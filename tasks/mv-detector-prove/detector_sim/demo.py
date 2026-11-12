"""The scripted demo, closed loop on the robot's own observations only.

Inputs, every control step, exactly what ``yam_arena`` gives any policy:
- the three RGB images ``top_cam``, ``left_cam``, ``right_cam`` (320 x 176 in the official command);
- ``joint_pos``: each arm's six joint angles and gripper opening.
It also uses fixed knowledge a person setting up the task has: the documented layout (``detector_sim.layout``, the
numbers in task.json ``setup``), the station's published camera calibration and the arm's URDF (``yam_arena``).
It never reads the simulator: no object poses, no prop joints, no light or button states, no scorer values.

How it perceives each thing:
- Detector in its cradle: the top camera's red pixels, matched against the detector's outline (head, body and
  spline tab, minus the black label and the lens on its face) rendered with the camera calibration; the best
  match over shifts in the table plane gives its position (the cradle height is documented).
- Tester: the red TEST button, the top-camera blob nearest the documented position, put on the button's known
  height; refined with the right wrist camera while arm B hovers over it. The end plate is 5.7 cm from the button.
- Bar: the right wrist camera's light-grey pixels at the start, matched against the outline of the bar on its
  two posts.
- The detector in the hand: where arm A's fingers closed on the stick (forward kinematics of arm A's joint readings)
  relative to where the camera saw the detector lying, so its pose then follows arm A's joint readings; the top
  camera checks it again after the lift and before each touch.
- Detector switched on: green range lights appearing on its face in the top camera after the press.
- The detector flashing on the tester: the two indicating lights' pixels in the top camera, located from the
  detector's pose, sampled every step while the top end is held on the plate; flashing = red and dark samples
  alternating.
- Touching: arm A's joint readings show the top end stopped against the plate or the bar.

Arm B presses ON with its closed fingers, arm A takes the stick (fingers down, closing across it at its balance
point) and lifts it. For each touch the top end goes 15 mm in front of the target, the arm's sag under the load is
measured from its joint readings and corrected, and the top end is pushed in along the stick past the target
(the plate or bar stops it). Arm B holds TEST during each prove. If the detector does not flash on the tester, the
demo withdraws 13 cm and holds still. DETECTOR_CONTROL=1 is the control run: on a failed prove it carries on to the
bar anyway (it must score 0).
"""

from __future__ import annotations

import math
import os

import numpy as np

from detector_sim import layout as L
from detector_sim import vision as V
from detector_sim.kit import CLOSED, DOWN, OPEN, ROOT_OFFSET, ArmTimeline, GraspModel, conj, q_axis, qmul, rotate
from yam_arena import ik, station
from yam_arena.ik import matrix_to_quat_xyzw
from yam_arena.urdf import quat_xyzw_to_matrix

CONTROL = os.environ.get("DETECTOR_CONTROL") == "1"

STANDOFF = 0.015
PUSH_STEP, PUSH_DT = 0.0015, 0.25
"""The push advances the commanded top end 1.5 mm every 0.25 s (6 mm/s)."""
PUSH_STALL = 0.0004
"""Stopped: the top end (from arm A's joint readings) advanced less than this in one push step."""
PUSH_HOLD = 0.0015
PUSH_MAX = 0.006
"""After it stops, the command stays this far past the stop: a light, steady touch."""
BACKOFF = 0.05
LIFT = 0.02
"""Moves between the tester and the bar run this much higher, clear of the detector's empty cradle."""
FINGER_END_BELOW_GRASP = 0.012
Q_REL = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
"""yam_arena's ``gripper`` frame = our flange frame * Rz(90) Rx(180) (see detector_sim.station)."""
TCP_EXTRA = 0.1347 - 0.134
FLASH_AMPLITUDE = 15
"""An indicating-light pixel's redness (R - max(G, B)) swings by 20-50 while it flashes (the renderer smooths the
0.2 s on / 0.2 s off flashes); a still pixel varies by under 5."""
FLASH_CROSSINGS = 4
"""Flashing needs at least this many swings across the middle in 1.5 s (it flashes at 2.5 Hz; the unit's 'alive'
blink every 4 s gives at most two)..."""
FLASH_PERIOD_S = 0.4
FLASH_COMPONENT = 6.0
"""...and a component of at least this amplitude at the flashing rate (2.5 Hz)."""
LENS_SEARCH_PX = 4
ARM = {"left": slice(0, 6), "right": slice(7, 13)}


def _mat(q) -> np.ndarray:
    return quat_xyzw_to_matrix(np.asarray(q, dtype=float))


class Demo:
    """``next_action(observation)`` gives the 14-D joint action for the next control step."""

    def __init__(self, control_hz: float = 20.0, speed: float = 0.10, log=print):
        self.dt = dt = 1.0 / control_hz
        self.speed = speed
        self.log = log
        self.k = 0
        self.obs = None
        self.planner = None
        self.watchers = []
        self.q_arm = {}
        self.worst = {"left": 0.0, "right": 0.0}
        self.hk = lambda s: max(1, round(s / dt))  # noqa: E731
        self.cams = None

    # -- observations ---------------------------------------------------------------------------------------------
    def joints(self, side) -> np.ndarray:
        return np.asarray(self.obs.state["joint_pos"], dtype=float)[ARM[side]]

    def image(self, name) -> np.ndarray:
        return np.asarray(self.obs.images[name])

    def hand(self, side) -> tuple[np.ndarray, np.ndarray]:
        """(desk position, quaternion in our flange convention) of one arm's grasp point, from its joint readings."""
        tcp, r = ik.tcp_pose(self.joints(side), station.BASES[side])
        quat = qmul(matrix_to_quat_xyzw(r), conj(Q_REL))
        pos = tcp - rotate(quat, (0.0, 0.0, -TCP_EXTRA)) - (0.0, 0.0, V.DESK_Z)
        return pos, quat

    def top(self) -> V.PinholeCam:
        h, w = self.image("top_cam").shape[:2]
        return V.top_cam(w, h)

    def wrist(self, side) -> V.PinholeCam:
        h, w = self.image(f"{side}_cam").shape[:2]
        return V.wrist_cam(side, self.joints(side), w, h)

    # -- control --------------------------------------------------------------------------------------------------
    def joint_action(self, side, pos_root, quat, grip):
        pos = pos_root + ROOT_OFFSET + (0.0, 0.0, V.DESK_Z)
        tcp = pos + rotate(quat, (0.0, 0.0, -TCP_EXTRA))
        q, pe, _ = ik.solve(tcp, qmul(quat, Q_REL), self.q_arm[side], base_pos=station.BASES[side], iters=60)
        self.q_arm[side] = q
        if pe > 0.005 and pe > self.worst[side]:
            self.log(f"IK {side} k={self.k} error {pe * 1000:.1f} mm at {np.round(pos - (0, 0, V.DESK_Z), 4)}")
        self.worst[side] = max(self.worst[side], pe)
        return np.r_[q, grip]

    def next_action(self, observation) -> np.ndarray:
        self.obs = observation
        if self.planner is None:
            for side in ("left", "right"):
                self.q_arm[side] = self.joints(side).copy()
            self.planner = Planner(self)
        k = self.k
        for w in list(self.watchers):
            if w(k) is False:
                self.watchers.remove(w)
        p = self.planner
        p.run_due(k)
        p.sync()
        if k >= len(p.L.steps):
            p.L.hold(1)
            p.R.hold(1)
        (lp, lq, lg), (rp, rq, rg) = p.L.steps[k], p.R.steps[k]
        self.k = k + 1
        return np.concatenate([self.joint_action("left", lp, lq, lg), self.joint_action("right", rp, rq, rg)])


class Planner:
    def __init__(self, demo: Demo):
        self.d = d = demo
        self.note = lambda msg: d.log(msg, flush=True)
        hk = d.hk
        l_pos, l_quat = d.hand("left")
        r_pos, r_quat = d.hand("right")
        self.L = ArmTimeline(l_pos - ROOT_OFFSET, l_quat, d.dt, d.speed)
        self.R = ArmTimeline(r_pos - ROOT_OFFSET, r_quat, d.dt, d.speed)
        self.events, self.readings, self.k = {}, [], -1
        self.q_tool = np.array(L.TOOL_QUAT)
        self.cmd_quat = self.q_tool.copy()
        self.sag = np.zeros(3)
        """Commanded detector orientation; corrected so that the joint readings show it level and along the stick."""
        self.grip_a = qmul(q_axis((0, 0, 1), -math.pi / 2), DOWN)
        self.model = V.DetectorModel(L.HEAD, L.BODY, L.LABEL, L.LENS, L.TAB)
        self.grasp = None
        self.tip0 = L.TIP_START.copy()
        self.test_top = L.TEST_TOP.copy()
        self.bar = L.BAR_CONTACT.copy()
        self.switched_on = False
        settle = hk(1.0)
        self.L.hold(settle)
        self.R.hold(settle)
        self.at(settle, self.look)

    # -- scheduling -----------------------------------------------------------------------------------------------
    def at(self, step, fn):
        step = max(step, self.k + 1)
        while step in self.events:
            step += 1
        self.events[step] = fn

    def run_due(self, k):
        self.k = k
        if k in self.events:
            self.events.pop(k)()

    def sync(self):
        n = max(len(self.L.steps), len(self.R.steps))
        self.L.pad_to(n)
        self.R.pad_to(n)
        return n

    def when_settled(self, side, fn, tries=15):
        tl = self.L if side == "left" else self.R

        def check():
            lag = np.linalg.norm(self.d.hand(side)[0] - (tl.pos + ROOT_OFFSET))
            still = check.prev is not None and abs(lag - check.prev) < 0.0003
            check.prev = lag
            if lag < 0.003 or still or check.tries <= 0:
                fn()
            else:
                check.tries -= 1
                tl.hold(self.d.hk(0.2))
                self.at(self.sync(), check)

        check.tries, check.prev = tries, None
        self.at(self.sync(), check)

    # -- the detector's pose, from arm A's joint readings ------------------------------------------------------
    def tool_pose(self) -> tuple[np.ndarray, np.ndarray]:
        g, q = self.d.hand("left")
        return g + rotate(q, self.grasp.rel_pos), qmul(q, self.grasp.rel_quat)

    def tip(self) -> np.ndarray:
        return self.tool_pose()[0]

    def tool_to(self, tip_pos, speed=None, min_steps=3):
        """Command the top end to tip_pos, offset by the arm's last measured sag under the load (self.sag)."""
        g, q = self.grasp.grasp_for(np.asarray(tip_pos) - self.sag, self.cmd_quat)
        self.L.move(g, q, speed=speed, min_steps=min_steps)

    # -- seeing ------------------------------------------------------------------------------------------------------
    def fit_detector(self, tip, quat, label, span=0.024):
        """Shift in the table plane that best matches the detector's outline to the top camera's red pixels."""
        img = self.d.image("top_cam")
        cam = self.d.top()
        obs = V.red_mask(img)
        rot = _mat(quat)
        pred0 = self.model.red(cam, tip, rot, obs.shape)
        ys, xs = np.nonzero(pred0)
        if len(xs) < 20:
            self.note(f"SEE {label}: detector outside the top camera's view")
            return np.zeros(3), 0.0
        roi = np.zeros_like(obs)
        roi[max(0, ys.min() - 8):ys.max() + 9, max(0, xs.min() - 8):xs.max() + 9] = True

        def score(dx, dy):
            return V.iou(self.model.red(cam, tip + (dx, dy, 0.0), rot, obs.shape), obs, roi)

        s, dx, dy = V.fit_shift(score, span=span)
        self.note(f"SEE {label}: detector outline match {score(0, 0):.2f} -> {s:.2f} at shift "
                  f"({dx * 1000:+.1f}, {dy * 1000:+.1f}) mm")
        return np.array([dx, dy, 0.0]), s

    def fit_yaw(self, tip, quat, label):
        """Logged check: the turn about the hand (vertical axis) that best matches the top camera's red pixels."""
        img = self.d.image("top_cam")
        cam = self.d.top()
        obs = V.red_mask(img)
        g = self.d.hand("left")[0]
        rot = _mat(quat)
        pred0 = self.model.red(cam, tip, rot, obs.shape)
        ys, xs = np.nonzero(pred0)
        roi = np.zeros_like(obs)
        roi[max(0, ys.min() - 10):ys.max() + 11, max(0, xs.min() - 10):xs.max() + 11] = True
        best = (-1.0, 0.0)
        for deg in np.arange(-8.0, 8.01, 0.5):
            rz = _mat(q_axis((0, 0, 1), math.radians(deg)))
            s = V.iou(self.model.red(cam, g + rz @ (tip - g), rz @ rot, obs.shape), obs, roi)
            if s > best[0]:
                best = (s, deg)
        self.note(f"SEE {label}: best turn about the hand {best[1]:+.1f} deg (match {best[0]:.2f})")
        return best[1]

    def locate_blob(self, cam, img, mask, nominal, label, max_px=6.0):
        """Desk point of the colour blob nearest a documented point (on that point's height)."""
        uv0 = cam.project(nominal)[0]
        blob = V.largest_blob(mask, near_uv=uv0, min_px=2)
        if blob is None:
            self.note(f"SEE {label}: no blob; using the documented position")
            return nominal.copy()
        vs, us = np.nonzero(blob)
        uv = np.array([us.mean(), vs.mean()])
        if np.linalg.norm(uv - uv0) > max_px:
            self.note(f"SEE {label}: nearest blob {np.round(uv, 1)} too far from {np.round(uv0, 1)}; documented")
            return nominal.copy()
        p = cam.on_plane_z(uv, nominal[2])
        self.note(f"SEE {label}: blob at {np.round(uv, 1)} (expected {np.round(uv0, 1)}) -> "
                  f"{np.round((p - nominal) * 1000, 1)} mm from the documented position")
        return p

    def fit_bar(self):
        """The bar on its posts in the right wrist camera (light-grey pixels), shift in the table plane."""
        img = self.d.image("right_cam")
        cam = self.d.wrist("right")
        hsv = V.hsv(img)
        obs = (hsv[..., 1] <= 40) & (hsv[..., 2] >= 150) & ~(V.yellow_mask(img))
        oak = (hsv[..., 0] >= 12) & (hsv[..., 0] <= 30) & (hsv[..., 1] > 25)
        obs &= ~oak
        a, b = L.BAR_SPAN
        rot = np.eye(3)
        axis = L.BAR_CONTACT + (0.0, -L.BAR_SIDE / 2, 0.0)  # bar axis below the contact face

        def render(dx, dy):
            c = axis + (dx, dy, 0.0)
            m = V.box_silhouette(cam, c + ((a + b) / 2, 0, 0), rot, (b - a, L.BAR_SIDE, L.BAR_SIDE), obs.shape)
            for sk in (a + 0.011, b - 0.02):
                h = L.TOOL_Z - L.BAR_SIDE / 2
                m |= V.box_silhouette(cam, (c[0] + sk, c[1], h / 2), rot, (0.022, 0.022, h), obs.shape)
            return m

        pred0 = render(0, 0)
        ys, xs = np.nonzero(pred0)
        if len(xs) < 20:
            self.note("SEE bar: outside the right wrist camera's view; using the documented position")
            return
        roi = np.zeros_like(obs)
        roi[max(0, ys.min() - 10):ys.max() + 11, max(0, xs.min() - 10):xs.max() + 11] = True
        s, dx, dy = V.fit_shift(lambda dx, dy: V.iou(render(dx, dy), obs, roi), span=0.018)
        s0 = V.iou(pred0, obs, roi)
        self.note(f"SEE bar: outline match {s0:.2f} -> {s:.2f} at shift ({dx * 1000:+.1f}, {dy * 1000:+.1f}) mm")
        if s > 0.4:
            self.bar = L.BAR_CONTACT + (dx, dy, 0.0)

    def lens_pixel(self, tip, quat) -> tuple[int, int]:
        """Top-camera pixel of the middle of the indicating-light lens, from the detector's pose."""
        u, v = self.model.face_point(self.d.top(), tip, _mat(quat), L.INDICATOR_FROM_TIP, 0.0)
        return int(round(u)), int(round(v))

    def lens_patch(self, uv) -> np.ndarray:
        """Redness of the top-camera pixels within +-LENS_SEARCH_PX of the lens (rows x cols)."""
        img = self.d.image("top_cam")
        u, v = uv
        r = LENS_SEARCH_PX
        patch = np.full((2 * r + 1, 2 * r + 1), -999, np.int16)
        red = V.redness(img)
        v0, v1, u0, u1 = max(0, v - r), min(img.shape[0], v + r + 1), max(0, u - r), min(img.shape[1], u + r + 1)
        if v0 < v1 and u0 < u1:
            patch[v0 - (v - r):v1 - (v - r), u0 - (u - r):u1 - (u - r)] = red[v0:v1, u0:u1]
        return patch

    def green_on_face(self, tip, quat) -> int:
        """Green (range-light) pixels on the detector's label in the top camera."""
        img = self.d.image("top_cam")
        cam = self.d.top()
        quad = V.face_quad(cam, tip, _mat(quat), L.LABEL[0], L.LABEL[1], L.LABEL[2], self.model.face_x)
        m = np.zeros(img.shape[:2], bool)
        V.fill_quad(m, quad, True)
        return int((V.green_mask(img) & m).sum())

    def tester_led(self) -> int:
        """Green pixels near the tester's LED in the right wrist camera (logged only)."""
        img = self.d.image("right_cam")
        cam = self.d.wrist("right")
        led = self.test_top + L.D * (L.LED_FROM_PLATE - L.TEST_FROM_PLATE) - (0.0, 0.0, 0.002)
        u, v = cam.project(led)[0]
        if not (0 <= u < img.shape[1] and 0 <= v < img.shape[0]):
            return -1
        u, v = int(round(u)), int(round(v))
        return int(V.green_mask(img)[max(0, v - 6):v + 7, max(0, u - 6):u + 7].sum())

    # -- the plan -----------------------------------------------------------------------------------------------
    def look(self):
        """First look, arms at the ready pose: the detector in its cradle, the TEST button, the bar."""
        shift, s = self.fit_detector(L.TIP_START, self.q_tool, "detector in cradle")
        if s > 0.5:
            self.tip0 = L.TIP_START + shift
        self.test_top = self.locate_blob(self.d.top(), self.d.image("top_cam"), V.red_mask(self.d.image("top_cam")),
                                         L.TEST_TOP, "TEST button (top camera)")
        self.fit_bar()
        self.press_on()

    # Self-test --------------------------------------------------------------------------------------------------
    def on_button_top(self):
        local = (L.BODY[0] / 2 + 0.002, L.ON_BUTTON[1], L.ON_BUTTON[0])
        return self.tip0 + rotate(self.q_tool, local)

    def press_on(self, extra_depth=0.0):
        hk = self.d.hk
        g = self.on_button_top() + np.array([0.0, 0.0, FINGER_END_BELOW_GRASP - 0.0015 - extra_depth])
        self.R.grip_to(CLOSED, hk(0.4))
        self.R.move(g + np.array([0.0, 0.0, 0.05]), DOWN, min_steps=hk(3.5))
        self.R.hold(hk(0.5))
        self.R.move(g, speed=0.02)
        self.R.hold(hk(1.2))
        self.R.move(g + np.array([0.0, 0.0, 0.05]), speed=0.04)
        self.test_g = self.test_top + np.array([0.0, 0.0, FINGER_END_BELOW_GRASP - 0.0015])
        self.R.move(self.test_g + np.array([0.0, 0.0, 0.04]), DOWN, min_steps=hk(2.0))
        self.R.hold(hk(0.6))
        self.at(len(self.R.steps), lambda: self.check_on(extra_depth))
        if extra_depth == 0.0:
            self.plan_pick()

    def check_on(self, extra_depth):
        """Arm B is off the detector now: its lowest range light stays green once the unit is on."""
        green = self.green_on_face(self.tip0 if self.grasp is None else self.tip(),
                                   self.q_tool if self.grasp is None else self.tool_pose()[1])
        self.switched_on = green > 0
        self.on_seen_k = self.k
        self.note(f"SEE switched on: {green} green pixels on the detector's face -> {self.switched_on}")
        self.refine_test()
        if not self.switched_on and extra_depth < 0.002 and self.grasp is None and self.k < self.pick_close_k:
            self.note("  pressing ON again, 1 mm deeper")
            self.press_on(extra_depth + 0.001)

    def refine_test(self):
        """Arm B hovers over TEST: find the red button in its wrist camera."""
        p = self.locate_blob(self.d.wrist("right"), self.d.image("right_cam"), V.red_mask(self.d.image("right_cam")),
                             self.test_top, "TEST button (right wrist camera)", max_px=25.0)
        if np.linalg.norm(p - L.TEST_TOP) < 0.012:
            self.test_top = p
            self.test_g = self.test_top + np.array([0.0, 0.0, FINGER_END_BELOW_GRASP - 0.0015])

    # Pick -------------------------------------------------------------------------------------------------------
    def plan_pick(self):
        hk = self.d.hk
        nominal = GraspModel.nominal(self.grip_a, self.q_tool, (0.0, 0.0, L.GRASP_FROM_TIP))
        grasp, _ = nominal.grasp_for(self.tip0, self.q_tool)
        self.pick_grasp = grasp
        self.pick_close_k = 10**9
        self.L.grip_to(OPEN, hk(0.2))
        self.L.move(self.L.pos + ROOT_OFFSET + np.array([0.0, 0.0, 0.05]), min_steps=hk(1.0))
        self.L.move(grasp + np.array([0.0, 0.0, 0.07]), self.grip_a, min_steps=hk(4.0))
        self.L.hold(hk(1.0))
        self.sync()
        self.when_settled("left", self.descend)

    def descend(self):
        self.L.move(self.pick_grasp, speed=0.02)
        self.L.hold(self.d.hk(0.6))
        self.sync()
        self.when_settled("left", self.close)

    def close(self):
        e = self.d.hand("left")[0] - self.pick_grasp
        self.note(f"PICK: hand error before closing (mm) {np.round(e * 1000, 1)}")
        self.pick_close_k = self.k
        hk = self.d.hk
        self.L.grip_to(CLOSED, hk(1.4))
        self.sync()
        self.at(self.sync(), self.measure_grasp)

    def measure_grasp(self):
        """Fingers closed on the stick, which still lies in its cradle where the camera saw it: the grasp is where
        the hand is now (joint readings) relative to that pose."""
        g, q = self.d.hand("left")
        self.grasp = GraspModel(rotate(conj(q), self.tip0 - g), qmul(conj(q), self.q_tool))
        self.note(f"GRASP: top end in the hand frame (mm) {np.round(self.grasp.rel_pos * 1000, 1)}")
        self.L.move(self.pick_grasp + np.array([0.0, 0.0, L.TOOL_Z - L.CRADLE_Z + 0.01]), speed=0.02)
        self.L.hold(self.d.hk(1.0))
        self.sync()
        self.when_settled("left", self.after_pick)

    def after_pick(self):
        tip, quat = self.tool_pose()
        shift, s = self.fit_detector(tip, quat, "detector in hand")
        self.note(f"  top end from the joint readings {np.round(tip, 4)}")
        if not self.switched_on:
            self.note("WARNING: the detector's range lights were not seen; carrying on")
        self.prove(self.after_prove1)

    # Touch a contact ----------------------------------------------------------------------------------------------
    def touch(self, point, then, label):
        """Top end to STANDOFF in front of `point` (along -D), correct the arm's sag from its joint readings,
        check with the top camera, push in past the surface."""
        self.tool_to(point - BACKOFF * L.D, speed=0.06, min_steps=self.d.hk(1.5))
        self.sync()
        self.tool_to(point - STANDOFF * L.D, speed=0.03)
        self.L.hold(self.d.hk(0.4))
        self.sync()
        self.when_settled("left", lambda: self._correct(point, 0, then, label))

    def _correct(self, point, attempt, then, label):
        target = point - STANDOFF * L.D
        tip, quat = self.tool_pose()
        err = tip - target
        e_rot = qmul(quat, conj(self.q_tool))
        if e_rot[3] < 0:
            e_rot = -e_rot
        ang = 2 * math.degrees(math.acos(min(1.0, e_rot[3])))
        self.note(f"  {label} standoff {attempt}: top end error from joint readings (mm) {np.round(err * 1000, 1)}, "
                  f"tilt {ang:.1f} deg, axis {np.round(rotate(quat, (0, 0, -1)), 3)}")
        self.sag = self.sag + err
        if ang > 0.4:
            self.cmd_quat = qmul(conj(e_rot), self.cmd_quat)
            self.cmd_quat /= np.linalg.norm(self.cmd_quat)
        if attempt < 3 and (np.abs(err).max() > 0.0008 or ang > 0.4):
            self.tool_to(target, speed=0.01)
            self.L.hold(self.d.hk(0.4))
            self.sync()
            self.when_settled("left", lambda: self._correct(point, attempt + 1, then, label))
            return
        tip, quat = self.tool_pose()
        self.fit_detector(tip, quat, f"detector at the {label} standoff", span=0.018)
        self.fit_yaw(tip, quat, label)
        self.push(point, -STANDOFF, None, then, label)

    def push(self, point, cmd_along, prev_along, then, label, n=0):
        """Advance the commanded top end PUSH_STEP along the stick every PUSH_DT until arm A's joint readings show
        it no longer advancing (the plate or bar stops it), then hold PUSH_HOLD past the stop: a light touch."""
        along = float(np.dot(self.tip() - point, L.D))
        stalled = prev_along is not None and along - prev_along < PUSH_STALL and along > -STANDOFF + 0.004
        if stalled or cmd_along >= PUSH_MAX:
            self.tool_to(point + (along + PUSH_HOLD) * L.D, speed=0.01)
            self.note(f"  {label}: top end stopped {along * 1000:+.1f} mm along the stick from the surface "
                      f"(joint readings; commanded {cmd_along * 1000:+.1f}); holding {PUSH_HOLD * 1000:.1f} mm past")
            self.L.hold(self.d.hk(0.5))
            self.at(self.sync(), then)
            return
        cmd_along += PUSH_STEP
        self.tool_to(point + cmd_along * L.D, min_steps=self.d.hk(PUSH_DT))
        self.at(self.sync(), lambda: self.push(point, cmd_along, along, then, label, n + 1))

    def withdraw(self, point, then, back=BACKOFF):
        self.tool_to(self.tip() - 0.015 * L.D, speed=0.02)
        self.tool_to(point - back * L.D + (0.0, 0.0, LIFT), speed=0.05)
        self.at(self.sync(), then)

    def watch_lights(self, n_steps, done):
        """Watch the indicating lights in the top camera for n_steps (the arm holds still), then call
        done(flashing). Every pixel near the lens gets a redness series; the lights flash if some pixel swings
        between dark and red again and again. Steady red body, table or a single 'alive' blink do not count."""
        uv = self.lens_pixel(*self.tool_pose())
        frames, leds = [], []
        end = self.k + n_steps

        def w(k):
            frames.append(self.lens_patch(uv))
            leds.append(self.tester_led())
            if k < end:
                return True
            s = np.stack(frames).astype(float)  # (T, rows, cols)
            lo, hi = np.percentile(s, 10, axis=0), np.percentile(s, 90, axis=0)
            above = s > (lo + hi) / 2
            crossings = (above[1:] != above[:-1]).sum(0)
            t = np.arange(len(s)) * self.d.dt
            wave = np.exp(-2j * np.pi * t / FLASH_PERIOD_S)[:, None, None]
            comp = 2 * np.abs(((s - s.mean(0)) * wave).mean(0))
            ok = ((hi - lo >= FLASH_AMPLITUDE) & (crossings >= FLASH_CROSSINGS) & (comp >= FLASH_COMPONENT)
                  & (lo > -900))
            flashing = int(ok.sum()) >= 2
            i, j = np.unravel_index(np.argmax(np.where(lo > -900, comp, -1)), lo.shape)
            self.note(f"SEE lights at pixel {uv}: {int(ok.sum())} flashing pixels -> flashing {flashing}; strongest "
                      f"({i - LENS_SEARCH_PX:+d}, {j - LENS_SEARCH_PX:+d}) swing {hi[i, j] - lo[i, j]:.0f}, "
                      f"{crossings[i, j]} crossings, 2.5 Hz part {comp[i, j]:.0f}, redness {s[:, i, j].astype(int).tolist()}; "
                      f"tester LED green px {leds[len(leds) // 2]}")
            done(flashing)
            return False

        self.d.watchers.append(w)

    # Prove / test -------------------------------------------------------------------------------------------------
    def plate(self):
        return self.test_top - L.TEST_FROM_PLATE * L.D - (0.0, 0.0, L.TEST_TOP[2] - L.PLATE[2])

    def prove(self, then):
        plate = self.plate()
        self.R.move(self.test_g + np.array([0.0, 0.0, 0.04]), DOWN, speed=0.04)
        self.R.move(self.test_g, speed=0.02)
        self.R.hold(self.d.hk(0.4))

        def read():
            self.L.hold(self.d.hk(1.6))
            self.sync()

            def decided(lit):
                self.readings.append(("prove", lit))
                self.R.move(self.test_g + np.array([0.0, 0.0, 0.04]), speed=0.03)
                self.withdraw(plate, lambda: then(lit))

            self.watch_lights(self.d.hk(1.5), decided)

        self.touch(plate, read, "tester")

    def test_target(self, then):
        c = self.bar

        def read():
            self.L.hold(self.d.hk(1.6))
            self.sync()

            def decided(lit):
                self.readings.append(("bar", lit))
                self.withdraw(c, lambda: then(lit))

            self.watch_lights(self.d.hk(1.5), decided)

        self.touch(c, read, "bar")

    def after_prove1(self, lit):
        if lit or CONTROL:
            if not lit:
                self.note("CONTROL: the prove failed; carrying on to the bar anyway")
            self.test_target(self.after_test)
        else:
            self.note("STOP: the detector did not flash on the tester; not going to the bar.")
            self.tool_to(self.plate() - 0.13 * L.D, speed=0.04)
            self.sync()
            self.L.hold(self.d.hk(8.0))
            self.R.move(self.test_g + np.array([0.0, 0.0, 0.08]), speed=0.04)
            self.sync()
            self.at(len(self.L.steps), lambda: self.note(f"DONE (stopped): readings {self.readings}"))

    def after_test(self, lit):
        self.prove(self.after_prove2)

    def after_prove2(self, lit):
        self.note(f"DONE: readings {self.readings}")
        self.L.hold(self.d.hk(3.0))
        self.sync()
