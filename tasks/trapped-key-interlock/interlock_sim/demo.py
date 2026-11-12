"""The scripted demo, closed loop on robot-observable inputs only.

What it reads, every control step: the observation Inspect Robots hands any policy, i.e. the three RGB images
(``top_cam``, ``left_cam``, ``right_cam``) and the 14-D ``joint_pos`` state. Nothing else comes from the simulator:
no object poses, no prop joint angles, no latch or scorer state. Fixed knowledge it uses: the documented setup
(task.json ``setup``: where each part should be, part sizes, label colours), the station's published camera
calibration and the arm's URDF (``yam_arena.station``, ``yam_arena.urdf``), as a real robot's script would.

How it perceives each thing:
- Which pair is A1, and roughly where: in the first top-camera frame, the yellow A1 labels (door enclosure top,
  breaker face) are segmented by colour and back-projected onto their documented heights; the offset from the
  documented layout (clipped to 3 cm) shifts the look poses below.
- Door A1's interlock and breaker A1's interlock (where key A1 is): the left arm moves its wrist camera over each
  lock and finds the lock's red nameplate; the plate's outline is back-projected onto its known plane (camera pose
  from forward kinematics of the measured joints) and a rectangle with the documented size and orientation is
  fitted; the keyway axis is the documented offset from the plate's centre. Three frames are averaged.
- Key A1 in the hand: the key is pinched while it is still in breaker A1's lock, so the key's tip is on that lock's
  keyway axis (camera estimate) at the documented seated depth. From then on the tip is predicted from forward
  kinematics of the measured joints and that grasp offset (the parallel pinch keeps the bow's plane on the pads,
  so the key turns with the wrist).
- Insertion: the predicted tip is servoed onto the camera estimate of door A1's keyway, then lowered. Whether the
  key went home is read from proprioception (how far down the hand got: a seated key sits 15.7 mm below the
  face, one resting on the plug's rim about 8 mm); if not, the demo lifts 2 mm and tries the next point of a small
  spiral (0.3 mm pitch) around the estimate.
- Turns, settling and release: commanded wrist roll, measured joint angles and timing.

Grasp-point targets become yam_arena's 14-D joint action through yam_arena's own IK (``yam_arena.ik``).
``--policy interlock-demo`` (interlock_sim.policy) plays it. ``INTERLOCK_DEMO_DEBUG=<n>`` prints the estimates every
n steps.
"""

from __future__ import annotations

import math
import os

import numpy as np

from yam_arena import ik, station
from yam_arena.ik import matrix_to_quat_xyzw

from interlock_sim import vision as V
from interlock_sim.kit import CLOSED, DOWN, OPEN, ROOT_OFFSET, ArmTimeline, GraspModel, conj, q_axis, qmul, rotate
from interlock_sim.spec import BREAKER_XY, DOOR_XY

MM = 0.001
# Documented geometry (task.json setup; interlock_sim/usd/props.py), desk frame: table top z = 0, x forward, y left.
LOCK_FACE_Z_BRK = 0.146          # breaker interlock's cylinder face above the table
LOCK_FACE_Z_DOOR = 0.0539        # door interlock's cylinder face above the table
BRK_LOCK_IN_BRK = (0.0521, -0.011)   # keyway from the breaker's centre
LOCK_YAW = {"breaker": 0.0, "door": -90.0}  # nameplate orientation (lock frame) seen from above
KEY_TIP_SEATED = -0.0157         # a seated key's tip below the face
GRASP_ABOVE_TIP = 0.037          # middle of the key's bow above its tip
LOOK_LIFT = 0.03                 # key lifted this far above its seated position for the matched nameplate looks
LABEL_YELLOW = ((22, 45), (170, 255), (150, 255))
LABEL_BLUE = ((130, 155), (110, 255), (120, 255))
DOOR_LABEL = ((0.062, 0.138), (0.040, 0.140), 0.152)   # enclosure-top label: x, y ranges from the door keyway, z
BRK_LABEL = ((-0.051, -0.017), (-0.042, 0.042), 0.086)  # breaker-face label: x, y ranges from breaker centre, z

Q_REL = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
"""yam_arena's ``gripper`` frame = our flange frame * Q_REL (fingers along the flange's -z)."""
TCP_EXTRA = 0.1347 - 0.134
"""Our grasp point (between the finger pads) sits this much further out than yam_arena's TCP."""


def yaw_q(deg):
    return q_axis((0, 0, 1), math.radians(deg))


def quat_yaw(q) -> float:
    """Heading (deg) of a key frame quaternion (rotation about z)."""
    return math.degrees(math.atan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)))


def spiral(pitch=0.0003, rings=3):
    pts = []
    for r in range(1, rings + 1):
        n = 6 * r
        pts += [(r * pitch * math.cos(2 * math.pi * i / n), r * pitch * math.sin(2 * math.pi * i / n)) for i in range(n)]
    return pts


class Demo:
    """``next_action(observation)`` gives the 14-D joint action for the next control step."""

    def __init__(self, target: str = "A1", control_hz: float = 20.0, speed: float = 0.10, log=None):
        self.log = log = log or (lambda *a: print(*a, flush=True))
        self.tgt = tgt = target
        self.side = side = "left" if tgt == "A1" else "right"
        self.rest = "right" if side == "left" else "left"
        self.sgn = 1.0 if side == "left" else -1.0
        self.hand_yaw = 0.0 if side == "left" else 180.0
        self.dt = 1.0 / control_hz
        self.speed = speed
        self.debug_every = int(os.environ.get("INTERLOCK_DEMO_DEBUG", "0"))
        self.k = 0
        self.obs = None
        self.events: dict[int, list] = {}
        self.A = None
        bx, by = BREAKER_XY[tgt]
        # Documented positions (desk frame): keyway axes at their faces.
        self.K_doc = np.array([bx + BRK_LOCK_IN_BRK[0], by + BRK_LOCK_IN_BRK[1], LOCK_FACE_Z_BRK])
        self.D_doc = np.array([*DOOR_XY[tgt], LOCK_FACE_Z_DOOR])
        self.K, self.D = self.K_doc.copy(), self.D_doc.copy()
        self.model = None
        self.search = spiral()
        self.search_i = 0
        self.q_cmd = None
        self.worst = 0.0

    # -- observation helpers -------------------------------------------------------------------------------------

    def hold_k(self, s):
        return max(1, round(s / self.dt))

    def q(self, side=None):
        return V.split_state(self.obs.state["joint_pos"])[side or self.side]

    def img(self, name):
        return np.asarray(self.obs.images[name])

    def grasp_point(self):
        """Measured (desk position, flange quaternion) of the working arm's grasp point, from FK of its joints."""
        p, r = V.tcp_desk(self.q()[0], self.side)
        q_theirs = matrix_to_quat_xyzw(r)
        return p + r @ np.array([0.0, 0.0, TCP_EXTRA]), qmul(q_theirs, conj(Q_REL))

    def key_est(self):
        """Predicted key tip (desk) and heading (deg) from the measured joints and the grasp offset."""
        g, gq = self.grasp_point()
        return g + rotate(gq, self.model.rel_pos), quat_yaw(qmul(gq, self.model.rel_quat))

    def wrist_cam(self):
        im = self.img(f"{self.side}_cam")
        return V.Cam.wrist(self.side, self.q()[0], im.shape[1], im.shape[0]), im

    # -- scheduling ------------------------------------------------------------------------------------------------

    def at(self, step, fn):
        self.events.setdefault(step, []).append(fn)

    def now(self, fn):
        self.at(len(self.A.steps), fn)

    def when_settled(self, fn, tries=30, tol=0.002):
        A = self.A

        def check():
            lag = np.linalg.norm(self.grasp_point()[0] - (A.pos + ROOT_OFFSET))
            if lag < tol or check.tries <= 0:
                if lag >= tol:
                    self.log(f"SETTLE timeout: lag {lag * 1000:.1f} mm")
                fn()
            else:
                check.tries -= 1
                A.hold(self.hold_k(0.1))
                self.now(check)

        check.tries = tries
        self.now(check)

    # -- perception ----------------------------------------------------------------------------------------------------

    def top_survey(self):
        """Coarse: the yellow A1 labels in the top camera; offsets of door and breaker from the documented layout."""
        im = self.img("top_cam")
        cam = V.Cam.top(im.shape[1], im.shape[0])
        colour = LABEL_YELLOW if self.tgt == "A1" else LABEL_BLUE
        bl = V.blobs(V.mask_range(im, *colour), min_px=6)
        self.coarse = {}
        for name, (xs, ys, z), ref in (("door", DOOR_LABEL, self.D_doc), ("breaker", BRK_LABEL,
                                       np.array([*BREAKER_XY[self.tgt], 0.0]))):
            doc = np.array([ref[0] + np.mean(xs), ref[1] + np.mean(ys), z])
            b = V.nearest_blob(bl, cam.project(doc)[0], 25.0)
            if b is None:
                self.log(f"TOP CAM: no {self.tgt} label found near the documented {name}; using the layout")
                off = np.zeros(2)
            else:
                est = cam.on_plane(b["centroid"], z)
                off = np.clip(est[:2] - doc[:2], -0.03, 0.03)
                self.log(f"TOP CAM: {self.tgt} {name} label at {np.round(est[:2], 4)} ({b['area']} px), "
                         f"offset from layout {np.round(off * 1000, 1)} mm")
            self.coarse[name] = off

    def measure_lock(self, which, then, frames=3, tries=2):
        """Fine: the lock's keyway from its nameplate in the wrist camera, averaged over a few still frames."""
        guess = self.D if which == "door" else self.K
        got = []

        def grab():
            cam, im = self.wrist_cam()
            r = V.locate_lock(im, cam, guess, LOCK_YAW[which])
            if r is not None:
                got.append(r)
                self.log(f"WRIST CAM {which}: keyway {np.round(r['xy'], 4)}, plate {np.round(r['size_mm'], 1)} mm "
                         f"({r['px']} px)")
            if len(got) < frames and grab.left > 0:
                grab.left -= 1
                self.A.hold(2)
                self.now(grab)
                return
            if got:
                xy = np.mean([g["xy"] for g in got], axis=0)
                size = np.mean([g["size_mm"] for g in got], axis=0)
                ok = abs(size[0] - 17.8) < 3.0 and abs(size[1] - 40.0) < 5.0
                if ok:
                    guess[:2] = xy
                self.log(f"{which.upper()} KEYWAY {'from camera' if ok else 'camera fit rejected, layout kept'}: "
                         f"{np.round(guess[:2], 4)} (layout {np.round((self.D_doc if which == 'door' else self.K_doc)[:2], 4)})")
            else:
                self.log(f"{which.upper()} KEYWAY: nameplate not seen, layout kept")
            then()

        grab.left = frames + tries
        self.now(grab)

    # -- the plan --------------------------------------------------------------------------------------------------

    def start(self):
        self.top_survey()
        self.D[:2] += self.coarse["door"]
        A = self.A
        A.grip_to(OPEN, self.hold_k(0.2))
        # Look at breaker A1's lock from a pose that sees its nameplate past the key, to aim the pinch.
        off, yaw, tilt = self.LOOK_PRE
        look_q = qmul(q_axis((0, 1, 0), math.radians(tilt)), qmul(yaw_q(yaw), DOWN))
        A.move(self.K + np.array(off), look_q, min_steps=self.hold_k(2.5))
        A.hold(self.hold_k(0.4))
        self.when_settled(lambda: self.measure_lock("breaker", self.plan_pick), tol=0.003)

    def plan_pick(self):
        A = self.A
        tip = self.K + np.array([0.0, 0.0, KEY_TIP_SEATED])
        hand = qmul(yaw_q(self.hand_yaw), DOWN)
        nominal = GraspModel.nominal(hand, yaw_q(0.0), (0.0, 0.0, GRASP_ABOVE_TIP))
        self.rel_quat = nominal.rel_quat
        grasp, gq = nominal.grasp_for(tip, yaw_q(0.0))
        A.move(grasp + np.array([0.0, 0.0, 0.07]), gq, min_steps=self.hold_k(1.5))
        A.move(grasp + np.array([0.0, 0.0, 0.015]), speed=0.04)
        self.when_settled(lambda: self._final_approach(grasp))

    def _final_approach(self, grasp):
        err = self.grasp_point()[0] - (self.A.pos + ROOT_OFFSET)
        self.A.move(grasp - np.array([err[0], err[1], 0.0]), speed=0.02)
        self.A.hold(self.hold_k(0.4))
        self.A.grip_to(CLOSED, self.hold_k(1.2))
        self.when_settled(self.turn_breaker)

    def measure_grasp(self, tip):
        """Grasp offset of the key: its tip is at ``tip`` (in a lock) and its heading follows the pads."""
        g, gq = self.grasp_point()
        self.model = GraspModel(rotate(conj(gq), tip - g), self.rel_quat)

    def turn_breaker(self):
        tip = self.K + np.array([0.0, 0.0, KEY_TIP_SEATED])
        self.measure_grasp(tip)  # provisional, refined below
        self.log(f"GRASP: key tip offset in the hand (mm) {np.round(self.model.rel_pos * 1000, 1)}")
        self._turn(tip, 0.0, 91.0, 4.0)
        self.A.hold(self.hold_k(0.6))
        self.when_settled(self.pull_out, tol=0.003)

    def _key_to(self, pos, yaw_deg, speed):
        g, q = self.model.grasp_for(np.asarray(pos), yaw_q(yaw_deg))
        self.A.move(g, q, speed=speed)

    def _turn(self, pos, yaw0, yaw1, seconds):
        n = self.hold_k(seconds)
        for i in range(1, n + 1):
            yaw = yaw0 + (yaw1 - yaw0) * i / n
            self.A.path([self.model.grasp_for(np.asarray(pos), yaw_q(yaw))])

    def pull_out(self):
        # The key is still in the lock (tip on the keyway axis, seated): keep the hand pose for the grasp offset.
        self.g_turned = self.grasp_point()
        _, yaw = self.key_est()
        self.log(f"TURNED breaker key: wrist heading gives key at {yaw:.1f} deg")
        tip = self.K + np.array([0.0, 0.0, KEY_TIP_SEATED])
        self._key_to(tip + np.array([0.0, 0.0, LOOK_LIFT]), 90.0, 0.02)
        self.A.hold(self.hold_k(0.6))
        # Key out and lifted, hand at the key's heading (90 deg): look at the empty lock's nameplate. The door
        # lock is looked at later from the same pose relative to its own nameplate, so the camera fit's bias
        # (the same in both lock frames) cancels between the grasp offset and the door target.
        self.when_settled(lambda: self.measure_lock("breaker", self.grasp_from_turn), tol=0.003)

    def grasp_from_turn(self):
        g, gq = self.g_turned
        tip = self.K + np.array([0.0, 0.0, KEY_TIP_SEATED])
        self.model = GraspModel(rotate(conj(gq), tip - g), self.rel_quat)
        self.log(f"GRASP (refined): key tip offset in the hand (mm) {np.round(self.model.rel_pos * 1000, 1)}")
        self.carry()

    def carry(self):
        kp, _ = self.key_est()
        self.log(f"OUT: key tip at {np.round(kp, 4)} (breaker face z {self.K[2]:.4f})")
        tx, ty, tz = self.D
        carry_z = 0.20 - GRASP_ABOVE_TIP
        self._key_to(np.array([kp[0], kp[1], carry_z]), 90.0, 0.06)
        self._key_to(np.array([tx, ty, carry_z]), 0.0, 0.08)
        self._key_to(np.array([tx, ty, tz + KEY_TIP_SEATED + LOOK_LIFT]), 0.0, 0.04)
        self.A.hold(self.hold_k(0.6))
        self.when_settled(lambda: self.measure_lock("door", self.to_entry), tol=0.003)

    def to_entry(self):
        tx, ty, tz = self.D
        self.entry = np.array([tx, ty, tz + 0.006])
        self.yaw_c = 0.0
        self._key_to(self.entry, 0.0, 0.02)
        self.A.hold(self.hold_k(0.6))
        self.when_settled(lambda: self._align(self.entry, np.zeros(3), 0, 0.0004, self._into_hole))

    def _align(self, target, correction, attempt, tol, then, max_attempts=8):
        """Servo the predicted key tip onto ``target`` (x, y) with heading 0."""
        kp, yerr = self.key_est()
        err = kp - target
        self.log(f"ALIGN {target[2] - self.D[2]:+.3f} attempt {attempt}: predicted tip error (mm) "
                 f"{np.round(err * 1000, 2)}, heading {yerr:.1f}")
        correction = correction + np.array([err[0], err[1], 0.0])
        self.yaw_c += yerr
        if attempt < max_attempts and (max(abs(err[0]), abs(err[1])) > tol or abs(yerr) > 1.0):
            self._key_to(target - correction, -self.yaw_c, 0.005)
            self.A.hold(self.hold_k(0.6))
            self.when_settled(lambda: self._align(target, correction, attempt + 1, tol, then, max_attempts))
            return
        then(correction)

    def _into_hole(self, correction, offset=(0.0, 0.0)):
        # Tip 6 mm into the face's 9 mm hole, 2 mm above the plug: align again, then seat.
        self.offset = np.array([offset[0], offset[1], 0.0])
        self.in_hole = np.array([self.D[0], self.D[1], self.D[2] - 0.006]) + self.offset
        self._key_to(self.in_hole - correction, -self.yaw_c, 0.006)
        self.A.hold(self.hold_k(0.5))
        # In the hole the shank can rest on the plug's rim, where friction stops small corrections: 4 tries at most.
        self.when_settled(lambda: self._align(self.in_hole, correction, 0, 0.00015, self._seat, 4))

    def _seat(self, correction):
        self.correction = correction
        home = np.array([self.D[0], self.D[1], self.D[2] + KEY_TIP_SEATED]) + self.offset
        self._key_to(home - correction + np.array([0.0, 0.0, -0.002]), -self.yaw_c, 0.004)
        self.A.hold(self.hold_k(0.8))
        self.when_settled(self.check_seated, tol=0.003)

    def check_seated(self):
        kp, yaw = self.key_est()
        depth = self.D[2] - kp[2]
        self.log(f"INSERT try {self.search_i}: hand says tip {depth * 1000:.1f} mm below the face, "
                 f"offset {np.round(self.offset[:2] * 1000, 2)} mm")
        if depth < 0.012 and self.search_i < len(self.search):
            off = self.search[self.search_i]
            self.search_i += 1
            self._into_hole(self.correction, off)
            return
        self._turn(kp, yaw, -93.0, 4.0)
        self.A.hold(self.hold_k(0.6))
        self.when_settled(self.release, tol=0.003)

    def release(self):
        _, yaw = self.key_est()
        self.log(f"TURNED door key: wrist heading gives key at {yaw:.1f} deg")
        A = self.A
        A.grip_to(OPEN, self.hold_k(1.0))
        A.move(A.pos + ROOT_OFFSET + np.array([0.0, 0.0, 0.08]), speed=0.04)
        A.move(np.array([0.15, self.sgn * 0.30, 0.28]), qmul(yaw_q(self.hand_yaw), DOWN), speed=0.10)
        A.hold(self.hold_k(6.0))

    # Wrist-camera look before the pinch: grasp point offset from the breaker keyway (desk, m) and hand yaw (deg),
    # chosen offline (URDF reach, calibrated camera) so the nameplate is in full view past the key and fingers.
    LOOK_PRE = ((0.01, 0.0, 0.08), 90.0, -25.0)  # hand tilted 25 deg about y

    # -- the action ----------------------------------------------------------------------------------------------------

    def _ik(self, side, pos_root, quat, grip):
        pos = pos_root + ROOT_OFFSET + np.array([0.0, 0.0, V.DESK_Z])  # world
        tcp = pos + rotate(quat, (0.0, 0.0, -TCP_EXTRA))
        rot = qmul(quat, Q_REL)
        q, pe, _ = ik.solve(tcp, rot, self.q_cmd, base_pos=station.BASES[side], iters=60)
        if pe > 0.001:  # warm start stuck: retry from other seeds
            for seed in (self.q()[0], np.array([0.0, 1.0, 1.0, -0.6, 0.0, 0.0]), np.array([0.4, 1.6, 1.8, -1.6, 0.0, 0.4])):
                q2, pe2, _ = ik.solve(tcp, rot, seed, base_pos=station.BASES[side], iters=200)
                if pe2 < pe and np.max(np.abs(q2 - self.q_cmd)) < 0.5:
                    q, pe = q2, pe2
        self.q_cmd = q
        if pe > 0.005 and pe > self.worst:
            self.log(f"IK {side} k={self.k} error {pe * 1000:.1f} mm at {np.round(pos_root + ROOT_OFFSET, 4)}")
        self.worst = max(self.worst, pe)
        return np.r_[q, grip]

    def next_action(self, obs) -> np.ndarray:
        self.obs = obs
        if self.A is None:
            g, gq = self.grasp_point()
            self.A = ArmTimeline(g - ROOT_OFFSET, gq, self.dt, self.speed)
            self.q_cmd = self.q()[0].copy()
            self.rest_q = self.q(self.rest)[0].copy()
            self.A.hold(self.hold_k(0.5))
            self.at(self.hold_k(0.5), self.start)
        while self.k in self.events:  # an event may schedule another for this same step
            for fn in self.events.pop(self.k):
                fn()
        A = self.A
        if self.k >= len(A.steps):
            A.hold(1)
        ap, aq, ag = A.steps[self.k]
        acts = {self.side: self._ik(self.side, ap, aq, ag), self.rest: np.r_[self.rest_q, CLOSED]}
        if self.debug_every and self.k % self.debug_every == 0 and self.model is not None:
            kp, yaw = self.key_est()
            self.log(f"DBG k={self.k} key tip {np.round(kp, 4)} heading {yaw:.1f}")
        self.k += 1
        return np.concatenate([acts["left"], acts["right"]])
