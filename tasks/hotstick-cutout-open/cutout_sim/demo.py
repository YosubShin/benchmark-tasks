"""The scripted demo, closed loop on robot-observable inputs only.

What it reads, every control step: the observation Inspect Robots hands any policy, i.e. the three RGB images
(``top_cam``, ``left_cam``, ``right_cam``) and the 14-D ``joint_pos`` state. Nothing else comes from the simulator:
no object poses, no prop joint angles, no latch or scorer state. Fixed knowledge it uses: the documented setup
(task.json ``setup``: where the cutout, its hinge, its pull ring and the stick rack are, part sizes, colours), the
station's published camera calibration and the arm's URDF (``yam_arena.station``, ``yam_arena.urdf``), as a real
robot's script would.

Plan (right arm = stick arm, left arm = camera arm):
1. Stick: the right wrist camera, at the start pose, sees the yellow stick in its rack; the shaft's pixels are
   back-projected onto the plane of its top face and a line fitted: the stick's lateral position (y) corrects the
   documented one (clipped to 1.5 cm). Its position along x is the documented one (butt against the rack's end stop).
2. Grasp: fingers down across the handgrip; once closed, the stick is still in the cradles, so its pose is known and
   the grasp offset (stick in hand) is measured from forward kinematics of the measured joints at that moment.
   From then on the prong is predicted from the measured joints plus that offset.
3. Ring: meanwhile the left arm moves its wrist camera to a fixed look pose (from the documented layout, 32 cm
   from the ring, outside the 15 cm keep-out) and looks at the orange pull ring from the side; an ellipse is
   fitted to the ring's outline and its centre back-projected onto the ring's plane (the swing plane, known), three
   frames averaged; the estimate replaces the documented ring centre (if within 2 cm of it). The left arm then goes
   back to its start pose.
4. Hook: the stick is raised and pitched up 32 deg so its axis points along the ring's opening path; the prong is
   servoed (predicted from the measured joints) to 35 mm right of the ring centre, then slid left through the ring.
5. Pull: the prong's target moves along the ring's arc about the documented hinge axis: first a slow pull 20 deg
   ahead (the latch holds the tube; the arm's lag builds the pull to the release force), then on along the arc to
   115 deg while the stick pitches down, lowering the tube (which rests on the prong) instead of letting it slam.
6. Unhook: the prong is slid right, out of the ring; the tube swings down the last bit and hangs.
7. Rack (the episode usually ends on success during this step, once the tube has hung still for 2 s): the stick
   is carried back over its rack, the predicted stick pose servoed onto the documented rack pose
   (corrected y), lowered into the cradles, released, and the arm backs away.

``mode="control"``: no stick; the right gripper reaches for the pull ring and pulls it (the keep-out hazard).
``mode="hold"``: both arms keep their start pose.
Grasp-point targets become yam_arena's 14-D joint action through yam_arena's own IK (``yam_arena.ik``).
``CUTOUT_DEMO_DEBUG=<n>`` prints the predicted prong every n steps.
"""

from __future__ import annotations

import math
import os

import numpy as np

from yam_arena import ik, station
from yam_arena.ik import matrix_to_quat_xyzw
from yam_arena.urdf import quat_xyzw_to_matrix

from cutout_sim import vision as V
from cutout_sim.kit import CLOSED, DOWN, OPEN, GraspModel, conj, q_axis, qmul, rotate, slerp
from cutout_sim.usd import props as P

MM = 0.001
Q_REL = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
"""yam_arena's ``gripper`` frame = our flange frame * Q_REL (fingers along the flange's -z)."""
TCP_EXTRA = 0.1347 - 0.134

# Documented layout (task.json setup; cutout_sim/usd/props.py), desk frame.
HINGE = np.array(P.HINGE)
RING = HINGE + np.array(P.ring_center_hinge())
STICK_RACK = np.array([P.STICK_BUTT[0] + 0.005, P.STICK_BUTT[1], P.STICK_AXIS_Z])
"""Butt 5 mm off the end stop."""
STICK_TOP_Z = P.STICK_AXIS_Z + P.STICK_HALF * MM
_A = math.radians(P.PRONG_SWEEP_DEG)
PRONG_DIR = np.array([-math.sin(_A), math.cos(_A), 0.0])
E_IN_STICK = np.array([P.PRONG_ROOT[0] * MM, 0.0, 0.0]) + 0.022 * PRONG_DIR
"""The prong point that should pass through the ring's centre (22 mm along the prong from its root)."""
GRIP_IN_STICK = (P.GRIP_X * MM, 0.0, 0.0)

PITCH_HOOK = 32.0
PITCH_END = -10.0
SIDE_OFF = 0.035
PULL_LEAD_DEG = 20.0
ARC_END_DEG = 115.0
LOOK = (0.32, (-0.4, 1.0, 0.4))
"""Left wrist camera: distance from the ring and direction (desk) it looks from."""


def stick_q(pitch_deg):
    """Stick orientation pitched tip-up by ``pitch_deg`` (axis along +x, prong toward +y)."""
    return q_axis((0, 1, 0), -math.radians(pitch_deg))


def arc_point(phi_deg, center=None, radius_scale=1.0):
    """The ring centre rotated by phi about the hinge axis, opening direction (desk)."""
    c = RING if center is None else center
    r = (c - HINGE) * radius_scale
    t = math.radians(phi_deg)
    return HINGE + np.array([r[0] * math.cos(t) - r[2] * math.sin(t), r[1], r[0] * math.sin(t) + r[2] * math.cos(t)])


class Arm:
    def __init__(self, side):
        self.side = side
        self.pos = None
        self.quat = None
        self.grip = OPEN
        self.q_cmd = None
        self.joint_target = None  # when set, a direct joint target (no IK)
        self.gen = None


class Demo:
    """``next_action(observation)`` gives the 14-D joint action for the next control step."""

    def __init__(self, mode: str = "demo", control_hz: float = 20.0, speed: float = 0.10, log=None):
        self.log = log or (lambda *a: print(*a, flush=True))
        self.mode = mode
        self.dt = 1.0 / control_hz
        self.speed = speed
        self.debug_every = int(os.environ.get("CUTOUT_DEMO_DEBUG", "0"))
        self.k = 0
        self.obs = None
        self.R = Arm("right")
        self.L = Arm("left")
        self.ring = None
        self.stick_y = P.STICK_BUTT[1]
        self.model = None
        self.worst = 0.0

    # -- observation helpers ---------------------------------------------------------------------------------------
    def steps(self, s):
        return max(1, round(s / self.dt))

    def qm(self, side):
        return V.split_state(self.obs.state["joint_pos"])[side]

    def tcp(self, side):
        """Measured (desk position, our flange quaternion) of an arm's grasp point, from FK of its joints."""
        p, r = V.tcp_desk(self.qm(side)[0], side)
        return p + r @ np.array([0.0, 0.0, TCP_EXTRA]), qmul(matrix_to_quat_xyzw(r), conj(Q_REL))

    def stick_est(self):
        """Predicted stick pose (butt position, quaternion) from the measured joints and the grasp offset."""
        g, gq = self.tcp("right")
        return g + rotate(gq, self.model.rel_pos), qmul(gq, self.model.rel_quat)

    def prong_est(self):
        p, q = self.stick_est()
        return p + rotate(q, E_IN_STICK)

    def img(self, name):
        return np.asarray(self.obs.images[name])

    # -- behaviours (generators: one yield per control step) -------------------------------------------------------
    def move(self, arm, pos, quat=None, speed=None, min_steps=3):
        pos = np.array(pos, float)
        quat = arm.quat if quat is None else np.array(quat, float)
        p0, q0 = arm.pos.copy(), arm.quat.copy()
        n = max(min_steps, int(math.ceil(np.linalg.norm(pos - p0) / ((speed or self.speed) * self.dt))))
        for i in range(1, n + 1):
            a = i / n
            arm.pos, arm.quat = p0 + a * (pos - p0), slerp(q0, quat, a)
            yield

    def hold(self, n):
        for _ in range(n):
            yield

    def grip(self, arm, value, s):
        arm.grip = value
        yield from self.hold(self.steps(s))

    def settle(self, arm, tol=0.002, max_s=2.0):
        for _ in range(self.steps(max_s)):
            if np.linalg.norm(self.tcp(arm.side)[0] - arm.pos) < tol:
                return
            yield
        self.log(f"SETTLE {arm.side}: lag {np.linalg.norm(self.tcp(arm.side)[0] - arm.pos) * 1000:.1f} mm")

    def stick_to(self, pos, quat, speed=None, min_steps=3):
        """Move the right arm so the (predicted) stick butt goes to ``pos`` with orientation ``quat``."""
        g, gq = self.model.grasp_for(np.asarray(pos), np.asarray(quat))
        yield from self.move(self.R, g, gq, speed=speed, min_steps=min_steps)

    def prong_to(self, target, pitch, speed=None, correction=np.zeros(3), min_steps=3):
        q = stick_q(pitch)
        butt = np.asarray(target) - rotate(q, E_IN_STICK) - correction
        yield from self.stick_to(butt, q, speed=speed, min_steps=min_steps)

    def servo_prong(self, target, pitch, tol=0.0008, tries=6):
        """Bring the predicted prong point onto ``target`` (closed loop on the measured joints)."""
        corr = np.zeros(3)
        for i in range(tries):
            yield from self.prong_to(target, pitch, speed=0.01, correction=corr)
            yield from self.hold(self.steps(0.4))
            yield from self.settle(self.R, tol=0.0015, max_s=1.5)
            err = self.prong_est() - target
            self.log(f"SERVO prong try {i}: error (mm) {np.round(err * 1000, 2)}")
            if np.max(np.abs(err)) < tol:
                break
            corr = corr + err
        return corr

    # -- the camera arm (left) -------------------------------------------------------------------------------------
    def left_plan(self):
        L = self.L
        dist, d = LOOK
        d = np.asarray(d, float) / np.linalg.norm(d)
        cam_pos = RING + dist * d
        f = -d
        z = -f
        x = np.cross((0.0, 0.0, 1.0), z)
        x /= np.linalg.norm(x)
        rc = np.stack([x, np.cross(z, x), z], 1)
        cam = station.wrist_camera("left")
        rg = rc @ quat_xyzw_to_matrix(np.asarray(cam.rot_xyzw)).T
        pg = cam_pos - rg @ np.asarray(cam.pos)
        tcp = pg + rg @ np.array([0.0, 0.0, 0.134]) + np.array([0.0, 0.0, V.DESK_Z])
        q_look, pe, re = ik.solve(tcp, matrix_to_quat_xyzw(rg), np.array([0.03, 1.37, 0.93, 0.69, 1.22, -0.23]),
                                  base_pos=station.BASES["left"], iters=300)
        self.log(f"LEFT look pose: IK error {pe * 1000:.1f} mm {math.degrees(re):.1f} deg")
        q0 = self.qm("left")[0].copy()
        n = self.steps(3.0)
        for i in range(1, n + 1):
            a = 0.5 - 0.5 * math.cos(math.pi * i / n)
            L.joint_target = q0 + a * (q_look - q0)
            yield
        yield from self.hold(self.steps(1.0))
        got = []
        for _ in range(12):
            q6 = self.qm("left")[0]
            if np.max(np.abs(q6 - q_look)) < 0.01 or len(got) or _ > 6:
                im = self.img("left_cam")
                cam = V.Cam.wrist("left", q6, im.shape[1], im.shape[0])
                r = V.locate_ring(im, cam, RING)
                if r is not None:
                    got.append(r["xz"])
                    self.log(f"LEFT CAM ring: centre x {r['xz'][0]:.4f} z {r['xz'][1]:.4f} (layout {RING[0]:.4f} "
                             f"{RING[2]:.4f}), ellipse {np.round(r['axes_px'], 1)} px, {r['px']} px")
                if len(got) >= 3:
                    break
            yield from self.hold(2)
        ring = RING.copy()
        if got:
            xz = np.mean(got, axis=0)
            if np.hypot(xz[0] - RING[0], xz[1] - RING[2]) < 0.02:
                ring[0], ring[2] = xz
            else:
                self.log("LEFT CAM ring estimate more than 2 cm from the layout: layout kept")
        else:
            self.log("LEFT CAM: ring not seen, layout kept")
        self.ring = ring
        self.log(f"RING centre used: {np.round(ring, 4)}")
        for i in range(1, n + 1):
            a = 0.5 - 0.5 * math.cos(math.pi * i / n)
            L.joint_target = q_look + a * (q0 - q_look)
            yield
        while True:
            yield

    # -- the stick arm (right) -------------------------------------------------------------------------------------
    def right_plan(self):
        R = self.R
        im = self.img("right_cam")
        cam = V.Cam.wrist("right", self.qm("right")[0], im.shape[1], im.shape[0])
        s = V.locate_stick(im, cam, STICK_TOP_Z, P.STICK_BUTT[1])
        if s is not None:
            y = s["y_at"](STICK_RACK[0] + P.GRIP_X * MM)
            off = float(np.clip(y - P.STICK_BUTT[1], -0.015, 0.015))
            self.stick_y = P.STICK_BUTT[1] + off
            self.log(f"RIGHT CAM stick: axis y {y:.4f} at the handgrip (layout {P.STICK_BUTT[1]:.4f}), yaw "
                     f"{s['yaw_deg']:.1f} deg, {s['n']} px")
        else:
            self.log("RIGHT CAM: stick not seen, layout kept")
        rack = STICK_RACK.copy()
        rack[0] = P.STICK_BUTT[0]
        rack[1] = self.stick_y
        nominal = GraspModel.nominal(DOWN, np.array([0.0, 0.0, 0.0, 1.0]), GRIP_IN_STICK)
        grasp, gq = nominal.grasp_for(rack, np.array([0.0, 0.0, 0.0, 1.0]))
        yield from self.grip(R, OPEN, 0.2)
        yield from self.move(R, grasp + np.array([0.0, 0.0, 0.08]), gq, speed=0.12, min_steps=self.steps(2.0))
        yield from self.move(R, grasp + np.array([0.0, 0.0, 0.01]), speed=0.04)
        yield from self.settle(R)
        err = self.tcp("right")[0] - R.pos
        yield from self.move(R, grasp - np.array([err[0], err[1], 0.0]), speed=0.02)
        yield from self.hold(self.steps(0.4))
        yield from self.grip(R, CLOSED, 1.2)
        # Measure the grasp: the stick is still lying in its cradles, at the documented pose (camera-corrected y).
        g, gq_m = self.tcp("right")
        self.model = GraspModel(rotate(conj(gq_m), rack - g), qmul(conj(gq_m), np.array([0.0, 0.0, 0.0, 1.0])))
        self.log(f"GRASP: stick butt in the hand (mm) {np.round(self.model.rel_pos * 1000, 1)}")
        if self.mode == "measure":
            return
        # Lift straight out of the cradles, then raise and pitch up beside the cutout.
        butt, q = self.stick_est()
        yield from self.stick_to(butt + np.array([0.0, 0.0, 0.08]), q, speed=0.06)
        while self.ring is None:
            yield
        ring = self.ring
        pre = ring - np.array([0.0, SIDE_OFF, 0.0])
        yield from self.prong_to(pre + np.array([-0.04, -0.03, -0.03]), PITCH_HOOK, speed=0.10,
                                 min_steps=self.steps(2.5))
        corr = yield from self.servo_prong(pre, PITCH_HOOK)
        # Through the ring: slide left along y.
        yield from self.prong_to(ring, PITCH_HOOK, speed=0.012, correction=corr)
        yield from self.hold(self.steps(0.6))
        e = self.prong_est() - ring
        self.log(f"HOOKED: predicted prong point from the ring centre (mm) {np.round(e * 1000, 1)}")
        # Pull: along the ring's arc about the hinge, slowly at first (the latch holds), then on, lowering the tube.
        n_pull = self.steps(2.0)
        for i in range(1, n_pull + 1):
            phi = PULL_LEAD_DEG * i / n_pull
            pitch = PITCH_HOOK + (PITCH_END - PITCH_HOOK) * phi / ARC_END_DEG
            g, gq = self.model.grasp_for(arc_point(phi, ring) - rotate(stick_q(pitch), E_IN_STICK) - corr,
                                         stick_q(pitch))
            R.pos, R.quat = g, gq
            yield
        n_arc = self.steps(3.0)
        for i in range(1, n_arc + 1):
            a = 0.5 - 0.5 * math.cos(math.pi * i / n_arc)
            phi = PULL_LEAD_DEG + (ARC_END_DEG - PULL_LEAD_DEG) * a
            pitch = PITCH_HOOK + (PITCH_END - PITCH_HOOK) * phi / ARC_END_DEG
            g, gq = self.model.grasp_for(arc_point(phi, ring) - rotate(stick_q(pitch), E_IN_STICK) - corr,
                                         stick_q(pitch))
            R.pos, R.quat = g, gq
            yield
        yield from self.hold(self.steps(0.8))
        # Unhook: slide the prong right, out of the ring, then down and back.
        end = arc_point(ARC_END_DEG, ring)
        yield from self.prong_to(end - np.array([0.0, 0.05, 0.0]), PITCH_END, speed=0.03, correction=corr)
        yield from self.prong_to(end - np.array([0.06, 0.08, -0.02]), PITCH_END, speed=0.06, correction=corr)
        yield from self.hold(self.steps(1.0))
        # Back to the rack.
        rack_pose = np.array([STICK_RACK[0], self.stick_y, STICK_RACK[2]])
        q0 = np.array([0.0, 0.0, 0.0, 1.0])
        yield from self.stick_to(rack_pose + np.array([0.0, 0.0, 0.09]), q0, speed=0.08, min_steps=self.steps(2.5))
        # Servo the predicted stick (butt position and heading) onto the rack line, 5 cm above it.
        corr, yaw_c = np.zeros(3), 0.0
        above = rack_pose + np.array([0.0, 0.0, 0.05])
        for i in range(5):
            yield from self.stick_to(above - corr, q_axis((0, 0, 1), -yaw_c), speed=0.02)
            yield from self.settle(self.R, tol=0.0015, max_s=1.5)
            p, q = self.stick_est()
            ax = rotate(q, (1.0, 0.0, 0.0))
            err = p - above
            yaw = math.atan2(ax[1], ax[0])
            self.log(f"RACK servo try {i}: stick butt error (mm) {np.round(err * 1000, 1)}, heading "
                     f"{math.degrees(yaw):.2f} deg, pitch {math.degrees(math.asin(ax[2])):.2f} deg")
            if np.max(np.abs(err[:2])) < 0.0015 and abs(yaw) < math.radians(0.4):
                break
            corr = corr + np.array([err[0], err[1], 0.0])
            yaw_c += yaw
        # Lower it into the cradles and press 6 mm further: whichever end touches first, the stick turns in the
        # fingers until it lies on both cradle floors (it may have slipped a few degrees in the hand while the tube
        # rested on the prong).
        yield from self.stick_to(rack_pose - corr + np.array([0.0, 0.0, -0.006]), q_axis((0, 0, 1), -yaw_c),
                                 speed=0.015)
        yield from self.hold(self.steps(0.6))
        yield from self.grip(R, OPEN, 1.0)
        yield from self.move(R, R.pos + np.array([0.0, 0.0, 0.08]), speed=0.05)
        yield from self.move(R, np.array([0.22, -0.30, 0.20]), speed=0.10)
        while True:
            yield

    def control_plan(self):
        """Control: the right gripper itself takes the pull ring (pinching its lower bar, fingers pointing left
        through the ring) and pulls it along its path, no stick."""
        R = self.R
        bar = RING + np.array([0.0, 0.0, -(P.RING_R_IN + P.RING_R_OUT) / 2 * MM])
        hand = qmul(q_axis((1, 0, 0), math.radians(90.0)), DOWN)
        yield from self.grip(R, OPEN, 0.2)
        yield from self.move(R, bar + np.array([0.0, -0.06, 0.0]), hand, speed=0.10, min_steps=self.steps(3.5))
        yield from self.move(R, bar, speed=0.02)
        yield from self.grip(R, CLOSED, 1.0)
        n = self.steps(3.0)
        for i in range(1, n + 1):
            R.pos = arc_point(90.0 * i / n, bar)
            yield
        while True:
            yield

    def hazards_plan(self):
        """Hazard checks: the left gripper goes into the keep-out (8 cm in front of the top contact) twice."""
        L = self.L
        top = HINGE + np.array(P.top_contact_hinge())
        q0 = self.qm("left")[0].copy()
        tgt = top + np.array([-0.08, 0.0, 0.0]) + np.array([0.0, 0.0, V.DESK_Z])
        q_in, pe, _ = ik.solve(tgt, None, np.array([-0.3, 1.6, 1.2, -0.8, 0.0, 0.0]), base_pos=station.BASES["left"],
                               iters=300)
        self.log(f"HAZARD keep-out pose: IK error {pe * 1000:.1f} mm")
        for start, end in ((160, 200), (240, 285)):
            while self.k < start:
                yield
            n = self.steps(1.5)
            for i in range(1, n + 1):
                L.joint_target = q0 + (q_in - q0) * i / n
                yield
            while self.k < end:
                yield
            for i in range(1, n + 1):
                L.joint_target = q_in + (q0 - q_in) * i / n
                yield
        while True:
            yield

    # -- the action ------------------------------------------------------------------------------------------------
    def _ik(self, arm):
        if arm.joint_target is not None and arm.gen is not None and arm.side == "left":
            return np.r_[arm.joint_target, arm.grip]
        pos = arm.pos + np.array([0.0, 0.0, V.DESK_Z])
        tcp = pos + rotate(arm.quat, (0.0, 0.0, -TCP_EXTRA))
        rot = qmul(arm.quat, Q_REL)
        q, pe, _ = ik.solve(tcp, rot, arm.q_cmd, base_pos=station.BASES[arm.side], iters=60)
        if pe > 0.001:
            for seed in (self.qm(arm.side)[0], np.array([0.0, 1.0, 1.0, -0.6, 0.0, 0.0]),
                         np.array([0.7, 1.5, 1.3, -1.2, 0.0, 0.7])):
                q2, pe2, _ = ik.solve(tcp, rot, seed, base_pos=station.BASES[arm.side], iters=200)
                if pe2 < pe and np.max(np.abs(q2 - arm.q_cmd)) < 0.5:
                    q, pe = q2, pe2
        arm.q_cmd = q
        if pe > 0.005 and pe > self.worst:
            self.log(f"IK {arm.side} k={self.k} error {pe * 1000:.1f} mm at {np.round(arm.pos, 4)}")
        self.worst = max(self.worst, pe)
        return np.r_[q, arm.grip]

    def next_action(self, obs) -> np.ndarray:
        self.obs = obs
        if self.k == 0:
            for arm in (self.R, self.L):
                arm.pos, arm.quat = self.tcp(arm.side)
                arm.q_cmd = self.qm(arm.side)[0].copy()
                arm.joint_target = arm.q_cmd.copy()
        if self.k == self.steps(0.5) and self.mode != "hold":
            if self.mode == "control":
                self.R.gen = self.control_plan()
            elif self.mode == "hazards":
                self.L.gen = self.hazards_plan()
            else:
                self.R.gen = self.right_plan()
                self.L.gen = self.left_plan()
        acts = {}
        for arm in (self.R, self.L):
            if arm.gen is not None:
                try:
                    next(arm.gen)
                except StopIteration:
                    arm.gen = iter(())
            if arm.gen is None:
                acts[arm.side] = np.r_[arm.joint_target, arm.grip]
            else:
                acts[arm.side] = self._ik(arm)
        if self.debug_every and self.k % self.debug_every == 0 and self.model is not None:
            p, q = self.stick_est()
            self.log(f"DBG demo k={self.k} prong {np.round(self.prong_est(), 4)} butt {np.round(p, 3)} "
                     f"axis {np.round(rotate(q, (1.0, 0.0, 0.0)), 3)}")
        self.k += 1
        return np.concatenate([acts["left"], acts["right"]])
