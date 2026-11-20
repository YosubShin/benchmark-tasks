"""The scripted demo, closed loop on robot-observable inputs only.

What it reads, every control step: the observation Inspect Robots hands any policy, i.e. the three RGB images
(``top_cam``, ``left_cam``, ``right_cam``) and the 14-D ``joint_pos`` state. Nothing else comes from the simulator:
no drum or star wheel angle, no drag value, no spoon pose, no scorer state. Fixed knowledge it uses: the documented
setup (task.json ``setup``: where the hub, the adjusting hole, the studs and the spoon's rests are; the star wheel's
size and depth behind the hole; the spoon's size; that teeth moved upward through the hole expand the shoes), the
station's published camera calibration, the arm's URDF and the i2rt drive gains (``yam_arena.constants``), as a real
robot's script would.

Plan (left arm = arm A turns the drum; right arm = arm B works the spoon):
1. A takes the wheel stud level with the axle on the robot's side, fingers pointing at the drum's face along the
   stud, jaws closing along the stud's path round the axle (the pads push the stud; no reliance on friction). B picks the spoon off its rests by the handle.
2. B carries the spoon to the hole, tilts it so its underside will bear on the hole's lower edge (the fulcrum),
   and slides the tip in along the spoon until it seats between two teeth (by feel: the gap narrows; the arm's
   push is soft and the tip stops where the teeth wedge it).
3. Drag check (feel): A turns the drum 25 degrees by the stud and holds, then turns it back and holds. In each hold
   the drum's Coulomb drag holds the arm off its command; the joint errors times the i2rt gains give the joint
   torques, the arm's Jacobian turns them into the wrench at the fingers and that into the torque about the axle.
   Half the difference between the two holds (opposite turning directions) is the drum's turning torque; arm
   gravity and friction offsets common to both cancel. The first check (shoes clear) is the baseline.
4. Strokes: B, pressing the spoon in, levers the handle down so the tip moves the tooth in front of it up along the
   wheel's circle by a tooth and a bit (21 degrees of the wheel; the pawl clicks at 15), slides the spoon straight
   back along itself until the tip is clear of the teeth, levers back and slides it in again.
   Two strokes between checks while the drag is at the baseline; one stroke per check once it has risen
   (contact). It stops when the drag is STOP_RISE above the baseline (light drag).
5. B draws the spoon out of the hole and lifts it away (still holding it); A lets go of the stud and moves away.

``control=True`` (the control run): after the baseline check it gives 24 strokes without feeling the drag, then
finishes the same way. ``DBA_DEMO_DEBUG=<n>`` prints estimates every n steps.
"""

from __future__ import annotations

import math
import os

import numpy as np

from yam_arena import ik, station
from yam_arena.ik import matrix_to_quat_xyzw

from drumbrake_sim import spec as S
from drumbrake_sim import vision as V
from drumbrake_sim.kit import CLOSED, DOWN, ROOT_OFFSET, ArmTimeline, conj, q_axis, qmul, rotate, slerp

Q_REL = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
TCP_EXTRA = 0.1347 - 0.134
GRIP_OPEN = 0.55
TRIM_GAIN, TRIM_MAX = 0.05, 0.06
KP = np.array([80.0, 80.0, 80.0, 10.0, 10.0, 10.0])
"""i2rt drive stiffness [N m / rad] (yam_arena.constants SHOULDER / WRIST)."""
R_SEAT = float(os.environ.get("DBA_R_SEAT", "0.0084"))
"""The circle the tip works on, just outside the tooth root (7.6 mm)."""
R_CLEAR = 0.0118
"""Tip radius when slid back between strokes: clear of the 10.1 mm tooth tips."""
PUSH = float(os.environ.get("DBA_PUSH", "0.0015"))
"""How much further in along the spoon the tip is commanded while seated and levering: the soft arm then presses it
into the gap firmly (as a technician leans the spoon into the teeth)."""
R_BACK = 0.0112
"""Tip radius while levering back (clear of the 10.1 mm tooth tips)."""
R_ENTRY = 0.0135
"""Tip radius where it is through the plate but short of the tooth tips."""
ENTRY_DZ = 0.0015
STROKE_DEG = float(os.environ.get("DBA_STROKE", "21.0"))
EASE_DEG = 4.0
STROKE_IN = float(os.environ.get("DBA_STROKE_IN", "0.0006"))
"""The tip drifts this much toward the wheel's axis over a stroke (keeps it bearing on the tooth's flank)."""
CHECK_DEG = 25.0
CHECK_MOVE_S, CHECK_HOLD_S = 1.0, 1.0
STOP_RISE = 0.25
CONTACT_RISE = 0.08
FAR_STROKES = 2
CONTROL_STROKES = 24
MAX_STROKES = 30
CAM_DEADBAND = 0.008
RED = ((245, 10), (150, 255), (40, 255))
"""The spoon's red dipped handle in HSV (OpenCV full range)."""
PRESS = float(os.environ.get("DBA_PRESS", "0.0"))
"""The blade's underside planned this far below the hole's lower edge, so it bears on it: the edge is the lever's
fulcrum and carries the tooth's reaction; the fingers then need no moment on the handle."""
SPOON_YAW = math.radians(90.0)


def q_mat(q):
    from yam_arena.urdf import quat_xyzw_to_matrix

    return quat_xyzw_to_matrix(np.asarray(q, dtype=float))


def down(yaw):
    return qmul(q_axis((0, 0, 1), yaw), DOWN)


Q_STUD0 = qmul(q_axis((1, 0, 0), -math.pi / 2), DOWN)
"""Hand on the stud at alpha = 0: fingers along -y (toward the drum's face, along the stud), jaws closing vertically,
i.e. along the stud's path round the axle: the turning force is carried by the pads' faces, not by friction."""


def stud_grasp(alpha_deg):
    """Grasp pose on the turning stud at angle alpha (the alpha = 0 hand turned with the drum about the axle)."""
    a = math.radians(alpha_deg)
    return np.array(S.stud_pos(alpha_deg)), qmul(q_axis((0, 1, 0), a), Q_STUD0)


def spoon_pose(theta_deg, r, dz=0.0, push=0.0):
    """Spoon pose (tip position, quaternion) with the tip on the wheel's circle of radius r at angle theta (up from
    the hole's direction), the blade's underside over the hole's lower edge; ``dz`` raises the whole line (entry);
    ``push`` puts the commanded tip that much further in along the spoon (the soft arm then presses the tip into the
    gap with a small steady force while it follows the circle)."""
    th = math.radians(theta_deg)
    tip = np.array([S.SW[0], S.SW[1] - r * math.cos(th), S.SW[2] + r * math.sin(th) + dz])
    f = np.array([S.SW[0], S.BP_Y, S.SLOT_Z0 + S.BLADE_T / 2 - PRESS + dz])
    d = tip - f
    phi = math.atan2(d[2], d[1])
    return tip + push * d / np.linalg.norm(d), q_axis((1, 0, 0), phi), phi


GRIP_FROM_END = float(os.environ.get("DBA_GRIP_FROM_END", str(S.GRIP_FROM_END)))
GRIP_LOCAL = np.array([0.0, -(S.SPOON_L - GRIP_FROM_END), 0.0])
"""Where the grasp point sits on the spoon (spoon frame); replaced by the measured grasp once the hand closes."""


def fulcrum():
    return np.array([S.SW[0], S.BP_Y, S.SLOT_Z0 + S.BLADE_T / 2 - PRESS])


def line_pose(phi, s):
    """Spoon pose with its centre line through the fulcrum at pitch phi and its tip s along the line from it."""
    return fulcrum() + s * np.array([0.0, math.cos(phi), math.sin(phi)]), q_axis((1, 0, 0), phi)


def line_of(theta_deg, r, push=0.0):
    """(pitch, distance from the fulcrum) of spoon_pose(theta, r, push=push)."""
    tip, _, phi = spoon_pose(theta_deg, r, push=push)
    return phi, float(np.linalg.norm(tip - fulcrum()))


def hand_for_spoon(tip, q_spoon):
    """Grasp-point pose that puts the spoon's tip at ``tip`` with orientation ``q_spoon``."""
    return tip + rotate(q_spoon, GRIP_LOCAL), qmul(q_spoon, down(SPOON_YAW))


class Demo:
    def __init__(self, control_hz: float = 20.0, speed: float = 0.10, control: bool = False, log=None):
        self.log = log or (lambda *a: print(*a, flush=True))
        self.dt = 1.0 / control_hz
        self.speed = speed
        self.control = bool(control)
        self.debug_every = int(os.environ.get("DBA_DEMO_DEBUG", "0") or 0)
        self.k = 0
        self.obs = None
        self.events: dict[int, list] = {}
        self.A = self.B = None
        self.q_cmd = {}
        self.q_sent = {}
        self.worst = 0.0
        self.trim = {"left": np.zeros(6), "right": np.zeros(6)}
        self.freeze = {"left": False, "right": False}
        self.q_prev = {}
        self.strokes = 0
        self.baseline = None
        self.contact = False
        self.samples = []
        self.checks = []
        self.ready = 0

    # -- helpers ---------------------------------------------------------------------------------------------------
    def hk(self, s):
        return max(1, round(s / self.dt))

    def q(self, side):
        return V.split_state(self.obs.state["joint_pos"])[side]

    def grasp_point(self, side, q6=None):
        p, r = V.tcp_desk(self.q(side)[0] if q6 is None else q6, side)
        return p + r @ np.array([0.0, 0.0, TCP_EXTRA]), qmul(matrix_to_quat_xyzw(r), conj(Q_REL))

    def at(self, step, fn):
        self.events.setdefault(step, []).append(fn)

    def now(self, fn):
        self.at(max(len(self.A.steps), len(self.B.steps)), fn)

    def sync(self):
        n = max(len(self.A.steps), len(self.B.steps))
        self.A.pad_to(n)
        self.B.pad_to(n)

    def when_settled(self, T, side, fn, tries=40, tol=0.0015):
        """Call fn once the arm's measured grasp point is within tol of its target (or after tries x 0.1 s)."""
        self.sync()

        def check():
            lag = float(np.linalg.norm(self.grasp_point(side)[0] - (T.pos + ROOT_OFFSET)))
            if lag < tol or check.tries <= 0:
                if lag >= tol:
                    self.log(f"SETTLE {side} timeout: lag {lag * 1000:.1f} mm")
                fn()
            else:
                check.tries -= 1
                T.hold(self.hk(0.1))
                self.sync()
                self.now(check)

        check.tries = tries
        self.now(check)

    def solve_side(self, side, pos_world, quat, seed):
        from yam_arena.ik import LOWER, UPPER

        tcp = np.asarray(pos_world, dtype=float) + q_mat(quat) @ np.array([0.0, 0.0, -TCP_EXTRA]) + \
            np.array([0.0, 0.0, V.DESK_Z])
        sols = []
        seed = np.asarray(seed, dtype=float)
        for sd in (seed, np.array([0.0, 1.0, 1.0, -0.6, 0.0, 0.0]), np.array([-0.5, 1.9, 1.3, -0.9, 0.0, 0.0]),
                   np.array([0.5, 1.9, 1.3, -0.9, 0.0, 0.0]), np.array([0.6, 1.6, 1.4, -1.0, 0.0, 1.0]),
                   np.array([0.6, 1.6, 1.4, -1.0, 0.0, -1.0])):
            q, pe, re = ik.solve(tcp, qmul(quat, Q_REL), sd, base_pos=station.BASES[side], iters=200)
            sols.append((q, pe, pe + 0.02 * re))
            if sd is seed and pe + 0.02 * re < 0.0015:
                break
        good = [t for t in sols if t[2] < 0.0015]
        if good:
            def key(t):
                margin = min(np.min(t[0] - LOWER), np.min(UPPER - t[0]))
                return -min(margin, 0.25) * 4 + np.linalg.norm(t[0] - seed) * 0.2
            q, pe, _ = min(good, key=key)
            return q, pe
        q, pe, _ = min(sols, key=lambda t: t[2])
        return q, pe

    def jmove(self, T, side, pos_world, quat, seconds):
        q0 = getattr(T, "last_q", None)
        if q0 is None:
            q0, _ = self.solve_side(side, T.pos + ROOT_OFFSET, T.quat, self.q_cmd[side])
        q1, pe = self.solve_side(side, pos_world, quat, q0)
        if pe > 0.002:
            self.log(f"JMOVE {side}: target off by {pe * 1000:.1f} mm")
        n = self.hk(seconds)
        for i in range(1, n + 1):
            a = i / n
            a = a * a * (3 - 2 * a)
            T.steps.append((None, q0 + a * (q1 - q0), T.grip))
        T.pos, T.quat, T.last_q = np.asarray(pos_world, dtype=float) - ROOT_OFFSET, np.asarray(quat, dtype=float), q1

    def cpath(self, T, poses, seconds):
        """Cartesian path through (world position, quaternion) poses, interpolated smoothly, IK per step."""
        n = self.hk(seconds)
        out = []
        for i in range(1, n + 1):
            a = i / n
            a = a * a * (3 - 2 * a)
            out.append(poses(a))
        T.path(out)

    # -- drag sensing ------------------------------------------------------------------------------------------------
    def wrench_torque(self):
        """Torque [N m] the left hand applies to the drum about +y, from joint errors x gains (proprioception)."""
        q = self.q("left")[0]
        tau = KP * (self.q_sent["left"] - q)
        p0, r0 = V.tcp_desk(q, "left")
        jac = np.zeros((6, 6))
        eps = 1e-4
        for i in range(6):
            dq = q.copy()
            dq[i] += eps
            p1, r1 = V.tcp_desk(dq, "left")
            jac[:3, i] = (p1 - p0) / eps
            w = (r1 @ r0.T - np.eye(3)) / eps
            jac[3:, i] = [w[2, 1], w[0, 2], w[1, 0]]
        wrench = np.linalg.lstsq(jac.T, tau, rcond=None)[0]
        f, m = wrench[:3], wrench[3:]
        arm = p0 - np.array(S.AXLE)
        return float(np.cross(arm, f)[1] + m[1])

    def sample(self):
        self.samples.append(self.wrench_torque())

    def sampler(self, T, seconds):
        """Hold seconds; sample the torque every step of the last half."""
        n = self.hk(seconds)
        start = max(len(self.A.steps), len(self.B.steps))
        T.hold(n)
        self.sync()
        for i in range(n // 2, n):
            self.at(start + i, self.sample)

    # -- perception ------------------------------------------------------------------------------------------------
    def find_handle(self):
        """Top camera: the spoon's red handle, back-projected onto the plane of its top face, gives where the
        spoon lies (x across, y along); returns the tip position it implies, or None."""
        im = np.asarray(self.obs.images["top_cam"])
        cam = V.Cam.top(im.shape[1], im.shape[0])
        bl = V.blobs(V.mask_range(im, *RED), min_px=3)
        if not bl:
            return None
        vs, us = np.nonzero(bl[0]["mask"])
        pts = np.array([cam.on_plane((u + 0.5, v + 0.5), S.SPOON_Z0 + S.HANDLE_T / 2) for u, v in zip(us, vs)])
        x = float(np.median(pts[:, 0]))
        y_join = float(pts[:, 1].max())  # the handle's end at the blade (lit; its free end lies in the arm's shadow)
        tip = np.array([x, y_join + (S.SPOON_L - S.HANDLE_L), S.SPOON_Z0])
        return tip, len(pts), pts[:, 1].max() - pts[:, 1].min()

    # -- plan ------------------------------------------------------------------------------------------------------
    def start(self):
        A, B = self.A, self.B
        # A: to the stud along the finger axis.
        p, q = stud_grasp(S.STUD_ALPHA0)
        A.grip = GRIP_OPEN
        self.jmove(A, "left", p + np.array([0.0, 0.06, 0.0]), q, 2.2)
        A.move(p, speed=0.02)
        A.hold(self.hk(0.3))
        # B: over the spoon's handle, fingers down, closing across it (where the top camera sees it).
        tip0 = np.array(S.spoon_start())
        seen = self.find_handle()
        # The camera checks the layout: at 320x176 a pixel is about 4.5 mm here, so a shift is only acted on beyond
        # CAM_DEADBAND (a spoon put back off its rests); otherwise the documented place stands.
        if seen is None:
            self.log("TOP CAM: spoon handle not seen; using the layout")
        else:
            off = float(np.linalg.norm(seen[0][:2] - tip0[:2]))
            self.log(f"TOP CAM: spoon handle {seen[1]} px; tip at {np.round(seen[0], 4)}, {off * 1000:.1f} mm from the "
                     f"layout's {np.round(tip0, 4)}")
            if CAM_DEADBAND < off < 0.05:
                tip0 = seen[0]
                self.log("TOP CAM: spoon moved; picking it where seen")
        g, qg = hand_for_spoon(tip0, np.array([0.0, 0.0, 0.0, 1.0]))
        B.grip = GRIP_OPEN
        self.jmove(B, "right", g + np.array([0.0, 0.0, 0.07]), qg, 2.2)
        B.move(g, speed=0.02)
        B.hold(self.hk(0.3))
        self.tip0 = tip0
        self.when_settled(A, "left", self.grip_stud, tol=0.0015)

    def grip_stud(self):
        self.A.grip_to(CLOSED, self.hk(0.6))
        self.when_settled(self.B, "right", self.pick_spoon, tol=0.0006, tries=60)

    def pick_spoon(self):
        """Close on the handle. The spoon lies on its rests at a known height and is centred across by the fingers,
        so where it sits in the hand follows from where the hand's joints say the grasp point is as it closes."""
        global GRIP_LOCAL
        B = self.B
        g = self.grasp_point("right")[0]
        rel = g - self.tip0
        GRIP_LOCAL = np.array([0.0, rel[1], rel[2]])
        self.log(f"GRASP: hand at {np.round(g, 4)}; spoon grasp point at {np.round(GRIP_LOCAL, 4)} on the spoon "
                 f"(nominal {np.round([0.0, S.GRIP_LOCAL_Y, 0.0], 4)})")
        B.grip_to(CLOSED, self.hk(0.8))
        g = B.pos + ROOT_OFFSET
        B.move(g + np.array([0.0, 0.0, 0.03]), speed=0.03)
        # Over to the hole's line, the tip still 3 cm behind the plate, then tilt to the approach angle.
        tip, qs, phi = spoon_pose(0.0, R_SEAT)
        out = tip - 0.030 * rotate(qs, (0.0, 1.0, 0.0))
        h, qh = hand_for_spoon(out + np.array([0.0, -0.01, 0.03]), qs)
        self.jmove(B, "right", h, qh, 2.2)
        h, qh = hand_for_spoon(out, qs)
        B.move(h, qh, speed=0.02)
        B.hold(self.hk(0.4))
        self.log(f"SPOON approach: tilt {math.degrees(phi):.1f} deg")
        self.when_settled(B, "right", self.insert, tol=0.0015)

    def enter(self, fast=False):
        """Tip through the hole 1.5 mm above the lever line (clear of the hole's lower corner) until it is past the
        plate, down onto the line, then along it into the teeth, pressing it into whichever gap it meets."""
        B = self.B
        tip_h, qh, _ = spoon_pose(0.0, R_ENTRY, dz=ENTRY_DZ)
        out = tip_h - 0.030 * rotate(qh, (0.0, 1.0, 0.0))
        k = 0.6 if fast else 1.0
        self.cpath(B, lambda a: hand_for_spoon(out + a * (tip_h - out), qh), 1.6 * k)
        self.cpath(B, lambda a: hand_for_spoon(*spoon_pose(0.0, R_ENTRY, dz=ENTRY_DZ * (1 - a))[:2]), 0.5 * k)
        self.cpath(B, lambda a: hand_for_spoon(*spoon_pose(0.0, R_ENTRY + a * (R_SEAT - R_ENTRY), push=a * PUSH)[:2]), 0.8 * k)
        return out, qh

    def insert(self):
        self.enter()
        self.B.hold(self.hk(0.5))
        self.sync()
        self.now(self.check)

    def check(self):
        """Drag check: turn the drum CHECK_DEG by the stud (toward smaller alpha), hold, turn back, hold."""
        A = self.A
        self.freeze["left"] = True
        a0 = S.STUD_ALPHA0
        self.samples = []
        self.cpath(A, lambda a: stud_grasp(a0 - a * CHECK_DEG), CHECK_MOVE_S)
        self.sampler(A, CHECK_HOLD_S)
        self.now(self._mid_check)

    def _mid_check(self):
        A = self.A
        self.fwd = float(np.median(self.samples)) if self.samples else 0.0
        self.samples = []
        a0 = S.STUD_ALPHA0
        self.cpath(A, lambda a: stud_grasp(a0 - CHECK_DEG + a * CHECK_DEG), CHECK_MOVE_S)
        self.sampler(A, CHECK_HOLD_S)
        self.now(self._end_check)

    def _end_check(self):
        back = float(np.median(self.samples)) if self.samples else 0.0
        t = (back - self.fwd) / 2.0
        self.checks.append((self.strokes, t))
        if self.baseline is None:
            self.baseline = t
        rise = t - self.baseline
        self.log(f"DRAG CHECK after {self.strokes} strokes: turning torque {t:.3f} N m (pull {self.fwd:+.3f}, "
                 f"push back {back:+.3f}); rise {rise:+.3f} over the baseline {self.baseline:.3f}")
        self.freeze["left"] = False
        if not self.control and rise >= STOP_RISE:
            self.log(f"LIGHT DRAG: stop after {self.strokes} strokes")
            self.now(self.finish)
            return
        if rise >= CONTACT_RISE:
            self.contact = True
        if self.strokes >= MAX_STROKES or (self.control and self.strokes >= CONTROL_STROKES):
            self.now(self.finish)
            return
        n = CONTROL_STROKES if self.control else (1 if self.contact else FAR_STROKES)
        self.todo = n
        self.now(self.stroke)

    def stroke(self):
        """One notch: lever the handle so the tip moves up the wheel's circle by a tooth and a bit, pushing the tooth in
        front of it; slide the spoon straight back along itself until the tip is clear of the teeth, lever back, and
        slide it in again so it takes the next tooth."""
        B = self.B
        self.strokes += 1
        self.cpath(B, lambda a: hand_for_spoon(*spoon_pose(a * STROKE_DEG, R_SEAT, push=PUSH)[:2]), 1.0)
        B.hold(self.hk(0.1))
        phi1, s1 = line_of(STROKE_DEG, R_SEAT, push=PUSH)
        phi0, s0 = line_of(0.0, R_SEAT, push=PUSH)
        _, s_clear = line_of(0.0, R_CLEAR)
        self.cpath(B, lambda a: hand_for_spoon(*line_pose(phi1, s1 + a * (s_clear - s1))), 0.4)
        self.cpath(B, lambda a: hand_for_spoon(*line_pose(phi1 + a * (phi0 - phi1), s_clear)), 0.5)
        self.cpath(B, lambda a: hand_for_spoon(*line_pose(phi0, s_clear + a * (s0 - s_clear))), 0.4)
        B.hold(self.hk(0.1))
        self.sync()
        self.todo -= 1
        if self.todo > 0:
            self.now(self.stroke)
        else:
            self.now(self.check)

    def finish(self):
        A, B = self.A, self.B
        tip, qs, _ = spoon_pose(0.0, R_BACK)
        out = tip - 0.060 * rotate(qs, (0.0, 1.0, 0.0))
        self.cpath(B, lambda a: hand_for_spoon(tip + a * (out - tip), qs), 1.5)
        h, qh = hand_for_spoon(out + np.array([0.0, -0.03, 0.06]), np.array([0.0, 0.0, 0.0, 1.0]))
        self.jmove(B, "right", h, qh, 2.0)
        A.grip_to(GRIP_OPEN, self.hk(0.6))
        p, q = stud_grasp(S.STUD_ALPHA0)
        A.move(p + np.array([0.0, 0.07, 0.0]), speed=0.04)
        self.jmove(A, "left", p + np.array([-0.08, 0.15, 0.06]), q, 2.0)
        A.hold(self.hk(12.0))
        B.hold(self.hk(12.0))
        self.sync()
        self.log(f"FINISH: {self.strokes} strokes; checks {[(s, round(t, 3)) for s, t in self.checks]}")

    # -- action ------------------------------------------------------------------------------------------------------
    def _ik(self, side, pos_root, quat, grip):
        pos = pos_root + ROOT_OFFSET + np.array([0.0, 0.0, V.DESK_Z])
        tcp = pos + q_mat(quat) @ np.array([0.0, 0.0, -TCP_EXTRA])
        rot = qmul(quat, Q_REL)
        q, pe, _ = ik.solve(tcp, rot, self.q_cmd[side], base_pos=station.BASES[side], iters=60)
        if pe > 0.001:
            for seed in (self.q(side)[0], np.array([0.0, 1.0, 1.0, -0.6, 0.0, 0.0])):
                q2, pe2, _ = ik.solve(tcp, rot, seed, base_pos=station.BASES[side], iters=200)
                if pe2 < pe and np.max(np.abs(q2 - self.q_cmd[side])) < 0.6:
                    q, pe = q2, pe2
        if pe > 0.02:
            q = self.q_cmd[side]
        self.q_cmd[side] = q
        if pe > 0.005 and pe > self.worst:
            self.log(f"IK {side} k={self.k} error {pe * 1000:.1f} mm at {np.round(pos_root + ROOT_OFFSET, 4)}")
            self.worst = max(self.worst, pe)
        return np.r_[q, grip]

    def next_action(self, obs) -> np.ndarray:
        self.obs = obs
        if self.A is None:
            for side in ("left", "right"):
                self.q_cmd[side] = self.q(side)[0].copy()
                self.q_sent[side] = self.q_cmd[side].copy()
            ga, qa = self.grasp_point("left")
            gb, qb = self.grasp_point("right")
            self.A = ArmTimeline(ga - ROOT_OFFSET, qa, self.dt, self.speed)
            self.B = ArmTimeline(gb - ROOT_OFFSET, qb, self.dt, self.speed)
            self.A.grip = self.B.grip = 1.0
            self.A.hold(self.hk(0.5))
            self.B.hold(self.hk(0.5))
            self.at(self.hk(0.5), self.start)
        while self.k in self.events:
            for fn in self.events.pop(self.k):
                fn()
        for T in (self.A, self.B):
            if self.k >= len(T.steps):
                T.hold(self.k + 1 - len(T.steps))
        acts = []
        for side, T in (("left", self.A), ("right", self.B)):
            st = T.steps[self.k]
            if st[0] is None:
                if st[1] is not None:
                    self.q_cmd[side] = np.asarray(st[1], dtype=float)
                acts.append(np.r_[self.q_cmd[side], st[2]])
            else:
                acts.append(self._ik(side, *st))
        for i, side in enumerate(("left", "right")):
            q_des = acts[i][:6]
            prev = self.q_prev.get(side)
            if prev is not None and not self.freeze[side]:
                still = np.abs(q_des - prev) < 0.0005
                err = q_des - self.q(side)[0]
                self.trim[side] = np.where(still, np.clip(self.trim[side] + TRIM_GAIN * err, -TRIM_MAX, TRIM_MAX),
                                           self.trim[side])
            self.q_prev[side] = q_des.copy()
            acts[i] = np.r_[q_des + self.trim[side], acts[i][6]]
            self.q_sent[side] = acts[i][:6].copy()
        act = np.concatenate(acts)
        if self.debug_every and self.k % self.debug_every == 0:
            gb = self.grasp_point("right")[0]
            ga = self.grasp_point("left")[0]
            self.log(f"DBG k={self.k} A {np.round(ga, 4)} B {np.round(gb, 4)} torque {self.wrench_torque():+.3f}")
        self.k += 1
        return act
