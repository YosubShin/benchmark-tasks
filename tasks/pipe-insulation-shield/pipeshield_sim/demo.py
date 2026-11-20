"""The scripted demo, closed loop on robot-observable inputs only.

What it reads, every control step: the observation Inspect Robots hands any policy, i.e. the three RGB images
(``top_cam``, ``left_cam``, ``right_cam``) and the 14-D ``joint_pos`` state. Nothing else comes from the simulator:
no pipe or shield pose, no scorer state. Fixed knowledge it uses: the documented setup (task.json ``setup``: where
the pipe, the hangers and the shield's post are; the pipe's, shield's and hanger's sizes), the station's published camera
calibration and the arm's URDF (``yam_arena.station``, ``yam_arena.urdf``), as a real robot's script would.

Plan (left arm = arm A holds the pipe; right arm = arm B handles the shield):
1. Top camera: the red tape round the shield's middle gives where the shield lies along the pipe on its post.
2. Arm A takes the pipe 21 cm left of hanger A (fingers tilted 60 degrees forward, closed to a light squeeze on the
   insulation) and raises it there: the pipe pivots on hanger B, so it comes up about 11 mm at hanger A, enough for
   the 1.2 mm shield to pass; the rise comes from arm A's joints (forward kinematics) and the documented hangers.
3. Arm B opens its fingers wide and slides them, level and pointing forward, under the shield either side of its post
   (a two-tine fork across its underside), pitches the hand up 18 degrees (the fingers' upper edges then lie about
   level), lifts the shield off the post, carries it over to the right, down below the pipe's underside and forward
   in under the pipe to the right of hanger A, and raises it (pitched 19 degrees) to 3 mm above the hanger's strap,
   checking the height from its joints (the arm sags a few mm under the load).
4. It slides the shield along the pipe into the hanger until its front end is 2 cm past the strap, and sets it
   down: its front part on the strap, the rest on the fingers.
5. The fingers cannot follow it past the strap, so the shield is pushed the rest of the way by its end edge: arm A
   lowers the pipe onto the shield's front part (the pipe's weight holds it), arm B drops its fingers, moves back
   along the shield and comes up again rolled 5 degrees, the leading finger under the shield and the trailing one
   (higher) behind its end edge; arm A raises the pipe again; arm B pushes the shield along until, by its joints,
   the shield's middle is on the hanger's centre line.
6. Top camera: the red tape, seen on the shield's near side under the pipe, against the hanger's centre line; a short
   extra push if the shield is short.
7. Arm A lowers the pipe onto the shield and lets go; arm B lowers its fingers away from the shield and draws them
   out. Both arms move clear.

``-P no_centre=1`` (control) leaves out steps 5 and 6: the shield is left where step 4 put it (about 12 cm off
centre, its front part on the strap) and the pipe is lowered onto it there. ``PIS_DEMO_DEBUG=<n>`` prints estimates every n steps.

A small joint-space integral trim (proprioception only) removes the station arms' steady error of a few hundredths of
a radian under their soft gains.
"""

from __future__ import annotations

import math
import os

import numpy as np

from yam_arena import ik, station
from yam_arena.ik import matrix_to_quat_xyzw

from pipeshield_sim import spec as S
from pipeshield_sim import vision as V
from pipeshield_sim.kit import DOWN, FORWARD, ROOT_OFFSET, ArmTimeline, conj, q_axis, qmul

Q_REL = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
TCP_EXTRA = 0.1347 - 0.134
RED = ((235, 12), (120, 255), (60, 255))
TRIM_GAIN, TRIM_MAX = 0.05, 0.06
FORK_PITCH = math.radians(18.0)
"""The fingers point forward and 15 degrees up: their upper edges (which rise toward the hand near the tips) are then
about level, peaking 11 mm behind the grasp point, and the wrist camera stays clear of the shield's near rim."""
INSERT_PITCH = math.radians(float(os.environ.get("PIS_INSERT_PITCH", "19.3")))
"""Under the pipe the hand pitches a little more, which levels the shield on the fingers for the hanger's 2.7 mm a side."""
FORK_TOP = float(os.environ.get("PIS_FORK_TOP", "0.0125"))
"""Height of the fingers' upper edges (their peak) above the grasp point, in the pitched fork pose."""
FORK_BACK = float(os.environ.get("PIS_FORK_BACK", "0.011"))
SEAT_BACK = 0.019
"""Where the shield's axis ends up behind the grasp point once it is lifted (it settles 4 mm further back on the
fingers' edges than where they went in under it, measured in development runs); used to put its axis under the pipe's.""""""The peak (where the shield's underside sits) is this far behind the grasp point along x."""
FINGER_HALF = 0.052
"""Outer face of each finger from the grasp point, fingers wide open (pads 94 mm apart, fingers about 5 mm thick)."""
GRIP_Y = 0.210
"""Where arm A takes the pipe (left of hanger A, clear of the shield's 152 mm half-length)."""
TILT = math.radians(60.0)
LIFT_HAND = 0.022
"""Rise at arm A's grip; at hanger A it is LIFT_HAND x (A - B) / (grip - B) (the pipe pivots on hanger B)."""
PIPE_GRIP = 0.62
"""Gripper command closing on the 72 mm insulation (0 closed .. 1 = 94 mm): a light squeeze, not a crush."""
PUSH_ROLL = math.radians(5.0)
"""Roll about the fingers' axis for the push: the trailing finger 8.6 mm higher than the leading one, so it meets the
shield's end edge while the leading finger carries the shield's underside."""
PAD_HALF = 0.047
"""Each finger's inner face from the grasp point, fingers wide open (pads 94-98 mm apart)."""
TUCK_Q = (0.0, 0.05, 0.05, 0.0, 0.0, 0.0)


def q_mat(q):
    from yam_arena.urdf import quat_xyzw_to_matrix

    return quat_xyzw_to_matrix(np.asarray(q, dtype=float))


def pipe_hand_quat():
    """Arm A: fingers pointing forward and 60 degrees down, closing across the pipe (in the x-z plane)."""
    return qmul(q_axis((0, 1, 0), -TILT), qmul(q_axis((0, 0, 1), math.pi / 2), DOWN))


def fork_quat(roll=0.0, pitch=None):
    """Arm B: fingers pointing forward (+x) and FORK_PITCH up, spread along the pipe; roll about the fingers' axis."""
    pitch = FORK_PITCH if pitch is None else pitch
    return qmul(q_axis((0, 1, 0), -pitch), qmul(FORWARD, q_axis((0, 0, 1), roll)))


class Demo:
    def __init__(self, control_hz: float = 20.0, speed: float = 0.10, no_centre: bool = False, log=None, **_):
        self.log = log or (lambda *a: print(*a, flush=True))
        self.dt = 1.0 / control_hz
        self.speed = speed
        self.no_centre = bool(no_centre)
        self.debug_every = int(os.environ.get("PIS_DEMO_DEBUG", "0") or 0)
        self.k = 0
        self.obs = None
        self.events: dict[int, list] = {}
        self.A = self.B = None
        self.q_cmd = {}
        self.worst = 0.0
        self.trim = {"left": np.zeros(6), "right": np.zeros(6)}
        self.q_prev = {}
        self.shield_xy = np.array([S.SH0_X, S.SH0_Y])
        self.roll_sign = 1.0
        self.corrections = 0

    # -- helpers ---------------------------------------------------------------------------------------------------
    def hk(self, s):
        return max(1, round(s / self.dt))

    def q(self, side):
        return V.split_state(self.obs.state["joint_pos"])[side]

    def img(self, name):
        return np.asarray(self.obs.images[name])

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

    def when_settled(self, fn, side="right", tries=40, tol=0.0015, axis=None):
        """Call fn once the arm's measured grasp point is within tol of its target (or after tries x 0.1 s)."""
        self.sync()
        T = self.B if side == "right" else self.A

        def check():
            err = self.grasp_point(side)[0] - (T.pos + ROOT_OFFSET)
            lag = abs(err[axis]) if axis is not None else float(np.linalg.norm(err))
            if lag < tol or check.tries <= 0:
                if lag >= tol:
                    self.log(f"SETTLE {side} timeout: lag {lag * 1000:.1f} mm; target {np.round(T.pos + ROOT_OFFSET, 4)}"
                             f", at {np.round(self.grasp_point(side)[0], 4)}")
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
        for sd in (seed, np.array([0.0, 1.0, 1.0, -0.6, 0.0, 0.0]), np.array([0.0, 1.6, 1.4, -0.2, 0.0, 0.0]),
                   np.array([-0.5, 1.9, 1.3, -0.9, 0.0, 0.0]), np.array([0.5, 1.9, 1.3, -0.9, 0.0, 0.0])):
            q, pe, re = ik.solve(tcp, qmul(quat, Q_REL), sd, base_pos=station.BASES[side], iters=200)
            sols.append((q, pe, pe + 0.02 * re))
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
        return q1

    def jq(self, T, side, q1, seconds):
        q0 = getattr(T, "last_q", None)
        if q0 is None:
            q0, _ = self.solve_side(side, T.pos + ROOT_OFFSET, T.quat, self.q_cmd[side])
        q1 = np.asarray(q1, dtype=float)
        n = self.hk(seconds)
        for i in range(1, n + 1):
            a = i / n
            a = a * a * (3 - 2 * a)
            T.steps.append((None, q0 + a * (q1 - q0), T.grip))
        p, r = V.tcp_desk(q1, side)
        T.pos = p + r @ np.array([0.0, 0.0, TCP_EXTRA]) - ROOT_OFFSET
        T.quat, T.last_q = qmul(matrix_to_quat_xyzw(r), conj(Q_REL)), q1

    def grip(self, T, value, seconds):
        T.grip = value
        q = getattr(T, "last_q", None)
        for _ in range(self.hk(seconds)):
            T.steps.append((None, q, T.grip) if q is not None else (T.pos.copy(), T.quat.copy(), T.grip))

    def lin(self, T, pos_world, speed, quat=None):
        T.move(np.asarray(pos_world, dtype=float), quat=quat, speed=speed)

    # -- perception ------------------------------------------------------------------------------------------------
    def red_pixels(self, cam_name="top_cam"):
        im = self.img(cam_name)
        bl = V.blobs(V.mask_range(im, *RED), min_px=2)
        if not bl:
            return None, None
        vs, us = np.nonzero(bl[0]["mask"])
        return np.stack([us + 0.5, vs + 0.5], axis=1), int(bl[0]["area"])

    def find_shield_on_post(self):
        """The red tape round the shield's middle, back-projected onto a level plane halfway up the shield: its
        middle along the pipe (across, the documented position of the post is used)."""
        uv, area = self.red_pixels()
        if uv is None:
            return None
        im = self.img("top_cam")
        cam = V.Cam.top(im.shape[1], im.shape[0])
        z = S.SPINE0 + S.SH_RO * 0.45
        pts = np.array([cam.on_plane(p, z) for p in uv])
        return float(np.median(pts[:, 1])), area

    def shield_mark_in_hanger(self):
        """Under the pipe: the red line's visible part (on the shield's near outside, below the pipe's near side),
        back-projected onto the vertical plane x = pipe axis - shield radius: the shield's middle along the pipe."""
        uv, area = self.red_pixels()
        if uv is None:
            return None
        im = self.img("top_cam")
        cam = V.Cam.top(im.shape[1], im.shape[0])
        x_plane = S.PIPE_X - S.SH_RO * 0.95
        ys = []
        for p in uv:
            d = cam.ray(p)
            t = (x_plane - cam.p[0]) / d[0]
            ys.append(cam.p[1] + t * d[1])
        return float(np.median(ys)), area

    # -- plan: arm A (pipe) ------------------------------------------------------------------------------------------
    def pipe_lift_at_a(self):
        """The pipe's rise at hanger A from arm A's measured grasp point (the pipe pivots on hanger B)."""
        g = self.grasp_point("left")[0]
        rise = g[2] - self.grip_z0
        return rise * (S.HANGER_A_Y - S.HANGER_B_Y) / (GRIP_Y - S.HANGER_B_Y)

    def start(self):
        r = self.find_shield_on_post()
        if r is None:
            self.log("TOP CAM: shield's red line not found; using the layout")
        else:
            self.shield_xy = np.array([S.SH0_X, r[0]])
            self.log(f"TOP CAM: shield's centre line at y {r[0]:.4f} ({r[1]} px; layout {S.SH0_Y:.4f})")
        # Arm A: approach along the fingers' axis, take the pipe, raise it.
        A = self.A
        qa = pipe_hand_quat()
        fdir = np.array([math.sin(TILT), 0.0, -math.cos(TILT)])
        self.grip_z0 = S.PIPE_Z_BARE
        g = np.array([S.PIPE_X, GRIP_Y, self.grip_z0])
        A.grip = 1.0
        p0 = A.pos + ROOT_OFFSET
        self.jmove(A, "left", np.array([p0[0] - 0.06, p0[1], 0.44]), A.quat, 1.2)
        self.jmove(A, "left", g - 0.09 * fdir + np.array([0.0, 0.0, 0.03]), qa, 2.5)
        self.jmove(A, "left", g - 0.09 * fdir, qa, 0.8)
        self.jmove(A, "left", g, qa, 2.0)
        A.hold(self.hk(0.4))
        self.grip(A, PIPE_GRIP, 0.8)
        self.jmove(A, "left", g + np.array([0.0, 0.0, LIFT_HAND]), qa, 2.0)
        A.hold(self.hk(0.5))
        # Arm B (meanwhile): fork under the shield either side of its post.
        B = self.B
        sx, sy = self.shield_xy
        B.grip = 1.0
        zf = S.SPINE0 - FORK_TOP - float(os.environ.get("PIS_FORK_BELOW", "0.012"))
        p0 = B.pos + ROOT_OFFSET
        self.jmove(B, "right", np.array([p0[0] - 0.06, p0[1], 0.40]), B.quat, 1.2)
        # Level fingers reach close in; they pass under the shield (their edges below its underside), then the hand
        # pitches up in place.
        level = fork_quat(pitch=0.0)
        self.jmove(B, "right", np.array([sx - 0.09, sy - 0.06, zf + 0.12]), level, 2.0)
        self.jmove(B, "right", np.array([sx - 0.09, sy, zf]), level, 1.5)
        self.lin(B, np.array([sx + FORK_BACK, sy, zf]), 0.04)
        B.hold(self.hk(0.3))
        self.when_settled(self.lift_shield, tol=0.002)

    def lift_shield(self):
        """Lift the shield off its post until the fingers clear the post's top (it is not under the pipe yet), move
        it to the right, lower it below the pipe's underside, carry it forward in under the
        pipe to the right of hanger A."""
        B = self.B
        sx, sy = self.shield_xy
        x = sx + FORK_BACK
        z1 = S.SPINE0 - FORK_TOP + 0.010
        self.jmove(B, "right", np.array([x, sy, z1]), fork_quat(), 1.5)
        B.hold(self.hk(0.4))
        self.when_settled(self.carry, tol=0.0015, axis=2)

    def carry(self):
        B = self.B
        sx, sy = self.shield_xy
        x = sx + FORK_BACK
        g = self.grasp_point("right")[0]
        self.log(f"LIFTED: hand at {np.round(g, 4)} (fingers' edges {(g[2] + FORK_TOP - S.SPINE0) * 1000:+.1f} mm "
                 f"above the shield's resting underside)")
        z_high = S.SPINE0 + 0.040
        """The fingers' lowest edges (near their roots, about 3.5 cm below the grasp point with the hand pitched up)
        above the post's top."""
        self.lin(B, np.array([x, sy, z_high]), 0.015)
        y0 = S.HANGER_A_Y - S.SH_L / 2 - S.STRAP_W / 2 - 0.012
        self.lin(B, np.array([x, y0, z_high]), 0.012)
        B.hold(self.hk(0.4))
        z_low = S.Z_CB - S.SH_RO - 0.004 - FORK_TOP
        """Shield's rims 4 mm below the strap's top: below the raised pipe's underside here."""
        self.lin(B, np.array([x, y0, z_low]), 0.01)
        self.lin(B, np.array([S.PIPE_X + SEAT_BACK, y0, z_low]), 0.02)
        self.sync()
        self.now(self.under_pipe)

    def under_pipe(self):
        lift = self.pipe_lift_at_a()
        self.log(f"PIPE: arm A's joints put the pipe {lift * 1000:.1f} mm up at hanger A")
        B = self.B
        y0 = S.HANGER_A_Y - S.SH_L / 2 - S.STRAP_W / 2 - 0.012
        z_slide = S.Z_CB + 0.004 - FORK_TOP
        self.jmove(B, "right", np.array([S.PIPE_X + SEAT_BACK, y0, z_slide]), fork_quat(pitch=INSERT_PITCH), 2.0)
        B.hold(self.hk(1.5))
        self.when_settled(self.slide_in, tol=0.0015)

    def slide_in(self):
        B = self.B
        g = self.grasp_point("right")[0]
        short = S.Z_CB + 0.003 - (g[2] + FORK_TOP)
        if short > 0.0005 and not getattr(self, "_raised", False):
            # The arm sags under the load: raise the target by what its joints say is missing, then check again.
            self._raised = True
            self.log(f"SLIDE HEIGHT: fingers' edges {short * 1000:.1f} mm short of 3 mm above the strap; raising")
            p = B.pos + ROOT_OFFSET
            self.lin(B, p + np.array([0.0, 0.0, short + 0.001]), 0.005)
            B.hold(self.hk(1.0))
            self.sync()
            self.now(self.slide_in)
            return
        z_slide = B.pos[2] + ROOT_OFFSET[2]
        # In until the shield's leading end is 2 cm past the strap (its front part over the strap; the fingers stay
        # well short of it), then set it down there.
        y1 = S.HANGER_A_Y + S.STRAP_W / 2 + 0.020 - S.SH_L / 2
        self.lin(B, np.array([S.PIPE_X + SEAT_BACK, y1, z_slide]), 0.012)
        B.hold(self.hk(0.4))
        self.lin(B, np.array([S.PIPE_X + SEAT_BACK, y1, S.Z_CB - FORK_TOP - 0.002]), 0.005)
        B.hold(self.hk(0.5))
        self.when_settled(self.after_slide, tol=0.0015)

    def after_slide(self):
        g = self.grasp_point("right")[0]
        self.slid_y = float(g[1])
        self.log(f"SLID IN: hand at {np.round(g, 4)} (the shield's middle about {g[1] * 1000:.0f} mm from the hanger, "
                 f"its front {(g[1] + S.SH_L / 2) * 1000:.0f} mm past the hanger's centre line)")
        if self.no_centre:
            self.log("CONTROL: no centring")
            self.sync()
            self.now(self.lower_pipe)
            return
        self.clamp()

    def pipe_hand_z(self, rise_at_a):
        return self.grip_z0 + rise_at_a * (GRIP_Y - S.HANGER_B_Y) / (S.HANGER_A_Y - S.HANGER_B_Y)

    def clamp(self):
        """Lower the pipe onto the shield's front part (on the strap): the pipe's weight holds the shield while the
        hand moves round behind it."""
        A = self.A
        self.jmove(A, "left", np.array([S.PIPE_X, GRIP_Y, self.pipe_hand_z(S.SH_T - 0.002)]), pipe_hand_quat(), 2.0)
        A.hold(self.hk(1.0))
        self.sync()
        self.now(self.regrip)

    def regrip(self):
        """Fingers down off the shield, back along it, then up again rolled a little: the leading finger under the
        shield, the trailing one (higher) behind its end edge."""
        B = self.B
        p = B.pos + ROOT_OFFSET
        self.lin(B, p + np.array([0.0, 0.0, -0.014]), 0.01)
        y_end = self.slid_y - S.SH_L / 2
        self.push_y0 = y_end + PAD_HALF - 0.003
        """Grasp point when the trailing finger's inner face is 3 mm behind the end edge."""
        self.lin(B, np.array([p[0], self.push_y0, p[2] - 0.014]), 0.02)
        z_push = S.Z_CB - FORK_TOP + FINGER_HALF * math.sin(PUSH_ROLL)
        self.jmove(B, "right", np.array([p[0], self.push_y0, z_push - 0.010]), fork_quat(PUSH_ROLL, INSERT_PITCH), 1.5)
        self.lin(B, np.array([p[0], self.push_y0, z_push]), 0.005)
        B.hold(self.hk(0.5))
        # Raise the pipe off the shield again.
        A = self.A
        self.sync()
        self.jmove(A, "left", np.array([S.PIPE_X, GRIP_Y, self.grip_z0 + LIFT_HAND]), pipe_hand_quat(), 2.0)
        A.hold(self.hk(0.8))
        self.sync()
        self.now(self.push)

    def push(self):
        """Push the shield along by its end edge until its middle is on the hanger's centre line."""
        B = self.B
        p = B.pos + ROOT_OFFSET
        y_goal = S.HANGER_A_Y - S.SH_L / 2 + PAD_HALF
        self.lin(B, np.array([p[0], y_goal, p[2]]), 0.008)
        B.hold(self.hk(0.8))
        self.when_settled(self.centre_check, tol=0.0012, axis=1)

    def centre_check(self):
        r = self.shield_mark_in_hanger()
        g = self.grasp_point("right")[0]
        self.log(f"PUSHED: hand at {np.round(g, 4)} (by its joints, the shield's middle {(g[1] - PAD_HALF + S.SH_L / 2) * 1000:+.1f}"
                 f" mm from the hanger)")
        B = self.B
        if r is None:
            self.log("TOP CAM: red tape not seen under the pipe; going by the hand's position")
            self.sync()
            self.now(self.lower_pipe)
            return
        err = r[0] - S.HANGER_A_Y
        self.log(f"TOP CAM: shield's centre tape {err * 1000:+.1f} mm from the hanger ({r[1]} px)")
        if err < -0.004 and self.corrections < 2:
            self.corrections += 1
            p = B.pos + ROOT_OFFSET
            self.lin(B, p + np.array([0.0, -err, 0.0]), 0.005)
            B.hold(self.hk(0.8))
            self.when_settled(self.centre_check, tol=0.0012, axis=1)
            return
        self.sync()
        self.now(self.lower_pipe)

    def lower_pipe(self):
        A = self.A
        g = np.array([S.PIPE_X, GRIP_Y, self.grip_z0])
        rest = g + np.array([0.0, 0.0, S.SH_T * (GRIP_Y - S.HANGER_B_Y) / (S.HANGER_A_Y - S.HANGER_B_Y)])
        self.jmove(A, "left", rest, pipe_hand_quat(), 2.5)
        A.hold(self.hk(0.5))
        self.sync()
        self.now(lambda: self.log(f"PIPE LOWERED: arm A's joints put the pipe {self.pipe_lift_at_a() * 1000:.1f} mm above "
                                  f"its bare-strap height at hanger A (on the shield: 1.2 mm)"))
        self.grip(A, 1.0, 0.8)
        fdir = np.array([math.sin(TILT), 0.0, -math.cos(TILT)])
        self.jmove(A, "left", rest - 0.09 * fdir, pipe_hand_quat(), 1.5)
        self.jq(A, "left", TUCK_Q, 2.5)
        self.sync()
        self.now(self.withdraw)

    def withdraw(self):
        B = self.B
        p = B.pos + ROOT_OFFSET
        self.lin(B, p + np.array([0.0, 0.0, -0.018]), 0.01)
        self.lin(B, np.array([S.PIPE_X - 0.12, p[1], p[2] - 0.018]), 0.04)
        self.jq(B, "right", TUCK_Q, 2.5)
        B.hold(self.hk(10.0))
        self.A.hold(self.hk(10.0))
        self.sync()

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
            if prev is not None:
                still = np.abs(q_des - prev) < 0.0005
                err = q_des - self.q(side)[0]
                self.trim[side] = np.where(still, np.clip(self.trim[side] + TRIM_GAIN * err, -TRIM_MAX, TRIM_MAX),
                                           self.trim[side])
            self.q_prev[side] = q_des.copy()
            acts[i] = np.r_[q_des + self.trim[side], acts[i][6]]
        act = np.concatenate(acts)
        if self.debug_every and self.k % self.debug_every == 0:
            ga = self.grasp_point("left")[0]
            gb = self.grasp_point("right")[0]
            self.log(f"DBG k={self.k} A at {np.round(ga, 4)} B at {np.round(gb, 4)} grip {self.q('left')[1]:.2f}/"
                     f"{self.q('right')[1]:.2f}")
        self.k += 1
        return act
