"""The scripted demo, closed loop on robot-observable inputs only.

What it reads, every control step: the observation Inspect Robots hands any policy, i.e. the three RGB images
(``top_cam``, ``left_cam``, ``right_cam``) and the 14-D ``joint_pos`` state. Nothing else comes from the simulator:
no object poses, no prop joint angles, no scorer state. Fixed knowledge it uses: the documented setup (task.json
``setup``: where the bracket, its chain hole, its J-hook and the cylinder are, part sizes, the red sleeve on the
chain's end link), the station's published camera calibration and the arm's URDF (``yam_arena.station``,
``yam_arena.urdf``), as a real robot's script would.

Plan (left arm = arm A takes the chain round, right arm = arm B guards the cylinder):
1. Top camera: find the red sleeve on the chain's end (it hangs below the bracket's centre tongue).
2. Arm A holds its wrist camera in front of it (fingers forward): the sleeve's pixels, back-projected onto the
   vertical plane the chain hangs in (through the documented chain hole), give its height and side position
   (three frames averaged).
3. Arm A, fingers forward and part open, closes on the sleeve (a pinch across the end link).
4. Arm A carries the end round the cylinder's front at chain height (31 cm) to beyond the J-hook on the cylinder's
   left, pulling the chain round the front and over the hook's tip; meanwhile arm B comes in from the right and holds
   its closed fingers about 1 cm off the cylinder's right side, 10 cm up (a guard; it does not push).
5. Arm A lowers the end so the chain drops onto the hook's tip (the link there threads on it), eases toward the hook
   so the link settles in the bend, lets go and backs away in front of the chain; arm B backs away; both wait.

``-P slack=<n>`` (control) brings the end n links' worth (28 mm each) less far, so a link further along the chain
goes on the hook and the chain is left slack. ``CYL_DEMO_DEBUG=<n>`` prints estimates every n steps.

A small joint-space integral trim on both arms (proprioception only) removes the station arms' steady error of a
few hundredths of a radian under their soft gains.
"""

from __future__ import annotations

import math
import os

import numpy as np

from yam_arena import ik, station
from yam_arena.ik import matrix_to_quat_xyzw

from cylchain_sim import spec as S
from cylchain_sim import vision as V
from cylchain_sim.kit import CLOSED, DOWN, FORWARD, OPEN, ROOT_OFFSET, ArmTimeline, conj, q_axis, qmul

Q_REL = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
TCP_EXTRA = 0.1347 - 0.134
RED = ((235, 12), (120, 255), (60, 255))
GRIP_OPEN = 0.45
SLEEVE_Z = 0.008
SLEEVE_TOP_Z = 0.012
GRASP_Z = 0.014
TRIM_GAIN, TRIM_MAX = 0.05, 0.06
SIDE = None  # set below
CHAIN_Z = 0.312
"""Height at which the end is carried round (just under the plate: the chain then lies in the upper third)."""
PULL = (0.37, 0.145, 0.33)
PULL_DIR = (0.3, -0.95, 0.0)
"""The hand there, fingers pointing back toward the hook: the taut chain then runs from the cylinder's front over the
gap between the hook's tip and stem (chosen offline from the cylinder's documented position)."""
"""Where the end (two links beyond the one that goes on the hook) is held: the taut chain then runs from the cylinder
over the hook's tip, the link to hook just above it."""
DROP_X, DROP_Y, DROP_Z = S.HOOK_X - S.HOOK_W / 2 + 0.012, S.HOOK_Y + 0.05, 0.20
"""The end is taken down past the hook 5 cm outside it, to 20 cm above the table."""
GUARD_GAP, GUARD_Z = 0.024, 0.10
PRESS = (S.HOOK_X - 0.004, S.HOOK_Y + 0.022, 0.297)
"""Arm B's grasp point at the end of its press (fingers forward, closed, their lower edge about 3.4 cm lower):
22 mm outside the hook, on the side of arm A's hand."""
TUCK_Q = (0.0, 0.05, 0.05, 0.0, 0.0, 0.0)


def q_mat(q):
    from yam_arena.urdf import quat_xyzw_to_matrix

    return quat_xyzw_to_matrix(np.asarray(q, dtype=float))


def mat_q(m):
    return matrix_to_quat_xyzw(np.asarray(m, dtype=float))


SIDE = mat_q(np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]))


def q_fingers(direction, side=(0.0, 0.0, 1.0)):
    """Hand orientation (our flange convention) with the fingers along ``direction``."""
    z = np.asarray(direction, dtype=float)
    z /= np.linalg.norm(z)
    sd = np.asarray(side, dtype=float)
    sd = sd - z * np.dot(sd, z)
    sd /= np.linalg.norm(sd)
    r = np.stack([sd, np.cross(z, sd), z], 1)  # yam_arena gripper frame: +z along the fingers
    return qmul(mat_q(r), conj(Q_REL))


LOOK = ((0.22, 0.0, 0.17), (0.4, -0.3, -0.87))
"""Wrist-camera look at the chain's end: hand position and finger direction (chosen offline for reach)."""
"""Arm B's hand: fingers pointing +y (at the cylinder's right side)."""


class Demo:
    def __init__(self, control_hz: float = 20.0, speed: float = 0.10, slack: int = 0, log=None):
        self.log = log or (lambda *a: print(*a, flush=True))
        self.dt = 1.0 / control_hz
        self.speed = speed
        self.slack = int(slack)
        self.debug_every = int(os.environ.get("CYL_DEMO_DEBUG", "0"))
        self.k = 0
        self.obs = None
        self.events: dict[int, list] = {}
        self.A = self.B = None
        self.q_cmd = {}
        self.worst = 0.0
        self.end = np.array([0.24, S.ANCHOR[1], SLEEVE_Z])
        self.heading = 180.0
        self.trim = {"left": np.zeros(6), "right": np.zeros(6)}
        self.q_prev = {}

    # -- helpers ---------------------------------------------------------------------------------------------------
    def hk(self, s):
        return max(1, round(s / self.dt))

    def q(self, side):
        return V.split_state(self.obs.state["joint_pos"])[side]

    def img(self, name):
        return np.asarray(self.obs.images[name])

    def grasp_point(self, side):
        p, r = V.tcp_desk(self.q(side)[0], side)
        return p + r @ np.array([0.0, 0.0, TCP_EXTRA]), qmul(matrix_to_quat_xyzw(r), conj(Q_REL))

    def at(self, step, fn):
        self.events.setdefault(step, []).append(fn)

    def now(self, fn):
        self.at(max(len(self.A.steps), len(self.B.steps)), fn)

    def sync(self):
        n = max(len(self.A.steps), len(self.B.steps))
        self.A.pad_to(n)
        self.B.pad_to(n)

    def when_settled(self, fn, tries=40, tol=0.002):
        self.sync()

        def check():
            lag = np.linalg.norm(self.grasp_point("left")[0] - (self.A.pos + ROOT_OFFSET))
            if lag < tol or check.tries <= 0:
                if lag >= tol:
                    self.log(f"SETTLE timeout: lag {lag * 1000:.1f} mm; target {np.round(self.A.pos + ROOT_OFFSET, 4)}, "
                             f"at {np.round(self.grasp_point('left')[0], 4)}; joints cmd {np.round(self.q_cmd['left'], 3)} "
                             f"meas {np.round(self.q('left')[0], 3)}")
                fn()
            else:
                check.tries -= 1
                self.A.hold(self.hk(0.1))
                self.sync()
                self.now(check)

        check.tries = tries
        self.now(check)

    def best_quat(self, side, pos, quats):
        """The orientation among ``quats`` that the arm reaches best at ``pos`` (URDF IK)."""
        best = None
        for q in quats:
            tcp = np.asarray(pos, dtype=float) + np.array([0.0, 0.0, V.DESK_Z])
            e = min((ik.solve(tcp, qmul(q, Q_REL), seed, base_pos=station.BASES[side], iters=150)
                     for seed in (self.q(side)[0], np.array([0.0, 1.0, 1.0, -0.6, 0.0, 0.0]))),
                    key=lambda r: r[1] + 0.02 * r[2])
            c = e[1] + 0.02 * e[2]
            if best is None or c < best[0]:
                best = (c, q)
        return best[1]

    def solve_side(self, side, pos_world, quat, seed):
        """URDF IK for a grasp-point pose (desk frame), best of a few seeds; returns (q, error m)."""
        tcp = np.asarray(pos_world, dtype=float) + q_mat(quat) @ np.array([0.0, 0.0, -TCP_EXTRA]) + \
            np.array([0.0, 0.0, V.DESK_Z])
        from yam_arena.ik import LOWER, UPPER

        sols = []
        seed = np.asarray(seed, dtype=float)
        for sd in (seed, np.array([0.0, 1.0, 1.0, -0.6, 0.0, 0.0]), np.array([-0.5, 1.9, 1.3, -0.9, 0.0, 0.0]),
                   np.array([-1.1, 2.0, 1.5, 0.6, -1.1, 0.0]), np.array([0.5, 1.9, 1.3, -0.9, 0.0, 0.0]),
                   np.array([-1.4, 1.5, 1.6, -1.1, -1.3, -1.0]), np.array([-1.1, 2.2, 2.3, -0.7, -1.0, -0.5])):
            q, pe, re = ik.solve(tcp, qmul(quat, Q_REL), sd, base_pos=station.BASES[side], iters=200)
            sols.append((q, pe, pe + 0.02 * re))
        good = [t for t in sols if t[2] < 0.0015]
        if good:  # exact solutions: prefer joint-limit margin, then nearness to the seed
            def key(t):
                margin = min(np.min(t[0] - LOWER), np.min(UPPER - t[0]))
                return -min(margin, 0.25) * 4 + np.linalg.norm(t[0] - seed) * 0.2
            q, pe, _ = min(good, key=key)
            return q, pe
        q, pe, _ = min(sols, key=lambda t: t[2])
        return q, pe

    def jmove(self, T, side, pos_world, quat, seconds):
        """Joint-space move (for large turns of the hand): IK at both ends, joints interpolated (smoothstep)."""
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

    def jq(self, T, side, q1, seconds):
        """Joint-space move to joint angles q1 (smoothstep)."""
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
        T.pos = p - ROOT_OFFSET
        T.quat, T.last_q = qmul(matrix_to_quat_xyzw(r), conj(Q_REL)), q1

    def solve_side(self, side, pos_world, quat, seed):
        from yam_arena.ik import LOWER, UPPER

        tcp = np.asarray(pos_world, dtype=float) + q_mat(quat) @ np.array([0.0, 0.0, -TCP_EXTRA]) + \
            np.array([0.0, 0.0, V.DESK_Z])
        sols = []
        seed = np.asarray(seed, dtype=float)
        for sd in (seed, np.array([0.0, 1.0, 1.0, -0.6, 0.0, 0.0]), np.array([-0.5, 1.9, 1.3, -0.9, 0.0, 0.0]),
                   np.array([-1.1, 2.0, 1.5, 0.6, -1.1, 0.0]), np.array([0.5, 1.9, 1.3, -0.9, 0.0, 0.0]),
                   np.array([-1.2, 2.4, 1.1, 0.4, -1.2, 0.0]), np.array([0.0, 2.2, 0.8, 1.4, 0.0, -1.57])):
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
        """Joint-space move: IK at both ends, joints interpolated (smoothstep)."""
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
        T.pos = p - ROOT_OFFSET
        T.quat, T.last_q = qmul(matrix_to_quat_xyzw(r), conj(Q_REL)), q1

    # -- perception ------------------------------------------------------------------------------------------------
    def find_sleeve(self, im, cam, min_px=4):
        """Red sleeve -> its centre (on its known height above the table) and heading, or None."""
        bl = V.blobs(V.mask_range(im, *RED), min_px=min_px)
        if not bl:
            return None
        b = bl[0]
        vs, us = np.nonzero(b["mask"])
        pts = np.array([cam.on_plane((u + 0.5, v + 0.5), SLEEVE_TOP_Z) for u, v in zip(us, vs)])
        c = pts.mean(axis=0)
        w, vecs = np.linalg.eigh(np.cov((pts[:, :2] - c[:2]).T))
        h = vecs[:, -1]
        return {"pos": np.array([c[0], c[1], SLEEVE_Z]), "heading": math.degrees(math.atan2(h[1], h[0])),
                "px": b["area"], "span_mm": float(np.ptp((pts[:, :2] - c[:2]) @ h)) * 1000}

    def top_survey(self):
        im = self.img("top_cam")
        r = self.find_sleeve(im, V.Cam.top(im.shape[1], im.shape[0]), min_px=2)
        if r is None:
            self.log("TOP CAM: sleeve not found; using the layout")
        else:
            self.end = r["pos"]
            self.log(f"TOP CAM: chain end at {np.round(r['pos'], 4)} ({r['px']} px)")

    def wrist_look(self, then, frames=3):
        got = []

        def grab():
            im = self.img("left_cam")
            cam = V.Cam.wrist("left", self.q("left")[0], im.shape[1], im.shape[0])
            r = self.find_sleeve(im, cam)
            if r is not None:
                got.append(r["pos"])
                self.heading = r["heading"]
                self.log(f"WRIST CAM: chain end at {np.round(r['pos'], 4)}, heading {r['heading']:.0f} ({r['px']} px, "
                         f"{r['span_mm']:.1f} mm long)")
            if len(got) < frames and grab.left > 0:
                grab.left -= 1
                self.A.hold(2)
                self.sync()
                self.now(grab)
                return
            if got:
                self.end = np.mean(got, axis=0)
            self.log(f"END: {np.round(self.end, 4)}")
            then()

        grab.left = frames + 3
        self.now(grab)

    # -- plan --------------------------------------------------------------------------------------------------------
    def start(self):
        self.top_survey()
        e = self.end
        A = self.A
        A.grip_to(GRIP_OPEN, 2)
        self.jmove(A, "left", np.array(LOOK[0]), q_fingers(LOOK[1], (0.0, -1.0, 0.0)), 3.0)
        A.hold(self.hk(0.5))
        self.when_settled(lambda: self.wrist_look(self.grasp), tol=0.004)

    def grasp(self):
        e = self.end
        A = self.A
        # Fingers down, closing across the sleeve (the link lies along its heading).
        q = qmul(q_axis((0, 0, 1), math.radians(self.heading)), DOWN)
        self.jmove(A, "left", np.array([e[0], e[1], 0.06]), q, 1.5)
        A.move(np.array([e[0], e[1], GRASP_Z]), speed=0.02)
        A.hold(self.hk(0.4))
        A.grip_to(CLOSED, self.hk(1.0))
        self.when_settled(self.carry, tol=0.006)

    def carry(self):
        A, B = self.A, self.B
        e = self.end
        A.move(np.array([e[0], e[1], 0.08]), speed=0.04)
        pull = np.array(PULL) - np.array([0.0, S.LINK_PITCH * self.slack, 0.0])
        cx, cy = S.CYL_XY
        prev = A.pos + ROOT_OFFSET
        qp = q_fingers(PULL_DIR, (0.0, 0.0, 1.0))
        for p, q in (((cx - 0.13, -0.06, 0.20), FORWARD), ((cx - 0.14, 0.0, CHAIN_Z), FORWARD),
                     ((cx - 0.12, 0.08, CHAIN_Z), FORWARD), ((pull[0] - 0.05, pull[1] + 0.005, pull[2]), qp),
                     (tuple(pull), qp)):
            p = np.array(p)
            self.jmove(A, "left", p, q, max(1.5, np.linalg.norm(p - prev) / 0.05))
            prev = p
        # Arm B comes over above the chain just outside the hook while arm A holds it taut.
        self.sync()
        B.grip_to(CLOSED, 4)
        self.jmove(B, "right", np.array(PRESS[:2] + (PRESS[2] + 0.07,)), FORWARD, 3.0)
        A.hold(self.hk(0.5))
        self.sync()
        self.when_settled(self.drop, tol=0.004)

    def drop(self):
        A, B = self.A, self.B
        self.log(f"PULL at {np.round(self.grasp_point('left')[0], 4)} (target {np.round(A.pos + ROOT_OFFSET, 4)})")
        # Arm B presses the taut chain down just outside the hook (the lower edge of its closed fingers, 3.4 cm below
        # its grasp point): the chain is drawn down over the hook and drops into the gap between its stem and tip.
        B.move(np.array(PRESS), speed=0.01)
        B.hold(self.hk(1.0))
        self.sync()
        # Arm A eases in toward the hook and down, so the hooked link settles in the bend; lets go and backs off.
        A.move(A.pos + ROOT_OFFSET + np.array([-0.01, -0.012, -0.03]), speed=0.01)
        A.hold(self.hk(0.6))
        A.grip_to(OPEN, self.hk(1.0))
        A.move(A.pos + ROOT_OFFSET + np.array([-0.02, 0.06, 0.0]), speed=0.04)
        A.move(np.array([0.15, 0.22, 0.30]), speed=0.08)
        self.sync()
        B.hold(self.hk(0.5))
        B.move(B.pos + ROOT_OFFSET + np.array([0.0, 0.0, 0.07]), speed=0.03)
        B.move(B.pos + ROOT_OFFSET + np.array([-0.10, 0.0, 0.0]), speed=0.05)
        self.jq(B, "right", TUCK_Q, 2.5)
        self.sync()
        A.hold(self.hk(8.0))
        B.hold(self.hk(8.0))

    # -- action ------------------------------------------------------------------------------------------------------
    def _ik(self, side, pos_root, quat, grip):
        pos = pos_root + ROOT_OFFSET + np.array([0.0, 0.0, V.DESK_Z])
        tcp = pos + q_mat(quat) @ np.array([0.0, 0.0, -TCP_EXTRA])
        rot = qmul(quat, Q_REL)
        q, pe, _ = ik.solve(tcp, rot, self.q_cmd[side], base_pos=station.BASES[side], iters=60)
        if pe > 0.001:
            for seed in (self.q(side)[0], np.array([0.0, 1.0, 1.0, -0.6, 0.0, 0.0]),
                         np.array([0.4, 1.6, 1.8, -1.6, 0.0, 0.4]), np.array([-0.4, 1.6, 1.8, -1.6, 0.0, -0.4])):
                q2, pe2, _ = ik.solve(tcp, rot, seed, base_pos=station.BASES[side], iters=200)
                if pe2 < pe and np.max(np.abs(q2 - self.q_cmd[side])) < 0.6:
                    q, pe = q2, pe2
        if pe > 0.02:  # no good solution near the last command: hold there rather than jump
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
            if st[0] is None:  # joint-space step
                self.q_cmd[side] = np.asarray(st[1], dtype=float)
                acts.append(np.r_[self.q_cmd[side], st[2]])
            else:
                acts.append(self._ik(side, *st))
        a, b = self.A.steps[self.k], self.B.steps[self.k]
        # Joint-space integral trim: the station arms' soft joint gains leave a steady error of a few hundredths of
        # a radian (several mm at the fingers); while a joint's target is (nearly) still, integrate the measured
        # error into a small offset, as a real robot's controller would.
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
            ta = None if a[0] is None else np.round(a[0] + ROOT_OFFSET, 3)
            self.log(f"DBG k={self.k} A target {ta} at {np.round(ga, 3)} trimA {np.round(self.trim['left'], 3)} "
                     f"qcmdA {np.round(self.q_cmd['left'], 2)}")
        self.k += 1
        return act
