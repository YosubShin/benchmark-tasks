"""The scripted demo, closed loop on robot-observable inputs only.

What it reads, every control step: the observation Inspect Robots hands any policy, i.e. the three RGB images
(``top_cam``, ``left_cam``, ``right_cam``) and the 14-D ``joint_pos`` state (each arm's six joint angles and its
gripper opening). Nothing else comes from the simulator: no object poses, no hasp, handle or shackle joint angles, no
latch or scorer state. Fixed knowledge it uses: the documented setup (``lockout_sim.layout``: the numbers in task.json
``setup``, part sizes, colours), the station's published camera calibration and the arm's URDF (``yam_arena.station``,
``yam_arena.urdf``, ``yam_arena.ik``), as a real robot's script would.

How it perceives each thing:
- The padlock, the hasp and the coworker's lock, roughly: in the first top-camera frame, the purple padlock, the red
  hasp tail and the yellow lock are segmented by colour near their documented places. At about 5 mm per pixel this
  only confirms each part is there; the wrist camera locates them.
- The padlock on its pedestal: arm A's wrist camera from a look pose above it; purple pixels matched against the
  outline of the body's upward face (rendered through the calibrated camera placed by forward kinematics of arm A's
  joints): a coarse search, then each long edge on its own, the body put midway between them.
- The padlock in arm A's hand: the fingers close on it where the camera saw it; the grasp does not move it, so its
  pose relative to the hand is the camera estimate taken just before. From then on it follows arm A's joints.
- The hasp's free hole: with the padlock held beside the tail, arm A's wrist camera sees the red tail and its six
  square holes; the tail's outline and holes are matched to the red pixels (shift in the tail's plane and the
  hasp's swing about its ring), averaged over 3 frames. The robot-side bottom hole is free (documented; the yellow
  lock is in the wall-side one).
- Threading the shackle: the short leg's tip (forward kinematics plus the grasp) is put 6 mm in front of the hole and
  pushed through along the hole's axis. That it went through is read from arm A's joints: the tip advanced to the
  commanded depth, or it stopped against the tail (a miss: back off and try the next point of a small search
  pattern).
- Snapping the shackle shut: arm B pushes the shackle's bend while arm A holds the body; arm B's fingertip (its
  joints) stops advancing under the full push once the shackle has bottomed and latched.
- Release and park: commanded poses, measured joint angles and timing.

Grasp-point targets become yam_arena's 14-D joint action through yam_arena's own IK (``yam_arena.ik``).
``--policy lockout-demo`` (lockout_sim.policy) plays it. LOCKOUT_DEMO_DEBUG=1 prints extra perception details.
"""

from __future__ import annotations

import math
import os

import numpy as np

from yam_arena import ik, station
from yam_arena.ik import matrix_to_quat_xyzw
from yam_arena.urdf import quat_xyzw_to_matrix

from lockout_sim import layout as L
from lockout_sim import vision as V
from lockout_sim.kit import CLOSED, FORWARD, OPEN, ROOT_OFFSET, ArmTimeline, GraspModel, conj, q_axis, qmul, rotate

DEBUG = os.environ.get("LOCKOUT_DEMO_DEBUG") == "1"

Q_REL = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
"""yam_arena's ``gripper`` frame = our flange frame * Rz(90) Rx(180)."""
TCP_EXTRA = 0.1347 - 0.134
ARM = {"left": slice(0, 6), "right": slice(7, 13)}

EX, EY, EZ = L.EX, L.EY, L.EZ
LOCK_QUAT = np.array(L.LOCK_LYING_QUAT)
SIDE_A = qmul(q_axis((0, 0, 1), -math.pi / 2), FORWARD)
QA = qmul(L.YAW_Q, qmul(q_axis((1, 0, 0), math.radians(40)), SIDE_A))
A_FINGER_DIR = L.turn((0.0, -math.cos(math.radians(40)), -math.sin(math.radians(40))))
YAW_B = math.radians(60)
QB_WAIT = qmul(q_axis((-math.sin(YAW_B), math.cos(YAW_B), 0), math.radians(20)), qmul(q_axis((0, 0, 1), YAW_B), FORWARD))
QB = qmul(L.YAW_Q, QB_WAIT)
B_FINGER_DIR = L.turn((math.cos(YAW_B) * math.cos(math.radians(20)), math.sin(YAW_B) * math.cos(math.radians(20)),
                       -math.sin(math.radians(20))))
FINGER_END = 0.012
GRASP_IN_LOCK = (0.0, 0.0, -0.012)
R_WAIT = np.array([0.18, -0.26, 0.24])
HOLE = (0, 0)
"""The robot-side hole of the bottom row (free; the coworker's lock is in the wall-side one)."""
FACE_GAP = 0.006
PIERCE = 0.010
LOOK_Q = qmul(q_axis((0, 1, 0), math.radians(65.0)), FORWARD)
"""Wrist-camera look at the padlock: hand pointing forward and 65 deg down."""
HASP_LOOK_Q = qmul(q_axis((0, 0, 1), math.radians(-60.0)), qmul(q_axis((0, 1, 0), math.radians(10.0)), FORWARD))
"""Wrist-camera look at the hasp's tail from its left (+EY) side, nearly face on."""
PURPLE = ((190, 235), (40, 255), (60, 255))
PURPLE_TOP = ((190, 235), (40, 255), (60, 255))
RED = ((236, 12), (100, 255), (40, 255))
YELLOW = ((25, 50), (120, 255), (120, 255))


def yaw_q(deg):
    return q_axis((0, 0, 1), math.radians(deg))


class Demo:
    """``next_action(observation)`` gives the 14-D joint action for the next control step."""

    def __init__(self, control_hz: float = 20.0, speed: float = 0.10, log=print):
        self.dt = 1.0 / control_hz
        self.speed = speed
        self.log = log
        self.k = 0
        self.obs = None
        self.planner = None
        self.q_arm = {}
        self.worst = {"left": 0.0, "right": 0.0}
        self.cam_in_hand = {}

    def hk(self, s):
        return max(1, round(s / self.dt))

    def joints(self, side) -> np.ndarray:
        return np.asarray(self.obs.state["joint_pos"], dtype=float).reshape(-1)[ARM[side]]

    def image(self, name) -> np.ndarray:
        return np.asarray(self.obs.images[name])

    def hand(self, side):
        """(desk position, quaternion in our flange convention) of one arm's grasp point, from its joint readings."""
        tcp, r = ik.tcp_pose(self.joints(side), station.BASES[side])
        quat = qmul(matrix_to_quat_xyzw(r), conj(Q_REL))
        pos = np.asarray(tcp) + rotate(quat, (0.0, 0.0, TCP_EXTRA)) - (0.0, 0.0, V.DESK_Z)
        return pos, quat

    def top(self):
        im = self.image("top_cam")
        return V.Cam.top(im.shape[1], im.shape[0]), im

    def wrist(self, side):
        im = self.image(f"{side}_cam")
        return V.Cam.wrist(side, self.joints(side), im.shape[1], im.shape[0]), im

    def look_pose(self, side, target, quat, dist):
        """Grasp-point position that puts ``target`` on the wrist camera's optical axis, ``dist`` in front of it."""
        o, rc = self.cam_in_hand[side]
        rh = quat_xyzw_to_matrix(np.asarray(quat, dtype=float))
        fwd = rh @ rc @ np.array([0.0, 0.0, -1.0])
        return np.asarray(target) - dist * fwd - rh @ o

    def joint_action(self, side, pos_root, quat, grip):
        pos = pos_root + ROOT_OFFSET + (0.0, 0.0, V.DESK_Z)
        tcp = pos + rotate(quat, (0.0, 0.0, -TCP_EXTRA))
        q, pe, _ = ik.solve(tcp, qmul(quat, Q_REL), self.q_arm[side], base_pos=station.BASES[side], iters=60)
        self.q_arm[side] = q
        if pe > 0.005 and pe > self.worst[side]:
            self.log(f"IK {side} k={self.k} error {pe * 1000:.1f} mm at {np.round(pos - (0, 0, V.DESK_Z), 4)}",
                     flush=True)
        self.worst[side] = max(self.worst[side], pe)
        return np.r_[q, grip]

    def next_action(self, observation) -> np.ndarray:
        self.obs = observation
        if self.planner is None:
            for side in ("left", "right"):
                self.q_arm[side] = self.joints(side).copy()
                p, q = self.hand(side)
                cam, _ = self.wrist(side)
                rh = quat_xyzw_to_matrix(np.asarray(q))
                self.cam_in_hand[side] = (rh.T @ (cam.p - p), rh.T @ cam.r)
            self.planner = Planner(self)
        k = self.k
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
        self.hk = d.hk
        l_pos, l_quat = d.hand("left")
        r_pos, r_quat = d.hand("right")
        self.L = ArmTimeline(l_pos - ROOT_OFFSET, l_quat, d.dt, d.speed)
        self.R = ArmTimeline(r_pos - ROOT_OFFSET, r_quat, d.dt, d.speed)
        self.events = {}
        self.k = 0
        self.lock_start = L.LOCK_START.copy()
        self.ring = L.HASP_RING.copy()
        self.swing = 0.0
        self.R.grip = CLOSED
        self.R.move(R_WAIT, QB_WAIT, min_steps=self.hk(3.0))
        settle = self.hk(2.0)
        self.L.hold(settle)
        self.at(settle, self.survey)

    # -- scheduling -----------------------------------------------------------------------------------------------
    def at(self, step, fn):
        self.events.setdefault(max(step, self.k), []).append(fn)

    def run_due(self, k):
        self.k = k
        while k in self.events:
            for fn in self.events.pop(k):
                fn()

    def sync(self):
        n = max(len(self.L.steps), len(self.R.steps))
        self.L.pad_to(n)
        self.R.pad_to(n)
        return n

    def tl(self, side):
        return self.L if side == "left" else self.R

    def lag(self, side):
        tl = self.tl(side)
        k = min(self.k, len(tl.steps) - 1)
        return self.d.hand(side)[0] - (tl.steps[k][0] + ROOT_OFFSET)

    def when_settled(self, side, fn, tries=30, tol=0.003):
        tl = self.tl(side)

        def check():
            lag = np.linalg.norm(self.d.hand(side)[0] - (tl.pos + ROOT_OFFSET))
            if lag < tol or check.tries <= 0:
                if lag >= tol:
                    self.note(f"SETTLE timeout ({side}): lag {lag * 1000:.1f} mm")
                fn()
            else:
                check.tries -= 1
                tl.hold(self.hk(0.1))
                self.at(len(tl.steps), check)

        check.tries = tries
        self.at(len(tl.steps), check)

    def then(self, side, fn):
        self.at(len(self.tl(side).steps), fn)

    def frames(self, side, measure, n, then, gap=2):
        got = []

        def grab():
            cam, im = self.d.wrist(side)
            r = measure(cam, im)
            if r is not None:
                got.append(r)
            if len(got) < n and grab.left > 0:
                grab.left -= 1
                self.tl(side).hold(gap)
                self.then(side, grab)
                return
            then(got)

        grab.left = n + 2
        self.then(side, grab)

    # -- perception ---------------------------------------------------------------------------------------------
    def survey(self):
        """First top-camera frame: confirm the padlock, the hasp's tail and the coworker's lock are there."""
        cam, im = self.d.top()
        bx, by, bz = L.PADLOCK.body
        lock_pts = L.LOCK_START + np.array([rotate(LOCK_QUAT, (bx / 2, y, z)) for y in np.linspace(-by / 2, by / 2, 5)
                                            for z in np.linspace(-bz / 2, bz / 2, 9)])
        red, _ = L.tail_points(0.004)
        tail_pts = np.array([L.hasp_point(p) for p in red])
        cw = L.hole(*L.COWORKER) + np.array([0.0, 0.0, -0.03])
        for name, colour, pts in (("purple padlock", PURPLE_TOP, lock_pts), ("red hasp tail", RED, tail_pts),
                                  ("yellow coworker's lock", YELLOW, cw[None, :])):
            pred = cam.project(pts).mean(axis=0)
            b = V.nearest_blob(V.blobs(V.mask_range(im, *colour), min_px=2), pred, 12.0)
            if b is None:
                self.note(f"SEE top camera: no {name} near its documented place")
            else:
                self.note(f"SEE top camera: {name} found ({b['area']} px), {np.round(b['centroid'] - pred, 1)} px "
                          f"from where the layout puts it")
        self.plan_hasp_look()

    def plan_hasp_look(self):
        """Arm A, hand empty, looks at the hasp's tail nearly face on (it hangs, so it may have swung a little)."""
        L_ = self.L
        tgt = L.hasp_point((0.0, 0.0, (L.TAIL_TOP + L.TAIL_BOTTOM) / 2))
        pos = self.d.look_pose("left", tgt, HASP_LOOK_Q, 0.16)
        L_.move(pos + np.array([-0.03, 0.0, 0.03]), HASP_LOOK_Q, min_steps=self.hk(2.5))
        L_.move(pos, speed=0.05)
        L_.hold(self.hk(0.5))
        self.when_settled("left", lambda: self.frames("left", self.measure_hasp, 3, self.hasp_seen))

    # ================================================================================================================
    # Arm A: the padlock off its pedestal
    # ================================================================================================================
    def plan_look(self):
        L_ = self.L
        L_.grip_to(OPEN, self.hk(0.2))
        top = self.lock_start + np.array([0.0, 0.0, L.PADLOCK.body[0] / 2])
        pos = self.d.look_pose("left", top, LOOK_Q, 0.12)
        L_.move(pos + np.array([0.0, 0.0, 0.03]), LOOK_Q, min_steps=self.hk(2.5))
        L_.move(pos, speed=0.05)
        L_.hold(self.hk(0.5))
        self.when_settled("left", lambda: self.frames("left", self.measure_lock, 3, self.lock_seen))

    def measure_lock(self, cam, im):
        """Purple pixels matched against the body's upward face (a 19 x 44 mm rectangle); each long edge on its own,
        the body put midway between them (the face nearer the camera shows purple beside its edge)."""
        m = V.mask_range(im, *PURPLE)
        bx, by, bz = L.PADLOCK.body
        top = bx / 2
        ins = [(top, y, z) for y in np.linspace(-by / 2 + 0.0006, by / 2 - 0.0006, 9)
               for z in np.linspace(-bz / 2 + 0.0006, bz / 2 - 0.0006, 18)]
        ends = [(top, y, -bz / 2 - d) for y in np.linspace(-by / 2, by / 2, 8) for d in (0.0008, 0.0018)]
        to_desk = lambda pts: self.lock_start + np.array([rotate(LOCK_QUAT, p) for p in pts])  # noqa: E731
        offs, scores = [], []
        for sy in (-1, 1):
            out = ends + [(top, sy * (by / 2 + d), z) for z in np.linspace(-bz / 2 - 0.002, bz / 2 + 0.002, 20)
                          for d in (0.0008, 0.0018)]
            off0, _, _ = V.fit_shift(m, cam, to_desk(ins), to_desk(out), ((1, 0, 0), (0, 1, 0)), 0.015, 0.0015,
                                     refine=False)
            off, sc, _ = V.fit_shift(m, cam, to_desk(ins) + off0, to_desk(out) + off0, ((1, 0, 0), (0, 1, 0)), 0.002,
                                     0.0005)
            offs.append(off0 + off)
            scores.append(sc)
        off, s = np.mean(offs, axis=0), float(np.mean(scores))
        if DEBUG:
            self.note(f"  padlock fit: edges {np.round(offs[0] * 1000, 2)} / {np.round(offs[1] * 1000, 2)} mm, "
                      f"score {s:.2f}")
        return None if s < 0.5 else (self.lock_start + off, s)

    def lock_seen(self, got):
        if got:
            self.lock_start = np.mean([g[0] for g in got], axis=0)
            self.note(f"SEE left wrist camera: padlock body centre at {np.round(self.lock_start, 4)} (layout "
                      f"{np.round(L.LOCK_START, 4)})")
        else:
            self.note("SEE left wrist camera: padlock not matched; layout kept")
        self.plan_pick()

    def plan_pick(self):
        grasp, _ = GraspModel.nominal(QA, LOCK_QUAT, GRASP_IN_LOCK).grasp_for(self.lock_start, LOCK_QUAT)
        self.grasp = grasp
        L_ = self.L
        L_.move(grasp - 0.07 * A_FINGER_DIR, QA, min_steps=self.hk(2.5))
        L_.hold(self.hk(0.4))
        L_.move(grasp - 0.015 * A_FINGER_DIR, speed=0.04)
        L_.hold(self.hk(0.4))
        self.when_settled("left", self.final_approach)

    def final_approach(self):
        err = self.lag("left")
        self.L.move(self.grasp - err, speed=0.02)
        self.L.hold(self.hk(0.4))
        self.L.grip_to(CLOSED, self.hk(1.0))
        self.then("left", self.grasped)

    def grasped(self):
        gp, gq = self.d.hand("left")
        self.model = GraspModel(rotate(conj(gq), self.lock_start - gp), qmul(conj(gq), LOCK_QUAT))
        self.note(f"LOCK grasped: body centre in the hand (mm) {np.round(self.model.rel_pos * 1000, 1)}")
        self.L.move(gp + np.array([0.0, 0.0, 0.035]), speed=0.03)
        self.when_settled("left", self.to_hasp)

    # -- the held lock -------------------------------------------------------------------------------------------
    def lock_pose(self):
        gp, gq = self.d.hand("left")
        return gp + rotate(gq, self.model.rel_pos), qmul(gq, self.model.rel_quat)

    def tip(self):
        p, q = self.lock_pose()
        return p + rotate(q, L.PADLOCK.short_tip(L.OPEN_TRAVEL))

    def lock_center_for_tip(self, tip_target):
        return np.asarray(tip_target) - rotate(LOCK_QUAT, L.PADLOCK.short_tip(L.OPEN_TRAVEL))

    def lock_to(self, center, speed, min_steps=3):
        self.L.move(self.model.grasp_for(np.asarray(center), LOCK_QUAT)[0], speed=speed, min_steps=min_steps)

    def hole(self, offset=(0.0, 0.0)):
        """The chosen hole's centre on the tail's mid-plane, from the hasp estimate (plus a search offset)."""
        c, r = L.HOLE_COLS[HOLE[0]], L.HOLE_ROWS[HOLE[1]]
        s, co = math.sin(self.swing), math.cos(self.swing)
        local = (co * c + s * r, 0.0, -s * c + co * r)
        return L.hasp_point(local, self.ring) + offset[0] * EX + offset[1] * EZ

    # ================================================================================================================
    # Arm A: to the hasp, find the free hole, thread the shackle
    # ================================================================================================================
    def to_hasp(self):
        self.tip_entry = self.hole() - FACE_GAP * EY - 0.020 * EX
        c = self.lock_center_for_tip(self.tip_entry)
        self.lock_to(c + 0.03 * EY + 0.03 * EZ, 0.06)
        self.lock_to(c, 0.03)
        self.L.hold(self.hk(0.6))
        self.when_settled("left", self.start_search)

    def measure_hasp(self, cam, im):
        """Red pixels matched against the tail's outline (its face red; just beyond its long sides and its bottom edge
        not). Shift in the tail's plane, then the swing about the ring."""
        red = V.mask_range(im, *RED)
        face_rel, holes_rel = L.tail_outline(0.0015)

        def pts(rel, ring, swing):
            s, c = math.sin(swing), math.cos(swing)
            rot = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
            return np.array([L.hasp_point(rot @ p, ring) for p in rel])

        best = None
        off0, _, _ = V.fit_shift(red, cam, pts(face_rel, self.ring, 0.0), pts(holes_rel, self.ring, 0.0), (EX, EZ),
                                 0.012, 0.0015, refine=False)
        for sw in np.radians(np.arange(-4.0, 4.01, 1.0)):
            off, s, si = V.fit_shift(red, cam, pts(face_rel, self.ring + off0, sw),
                                     pts(holes_rel, self.ring + off0, sw), (EX, EZ), 0.003, 0.0005)
            if best is None or s > best[1]:
                best = (off0 + off, s, si, sw)
        off, s, si, sw = best
        for sw2 in sw + np.radians(np.arange(-0.75, 0.76, 0.25)):  # finer swing
            o2, s2, si2 = V.fit_shift(red, cam, pts(face_rel, self.ring + off, sw2), pts(holes_rel, self.ring + off, sw2),
                                      (EX, EZ), 0.001, 0.00025, refine=False)
            if s2 > s:
                off, s, si, sw = off + o2, s2, si2, sw2
        if DEBUG:
            self.note(f"  hasp fit: ring shift {np.round(off * 1000, 2)} mm, swing {math.degrees(sw):.1f} deg, "
                      f"score {s:.2f} ({si:.2f})")
        return None if s < 0.4 else (self.ring + off, sw, s)

    def hasp_seen(self, got):
        if got:
            self.ring = np.mean([g[0] for g in got], axis=0)
            self.swing = float(np.mean([g[1] for g in got]))
            self.note(f"SEE left wrist camera: hasp ring at {np.round(self.ring, 4)} (layout {np.round(L.HASP_RING, 4)})"
                      f", swung {math.degrees(self.swing):.1f} deg; free hole at {np.round(self.hole(), 4)}")
        else:
            self.note("SEE left wrist camera: hasp tail not matched; layout kept")
        self.L.move(self.L.pos + ROOT_OFFSET + np.array([-0.04, 0.0, 0.03]), speed=0.05)
        self.plan_look()

    def start_search(self):
        self.search = [(0.0, 0.0)] + [(r * math.cos(a), r * math.sin(a)) for r in (0.0006, 0.0012)
                                      for a in np.radians(np.arange(0, 360, 45))]
        self.search_i = 0
        self.correction = np.zeros(3)
        self.entry(0)

    def entry(self, attempt):
        off = self.search[self.search_i]
        self.tip_entry = self.hole(off) - FACE_GAP * EY - 0.020 * EX
        err = self.tip() - self.tip_entry
        if attempt == 0:
            self.note(f"ENTRY (try {self.search_i}, offset {np.round(np.array(off) * 1000, 1)} mm): tip error from "
                      f"arm A's joints (mm) {np.round(err * 1000, 1)}")
        self.correction = self.correction + err
        if attempt < 4 and np.abs(err).max() > 0.0008:
            self.lock_to(self.lock_center_for_tip(self.tip_entry - self.correction), 0.01)
            self.L.hold(self.hk(0.5))
            self.when_settled("left", lambda: self.entry(attempt + 1))
            return
        self.over(0)

    def over(self, attempt):
        off = self.search[self.search_i]
        self.tip_over = self.hole(off) - FACE_GAP * EY
        if attempt == 0:
            self.lock_to(self.lock_center_for_tip(self.tip_over - self.correction), 0.012)
            self.L.hold(self.hk(0.6))
            self.when_settled("left", lambda: self.over(attempt + 1))
            return
        err = self.tip() - self.tip_over
        ex, ez = float(err @ EX), float(err[2])
        self.correction = self.correction + ex * EX + ez * EZ
        if attempt < 5 and max(abs(ex), abs(ez)) > 0.0006:
            self.lock_to(self.lock_center_for_tip(self.tip_over - self.correction), 0.008)
            self.L.hold(self.hk(0.6))
            self.when_settled("left", lambda: self.over(attempt + 1))
            return
        self.note(f"OVER the hole (try {self.search_i}): tip error from arm A's joints (mm) "
                  f"{np.round([ex * 1000, ez * 1000], 2)}")
        self.through = self.tip_over + (FACE_GAP + PIERCE) * EY
        self.lock_to(self.lock_center_for_tip(self.through - self.correction), 0.006)
        self.L.hold(self.hk(0.8))
        self.when_settled("left", self.check_through, tries=10)

    def check_through(self):
        depth = float((self.tip() - self.tip_over) @ EY)
        want = FACE_GAP + PIERCE
        if depth > want - 0.003:
            self.note(f"THREADED: the tip went {depth * 1000:.1f} mm in along the hole's axis (commanded "
                      f"{want * 1000:.0f}; arm A's joints): through the hole")
            self.press_start()
            return
        self.note(f"BLOCKED: the tip stopped {depth * 1000:.1f} mm in (commanded {want * 1000:.0f}): backing out, "
                  f"next search point")
        self.search_i = (self.search_i + 1) % len(self.search)
        self.lock_to(self.lock_center_for_tip(self.tip_over - 0.004 * EY - self.correction), 0.01)
        self.L.hold(self.hk(0.3))
        self.when_settled("left", lambda: self.entry(0))

    # ================================================================================================================
    # Arm B presses the shackle shut while arm A holds the body
    # ================================================================================================================
    def bend_outer(self):
        pos, quat = self.lock_pose()
        return pos + rotate(quat, (0.0, 0.0, L.PADLOCK.bend_top(L.OPEN_TRAVEL)))

    def press_start(self):
        R = self.R
        self.sync()
        b = self.bend_outer()
        self.press_g = b - 0.012 * EY - FINGER_END * B_FINGER_DIR
        R.move(self.press_g - 0.06 * B_FINGER_DIR, QB, min_steps=self.hk(2.5))
        R.move(self.press_g, speed=0.04)
        R.hold(self.hk(0.6))
        self.sync()
        self.when_settled("right", self.press_line_up)

    def press_line_up(self):
        err = self.lag("right")
        self.press_g = self.press_g - (err - float(err @ EY) * EY)
        self.note(f"PRESS line-up: arm B tracking error (mm) {np.round(err * 1000, 1)}")
        self.R.move(self.press_g, speed=0.02)
        self.R.hold(self.hk(0.3))
        self.press_s = 0.0
        self.press_hist = []
        self.a_cmd0 = None
        self.sync()
        self.when_settled("right", lambda: self.press(0))

    def press(self, k):
        gp, _ = self.d.hand("right")
        adv = float((gp - self.press_g) @ EY)
        self.press_hist = (self.press_hist + [adv])[-3:]
        stopped = len(self.press_hist) == 3 and max(self.press_hist) - min(self.press_hist) < 0.0004
        bound = 0.012 + L.OPEN_TRAVEL + 0.010
        if (stopped and self.press_s >= bound - 1e-6) or k > 30:
            self.note(f"PRESS done after {k} steps: arm B's fingertip stopped {adv * 1000:.1f} mm along the push "
                      f"under the full push (the shackle bottomed)")
            self.R.move(self.R.pos + ROOT_OFFSET - 0.004 * EY, speed=0.02)
            self.R.hold(self.hk(0.4))
            self.sync()
            self.release()
            return
        if k % 3 == 0:
            self.note(f"  press {k}: arm B's fingertip {adv * 1000:.1f} mm along the push")
        # Arm A holds the body against the push: its command is moved back by a running sum of how far its hand has
        # been pushed along the push direction since the press began (integral action; bounded).
        a_meas = self.d.hand("left")[0]
        if self.a_cmd0 is None:
            self.a_cmd0, self.a_meas0, self.a_back = self.L.pos + ROOT_OFFSET, a_meas, 0.0
        self.a_back = float(np.clip(self.a_back + 0.6 * float((a_meas - self.a_meas0) @ EY), 0.0, 0.02))
        self.L.move(self.a_cmd0 - self.a_back * EY, speed=0.02, min_steps=1)
        self.press_s = min(self.press_s + 0.004, bound)
        self.R.move(self.press_g + self.press_s * EY, speed=0.02, min_steps=2)
        self.R.hold(self.hk(0.1))
        self.sync()
        self.then("right", lambda: self.press(k + 1))

    def release(self):
        R, L_ = self.R, self.L
        R.move(R.pos + ROOT_OFFSET - 0.04 * EY, speed=0.04)
        R.move(R_WAIT, speed=0.10)
        L_.pad_to(len(R.steps) - self.hk(1.2))
        L_.grip_to(0.55, self.hk(1.0))
        L_.move(L_.pos + ROOT_OFFSET - 0.05 * A_FINGER_DIR, speed=0.03)
        L_.move(L_.pos + ROOT_OFFSET + L.turn((-0.05, 0.06, 0.04)), speed=0.06)
        L_.hold(self.hk(4.0))
