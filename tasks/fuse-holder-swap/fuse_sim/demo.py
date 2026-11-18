"""The scripted demo, closed loop on robot-observable inputs only.

What it reads, every control step: the observation Inspect Robots hands any policy, i.e. the three RGB images
(``top_cam``, ``left_cam``, ``right_cam``) and the 14-D ``joint_pos`` state. Nothing else comes from the simulator:
no carrier angle, no fuse poses, no task layout record, no scorer state. Fixed knowledge it uses: the documented
layout (task.json ``setup``; ``fuse_sim.layout``: the holders, their hinge and grip tab, the fuse seat, the legends,
the spare cartons and their printed lids, the tray), the station's published camera calibration and the arm's URDF
(``yam_arena.station``, ``yam_arena.urdf``), as a real robot's script would.

How it perceives each thing:
- Which spare to fit: in the first top-camera frame, the colour of F2's legend's rating field and of the rating
  fields printed inside the two spare cartons' open lids, each classified as the nearest of the three rating colours
  (pink 2 A, green 6 A, red 10 A); the carton whose lid matches F2's legend holds the one. Its tube (grey-white in
  the brown carton) is found in the same frame to check it lies in the carton's middle slot, against the far wall,
  where the layout puts it.
- F2's carrier opening: while arm A pinches the grip tab and pulls it round its arc, the carrier's angle is the pinch
  point's angle about the documented hinge (forward kinematics of arm A's joints). After A lets go, the top camera
  checks the red grip tabs every second: once F2's is seen twice running where an open carrier's is (on its 72 degree
  stop), it has come to rest; F1's and F3's are checked where shut ones' are.
- The blown fuse: in the open carrier's documented seat.
- The spare in the hand: where arm B's fingers closed on it (forward kinematics) relative to where the camera saw it
  lying; from then on its pose follows arm B's joint readings, and the hand is servoed until the predicted fuse is
  lined up with the open carrier's seat.
- Closing: arm A's closed fingers push the tab's underside up its arc; the carrier's angle is their angle about the
  hinge (arm A's joints), and the top camera checks the tab is back where a shut carrier's is.

1. Arm A (left), fingers level, pinches the sides of F2's grip tab from the front and pulls it toward the robot
   along its arc (pulling further while the carrier does not follow) until it leaves the latch, then lets go:
   gravity swings the carrier open to its stop.
2. Arm B (right), fingers down and leaning 40 deg toward the holder, closes across the blown fuse's outer ferrule,
   lifts it toward the clip's open side in short steps, draws it out toward the robot, and drops it in the tray. If
   the gripper reading shows the fingers closed on nothing after the draw (the fuse jumped out of them as it left
   the clip), it goes back for it, at most twice.
3. Arm B reaches into the matching carton through its front cut-out, picks the spare by its near ferrule, lines it up with the open carrier's axis 9 mm
   above its seat, slides it in over the seat, lowers it into the clip and lets go.
4. Arm A, closed fingers level, lifts the carrier by the underside of its grip tab and follows it round its arc
   until it is shut, then backs away. The task's hold test then pulls the carrier open with 4 N at the tab.

``--policy fuse-demo`` (fuse_sim.policy) plays it; grasp-point targets become yam_arena's 14-D joint action through
yam_arena's own IK every control step.

Control (``FUSE_CONTROL=1``): the demo is unchanged, but the task turns the carrier latch off (fuse_sim.task.NO_LATCH),
so nothing holds the shut carrier against the hold test's pull.
"""

from __future__ import annotations

import math
import os

import numpy as np

from yam_arena import ik, station
from yam_arena.ik import matrix_to_quat_xyzw

from fuse_sim import layout as L_
from fuse_sim import vision as V
from fuse_sim.kit import CLOSED, DOWN, FORWARD, ROOT_OFFSET, ArmTimeline, GraspModel, conj, q_axis, qmul, rotate

Q_REL = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)
"""yam_arena's ``gripper`` frame = our flange frame * Q_REL."""
TCP_EXTRA = 0.1347 - 0.134
"""Our grasp point (between the finger pads) sits this much further out than yam_arena's TCP."""


class _Props:
    """The documented geometry under the names the motion code uses."""
    TAB_D, TAB_Z, OPEN_DEG = L_.TAB_D, L_.TAB_Z, L_.OPEN_DEG
    FUSE_L, HINGE_D, HINGE_Z = L_.FUSE_L, L_.HINGE_D, L_.HINGE_Z
    hinge_world = staticmethod(L_.hinge_world)
    seat_in_carrier = staticmethod(L_.seat_in_carrier)


class Demo:
    """``next_action(observation)`` gives the 14-D joint action for the next control step."""

    def __init__(self, control_hz: float = 20.0, speed: float = 0.10, log=print):
        props = _Props
        TRAY_XY = L_.TRAY_XY
        obs_box = [None]
        self._obs_box = obs_box
        state_k = [0]
        self.state_k = state_k

        def q_meas(side):
            return V.split_state(obs_box[0].state["joint_pos"])[side][0]

        def grasp_point(side):
            """Measured (desk position, flange quaternion) of an arm's grasp point: forward kinematics of its joints."""
            p, r = V.tcp_desk(q_meas(side), side)
            return p + r @ np.array([0.0, 0.0, TCP_EXTRA]), qmul(matrix_to_quat_xyzw(r), conj(Q_REL))

        def arc_angle(p, p0):
            """Angle (rad) by which p is turned from p0 about F2's hinge in the x-z plane (the carrier opens by turning
            its points from +z toward -x, so opening is positive)."""
            h = np.array(L_.hinge_world(1))
            a = math.atan2(p[2] - h[2], p[0] - h[0]) - math.atan2(p0[2] - h[2], p0[0] - h[0])
            return (a + math.pi) % (2 * math.pi) - math.pi

        self._grasp_point = grasp_point
        self._q_meas = q_meas
        k10 = control_hz / 10.0

        def n(steps):
            """A step count of the 10 Hz original, at this control rate."""
            return max(1, round(steps * k10))

        FINGER_END = 0.012
        TIP_HALF_W = 0.0075
        PRESS_DEG = 24.0
        PINCH_D = -0.0005 - FINGER_END
        """Pinch grasp point: the fingertips end 0.5 mm in front of the carrier face, so the pads cover the tab's
        front 8.5 mm and the fingers stay clear of the holder body's side walls behind it."""
        PINCH_Z = 0.5 * (props.TAB_Z[0] + props.TAB_Z[1])
        PINCH_OPEN = 0.30
        FUSE_GRIP = 0.0  # squeeze fully: the station gripper is a 5000 N/m spring capped at 50 N
        FUSE_OPEN_CMD = 0.35
        SPARE_OPEN_CMD = 0.22
        """21 mm between the fingers for the spare: they pass through the carton's 44 mm finger cut-out."""
        TOP_LEAN = math.radians(40.0)
        ALONG, BELOW = 0.002, 0.008
        Q_TOP = qmul(q_axis((0, 1, 0), -TOP_LEAN), DOWN)
        OPEN_RAD = math.radians(props.OPEN_DEG)
        HOVER = 0.009
        SPARE_VIRTUAL_Q = q_axis((0, 1, 0), -math.pi / 2)
        CLOSE_CONTACT = np.array([props.TAB_D[0], 0.0, props.TAB_Z[0] + 0.002])
        CLOSE_LEAD = 0.0015
        LIFT_SEGS = 2
        """Lift the blown fuse 6 mm twice toward the clip's open side, then draw it out toward the robot and up: the
        clip lets go during the draw, while the hand is already moving out (lifting further first, it let go with the
        arm's lag stored and the fuse jumped out of the fingers)."""
        L_PARK = np.array([0.13, 0.20, 0.20])

        def grasp_in_fuse(tip_z, below):
            cq = q_axis((0, 1, 0), -OPEN_RAD)
            f_world = np.array([math.sin(TOP_LEAN), 0.0, -math.cos(TOP_LEAN)])
            f_c = rotate(conj(cq), f_world)
            tip = np.array([0.0, 0.0, tip_z]) + f_c * below
            return tip - f_c * FINGER_END, qmul(conj(cq), Q_TOP)

        GRASP_F, REL_Q = grasp_in_fuse(props.FUSE_L / 2 - ALONG, BELOW)

        def carrier_q(theta):
            return q_axis((0, 1, 0), -theta)

        def carrier_pt(theta, d, z, slot=1):
            h = np.array(props.hinge_world(slot))
            return h + rotate(carrier_q(theta), np.array([d - props.HINGE_D, 0.0, z - props.HINGE_Z]))

        class Planner:
            def __init__(self, L, R):
                self.L, self.R, self.events = L, R, {}
                self.th = 0.0  # what the demo knows of F2's carrier angle (rad)
                self.spare_k = None  # which spare (0, 1) matches, from the cameras
                self.spare_pos = None

            def at(self, step, fn):
                self.events.setdefault(step, []).append(fn)

            def run_due(self, k):
                for fn in self.events.pop(k, []):
                    fn()

            def when_settled(self, tl, side, fn, tries=15, tol=0.003):
                def check():
                    lag = np.linalg.norm(grasp_point(side)[0] - (tl.pos + ROOT_OFFSET))
                    if lag < tol or check.tries <= 0:
                        if lag >= tol:
                            log(f"SETTLE timeout ({side}): lag {lag * 1000:.1f} mm", flush=True)
                        fn()
                    else:
                        check.tries -= 1
                        tl.hold(n(2))
                        self.at(len(tl.steps), check)
                check.tries = tries
                self.at(len(tl.steps), check)

            def sync(self):
                m = max(len(self.L.steps), len(self.R.steps))
                self.L.pad_to(m)
                self.R.pad_to(m)

            # what the demo knows -------------------------------------------------------------------------
            def theta(self):
                return self.th

            def theta_from_pinch(self):
                """Carrier angle while arm A pinches the grip tab: the pinch point's angle about the hinge (forward
                kinematics of arm A's joints), against the pinch point's angle at 0."""
                return arc_angle(grasp_point("left")[0], self.pinch_target(0.0))

            def theta_from_push(self, lead):
                """Carrier angle while arm A's closed fingers push the tab's underside up its arc."""
                return arc_angle(grasp_point("left")[0], self.close_target(0.0, lead))

            def carrier_pose(self):
                """F2's carrier frame (at its documented hinge) at the angle the demo knows."""
                return np.array(L_.hinge_world(1)), carrier_q(self.th)

            def lag(self, side, tl):
                return np.linalg.norm(grasp_point(side)[0] - (tl.pos + ROOT_OFFSET)) * 1000

            def report(self, tag):
                log(f"{tag}: carrier at {math.degrees(self.th):.1f} deg (as the demo knows it)", flush=True)

            def look_top(self, what):
                """Top camera: is F2's grip tab where a shut carrier's is, or where an open one's is? Also F1, F3."""
                im = np.asarray(obs_box[0].images["top_cam"])
                cam = V.Cam.top(im.shape[1], im.shape[0])
                out = {}
                for slot in range(3):
                    hits = {}
                    for name, th in (("shut", 0.0), ("open", OPEN_RAD)):
                        p = L_.tab_world(slot, th)
                        r = V.find_blob(im, cam, p, V.RED, p[2], max_px=6, min_px=1)
                        if r is not None:
                            hits[name] = float(np.linalg.norm(r["uv"] - r["uv_doc"]))
                    out[slot] = min(hits, key=hits.get) if hits else "not seen"
                log(f"TOP CAM ({what}): F1 tab {out[0]}, F2 tab {out[1]}, F3 tab {out[2]}", flush=True)
                return out

            def survey(self):
                """Top camera, first frame: which spare carton's lid has F2's legend colour, and where its fuse lies."""
                im = np.asarray(obs_box[0].images["top_cam"])
                cam = V.Cam.top(im.shape[1], im.shape[0])
                refs = {amps: V.hue_of_rgb(rgb) for amps, (_, rgb) in L_.RATING_COLOURS.items()}

                def rating(h):
                    return None if h is None else min(refs, key=lambda a: V.hue_dist(h, refs[a]))

                legend = rating(V.hue_at(im, cam, L_.legend_field_world(1)))
                labels = [rating(V.hue_at(im, cam, L_.carton_label_world(k))) for k in range(2)]
                names = {a: n for a, (n, _) in L_.RATING_COLOURS.items()}
                log(f"TOP CAM: F2 legend field {names.get(legend)}; carton lids {[names.get(x) for x in labels]}",
                    flush=True)
                match = [k for k in range(2) if labels[k] is not None and labels[k] == legend]
                if len(match) != 1:
                    raise RuntimeError("could not match a spare to F2's legend by colour")
                self.spare_k = match[0]
                if os.environ.get("FUSE_HAZARD_TEST") == "wrong_spare":  # scorer check only: fit the other rating
                    self.spare_k = 1 - self.spare_k
                p = np.array(L_.spare_pose(self.spare_k)[:3])
                r = V.find_blob(im, cam, p, V.BRIGHT, L_.SPARE_AXIS_Z + L_.FUSE_R, max_px=8, window_px=(4, 4, 6, 2))
                # The carton's walls keep the spare in its slot against the far wall, where the layout puts it; the
                # camera checks it is there (at 320x176 its tube is 4 px wide; good to a few millimetres only). Only
                # the pixels round the fuse's middle and far half count: in the image its near end touches the lit
                # front edge under the carton.
                if r is None or np.linalg.norm(r["xyz"][:2] - p[:2]) > 0.010:
                    raise RuntimeError("the matching spare is not in its carton's slot")
                log(f"TOP CAM: spare tube seen at {np.round(r['xyz'][:2], 4)} (layout {np.round(p[:2], 4)}): in its "
                    f"carton", flush=True)
                self.spare_pos = p
                self.look_top("start")

            # 1. Open (arm A) -----------------------------------------------------------------------------
            def plan_open(self):
                """Pinch the grip tab's sides from the front (fingers level, closing across the rail, the fingertips
                ending 0.5 mm short of the carrier face) and pull it toward the robot along its arc, as a person
                pulls the tab; past the latch, let go and gravity swings the carrier open."""
                L = self.L
                self.ycorr = 0.0
                L.grip_to(PINCH_OPEN, n(2))
                g0 = self.pinch_target(0.0)
                L.move(g0 + np.array([-0.05, 0.0, 0.02]), FORWARD, min_steps=n(30))
                L.move(g0 + np.array([-0.02, 0.0, 0.0]), speed=0.04)
                L.move(g0, speed=0.015)
                L.hold(n(3))
                self.when_settled(L, "left", self.pinch)

            @staticmethod
            def pinch_target(th):
                return carrier_pt(th, PINCH_D, PINCH_Z)

            def pinch(self):
                log(f"ON TAB: lag {self.lag('left', self.L):.1f} mm, grasp {np.round(grasp_point('left')[0], 4)}",
                    flush=True)
                self.L.grip_to(0.0, n(8))
                self.at(len(self.L.steps), lambda: self.pull_open(0.0, 0.0, 0, 0.0))

            def pull_open(self, th_cmd, pull, k, th_prev):
                """Follow the tab's arc 4 deg ahead of the carrier, pulling 3 mm further whenever it stalls."""
                th = self.th = max(0.0, self.theta_from_pinch())
                if k:
                    log(f"PULL {k}: carrier at (arm A's joints) {math.degrees(th):.1f} deg, extra pull {pull * 1000:.0f} mm, "
                        f"lag {self.lag('left', self.L):.1f} mm", flush=True)
                if math.degrees(th) >= PRESS_DEG or k > 40:
                    self.let_go()
                    return
                if k and th - th_prev < math.radians(0.5):
                    pull = min(pull + 0.003, 0.030)
                th_cmd = min(max(th_cmd, th) + math.radians(4.0), math.radians(PRESS_DEG + 4.0))
                L = self.L
                start = L.pos + ROOT_OFFSET
                goal = self.pinch_target(th_cmd) + np.array([-pull, 0.0, 0.0])
                m = n(5)
                for i in range(1, m + 1):
                    L.path([(start + (goal - start) * i / m, FORWARD)])
                L.hold(n(2))
                self.at(len(L.steps), lambda: self.pull_open(th_cmd, pull, k + 1, th))

            def let_go(self):
                log(f"PULLED: carrier at {math.degrees(self.th):.1f} deg (arm A's joints), lag {self.lag('left', self.L):.1f} mm",
                    flush=True)
                L = self.L
                L.grip_to(PINCH_OPEN, n(4))
                L.move(L.pos + ROOT_OFFSET + np.array([-0.03, 0.0, 0.03]), speed=0.05)
                L.move(L_PARK, FORWARD, speed=0.10, min_steps=n(15))
                self.sync()
                self.wait_open(0)

            def wait_open(self, tries):
                # Arm A is away: the top camera tells whether the carrier fell open onto its stop.
                seen = self.look_top(f"after letting go, check {tries}")[1] if tries % 5 == 0 else None
                if seen == "open":
                    self.open_seen = getattr(self, "open_seen", 0) + 1
                elif seen is not None:
                    self.open_seen = 0
                # Open twice in a row, 1 s apart: it has come to rest on its stop.
                if getattr(self, "open_seen", 0) >= 2 or tries > 80:
                    self.th = OPEN_RAD  # resting on its documented stop
                    self.report(f"OPEN after {tries} checks")
                    self.plan_remove()
                    return
                if seen == "shut" and tries >= 10:  # it fell back shut: pull again
                    self.th = 0.0
                    self.plan_open()
                    return
                self.R.hold(n(2))
                self.sync()
                self.at(len(self.R.steps), lambda: self.wait_open(tries + 1))

            # 2. Old fuse out (arm B) ----------------------------------------------------------------------
            @staticmethod
            def hand_for(pos, vq):
                return pos + rotate(vq, GRASP_F), qmul(vq, REL_Q)

            def plan_remove(self):
                R = self.R
                c_pos, c_quat = self.carrier_pose()
                pos = c_pos + rotate(c_quat, L_.seat_in_carrier())  # the blown fuse in its seat (documented)
                g, q = self.hand_for(pos, c_quat)
                f_dir = rotate(q, (0.0, 0.0, -1.0))
                self.outward = rotate(c_quat, (0.0, 0.0, 1.0))
                log(f"OLD FUSE at {np.round(pos, 4)}, grasp {np.round(g, 4)}", flush=True)
                R.grip_to(FUSE_OPEN_CMD, n(2))
                R.move(g - f_dir * 0.06 + self.outward * 0.01, q, min_steps=n(35))
                R.move(g - f_dir * 0.03, speed=0.05)
                R.move(g, speed=0.02)
                R.hold(n(4))
                self.when_settled(R, "right", self.grab_old)

            def grab_old(self):
                log(f"AT OLD FUSE: lag {self.lag('right', self.R):.1f} mm", flush=True)
                self.R.grip_to(FUSE_GRIP, n(10))
                self.at(len(self.R.steps), self.felt_old)

            def felt_old(self):
                # Fingers closed on the fuse: they stay open by its width (arm B's gripper reading).
                self.grip_full = self.gripper("right")
                log(f"GRIPPED OLD FUSE: gripper reads {self.grip_full:.3f} (closed on nothing: 0)", flush=True)
                self.draw_out()

            def gripper(self, side):
                return V.split_state(obs_box[0].state["joint_pos"])[side][1]

            def holding(self, side):
                return self.gripper(side) > 0.5 * self.grip_full

            def draw_out(self, seg=0):
                R = self.R
                _, c_quat = self.carrier_pose()
                up = rotate(c_quat, (1.0, 0.0, 0.0))
                if seg:
                    log(f"LIFT {seg}: lag {self.lag('right', R):.1f} mm", flush=True)
                if seg < LIFT_SEGS:
                    R.move(R.pos + ROOT_OFFSET + up * 0.006, speed=0.006)
                    R.hold(n(2))
                    self.when_settled(R, "right", lambda: self.draw_out(seg + 1), tries=10, tol=0.002)
                    return
                R.move(R.pos + ROOT_OFFSET + np.array([-0.03, 0.0, 0.02]), speed=0.03)
                self.when_settled(R, "right", self.check_drawn, tries=8)

            def check_drawn(self):
                g = self.gripper("right")
                if self.holding("right"):
                    log(f"OLD FUSE IN HAND: gripper reads {g:.3f}", flush=True)
                    self.to_tray()
                    return
                self.tries_old = getattr(self, "tries_old", 0) + 1
                log(f"OLD FUSE NOT IN HAND (gripper reads {g:.3f}): it slipped out; trying again "
                    f"({self.tries_old})", flush=True)
                if self.tries_old > 2:
                    self.to_tray()
                    return
                R = self.R
                R.grip_to(FUSE_OPEN_CMD, n(4))
                R.move(R.pos + ROOT_OFFSET + np.array([-0.04, 0.0, 0.04]), speed=0.05)
                self.sync()
                self.at(len(R.steps), self.plan_remove)

            def to_tray(self):
                self.report("OLD DRAWN OUT")
                R = self.R
                tx, ty = TRAY_XY
                # Back toward the robot first, then across to the tray: the spare cartons (open lids up to 20 cm) stand
                # between the holder and the tray.
                here = R.pos + ROOT_OFFSET
                R.move(np.array([here[0], here[1], max(here[2], 0.22)]), speed=0.06)
                R.move(np.array([0.17, -0.02, 0.22]), speed=0.08)
                R.move(np.array([tx, ty, 0.16]), speed=0.08, min_steps=n(25))
                R.hold(n(5))
                R.grip_to(0.6, n(15))
                R.move(np.array([tx - 0.02, ty + 0.03, 0.19]), speed=0.08)
                self.sync()
                self.at(len(R.steps), self.plan_pick)

            # 3. Spare in (arm B) ---------------------------------------------------------------------------
            def plan_pick(self):
                self.report("OLD DROPPED")
                R = self.R
                pos, quat = self.spare_pos, np.array(L_.spare_pose(self.spare_k)[3:])
                self.spare_quat = quat
                self.spare_d = qmul(conj(SPARE_VIRTUAL_Q), quat)
                g, q = self.hand_for(pos, SPARE_VIRTUAL_Q)
                f_dir = rotate(q, (0.0, 0.0, -1.0))
                log(f"SPARE {L_.SPARE_RATINGS[self.spare_k]} A at {np.round(pos, 4)}, grasp {np.round(g, 4)}", flush=True)
                R.grip_to(SPARE_OPEN_CMD, n(2))
                R.move(g - f_dir * 0.06 + np.array([0.0, 0.0, 0.02]), q, min_steps=n(30))
                R.move(g - f_dir * 0.03, speed=0.05)
                R.move(g, speed=0.02)
                R.hold(n(4))
                self.when_settled(R, "right", self.grab_spare)

            def grab_spare(self):
                log(f"AT SPARE: lag {self.lag('right', self.R):.1f} mm", flush=True)
                R = self.R
                R.grip_to(FUSE_GRIP, n(10))
                # Fingers closed on the spare where the camera saw it: how it sits in the hand (arm B's joints).
                self.at(len(R.steps), self.measure_grasp)
                R.move(R.pos + ROOT_OFFSET + np.array([0.0, 0.0, 0.03]), speed=0.03)
                R.move(R.pos + ROOT_OFFSET + np.array([-0.03, 0.0, 0.01]), speed=0.05)
                self.when_settled(R, "right", self.to_carrier, tries=8)

            def measure_grasp(self):
                g, gq = grasp_point("right")
                self.grasp = GraspModel(rotate(conj(gq), self.spare_pos - g), qmul(conj(gq), self.spare_quat))
                log(f"SPARE GRASPED: fuse centre in the hand {np.round(self.grasp.rel_pos * 1000, 1)} mm", flush=True)

            def fuse_est(self):
                g, gq = grasp_point("right")
                return g + rotate(gq, self.grasp.rel_pos), qmul(gq, self.grasp.rel_quat)

            def to_carrier(self):
                c_pos, c_quat = self.carrier_pose()
                seat = c_pos + rotate(c_quat, props.seat_in_carrier())
                outward = rotate(c_quat, (0.0, 0.0, 1.0))
                quat = qmul(c_quat, self.spare_d)
                self.ins = (seat, quat, outward)
                self.up = rotate(c_quat, (1.0, 0.0, 0.0))
                self.corr = np.zeros(3)
                g, q = self.grasp.grasp_for(seat + self.up * HOVER + outward * 0.045, quat)
                log(f"SPARE HELD: rel {np.round(self.grasp.rel_pos * 1000, 1)} mm; seat {np.round(seat, 4)}",
                    flush=True)
                R = self.R
                R.move(g + np.array([-0.02, 0.0, 0.03]), q, speed=0.06, min_steps=n(30))
                R.move(g, q, speed=0.03)
                R.hold(n(5))
                self.when_settled(R, "right", lambda: self.insert(0))

            def insert(self, attempt):
                seat, quat, outward = self.ins
                f_pos, f_quat = self.fuse_est()
                want = seat + self.up * HOVER + outward * 0.045
                inner = f_pos + rotate(f_quat, (0.0, 0.0, props.FUSE_L / 2))
                err = inner - (want + rotate(quat, (0.0, 0.0, props.FUSE_L / 2)))
                tilt = math.degrees(math.acos(min(1.0, abs(float(np.dot(rotate(f_quat, (0, 0, 1)),
                                                                        rotate(quat, (0, 0, 1))))))))
                log(f"PRE-PLACE {attempt}: inner end error {np.round(err * 1000, 1)} mm, axis off {tilt:.1f} deg",
                    flush=True)
                if attempt < 4 and (np.abs(err).max() > 0.0006 or tilt > 1.5):
                    if attempt:
                        self.corr = self.corr + err
                    g, q = self.grasp.grasp_for(want - self.corr, quat)
                    self.R.move(g, q, speed=0.01)
                    self.R.hold(n(4))
                    self.when_settled(self.R, "right", lambda: self.insert(attempt + 1), tol=0.0015)
                    return
                self.slide(0)

            def slide(self, seg):
                seat, quat, outward = self.ins
                if seg == 3:
                    self.lower(0)
                    return
                g, q = self.grasp.grasp_for(seat + self.up * HOVER + outward * (0.030, 0.015, 0.0)[seg] - self.corr,
                                            quat)
                self.R.move(g, q, speed=0.006)
                self.R.hold(n(3))
                self.when_settled(self.R, "right", lambda: self.slide(seg + 1), tries=8, tol=0.0015)

            def lower(self, seg):
                seat, quat, outward = self.ins
                if seg:
                    f_pos, _ = self.fuse_est()
                    log(f"LOWER {seg}: fuse off seat (arm B's joints) {np.round((f_pos - seat) * 1000, 1)} mm", flush=True)
                if seg == 2:
                    self.release_spare()
                    return
                g, q = self.grasp.grasp_for(seat + self.up * (0.004, 0.0005)[seg] + outward * 0.001 - self.corr, quat)
                self.R.move(g, q, speed=0.004)
                self.R.hold(n(4))
                self.when_settled(self.R, "right", lambda: self.lower(seg + 1), tries=8, tol=0.0015)

            def release_spare(self):
                f_pos, _ = self.fuse_est()
                log(f"INSERTED: fuse off seat as let go (arm B's joints) {np.round((f_pos - self.ins[0]) * 1000, 1)} mm",
                    flush=True)
                R = self.R
                g_quat = grasp_point("right")[1]
                f_dir = rotate(g_quat, (0.0, 0.0, -1.0))
                R.grip_to(FUSE_OPEN_CMD, n(8))
                R.move(R.pos + ROOT_OFFSET - f_dir * 0.05, speed=0.03)
                R.hold(n(5))
                self.at(len(R.steps), self.after_release)

            def after_release(self):
                R = self.R
                R.move(np.array([0.15, -0.20, 0.22]), speed=0.08, min_steps=n(15))
                self.sync()
                self.at(len(R.steps), self.plan_close)

            # 4. Close (arm A) ------------------------------------------------------------------------------
            @staticmethod
            def close_target(th, lead=CLOSE_LEAD):
                edge = carrier_pt(th, CLOSE_CONTACT[0], CLOSE_CONTACT[2])
                edge = edge + np.array([math.cos(th), 0.0, math.sin(th)]) * lead
                return edge - np.array([FINGER_END, 0.0, TIP_HALF_W])

            def plan_close(self):
                self.report("SPARE RELEASED")
                L = self.L
                th0 = self.theta()
                start = self.close_target(th0)
                L.grip_to(CLOSED, n(2))
                L.move(start + np.array([-0.04, 0.0, -0.01]), FORWARD, min_steps=n(30))
                L.move(start + np.array([-0.01, 0.0, -0.006]), speed=0.04)
                L.move(start + np.array([0.0, 0.0, -0.004]), speed=0.02)
                L.hold(n(3))
                self.stop_rad = math.radians(-2.0)
                self.stuck = 0
                self.when_settled(L, "left", lambda: self.sweep(th0, CLOSE_LEAD, 0, th0))

            def sweep(self, th_cmd, lead, k, th_prev):
                """Lift the tab along the arc in 6-deg steps ahead of the carrier; push 2 mm harder if it stalls."""
                th = self.th = min(OPEN_RAD, max(0.0, self.theta_from_push(0.0))) if k else self.th
                log(f"SWEEP {k}: carrier at {math.degrees(th):.1f} deg (arm A's joints), lead {lead * 1000:.1f} mm, "
                    f"lag {self.lag('left', self.L):.1f} mm", flush=True)
                # Shut: the fingers' contact is back at the shut tab's place, or they have stopped at the end of
                # their arc with the push at its strongest (the latch holds the carrier there).
                self.stuck = self.stuck + 1 if k and abs(th - th_prev) < math.radians(0.3) else 0
                done = th < math.radians(2.0) or (th_cmd <= self.stop_rad + 1e-6 and lead >= 0.012 and self.stuck >= 3)
                if done or k > 40:
                    self.L.hold(n(8))
                    self.at(len(self.L.steps), self.back_off)
                    return
                if k and th_prev - th < math.radians(0.5):
                    lead = min(lead + 0.002, 0.012)
                th_cmd = max(min(th_cmd, th) - math.radians(6.0), self.stop_rad)
                L = self.L
                start = L.pos + ROOT_OFFSET
                goal = self.close_target(th_cmd, lead)
                m = n(6)
                for i in range(1, m + 1):
                    L.path([(start + (goal - start) * i / m, FORWARD)])
                L.hold(n(2))
                self.at(len(L.steps), lambda: self.sweep(th_cmd, lead, k + 1, th))

            def back_off(self):
                self.report("SWEPT")
                L = self.L
                L.move(L.pos + ROOT_OFFSET + np.array([-0.07, 0.0, 0.0]), speed=0.06)
                L.move(L_PARK, speed=0.10)
                self.sync()
                L.hold(n(5))
                self.sync()
                self.at(len(L.steps), self.check_shut)

            def check_shut(self):
                seen = self.look_top("after closing")[1]
                if seen == "open" and not getattr(self, "reclosed", False):
                    self.reclosed = True
                    self.th = OPEN_RAD
                    log("F2 still open: closing again", flush=True)
                    self.plan_close()
                    return
                self.th = 0.0 if seen == "shut" else self.th
                self.L.hold(n(60))
                self.sync()

        self._Planner = Planner
        self._dt = 1.0 / control_hz
        self._speed = speed
        self._log = log
        self.planner = None
        self.worst = {"left": 0.0, "right": 0.0}

    def _start(self):
        n = lambda steps: max(1, round(steps * (1.0 / self._dt) / 10.0))  # noqa: E731
        l_pos, l_quat = self._grasp_point("left")
        r_pos, r_quat = self._grasp_point("right")
        L = ArmTimeline(l_pos - ROOT_OFFSET, l_quat, self._dt, self._speed)
        R = ArmTimeline(r_pos - ROOT_OFFSET, r_quat, self._dt, self._speed)
        planner = self._Planner(L, R)
        settle = n(15)
        L.hold(settle)
        R.hold(settle)
        if os.environ.get("FUSE_HOLD_CHECK") != "1":  # hold check (fuse_sim.task.HOLD_CHECK): the arms stay at ready
            planner.at(1, planner.survey)
            planner.at(settle, planner.plan_open)
        self.planner = planner
        self._L, self._R = L, R
        self._q_arm = {s: self._q_meas(s).astype(float).copy() for s in ("left", "right")}

    def _joint_action(self, side, pos_root, quat, grip):
        pos = pos_root + ROOT_OFFSET + np.array([0.0, 0.0, V.DESK_Z])  # world
        tcp = pos + rotate(quat, (0.0, 0.0, -TCP_EXTRA))
        rot = qmul(quat, Q_REL)
        q, pe, _ = ik.solve(tcp, rot, self._q_arm[side], base_pos=station.BASES[side], iters=60)
        self._q_arm[side] = q
        if pe > 0.005 and pe > self.worst[side] + 0.002:
            self._log(f"IK {side} k={self.state_k[0]} error {pe * 1000:.1f} mm at {np.round(pos_root + ROOT_OFFSET, 4)}",
                      flush=True)
        self.worst[side] = max(self.worst[side], pe)
        return np.r_[q, grip]

    def next_action(self, obs) -> np.ndarray:
        self._obs_box[0] = obs
        if self.planner is None:
            self._start()
        k = self.state_k[0]
        L, R = self._L, self._R
        self.planner.run_due(k)
        L.pad_to(len(R.steps))
        R.pad_to(len(L.steps))
        if k >= len(L.steps):
            L.hold(1)
            R.hold(1)
        (lp, lq, lg), (rp, rq, rg) = L.steps[k], R.steps[k]
        self.state_k[0] = k + 1
        return np.concatenate([self._joint_action("left", lp, lq, lg), self._joint_action("right", rp, rq, rg)])
