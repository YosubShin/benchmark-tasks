"""The scripted demo, closed loop.

1. Arm B (right), fingers down, closes across the open cover's back half (one finger on the hinge barrel, the other
   on the back half's ear), lifts it off its block, carries it over the START button 3 mm beyond its axis (so the back
   half's bottom lip clears the base's flange), lowers it until it hovers 0.7 mm above the panel and slides it onto
   the axis: the lip goes under the flange. It lets go and holds the back half down with its shut fingertips on the
   cover's top.
2. Arm A (left), fingers shut and tilted so its wrist stays clear of arm B's hand, pushes the open front half's ear
   toward the back half's, following it as it swings about the hinge, until the cover is shut (the front half's lip
   goes under the flange as it swings in). Both arms back off.
3. Arm A picks the open padlock off its pedestal from above (lying level, shackle toward the cover's ears; fingers on
   the body's sides), slides it so the two
   ears enter the gap under the shackle's short leg, lines the leg up with the ear holes from the measured tip
   position and draws the lock back so the leg passes through both ears; arm B pushes the shackle's bend shut while
   arm A holds the body; both let go.

``control=True``: steps 1 and 2 only; once both hands are clear the after-release check is forced (the cover, shut but
not padlocked, must open).

It reads the simulator's state (the cover's and padlock's poses), like any scripted demo with privileged state;
``--policy pushbutton-lockout-demo`` (pushbutton_sim.policy) plays it through Inspect Robots. Grasp-point targets
become yam_arena's 14-D joint action through yam_arena's own IK every control step. Import after Isaac Sim has
started.
"""

from __future__ import annotations

import math
import os

import numpy as np

from yam_arena import ik, station
from yam_arena.constants import READY

TIP_BELOW_GRASP = 0.013
"""The fingertips' ends are 13 mm past the grasp point (yam_arena PAD_TIP_PAST_TCP)."""
OFFSET_X = 0.003
"""The cover is set down this far beyond the button's axis (+x), then slid onto it."""
PUSH_EAR_Z = 0.008
"""Arm A pushes the front half's ear this high above the cover's top."""
PUSH_DEPTH = 0.003
FINGER_T = 0.005
"""Thickness of a finger near its end (the pushing face is this far from the grasp point when the fingers are shut)."""
CLOSED_DONE_DEG = 1.0
HOVER = 0.0007
"""The cover's bottom hovers this far above the panel while it is slid onto the axis."""


class Demo:
    """``next_action()`` gives the 14-D joint action for the next control step."""

    def __init__(self, STATION, control_hz: float = 20.0, speed: float = 0.10, control: bool = False, log=print):
        from pushbutton_sim import station as hb
        from pushbutton_sim.kit import (CLOSED, DOWN, FORWARD, OPEN, ROOT_OFFSET, ArmTimeline, GraspModel, conj, q_axis,
                                        qmul, rotate)
        from pushbutton_sim.kit import grasp_point as grasp_point_world
        from pushbutton_sim.spec import BUTTON_XY, PANEL_Z
        from pushbutton_sim.task import PV, lock_start_pose
        from pushbutton_sim.usd import pb_props as props

        view, task = STATION.env, STATION.task
        st = task.state(view)
        state_k = [0]
        self.state_k = state_k
        origin = view.scene.env_origins[0].cpu().numpy()
        robot = view.scene["robot"]
        dt = 1.0 / control_hz
        hold_k = lambda s: max(1, round(s / dt))  # noqa: E731
        EX, EY, EZ = np.eye(3)
        YAW90 = q_axis((0, 0, 1), math.pi / 2)
        DOWN_NX = qmul(q_axis((0, 0, 1), -math.pi / 2), DOWN)
        BX, BY = BUTTON_XY["start"]
        BUTTON = np.array([BX, BY, PANEL_Z])
        cover = st.cover
        lock = view.scene["padlock"]
        lock_body = lock.find_bodies("body")[0][0]
        shackle_joint = lock.find_joints("shackle_joint")[0][0]

        def grasp_point(side):
            p, q = grasp_point_world(robot, side)
            return p - origin, q

        for side, name in (("left", "left_robot"), ("right", "right_robot")):
            qj = view.scene[name].data.joint_pos.torch[0, :6].cpu().numpy()
            tcp, _ = ik.tcp_pose(qj, station.BASES[side])
            log(f"CHECK {side}: their TCP {np.round(tcp, 4)} our grasp point {np.round(grasp_point(side)[0] + origin, 4)}",
                flush=True)

        def half_pose(which):
            i = st.half_b if which == "b" else st.half_f
            pos = cover.data.body_pos_w.torch[0, i].cpu().numpy() - origin
            return pos, cover.data.body_quat_w.torch[0, i].cpu().numpy()

        def on_half(which, p):
            pos, q = half_pose(which)
            return pos + rotate(q, p)

        def hinge_deg():
            return math.degrees(float(cover.data.joint_pos.torch[0, st.hinge]))

        # Arm B's grasp on the back half (cover frame): between the hinge barrel's outside and the back ear's outside.
        y_barrel = -props.HINGE_R - props.BARREL / 2
        y_ear = props.EAR_Y_IN[1]
        COVER_GRASP = np.array([0.0, (y_barrel + y_ear) / 2, props.HEIGHT + 0.004])
        log(f"COVER grasp span {(y_ear - y_barrel) * 1000:.1f} mm", flush=True)

        FINGER_END, GRASP_IN_LOCK, FACE_GAP, PIERCE = 0.012, (0.0, 0.0, -0.012), 0.0013, 0.010
        LOCK_QUAT = np.array(lock_start_pose()[3:])
        # Arm A holds the lying padlock's body near its bottom end from above, closing on the body's two 19 mm sides,
        # fingers tilted 35 degrees toward the cover (its wrist stays to the left, clear of arm B's).
        QA = qmul(q_axis((1, 0, 0), math.radians(-35)), qmul(q_axis((0, 0, 1), -math.pi / 2), DOWN))
        A_FINGER_DIR = rotate(QA, (0.0, 0.0, -1.0))
        Q_PRESS = qmul(q_axis((1, 0, 0), math.radians(45)), DOWN)
        F_PRESS = rotate(Q_PRESS, (0.0, 0.0, -1.0))
        """Arm B pushes the shackle's bend with its shut fingertips, fingers toward +y and 45 degrees down (its wrist
        stays well to the right of arm A's)."""
        R_WAIT = np.array((0.16, -0.20, 0.20))
        L_WAIT = np.array((0.14, 0.18, 0.20))
        Q_EAR = qmul(q_axis((1, 0, 0), math.radians(-45)), qmul(q_axis((0, 0, 1), -math.pi / 2), DOWN))
        F_EAR = rotate(Q_EAR, (0.0, 0.0, -1.0))
        """Arm A pushes the front half's ear shut with its shut fingers closing along x and pointing toward -y and 45
        degrees down, so its wrist stays well to the left of arm B's hand on the cover's top."""
        EAR_SIDE = props.EAR_T + props.EAR_GAP / 2
        """From the ear holes' centre to the inner ear's inner face (along the hole axis)."""

        class Planner:
            def __init__(self, L, R):
                self.L, self.R, self.events = L, R, {}

            def at(self, step, fn):
                self.events.setdefault(step, []).append(fn)

            def run_due(self, k):
                for fn in self.events.pop(k, []):
                    fn()

            def when_settled(self, tl, side, fn, tries=30, tol=0.003):
                def check():
                    lag = np.linalg.norm(grasp_point(side)[0] - (tl.pos + ROOT_OFFSET))
                    if lag < tol or check.tries <= 0:
                        if lag >= tol:
                            log(f"SETTLE timeout ({side}): lag {lag * 1000:.1f} mm", flush=True)
                        fn()
                    else:
                        check.tries -= 1
                        tl.hold(hold_k(0.1))
                        self.at(len(tl.steps), check)

                check.tries = tries
                self.at(len(tl.steps), check)

            def sync(self):
                n = max(len(self.L.steps), len(self.R.steps))
                self.L.pad_to(n)
                self.R.pad_to(n)
                return n

            def report(self, tag):
                rep = task.episode_report(view, 0)
                log(f"{tag}: milestone {rep['milestone']} off axis {rep['cover_off_axis_mm']} mm seat "
                    f"{rep['cover_seat_mm']} mm tilt {rep['cover_tilt_deg']} hinge {rep['hinge_deg']} deg, START "
                    f"{rep['start_travel_max_mm']} STOP {rep['stop_travel_max_mm']} mm, events {rep['semantic_events']}",
                    flush=True)
                return rep

            # -- 1. carry the cover onto the START button --------------------------------------------------------------
            def plan_cover_pick(self):
                pos, q = half_pose("b")
                self.grasp = pos + rotate(q, COVER_GRASP)
                R = self.R
                R.grip_to(OPEN, hold_k(0.2))
                R.move(self.grasp + np.array([0.0, 0.0, 0.06]), DOWN, min_steps=hold_k(2.5))
                R.move(self.grasp + np.array([0.0, 0.0, 0.02]), speed=0.05)
                R.hold(hold_k(0.3))
                self.when_settled(R, "right", self._cover_descend)

            def _cover_descend(self):
                err = grasp_point("right")[0] - (self.R.pos + ROOT_OFFSET)
                self.R.move(self.grasp - err * np.array([1.0, 1.0, 0.0]), speed=0.02)
                self.R.hold(hold_k(0.4))
                self.when_settled(self.R, "right", self._cover_close)

            def _cover_close(self):
                g = grasp_point("right")[0]
                log(f"GRASP cover: hand {np.round(g, 4)} planned {np.round(self.grasp, 4)}", flush=True)
                self.R.grip_to(CLOSED, hold_k(1.0))
                self.at(len(self.R.steps), self._cover_lift)

            def _cover_to(self, pos, speed, quat=None):
                g, q = self.cmodel.grasp_for(pos, self.cover_q if quat is None else quat)
                self.R.move(g, q, speed=speed)

            def _cover_lift(self):
                self.cmodel = GraspModel.measure(robot, "right", cover)
                p, q = half_pose("b")
                self.cover_q = np.array([0.0, 0.0, 0.0, 1.0])
                log(f"COVER in hand (mm) {np.round(self.cmodel.rel_pos * 1000, 1)}, hinge {hinge_deg():.1f} deg", flush=True)
                carry_z = PANEL_Z + 0.04
                self._cover_to(np.array([p[0], p[1], carry_z]), 0.04)
                above = BUTTON + np.array([OFFSET_X, 0.0, 0.03])
                self._cover_to(np.array([above[0], above[1], carry_z]), 0.06)
                self._cover_to(above, 0.05)
                self.R.hold(hold_k(0.5))
                self.place_corr = np.zeros(3)
                self.when_settled(self.R, "right", lambda: self._place(0))

            def _place(self, attempt):
                """Lower the cover to hover HOVER above the panel, OFFSET_X beyond the axis (lips clear of the flange,
                below it), correcting from the measured cover position."""
                p, _ = half_pose("b")
                goal = BUTTON + np.array([OFFSET_X, 0.0, 0.006 if attempt <= 1 else HOVER])
                err = p - goal
                log(f"PLACE {attempt}: cover error (mm) {np.round(err * 1000, 1)}, hinge {hinge_deg():.1f} deg", flush=True)
                if attempt == 0:
                    self._cover_to(goal, 0.02)
                elif attempt == 1:
                    self.place_corr = self.place_corr + err
                    self._cover_to(goal - self.place_corr - np.array([0.0, 0.0, 0.006 - HOVER]), 0.01)
                elif attempt < 5 and np.abs(err).max() > 0.0006:
                    self.place_corr = self.place_corr + err
                    self._cover_to(goal - self.place_corr, 0.006)
                else:
                    self._slide(0)
                    return
                self.R.hold(hold_k(0.5))
                self.when_settled(self.R, "right", lambda: self._place(attempt + 1))

            def _slide(self, attempt):
                """Slide onto the button's axis, still hovering: the back half's lip goes under the flange."""
                p, _ = half_pose("b")
                goal = BUTTON + np.array([0.0, 0.0, HOVER])
                err = p - goal
                log(f"SLIDE {attempt}: cover off the hover pose on the axis (mm) {np.round(err * 1000, 1)}", flush=True)
                if attempt < 5 and np.abs(err).max() > 0.0005:
                    self.place_corr = self.place_corr + (err if attempt > 0 else np.zeros(3))
                    self._cover_to(goal - self.place_corr, 0.004)
                    self.R.hold(hold_k(0.5))
                    self.when_settled(self.R, "right", lambda: self._slide(attempt + 1))
                    return
                R = self.R
                R.grip_to(OPEN, hold_k(0.6))
                R.hold(hold_k(0.4))
                self.at(len(R.steps), self._placed)

            def _placed(self):
                """Arm B lets go and holds the back half down with its shut fingertips on the cover's top."""
                self.report("PLACED")
                R = self.R
                R.move(R.pos + ROOT_OFFSET + np.array([0.0, 0.0, 0.03]), speed=0.03)
                R.grip_to(CLOSED, hold_k(0.4))
                top = on_half("b", (0.016, 0.0, props.HEIGHT))
                R.move(top + np.array([0.0, 0.0, TIP_BELOW_GRASP + 0.01]), DOWN, speed=0.04)
                R.move(top + np.array([0.0, 0.0, TIP_BELOW_GRASP - 0.002]), speed=0.01)
                R.hold(hold_k(0.4))
                self.at(len(R.steps), self.push_start)

            # -- 2. close the cover -----------------------------------------------------------------------------------
            def push_point(self):
                """The leading edge of the front half's ear (the edge that meets the back half's ear) and the direction
                it moves as the cover closes, both in the world."""
                pos, q = half_pose("f")
                p_loc = np.array([props.EAR_HW, sum(props.EAR_Y_OUT) / 2, props.HEIGHT + PUSH_EAR_Z])
                d = p_loc - props.hinge_point()
                tangent = np.array([d[1], -d[0], 0.0]) / math.hypot(d[0], d[1])
                return pos + rotate(q, p_loc), rotate(q, tangent)

            def push_grasp(self, p, d, depth):
                """Arm A's grasp point that puts its shut fingers' face (the one toward +x) on p, ``depth`` past it."""
                return p - (FINGER_T - depth) * d - TIP_BELOW_GRASP * F_EAR

            def push_start(self):
                self.report("HOLDING back half")
                self.sync()
                p, d = self.push_point()
                L = self.L
                L.grip_to(CLOSED, hold_k(0.3))
                start = self.push_grasp(p, d, -0.012)
                L.move(start - 0.04 * F_EAR, Q_EAR, min_steps=hold_k(3.0))
                L.move(start, speed=0.03)
                L.hold(hold_k(0.3))
                self.when_settled(L, "left", lambda: self._push(0))

            def _push(self, k):
                th = hinge_deg()
                p, d = self.push_point()
                if k % 3 == 0:
                    rep = task.episode_report(view, 0)
                    log(f"PUSH {k}: hinge {th:.1f} deg, ear edge {np.round(p, 4)}, dir {np.round(d, 2)}, hand "
                        f"{np.round(grasp_point('left')[0], 4)}, cover off axis {rep['cover_off_axis_mm']} mm", flush=True)
                if th < CLOSED_DONE_DEG or k > 60:
                    self.L.hold(hold_k(0.5))
                    self.at(len(self.L.steps), self._push_done)
                    return
                self.L.move(self.push_grasp(p, d, PUSH_DEPTH), speed=0.02, min_steps=2)
                self.L.hold(hold_k(0.15))
                self.at(len(self.L.steps), lambda: self._push(k + 1))

            def _push_done(self):
                self.report("CLOSED")
                L, R = self.L, self.R
                L.move(L.pos + ROOT_OFFSET - 0.04 * F_EAR, speed=0.03)
                L.move(np.array([0.17, 0.03, 0.20]), speed=0.06)
                self.sync()
                R.move(R.pos + ROOT_OFFSET + np.array([0.0, 0.0, 0.04]), speed=0.03)
                R.move(R_WAIT, speed=0.08)
                if control:
                    L.move(L_WAIT, speed=0.08)
                    self.sync()
                    L.hold(hold_k(1.5))
                    self.at(len(L.steps), lambda: (self.report("CONTROL check"), task.force_function_test(view)))
                    L.hold(hold_k(4.0))
                    return
                self.sync()
                self.at(len(L.steps), self.plan_pick)

            # -- 3. padlock -------------------------------------------------------------------------------------------
            def lock_pose(self):
                return (lock.data.body_pos_w.torch[0, lock_body].cpu().numpy() - origin,
                        lock.data.body_quat_w.torch[0, lock_body].cpu().numpy())

            def travel(self):
                return float(lock.data.joint_pos.torch[0, shackle_joint])

            def tip(self):
                pos, quat = self.lock_pose()
                return pos + rotate(quat, PV.short_tip(self.travel()))

            def hole(self):
                return on_half("b", (0.0, props.EAR_HOLE_Y, props.EAR_HOLE_Z))

            def face_point(self, hole):
                return hole - (EAR_SIDE + FACE_GAP) * EY

            def lock_center_for_tip(self, tip_target):
                return np.asarray(tip_target) - rotate(LOCK_QUAT, PV.short_tip(self.travel()))

            def _lock_to(self, center, speed):
                self.L.move(self.lmodel.grasp_for(center, LOCK_QUAT)[0], speed=speed)

            def plan_pick(self):
                self.report("PICK padlock")
                start = np.array(lock_start_pose()[:3])
                grasp, _ = GraspModel.nominal(QA, LOCK_QUAT, GRASP_IN_LOCK).grasp_for(start, LOCK_QUAT)
                L = self.L
                L.grip_to(OPEN, hold_k(0.2))
                L.move(grasp - 0.05 * A_FINGER_DIR, QA, min_steps=hold_k(3.0))
                L.hold(hold_k(0.4))
                L.move(grasp - 0.015 * A_FINGER_DIR, speed=0.04)
                L.hold(hold_k(0.4))
                self.when_settled(L, "left", self._final_approach)

            def _final_approach(self):
                start = np.array(lock_start_pose()[:3])
                grasp, _ = GraspModel.nominal(QA, LOCK_QUAT, GRASP_IN_LOCK).grasp_for(start, LOCK_QUAT)
                err = grasp_point("left")[0] - (self.L.pos + ROOT_OFFSET)
                self.L.move(grasp - err, speed=0.02)
                self.L.hold(hold_k(0.4))
                self.L.grip_to(CLOSED, hold_k(1.0))
                self.L.move(grasp - err + np.array([0.0, 0.0, 0.03]), speed=0.03)
                self.when_settled(self.L, "left", self.lock_carry)

            def lock_carry(self):
                self.lmodel = GraspModel.measure(robot, "left", lock)
                log("LOCK grasp rel_pos (mm)", np.round(self.lmodel.rel_pos * 1000, 1), flush=True)
                self.tip_entry = self.face_point(self.hole()) - 0.022 * EX
                c = self.lock_center_for_tip(self.tip_entry)
                here = self.lock_pose()[0]
                self._lock_to(here + 0.03 * EZ, 0.04)
                self._lock_to(c + 0.03 * EZ + 0.01 * EY, 0.06)
                self._lock_to(c + 0.01 * EY, 0.03)
                self._lock_to(c, 0.02)
                self.L.hold(hold_k(0.6))
                self.when_settled(self.L, "left", lambda: self._entry_correct(np.zeros(3), 0))

            def _entry_correct(self, correction, attempt):
                err = self.tip() - self.tip_entry
                bp, _ = self.lock_pose()
                log(f"ENTRY attempt {attempt}: tip error (mm) {np.round(err * 1000, 1)}, shackle out "
                    f"{self.travel() * 1000:.1f} mm, body {np.round(bp, 4)}", flush=True)
                correction = correction + err
                if attempt < 4 and np.abs(err).max() > 0.001:
                    self._lock_to(self.lock_center_for_tip(self.tip_entry - correction), 0.01)
                    self.L.hold(hold_k(0.6))
                    self.when_settled(self.L, "left", lambda: self._entry_correct(correction, attempt + 1))
                    return
                self.correction = correction
                self.tip_over = self.face_point(self.hole())
                self._lock_to(self.lock_center_for_tip(self.tip_over - correction), 0.012)
                self.L.hold(hold_k(0.6))
                self.when_settled(self.L, "left", lambda: self._over_correct(0))

            def _over_correct(self, attempt):
                self.tip_over = self.face_point(self.hole())
                err = self.tip() - self.tip_over
                log(f"OVER hole attempt {attempt}: tip error (mm) {np.round(err * 1000, 1)}, lock rot vs plan "
                    f"{np.round(qmul(self.lock_pose()[1], conj(LOCK_QUAT)), 3)}", flush=True)
                self.correction = self.correction + err
                if attempt < 5 and np.abs(err).max() > 0.0008:
                    self._lock_to(self.lock_center_for_tip(self.tip_over - self.correction), 0.008)
                    self.L.hold(hold_k(0.6))
                    self.when_settled(self.L, "left", lambda: self._over_correct(attempt + 1))
                    return
                self._pierce(0, 0.0)

            def _pierce(self, k, dy):
                """Draw the lock back (+y) in 2 mm steps so the short leg passes through both ears, re-centring it on
                the holes from the measured tip each step."""
                tip = self.tip()
                err = tip - (self.tip_over + float((tip - self.tip_over) @ EY) * EY)
                if k % 2 == 0 or k < 4:
                    log(f"PIERCE {k}: tip past start {float((tip - self.tip_over) @ EY) * 1000:.1f} mm, off the hole's "
                        f"axis (mm) {np.round(err * 1000, 1)}", flush=True)
                if dy >= PIERCE or k > 25:
                    self.L.hold(hold_k(0.6))
                    self.when_settled(self.L, "left", self._threaded)
                    return
                self.correction = self.correction + 0.7 * (err - float(err @ EY) * EY)
                dy = min(dy + 0.002, PIERCE)
                self._lock_to(self.lock_center_for_tip(self.tip_over + dy * EY - self.correction), 0.006)
                self.L.hold(hold_k(0.3))
                self.at(len(self.L.steps), lambda: self._pierce(k + 1, dy))

            def _threaded(self):
                log(f"PIERCED: tip {np.round(self.tip(), 4)} hole {np.round(self.hole(), 4)}", flush=True)
                self.press_start()

            def bend_outer(self):
                pos, quat = self.lock_pose()
                return pos + rotate(quat, (0.0, 0.0, PV.bend_top(self.travel())))

            def press_start(self):
                R = self.R
                R.pad_to(len(self.L.steps))
                R.grip = CLOSED
                tip = self.bend_outer() - 0.002 * EY
                g = tip - TIP_BELOW_GRASP * F_PRESS
                R.move(g - 0.04 * F_PRESS, Q_PRESS, min_steps=hold_k(2.5))
                R.move(g, speed=0.03)
                R.hold(hold_k(0.6))
                self.body0 = self.lock_pose()[0]
                self.when_settled(R, "right", lambda: self._press(0))

            def _press(self, k):
                latched = bool(st.latched[0])
                if latched or k > 30:
                    log(f"PRESS done after {k}: travel {self.travel() * 1000:.1f} mm, latched {latched}, through "
                        f"{bool(st.latched_through[0])}", flush=True)
                    self.R.move(self.R.pos + ROOT_OFFSET + 0.003 * EY, speed=0.02)
                    self.R.hold(hold_k(0.4))
                    self.L.pad_to(len(self.R.steps))
                    self.release()
                    return
                if k % 3 == 0:
                    log(f"PRESS {k}: travel {self.travel() * 1000:.1f} mm, bend {np.round(self.bend_outer(), 4)}, hand "
                        f"{np.round(grasp_point('right')[0], 4)}", flush=True)
                drift = float((self.lock_pose()[0] - self.body0) @ EY)
                self.L.pad_to(len(self.R.steps))
                self.L.move(self.L.pos + ROOT_OFFSET - drift * EY, speed=0.02)
                self.R.move(self.R.pos + ROOT_OFFSET + 0.004 * EY, speed=0.02)
                self.R.hold(hold_k(0.2))
                self.sync()
                self.at(len(self.R.steps), lambda: self._press(k + 1))

            def release(self):
                R, L = self.R, self.L
                R.move(R.pos + ROOT_OFFSET - 0.04 * F_PRESS, speed=0.03)
                R.move(R_WAIT, DOWN, speed=0.10)
                L.pad_to(len(R.steps) - hold_k(1.2))
                L.grip_to(0.55, hold_k(1.0))
                L.move(L.pos + ROOT_OFFSET - 0.05 * A_FINGER_DIR, speed=0.03)
                L.move(L_WAIT, speed=0.06)
                L.hold(hold_k(1.0))
                self.sync()
                self.at(len(L.steps), lambda: self.report("RELEASED"))
                L.hold(hold_k(4.0))

        l_pos, l_quat = grasp_point("left")
        r_pos, r_quat = grasp_point("right")
        L = ArmTimeline(l_pos - ROOT_OFFSET, l_quat, dt, speed)
        R = ArmTimeline(r_pos - ROOT_OFFSET, r_quat, dt, speed)
        planner = Planner(L, R)
        settle = hold_k(1.0)
        R.hold(settle)
        L.hold(settle)
        start = os.environ.get("PBLOCK_DEMO_START", "")  # development only: "padlock" (cover shut on the button)
        if start == "padlock":
            import torch

            z = torch.zeros(1, 1, device=view.device)
            cover.write_joint_state_to_sim(z, z, joint_ids=[st.hinge])
            p = torch.tensor([[BX, BY, PANEL_Z + 0.0003, 0.0, 0.0, 0.0, 1.0]], device=view.device)
            p[:, :3] += view.scene.env_origins[0]
            cover.write_root_pose_to_sim(p)
            cover.write_root_velocity_to_sim(torch.zeros(1, 6, device=view.device))
        planner.at(settle, {"padlock": planner.plan_pick}.get(start, planner.plan_cover_pick))

        q_rel = np.array(hb.Q_REL)
        tcp_extra = 0.1347 - 0.134
        q_arm = {s: view.scene[f"{s}_robot"].data.joint_pos.torch[0, :6].cpu().numpy().astype(float) for s in ("left", "right")}
        worst = {"left": 0.0, "right": 0.0}

        def joint_action(side, pos_root, quat, grip):
            pos = pos_root + ROOT_OFFSET + origin
            tcp = pos + rotate(quat, (0.0, 0.0, -tcp_extra))
            rot = qmul(quat, q_rel)
            q, pe, _ = ik.solve(tcp, rot, q_arm[side], base_pos=station.BASES[side], iters=60)
            if pe > 0.004:
                # Stuck in a poor branch after a change of hand orientation: try again from the ready pose.
                q2, pe2, _ = ik.solve(tcp, rot, np.array(READY, dtype=float), base_pos=station.BASES[side], iters=200)
                if pe2 < pe - 0.002:
                    q, pe = q2, pe2
            q_arm[side] = q
            if pe > 0.005 and (pe > worst[side] + 0.002 or state_k[0] % 40 == 0):
                log(f"IK {side} k={state_k[0]} error {pe * 1000:.1f} mm at {np.round(pos - origin, 4)}", flush=True)
            worst[side] = max(worst[side], pe)
            return np.r_[q, grip]

        self._L, self._R, self._planner, self._joint_action = L, R, planner, joint_action
        self.worst = worst

    def next_action(self) -> np.ndarray:
        k = self.state_k[0]
        L, R = self._L, self._R
        self._planner.run_due(k)
        L.pad_to(len(R.steps))
        R.pad_to(len(L.steps))
        if k >= len(L.steps):
            L.hold(1)
            R.hold(1)
        (lp, lq, lg), (rp, rq, rg) = L.steps[k], R.steps[k]
        self.state_k[0] = k + 1
        return np.concatenate([self._joint_action("left", lp, lq, lg), self._joint_action("right", rp, rq, rg)])
