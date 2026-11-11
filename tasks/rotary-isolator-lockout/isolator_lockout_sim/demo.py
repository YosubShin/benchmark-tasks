"""The scripted demo, closed loop.

Arm B (right) points its fingers level at Q1's handle (ON, pointing up), squeezes the grip with the gripper's full
force 17 mm above the axis and turns it a quarter turn anticlockwise to OFF by rolling the wrist about the handle's
axis while the hand follows the grip's arc, leading the measured handle angle by a few degrees (pinch-and-twist). It
lets go, pinches the red slide's finger tab below the grip's tip across its thickness and pulls it straight down 32 mm,
out, then goes back to its start. Arm A (left) picks the open padlock off its post (level fingers closing on the
lying lock's broad faces near its bottom end), carries it in front of its place, in along +x under the grip (the
legs pass over the pulled-out stripe, the short leg's tip stops short of the tab), lowers it onto the hole's axis and
slides it along the grip so the short leg goes 14 mm through the tab's hole (the short leg's measured tip is corrected
onto each target). Arm B, fingers closed and pitched 35 deg down, comes up beside the shackle's bend on the hub side
and holds it, while arm A pushes the body onto the shackle until it latches. Both let go; the task then runs its
after-release test (off camera).

``control=True``: the same up to the pulled-out slide, no padlock; the test is forced 1 s after arm B lets go.

It reads the simulator's state (handle angle, slide, padlock pose), like any scripted demo with privileged state, and
sends yam_arena's 14-D joint action through yam_arena's IK (``yam_arena.ik``). Import after Isaac Sim has started.
"""

from __future__ import annotations

import math
import os

import numpy as np

from yam_arena import ik, station

STOP_AFTER = os.environ.get("ISO_DEMO_STOP", "")
START_AT = os.environ.get("ISO_DEMO_START", "")
"""Debugging: 'pick' puts Q1 at OFF with its slide out at the start and begins with the padlock."""
"""Debugging: 'turn' or 'slide' ends the demo (both arms park) after that stage."""


def _log(*args, **kwargs):
    kwargs["flush"] = True
    print("[isolator-demo]", *args, **kwargs)


class Demo:
    """``next_action()`` gives the 14-D joint action for the next control step."""

    def __init__(self, STATION, control_hz: float = 20.0, speed: float = 0.10, control: bool = False, log=_log):
        from isolator_lockout_sim import station as hb
        from isolator_lockout_sim.kit import CLOSED, DOWN, FORWARD, OPEN, ROOT_OFFSET, ArmTimeline, GraspModel, q_axis, qmul
        from isolator_lockout_sim.kit import grasp_point as grasp_point_world
        from isolator_lockout_sim.kit import rotate
        from isolator_lockout_sim.task import LOCK_QUAT, PV, TARGET_SLOT, lock_start_pose
        from isolator_lockout_sim.usd import isolator_props as props

        view = STATION.env
        task = STATION.task
        state_k = [0]
        self.state_k = state_k
        origin = view.scene.env_origins[0].cpu().numpy()
        robot = view.scene["robot"]

        def grasp_point(side):
            p, q = grasp_point_world(robot, side)
            return p - origin, q

        last_err = {}
        LQ = np.array(LOCK_QUAT)
        PITCH = math.radians(35)
        QL = qmul(q_axis((1, 0, 0), math.pi / 2), FORWARD)
        """Arm A (left) holds the padlock with level fingers pointing at the enclosure, closing vertically on the lying
        body's broad faces (gravity cannot twist it about the pads' axis, and the gripper stays clear of the
        enclosure)."""
        L_FINGER_DIR = np.array([1.0, 0.0, 0.0])
        QP = qmul(q_axis((-1, 0, 0), PITCH), qmul(q_axis((0, 0, 1), math.pi / 2), FORWARD))
        """Arm B (right) pushes the shackle's bend toward +y with its closed fingertips, fingers pointing along the grip
        (+y) and pitched 35 deg down, from in front of the hub."""
        P_FINGER_DIR = np.array([0.0, math.cos(PITCH), -math.sin(PITCH)])
        GRASP_IN_LOCK = (0.0, 0.0, -0.012)
        """In the body's bottom half (the end away from the shackle), on its centre line."""
        FINGER_END = 0.012
        Y0, H = props.SLOT_Y[TARGET_SLOT], props.AXIS_Z
        GRIP_X = props.FACE_X - 0.043
        """Grasp point 43 mm in front of the enclosure: the fingertips stop at the grip's back face, clear of the hub."""
        PINCH_R = 0.017
        """Well inside the finger tab at the grip's tip (36-42 mm out along the grip): the pads must clear it."""
        GRIP_CMD = 0.0
        """Squeeze the grip with the gripper's full 50 N: the wrist's roll then turns it through the pads' friction."""
        TURN_LEAD0, TURN_LEAD_MAX = math.radians(6.0), math.radians(40.0)
        TAB_X = props.FACE_X - (props.TAB_D[0] + props.TAB_D[1]) / 2
        TAB_Y = Y0 + props.TAB_YC
        TAB_Z = H - props.GRIP_HW - 0.009
        HOLE = np.array(props.hole_world(TARGET_SLOT))
        FRONT_DX = 0.06
        """The padlock comes in along +x (toward the enclosure) from this far in front of its place."""
        ABOVE_DZ = 0.007
        """... at this height above its place (its legs then clear the pulled-out slide's stripe), then drops."""
        PRESS_STEP, PRESS_MAX = 0.002, 45
        TIP_IN_HOLE_DY = 0.014
        """The short leg goes 14 mm along through the hole (its tip 5 mm past the tab), so the body has room to close
        onto it."""
        PRESS_DX = 0.0
        BODY_IN = 0.017
        """Arm A brings the body this far toward the hub while arm B pushes (its top face stops 1 mm short of the tab)."""
        """Arm B's grasp point starts this far on the hub side of the bend's outer edge."""
        R_WAIT = np.array([0.15, -0.24, 0.25])
        L_WAIT = np.array([0.15, 0.24, 0.25])

        iso, lock = view.scene["isolators"], view.scene["padlock"]
        body_id = lock.find_bodies("body")[0][0]
        shackle_joint = lock.find_joints("shackle_joint")[0][0]
        handle_joint = iso.find_joints(f"handle_{TARGET_SLOT}_joint")[0][0]
        slide_joint = iso.find_joints(f"slide_{TARGET_SLOT}_joint")[0][0]
        dt = 1.0 / control_hz
        hold_k = lambda s: max(1, round(s / dt))  # noqa: E731

        def hand_pose(q):
            """Grasp-point pose on the grip at PINCH_R, level fingers (FORWARD) rolled with the handle (q from OFF)."""
            pos = np.array([GRIP_X, Y0 + PINCH_R * math.cos(q), H + PINCH_R * math.sin(q)])
            return pos, qmul(q_axis((1.0, 0.0, 0.0), q - math.pi / 2), FORWARD)

        class Planner:
            def __init__(self, L, R):
                self.L, self.R, self.events = L, R, {}
                self.corr = np.zeros(3)

            def at(self, step, fn):
                self.events.setdefault(step, []).append(fn)

            def run_due(self, k):
                for fn in self.events.pop(k, []):
                    fn()

            def sync(self):
                n = max(len(self.L.steps), len(self.R.steps))
                self.L.pad_to(n)
                self.R.pad_to(n)
                return n

            def when_settled(self, tl, side, fn, tries=30, tol=0.003):
                def check():
                    lag = np.linalg.norm(grasp_point(side)[0] - (tl.pos + ROOT_OFFSET))
                    if lag < tol or check.tries <= 0:
                        if lag >= tol:
                            log(f"SETTLE timeout ({side}): lag {lag * 1000:.1f} mm")
                        fn()
                    else:
                        check.tries -= 1
                        tl.hold(hold_k(0.1))
                        self.at(len(tl.steps), check)
                check.tries = tries
                self.at(len(tl.steps), check)

            # state
            def handle_q(self):
                return float(iso.data.joint_pos.torch[0, handle_joint])

            def slide_q(self):
                return float(iso.data.joint_pos.torch[0, slide_joint])

            def body_pose(self):
                pos = lock.data.body_pos_w.torch[0, body_id].cpu().numpy() - origin
                return pos, lock.data.body_quat_w.torch[0, body_id].cpu().numpy()

            def travel(self):
                return float(lock.data.joint_pos.torch[0, shackle_joint])

            def tip(self):
                pos, quat = self.body_pose()
                return pos + rotate(quat, PV.short_tip(self.travel()))

            def lock_center_for_tip(self, tip_target):
                return np.asarray(tip_target) - rotate(LQ, PV.short_tip(self.travel()))

            def lock_to(self, center, speed):
                self.L.move(self.model.grasp_for(center, LQ)[0], speed=speed)

            # Arm B: turn the handle ---------------------------------------------------------------
            def plan_turn(self):
                R = self.R
                R.grip = OPEN
                p0, q0 = hand_pose(math.pi / 2)
                # Stay high over the padlock's stand on the way to the handle.
                R.move(np.array([0.17, -0.12, 0.30]), q0, min_steps=hold_k(2.0))
                R.move(p0 + np.array([-0.08, 0.0, 0.0]), q0, min_steps=hold_k(1.5))
                R.hold(hold_k(0.3))
                self.when_settled(R, "right", self.approach)

            def approach(self):
                R = self.R
                err = grasp_point("right")[0] - (R.pos + ROOT_OFFSET)
                self.corr = np.clip(np.array([0.0, -err[1], -err[2]]), -0.01, 0.01)
                log(f"APPROACH: grasp-point error (mm) {np.round(err * 1000, 1)}")
                p0, _ = hand_pose(math.pi / 2)
                R.move(p0 + np.array([-0.012, 0.0, 0.0]) + self.corr, speed=0.04)
                R.move(p0 + self.corr, speed=0.02)
                R.hold(hold_k(0.3))
                self.when_settled(R, "right", self.pinch, tol=0.002)

            def pinch(self):
                R = self.R
                err = grasp_point("right")[0] - (R.pos + ROOT_OFFSET)
                log(f"AT GRIP: grasp-point error (mm) {np.round(err * 1000, 1)}, handle "
                    f"{math.degrees(self.handle_q()):.1f} deg")
                R.grip_to(GRIP_CMD, hold_k(0.8))
                self.at(len(R.steps), lambda: self.turn(0, TURN_LEAD0, self.handle_q()))

            def turn(self, k, lead, last_q):
                """One step of the turn, closed loop: aim the hand ``lead`` past the measured handle angle along the
                grip's arc, rolling the wrist with it. The lead grows while the handle is slow."""
                q = self.handle_q()
                if k % 5 == 0:
                    log(f"TURN {k}: handle {math.degrees(q):.1f} deg from OFF, lead {math.degrees(lead):.0f} deg, slide "
                        f"{-self.slide_q() * 1000:.1f} mm out")
                if q <= math.radians(1.5) or k > 200:
                    self.R.hold(hold_k(0.4))
                    self.at(len(self.R.steps), self.turned)
                    return
                rate = last_q - q
                if rate < math.radians(1.0):
                    lead = min(TURN_LEAD_MAX, lead + math.radians(2.0))
                elif rate > math.radians(3.0):
                    lead = max(TURN_LEAD0, lead - math.radians(2.0))
                target = max(math.radians(-6.0), q - lead)
                pos, quat = hand_pose(target)
                self.R.move(pos + self.corr, quat, speed=0.05, min_steps=hold_k(0.15))
                self.at(len(self.R.steps), lambda: self.turn(k + 1, lead, q))

            def turned(self):
                log(f"TURNED: handle {math.degrees(self.handle_q()):.1f} deg from OFF")
                R = self.R
                R.grip_to(0.7, hold_k(0.5))
                pos, quat = hand_pose(0.0)
                R.move(pos + np.array([-0.07, 0.0, 0.0]), quat, speed=0.05)
                if STOP_AFTER == "turn":
                    self.finish()
                    return
                self.plan_slide()

            # Arm B: pull the padlock slide out --------------------------------------------------------
            def plan_slide(self):
                R = self.R
                g = np.array([TAB_X, TAB_Y, TAB_Z])
                R.move(g + np.array([-0.07, 0.0, 0.0]), FORWARD, min_steps=hold_k(1.5))
                R.grip_to(0.25, hold_k(0.3))
                R.hold(hold_k(0.3))
                self.when_settled(R, "right", lambda: self.slide_approach(g))

            def slide_approach(self, g):
                R = self.R
                err = grasp_point("right")[0] - (R.pos + ROOT_OFFSET)
                corr = np.clip(np.array([0.0, -err[1], -err[2]]), -0.01, 0.01)
                R.move(g + corr + np.array([-0.012, 0.0, 0.0]), speed=0.04)
                R.move(g + corr, speed=0.02)
                R.hold(hold_k(0.3))
                self.when_settled(R, "right", lambda: self.slide_grip(g), tol=0.002)

            def slide_grip(self, g):
                R = self.R
                err = grasp_point("right")[0] - (R.pos + ROOT_OFFSET)
                log(f"AT SLIDE: grasp-point error (mm) {np.round(err * 1000, 1)}")
                R.grip_to(CLOSED, hold_k(0.8))
                R.move(R.pos + ROOT_OFFSET + np.array([0.0, 0.0, -(props.SLIDE_TRAVEL + 0.004)]), speed=0.01)
                R.hold(hold_k(0.5))
                self.when_settled(R, "right", lambda: self.slide_done(0), tries=8)

            def slide_done(self, attempt):
                q = self.slide_q()
                log(f"SLIDE attempt {attempt}: {-q * 1000:.1f} mm out (full {props.SLIDE_TRAVEL * 1000:.0f}), handle "
                    f"{math.degrees(self.handle_q()):.1f} deg")
                R = self.R
                if q > -(props.SLIDE_TRAVEL - 0.0015) and attempt < 3:
                    R.move(R.pos + ROOT_OFFSET + np.array([0.0, 0.0, -(props.SLIDE_TRAVEL + q) - 0.003]), speed=0.01)
                    R.hold(hold_k(0.5))
                    self.when_settled(R, "right", lambda: self.slide_done(attempt + 1), tries=8)
                    return
                # Open only a little (fully open, the outer finger would reach the padlock's shackle on its post).
                R.grip_to(0.3, hold_k(0.6))
                R.move(R.pos + ROOT_OFFSET + np.array([-0.07, 0.0, 0.0]), speed=0.05)
                # Back to where arm B started (out of the padlock's way, and the IK branch its press move starts from).
                R.move(self.r_home[0], self.r_home[1], speed=0.08)
                self.at(len(R.steps) - 1, reseed_ready)
                self.sync()
                if control:
                    R.move(R_WAIT, speed=0.10)
                    self.sync()
                    self.at(len(R.steps) + hold_k(1.0), lambda: task.force_function_test(view))
                elif STOP_AFTER == "slide":
                    self.finish()
                else:
                    self.plan_pick()

            # Arm A: the padlock -----------------------------------------------------------------------
            def plan_pick(self):
                start = np.array(lock_start_pose()[:3])
                grasp, _ = GraspModel.nominal(QL, LQ, GRASP_IN_LOCK).grasp_for(start, LQ)
                L = self.L
                L.grip_to(OPEN, hold_k(0.2))
                # Back and up from the ready pose first (it is just beside the padlock's post), then in front of it.
                L.move(np.array([0.14, 0.25, 0.26]), L.quat, min_steps=hold_k(1.5))
                L.move(grasp - 0.07 * L_FINGER_DIR + np.array([0.0, 0.0, 0.02]), QL, min_steps=hold_k(2.5))
                L.move(grasp - 0.07 * L_FINGER_DIR, QL, min_steps=hold_k(0.8))
                L.move(grasp - 0.015 * L_FINGER_DIR, speed=0.04)
                L.hold(hold_k(0.4))
                self.when_settled(L, "left", self.pick_final)

            def pick_final(self):
                start = np.array(lock_start_pose()[:3])
                grasp, _ = GraspModel.nominal(QL, LQ, GRASP_IN_LOCK).grasp_for(start, LQ)
                err = grasp_point("left")[0] - (self.L.pos + ROOT_OFFSET)
                log(f"PICK: grasp-point error (mm) {np.round(err * 1000, 1)}, padlock at {np.round(self.body_pose()[0], 4)}")
                self.L.move(grasp - err, speed=0.02)
                self.L.hold(hold_k(0.4))
                self.L.grip_to(CLOSED, hold_k(1.0))
                self.L.move(grasp - err + np.array([0.0, 0.0, 0.04]), speed=0.03)
                self.when_settled(self.L, "left", self.carry)

            def carry(self):
                self.model = GraspModel.measure(robot, "left", lock)
                log("LOCK grasp rel_pos (mm)", np.round(self.model.rel_pos * 1000, 1), "padlock quat",
                    np.round(self.body_pose()[1], 3))
                # Final: the short leg's tip 3 mm short of the tab (toward the hub), on the hole's axis.
                self.tip_final = np.array([HOLE[0], TAB_Y - props.SLIDE_T / 2 - 0.003, HOLE[2]])
                self.tip_above = self.tip_final + np.array([0.0, 0.0, ABOVE_DZ])
                self.tip_front = self.tip_above - np.array([FRONT_DX, 0.0, 0.0])
                lift_z = self.body_pose()[0][2]
                c_front = self.lock_center_for_tip(self.tip_front)
                self.lock_to(np.array([c_front[0], c_front[1], max(lift_z, c_front[2])]), 0.05)
                self.lock_to(c_front, 0.04)
                self.L.hold(hold_k(0.6))
                self.when_settled(self.L, "left", lambda: self.correct("FRONT", self.tip_front, np.zeros(3), 0,
                                                                       self.go_above))

            def correct(self, name, target, correction, attempt, then, tol=0.0015, tries=4):
                err = self.tip() - target
                log(f"{name} attempt {attempt}: tip error (mm) {np.round(err * 1000, 1)}")
                correction = correction + err
                if attempt < tries and np.abs(err).max() > tol:
                    self.lock_to(self.lock_center_for_tip(target - correction), 0.01)
                    self.L.hold(hold_k(0.6))
                    self.when_settled(self.L, "left", lambda: self.correct(name, target, correction, attempt + 1, then,
                                                                           tol, tries))
                    return
                self.correction = correction
                then()

            def go_above(self):
                # In along +x under the grip: the legs pass over the pulled-out stripe, the short leg's tip stops short
                # of the tab.
                self.lock_to(self.lock_center_for_tip(self.tip_above - self.correction), 0.01)
                self.L.hold(hold_k(0.8))
                self.when_settled(self.L, "left", lambda: self.correct("ABOVE", self.tip_above, self.correction, 0,
                                                                       self.go_down))

            def go_down(self):
                self.lock_to(self.lock_center_for_tip(self.tip_final - self.correction), 0.006)
                self.L.hold(hold_k(0.8))
                self.when_settled(self.L, "left", lambda: self.correct("AT HOLE", self.tip_final, self.correction, 0,
                                                                       self.into_hole, tol=0.0008, tries=5))

            def into_hole(self):
                # Slide the padlock along the grip so the short leg's tip goes into the tab's hole: the hole's edges then
                # carry the leg while arm B pushes the bend.
                self.tip_final = self.tip_final + np.array([0.0, TIP_IN_HOLE_DY, 0.0])
                self.lock_to(self.lock_center_for_tip(self.tip_final - self.correction), 0.005)
                self.L.hold(hold_k(0.6))
                self.when_settled(self.L, "left", self.in_hole)

            def in_hole(self):
                log(f"IN HOLE: tip error (mm) {np.round((self.tip() - self.tip_final) * 1000, 1)}, slide "
                    f"{self.slide_q() * 1000:.1f} mm, handle {math.degrees(self.handle_q()):.1f} deg")
                self.sync()
                self.press_start()

            # Arm B: shut the shackle ---------------------------------------------------------------
            def bend_top(self):
                pos, quat = self.body_pose()
                return pos + rotate(quat, (0.0, 0.0, PV.bend_top(self.travel())))

            def press_start(self):
                """Arm B, fingers closed, pointing at the enclosure and 45 deg down, comes up beside the shackle's bend
                on the hub side (in front of the grip) and stops against it with the flat side of its fingers: an anvil.
                Arm A then pushes the padlock's body toward the hub, so the shackle slides shut into the body."""
                R = self.R
                b = self.bend_top()
                # 12 mm toward the robot from the bend's middle, on its fixed-leg shoulder: the fingers then stay in
                # front of the grip (pushing at the middle, their upper edge caught under the grip).
                start = b - FINGER_END * P_FINGER_DIR + np.array([-PRESS_DX, -0.006, 0.0])
                R.grip = CLOSED
                R.move(np.array([0.17, -0.14, 0.25]), QP, min_steps=hold_k(2.0))
                R.move(start - 0.05 * P_FINGER_DIR - np.array([0.0, 0.01, 0.0]), QP, min_steps=hold_k(2.0))
                R.move(start, speed=0.03)
                R.hold(hold_k(0.4))
                self.when_settled(R, "right", lambda: self.anvil(0))

            def anvil(self, k):
                """Arm B steps toward the bend until it touches (the bend starts to move)."""
                b = self.bend_top()
                if k == 0:
                    self.bend0 = b.copy()
                moved = b[1] - self.bend0[1]
                lag = (self.R.pos + ROOT_OFFSET)[1] - grasp_point("right")[0][1]
                if moved > 0.0008 or lag > 0.003 or k > 20:
                    log(f"ANVIL at step {k}: bend moved {moved * 1000:.1f} mm, arm B lag {lag * 1000:.1f} mm, travel "
                        f"{self.travel() * 1000:.1f} mm")
                    self.anvil_pos = grasp_point("right")[0].copy()
                    self.R.move(self.anvil_pos, speed=0.02)
                    self.sync()
                    self.body0 = self.body_pose()[0]
                    self.at(len(self.L.steps), lambda: self.press(0))
                    return
                self.R.move(self.R.pos + ROOT_OFFSET + np.array([0.0, 0.0015, 0.0]), speed=0.02)
                self.R.hold(hold_k(0.15))
                self.sync()
                self.at(len(self.R.steps), lambda: self.anvil(k + 1))

            def press(self, k):
                latched = bool(view._task_state.latched[0])
                if latched or k > PRESS_MAX:
                    log(f"PRESS done after {k}: travel {self.travel() * 1000:.1f} mm, latched {latched}, "
                        f"through {bool(view._task_state.debug['through'][0])}")
                    self.L.hold(hold_k(0.3))
                    self.R.move(self.R.pos + ROOT_OFFSET + np.array([0.0, -0.02, 0.0]) - 0.03 * P_FINGER_DIR, speed=0.03)
                    self.sync()
                    self.release()
                    return
                if k % 3 == 0:
                    log(f"PRESS {k}: travel {self.travel() * 1000:.1f} mm, bend {np.round(self.bend_top(), 4)}, arm B "
                        f"{np.round(grasp_point('right')[0], 4)}, body y {self.body_pose()[0][1]:.4f} (start "
                        f"{self.body0[1]:.4f}), tip error (mm) {np.round((self.tip() - self.tip_final) * 1000, 1)}")
                # Arm B is the anvil: it holds the bend where it touched it (integral action against its soft gains).
                bend = self.bend_top()
                gp = grasp_point("right")[0]
                tgt = self.R.pos + ROOT_OFFSET + np.array([0.0, 0.8 * (self.bend0[1] + 0.0005 - bend[1]),
                                                          0.8 * (self.anvil_pos[2] - gp[2])])
                tgt[2] = min(max(tgt[2], self.anvil_pos[2] - 0.01), self.anvil_pos[2] + 0.01)
                self.R.move(tgt, speed=0.02)
                self.R.hold(hold_k(0.2))
                # Arm A pushes the body onto the shackle (toward the hub), keeping it level and in line.
                body = self.body_pose()[0]
                target_y = self.body0[1] - min(BODY_IN, PRESS_STEP * (k + 1))
                corr = np.array([0.6 * (self.body0[0] - body[0]), 0.8 * (target_y - body[1]),
                                 0.6 * (self.body0[2] - body[2])])
                self.L.move(self.L.pos + ROOT_OFFSET + corr, speed=0.02)
                self.sync()
                self.at(len(self.R.steps), lambda: self.press(k + 1))

            def release(self):
                R, L = self.R, self.L
                R.move(R_WAIT, speed=0.10)
                L.grip_to(OPEN, hold_k(1.5))
                # Drop the lower finger clear of the padlock (it now hangs from the slide), then back straight off.
                L.move(L.pos + ROOT_OFFSET + np.array([0.0, 0.0, -0.006]), speed=0.01)
                L.move(L.pos + ROOT_OFFSET - 0.06 * L_FINGER_DIR, speed=0.015)
                L.move(L_WAIT, QL, speed=0.08)
                self.sync()
                R.hold(hold_k(4.0))
                L.hold(hold_k(4.0))

            def finish(self):
                R = self.R
                R.move(R_WAIT, speed=0.08)
                self.sync()
                R.hold(hold_k(2.0))
                self.at(len(R.steps), lambda: log(f"STOP ({STOP_AFTER}): {task.episode_report(view, 0)}"))

        l_pos, l_quat = grasp_point("left")
        r_pos, r_quat = grasp_point("right")
        L = ArmTimeline(l_pos - ROOT_OFFSET, l_quat, dt, speed)
        R = ArmTimeline(r_pos - ROOT_OFFSET, r_quat, dt, speed)
        planner = Planner(L, R)
        planner.r_home = (r_pos.copy(), r_quat.copy())

        def reseed_ready():
            """Arm B is back at its start pose: restart its IK from the READY joints, so its next moves take the same
            joint branch as from the start (from the twisted turn-and-pull branch the press move stalls)."""
            from yam_arena.constants import READY

            q_arm["right"] = np.array(READY, dtype=float)
        settle = hold_k(1.0)
        L.hold(settle)
        R.hold(settle)
        if START_AT == "pick":
            import torch

            jq = torch.tensor([[0.0, -props.SLIDE_TRAVEL]], device=iso.data.joint_pos.torch.device)
            ids = [handle_joint, slide_joint]
            iso.write_joint_state_to_sim(jq, torch.zeros_like(jq), joint_ids=ids)
            iso.set_joint_position_target(jq, joint_ids=ids)
            planner.at(settle, planner.plan_pick)
        else:
            planner.at(settle, planner.plan_turn)
        log(f"control {control}, stop after {STOP_AFTER or '-'}, grip command {GRIP_CMD:.3f}")

        q_rel = np.array(hb.Q_REL)
        tcp_extra = 0.1347 - 0.134
        q_arm = {s: view.scene[f"{s}_robot"].data.joint_pos.torch[0, :6].cpu().numpy().astype(float)
                 for s in ("left", "right")}
        worst = {"left": 0.0, "right": 0.0}
        last_err.update({"left": 0.0, "right": 0.0})

        def joint_action(side, pos_root, quat, grip):
            pos = pos_root + ROOT_OFFSET + origin
            tcp = pos + rotate(quat, (0.0, 0.0, -tcp_extra))
            rot = qmul(quat, q_rel)
            q, pe, _ = ik.solve(tcp, rot, q_arm[side], base_pos=station.BASES[side], iters=60)
            q_arm[side] = q
            if pe > 0.005 and pe > worst[side]:
                log(f"IK {side} k={state_k[0]} error {pe * 1000:.1f} mm at {np.round(pos - origin, 4)}")
            worst[side] = max(worst[side], pe)
            last_err[side] = pe
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
