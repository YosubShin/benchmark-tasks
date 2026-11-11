"""The scripted demo, closed loop, two hands on the lockout.

Arm B (right) picks the lockout off its post by the sides of its body (fingers down) and holds it in front of the
blue breaker, lined up from its measured pose. Arm A (left), fingers closed and pointing forward, presses the black
button home and keeps pressing: the pins swing outward. Both arms then move in together (arm A keeps the button
pressed) until the nose sits on the breaker face and the pins' feet are through the holes. Arm A backs off: the
return spring brings the button out and the pins swing in behind the holes' edges. Arm B lets go. Arm A takes the
padlock off its column, rolls it 90 degrees so the shackle's legs point along the breaker row, and pushes the open
short leg sideways through the stem's hole; arm B pushes the shackle's bend shut. Both let go.

Grasp-point targets become yam_arena's 14-D joint action through yam_arena's own IK (``yam_arena.ik``) every control
step. It reads the simulator's state (the lockout's and padlock's poses), like any scripted demo with privileged
state; ``--policy pinout-lockout-demo`` (pinout_sim.policy) plays it through Inspect Robots. Import after Isaac Sim
has started. With PINOUT_LOCKOUT_CONTROL=1 it stops after setting the lockout and forces the after-release check.
"""

from __future__ import annotations

import math
import os

import numpy as np

from yam_arena import ik, station

CONTROL = os.environ.get("PINOUT_LOCKOUT_CONTROL") == "1"


class Demo:
    """``next_action()`` gives the 14-D joint action for the next control step."""

    def __init__(self, STATION, control_hz: float = 20.0, speed: float = 0.10, log=print):
        from pinout_sim import station as hb
        from pinout_sim.kit import (CLOSED, DOWN, FORWARD, GRIPPER_OPEN_M, OPEN, ROOT_OFFSET, ArmTimeline,
                                     GraspModel, q_axis, qmul, rotate)
        from pinout_sim.kit import grasp_point as grasp_point_world
        from pinout_sim.task import PADLOCK as PV
        from pinout_sim.usd import breaker_props as P
        from pinout_sim.usd import padlock_props as PP

        view = STATION.env
        task = STATION.task
        st = task.state(view)
        state_k = [0]
        self.state_k = state_k
        origin = view.scene.env_origins[0].cpu().numpy()
        robot = view.scene["robot"]
        dt = 1.0 / control_hz
        hold_k = lambda s: max(1, round(s / dt))  # noqa: E731

        def grasp_point(side):
            p, q = grasp_point_world(robot, side)
            return p - origin, q

        for side, name in (("left", "left_robot"), ("right", "right_robot")):
            qj = view.scene[name].data.joint_pos.torch[0, :6].cpu().numpy()
            tcp, _ = ik.tcp_pose(qj, station.BASES[side])
            log(f"CHECK {side}: their TCP {np.round(tcp, 4)} our grasp point {np.round(grasp_point(side)[0] + origin, 4)}",
                flush=True)

        IDQ = np.array([0.0, 0.0, 0.0, 1.0])
        BODY_GRIP_X = -0.026
        """Arm B's grasp point on the body (lockout frame): between the flange and the nose."""
        TIP_AHEAD = 0.0108
        """Closed fingertips' ends ahead of the grasp point."""
        PRESS_OVER = 0.004
        PRE_X = 0.040
        """Lockout held this far out from its seat before the press."""
        ROLLED = q_axis((1, 0, 0), math.pi / 2)
        YAW_B, PITCH_B = math.radians(60), math.radians(20)
        QB = qmul(q_axis((-math.sin(YAW_B), math.cos(YAW_B), 0), PITCH_B), qmul(q_axis((0, 0, 1), YAW_B), FORWARD))
        """Arm B pressing the bend: fingers closed, pointing 60 degrees inboard (+y) and 20 degrees down."""
        B_FINGER_DIR = np.array([math.cos(YAW_B) * math.cos(PITCH_B), math.sin(YAW_B) * math.cos(PITCH_B),
                                 -math.sin(PITCH_B)])
        FINGER_END = 0.012
        L_PARK, R_PARK = np.array([0.17, 0.20, 0.22]), np.array([0.18, -0.26, 0.24])
        seat = np.array(P.lockout_seat_world(st.layout.named_slot))
        lo, lock = st.lo, st.lock

        def width_cmd(width):
            return float(np.clip(width / (2 * GRIPPER_OPEN_M), 0.0, 1.0))

        def note(msg):
            log(msg, flush=True)

        class Planner:
            def __init__(self, L, R):
                self.L, self.R, self.events = L, R, {}

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
                            note(f"SETTLE timeout ({side}): lag {lag * 1000:.1f} mm")
                        fn()
                    else:
                        check.tries -= 1
                        tl.hold(hold_k(0.1))
                        self.at(len(tl.steps), check)

                check.tries = tries
                self.at(len(tl.steps), check)

            def report(self):
                return task.episode_report(view, 0)

            # state
            def lo_pose(self):
                d = lo.data
                return d.body_pos_w.torch[0, st.lo_body].cpu().numpy() - origin, d.body_quat_w.torch[0, st.lo_body].cpu().numpy()

            def btn_pose(self):
                d = lo.data
                return (d.body_pos_w.torch[0, st.lo_button].cpu().numpy() - origin,
                        d.body_quat_w.torch[0, st.lo_button].cpu().numpy())

            def hole(self):
                p, q = self.btn_pose()
                return p + rotate(q, P.hole_center(0.0))

            def lock_pose(self):
                d = lock.data
                return (d.body_pos_w.torch[0, st.lock_body].cpu().numpy() - origin,
                        d.body_quat_w.torch[0, st.lock_body].cpu().numpy())

            def travel(self):
                return float(lock.data.joint_pos.torch[0, st.shackle_joint])

            def tip(self):
                p, q = self.lock_pose()
                return p + rotate(q, PV.short_tip(self.travel()))

            def btn_q(self):
                return float(lo.data.joint_pos.torch[0, st.button_joint])

            # Arm B: the lockout, by its body
            def btn_frac(self):
                return self.btn_q() / P.GAP

            def pins_deg(self):
                return np.degrees(lo.data.joint_pos.torch[0, st.pin_joints].cpu().numpy())

            def plan(self):
                R = self.R
                start = np.array(P.lockout_start_world()[:3])
                nominal = GraspModel.nominal(DOWN, IDQ, (BODY_GRIP_X, 0.0, 0.0))
                self.body_grasp, _ = nominal.grasp_for(start, IDQ)
                g = self.body_grasp
                R.grip_to(OPEN, hold_k(0.2))
                R.move(g + np.array([0.0, 0.0, 0.07]), DOWN, min_steps=hold_k(3.0))
                R.move(g + np.array([0.0, 0.0, 0.025]), speed=0.06)
                R.hold(hold_k(0.4))
                self.when_settled(R, "right", self.body_approach)

            def body_approach(self):
                g = self.body_grasp
                err = grasp_point("right")[0] - (self.R.pos + ROOT_OFFSET)
                note(f"BODY approach: grasp-point error (mm) {np.round(err * 1000, 1)}")
                self.R.move(g - err, speed=0.02)
                self.R.hold(hold_k(0.4))
                self.R.grip_to(width_cmd(2 * P.BODY_R - 0.008), hold_k(1.0))
                self.R.move(g - err + np.array([0.0, 0.0, 0.03]), speed=0.03)
                self.when_settled(self.R, "right", self.carry_lockout)

            def carry_lockout(self):
                self.lo_model = GraspModel.measure(robot, "right", lo)
                note(f"LOCKOUT picked: grasp rel_pos (mm) {np.round(self.lo_model.rel_pos * 1000, 1)}")
                self.corr = np.zeros(3)
                pre = seat + np.array([-PRE_X, 0.0, 0.0])
                g, q = self.lo_model.grasp_for(pre + np.array([-0.03, 0.0, 0.03]), IDQ)
                self.R.move(g, q, speed=0.08)
                g, q = self.lo_model.grasp_for(pre, IDQ)
                self.R.move(g, q, speed=0.04)
                self.R.hold(hold_k(0.4))
                self.when_settled(self.R, "right", lambda: self.align_lockout(0))

            def r_to(self, lo_target, speed, min_steps=3):
                g, q = self.lo_model.grasp_for(lo_target - self.corr, IDQ)
                self.R.move(g, q, speed=speed, min_steps=min_steps)

            def align_lockout(self, attempt):
                pos, quat = self.lo_pose()
                pre = seat + np.array([-PRE_X, 0.0, 0.0])
                err = pos - pre
                tilt = math.degrees(2 * math.acos(min(1.0, abs(quat[3]))))
                note(f"LOCKOUT pre-seat attempt {attempt}: error (mm) {np.round(err * 1000, 2)}, tilt {tilt:.1f} deg")
                self.corr = self.corr + err
                if attempt < 3 and (np.abs(err).max() > 0.0008):
                    self.r_to(pre, 0.01)
                    self.R.hold(hold_k(0.4))
                    self.when_settled(self.R, "right", lambda: self.align_lockout(attempt + 1), tol=0.0015)
                    return
                self.lo_x = pre[0]
                self.sync()
                self.press_approach()

            # Arm A: press and hold the button
            def cap_outer(self):
                """World position of the released cap's outer face centre (the button may be pressed)."""
                pos, quat = self.lo_pose()
                return pos + rotate(quat, (P.cap_inner_x(0.0) - P.CAP_T, 0.0, 0.0))

            def press_approach(self):
                """Arm A first goes, fingers forward, to a waypoint well behind the lockout; there its IK is put on the
                branch that reaches the deepest press pose (the ready pose's branch does not), then it comes in."""
                L = self.L
                c = self.cap_outer()
                L.grip_to(CLOSED, hold_k(0.3))
                L.move(c + np.array([-TIP_AHEAD - 0.03, 0.0, 0.06]), FORWARD, min_steps=hold_k(3.0))
                L.hold(hold_k(0.3))
                self.sync()

                def prime():
                    deepest = np.array([seat[0] + P.cap_inner_x(P.GAP) - P.CAP_T - TIP_AHEAD + 0.004, seat[1], seat[2]])
                    self.prime_ik("left", deepest, FORWARD)
                    L.hold(hold_k(2.0))
                    self.sync()
                    self.at(len(L.steps), self.press_approach2)

                self.at(len(L.steps), prime)

            def press_approach2(self):
                L = self.L
                c = self.cap_outer()
                L.move(c + np.array([-TIP_AHEAD - 0.012, 0.0, 0.0]), speed=0.04)
                L.hold(hold_k(0.4))
                self.when_settled(L, "left", self.press_contact)

            def press_contact(self):
                c = self.cap_outer()
                err = grasp_point("left")[0] - (self.L.pos + ROOT_OFFSET)
                note(f"PRESS approach: grasp-point error (mm) {np.round(err * 1000, 1)}")
                self.l_corr = np.array([0.0, err[1], err[2]])
                self.depth = -0.004
                self.press_button(0)

            def l_target(self):
                """Arm A's grasp point for the current press depth, from the lockout's commanded pose (not its measured
                one: if arm B yields, arm A must not chase the lockout)."""
                c = np.array([self.lo_x + P.cap_inner_x(0.0) - P.CAP_T, seat[1], seat[2]])
                return c + np.array([-TIP_AHEAD + self.depth, 0.0, 0.0]) - self.l_corr

            def hold_lockout(self, gain=0.7):
                """Arm B holds the lockout at its commanded pose against arm A's push (its soft wrist yields)."""
                pos, _ = self.lo_pose()
                err = pos - np.array([self.lo_x, seat[1], seat[2]])
                self.corr = self.corr + gain * err
                self.r_to(np.array([self.lo_x, seat[1], seat[2]]), 0.02, min_steps=2)
                return err

            def press_button(self, k):
                f = self.btn_frac()
                if f > 0.93 or k > 40:
                    note(f"PRESSED after {k}: button {self.btn_q() * 1000:.1f} mm, pins {np.round(self.pins_deg(), 1)} deg, "
                         f"depth {self.depth * 1000:.1f} mm")
                    self.L.hold(hold_k(0.3))
                    self.sync()
                    self.at(len(self.L.steps), lambda: self.advance(0))
                    return
                err = self.hold_lockout()
                if k % 4 == 0:
                    note(f"PRESS {k}: button {f * 100:.0f}%, lockout off its hold (mm) {np.round(err * 1000, 1)}")
                self.depth = min(self.depth + 0.002, P.GAP + 0.012)
                self.L.move(self.l_target(), speed=0.02, min_steps=2)
                self.sync()
                self.L.hold(hold_k(0.1))
                self.R.hold(hold_k(0.1))
                self.at(len(self.L.steps), lambda: self.press_button(k + 1))

            def advance(self, k):
                """Both arms in together, arm A keeping the button home; y and z re-centred on the seat each step."""
                pos, quat = self.lo_pose()
                err = pos - seat
                f = self.btn_frac()
                if k % 5 == 0:
                    note(f"ADVANCE {k}: lockout-seat (mm) {np.round(err * 1000, 2)}, button {f * 100:.0f}%, "
                         f"pins {np.round(self.pins_deg(), 1)} deg")
                if err[0] > -0.0003 or k > 80:
                    note(f"SEATED after {k}: lockout-seat (mm) {np.round(err * 1000, 2)}, button {f * 100:.0f}%, "
                         f"milestone {self.report()['milestone']}")
                    self.L.hold(hold_k(0.4))
                    self.sync()
                    self.at(len(self.L.steps), lambda: self.let_button_go(0))
                    return
                if f < 0.9:
                    self.depth = min(self.depth + 0.0015, P.GAP + 0.014)
                if pos[0] > self.lo_x - 0.004:
                    self.lo_x = min(self.lo_x + 0.002, seat[0] + 0.002)
                self.hold_lockout(gain=0.5)
                self.L.move(self.l_target(), speed=0.02, min_steps=2)
                self.sync()
                self.L.hold(hold_k(0.1))
                self.R.hold(hold_k(0.1))
                self.at(len(self.L.steps), lambda: self.advance(k + 1))

            def let_button_go(self, k):
                f = self.btn_frac()
                if bool(st.is_set[0]) and (f < 0.03 or self.depth < -0.006) or k > 40:
                    note(f"BUTTON released after {k}: button {self.btn_q() * 1000:.1f} mm, pins "
                         f"{np.round(self.pins_deg(), 1)} deg, set {bool(st.is_set[0])}, milestone "
                         f"{self.report()['milestone']}")
                    L = self.L
                    # Up and slightly back: on the press branch the left hand cannot back straight out.
                    L.move(L.pos + ROOT_OFFSET + np.array([-0.02, 0.0, 0.08]), speed=0.03)
                    self.R.hold(hold_k(0.5))
                    # Open only a little past the body: wide open, the fingers (down at toggle height) sweep into the
                    # neighbouring breakers' toggles.
                    self.R.grip_to(width_cmd(2 * P.BODY_R + 0.008), hold_k(0.8))
                    # Straight back first (the opened fingers clear the flange and button): from this pose the
                    # right arm cannot rise straight up (68 mm IK error at +6 cm).
                    self.R.move(self.R.pos + ROOT_OFFSET + np.array([-0.07, 0.0, 0.0]), speed=0.03)
                    self.R.move(self.R.pos + ROOT_OFFSET + np.array([0.0, 0.0, 0.05]), speed=0.03)
                    self.sync()

                    def park():
                        # Back to the left arm's exact ready pose and its IK branch, where the padlock stage starts.
                        self.restore_ik("left")
                        L.move(l_pos, l_quat, speed=0.06, min_steps=hold_k(3.0))
                        L.hold(hold_k(1.0))
                        self.R.grip_to(OPEN, hold_k(0.3))
                        self.R.move(R_PARK, speed=0.08)
                        self.sync()
                        self.R.hold(hold_k(0.5))
                        self.L.hold(hold_k(0.5))
                        self.at(len(self.R.steps), self.after_set)

                    self.at(len(L.steps), park)
                    return
                self.depth = max(self.depth - 0.002, -0.008)
                # Arm B keeps the lockout pushed home while the button comes out: the feet can only swing in once
                # they are fully behind the face wall.
                self.lo_x = seat[0] + 0.0015
                self.r_to(np.array([self.lo_x, seat[1], seat[2]]), 0.02, min_steps=2)
                self.L.move(self.l_target(), speed=0.02, min_steps=2)
                self.sync()
                self.L.hold(hold_k(0.1))
                self.R.hold(hold_k(0.1))
                self.at(len(self.L.steps), lambda: self.let_button_go(k + 1))

            def after_set(self):
                r = self.report()
                note(f"LOCKOUT let go: set {bool(st.is_set[0])}, button {self.btn_q() * 1000:.1f} mm, "
                     f"pins {np.round(self.pins_deg(), 1)} deg, milestone {r['milestone']}")
                if CONTROL:
                    self.R.hold(hold_k(1.0))
                    self.sync()
                    self.at(len(self.R.steps), lambda: (note("CONTROL: after-release check forced, no padlock"),
                                                        task.force_function_test(view)))
                    return
                self.plan_padlock_pick()

            # Arm A: the padlock
            def plan_padlock_pick(self):
                self.sync()
                start = np.array(P.padlock_start_world()[:3])
                nominal = GraspModel.nominal(FORWARD, IDQ, PV.grasp_in_lock)
                self.lock_grasp, quat = nominal.grasp_for(start, IDQ)
                grasp = self.lock_grasp
                L = self.L
                L.grip_to(OPEN, hold_k(0.4))  # closed since the button press
                L.move((0.17, 0.20, 0.20), quat, min_steps=hold_k(2.0))
                L.move((0.17, grasp[1], 0.20), speed=0.08)
                L.move(grasp + np.array([-0.04, 0.0, 0.0]), speed=0.05)
                L.hold(hold_k(0.4))
                self.when_settled(L, "left", self.padlock_approach)

            def padlock_approach(self):
                grasp = self.lock_grasp
                err = grasp_point("left")[0] - (self.L.pos + ROOT_OFFSET)
                note(f"PADLOCK approach: grasp-point error (mm) {np.round(err * 1000, 1)}, lock off its start (mm) "
                     f"{np.round((self.lock_pose()[0] - np.array(P.padlock_start_world()[:3])) * 1000, 1)}")
                self.L.move(grasp - err, speed=0.03)
                self.L.hold(hold_k(0.4))
                self.L.grip_to(CLOSED, hold_k(1.0))
                self.L.move(grasp - err + np.array([0.0, 0.0, 0.03]), speed=0.03)
                self.when_settled(self.L, "left", self.padlock_picked)

            def padlock_picked(self):
                self.lock_model = GraspModel.measure(robot, "left", lock)
                note(f"LOCK picked: grasp rel_pos (mm) {np.round(self.lock_model.rel_pos * 1000, 1)}")
                mid = np.array([0.20, 0.03, 0.23])
                g, q = self.lock_model.grasp_for(mid, IDQ)
                self.L.move(g, q, speed=0.06)
                g, q = self.lock_model.grasp_for(mid, ROLLED)
                self.L.move(g, q, speed=0.04, min_steps=hold_k(2.5))
                self.L.hold(hold_k(0.4))
                self.correction = np.zeros(3)
                self.when_settled(self.L, "left", lambda: self.entry(0))

            def lock_target(self, tip_target):
                return tip_target - rotate(ROLLED, PV.short_tip(PP.OPEN_TRAVEL))

            def lock_to(self, lock_pos, speed, min_steps=3):
                g, q = self.lock_model.grasp_for(lock_pos - self.correction, ROLLED)
                self.L.move(g, q, speed=speed, min_steps=min_steps)

            def entry_tip(self, dz):
                return self.hole() + np.array([0.0, -0.0045, dz])

            def entry(self, attempt):
                self.lock_to(self.lock_target(self.entry_tip(-0.03)) + np.array([-0.03, 0.0, 0.0]), 0.06)
                self.lock_to(self.lock_target(self.entry_tip(-0.018)), 0.03)
                self.L.hold(hold_k(0.6))
                self.when_settled(self.L, "left", lambda: self.entry_correct(attempt))

            def entry_correct(self, attempt):
                self.lock_model = GraspModel.measure(robot, "left", lock)
                err = self.tip() - self.entry_tip(-0.018)
                note(f"ENTRY attempt {attempt}: tip error (mm) {np.round(err * 1000, 2)}")
                self.correction = self.correction + err
                if attempt < 5 and np.abs(err).max() > 0.0006:
                    self.lock_to(self.lock_target(self.entry_tip(-0.018)), 0.01)
                    self.L.hold(hold_k(0.6))
                    self.when_settled(self.L, "left", lambda: self.entry_correct(attempt + 1), tol=0.0015)
                    return
                self.slide_up(0)

            def slide_up(self, attempt):
                self.lock_to(self.lock_target(self.entry_tip(0.0)), 0.01)
                self.L.hold(hold_k(0.6))
                self.when_settled(self.L, "left", lambda: self.level_correct(attempt), tol=0.0015)

            def level_correct(self, attempt):
                err = self.tip() - self.entry_tip(0.0)
                note(f"LEVEL attempt {attempt}: tip error (mm) {np.round(err * 1000, 2)}")
                self.correction = self.correction + err
                if attempt < 5 and np.abs(err[[0, 2]]).max() > 0.0004:
                    self.slide_up(attempt + 1)
                    return
                self.push_y = self.entry_tip(0.0)[1]
                self.push_through(0)

            def push_through(self, k):
                """Push the short leg sideways through the hole in 1.5 mm steps, re-centring it on the hole in x and z
                from the measured tip every step (the soft wrist drifts about 2 mm along the stem as it pushes)."""
                h, tip = self.hole(), self.tip()
                err = tip - h
                if err[1] > 0.0045 or k > 40:
                    note(f"PUSH THROUGH after {k} steps: tip {np.round(err * 1000, 2)} mm from the hole centre")
                    self.L.hold(hold_k(0.5))
                    self.when_settled(self.L, "left", self.threaded, tries=5, tol=0.002)
                    return
                if k > 0:
                    self.correction = self.correction + 0.7 * np.array([err[0], 0.0, err[2]])
                self.push_y += 0.0015
                self.lock_to(self.lock_target(np.array([h[0], self.push_y, h[2]])), 0.01, min_steps=2)
                self.L.hold(hold_k(0.15))
                self.at(len(self.L.steps), lambda: self.push_through(k + 1))

            def threaded(self):
                r = self.report()
                note(f"THREADED: tip {np.round((self.tip() - self.hole()) * 1000, 2)} mm from the hole centre, "
                     f"milestone {r['milestone']}")
                pos, q = self.lock_pose()
                outer = pos + rotate(q, (0.0, 0.0, PV.bend_top(self.travel())))
                R = self.R
                self.sync()
                R.grip_to(CLOSED, hold_k(0.4))
                start = outer - np.array([0.0, 0.012, 0.0]) - FINGER_END * B_FINGER_DIR
                R.move(start - 0.06 * B_FINGER_DIR + np.array([0.0, 0.0, 0.03]), QB, min_steps=hold_k(2.5))
                R.move(start, speed=0.04)
                R.hold(hold_k(0.4))
                self.body0 = self.lock_pose()[0]
                self.when_settled(R, "right", lambda: self.press(0))

            def press(self, k):
                latched = bool(st.latched[0])
                if latched or k > 60:
                    note(f"PRESS done after {k}: travel {self.travel() * 1000:.1f} mm, latched {latched}, "
                         f"milestone {self.report()['milestone']}")
                    self.R.move(self.R.pos + ROOT_OFFSET + np.array([0.0, 0.002, 0.0]), speed=0.02)
                    self.R.hold(hold_k(0.3))
                    self.sync()
                    self.release()
                    return
                # Arm A holds the body against the push: it backs off by however far the body has moved along y.
                drift = float(self.lock_pose()[0][1] - self.body0[1])
                self.sync()
                self.L.move(self.L.pos + ROOT_OFFSET - np.array([0.0, drift, 0.0]), speed=0.02, min_steps=1)
                self.R.move(self.R.pos + ROOT_OFFSET + np.array([0.0, 0.003, 0.0]), speed=0.02, min_steps=2)
                self.sync()
                self.at(len(self.R.steps), lambda: self.press(k + 1))

            def release(self):
                R, L = self.R, self.L
                R.move(R.pos + ROOT_OFFSET + np.array([0.0, -0.04, 0.03]), speed=0.04)
                R.move(R_PARK, speed=0.08)
                L.grip_to(OPEN, hold_k(0.8))
                L.move(L.pos + ROOT_OFFSET + np.array([-0.06, 0.0, 0.0]), speed=0.04)
                L.move(L_PARK, speed=0.08)
                self.sync()
                L.hold(hold_k(4.0))

        l_pos, l_quat = grasp_point("left")
        r_pos, r_quat = grasp_point("right")
        L = ArmTimeline(l_pos - ROOT_OFFSET, l_quat, dt, speed)
        R = ArmTimeline(r_pos - ROOT_OFFSET, r_quat, dt, speed)
        planner = Planner(L, R)
        settle = hold_k(1.0)
        L.hold(settle)
        R.hold(settle)
        planner.at(settle, planner.plan)

        q_rel = np.array(hb.Q_REL)
        tcp_extra = 0.1347 - 0.134
        q_arm = {s: view.scene[f"{s}_robot"].data.joint_pos.torch[0, :6].cpu().numpy().astype(float)
                 for s in ("left", "right")}
        worst = {"left": 0.0, "right": 0.0}

        def joint_action(side, pos_root, quat, grip):
            pos = pos_root + ROOT_OFFSET + origin
            tcp = pos + rotate(quat, (0.0, 0.0, -tcp_extra))
            rot = qmul(quat, q_rel)
            q, pe, _ = ik.solve(tcp, rot, q_arm[side], base_pos=station.BASES[side], iters=60)
            q_arm[side] = q
            if pe > 0.005 and pe > worst[side]:
                log(f"IK {side} k={state_k[0]} error {pe * 1000:.1f} mm at {np.round(pos - origin, 4)}")
            worst[side] = max(worst[side], pe)
            return np.r_[q, grip]

        def prime_ik(side, pos_world, quat):
            """Put one arm's IK warm start on a branch that reaches ``pos_world`` (desk frame) with ``quat``: the
            warm-started 60-iteration solve otherwise stays on the ready pose's branch, which runs out of reach for
            the left hand pointing forward at the right-hand breakers (19 mm error seen). Called while the arm is
            still far from everything."""
            tcp = np.asarray(pos_world) + origin + rotate(quat, (0.0, 0.0, -tcp_extra))
            rot = qmul(quat, q_rel)
            rng = np.random.default_rng(0)
            best = None
            for i in range(16):
                q0 = q_arm[side] if i == 0 else rng.uniform(-1.5, 1.5, 6)
                q, pe, _ = ik.solve(tcp, rot, q0, base_pos=station.BASES[side], iters=300)
                key = (round(pe * 1000, 1), float(np.abs(q - q_arm[side]).sum()))
                if best is None or key < best[0]:
                    best = (key, q)
            log(f"IK primed {side}: {best[0][0]:.1f} mm at {np.round(pos_world, 3)}", flush=True)
            q_arm[side] = best[1]

        q_start = {k: v.copy() for k, v in q_arm.items()}

        def restore_ik(side):
            """Back to the warm start of the ready pose's branch (the padlock threading was built on it)."""
            q_arm[side] = q_start[side].copy()
            log(f"IK branch restored {side}", flush=True)

        self.prime_ik = prime_ik
        planner.prime_ik = prime_ik
        planner.restore_ik = restore_ik
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
