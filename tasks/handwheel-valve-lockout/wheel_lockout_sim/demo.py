"""The scripted demo, closed loop.

1. Arm B (right), fingers down, grips V-21's handwheel across its rim and turns it clockwise (seen from above) by
   turning its wrist, each command aimed a little past the measured wheel angle; at the end of the wrist's travel it
   lets go, turns back with the fingers open, grips again and carries on, until the wheel stops at its seat.
2. Arm B, fingers down, picks the lockout cover up across its rim (its loose half nested inside), carries it level
   beside the handwheel and slides it on sideways from the right, open sector first, until the stem sits in the
   cover's centre, then lets go.
3. Arm A's closed fingertips, tilted in toward the stem, push the edge of the loose half's ear round the stem until it
   lies under the fixed half's ear.
4. Arm A takes the open padlock off its column (from the left, fingers pitched 40 deg down, closing across the body's
   thickness), carries it upright to the ears and slides it in from the left so the two stacked ears pass between the
   body and the raised short leg, with the leg's tip over the inner holes (the tip's measured position is corrected
   onto it).
5. Arm B, fingers closed and pitched steeply down, pushes the shackle's bend down until it latches, while arm A holds
   the body and keeps the tip on the hole. Both let go; the task then runs its after-release test (off camera).

``control=True``: steps 1-3 only, no padlock; the after-release test is forced 1 s after both arms are clear.

It reads the simulator's state (wheel angle, the cover's pose and swing, the padlock's pose), like any scripted demo
with privileged state, and sends yam_arena's 14-D joint action through yam_arena's IK (``yam_arena.ik``). Import after
Isaac Sim has started.
"""

from __future__ import annotations

import math

import numpy as np

from yam_arena import ik, station


def _log(*args, **kwargs):
    kwargs["flush"] = True
    print("[wheel-demo]", *args, **kwargs)


class Demo:
    """``next_action()`` gives the 14-D joint action for the next control step."""

    def __init__(self, STATION, control_hz: float = 20.0, speed: float = 0.10, control: bool = False, upto: int = 99,
                 start: int = 1, log=_log):
        from wheel_lockout_sim import station as hb
        from wheel_lockout_sim.kit import CLOSED, DOWN, FORWARD, OPEN, ROOT_OFFSET, ArmTimeline, GraspModel, q_axis, qmul
        from wheel_lockout_sim.kit import grasp_point as grasp_point_world
        from wheel_lockout_sim.kit import rotate
        from wheel_lockout_sim.task import (LOCK_QUAT, PV, TARGET_SLOT, WHEEL_SEAT, cover_fitted_pos,
                                            lock_start_pose)
        from wheel_lockout_sim.usd import cover_props as cov
        from wheel_lockout_sim.usd import valve_props as props

        view = STATION.env
        task = STATION.task
        state_k = [0]
        self.state_k = state_k
        origin = view.scene.env_origins[0].cpu().numpy()
        robot = view.scene["robot"]

        def grasp_point(side):
            p, q = grasp_point_world(robot, side)
            return p - origin, q

        LQ0 = np.array(LOCK_QUAT)
        IDQ = np.array([0.0, 0.0, 0.0, 1.0])
        R_WAIT = np.array([0.15, -0.20, 0.22])
        A_WAIT = np.array([0.22, 0.20, 0.20])
        VX, VY = props.VALVE_X, props.VALVE_Y[TARGET_SLOT]
        FIT = np.array(cover_fitted_pos())
        # 1. Turning.
        Y_HI, Y_LO = 150.0, -55.0
        """Arm B's wrist yaw range for one grip (joint 6 within its +-120 deg)."""
        LEAD0, LEAD_MAX = 10.0, 25.0
        SEATED_DEG = 1.5
        # 2. Cover.
        COVER_YAW = 115.0
        """Arm B's yaw on the cover: fingers on A's rim at phi = 25 and 205 deg (clear of both ears)."""
        SLIDE_FROM = math.radians(292.0)
        SLIDE_D = 0.090
        CARRY_DZ = 0.003
        # 3. Swing.
        PUSH_R = 0.072
        PUSH_DZ = (cov.B_EAR_Z[0] + cov.B_EAR_Z[1]) / 2
        """The fingertips push the edge of B's ear, which lies in the plane of B's back face."""
        PUSH_BEHIND = math.radians(25.0)
        """The closed fingers come down this far behind B's ear (well clear of it), then advance along the arc."""
        PUSH_LEAD = math.radians(4.0)
        PUSH_TILT = math.radians(20.0)
        PUSH_REACH = math.radians(4.0)
        """The fingers are aimed no closer than this behind the ear's centre line (so they bear on its edge)."""
        HOLD_PHI, HOLD_R = math.radians(300.0), 0.034
        """Arm B's closed fingertips press here on A's front face while B is swung."""
        # 4-5. Padlock.
        SIDE_A = qmul(q_axis((0, 0, 1), -math.pi / 2), FORWARD)
        QA = qmul(q_axis((1, 0, 0), math.radians(40)), SIDE_A)
        """Arm A on the padlock: fingers pointing right (-y) and 40 deg down, closing across the body's thickness."""
        A_FINGER_DIR = np.array([0.0, -math.cos(math.radians(40)), -math.sin(math.radians(40))])
        GRASP_IN_LOCK = (-0.011, 0.0, -0.008)
        """Arm A's grasp point in the padlock frame: the body's lower left part (under the long leg)."""
        YAW_B, PITCH_B = math.radians(60), math.radians(70)
        QB = qmul(q_axis((-math.sin(YAW_B), math.cos(YAW_B), 0), PITCH_B), qmul(q_axis((0, 0, 1), YAW_B), FORWARD))
        B_FINGER_DIR = np.array([math.cos(YAW_B) * math.cos(PITCH_B), math.sin(YAW_B) * math.cos(PITCH_B),
                                 -math.sin(PITCH_B)])
        TIP_IN_A = np.array([cov.HOLE_XS[0], 0.0, cov.STACK_Z[1] + 0.0035])
        """The short leg's tip, in A's frame: over the inner holes, 3.5 mm above the upper (A's) ear."""
        ENTRY_DY = 0.018
        """The padlock comes in from the left: the ears slide between the body and the raised short leg."""
        FINGER_END = 0.012
        PRESS_STEP, PRESS_MAX = 0.003, 30

        lock, pipe, cover = view.scene["padlock"], view.scene["pipe_run"], view.scene["cover"]
        body_id = lock.find_bodies("body")[0][0]
        shackle_joint = lock.find_joints("shackle_joint")[0][0]
        wheel_joint = pipe.find_joints(f"wheel_{TARGET_SLOT}_joint")[0][0]
        swing_joint = cover.find_joints("swing_joint")[0][0]
        a_id = cover.find_bodies("half_a")[0][0]
        dt = 1.0 / control_hz
        hold_k = lambda s: max(1, round(s / dt))  # noqa: E731

        Z_UP = 0.178
        """Highest grasp point for fingers-down poses near the valve (the station arm's reach there)."""

        def up(p, dz):
            q = np.array(p, dtype=float)
            q[2] = max(q[2], min(q[2] + dz, Z_UP))
            return q

        def down_yaw(yaw_deg):
            return qmul(q_axis((0, 0, 1), math.radians(yaw_deg)), DOWN)

        def rz(a):
            return q_axis((0, 0, 1), a)

        class Planner:
            def __init__(self, L, R):
                self.L, self.R, self.events = L, R, {}
                self.r_yaw = Y_HI

            def at(self, step, fn):
                self.events.setdefault(step, []).append(fn)

            def run_due(self, k):
                for fn in self.events.pop(k, []):
                    fn()

            def sync(self):
                n = max(len(self.L.steps), len(self.R.steps))
                self.L.pad_to(n)
                self.R.pad_to(n)

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

            def here(self, tl):
                return tl.pos + ROOT_OFFSET

            # state
            def wheel_q(self):
                return float(pipe.data.joint_pos.torch[0, wheel_joint])

            def from_seat_deg(self):
                return math.degrees(self.wheel_q() - WHEEL_SEAT)

            def swing_q(self):
                return float(cover.data.joint_pos.torch[0, swing_joint])

            def a_pose(self):
                return (cover.data.body_pos_w.torch[0, a_id].cpu().numpy() - origin,
                        cover.data.body_quat_w.torch[0, a_id].cpu().numpy())

            def a_yaw(self):
                _, q = self.a_pose()
                x = rotate(q, (1.0, 0.0, 0.0))
                return math.atan2(x[1], x[0])

            def body_pose(self):
                pos = lock.data.body_pos_w.torch[0, body_id].cpu().numpy() - origin
                return pos, lock.data.body_quat_w.torch[0, body_id].cpu().numpy()

            def travel(self):
                return float(lock.data.joint_pos.torch[0, shackle_joint])

            def tip(self):
                pos, quat = self.body_pose()
                return pos + rotate(quat, PV.short_tip(self.travel()))

            def bend_top(self):
                pos, quat = self.body_pose()
                return pos + rotate(quat, (0.0, 0.0, PV.bend_top(self.travel())))

            def rotate_yaw(self, tl, yaw_from, yaw_to, step_deg=40.0, steps_each=6):
                """Turn the wrist in place through intermediate yaws (slerp would take the short way round)."""
                n = max(1, math.ceil(abs(yaw_to - yaw_from) / step_deg))
                for k in range(1, n + 1):
                    tl.move(self.here(tl), down_yaw(yaw_from + (yaw_to - yaw_from) * k / n), min_steps=steps_each)

            # 1. Arm B: turn the handwheel --------------------------------------------------------------
            def plan_turn(self):
                R = self.R
                R.grip = OPEN
                g = np.array([VX, VY, props.WHEEL_Z])
                R.move(up(g, 0.07), down_yaw(90.0), min_steps=hold_k(2.0))
                self.rotate_yaw(R, 90.0, Y_HI)
                R.move(g + np.array([0.0, 0.0, 0.015]), speed=0.05)
                R.hold(hold_k(0.3))
                self.r_yaw = Y_HI
                self.when_settled(R, "right", lambda: self.wheel_grip(g))

            def wheel_grip(self, g):
                R = self.R
                err = grasp_point("right")[0] - self.here(R)
                R.move(g - np.array([err[0], err[1], 0.0]), speed=0.02)
                R.hold(hold_k(0.2))
                R.grip_to(CLOSED, hold_k(0.8))
                self.at(len(R.steps), lambda: self.turn_start())

            def turn_start(self):
                self.q_grip = self.wheel_q()
                self.yaw_grip = self.r_yaw
                log(f"GRIP at yaw {self.yaw_grip:.0f}: wheel {self.from_seat_deg():.1f} deg from its seat")
                self.turn(0, LEAD0, self.from_seat_deg() + 10.0)

            def turn(self, k, lead, last):
                left = self.from_seat_deg()
                if k % 10 == 0:
                    log(f"TURN {k}: {left:.1f} deg from seat, lead {lead:.0f}")
                if left < SEATED_DEG or k > 600:
                    log(f"SEATED: {left:.1f} deg from seat")
                    self.R.hold(hold_k(0.4))
                    self.at(len(self.R.steps), self.turn_done)
                    return
                rate = last - left
                if rate < 0.8:
                    lead = min(LEAD_MAX, lead + 2.0)
                elif rate > 3.0:
                    lead = max(LEAD0, lead - 2.0)
                turned = math.degrees(self.q_grip - self.wheel_q())
                yaw = self.yaw_grip - turned - lead
                yaw = max(yaw, self.r_yaw - 8.0)
                if yaw < Y_LO:
                    self.regrip()
                    return
                self.r_yaw = yaw
                self.R.move(self.here(self.R), down_yaw(yaw), min_steps=hold_k(0.1))
                self.at(len(self.R.steps), lambda: self.turn(k + 1, lead, left))

            def regrip(self):
                R = self.R
                log(f"REGRIP at yaw {self.r_yaw:.0f}: {self.from_seat_deg():.1f} deg from seat")
                R.grip_to(OPEN, hold_k(0.6))
                self.rotate_yaw(R, self.r_yaw, Y_HI)
                self.r_yaw = Y_HI
                R.hold(hold_k(0.2))
                R.grip_to(CLOSED, hold_k(0.8))
                self.at(len(R.steps), self.turn_start)

            def turn_done(self):
                R = self.R
                R.grip_to(OPEN, hold_k(0.6))
                R.move(up(self.here(R), 0.06), speed=0.05)
                self.rotate_yaw(R, self.r_yaw, 0.0)
                if upto >= 2:
                    self.plan_cover_pick()

            # 2. Arm B: the cover onto the wheel ---------------------------------------------------------
            def plan_cover_pick(self):
                R = self.R
                a_pos, _ = self.a_pose()
                g = a_pos.copy()
                R.grip_to(OPEN, hold_k(0.2))
                R.move(g + np.array([0.0, 0.0, 0.08]), down_yaw(COVER_YAW), min_steps=hold_k(2.5))
                R.move(g + np.array([0.0, 0.0, 0.02]), speed=0.05)
                R.hold(hold_k(0.3))
                self.when_settled(R, "right", lambda: self.cover_grip(g))

            def cover_grip(self, g):
                R = self.R
                err = grasp_point("right")[0] - self.here(R)
                R.move(g + np.array([0.0, 0.0, 0.006]) - np.array([err[0], err[1], 0.0]), speed=0.02)
                R.hold(hold_k(0.3))
                R.grip_to(CLOSED, hold_k(1.0))
                R.move(self.here(R) + np.array([0.0, 0.0, 0.03]), speed=0.03)
                self.when_settled(R, "right", self.cover_carry)

            def cover_carry(self):
                self.cmodel = GraspModel.measure(robot, "right", cover)
                a_pos, _ = self.a_pose()
                log(f"COVER grasp rel_pos (mm) {np.round(self.cmodel.rel_pos * 1000, 1)}, A at {np.round(a_pos, 4)}, "
                    f"swing {math.degrees(self.swing_q()):.1f} deg")
                self.pre = FIT + np.array([SLIDE_D * math.cos(SLIDE_FROM), SLIDE_D * math.sin(SLIDE_FROM), CARRY_DZ])
                R = self.R
                R.move(self.cmodel.grasp_for(np.array([a_pos[0], a_pos[1], self.pre[2] + 0.01]), IDQ)[0], speed=0.04)
                R.move(self.cmodel.grasp_for(self.pre, IDQ)[0], speed=0.04)
                R.hold(hold_k(0.5))
                self.when_settled(R, "right", lambda: self.cover_pre(np.zeros(3), 0), tries=10)

            def cover_pre(self, corr, attempt):
                a_pos, _ = self.a_pose()
                err = a_pos - self.pre
                log(f"COVER pre {attempt}: A error (mm) {np.round(err * 1000, 1)}, yaw "
                    f"{math.degrees(self.a_yaw()):.1f}")
                corr = corr + err
                if attempt < 4 and np.abs(err).max() > 0.002:
                    self.R.move(self.cmodel.grasp_for(self.pre - corr, IDQ)[0], speed=0.02)
                    self.R.hold(hold_k(0.5))
                    self.when_settled(self.R, "right", lambda: self.cover_pre(corr, attempt + 1), tries=10)
                    return
                self.slide(corr, 0)

            def slide(self, corr, attempt):
                """Sideways onto the wheel in two legs, correcting x and y at the end of each."""
                a_pos, _ = self.a_pose()
                goal = FIT + np.array([0.0, 0.0, CARRY_DZ])
                if attempt > 0:
                    err = a_pos - goal
                    log(f"SLIDE {attempt}: A error (mm) {np.round(err * 1000, 1)}, wheel nut under A "
                        f"{np.round((a_pos[:2] - FIT[:2]) * 1000, 1)}")
                    corr = corr + np.array([err[0], err[1], 0.6 * err[2]])
                    if attempt >= 4 or max(abs(err[0]), abs(err[1])) < 0.0015:
                        self.cover_down(corr)
                        return
                self.R.move(self.cmodel.grasp_for(goal - corr, IDQ)[0], speed=0.015 if attempt == 0 else 0.008)
                self.R.hold(hold_k(0.5))
                self.when_settled(self.R, "right", lambda: self.slide(corr, attempt + 1), tries=10)

            def cover_down(self, corr):
                R = self.R
                R.move(self.cmodel.grasp_for(FIT - corr + np.array([0.0, 0.0, -0.002]), IDQ)[0], speed=0.01)
                R.hold(hold_k(0.5))
                R.grip_to(OPEN, hold_k(0.8))
                # Out between the open fingers' line and the rim (along the closing axis's normal, away from the
                # robot's left arm), then up: lifting straight up or moving across catches the rim.
                out = math.radians(COVER_YAW + 180.0)
                R.move(self.here(R) + 0.07 * np.array([math.cos(out), math.sin(out), 0.0]), speed=0.03)
                R.move(up(self.here(R), 0.04), speed=0.04)
                self.at(len(R.steps), self.cover_placed)

            def cover_placed(self):
                a_pos, _ = self.a_pose()
                log(f"COVER placed: A at {np.round(a_pos, 4)} (fit {np.round(FIT, 4)}), off axis "
                    f"{np.linalg.norm(a_pos[:2] - FIT[:2]) * 1000:.1f} mm, yaw {math.degrees(self.a_yaw()):.1f}, swing "
                    f"{math.degrees(self.swing_q()):.1f}, wheel {self.from_seat_deg():.1f} deg from seat")
                self.sync()
                if upto >= 3:
                    self.plan_swing()

            # 3. Arm A swings B shut -------------------------------------------------------------------
            def a_point(self, phi, r, dz):
                a_pos, a_quat = self.a_pose()
                return a_pos + rotate(a_quat, (r * math.cos(phi), r * math.sin(phi), dz))

            def tab_phi(self):
                """B's ear's angle about A's axis, in A's frame (180 deg + swing)."""
                return math.pi + self.swing_q()

            def pinch_quat(self, phi):
                """Closed fingers across the tangent at phi (yaw = phi - 180 deg keeps the left wrist's joint 6 in its
                range from phi = 20 to 180 deg), tilted PUSH_TILT so they point down and in toward the axis: the hand
                above them then leans out, clear of the cover's top."""
                w = self.a_yaw() + phi
                t = (-math.sin(w), math.cos(w), 0.0)
                return qmul(q_axis(t, PUSH_TILT), down_yaw(math.degrees(w) - 180.0))

            def plan_swing(self):
                R = self.R
                R.move(R_WAIT, down_yaw(0.0), speed=0.10)
                self.sync()
                self.plan_push()

            def plan_push(self):
                """Closed fingertips come down well behind B's ear, then advance along the arc and push its edge round
                the stem until B stops against A (a pinch on the ear would also catch the end of A's rim beside it)."""
                L = self.L
                L.grip = CLOSED
                phi = self.tab_phi() - PUSH_BEHIND
                s = self.a_point(phi, PUSH_R, PUSH_DZ)
                L.move(up(s, 0.04), self.pinch_quat(phi), min_steps=hold_k(1.5))
                L.move(s, speed=0.02)
                self.when_settled(L, "left", lambda: self.push(0, phi))

            def push(self, k, phi_cmd):
                sw = math.degrees(self.swing_q())
                if k % 5 == 0:
                    log(f"PUSH {k}: swing {sw:.1f} deg, A yaw {math.degrees(self.a_yaw()):.1f}")
                if sw > -0.8 or k > 150:
                    log(f"SWUNG: swing {sw:.1f} deg after {k} pushes, A yaw {math.degrees(self.a_yaw()):.1f}")
                    self.swung()
                    return
                # Advance a few degrees a step toward the ear; the closed fingers meet its edge and push it round.
                phi = min(phi_cmd + PUSH_LEAD, self.tab_phi() - PUSH_REACH, math.pi - PUSH_REACH + math.radians(1.5))
                phi = max(phi, phi_cmd - math.radians(1.0))
                self.L.move(self.a_point(phi, PUSH_R, PUSH_DZ), self.pinch_quat(phi), speed=0.02, min_steps=hold_k(0.1))
                self.at(len(self.L.steps), lambda: self.push(k + 1, phi))

            def swung(self):
                L, R = self.L, self.R
                L.move(up(self.here(L), 0.04), speed=0.04)
                L.move(A_WAIT, speed=0.08)
                self.sync()
                self.at(len(L.steps), self.report_fit)

            def report_fit(self):
                a_pos, _ = self.a_pose()
                st = view._task_state
                log(f"FITTED {bool(st.fitted_ever[0])}: A at {np.round(a_pos, 4)}, off axis "
                    f"{np.linalg.norm(a_pos[:2] - FIT[:2]) * 1000:.1f} mm, swing {math.degrees(self.swing_q()):.1f}, "
                    f"yaw {math.degrees(self.a_yaw()):.1f}, wheel {self.from_seat_deg():.1f} deg from seat")
                if control:
                    self.at(len(self.L.steps) + hold_k(1.0), lambda: task.force_function_test(view))
                elif upto >= 4:
                    self.plan_lock_pick()

            # 4. Arm A: the padlock ----------------------------------------------------------------------
            def lq(self):
                return qmul(rz(self.a_yaw()), LQ0)

            def a_vec(self, v):
                return rotate(rz(self.a_yaw()), np.asarray(v, dtype=float))

            def lock_to_tip(self, tip_target, speed):
                c = np.asarray(tip_target) - rotate(self.lq_fixed, PV.short_tip(self.travel()))
                self.L.move(self.lmodel.grasp_for(c, self.lq_fixed)[0], speed=speed)

            def plan_lock_pick(self):
                start = np.array(lock_start_pose()[:3])
                grasp, _ = GraspModel.nominal(QA, LQ0, GRASP_IN_LOCK).grasp_for(start, LQ0)
                L = self.L
                L.grip_to(OPEN, hold_k(0.2))
                L.move(grasp - 0.07 * A_FINGER_DIR, QA, min_steps=hold_k(3.0))
                L.hold(hold_k(0.4))
                L.move(grasp - 0.015 * A_FINGER_DIR, speed=0.04)
                L.hold(hold_k(0.4))
                self.when_settled(L, "left", lambda: self.lock_grip(grasp))

            def lock_grip(self, grasp):
                L = self.L
                err = grasp_point("left")[0] - self.here(L)
                L.move(grasp - err, speed=0.02)
                L.hold(hold_k(0.4))
                L.grip_to(CLOSED, hold_k(1.0))
                L.move(grasp - err + np.array([0.0, 0.0, 0.04]), speed=0.03)
                self.when_settled(L, "left", self.lock_carry)

            def lock_carry(self):
                self.lmodel = GraspModel.measure(robot, "left", lock)
                self.lq_fixed = self.lq()
                a_pos, a_quat = self.a_pose()
                self.tip_target = a_pos + rotate(a_quat, TIP_IN_A)
                self.entry = self.tip_target + self.a_vec((0.0, ENTRY_DY, 0.0))
                log(f"TIP target {np.round(self.tip_target, 4)} (A at {np.round(a_pos, 4)}, yaw "
                    f"{math.degrees(self.a_yaw()):.1f})")
                log("LOCK grasp rel_pos (mm)", np.round(self.lmodel.rel_pos * 1000, 1))
                self.lock_to_tip(self.entry + np.array([-0.02, 0.03, 0.03]), 0.06)
                self.lock_to_tip(self.entry, 0.03)
                self.L.hold(hold_k(0.6))
                self.when_settled(self.L, "left", lambda: self.entry_correct(np.zeros(3), 0))

            def entry_correct(self, corr, attempt):
                err = self.tip() - self.entry
                log(f"ENTRY attempt {attempt}: tip error (mm) {np.round(err * 1000, 1)}")
                corr = corr + err
                if attempt < 4 and np.abs(err).max() > 0.001:
                    self.lock_to_tip(self.entry - corr, 0.01)
                    self.L.hold(hold_k(0.6))
                    self.when_settled(self.L, "left", lambda: self.entry_correct(corr, attempt + 1))
                    return
                self.corr = corr
                self.lock_to_tip(self.tip_target - corr, 0.008)
                self.L.hold(hold_k(0.8))
                self.when_settled(self.L, "left", lambda: self.on_correct(0))

            def on_correct(self, attempt):
                a_pos, a_quat = self.a_pose()
                self.tip_target = a_pos + rotate(a_quat, TIP_IN_A)
                err = self.tip() - self.tip_target
                log(f"OVER hole attempt {attempt}: tip error (mm) {np.round(err * 1000, 1)}")
                self.corr = self.corr + np.array([err[0], err[1], 0.0])
                if attempt < 4 and max(abs(err[0]), abs(err[1])) > 0.0008:
                    self.lock_to_tip(self.tip_target - self.corr, 0.006)
                    self.L.hold(hold_k(0.6))
                    self.when_settled(self.L, "left", lambda: self.on_correct(attempt + 1))
                    return
                self.sync()
                if upto >= 5:
                    self.press_start()

            # 5. Arm B: shut the shackle -----------------------------------------------------------------
            def press_start(self):
                """Arm B's closed fingertips, pitched steeply down from the front right, rest on the shackle's bend
                (clear of arm A's hand, which holds the body from the left)."""
                R = self.R
                b = self.bend_top() + np.array([0.0, 0.004, 0.0])
                start = b - FINGER_END * B_FINGER_DIR + np.array([0.0, 0.0, 0.004])
                R.grip = CLOSED
                R.move(start - 0.05 * B_FINGER_DIR + np.array([0.0, 0.0, 0.02]), QB, min_steps=hold_k(3.0))
                R.move(start, speed=0.03)
                R.hold(hold_k(0.4))
                self.body0 = self.body_pose()[0]
                self.when_settled(R, "right", lambda: self.press(0))

            def press(self, k):
                latched = bool(view._task_state.latched[0])
                if latched or k > PRESS_MAX:
                    log(f"PRESS done after {k}: travel {self.travel() * 1000:.1f} mm, latched {latched}, through "
                        f"{bool(view._task_state.latched_through[0])}, swing {math.degrees(self.swing_q()):.1f}")
                    self.R.hold(hold_k(0.3))
                    self.R.move(self.here(self.R) - 0.03 * B_FINGER_DIR, speed=0.03)
                    self.sync()
                    self.release()
                    return
                if k % 3 == 0:
                    log(f"PRESS {k}: travel {self.travel() * 1000:.1f} mm, bend top {np.round(self.bend_top(), 4)}, "
                        f"tip error (mm) {np.round((self.tip() - self.tip_target) * 1000, 1)}")
                # Arm A holds the body against the push and keeps the short leg's tip over the hole.
                body = self.body_pose()[0]
                err = self.tip() - self.tip_target
                corr = np.array([-0.5 * err[0], -0.5 * err[1], 0.4 * (self.body0[2] - body[2])])
                self.L.pad_to(len(self.R.steps))
                if np.abs(corr).max() > 0.0003:
                    self.L.move(self.here(self.L) + corr, speed=0.02)
                self.R.move(self.here(self.R) + np.array([0.0, 0.0, -PRESS_STEP]), speed=0.02)
                self.R.hold(hold_k(0.2))
                self.sync()
                self.at(len(self.R.steps), lambda: self.press(k + 1))

            def release(self):
                R, L = self.R, self.L
                R.move(R_WAIT, down_yaw(0.0), speed=0.10)
                L.grip_to(0.6, hold_k(1.0))
                L.move(self.here(L) - 0.05 * A_FINGER_DIR, speed=0.03)
                L.move(A_WAIT, speed=0.08)
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
        if start >= 2:
            # Debug starts: V-21 already closed (written to the simulator); 3: the cover already on the wheel, open;
            # 4: on the wheel and swung shut.
            import torch
            qc = torch.tensor([[WHEEL_SEAT]], device=pipe.device)
            pipe.write_joint_state_to_sim(qc, torch.zeros_like(qc), joint_ids=[wheel_joint])
            pipe.set_joint_position_target(qc, joint_ids=[wheel_joint])
            view._task_state.wheel_target[:, TARGET_SLOT] = WHEEL_SEAT
            if start >= 3:
                env_o = view.scene.env_origins[0:1]
                pose = torch.tensor([[*FIT.tolist(), 0.0, 0.0, 0.0, 1.0]], device=pipe.device, dtype=torch.float32)
                pose[:, :3] += env_o
                cover.write_root_pose_to_sim(pose)
                cover.write_root_velocity_to_sim(torch.zeros(1, 6, device=pipe.device))
                q0 = torch.full((1, 1), 0.0 if start >= 4 else math.radians(cov.Q_OPEN), device=pipe.device)
                cover.write_joint_state_to_sim(q0, torch.zeros_like(q0), joint_ids=[swing_joint])
                cover.set_joint_position_target(q0, joint_ids=[swing_joint])
                view._task_state.swing_target[:] = float(q0[0, 0])
            planner.at(settle, {2: planner.plan_cover_pick, 3: planner.plan_swing}.get(start, planner.plan_lock_pick))
        elif upto >= 1:
            planner.at(settle, planner.plan_turn)
        self.planner = planner

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
            if pe > 0.003:
                # Stuck in a poor branch (large reorientation from the ready pose): retry from an elbow-up,
                # fingers-down seed aimed at the target, and keep whichever solution is closer.
                b = np.asarray(station.BASES[side])
                j1 = math.atan2(tcp[1] - b[1], tcp[0] - b[0])
                seed = np.array([j1, 1.2, 1.2, -0.9, 0.0, 0.0])
                q2, pe2, _ = ik.solve(tcp, rot, seed, base_pos=station.BASES[side], iters=200)
                if pe2 < pe - 0.001:
                    q, pe = q2, pe2
            q_arm[side] = q
            if pe > 0.005 and pe > worst[side] + 0.004:
                log(f"IK {side} k={state_k[0]} error {pe * 1000:.1f} mm at {np.round(pos - origin, 4)}")
            worst[side] = max(worst[side], pe)
            return np.r_[q, grip]

        def hand_yaw(side):
            _, q = grasp_point(side)
            y = rotate(q, (0.0, 1.0, 0.0))
            return math.degrees(math.atan2(-y[0], y[1]))

        def debug(k):
            a_pos, _ = planner.a_pose()
            jr = np.degrees(view.scene["right_robot"].data.joint_pos.torch[0, :6].cpu().numpy())
            log(f"ARMS k={k}: right joints {np.round(jr, 1)}, right hand yaw {hand_yaw('right'):.1f} (cmd "
                f"{planner.r_yaw:.1f}), left hand yaw {hand_yaw('left'):.1f}")
            log(f"STATE k={k}: wheel {planner.from_seat_deg():.1f} deg from seat, A {np.round(a_pos, 4)} yaw "
                f"{math.degrees(planner.a_yaw()):.1f} swing {math.degrees(planner.swing_q()):.1f}, L grasp "
                f"{np.round(grasp_point('left')[0], 4)} cmd {np.round(L.pos + ROOT_OFFSET, 4)}, R grasp "
                f"{np.round(grasp_point('right')[0], 4)} cmd {np.round(R.pos + ROOT_OFFSET, 4)}, lock "
                f"{np.round(planner.body_pose()[0], 4)}")

        self._debug = debug
        self._L, self._R, self._planner, self._joint_action = L, R, planner, joint_action
        self.worst = worst

    def next_action(self) -> np.ndarray:
        k = self.state_k[0]
        L, R = self._L, self._R
        self._planner.run_due(k)
        if k % 20 == 0:
            self._debug(k)
        L.pad_to(len(R.steps))
        R.pad_to(len(L.steps))
        if k >= len(L.steps):
            L.hold(1)
            R.hold(1)
        (lp, lq, lg), (rp, rq, rg) = L.steps[k], R.steps[k]
        self.state_k[0] = k + 1
        return np.concatenate([self._joint_action("left", lp, lq, lg), self._joint_action("right", rp, rq, rg)])
