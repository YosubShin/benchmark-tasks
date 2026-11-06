"""The scripted demo, closed loop.

1. Arm B (right), fingers down, pinches V-12's lever by its vinyl grip and turns it a quarter turn clockwise to the
   closed stop, each move aimed a few degrees past the measured lever angle.
2. Arm A (left), fingers down, picks the lockout up by its channel and lowers it over the closed lever: the lever in
   the channel, A's legs just in front of the pipe, B's legs (opened to 40 mm) beyond it.
3. Arm A lets go, turns its fingers to close along x and squeezes the two halves together across the pipe, as a hand
   does, until B's housing reaches the first setting (its first hole lined up with A's tongue).
4. Arm B takes the open padlock off its cradle (fingers down, closing across the body's thickness), carries it on its
   side to the right of the lockout and lowers it so the housing's wall slides between the body and the raised short
   leg, with the leg's tip over the lined-up hole (the tip's measured position is corrected onto the target).
5. Arm A, fingers closed, pushes the shackle's bend toward the body until it latches, while arm B holds the body and
   keeps the tip on the hole. Both let go; the task then runs its after-release test (off camera).

``control=True``: steps 1-3 only, no padlock; the after-release test is forced 1 s after arm A lets go.

It reads the simulator's state (lever angle, slide, padlock and lockout poses), like any scripted demo with privileged
state, and sends yam_arena's 14-D joint action through yam_arena's IK (``yam_arena.ik``). Import after Isaac Sim has
started.
"""

from __future__ import annotations

import math

import numpy as np

from yam_arena import ik, station


def _log(*args, **kwargs):
    kwargs["flush"] = True
    print("[clamp-demo]", *args, **kwargs)


class Demo:
    """``next_action()`` gives the 14-D joint action for the next control step."""

    def __init__(self, STATION, control_hz: float = 20.0, speed: float = 0.10, control: bool = False, upto: int = 99,
                 start: int = 1, log=_log):
        from clamp_valve_sim import station as hb
        from clamp_valve_sim.kit import CLOSED, DOWN, FORWARD, OPEN, ROOT_OFFSET, ArmTimeline, GraspModel, q_axis, qmul
        from clamp_valve_sim.kit import grasp_point as grasp_point_world
        from clamp_valve_sim.kit import rotate
        from clamp_valve_sim.task import LOCK_QUAT, PV, TARGET_SLOT, device_fitted_pos, device_start_pose, lock_start_pose
        from clamp_valve_sim.usd import clamp_props as dev
        from clamp_valve_sim.usd import valve_props as props

        view = STATION.env
        task = STATION.task
        state_k = [0]
        self.state_k = state_k
        origin = view.scene.env_origins[0].cpu().numpy()
        robot = view.scene["robot"]

        def grasp_point(side):
            p, q = grasp_point_world(robot, side)
            return p - origin, q

        LQ = np.array(LOCK_QUAT)
        IDQ = np.array([0.0, 0.0, 0.0, 1.0])
        PINCH_R = 0.070
        TURN_LEAD0, TURN_LEAD_MAX, TURN_END = 6.0, 35.0, 110.0
        R_WAIT = np.array([0.15, -0.20, 0.20])
        A_WAIT = np.array([0.15, 0.20, 0.20])
        VX, VY = props.VALVE_X, props.VALVE_Y[TARGET_SLOT]
        FIT = np.array(device_fitted_pos())
        DEV_GRASP = np.array([-0.016, 0.0, -0.010])
        """Arm A's grasp point, in A's frame: across the outside of A's two legs (B's wall on the right one), near the
        device's centre of mass, so it does not swing about the fingers' axis."""
        CARRY_Z = 0.202
        SQUEEZE_Y, SQUEEZE_Z = 0.033, props.LEVER_Z - 0.007
        LOCK_GRASP = np.array([-0.009, 0.0, 0.0])
        """Arm B's grasp point in the padlock frame: 10 mm below the body's upper side face (lock -x is up)."""
        TIP_IN_A = np.array([0.0, dev.TONGUE_Y[1] + 0.0022, dev.HOLE_Z])
        """The short leg's tip, in A's frame: on the tongue hole's axis, 2.2 mm in from the tongue's inner face."""
        TIP_TARGET = np.array([VX, VY, props.LEVER_Z]) + TIP_IN_A
        ENTRY_DZ = 0.030
        PRESS_PITCH = math.radians(15)
        PRESS_Q = qmul(q_axis((1, 0, 0), PRESS_PITCH), qmul(q_axis((0, 0, 1), -math.pi / 2), FORWARD))
        PRESS_DIR = np.array([0.0, -math.cos(PRESS_PITCH), -math.sin(PRESS_PITCH)])
        """Arm A's finger direction while it shuts the shackle."""
        FINGER_END = 0.012
        PRESS_STEP, PRESS_MAX = 0.002, 45

        lock, pipe, lockout = view.scene["padlock"], view.scene["pipe_run"], view.scene["lockout"]
        body_id = lock.find_bodies("body")[0][0]
        shackle_joint = lock.find_joints("shackle_joint")[0][0]
        lever_joint = pipe.find_joints(f"lever_{TARGET_SLOT}_joint")[0][0]
        slide_joint = lockout.find_joints("slide_joint")[0][0]
        a_id = lockout.find_bodies("half_a")[0][0]
        dt = 1.0 / control_hz
        hold_k = lambda s: max(1, round(s / dt))  # noqa: E731

        def down_yaw(yaw):
            return qmul(q_axis((0, 0, 1), yaw), DOWN)

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
            def lever_deg(self):
                return -math.degrees(float(pipe.data.joint_pos.torch[0, lever_joint]))

            def slide_q(self):
                return float(lockout.data.joint_pos.torch[0, slide_joint])

            def a_pose(self):
                return (lockout.data.body_pos_w.torch[0, a_id].cpu().numpy() - origin,
                        lockout.data.body_quat_w.torch[0, a_id].cpu().numpy())

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

            # 1. Arm B: turn the lever ---------------------------------------------------------------
            def grip_pose(self, lever_yaw):
                pos = np.array([VX + PINCH_R * math.cos(lever_yaw), VY + PINCH_R * math.sin(lever_yaw),
                                props.LEVER_Z])
                return pos, down_yaw(lever_yaw + math.pi)

            def plan_turn(self):
                R = self.R
                R.grip = OPEN
                g0, q0 = self.grip_pose(-math.pi / 2)
                R.move(g0 + np.array([0.0, 0.0, 0.06]), q0, min_steps=hold_k(3.0))
                R.move(g0 + np.array([0.0, 0.0, 0.015]), speed=0.04)
                R.hold(hold_k(0.3))
                self.when_settled(R, "right", lambda: self.pinch(g0))

            def pinch(self, g0):
                R = self.R
                err = grasp_point("right")[0] - self.here(R)
                R.move(g0 - err, speed=0.02)
                R.hold(hold_k(0.3))
                R.grip_to(CLOSED, hold_k(1.0))
                self.at(len(R.steps), lambda: self.turn(0, TURN_LEAD0, -1.0))

            def turn(self, k, lead_deg, last_deg):
                deg = self.lever_deg()
                if k % 5 == 0:
                    log(f"TURN {k}: lever {deg:.1f} deg, lead {lead_deg:.0f} deg")
                if deg >= 89.0 or k > 240:
                    self.R.hold(hold_k(0.5))
                    self.at(len(self.R.steps), self.turned)
                    return
                rate = deg - last_deg
                if rate < 1.0:
                    lead_deg = min(TURN_LEAD_MAX, lead_deg + 2.0)
                elif rate > 3.0:
                    lead_deg = max(TURN_LEAD0, lead_deg - 2.0)
                phi = min(TURN_END, deg + lead_deg)
                pos, quat = self.grip_pose(-math.pi / 2 - math.radians(phi))
                self.R.move(pos, quat, speed=0.06, min_steps=hold_k(0.15))
                self.at(len(self.R.steps), lambda: self.turn(k + 1, lead_deg, deg))

            def turned(self):
                log(f"TURNED: lever {self.lever_deg():.1f} deg closed")
                R = self.R
                R.grip_to(OPEN, hold_k(0.6))
                R.move(grasp_point("right")[0] + np.array([0.0, 0.0, 0.04]), speed=0.05)
                if self.lever_deg() < 89.0:
                    self.plan_finish()
                    return
                self.turn_done()

            def plan_finish(self):
                """The pinch can stall a few degrees short of the stop (the wrist runs out of travel low down): push the
                grip's right side to the stop with closed fingertips instead."""
                R = self.R
                R.grip = CLOSED
                p = np.array([VX - 0.080, VY - 0.030, props.LEVER_Z + 0.004])
                R.move(p + np.array([0.0, 0.0, 0.04]), down_yaw(0.0), min_steps=hold_k(1.0))
                R.move(p, speed=0.03)
                R.hold(hold_k(0.3))
                self.at(len(R.steps), lambda: self.finish(0))

            def finish(self, k):
                deg = self.lever_deg()
                if k % 4 == 0:
                    log(f"FINISH {k}: lever {deg:.1f} deg")
                if deg >= 89.5 or k > 30:
                    log(f"FINISHED: lever {deg:.1f} deg")
                    self.R.move(self.here(self.R) + np.array([0.0, -0.02, 0.04]), speed=0.04)
                    self.turn_done()
                    return
                self.R.move(self.here(self.R) + np.array([0.0, 0.002, 0.0]), speed=0.02)
                self.R.hold(hold_k(0.1))
                self.at(len(self.R.steps), lambda: self.finish(k + 1))

            def turn_done(self):
                R = self.R
                R.move(R_WAIT, speed=0.10)
                self.sync()
                if upto >= 2:
                    self.plan_device_pick()

            # 2. Arm A: the lockout onto the lever -------------------------------------------------------
            def plan_device_pick(self):
                L = self.L
                start = np.array(device_start_pose()[:3])
                g = start + DEV_GRASP
                L.grip_to(OPEN, hold_k(0.2))
                L.move(g + np.array([0.0, 0.0, 0.07]), down_yaw(0.0), min_steps=hold_k(2.5))
                L.move(g + np.array([0.0, 0.0, 0.015]), speed=0.05)
                L.hold(hold_k(0.3))
                self.when_settled(L, "left", lambda: self.device_grip(g))

            def device_grip(self, g):
                L = self.L
                err = grasp_point("left")[0] - self.here(L)
                L.move(g - err, speed=0.02)
                L.hold(hold_k(0.3))
                L.grip_to(CLOSED, hold_k(1.0))
                L.move(self.here(L) + np.array([0.0, 0.0, 0.05]), speed=0.03)
                self.when_settled(L, "left", self.device_carry)

            def device_carry(self):
                self.dmodel = GraspModel.measure(robot, "left", lockout)
                a_pos, a_quat = self.a_pose()
                log(f"DEVICE grasp rel_pos (mm) {np.round(self.dmodel.rel_pos * 1000, 1)}, A at {np.round(a_pos, 4)}, "
                    f"slide {self.slide_q() * 1000:.1f} mm")
                L = self.L
                # Straight up, then forward and up over V-12's lever (the legs pass either side of it, the channel
                # above it), stopping short of the pipe: the soft wrist lets the held device sag, so its height is
                # corrected there before B's legs go over the pipe.
                L.move(self.dmodel.grasp_for(np.array([a_pos[0], a_pos[1], 0.16]), IDQ)[0], speed=0.04)
                # Across to the midline in front of the closed lever's tip (B's legs stay short of it).
                L.move(self.dmodel.grasp_for(np.array([0.105, 0.0, 0.16]), IDQ)[0], speed=0.04)
                pre = np.array([0.19, 0.0, CARRY_Z])
                L.move(self.dmodel.grasp_for(pre, IDQ)[0], speed=0.03)
                L.hold(hold_k(0.5))
                self.when_settled(L, "left", lambda: self.device_pre(pre, np.zeros(3), 0), tries=10)

            def device_pre(self, pre, corr, attempt):
                a_pos, _ = self.a_pose()
                err = a_pos - pre
                log(f"DEVICE pre {attempt}: A error (mm) {np.round(err * 1000, 1)}")
                corr = corr + err
                if attempt < 4 and np.abs(err).max() > 0.003:
                    self.L.move(self.dmodel.grasp_for(pre - corr, IDQ)[0], speed=0.02)
                    self.L.hold(hold_k(0.5))
                    self.when_settled(self.L, "left", lambda: self.device_pre(pre, corr, attempt + 1), tries=10)
                    return
                above = FIT + np.array([-0.002, 0.0, CARRY_Z - FIT[2]])
                self.L.move(self.dmodel.grasp_for(above - corr, IDQ)[0], speed=0.04)
                self.L.hold(hold_k(0.5))
                self.when_settled(self.L, "left", lambda: self.device_align(corr, 0), tries=10)

            def device_align(self, corr, attempt):
                a_pos, _ = self.a_pose()
                target = FIT + np.array([-0.002, 0.0, CARRY_Z - FIT[2]])
                err = a_pos - target
                log(f"DEVICE align {attempt}: A error (mm) {np.round(err * 1000, 1)}")
                corr = corr + np.array([err[0], err[1], 0.0])
                if attempt < 4 and max(abs(err[0]), abs(err[1])) > 0.0015:
                    self.L.move(self.dmodel.grasp_for(target - corr, IDQ)[0], speed=0.01)
                    self.L.hold(hold_k(0.5))
                    self.when_settled(self.L, "left", lambda: self.device_align(corr, attempt + 1), tries=10)
                    return
                self.lower(corr, 0)

            def lower(self, corr, attempt):
                """Down to 1 cm above the seat, correcting x and y on the way (the arm lags while it moves)."""
                a_pos, _ = self.a_pose()
                near = FIT + np.array([-0.002, 0.0, 0.010])
                err = a_pos - near
                if attempt > 0:
                    log(f"DEVICE lower {attempt}: A error (mm) {np.round(err * 1000, 1)}")
                    corr = corr + np.array([err[0], err[1], 0.0])
                if attempt == 0 or (attempt < 5 and max(abs(err[0]), abs(err[1])) > 0.0015):
                    self.L.move(self.dmodel.grasp_for(near - corr, IDQ)[0], speed=0.015)
                    self.L.hold(hold_k(0.6))
                    self.when_settled(self.L, "left", lambda: self.lower(corr, attempt + 1), tries=10)
                    return
                low = FIT + np.array([-0.002, 0.0, 0.002])
                self.L.move(self.dmodel.grasp_for(low - corr, IDQ)[0], speed=0.01)
                self.L.hold(hold_k(0.6))
                self.when_settled(self.L, "left", self.device_down, tries=15)

            def device_down(self):
                a_pos, a_quat = self.a_pose()
                log(f"DEVICE down: A at {np.round(a_pos, 4)} (fit {np.round(FIT, 4)}), slide {self.slide_q() * 1000:.1f}"
                    f" mm, lever {self.lever_deg():.1f} deg")
                L = self.L
                L.move(self.here(L) + np.array([0.0, 0.0, -0.006]), speed=0.01)
                L.grip_to(OPEN, hold_k(0.8))
                L.move(self.here(L) + np.array([0.0, 0.0, 0.05]), speed=0.04)
                self.at(len(L.steps), self.plan_squeeze)

            # 3. Arm A: squeeze the halves together ------------------------------------------------------
            def plan_squeeze(self):
                a_pos, _ = self.a_pose()
                log(f"SETTLED on valve: A at {np.round(a_pos, 4)}, slide {self.slide_q() * 1000:.1f} mm")
                L = self.L
                g = np.array([VX, VY + SQUEEZE_Y, SQUEEZE_Z])
                L.move(g + np.array([0.0, 0.0, 0.05]), down_yaw(math.pi / 2), min_steps=hold_k(1.5))
                L.move(g, speed=0.03)
                L.hold(hold_k(0.3))
                self.when_settled(L, "left", lambda: self.squeeze(g))

            def squeeze(self, g):
                L = self.L
                err = grasp_point("left")[0] - self.here(L)
                L.move(g - np.array([err[0], err[1], 0.0]), speed=0.01)
                n = hold_k(2.0)
                for k in range(1, n + 1):
                    L.steps.append((L.pos.copy(), L.quat.copy(), OPEN * (1 - k / n) + CLOSED * k / n))
                L.grip = CLOSED
                L.hold(hold_k(1.0))
                self.at(len(L.steps), self.squeezed)

            def squeezed(self):
                a_pos, _ = self.a_pose()
                log(f"SQUEEZED: slide {self.slide_q() * 1000:.1f} mm, A at {np.round(a_pos, 4)}, lever "
                    f"{self.lever_deg():.1f} deg, fitted {bool(view._task_state.fitted_ever[0])}")
                L = self.L
                L.grip_to(OPEN, hold_k(0.8))
                L.move(self.here(L) + np.array([0.0, 0.03, 0.06]), speed=0.04)
                L.move(A_WAIT, speed=0.08)
                self.sync()
                if control:
                    self.at(len(L.steps) + hold_k(1.0), lambda: task.force_function_test(view))
                elif upto >= 4:
                    self.plan_lock_pick()

            # 4. Arm B: the padlock ----------------------------------------------------------------------
            def plan_lock_pick(self):
                pos, quat = self.body_pose()
                g = pos + rotate(quat, LOCK_GRASP)
                R = self.R
                R.grip_to(OPEN, hold_k(0.2))
                R.move(g + np.array([0.0, 0.0, 0.07]), down_yaw(math.pi / 2), min_steps=hold_k(2.0))
                R.move(g + np.array([0.0, 0.0, 0.015]), speed=0.05)
                R.hold(hold_k(0.3))
                self.when_settled(R, "right", lambda: self.lock_grip(g))

            def lock_grip(self, g):
                R = self.R
                err = grasp_point("right")[0] - self.here(R)
                R.move(g - err, speed=0.02)
                R.hold(hold_k(0.3))
                R.grip_to(CLOSED, hold_k(1.0))
                R.move(self.here(R) + np.array([0.0, 0.0, 0.05]), speed=0.03)
                self.when_settled(R, "right", self.lock_carry)

            def lock_to_tip(self, tip_target, speed):
                c = np.asarray(tip_target) - rotate(LQ, PV.short_tip(self.travel()))
                self.R.move(self.lmodel.grasp_for(c, LQ)[0], speed=speed)

            def lock_carry(self):
                self.lmodel = GraspModel.measure(robot, "right", lock)
                a_pos, a_quat = self.a_pose()
                self.tip_target = a_pos + rotate(a_quat, TIP_IN_A)
                log(f"TIP target {np.round(self.tip_target, 4)} (A at {np.round(a_pos, 4)})")
                log("LOCK grasp rel_pos (mm)", np.round(self.lmodel.rel_pos * 1000, 1))
                self.entry = self.tip_target + np.array([0.0, 0.0, ENTRY_DZ])
                self.lock_to_tip(self.entry + np.array([-0.03, -0.03, 0.02]), 0.06)
                self.lock_to_tip(self.entry, 0.03)
                self.R.hold(hold_k(0.6))
                self.when_settled(self.R, "right", lambda: self.entry_correct(np.zeros(3), 0))

            def entry_correct(self, corr, attempt):
                err = self.tip() - self.entry
                log(f"ENTRY attempt {attempt}: tip error (mm) {np.round(err * 1000, 1)}")
                corr = corr + err
                if attempt < 4 and np.abs(err).max() > 0.001:
                    self.lock_to_tip(self.entry - corr, 0.01)
                    self.R.hold(hold_k(0.6))
                    self.when_settled(self.R, "right", lambda: self.entry_correct(corr, attempt + 1))
                    return
                self.corr = corr
                self.lock_to_tip(self.tip_target - corr, 0.008)
                self.R.hold(hold_k(0.8))
                self.when_settled(self.R, "right", lambda: self.on_correct(0))

            def on_correct(self, attempt):
                err = self.tip() - self.tip_target
                log(f"ON hole attempt {attempt}: tip error (mm) {np.round(err * 1000, 1)}")
                self.corr = self.corr + np.array([err[0], 0.0, err[2]])
                if attempt < 4 and max(abs(err[0]), abs(err[2])) > 0.0008:
                    self.lock_to_tip(self.tip_target - self.corr, 0.006)
                    self.R.hold(hold_k(0.6))
                    self.when_settled(self.R, "right", lambda: self.on_correct(attempt + 1))
                    return
                self.sync()
                if upto >= 5:
                    self.press_start()

            # 5. Arm A: shut the shackle -----------------------------------------------------------------
            def press_start(self):
                """Arm A's closed fingertips, pointing -y (across the lockout) and pitched 30 deg down, come in low from
                the left, clear of arm B's hand above the padlock, and rest on the bend's outer side."""
                L = self.L
                b = self.bend_top() + np.array([0.0, 0.003, 0.003])
                start = b - FINGER_END * PRESS_DIR
                L.grip = CLOSED
                L.move(start - 0.06 * PRESS_DIR + np.array([0.0, 0.0, 0.03]), PRESS_Q, min_steps=hold_k(3.0))
                L.move(start - 0.03 * PRESS_DIR, speed=0.03)
                L.move(start, speed=0.02)
                L.hold(hold_k(0.4))
                self.body0 = self.body_pose()[0]
                self.when_settled(L, "left", lambda: self.press(0))

            def press(self, k):
                latched = bool(view._task_state.latched[0])
                if latched or k > PRESS_MAX:
                    log(f"PRESS done after {k}: travel {self.travel() * 1000:.1f} mm, latched {latched}, through "
                        f"{bool(view._task_state.latched_through[0])}, slide {self.slide_q() * 1000:.1f} mm")
                    self.L.hold(hold_k(0.3))
                    self.sync()
                    self.release()
                    return
                if k % 3 == 0:
                    log(f"PRESS {k}: travel {self.travel() * 1000:.1f} mm, bend {np.round(self.bend_top(), 4)}, "
                        f"tip error (mm) {np.round((self.tip() - self.tip_target) * 1000, 1)}")
                body = self.body_pose()[0]
                err = self.tip() - self.tip_target
                corr = np.array([-0.8 * err[0], 0.6 * (self.body0[1] - body[1]), -0.8 * err[2]])
                self.R.pad_to(len(self.L.steps))
                if np.abs(corr).max() > 0.0003:
                    self.R.move(self.here(self.R) + corr, speed=0.02)
                self.L.move(self.here(self.L) + np.array([0.0, -PRESS_STEP, 0.0]), speed=0.02)
                self.L.hold(hold_k(0.15))
                self.sync()
                self.at(len(self.L.steps), lambda: self.press(k + 1))

            def release(self):
                R, L = self.R, self.L
                L.move(self.here(L) + np.array([0.0, 0.03, 0.03]), speed=0.03)
                L.move(A_WAIT, speed=0.08)
                R.grip_to(0.6, hold_k(1.0))
                R.move(self.here(R) + np.array([0.0, 0.0, 0.05]), speed=0.03)
                R.move(R_WAIT, speed=0.08)
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
            # Debug start: V-12 already closed (written to the simulator), begin with the lockout.
            import torch
            qc = torch.tensor([[math.radians(props.CLOSED_DEG)]], device=pipe.device)
            pipe.write_joint_state_to_sim(qc, torch.zeros_like(qc), joint_ids=[lever_joint])
            pipe.set_joint_position_target(qc, joint_ids=[lever_joint])
            if start >= 4:
                # Debug start: the lockout already fitted and pushed together.
                env_o = view.scene.env_origins[0:1]
                pose = torch.tensor([[*FIT.tolist(), 0.0, 0.0, 0.0, 1.0]], device=pipe.device, dtype=torch.float32)
                pose[:, :3] += env_o
                lockout.write_root_pose_to_sim(pose)
                lockout.write_root_velocity_to_sim(torch.zeros(1, 6, device=pipe.device))
                q0 = torch.zeros(1, 1, device=pipe.device)
                lockout.write_joint_state_to_sim(q0, q0, joint_ids=[slide_joint])
                lockout.set_joint_position_target(q0, joint_ids=[slide_joint])
                planner.at(settle, planner.plan_lock_pick)
            else:
                planner.at(settle, planner.plan_device_pick)
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
            q_arm[side] = q
            if pe > 0.005 and pe > worst[side] + 0.004:
                log(f"IK {side} k={state_k[0]} error {pe * 1000:.1f} mm at {np.round(pos - origin, 4)}")
            worst[side] = max(worst[side], pe)
            return np.r_[q, grip]

        def debug(k):
            a_pos, a_quat = planner.a_pose()
            x_ax = rotate(a_quat, (1.0, 0.0, 0.0))
            z_ax = rotate(a_quat, (0.0, 0.0, 1.0))
            log(f"STATE k={k}: A {np.round(a_pos, 4)} yaw {math.degrees(math.atan2(x_ax[1], x_ax[0])):.1f} "
                f"tilt {math.degrees(math.acos(min(1.0, z_ax[2]))):.1f} slide {planner.slide_q() * 1000:.1f} mm, "
                f"L grasp {np.round(grasp_point('left')[0], 4)} cmd {np.round(L.pos + ROOT_OFFSET, 4)}, lever "
                f"{planner.lever_deg():.1f}, lock {np.round(planner.body_pose()[0], 4)}")

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
