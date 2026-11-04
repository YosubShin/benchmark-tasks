"""The scripted demo, closed loop.

Arm B (right), fingers down, pinches V-12's lever by its vinyl grip and turns it a quarter turn clockwise to the
closed stop, each move aimed a few degrees past the measured lever angle (a fingertip push, which worked under stiffer
arm gains, slips over the grip under yam_arena's soft wrist gains). It then pinches the latch sleeve and slides it toward the stem so its tongue drops into the
stop plate's closed notch. Arm A (left) takes the open padlock off its pedestal (fingers toward -y, pitched 40 deg
down, closing across the body's thickness), brings it beside the lever with the shackle's legs across the lever, and
slides it -y so the lever passes between the body and the raised short leg until the short leg is over the lock hole
(the short leg's measured tip position is corrected onto the target before each move). Arm B, fingers closed and
pitched 70 deg down, pushes the shackle's bend down until it latches, while arm A holds the body and keeps the
short leg over the hole's centre. Both let go; the
task then runs its function test (off camera).

``control=True``: the same up to the latched sleeve, no padlock; the function test is forced 1 s after arm B lets go.

It reads the simulator's state (lever angle, sleeve, padlock pose), like any scripted demo with privileged state, and
sends yam_arena's 14-D joint action through yam_arena's IK (``yam_arena.ik``). Import after Isaac Sim has started.
"""

from __future__ import annotations

import math

import numpy as np

from yam_arena import ik, station


def _log(*args, **kwargs):
    kwargs["flush"] = True
    print("[valve-demo]", *args, **kwargs)


class Demo:
    """``next_action()`` gives the 14-D joint action for the next control step."""

    def __init__(self, STATION, control_hz: float = 20.0, speed: float = 0.10, control: bool = False, log=_log):
        from valve_lockout_sim import station as hb
        from valve_lockout_sim.kit import CLOSED, DOWN, FORWARD, OPEN, ROOT_OFFSET, ArmTimeline, GraspModel, q_axis, qmul
        from valve_lockout_sim.kit import grasp_point as grasp_point_world
        from valve_lockout_sim.kit import rotate
        from valve_lockout_sim.task import LOCK_QUAT, PV, TARGET_SLOT, lock_start_pose
        from valve_lockout_sim.usd import valve_props as props

        view = STATION.env
        task = STATION.task
        state_k = [0]
        self.state_k = state_k
        origin = view.scene.env_origins[0].cpu().numpy()  # desk frame origin in the world
        robot = view.scene["robot"]

        def grasp_point(side):
            p, q = grasp_point_world(robot, side)
            return p - origin, q

        for side, name in (("left", "left_robot"), ("right", "right_robot")):
            qj = view.scene[name].data.joint_pos.torch[0, :6].cpu().numpy()
            tcp, _ = ik.tcp_pose(qj, station.BASES[side])
            log(f"CHECK {side}: their TCP {np.round(tcp, 4)} our grasp point {np.round(grasp_point(side)[0] + origin, 4)}",
                flush=True)

        try:
            pv = view.scene["pipe_run"]
            log("PIPE actuators", {k: (a.cfg.stiffness, getattr(a.cfg, "effort_limit_sim", None)) for k, a in pv.actuators.items()},
                "max forces", pv.root_physx_view.get_dof_max_forces().torch[0].tolist() if hasattr(pv.root_physx_view.get_dof_max_forces(), "torch") else pv.root_physx_view.get_dof_max_forces()[0],
                "joints", pv.joint_names)
        except Exception as e:  # noqa: BLE001
            log("PIPE actuator debug failed", repr(e))
        LQ = np.array(LOCK_QUAT)
        SIDE_A = qmul(q_axis((0, 0, 1), -math.pi / 2), FORWARD)
        QA = qmul(q_axis((1, 0, 0), math.radians(40)), SIDE_A)
        A_FINGER_DIR = np.array([0.0, -math.cos(math.radians(40)), -math.sin(math.radians(40))])
        GRASP_IN_LOCK = (-0.011, 0.0, -0.008)
        FINGER_END = 0.012
        PINCH_R, PINCH_DZ = 0.088, 0.0
        TURN_LEAD0, TURN_LEAD_MAX, TURN_END = 6.0, 30.0, 100.0
        SLEEVE_DZ = 0.007
        R_WAIT = np.array([0.12, -0.28, 0.24])
        A_WAIT = np.array([0.14, 0.20, 0.25])
        VX, VY = props.VALVE_X, props.VALVE_Y[TARGET_SLOT]
        HOLE_X = VX - props.HOLE_S
        TIP_ABOVE_LEVER = 0.0055
        ENTRY_DY = 0.018
        PRESS_STEP, PRESS_MAX = 0.003, 30
        YAW_B, PITCH_B = math.radians(60), math.radians(70)
        QB = qmul(q_axis((-math.sin(YAW_B), math.cos(YAW_B), 0), PITCH_B), qmul(q_axis((0, 0, 1), YAW_B), FORWARD))
        B_FINGER_DIR = np.array([math.cos(YAW_B) * math.cos(PITCH_B), math.sin(YAW_B) * math.cos(PITCH_B),
                                 -math.sin(PITCH_B)])

        lock, pipe = view.scene["padlock"], view.scene["pipe_run"]
        body_id = lock.find_bodies("body")[0][0]
        shackle_joint = lock.find_joints("shackle_joint")[0][0]
        lever_joint = pipe.find_joints(f"lever_{TARGET_SLOT}_joint")[0][0]
        sleeve_joint = pipe.find_joints(f"sleeve_{TARGET_SLOT}_joint")[0][0]
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

            # state
            def lever_deg(self):
                return -math.degrees(float(pipe.data.joint_pos.torch[0, lever_joint]))

            def sleeve_q(self):
                return float(pipe.data.joint_pos.torch[0, sleeve_joint])

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

            # Arm B: turn the lever ----------------------------------------------------------------
            def grip_pose(self, lever_yaw):
                """Grasp point on the grip at PINCH_R, fingers down, closing across the grip's width."""
                pos = np.array([VX + PINCH_R * math.cos(lever_yaw), VY + PINCH_R * math.sin(lever_yaw),
                                props.LEVER_Z + PINCH_DZ])
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
                err = grasp_point("right")[0] - (R.pos + ROOT_OFFSET)
                R.move(g0 - err, speed=0.02)
                R.hold(hold_k(0.3))
                R.grip_to(CLOSED, hold_k(1.0))
                self.at(len(R.steps), lambda: self.turn(0, TURN_LEAD0, -1.0))

            def turn(self, k, lead_deg, last_deg):
                """One step of the turn, closed loop: aim the pinch ``lead_deg`` past the measured lever angle along
                the grip's arc. yam_arena's arm gains are soft (the wrist joints 10 N*m/rad), so the 16 N the lever
                needs takes a target well ahead; the lead grows while the lever is slow and shrinks when it is fast."""
                deg = self.lever_deg()
                if k % 5 == 0:
                    log(f"TURN {k}: lever {deg:.1f} deg, lead {lead_deg:.0f} deg, grasp R "
                        f"{np.round(grasp_point('right')[0], 4)} cmd {np.round(self.R.pos + ROOT_OFFSET, 4)}")
                if deg >= 89.0 or k > 240:
                    R = self.R
                    R.hold(hold_k(0.5))
                    self.at(len(R.steps), self.turned)
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
                R.move(grasp_point("right")[0] + np.array([0.0, 0.0, 0.05]), speed=0.05)
                self.plan_sleeve()

            # Arm B: slide the latch sleeve in ------------------------------------------------------
            def plan_sleeve(self):
                R = self.R
                s_mid = (props.SLEEVE_IN_S[0] + props.SLEEVE_IN_S[1]) / 2 + props.SLEEVE_TRAVEL
                g = np.array([VX - s_mid, VY, props.LEVER_Z + SLEEVE_DZ])
                R.move(g + np.array([0.0, 0.0, 0.05]), down_yaw(0.0), min_steps=hold_k(2.0))
                R.move(g, speed=0.03)
                R.hold(hold_k(0.3))
                self.when_settled(R, "right", lambda: self.sleeve_grip(g))

            def sleeve_grip(self, g):
                R = self.R
                err = grasp_point("right")[0] - (R.pos + ROOT_OFFSET)
                R.move(g - err, speed=0.02)
                R.grip_to(CLOSED, hold_k(1.0))
                R.move(g - err + np.array([props.SLEEVE_TRAVEL + 0.004, 0.0, 0.0]), speed=0.01)
                R.hold(hold_k(0.6))
                self.when_settled(R, "right", lambda: self.sleeve_done(0), tries=8)

            def sleeve_done(self, attempt):
                q = self.sleeve_q()
                log(f"SLEEVE attempt {attempt}: q {q * 1000:.1f} mm (0 = in), lever {self.lever_deg():.1f} deg")
                R = self.R
                if q < -0.002 and attempt < 3:
                    R.move(R.pos + ROOT_OFFSET + np.array([-q + 0.003, 0.0, 0.0]), speed=0.01)
                    R.hold(hold_k(0.6))
                    self.when_settled(R, "right", lambda: self.sleeve_done(attempt + 1), tries=8)
                    return
                R.grip_to(OPEN, hold_k(0.8))
                R.move(R.pos + ROOT_OFFSET + np.array([0.0, 0.0, 0.06]), speed=0.05)
                R.move(R_WAIT, speed=0.10)
                self.sync()
                if control:
                    self.at(len(R.steps) + hold_k(1.0), lambda: task.force_function_test(view))
                else:
                    self.plan_pick()

            # Arm A: the padlock --------------------------------------------------------------------
            def plan_pick(self):
                start = np.array(lock_start_pose()[:3])
                grasp, _ = GraspModel.nominal(QA, LQ, GRASP_IN_LOCK).grasp_for(start, LQ)
                L = self.L
                L.grip_to(OPEN, hold_k(0.2))
                L.move(grasp - 0.07 * A_FINGER_DIR, QA, min_steps=hold_k(3.0))
                L.hold(hold_k(0.4))
                L.move(grasp - 0.015 * A_FINGER_DIR, speed=0.04)
                L.hold(hold_k(0.4))
                self.when_settled(L, "left", self.final_approach)

            def final_approach(self):
                start = np.array(lock_start_pose()[:3])
                grasp, _ = GraspModel.nominal(QA, LQ, GRASP_IN_LOCK).grasp_for(start, LQ)
                err = grasp_point("left")[0] - (self.L.pos + ROOT_OFFSET)
                self.L.move(grasp - err, speed=0.02)
                self.L.hold(hold_k(0.4))
                self.L.grip_to(CLOSED, hold_k(1.0))
                self.L.move(grasp - err + np.array([0.0, 0.0, 0.04]), speed=0.03)
                self.when_settled(self.L, "left", self.carry)

            def carry(self):
                self.model = GraspModel.measure(robot, "left", lock)
                log("LOCK grasp rel_pos (mm)", np.round(self.model.rel_pos * 1000, 1))
                self.tip_final = np.array([HOLE_X, VY, props.LEVER_Z + TIP_ABOVE_LEVER])
                self.tip_entry = self.tip_final + np.array([0.0, ENTRY_DY, 0.0])
                c = self.lock_center_for_tip(self.tip_entry)
                self.lock_to(c + np.array([0.0, 0.03, 0.03]), 0.06)
                self.lock_to(c, 0.03)
                self.L.hold(hold_k(0.6))
                self.when_settled(self.L, "left", lambda: self.entry_correct(np.zeros(3), 0))

            def entry_correct(self, correction, attempt):
                err = self.tip() - self.tip_entry
                log(f"ENTRY attempt {attempt}: tip error (mm) {np.round(err * 1000, 1)}")
                correction = correction + err
                if attempt < 4 and np.abs(err).max() > 0.0015:
                    self.lock_to(self.lock_center_for_tip(self.tip_entry - correction), 0.01)
                    self.L.hold(hold_k(0.6))
                    self.when_settled(self.L, "left", lambda: self.entry_correct(correction, attempt + 1))
                    return
                self.correction = correction
                self.lock_to(self.lock_center_for_tip(self.tip_final - correction), 0.008)
                self.L.hold(hold_k(0.8))
                self.when_settled(self.L, "left", lambda: self.over_correct(0))

            def over_correct(self, attempt):
                err = self.tip() - self.tip_final
                log(f"OVER hole attempt {attempt}: tip error (mm) {np.round(err * 1000, 1)}")
                self.correction = self.correction + np.array([err[0], err[1], 0.0])
                if attempt < 4 and max(abs(err[0]), abs(err[1])) > 0.001:
                    self.lock_to(self.lock_center_for_tip(self.tip_final - self.correction), 0.006)
                    self.L.hold(hold_k(0.6))
                    self.when_settled(self.L, "left", lambda: self.over_correct(attempt + 1))
                    return
                self.sync()
                self.press_start()

            # Arm B: shut the shackle ---------------------------------------------------------------
            def bend_top(self):
                pos, quat = self.body_pose()
                return pos + rotate(quat, (0.0, 0.0, PV.bend_top(self.travel())))

            def press_start(self):
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
                    log(f"PRESS done after {k}: travel {self.travel() * 1000:.1f} mm, latched {latched}, "
                        f"tip {np.round(self.tip(), 4)}")
                    self.R.hold(hold_k(0.3))
                    self.R.move(self.R.pos + ROOT_OFFSET - 0.03 * B_FINGER_DIR, speed=0.03)
                    self.sync()
                    self.release()
                    return
                if k % 3 == 0:
                    log(f"PRESS {k}: travel {self.travel() * 1000:.1f} mm, bend top {np.round(self.bend_top(), 4)}, "
                        f"tip error (mm) {np.round((self.tip() - self.tip_final) * 1000, 1)}, "
                        f"body z {self.body_pose()[0][2]:.4f} (start {self.body0[2]:.4f})")
                # Arm A holds the body against the push and keeps the short leg's tip over the hole's centre (arm B's
                # push and the soft wrists move both): a small integral correction each press step.
                body = self.body_pose()[0]
                err = self.tip() - self.tip_final
                corr = np.array([-0.5 * err[0], -0.5 * err[1], 0.4 * (self.body0[2] - body[2])])
                self.L.pad_to(len(self.R.steps))
                if np.abs(corr).max() > 0.0003:
                    self.L.move(self.L.pos + ROOT_OFFSET + corr, speed=0.02)
                self.R.move(self.R.pos + ROOT_OFFSET + np.array([0.0, 0.0, -PRESS_STEP]), speed=0.02)
                self.R.hold(hold_k(0.2))
                self.sync()
                self.at(len(self.R.steps), lambda: self.press(k + 1))

            def release(self):
                R, L = self.R, self.L
                R.move(R_WAIT, speed=0.10)
                L.grip_to(0.6, hold_k(1.0))
                L.move(L.pos + ROOT_OFFSET - 0.05 * A_FINGER_DIR, speed=0.03)
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
        planner.at(settle, planner.plan_turn)

        q_rel = np.array(hb.Q_REL)
        tcp_extra = 0.1347 - 0.134  # our grasp point sits this much further out than their TCP
        q_arm = {s: view.scene[f"{s}_robot"].data.joint_pos.torch[0, :6].cpu().numpy().astype(float)
                 for s in ("left", "right")}
        worst = {"left": 0.0, "right": 0.0}
        last_err = {"left": 0.0, "right": 0.0}

        def joint_action(side, pos_root, quat, grip):
            pos = pos_root + ROOT_OFFSET + origin  # world
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
