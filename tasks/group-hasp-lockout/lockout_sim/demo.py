"""The scripted demo, closed loop: arm A (left) picks the open padlock off its pedestal, carries it beside the hasp's
tail, lines the short leg up with a free tail hole from the measured tip position (up to five corrections) and pushes
it through both jaws; arm B (right), fingers closed, pushes the shackle's bend shut while arm A holds the body against
the push; both let go. Grasp-point targets become yam_arena's 14-D joint action through yam_arena's own IK
(``yam_arena.ik``) every control step.

It reads the simulator's state (the padlock's and hasp's poses), like any scripted demo with privileged state;
``--policy lockout-demo`` (lockout_sim.policy) plays it through Inspect Robots. Import after Isaac Sim has started.
"""

from __future__ import annotations

import math

import numpy as np

from yam_arena import ik, station


class Demo:
    """``next_action()`` gives the 14-D joint action for the next control step."""

    def __init__(self, STATION, control_hz: float = 20.0, speed: float = 0.10, log=print):
        from lockout_sim import station as hb

        view = STATION.env
        task = STATION.task
        state_k = [0]
        self.state_k = state_k
        # Imports that need Kit.
        from lockout_sim.kit import CLOSED, FORWARD, OPEN, ROOT_OFFSET, ArmTimeline, GraspModel, q_axis, qmul, rotate
        from lockout_sim.kit import grasp_point as grasp_point_world
        from lockout_sim.task import PV, _YAW_Q, layout_dir, lock_start_pose
        from lockout_sim.usd import hasp_props as props
        origin = view.scene.env_origins[0].cpu().numpy()  # desk frame origin in the world
        robot = view.scene["robot"]

        def grasp_point(side):
            p, q = grasp_point_world(robot, side)
            return p - origin, q

        # Sanity: their TCP (FK of the measured joints) against our grasp point through the ArmsView.
        for side, name in (("left", "left_robot"), ("right", "right_robot")):
            qj = view.scene[name].data.joint_pos.torch[0, :6].cpu().numpy()
            tcp, _ = ik.tcp_pose(qj, station.BASES[side])
            log(f"CHECK {side}: their TCP {np.round(tcp, 4)} our grasp point {np.round(grasp_point(side)[0] + origin, 4)}",
                  flush=True)

        # The plan of the unturned layout, turned with the prop assembly (lockout_sim.task.LAYOUT_YAW_DEG).
        turn = lambda v: np.array(layout_dir(v))  # noqa: E731
        turn_q = lambda q: qmul(_YAW_Q, q)  # noqa: E731
        EX, EY, EZ = turn((1.0, 0.0, 0.0)), turn((0.0, 1.0, 0.0)), np.array([0.0, 0.0, 1.0])
        LOCK_QUAT = np.array(lock_start_pose()[3:])
        SIDE_A = qmul(q_axis((0, 0, 1), -math.pi / 2), FORWARD)
        QA = turn_q(qmul(q_axis((1, 0, 0), math.radians(40)), SIDE_A))
        A_FINGER_DIR = turn((0.0, -math.cos(math.radians(40)), -math.sin(math.radians(40))))
        YAW_B = math.radians(60)
        QB_WAIT = qmul(q_axis((-math.sin(YAW_B), math.cos(YAW_B), 0), math.radians(20)), qmul(q_axis((0, 0, 1), YAW_B), FORWARD))
        QB = turn_q(QB_WAIT)
        B_FINGER_DIR = turn((math.cos(YAW_B) * math.cos(math.radians(20)), math.sin(YAW_B) * math.cos(math.radians(20)),
                             -math.sin(math.radians(20))))
        FINGER_END, GRASP_IN_LOCK, R_WAIT, HOLE, FACE_GAP, PIERCE = 0.012, (0.0, 0.0, -0.012), (0.18, -0.26, 0.24), (0, 0), 0.006, 0.010

        lock, hasp = view.scene["padlock"], view.scene["hasp"]
        body_id = lock.find_bodies("body")[0][0]
        shackle_joint = lock.find_joints("shackle_joint")[0][0]
        jaw_a = hasp.find_bodies("jaw_a")[0][0]
        dt = 1.0 / control_hz
        hold_k = lambda s: max(1, round(s / dt))  # noqa: E731  (the 10 Hz reference's step counts, in seconds)

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
                            log(f"SETTLE timeout ({side}): lag {lag * 1000:.1f} mm")
                        fn()
                    else:
                        check.tries -= 1
                        tl.hold(hold_k(0.1))
                        self.at(len(tl.steps), check)
                check.tries = tries
                self.at(len(tl.steps), check)

            def body_pose(self):
                pos = lock.data.body_pos_w.torch[0, body_id].cpu().numpy() - origin
                return pos, lock.data.body_quat_w.torch[0, body_id].cpu().numpy()

            def travel(self):
                return float(lock.data.joint_pos.torch[0, shackle_joint])

            def tip(self):
                pos, quat = self.body_pose()
                return pos + rotate(quat, PV.short_tip(self.travel()))

            def hasp_pose(self):
                pos = hasp.data.body_pos_w.torch[0, jaw_a].cpu().numpy() - origin
                return pos, hasp.data.body_quat_w.torch[0, jaw_a].cpu().numpy()

            def hole(self):
                pos, quat = self.hasp_pose()
                return pos + rotate(quat, props.hole_local(*HOLE))

            def face_point(self, hole):
                """The point FACE_GAP short of the hasp's mid-plane (on the robot lock's side), level with ``hole``."""
                s = float((self.hasp_pose()[0] - hole) @ EY)
                return hole + (s - FACE_GAP) * EY

            def lock_center_for_tip(self, tip_target):
                return np.asarray(tip_target) - rotate(LOCK_QUAT, PV.short_tip(self.travel()))

            def _lock_to(self, center, speed):
                self.L.move(self.model.grasp_for(center, LOCK_QUAT)[0], speed=speed)

            def plan_pick(self):
                start = np.array(lock_start_pose()[:3])
                grasp, _ = GraspModel.nominal(QA, LOCK_QUAT, GRASP_IN_LOCK).grasp_for(start, LOCK_QUAT)
                L = self.L
                L.grip_to(OPEN, hold_k(0.2))
                L.move(grasp - 0.07 * A_FINGER_DIR, QA, min_steps=hold_k(3.0))
                L.hold(hold_k(0.4))
                L.move(grasp - 0.015 * A_FINGER_DIR, speed=0.04)
                L.hold(hold_k(0.4))
                self.when_settled(L, "left", self._final_approach)

            def _final_approach(self):
                # Closed loop on the approach too: put the measured grasp point on the planned one before closing.
                start = np.array(lock_start_pose()[:3])
                grasp, _ = GraspModel.nominal(QA, LOCK_QUAT, GRASP_IN_LOCK).grasp_for(start, LOCK_QUAT)
                err = grasp_point("left")[0] - (self.L.pos + ROOT_OFFSET)
                self.L.move(grasp - err, speed=0.02)
                self.L.hold(hold_k(0.4))
                self.L.grip_to(CLOSED, hold_k(1.0))
                self.L.move(grasp - err + np.array([0.0, 0.0, 0.035]), speed=0.03)
                self.when_settled(self.L, "left", self.carry)

            def carry(self):
                self.model = GraspModel.measure(robot, "left", lock)
                log("LOCK grasp rel_pos (mm)", np.round(self.model.rel_pos * 1000, 1))
                self.tip_entry = self.face_point(self.hole()) - 0.020 * EX
                c = self.lock_center_for_tip(self.tip_entry)
                self._lock_to(c + 0.03 * EY + 0.03 * EZ, 0.06)
                self._lock_to(c, 0.03)
                self.L.hold(hold_k(0.6))
                self.when_settled(self.L, "left", lambda: self._entry_correct(np.zeros(3), 0))

            def _entry_correct(self, correction, attempt):
                err = self.tip() - self.tip_entry
                log(f"ENTRY attempt {attempt}: tip error (mm) {np.round(err * 1000, 1)}")
                correction = correction + err
                if attempt < 4 and np.abs(err).max() > 0.0015:
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
                log(f"OVER hole attempt {attempt}: tip error (mm) {np.round(err * 1000, 1)}")
                ex, ez = float(err @ EX), float(err[2])
                self.correction = self.correction + ex * EX + ez * EZ
                if attempt < 4 and max(abs(ex), abs(ez)) > 0.001:
                    self._lock_to(self.lock_center_for_tip(self.tip_over - self.correction), 0.008)
                    self.L.hold(hold_k(0.6))
                    self.when_settled(self.L, "left", lambda: self._over_correct(attempt + 1))
                    return
                through = self.tip_over + PIERCE * EY
                self._lock_to(self.lock_center_for_tip(through - self.correction), 0.006)
                self.L.hold(hold_k(0.8))
                self.when_settled(self.L, "left", self._threaded)

            def _threaded(self):
                rep = task.episode_report(view, 0)
                log(f"THREADED: tip {np.round(self.tip(), 4)} hole {np.round(self.hole(), 4)} milestone {rep['milestone']}",
                      flush=True)
                self.press_start()

            def bend_outer(self):
                pos, quat = self.body_pose()
                return pos + rotate(quat, (0.0, 0.0, PV.bend_top(self.travel())))

            def press_start(self):
                R = self.R
                R.pad_to(len(self.L.steps))
                b = self.bend_outer()
                start = b - 0.012 * EY - FINGER_END * B_FINGER_DIR
                R.move(start - 0.06 * B_FINGER_DIR, QB, min_steps=hold_k(2.5))
                R.move(start, speed=0.04)
                R.hold(hold_k(0.6))
                self.body0 = self.body_pose()[0]
                self.when_settled(R, "right", lambda: self._press(0))

            def _press(self, k):
                latched = bool(view._task_state.latched[0])
                if latched or k > 30:
                    log(f"PRESS done after {k}: travel {self.travel() * 1000:.1f} mm, latched {latched}")
                    self.R.move(self.R.pos + ROOT_OFFSET + 0.004 * EY, speed=0.02)
                    self.R.hold(hold_k(0.4))
                    self.L.pad_to(len(self.R.steps))
                    self.release()
                    return
                # Arm A holds the body against the push (its soft wrist would let the lock, and the hasp with the
                # coworker's lock on it, swing toward +y): it backs off by however far the body has moved.
                drift = float((self.body_pose()[0] - self.body0) @ EY)
                self.L.pad_to(len(self.R.steps))
                self.L.move(self.L.pos + ROOT_OFFSET - drift * EY, speed=0.02)
                self.R.move(self.R.pos + ROOT_OFFSET + 0.004 * EY, speed=0.02)
                self.R.hold(hold_k(0.2))
                self.L.pad_to(len(self.R.steps))
                self.R.pad_to(len(self.L.steps))
                self.at(len(self.R.steps), lambda: self._press(k + 1))

            def release(self):
                R, L = self.R, self.L
                R.move(R.pos + ROOT_OFFSET - 0.04 * EY, speed=0.04)
                R.move(R_WAIT, speed=0.10)
                L.pad_to(len(R.steps) - hold_k(1.2))
                L.grip_to(0.55, hold_k(1.0))
                L.move(L.pos + ROOT_OFFSET - 0.05 * A_FINGER_DIR, speed=0.03)
                L.move(L.pos + ROOT_OFFSET + turn((-0.05, 0.06, 0.04)), speed=0.06)
                L.hold(hold_k(4.0))

        l_pos, l_quat = grasp_point("left")
        r_pos, r_quat = grasp_point("right")
        L = ArmTimeline(l_pos - ROOT_OFFSET, l_quat, dt, speed)
        R = ArmTimeline(r_pos - ROOT_OFFSET, r_quat, dt, speed)
        R.grip = CLOSED
        R.move(R_WAIT, QB_WAIT, min_steps=hold_k(3.0))
        planner = Planner(L, R)
        settle = hold_k(2.0)
        L.hold(settle)
        planner.at(settle, planner.plan_pick)

        q_rel = np.array(hb.Q_REL)
        tcp_extra = 0.1347 - 0.134  # our grasp point sits this much further out than their TCP
        q_arm = {s: view.scene[f"{s}_robot"].data.joint_pos.torch[0, :6].cpu().numpy().astype(float) for s in ("left", "right")}
        worst = {"left": 0.0, "right": 0.0}

        def joint_action(side, pos_root, quat, grip):
            pos = pos_root + ROOT_OFFSET + origin  # world
            tcp = pos + rotate(quat, (0.0, 0.0, -tcp_extra))
            rot = qmul(quat, q_rel)
            q, pe, _ = ik.solve(tcp, rot, q_arm[side], base_pos=station.BASES[side], iters=60)
            q_arm[side] = q
            if pe > 0.005 and pe > worst[side]:
                log(f"IK {side} k={state_k[0]} error {pe * 1000:.1f} mm at {np.round(pos - origin, 4)}")
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
