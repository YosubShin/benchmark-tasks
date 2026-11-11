"""The scripted demo, closed loop.

Arm B (right) takes PUMP 1's key off the peg on that panel's side (fingers pitched 60 degrees down, closing across
the key), slides it off the peg, carries it over the lid's key slot, lines the blade's tip up with the slot from the
measured key pose, lowers it to just above the slot and lets go: the key drops into the box. Arm A (left) picks the
open padlock off its pedestal (fingers pitched 35 degrees down, gripping the body's lower rear corner), carries it so
the lid flange's edge passes between the short leg's tip and the body, lines the tip up with front hole 7 from the
measured tip position, and lowers it through both flanges; arm B, fingers closed, pushes the shackle's bend down
until it latches while arm A holds the body; both let go. Grasp-point targets become yam_arena's 14-D joint action
through yam_arena's own IK every control step.

It reads the simulator's state (the key's, padlock's and box's poses), like any scripted demo with privileged state;
``--policy lockbox-demo`` (lockbox_sim.policy) plays it through Inspect Robots. Import after Isaac Sim has started.
Under LOCKBOX_CONTROL=1 it skips the padlock (the control run).
"""

from __future__ import annotations

import math
import os

import numpy as np

from yam_arena import ik, station


class Demo:
    """``next_action()`` gives the 14-D joint action for the next control step."""

    def __init__(self, STATION, control_hz: float = 20.0, speed: float = 0.10, log=print):
        from lockbox_sim import spec as S
        from lockbox_sim import station as hb
        from lockbox_sim.kit import CLOSED, FORWARD, OPEN, ROOT_OFFSET, ArmTimeline, GraspModel, q_axis, qmul, rotate
        from lockbox_sim.kit import grasp_point as grasp_point_world
        from lockbox_sim.task import PV, lock_start_pose

        view = STATION.env
        task = STATION.task
        state_k = [0]
        self.state_k = state_k
        control = os.environ.get("LOCKBOX_CONTROL") == "1"
        origin = view.scene.env_origins[0].cpu().numpy()
        robot = view.scene["robot"]
        lock, key = view.scene["padlock"], view.scene["key"]
        body_id = lock.find_bodies("body")[0][0]
        shackle_joint = lock.find_joints("shackle_joint")[0][0]
        dt = 1.0 / control_hz
        hold_k = lambda s: max(1, round(s / dt))  # noqa: E731
        EZ = np.array([0.0, 0.0, 1.0])
        EX = np.array([1.0, 0.0, 0.0])
        EY = np.array([0.0, 1.0, 0.0])

        def grasp_point(side):
            p, q = grasp_point_world(robot, side)
            return p - origin, q

        def fwd(pitch_deg):
            return qmul(q_axis((0, 1, 0), math.radians(pitch_deg)), FORWARD)

        def fdir(pitch_deg):
            a = math.radians(pitch_deg)
            return np.array([math.cos(a), 0.0, -math.sin(a)])

        QA, FA = fwd(10), fdir(10)
        QB, FB = fwd(30), fdir(30)
        QP, FP = fwd(60), fdir(60)
        LOCK_QUAT = np.array([0.0, 0.0, 0.0, 1.0])
        GRASP_IN_LOCK = (-PV.body[0] / 2 + 0.010, 0.0, -PV.body[2] / 2 + 0.010)
        KEY_QUAT = np.array(S.KEY_QUAT)
        GRASP_IN_KEY = (0.0, 0.0, -0.039)
        HOLE = 7
        hx, hy = S.hole_xy(HOLE)
        FL_TOP = S.Z_LID_FLANGE[1]
        FINGER_END = 0.012
        KEY_OPEN = 0.35
        """Fingers part way open for the key (the key hangs 4 cm from its panel)."""
        R_WAIT = np.array([0.20, -0.22, 0.24])
        L_WAIT = np.array([0.16, 0.20, 0.22])

        def key_pose():
            return key.data.root_pos_w.torch[0].cpu().numpy() - origin, key.data.root_quat_w.torch[0].cpu().numpy()

        def key_tip():
            p, q = key_pose()
            return p + rotate(q, (0.0, 0.0, S.KEY_TIP_Z))

        def body_pose():
            pos = lock.data.body_pos_w.torch[0, body_id].cpu().numpy() - origin
            return pos, lock.data.body_quat_w.torch[0, body_id].cpu().numpy()

        def travel():
            return float(lock.data.joint_pos.torch[0, shackle_joint])

        def lock_tip():
            pos, quat = body_pose()
            return pos + rotate(quat, PV.short_tip(travel()))

        def bend_top():
            pos, quat = body_pose()
            return pos + rotate(quat, (0.0, 0.0, PV.bend_top(travel())))

        class Planner:
            def __init__(self, L, R):
                self.L, self.R, self.events = L, R, {}
                self.lock_ready = False
                self.key_done = False

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

            # Arm B: the key ---------------------------------------------------------------------------
            def key_pick(self):
                R = self.R
                kp, kq = np.array(S.key_hang_pos(S.TARGET_PANEL)), KEY_QUAT
                self.kgrasp, _ = GraspModel.nominal(QB, kq, GRASP_IN_KEY).grasp_for(kp, kq)
                R.grip_to(KEY_OPEN, hold_k(0.2))
                R.move(np.array([0.22, -0.20, 0.27]), min_steps=hold_k(1.5))
                R.move(self.kgrasp - 0.07 * FB + np.array([0.0, 0.0, 0.03]), QB, min_steps=hold_k(2.5))
                R.move(self.kgrasp - 0.03 * FB, speed=0.05)
                R.hold(hold_k(0.4))
                self.when_settled(R, "right", self._key_close)

            def _key_close(self):
                err = grasp_point("right")[0] - (self.R.pos + ROOT_OFFSET)
                self.R.move(self.kgrasp - err, speed=0.02)
                self.R.hold(hold_k(0.4))
                self.R.grip_to(CLOSED, hold_k(1.0))
                g = self.kgrasp - err
                self.at(len(self.R.steps), lambda: log(
                    f"KEY closed: grasp {np.round(grasp_point('right')[0], 4)} target {np.round(self.kgrasp, 4)} "
                    f"fingers {np.round(robot.data.joint_pos.torch[0, robot.find_joints(['right_joint7', 'right_joint8'])[0]].cpu().numpy(), 4)} "
                    f"key {np.round(key_pose()[0], 4)}"))
                # Lift the bow hole off the peg's top (2 mm play), slide it off the peg's end, then up.
                self.R.move(g + 0.0015 * EZ, speed=0.01)
                self.R.move(g + 0.0015 * EZ - 0.030 * EY, speed=0.02)
                self.R.move(g + 0.03 * EZ - 0.045 * EY, speed=0.03)
                self.when_settled(self.R, "right", self._key_carry)

            def _key_carry(self):
                self.kmodel = GraspModel.measure(robot, "right", key)
                log("KEY grasp rel_pos (mm)", np.round(self.kmodel.rel_pos * 1000, 1), "key", np.round(key_pose()[0], 4))
                self.slot_above = np.array([S.KEY_SLOT[0], S.KEY_SLOT[1], S.Z_TOP + 0.025])
                hand = grasp_point("right")[0]
                target = hand + (self.slot_above - key_tip())
                self.R.move(target + 0.04 * EZ, speed=0.06)
                self.R.move(target, speed=0.03)
                self.R.hold(hold_k(0.5))
                self.when_settled(self.R, "right", lambda: self._key_correct(self.slot_above, 0, self._key_lower))

            def _key_correct(self, tip_target, attempt, then):
                err = key_tip() - tip_target
                log(f"KEY attempt {attempt}: tip error (mm) {np.round(err * 1000, 1)}")
                if attempt < 5 and np.abs(err).max() > 0.0012:
                    self.R.move(self.R.pos + ROOT_OFFSET - err, speed=0.01)
                    self.R.hold(hold_k(0.5))
                    self.when_settled(self.R, "right", lambda: self._key_correct(tip_target, attempt + 1, then))
                    return
                then()

            def _key_lower(self):
                low = np.array([S.KEY_SLOT[0], S.KEY_SLOT[1], S.Z_TOP + 0.010])
                self.R.move(self.R.pos + ROOT_OFFSET + (low - key_tip()), speed=0.01)
                self.R.hold(hold_k(0.5))
                self.when_settled(self.R, "right", lambda: self._key_correct(low, 0, self._key_release))

            def _key_release(self):
                log(f"KEY at release: pos {np.round(key_pose()[0], 4)} quat {np.round(key_pose()[1], 3)} tip {np.round(key_tip(), 4)}")
                R = self.R
                R.grip_to(KEY_OPEN, hold_k(1.0))
                R.move(R.pos + ROOT_OFFSET - 0.05 * FB + 0.03 * EZ, speed=0.04)
                R.move(R_WAIT, speed=0.08)
                R.hold(hold_k(1.0))
                self.at(len(R.steps), self._key_check)

            def _key_check(self):
                log(f"KEY released: key at {np.round(key_pose()[0], 4)}, milestone {task.episode_report(view, 0)['milestone']}")
                self.key_done = True
                self._maybe_press()

            # Arm A: the padlock -----------------------------------------------------------------------
            def lock_pick(self):
                start = np.array(lock_start_pose()[:3])
                self.lgrasp, _ = GraspModel.nominal(QA, LOCK_QUAT, GRASP_IN_LOCK).grasp_for(start, LOCK_QUAT)
                L = self.L
                L.grip_to(OPEN, hold_k(0.2))
                pre = self.lgrasp - 0.04 * FA
                L.move(np.array([0.24, 0.22, 0.30]), min_steps=hold_k(1.5))
                L.move(np.array([pre[0] + 0.02, pre[1], 0.25]), QA, min_steps=hold_k(2.0))
                L.move(pre, speed=0.05)
                L.hold(hold_k(0.4))
                L.move(self.lgrasp - 0.015 * FA, speed=0.04)
                L.hold(hold_k(0.4))
                self.when_settled(L, "left", self._lock_close)

            def _lock_close(self):
                err = grasp_point("left")[0] - (self.L.pos + ROOT_OFFSET)
                g = self.lgrasp - err
                self.L.move(g, speed=0.02)
                self.L.hold(hold_k(0.4))
                self.L.grip_to(CLOSED, hold_k(1.0))
                self.L.move(g + 0.035 * EZ, speed=0.03)
                self.when_settled(self.L, "left", self._lock_carry)

            def _lock_carry(self):
                self.lmodel = GraspModel.measure(robot, "left", lock)
                log("LOCK grasp rel_pos (mm)", np.round(self.lmodel.rel_pos * 1000, 1))
                self.tip_entry = np.array([hx - 0.022, hy, FL_TOP + 0.0045])
                d = self.tip_entry - lock_tip()
                self.L.move(self.L.pos + ROOT_OFFSET + d - 0.03 * EX + 0.03 * EZ, speed=0.06)
                self.L.move(self.L.pos + ROOT_OFFSET + 0.03 * EX - 0.03 * EZ, speed=0.03)
                self.L.hold(hold_k(0.6))
                self.when_settled(self.L, "left", lambda: self._lock_correct(self.tip_entry, 0, self._lock_over, 0.0015))

            def _lock_correct(self, tip_target, attempt, then, tol):
                err = lock_tip() - tip_target
                log(f"LOCK attempt {attempt}: tip error (mm) {np.round(err * 1000, 1)}")
                if attempt < 5 and np.abs(err).max() > tol:
                    self.L.move(self.L.pos + ROOT_OFFSET - err, speed=0.01)
                    self.L.hold(hold_k(0.6))
                    self.when_settled(self.L, "left", lambda: self._lock_correct(tip_target, attempt + 1, then, tol))
                    return
                then()

            def _lock_over(self):
                self.tip_over = np.array([hx, hy, FL_TOP + 0.0045])
                self.L.move(self.L.pos + ROOT_OFFSET + (self.tip_over - lock_tip()), speed=0.008)
                self.L.hold(hold_k(0.6))
                self.when_settled(self.L, "left", lambda: self._lock_correct(self.tip_over, 0, self._lock_down, 0.001))

            def _lock_down(self):
                self.L.move(self.L.pos + ROOT_OFFSET - 0.016 * EZ, speed=0.006)
                self.L.hold(hold_k(0.8))
                self.when_settled(self.L, "left", self._threaded)

            def _threaded(self):
                log(f"THREADED: tip {np.round(lock_tip(), 4)} hole ({hx:.4f}, {hy:.4f}) "
                    f"milestone {task.episode_report(view, 0)['milestone']}", flush=True)
                self.lock_ready = True
                self._maybe_press()

            # Arm B presses the shackle shut ---------------------------------------------------------------
            def _maybe_press(self):
                if not (self.lock_ready and self.key_done) or getattr(self, "pressing", False):
                    return
                self.pressing = True
                R = self.R
                R.pad_to(len(self.L.steps))
                self.L.pad_to(len(R.steps))
                R.grip_to(CLOSED, hold_k(0.5))
                b = bend_top()
                start = b + 0.004 * EZ - FINGER_END * FP
                R.move(start - 0.04 * FP + 0.015 * EZ, QP, speed=0.08)
                R.move(start, speed=0.03)
                R.hold(hold_k(0.6))
                self.body0 = body_pose()[0]
                self.when_settled(R, "right", lambda: self._press(0))

            def _press(self, k):
                latched = bool(view._task_state.latched[0])
                if latched or k > 30:
                    log(f"PRESS done after {k}: travel {travel() * 1000:.1f} mm, latched {latched}")
                    self.R.move(self.R.pos + ROOT_OFFSET + 0.004 * EZ, speed=0.02)
                    self.R.hold(hold_k(0.4))
                    self.L.pad_to(len(self.R.steps))
                    self.release()
                    return
                drift = float((body_pose()[0] - self.body0) @ EZ)
                self.L.pad_to(len(self.R.steps))
                self.L.move(self.L.pos + ROOT_OFFSET - drift * EZ, speed=0.02)
                self.R.move(self.R.pos + ROOT_OFFSET - 0.004 * EZ, speed=0.02)
                self.R.hold(hold_k(0.2))
                self.L.pad_to(len(self.R.steps))
                self.R.pad_to(len(self.L.steps))
                self.at(len(self.R.steps), lambda: self._press(k + 1))

            def release(self):
                R, L = self.R, self.L
                R.move(R.pos + ROOT_OFFSET + 0.04 * EZ - 0.03 * EX, speed=0.04)
                R.move(R_WAIT, speed=0.10)
                L.pad_to(len(R.steps) - hold_k(1.2))
                L.grip_to(0.55, hold_k(1.0))
                L.move(L.pos + ROOT_OFFSET - 0.05 * FA, speed=0.03)
                L.move(L_WAIT, speed=0.06)
                L.hold(hold_k(4.0))

        l_pos, l_quat = grasp_point("left")
        r_pos, r_quat = grasp_point("right")
        L = ArmTimeline(l_pos - ROOT_OFFSET, l_quat, dt, speed)
        R = ArmTimeline(r_pos - ROOT_OFFSET, r_quat, dt, speed)
        planner = Planner(L, R)
        settle = hold_k(1.0)
        L.hold(settle)
        R.hold(settle)
        planner.at(settle, planner.key_pick)
        if control:
            planner.lock_ready = False
        else:
            planner.at(settle, planner.lock_pick)

        q_rel = np.array(hb.Q_REL)
        tcp_extra = 0.1347 - 0.134
        q_arm = {s: view.scene[f"{s}_robot"].data.joint_pos.torch[0, :6].cpu().numpy().astype(float) for s in ("left", "right")}
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

        self._L, self._R, self._planner, self._joint_action = L, R, planner, joint_action
        self.worst = worst

    def next_action(self) -> np.ndarray:
        k = self.state_k[0]
        L, R = self._L, self._R
        self._planner.run_due(k)
        L.pad_to(k + 1)
        R.pad_to(k + 1)
        (lp, lq, lg), (rp, rq, rg) = L.steps[k], R.steps[k]
        self.state_k[0] = k + 1
        return np.concatenate([self._joint_action("left", lp, lq, lg), self._joint_action("right", rp, rq, rg)])
