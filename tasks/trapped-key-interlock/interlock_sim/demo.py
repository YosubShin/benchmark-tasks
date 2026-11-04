"""The scripted demo, closed loop. The arm on the named pair's side (left for A1, right for B1), fingers down,
pinches the named key's bow in its breaker's interlock, turns it +90 deg about its axis (the plug drives the bolt
out in front of the breaker handle), lifts it out (the bit now passes the notch), carries it to the named door's
lock with the bit toward that lock's notch, measures the key's tip and yaw over the keyway and corrects, inserts
it, turns it -90 deg (the door's bolt withdraws and the key is trapped), lets go and backs off. The other arm stays
near its start.

Grasp-point targets become yam_arena's 14-D joint action through yam_arena's own IK (``yam_arena.ik``) every
control step. It reads the simulator's state (the key's pose, the locks' joints), like any scripted demo with
privileged state; ``--policy interlock-demo`` (interlock_sim.policy) plays it through Inspect Robots. Import after
Isaac Sim has started. ``INTERLOCK_DEMO_DEBUG=<n>`` prints the state every n steps.
"""

from __future__ import annotations

import math
import os

import numpy as np

from yam_arena import ik, station


class Demo:
    """``next_action()`` gives the 14-D joint action for the next control step."""

    def __init__(self, STATION, control_hz: float = 20.0, speed: float = 0.10, log=None):
        log = log or (lambda *a: print(*a, flush=True))
        from interlock_sim import station as hb
        from interlock_sim.kit import CLOSED, DOWN, OPEN, ROOT_OFFSET, ArmTimeline, GraspModel, q_axis, qmul, rotate
        from interlock_sim.kit import grasp_point as grasp_point_world
        from interlock_sim.task import breaker_lock, door_lock
        from interlock_sim.usd import props

        view, task = STATION.env, STATION.task
        st = task.state(view)
        tgt = st.tgt
        side = "left" if tgt == "A1" else "right"
        rest = "right" if side == "left" else "left"
        sgn = 1.0 if side == "left" else -1.0
        origin = view.scene.env_origins[0].cpu().numpy()  # desk frame origin in the world
        robot = view.scene["robot"]
        key = view.scene[f"key_{tgt}"]
        dt = 1.0 / control_hz
        hold_k = lambda s: max(1, round(s / dt))  # noqa: E731
        self.debug_every = int(os.environ.get("INTERLOCK_DEMO_DEBUG", "0"))
        self.log = log

        MM = 0.001
        GRASP_IN_KEY = np.array(props.GRASP_K) * MM
        TIP_HOME = props.KEY_TIP_IN_L * MM
        CARRY_Z = 0.20
        BRK_LOCK, DOOR_LOCK = np.array(breaker_lock(tgt)), np.array(door_lock(tgt))
        # The right arm holds the bow with its hand turned 180 deg (same pinch), which keeps its wrist roll
        # inside +-120 deg through both quarter turns.
        HAND_YAW = 0.0 if side == "left" else 180.0

        def grasp_point(s):
            p, q = grasp_point_world(robot, s)
            return p - origin, q

        for s, name in (("left", "left_robot"), ("right", "right_robot")):
            qj = view.scene[name].data.joint_pos.torch[0, :6].cpu().numpy()
            tcp, _ = ik.tcp_pose(qj, station.BASES[s])
            log(f"CHECK {s}: their TCP {np.round(tcp, 4)} our grasp point {np.round(grasp_point(s)[0] + origin, 4)}")

        def yaw_q(deg):
            return q_axis((0, 0, 1), math.radians(deg))

        def joints(js):
            return {k: float(js.q(k)[0]) for k in js.ids}

        class Planner:
            def __init__(self, A, P):
                self.A, self.P, self.events, self.model = A, P, {}, None

            def at(self, step, fn):
                self.events.setdefault(step, []).append(fn)

            def run_due(self, k):
                for fn in self.events.pop(k, []):
                    fn()

            def when_settled(self, tl, fn, tries=30, tol=0.002):
                def check():
                    lag = np.linalg.norm(grasp_point(side)[0] - (tl.pos + ROOT_OFFSET))
                    if lag < tol or check.tries <= 0:
                        if lag >= tol:
                            log(f"SETTLE timeout: lag {lag * 1000:.1f} mm")
                        fn()
                    else:
                        check.tries -= 1
                        tl.hold(hold_k(0.1))
                        self.at(len(tl.steps), check)
                check.tries = tries
                self.at(len(tl.steps), check)

            def key_pose(self):
                return (key.data.root_pos_w.torch[0].cpu().numpy() - origin, key.data.root_quat_w.torch[0].cpu().numpy())

            def key_yaw(self):
                q = self.key_pose()[1]
                return math.degrees(math.atan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)))

            def _key_to(self, pos, yaw_deg, speed):
                g, q = self.model.grasp_for(np.asarray(pos), yaw_q(yaw_deg))
                self.A.move(g, q, speed=speed)

            def _turn(self, pos, yaw0, yaw1, seconds):
                n = hold_k(seconds)
                for i in range(1, n + 1):
                    yaw = yaw0 + (yaw1 - yaw0) * i / n
                    self.A.path([self.model.grasp_for(np.asarray(pos), yaw_q(yaw))])

            def plan_pick(self):
                kp, kq = self.key_pose()
                hand = qmul(yaw_q(HAND_YAW), DOWN)
                grasp, gq = GraspModel.nominal(hand, yaw_q(0.0), GRASP_IN_KEY).grasp_for(kp, kq)
                A = self.A
                A.grip_to(OPEN, hold_k(0.2))
                A.move(grasp + np.array([0.0, 0.0, 0.07]), gq, min_steps=hold_k(3.0))
                A.move(grasp + np.array([0.0, 0.0, 0.015]), speed=0.04)
                self.when_settled(A, lambda: self._final_approach(grasp))

            def _final_approach(self, grasp):
                err = grasp_point(side)[0] - (self.A.pos + ROOT_OFFSET)
                self.A.move(grasp - np.array([err[0], err[1], 0.0]), speed=0.02)
                self.A.hold(hold_k(0.4))
                self.A.grip_to(CLOSED, hold_k(1.2))
                self.when_settled(self.A, self.turn_breaker)

            def turn_breaker(self):
                self.model = GraspModel.measure(robot, side, key)
                kp, _ = self.key_pose()
                log(f"GRASP rel_pos (mm) {np.round(self.model.rel_pos * 1000, 1)} key {np.round(kp, 4)} "
                    f"yaw {self.key_yaw():.1f}")
                self.home_pos = kp.copy()
                self._turn(kp, 0.0, 91.0, 4.0)
                self.A.hold(hold_k(0.6))
                self.when_settled(self.A, self.pull_out, tol=0.003)

            def pull_out(self):
                j = joints(st.brk[tgt])
                log(f"TURNED breaker: plug {math.degrees(j['plug_joint']):.1f} deg, bolt {-j['bolt_joint'] * 1000:.1f} mm, "
                    f"key yaw {self.key_yaw():.1f}")
                self.model = GraspModel.measure(robot, side, key)
                self._key_to(self.home_pos + np.array([0.0, 0.0, 0.03]), 90.0, 0.02)
                self.A.hold(hold_k(0.4))
                self.when_settled(self.A, self.carry)

            def carry(self):
                kp, _ = self.key_pose()
                log(f"OUT: key tip {np.round(kp, 4)} (breaker face z {BRK_LOCK[2]:.4f}), "
                    f"milestone {task.episode_report(view, 0)['milestone']}")
                self.model = GraspModel.measure(robot, side, key)
                g_z = CARRY_Z - GRASP_IN_KEY[2]
                tx, ty, tz = DOOR_LOCK
                self._key_to(np.array([kp[0], kp[1], g_z]), 90.0, 0.06)
                self._key_to(np.array([tx, ty, g_z]), 0.0, 0.08)
                self.entry = np.array([tx, ty, tz + 0.006])
                self.yaw_c = 0.0
                self._key_to(self.entry, 0.0, 0.04)
                self.A.hold(hold_k(0.6))
                self.when_settled(self.A, lambda: self._entry_correct(np.zeros(3), 0))

            def _align(self, target, correction, attempt, tol, then):
                """Put the key's tip on ``target`` (x, y) and its yaw at 0, from the measured pose."""
                kp, _ = self.key_pose()
                err = kp - target
                yerr = self.key_yaw()
                log(f"ALIGN {target[2] - DOOR_LOCK[2]:+.3f} attempt {attempt}: tip error (mm) {np.round(err * 1000, 2)}, "
                    f"key yaw {yerr:.1f}")
                correction = correction + np.array([err[0], err[1], 0.0])
                self.yaw_c += yerr
                if attempt < 8 and (max(abs(err[0]), abs(err[1])) > tol or abs(yerr) > 1.0):
                    self._key_to(target - correction, -self.yaw_c, 0.005)
                    self.A.hold(hold_k(0.6))
                    self.when_settled(self.A, lambda: self._align(target, correction, attempt + 1, tol, then))
                    return
                then(correction)

            def _entry_correct(self, correction, attempt):
                self._align(self.entry, correction, attempt, 0.0004, self._into_hole)

            def _into_hole(self, correction):
                # Tip 6 mm into the face's 9 mm hole, 2 mm above the plug's socket: align again, then seat.
                self.in_hole = np.array([DOOR_LOCK[0], DOOR_LOCK[1], DOOR_LOCK[2] - 0.006])
                self._key_to(self.in_hole - correction, -self.yaw_c, 0.006)
                self.A.hold(hold_k(0.6))
                self.when_settled(self.A, lambda: self._align(self.in_hole, correction, 0, 0.00015, self._seat))

            def _seat(self, correction):
                self.correction = correction
                home = np.array([DOOR_LOCK[0], DOOR_LOCK[1], DOOR_LOCK[2] + TIP_HOME])
                self._key_to(home - correction + np.array([0.0, 0.0, -0.002]), -self.yaw_c, 0.004)
                self.A.hold(hold_k(0.8))
                self.when_settled(self.A, self.turn_door, tol=0.003)

            def turn_door(self):
                kp, _ = self.key_pose()
                depth = DOOR_LOCK[2] - kp[2]
                log(f"INSERTED: tip {np.round(kp, 4)}, depth below face {depth * 1000:.1f} mm, "
                    f"yaw {self.key_yaw():.1f}")
                self.seat_tries = getattr(self, "seat_tries", 0) + 1
                if depth < -TIP_HOME - 0.002 and self.seat_tries <= 3:
                    log("NOT SEATED: lift and align again")
                    self._into_hole(self.correction)
                    return
                self.model = GraspModel.measure(robot, side, key)
                self._turn(kp, self.key_yaw(), -93.0, 4.0)
                self.A.hold(hold_k(0.6))
                self.when_settled(self.A, self.release, tol=0.003)

            def release(self):
                j = joints(st.door[tgt])
                rep = task.episode_report(view, 0)
                log(f"TURNED door: plug {math.degrees(j['plug_joint']):.1f} deg, bolt {-j['bolt_joint'] * 1000:.1f} mm, "
                    f"key yaw {self.key_yaw():.1f}, milestone {rep['milestone']}")
                A = self.A
                A.grip_to(OPEN, hold_k(1.0))
                A.move(A.pos + ROOT_OFFSET + np.array([0.0, 0.0, 0.08]), speed=0.04)
                A.move(np.array([0.15, sgn * 0.30, 0.28]), qmul(yaw_q(HAND_YAW), DOWN), speed=0.10)
                A.hold(hold_k(6.0))

        a_pos, a_quat = grasp_point(side)
        p_pos, p_quat = grasp_point(rest)
        A = ArmTimeline(a_pos - ROOT_OFFSET, a_quat, dt, speed)
        P = ArmTimeline(p_pos - ROOT_OFFSET, p_quat, dt, speed)
        P.grip = CLOSED
        P.hold(hold_k(1.0))
        planner = Planner(A, P)
        settle = hold_k(1.0)
        A.hold(settle)
        planner.at(settle, planner.plan_pick)

        q_rel = np.array(hb.Q_REL)
        tcp_extra = 0.1347 - 0.134  # our grasp point sits this much further out than their TCP
        q_arm = {s: view.scene[f"{s}_robot"].data.joint_pos.torch[0, :6].cpu().numpy().astype(float)
                 for s in ("left", "right")}
        worst = {"left": 0.0, "right": 0.0}
        state_k = [0]

        def joint_action(s, pos_root, quat, grip):
            pos = pos_root + ROOT_OFFSET + origin  # world
            tcp = pos + rotate(quat, (0.0, 0.0, -tcp_extra))
            q, pe, _ = ik.solve(tcp, qmul(quat, q_rel), q_arm[s], base_pos=station.BASES[s], iters=60)
            q_arm[s] = q
            if pe > 0.005 and pe > worst[s]:
                log(f"IK {s} k={state_k[0]} error {pe * 1000:.1f} mm at {np.round(pos - origin, 4)}")
            worst[s] = max(worst[s], pe)
            return np.r_[q, grip]

        self._A, self._P, self._planner, self._joint_action = A, P, planner, joint_action
        self._side, self._rest, self._state_k, self.worst = side, rest, state_k, worst
        self._st, self._task, self._view = st, task, view

    def next_action(self) -> np.ndarray:
        k = self._state_k[0]
        A, P = self._A, self._P
        self._planner.run_due(k)
        A.pad_to(len(P.steps))
        P.pad_to(len(A.steps))
        if k >= len(A.steps):
            A.hold(1)
            P.hold(1)
        (ap, aq, ag), (pp, pq, pg) = A.steps[k], P.steps[k]
        self._state_k[0] = k + 1
        acts = {self._side: self._joint_action(self._side, ap, aq, ag),
                self._rest: self._joint_action(self._rest, pp, pq, pg)}
        if self.debug_every and k % self.debug_every == 0:
            rep = self._task.episode_report(self._view, 0)
            kp, _ = self._planner.key_pose()
            self.log(f"DBG k={k} ms={rep['milestone']} key={np.round(kp, 4)} yaw={self._planner.key_yaw():.1f} "
                     f"brk_bolt={rep['breaker_bolt_mm']} door_bolt={rep['door_bolt_mm']} "
                     f"handles={rep['breaker_handle_deg']} phase={rep['function_test_phase']} "
                     f"ev={rep['semantic_events']} {rep['contact_events']}")
        return np.concatenate([acts["left"], acts["right"]])
