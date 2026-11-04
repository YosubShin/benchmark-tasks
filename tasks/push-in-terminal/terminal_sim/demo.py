"""The scripted demo, closed loop: arm B (right), fingers forward and 45 deg down, closing along y, picks the wire's
ferruled end off its stand by the insulation (the end lies in the pads' plane at 30 deg to the fingers, ferrule
beyond the fingertips), lifts it (it already slopes 15 deg down like the entry's axis), carries it in front of
terminal X1:4's upper entry, measures and corrects its pose, then pushes it in along the entry axis in 1.5 mm steps,
re-measuring each step, until the task reports it seated (the collar on the mouth and the clamping leg riding on the
ferrule). It lets go and backs away; the task then tugs the wire (off camera). Arm A (left) stays at its start.

``control=True``: the same, but the push stops once the ferrule's tip is past 4.5 mm into the entry (short of the clamping
leg), lets go, and the tug is forced once the hand is clear: an under-inserted wire (it must fail the tug).

It reads the simulator's state (the wire end's pose, the task's seated flag), like any scripted demo with
privileged state; ``--policy terminal-demo`` (terminal_sim.policy) plays it through Inspect Robots. Grasp-point
targets become yam_arena's 14-D joint action through yam_arena's own IK every control step. Import after Isaac Sim
has started.
"""

from __future__ import annotations

import math

import numpy as np

from yam_arena import ik, station

PRE_U = 0.012
"""Line-up stand-off: the collar 12 mm out of the mouth (the ferrule's tip 2 mm outside)."""
PUSH_STEP = 0.0015
PUSH_PAST = 0.040
"""How far past home the push may command the wire (the collar stops on the mouth; if the wire slides back in the pads the fingers end
up against the collar and push on it; under yam_arena's soft joint
gains the commanded overshoot is what makes the push force that opens the clamping leg)."""
SHORT_TIP_U = 0.0045
"""Control run: stop the push once the ferrule's tip is this far in (the clamping edge meets it at 7 mm)."""


class Demo:
    """``next_action()`` gives the 14-D joint action for the next control step."""

    def __init__(self, STATION, control_hz: float = 20.0, speed: float = 0.10, control: bool = False, log=print):
        from terminal_sim import station as hb
        from terminal_sim.kit import (CLOSED, FORWARD, OPEN, ROOT_OFFSET, ArmTimeline, GraspModel, conj, q_axis, qmul,
                                      rotate)
        from terminal_sim.kit import grasp_point as grasp_point_world
        from terminal_sim.spec import TARGET
        from terminal_sim.usd import terminal_props as props

        view, task = STATION.env, STATION.task
        st = task.state(view)
        state_k = [0]
        self.state_k = state_k
        origin = view.scene.env_origins[0].cpu().numpy()  # desk frame origin in the world
        robot = view.scene["robot"]
        wire = st.wire
        end_id = st.end_id

        def grasp_point(side):
            p, q = grasp_point_world(robot, side)
            return p - origin, q

        for side, name in (("left", "left_robot"), ("right", "right_robot")):
            qj = view.scene[name].data.joint_pos.torch[0, :6].cpu().numpy()
            tcp, _ = ik.tcp_pose(qj, station.BASES[side])
            log(f"CHECK {side}: their TCP {np.round(tcp, 4)} our grasp point {np.round(grasp_point(side)[0] + origin, 4)}",
                flush=True)

        W_QUAT = np.array(props.W_QUAT)
        U = np.array(props.u_hat())
        V = np.array(props.v_hat())
        IDENT = np.array([0.0, 0.0, 0.0, 1.0])
        PITCHED = qmul(q_axis((0, 1, 0), math.pi / 4), FORWARD)
        FINGER_DIR = rotate(PITCHED, (0.0, 0.0, 1.0))
        GRASP_IN_WIRE = np.array([props.GRIP_X, 0.0, 0.0]) - rotate(conj(W_QUAT), 0.006 * FINGER_DIR)
        home = np.array(props.wire_aim_root(TARGET)[:3])
        dt = 1.0 / control_hz
        hold_k = lambda s: max(1, round(s / dt))  # noqa: E731

        def end_pose():
            pos = wire.data.body_pos_w.torch[0, end_id].cpu().numpy() - origin
            return pos, wire.data.body_quat_w.torch[0, end_id].cpu().numpy()

        def measure():
            """The wire end's pose relative to the right grasp point, as it is now."""
            g_pos, g_quat = grasp_point("right")
            o_pos, o_quat = end_pose()
            return GraspModel(rotate(conj(g_quat), o_pos - g_pos), qmul(conj(g_quat), o_quat))

        def u_of(p):
            d = p - home
            return float(d @ U), np.array([d[1], float(d @ V)])

        class Planner:
            def __init__(self, R):
                self.R, self.events = R, {}
                self.correction = np.zeros(3)
                self.q_corr = IDENT.copy()
                self.retries = 0
                self.bias = np.zeros(3)

            def at(self, step, fn):
                self.events.setdefault(step, []).append(fn)

            def run_due(self, k):
                for fn in self.events.pop(k, []):
                    fn()

            def when_settled(self, fn, tries=30, tol=0.003):
                R = self.R

                def check():
                    lag = np.linalg.norm(grasp_point("right")[0] - (R.pos + ROOT_OFFSET))
                    if lag < tol or check.tries <= 0:
                        if lag >= tol:
                            log(f"SETTLE timeout: lag {lag * 1000:.1f} mm")
                        fn()
                    else:
                        check.tries -= 1
                        R.hold(hold_k(0.1))
                        self.at(len(R.steps), check)

                check.tries = tries
                self.at(len(R.steps), check)

            # Pick ---------------------------------------------------------------------------------
            def plan_pick(self):
                start, _ = end_pose()
                self.grasp, self.quat = GraspModel.nominal(PITCHED, W_QUAT, GRASP_IN_WIRE).grasp_for(start, W_QUAT)
                R = self.R
                R.grip_to(OPEN, hold_k(0.2))
                R.move(self.grasp + np.array([-0.03, 0.0, 0.07]), self.quat, min_steps=hold_k(3.0))
                R.move(self.grasp + np.array([0.0, 0.0, 0.02]), speed=0.05)
                R.hold(hold_k(0.4))
                self.when_settled(self._final_approach)

            def _final_approach(self):
                R = self.R
                err = grasp_point("right")[0] - (R.pos + ROOT_OFFSET)
                R.move(self.grasp - err, speed=0.02)
                R.hold(hold_k(0.4))
                R.grip_to(CLOSED, hold_k(1.0))
                R.move(self.grasp - err + np.array([0.0, 0.0, 0.04]), speed=0.03)
                self.when_settled(self.carry)

            def _wire_to(self, root_pos, speed):
                g, q = self.model.grasp_for(root_pos, qmul(self.q_corr, W_QUAT))
                self.R.move(g, q, speed=speed)

            def carry(self):
                self.model = measure()
                log("WIRE grasp rel_pos (mm)", np.round(self.model.rel_pos * 1000, 1), flush=True)
                pre = home - PRE_U * U
                self._wire_to(pre + np.array([-0.03, 0.0, 0.03]), 0.06)
                self._line_up(0)

            def _line_up(self, attempt):
                pre = home - PRE_U * U
                self._wire_to(pre - self.correction, 0.03 if attempt == 0 else 0.01)
                self.R.hold(hold_k(0.6))
                self.when_settled(lambda: self._line_up_check(attempt), tries=10, tol=0.006)

            def _line_up_check(self, attempt):
                pre = home - PRE_U * U
                if attempt == 0:  # the wire shifts in the fingers on the way (cable drag): measure it again here
                    self.model = measure()
                    self._line_up(1)
                    return
                p, q = end_pose()
                err = p - pre
                rot_err = qmul(q, conj(W_QUAT))
                ang = 2 * math.degrees(math.acos(min(1.0, abs(rot_err[3]))))
                log(f"LINE-UP {attempt}: pos error (mm) {np.round(err * 1000, 2)}, rot error {ang:.2f} deg", flush=True)
                self.correction += err
                self.q_corr = qmul(conj(rot_err), self.q_corr)
                self.q_corr /= np.linalg.norm(self.q_corr)
                if attempt < 6 and (np.abs(err).max() > 0.0003 or ang > 0.5):
                    self._line_up(attempt + 1)
                    return
                self.push_u, self.last_tip, self.stalled = -PRE_U, None, 0
                self._push(0)

            def _push(self, k):
                p, _ = end_pose()
                u, lat = u_of(p)
                tip_u = u + props.SLEEVE_LEN
                seated = bool(st.seated_once[0])
                rep = task.episode_report(view, 0)
                leg = props.LEAF_SIGN * math.degrees(float(st.strip.data.joint_pos.torch[0, st.leaf_joints[TARGET - 1]]))
                fm = st.sensor.data.force_matrix_w
                fm = (fm.torch if hasattr(fm, "torch") else fm)[0].reshape(-1, 3).sum(0).cpu().numpy()
                lag = grasp_point("right")[0] - (self.R.pos + ROOT_OFFSET)
                log(f"PUSH {k}: root u {u * 1000:.2f} mm (tip {tip_u * 1000:.2f}), lateral y/v (mm) "
                    f"{np.round(lat * 1000, 2)}, leg {leg:.2f} deg, milestone {rep['milestone']}, "
                    f"wrong {rep['semantic_events']}, resist {-float(fm @ U):.1f} N, hand lag (mm) "
                    f"{np.round(lag * 1000, 1)}, grasp {np.round(measure().rel_pos * 1000, 1)}", flush=True)
                stop_short = control and tip_u > SHORT_TIP_U
                if seated or stop_short or k > 60:
                    log(f"PUSH done after {k} steps: root u {u * 1000:.2f} mm, seated {seated}", flush=True)
                    self.release()
                    return
                # Stalled (the tip not advancing): on the mouth's lip, back off 3 mm, re-measure and line up again,
                # aiming off the axis against the offset measured at the stall; inside the entry, ease off 1.5 mm and
                # let the steering below centre it before pushing on.
                if self.last_tip is not None and tip_u < 0.005 and tip_u - self.last_tip < 0.0002:
                    self.stalled += 1
                else:
                    self.stalled = 0
                self.last_tip = tip_u
                eased = False
                if self.stalled >= 2 and self.retries < 8:
                    self.retries += 1
                    self.stalled = 0
                    if tip_u < 0.0005:
                        log(f"RETRY {self.retries}: stalled on the mouth, lining up again", flush=True)
                        self.bias -= 0.5 * (lat[0] * np.array([0.0, 1.0, 0.0]) + lat[1] * V)
                        self._line_up(0)
                        return
                    log(f"RETRY {self.retries}: stalled {tip_u * 1000:.1f} mm in, easing off", flush=True)
                    self.push_u = u - 0.0015
                    eased = True
                # Re-measure the wire in the fingers and steer it back onto the axis until the ferrule reaches the
                # leg; then the leg presses it onto the busbar and the hand follows.
                self.model = measure()
                d = p - home
                perp = d - (d @ U) * U
                if tip_u < 0.005:
                    self.correction += 0.6 * perp
                step = 0.001 if control else (0.0008 if -0.0015 < tip_u < 0.002 else PUSH_STEP)
                # Command at most 2.5 mm ahead of where the wire is (no pile-up of force on a stuck tip).
                if not eased:
                    lead = 0.0025 if tip_u < 0.005 else 0.042  # at the leg: push harder
                    self.push_u = min(self.push_u + step, u + lead, PUSH_PAST)
                target = home + self.push_u * U
                aim = target - (self.correction - (self.correction @ U) * U) + (self.bias if tip_u < 0.0005 else 0.0)
                self._wire_to(aim, 0.01)
                self.R.hold(hold_k(0.3))
                self.at(len(self.R.steps), lambda: self._push(k + 1))

            def release(self):
                R = self.R
                # The push commanded the hand up to 2 cm past where the wire stopped; bring the command back to where
                # the hand is, so that opening the fingers does not let it lunge into the strip.
                here, _ = grasp_point("right")
                R.move(here - 0.002 * U, speed=0.02, min_steps=hold_k(0.5))
                R.hold(hold_k(0.3))
                R.grip_to(OPEN, hold_k(0.8))
                R.move(R.pos + ROOT_OFFSET - 0.04 * FINGER_DIR, speed=0.03)
                R.move(R.pos + ROOT_OFFSET + np.array([-0.08, -0.04, 0.08]), speed=0.06)
                if control:  # nothing clamped: tug it anyway once the hand is clear
                    self.at(len(R.steps) + hold_k(0.5), lambda: task.force_function_test(view))
                R.hold(hold_k(4.0))

        r_pos, r_quat = grasp_point("right")
        R = ArmTimeline(r_pos - ROOT_OFFSET, r_quat, dt, speed)
        planner = Planner(R)
        settle = hold_k(1.0)
        R.hold(settle)
        planner.at(settle, planner.plan_pick)

        q_rel = np.array(hb.Q_REL)
        tcp_extra = 0.1347 - 0.134  # our grasp point sits this much further out than their TCP
        q_arm = view.scene["right_robot"].data.joint_pos.torch[0, :6].cpu().numpy().astype(float)
        self._left = np.r_[view.scene["left_robot"].data.joint_pos.torch[0, :6].cpu().numpy().astype(float), OPEN]
        worst = [0.0]

        def joint_action(pos_root, quat, grip):
            nonlocal q_arm
            pos = pos_root + ROOT_OFFSET + origin  # world
            tcp = pos + rotate(quat, (0.0, 0.0, -tcp_extra))
            rot = qmul(quat, q_rel)
            q, pe, _ = ik.solve(tcp, rot, q_arm, base_pos=station.BASES["right"], iters=80)
            q_arm = q
            if pe > 0.003 and pe > worst[0]:
                log(f"IK right k={state_k[0]} error {pe * 1000:.1f} mm at {np.round(pos - origin, 4)}")
            worst[0] = max(worst[0], pe)
            return np.r_[q, grip]

        self._R, self._planner, self._joint_action = R, planner, joint_action

    def next_action(self) -> np.ndarray:
        k = self.state_k[0]
        R = self._R
        self._planner.run_due(k)
        if k >= len(R.steps):
            R.hold(1)
        rp, rq, rg = R.steps[k]
        self.state_k[0] = k + 1
        return np.concatenate([self._left, self._joint_action(rp, rq, rg)])
