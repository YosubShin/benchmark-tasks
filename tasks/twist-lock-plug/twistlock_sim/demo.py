"""The scripted demo, closed loop: arm B (right), fingers level and pointing forward, closing across the cap's 57.5 mm
body, picks the cap off its cradle, carries it in front of the CKT 7 receptacle with its axis on the receptacle's axis
and its blades in the entry orientation, measures and corrects its pose, pushes it home in 2 mm steps (re-measuring
the cap in the fingers each step), then rolls the wrist clockwise in 5 degree steps until the blades stop at the
slots' ends. It lets go and backs away; the task then pulls the cap (off camera). Arm A (left) stays at its start.

``control=True``: the same, but the cap is not turned: pushed home, let go, and the pull test is forced once the hand
is clear (an unturned cap must come out).

It reads the simulator's state (the cap's pose), like any scripted demo with privileged state; ``--policy
twistlock-demo`` (twistlock_sim.policy) plays it through Inspect Robots. Grasp-point targets become yam_arena's 14-D
joint action through yam_arena's own IK every control step. Import after Isaac Sim has started.
"""

from __future__ import annotations

import math

import numpy as np

from yam_arena import ik, station

PRE_GAP = 0.006
"""Line-up stand-off: blade tips 6 mm in front of the face."""
PUSH_STEP = 0.002
PUSH_LEAD = 0.008
"""Command the cap at most this far ahead of where it is (the soft arm gains turn the lead into push force)."""
PUSH_PAST = 0.003
TURN_STEP_DEG = 3.0
TURN_MAX_DEG = 46.0


class Demo:
    """``next_action()`` gives the 14-D joint action for the next control step."""

    def __init__(self, STATION, control_hz: float = 20.0, speed: float = 0.10, control: bool = False, log=print):
        from twistlock_sim import station as hb
        from twistlock_sim.kit import (CLOSED, FORWARD, OPEN, ROOT_OFFSET, ArmTimeline, GraspModel, conj, q_axis, qmul,
                                       rotate)
        from twistlock_sim.kit import grasp_point as grasp_point_world
        from twistlock_sim.usd import twistlock_props as props

        view, task = STATION.env, STATION.task
        st = task.state(view)
        state_k = [0]
        self.state_k = state_k
        origin = view.scene.env_origins[0].cpu().numpy()
        robot = view.scene["robot"]
        dt = 1.0 / control_hz
        hold_k = lambda s: max(1, round(s / dt))  # noqa: E731
        home = np.array(props.rec_root("target"))
        IDENT = np.array([0.0, 0.0, 0.0, 1.0])
        GRASP_IN_PLUG = np.array([props.GRIP_X, 0.0, 0.0])

        def x_rot(deg):
            return q_axis((1.0, 0.0, 0.0), math.radians(deg))

        def grasp_point(side):
            p, q = grasp_point_world(robot, side)
            return p - origin, q

        def cap_pose():
            pos = st.cord.data.body_pos_w.torch[0, st.cap_id].cpu().numpy() - origin
            return pos, st.cord.data.body_quat_w.torch[0, st.cap_id].cpu().numpy()

        def measure():
            g_pos, g_quat = grasp_point("right")
            o_pos, o_quat = cap_pose()
            return GraspModel(rotate(conj(g_quat), o_pos - g_pos), qmul(conj(g_quat), o_quat))

        def cap():
            p, q = cap_pose()
            ex, ey = rotate(q, (1.0, 0.0, 0.0)), rotate(q, (0.0, 1.0, 0.0))
            tilt = float(np.degrees(np.arccos(np.clip(ex[0], -1.0, 1.0))))
            phi = float(np.degrees(math.atan2(ey[2], ey[1])))
            return home[0] - p[0], (p - home)[1:], tilt, phi

        def pre():
            return home - np.array([props.BLADE_L + PRE_GAP, 0.0, 0.0])

        class Planner:
            def __init__(self, R):
                self.R, self.events = R, {}
                self.correction = np.zeros(3)
                self.q_corr = IDENT.copy()
                self.phi_cmd = 0.0

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
                            log(f"SETTLE timeout: lag {lag * 1000:.1f} mm", flush=True)
                        fn()
                    else:
                        check.tries -= 1
                        R.hold(hold_k(0.1))
                        self.at(len(R.steps), check)

                check.tries = tries
                self.at(len(R.steps), check)

            # Pick ---------------------------------------------------------------------------------
            def plan_pick(self):
                start, q0 = cap_pose()
                self.grasp, self.quat = GraspModel.nominal(FORWARD, q0, GRASP_IN_PLUG).grasp_for(start, q0)
                R = self.R
                R.grip_to(OPEN, hold_k(0.2))
                # Up out from behind the plate (the ready pose holds the right hand over the plate's box), then over
                # the cap and down onto it from straight above (level fingers cannot reach in from behind this close
                # to the arms).
                R.move(np.array([0.31, -0.29, 0.30]), self.quat, min_steps=hold_k(1.5))
                R.move(np.array([0.24, -0.15, 0.30]), min_steps=hold_k(1.0))
                R.move(self.grasp + np.array([0.0, 0.0, 0.08]), min_steps=hold_k(1.5))
                R.hold(hold_k(0.4))
                self.when_settled(self._final_approach)

            def _final_approach(self):
                R = self.R
                err = grasp_point("right")[0] - (R.pos + ROOT_OFFSET)
                R.move(self.grasp - err * np.array([1.0, 1.0, 0.0]), speed=0.03)
                R.hold(hold_k(0.4))
                self.when_settled(self._close)

            def _close(self):
                R = self.R
                R.grip_to(CLOSED, hold_k(1.0))
                R.move(R.pos + ROOT_OFFSET + np.array([0.0, 0.0, 0.03]), speed=0.03)
                self.when_settled(self.carry)

            def _cap_to(self, root_pos, phi_deg, speed):
                q_obj = qmul(self.q_corr, x_rot(phi_deg))
                g, q = self.model.grasp_for(root_pos, q_obj)
                self.R.move(g, q, speed=speed)

            def carry(self):
                self.model = measure()
                log("CAP grasp rel_pos (mm)", np.round(self.model.rel_pos * 1000, 1), flush=True)
                p, _ = cap_pose()
                self._cap_to(np.array([p[0] - 0.02, p[1], home[2] + 0.02]), 0.0, 0.05)
                self._cap_to(pre() + np.array([-0.03, 0.0, 0.02]), 0.0, 0.06)
                self._line_up(0)

            def _line_up(self, attempt):
                self._cap_to(pre() - self.correction, self.phi_cmd, 0.03 if attempt == 0 else 0.01)
                self.R.hold(hold_k(0.6))
                self.when_settled(lambda: self._line_up_check(attempt), tries=10, tol=0.006)

            def _line_up_check(self, attempt):
                if attempt == 0:  # the cap shifts in the fingers on the way (cord drag): measure it again here
                    self.model = measure()
                    self._line_up(1)
                    return
                p, q = cap_pose()
                err = p - pre()
                rot_err = qmul(q, conj(x_rot(self.phi_cmd)))
                ang = 2 * math.degrees(math.acos(min(1.0, abs(rot_err[3]))))
                s, lat, tilt, phi = cap()
                log(f"LINE-UP {attempt}: pos error (mm) {np.round(err * 1000, 2)}, rot error {ang:.2f} deg, tilt "
                    f"{tilt:.2f}, phi {phi:.2f}", flush=True)
                self.correction += err
                self.q_corr = qmul(conj(rot_err), self.q_corr)
                self.q_corr /= np.linalg.norm(self.q_corr)
                if attempt < 7 and (np.abs(err[1:]).max() > 0.0003 or ang > 0.5):
                    self._line_up(attempt + 1)
                    return
                self.s_cmd = props.BLADE_L + PRE_GAP
                self.last_s, self.stalled, self.retries = None, 0, 0
                self.last_home_s, self.home_stalls = None, 0
                self._push(0)

            def _push(self, k):
                s, lat, tilt, phi = cap()
                rep = task.episode_report(view, 0)
                lag = grasp_point("right")[0] - (self.R.pos + ROOT_OFFSET)
                log(f"PUSH {k}: s {s * 1000:.2f} mm, lateral (mm) {np.round(lat * 1000, 2)}, tilt {tilt:.2f}, phi "
                    f"{phi:.2f}, milestone {rep['milestone']}, events {rep['semantic_events']}, hand lag (mm) "
                    f"{np.round(lag * 1000, 1)}", flush=True)
                near_stall = s < 0.0015 and self.last_home_s is not None and self.last_home_s - s < 0.0001
                self.home_stalls = self.home_stalls + 1 if near_stall else 0
                self.last_home_s = s
                if s < 0.0010 or self.home_stalls >= 3 or k > 60:
                    log(f"PUSH done after {k} steps: s {s * 1000:.2f} mm", flush=True)
                    if control:
                        self.release()
                    else:
                        self._turn(0)
                    return
                # Stalled with the blade tips on the face (off the slots by a fraction of a millimetre or a degree):
                # back off 3 mm, take out the measured offset and turn, and push again.
                if self.last_s is not None and s > props.BLADE_L - 0.002 and self.last_s - s < 0.0002:
                    self.stalled += 1
                else:
                    self.stalled = 0
                self.last_s = s
                if self.stalled >= 2 and self.retries < 6:
                    self.retries += 1
                    self.stalled, self.last_s = 0, None
                    log(f"RETRY {self.retries}: stalled at the face, backing off", flush=True)
                    p, q = cap_pose()
                    self.correction[1:] += (p - home)[1:]
                    rot_err = qmul(q, conj(x_rot(self.phi_cmd)))
                    self.q_corr = qmul(conj(rot_err), self.q_corr)
                    self.q_corr /= np.linalg.norm(self.q_corr)
                    self.s_cmd = props.BLADE_L + 0.003
                    target = home - np.array([self.s_cmd, 0.0, 0.0])
                    self._cap_to(target - self.correction * np.array([0.0, 1.0, 1.0]), self.phi_cmd, 0.01)
                    self.R.hold(hold_k(0.6))
                    self.when_settled(lambda: self._push(k + 1), tries=10, tol=0.006)
                    return
                self.model = measure()
                p, _ = cap_pose()
                if s > props.BLADE_L - 0.002:  # steer onto the axis until the blades are in the slots
                    self.correction[1:] += 0.6 * (p - home)[1:]
                lead = PUSH_LEAD
                self.s_cmd = max(min(self.s_cmd - PUSH_STEP, s - 0.0005), s - lead, -PUSH_PAST)
                target = home - np.array([self.s_cmd, 0.0, 0.0])
                self._cap_to(target - self.correction * np.array([0.0, 1.0, 1.0]), self.phi_cmd, 0.01)
                self.R.hold(hold_k(0.25))
                self.at(len(self.R.steps), lambda: self._push(k + 1))

            def _turn(self, k):
                s, lat, tilt, phi = cap()
                rep = task.episode_report(view, 0)
                springs = st.panel.data.joint_pos.torch[0, st.contact_joints[:3]].cpu().numpy() * 1000
                log(f"TURN {k}: cmd {self.phi_cmd:.1f}, phi {phi:.2f} deg, s {s * 1000:.2f} mm, tilt {tilt:.2f}, "
                    f"locked {rep['locked']}, milestone {rep['milestone']}, contact springs pushed in (mm) "
                    f"{np.round(-springs, 2)}", flush=True)
                if self.phi_cmd >= TURN_MAX_DEG or (phi > props.LOCK_DEG - 1.5 and k > 0):
                    log(f"TURN done: phi {phi:.2f} deg", flush=True)
                    self.release()
                    return
                # Re-measure the cap in the fingers and keep it centred in its slots while the wrist rolls (the hand
                # orbits the cap's axis, and a sideways load would press the blades into the slot walls).
                self.model = measure()
                p, _ = cap_pose()
                self.correction[1:] += 0.5 * (p - home)[1:]
                self.phi_cmd = min(self.phi_cmd + TURN_STEP_DEG, TURN_MAX_DEG)
                target = home + np.array([0.0005, 0.0, 0.0])  # keep it pressed home while it turns
                self._cap_to(target - self.correction * np.array([0.0, 1.0, 1.0]), self.phi_cmd, 0.01)
                self.R.hold(hold_k(0.25))
                self.at(len(self.R.steps), lambda: self._turn(k + 1))

            def release(self):
                R = self.R
                # Bring the command back to where the hand is, so opening the fingers does not let it lunge.
                here, q_here = grasp_point("right")
                R.move(here - np.array([0.001, 0.0, 0.0]), q_here, speed=0.02, min_steps=hold_k(0.5))
                R.hold(hold_k(0.3))
                R.grip_to(OPEN, hold_k(0.8))
                R.move(R.pos + ROOT_OFFSET + np.array([-0.06, 0.0, 0.0]), speed=0.03)
                R.move(R.pos + ROOT_OFFSET + np.array([0.0, 0.0, 0.06]), speed=0.05)
                if control:
                    self.at(len(R.steps) + hold_k(0.5), lambda: task.force_function_test(view))
                R.hold(hold_k(4.0))

        r_pos, r_quat = grasp_point("right")
        R = ArmTimeline(r_pos - ROOT_OFFSET, r_quat, dt, speed)
        planner = Planner(R)
        settle = hold_k(1.0)
        R.hold(settle)
        planner.at(settle, planner.plan_pick)

        q_rel = np.array(hb.Q_REL)
        tcp_extra = 0.1347 - 0.134
        q_arm = view.scene["right_robot"].data.joint_pos.torch[0, :6].cpu().numpy().astype(float)
        self._left = np.r_[view.scene["left_robot"].data.joint_pos.torch[0, :6].cpu().numpy().astype(float), OPEN]
        worst = [0.0]

        def joint_action(pos_root, quat, grip):
            nonlocal q_arm
            pos = pos_root + ROOT_OFFSET + origin
            tcp = pos + rotate(quat, (0.0, 0.0, -tcp_extra))
            rot = qmul(quat, q_rel)
            q, pe, _ = ik.solve(tcp, rot, q_arm, base_pos=station.BASES["right"], iters=80)
            q_arm = q
            if pe > 0.003 and pe > worst[0]:
                log(f"IK right k={state_k[0]} error {pe * 1000:.1f} mm at {np.round(pos - origin, 4)}", flush=True)
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
