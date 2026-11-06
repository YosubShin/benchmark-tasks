"""The scripted demo, closed loop, both arms.

Arm A (left) picks the WAGO operating tool off its stand by the rear end of its handle (fingers forward and 45 deg
down, rolled 35 deg about the tool so its wrist sits to the left), carries it in front of terminal X1:4's operating slot, measures and corrects
the blade's pose, then pushes the blade straight in (square to the front face) until the shaft is on the face and the
clamp is held open, and keeps hold of it. Before that, arm B (right), fingers forward and 45 deg down, has picked the
wire's insulated end off its stand and parked it well to the right; once the clamp is open it lines the stripped end
up with X1:4's upper entry (as in the ferrule task), pushes it in until the insulation stops in the mouth and holds it
there. A pulls the tool straight back out (the clamp closes on the strands) and lifts it away; B lets go and backs off. The task then tugs the wire
(off camera).

``control=True``: A leaves the tool on its stand; B pushes the stripped end into X1:4 just the same (the strands
bunch up against the closed clamp), lets go, and the tug is forced once the hand is clear.

It reads the simulator's state (the wire end's and the tool's poses, the leg angle), like any scripted demo with
privileged state; ``--policy terminal-tool-demo`` (terminal_tool_sim.policy) plays it through Inspect Robots.
Grasp-point targets become yam_arena's 14-D joint action through yam_arena's own IK every control step. Import after
Isaac Sim has started.
"""

from __future__ import annotations

import math

import numpy as np

from yam_arena import ik, station

PRE_U = 0.014
"""Wire line-up stand-off: the insulation 14 mm out of the mouth (the strands' tip 4 mm outside)."""
PUSH_STEP = 0.0015
PUSH_PAST = 0.030
"""How far past home the push may command the wire (the insulation stops in the mouth)."""
TOOL_PRE = 0.015
"""Tool line-up stand-off: the blade's tip 5 mm in front of the face."""
TOOL_PUSH_PAST = 0.045
TOOL_OPEN_DEG = 20.0


class Demo:
    """``next_action()`` gives the 14-D joint action for the next control step."""

    def __init__(self, STATION, control_hz: float = 20.0, speed: float = 0.10, control: bool = False, log=print):
        from terminal_tool_sim import station as hb
        from terminal_tool_sim.kit import (CLOSED, FORWARD, OPEN, ROOT_OFFSET, ArmTimeline, GraspModel, conj, q_axis,
                                           qmul, rotate)
        from terminal_tool_sim.kit import grasp_point as grasp_point_world
        from terminal_tool_sim.spec import TARGET
        from terminal_tool_sim.usd import terminal_props as props

        view, task = STATION.env, STATION.task
        st = task.state(view)
        state_k = [0]
        self.state_k = state_k
        origin = view.scene.env_origins[0].cpu().numpy()  # desk frame origin in the world
        robot = view.scene["robot"]
        wire, tool = st.wire, st.tool
        end_id = st.end_id

        def grasp_point(side):
            p, q = grasp_point_world(robot, side)
            return p - origin, q

        W_QUAT = np.array(props.W_QUAT)
        U = np.array(props.u_hat())
        V = np.array(props.v_hat())
        X = np.array([1.0, 0.0, 0.0])
        IDENT = np.array([0.0, 0.0, 0.0, 1.0])
        PITCHED = qmul(q_axis((0, 1, 0), math.pi / 4), FORWARD)
        # The tool hand grips the handle's rear end from above, behind and to the left (fingers forward and 45 deg
        # down, rolled 35 deg about the tool), so its hand and wrist stay clear of B's at the entry.
        TOOL_HAND = qmul(q_axis((1, 0, 0), math.radians(-35.0)), PITCHED)  # rolled about the tool: wrist to the left
        # The wire hand: fingers forward and 45 deg down, rolled 35 deg about the wire so its fingers and wrist sit to
        # the right, clear of the tool's shaft below the entry and of A's hand.
        WIRE_HAND = qmul(q_axis(U, math.radians(35.0)), PITCHED)
        FINGER_DIR = rotate(WIRE_HAND, (0.0, 0.0, 1.0))
        TOOL_FINGER_DIR = rotate(TOOL_HAND, (0.0, 0.0, 1.0))
        GRASP_IN_WIRE = np.array([-0.045, 0.0, 0.0]) - rotate(conj(W_QUAT), 0.010 * FINGER_DIR)  # 45 mm back: clear of the tool's shaft
        GRASP_IN_TOOL = np.array([props.TOOL_GRIP_X, 0.0, 0.0]) - 0.003 * TOOL_FINGER_DIR
        home = np.array(props.wire_aim_root(TARGET)[:3])
        tool_home = np.array(props.tool_home_tip(TARGET)[:3])
        leg_id = st.leaf_joints[TARGET - 1]
        dt = 1.0 / control_hz
        hold_k = lambda s: max(1, round(s / dt))  # noqa: E731
        flags = {"tool_lifted": False, "wire_parked": False, "tool_in": False, "wire_home": False, "tool_out": False}
        PART_OPEN = 0.45
        """Fingers part open (about 4 cm apart): enough for the wire or the handle, clear of things beside them."""

        def end_pose():
            pos = wire.data.body_pos_w.torch[0, end_id].cpu().numpy() - origin
            return pos, wire.data.body_quat_w.torch[0, end_id].cpu().numpy()

        def tool_pose():
            pos = tool.data.root_pos_w.torch[0].cpu().numpy() - origin
            return pos, tool.data.root_quat_w.torch[0].cpu().numpy()

        def leg_deg():
            return props.LEAF_SIGN * math.degrees(float(st.strip.data.joint_pos.torch[0, leg_id]))

        def measure(side, pose_fn):
            g_pos, g_quat = grasp_point(side)
            o_pos, o_quat = pose_fn()
            return GraspModel(rotate(conj(g_quat), o_pos - g_pos), qmul(conj(g_quat), o_quat))

        def u_of(p):
            d = p - home
            return float(d @ U), np.array([d[1], float(d @ V)])

        class Arm:
            """One arm's timeline and its own event queue (steps are shared: both timelines advance together)."""

            def __init__(self, side, T):
                self.side, self.T, self.events = side, T, {}

            def at(self, step, fn):
                self.events.setdefault(step, []).append(fn)

            def soon(self, fn, s=0.0):
                self.at(len(self.T.steps) + hold_k(s) if s else len(self.T.steps), fn)

            def run_due(self, k):
                while k in self.events:  # an event may schedule another at this same step
                    for fn in self.events.pop(k):
                        fn()

            def lag(self):
                return np.linalg.norm(grasp_point(self.side)[0] - (self.T.pos + ROOT_OFFSET))

            def when_settled(self, fn, tries=30, tol=0.003):
                def check():
                    lag = self.lag()
                    if lag < tol or check.tries <= 0:
                        if lag >= tol:
                            log(f"SETTLE {self.side} timeout: lag {lag * 1000:.1f} mm; at (mm) "
                                f"{np.round(grasp_point(self.side)[0] * 1000, 0)}, target "
                                f"{np.round((self.T.pos + ROOT_OFFSET) * 1000, 0)}, other hand at "
                                f"{np.round(grasp_point('right' if self.side == 'left' else 'left')[0] * 1000, 0)}")
                        fn()
                    else:
                        check.tries -= 1
                        self.T.hold(hold_k(0.1))
                        self.soon(check)

                check.tries = tries
                self.soon(check)

            def wait_for(self, cond, fn, poll=0.2):
                def check():
                    if cond():
                        fn()
                    else:
                        self.T.hold(hold_k(poll))
                        self.soon(check)

                self.soon(check)

        # -- arm A (left): the operating tool ----------------------------------------------------------------
        class ToolHand(Arm):
            def plan_pick(self):
                p, q = tool_pose()
                self.grasp, self.quat = GraspModel.nominal(TOOL_HAND, q, GRASP_IN_TOOL).grasp_for(p, q)
                T = self.T
                T.grip_to(OPEN, hold_k(0.2))
                T.move(self.grasp + np.array([-0.03, 0.0, 0.07]), self.quat, min_steps=hold_k(3.0))
                T.move(self.grasp + np.array([0.0, 0.0, 0.02]), speed=0.05)
                T.hold(hold_k(0.4))
                self.when_settled(self._final_approach)

            def _final_approach(self):
                T = self.T
                err = grasp_point("left")[0] - (T.pos + ROOT_OFFSET)
                T.move(self.grasp - err, speed=0.02)
                T.hold(hold_k(0.4))
                T.grip_to(CLOSED, hold_k(1.0))
                T.move(self.grasp - err + np.array([0.0, 0.0, 0.05]), speed=0.03)
                self.when_settled(self.carry)

            def _tool_to(self, tip_pos, speed):
                g, q = self.model.grasp_for(tip_pos, qmul(self.q_corr, IDENT))
                self.T.move(g, q, speed=speed)

            def carry(self):
                self.model = measure("left", tool_pose)
                self.correction, self.q_corr, self.retries = np.zeros(3), IDENT.copy(), 0
                log("TOOL grasp rel_pos (mm)", np.round(self.model.rel_pos * 1000, 1), flush=True)
                # Lift the tool clear and wait up and to the left while B fetches the wire (the wire's cable sweeps
                # past the tool's stand), then go to the slot.
                self._tool_to(np.array([0.26, 0.06, 0.20]), 0.05)
                self.T.hold(hold_k(0.3))
                flags["tool_lifted"] = True
                self.wait_for(lambda: flags["wire_parked"], self._to_slot)

            def _to_slot(self):
                pre = tool_home - TOOL_PRE * X
                self._tool_to(pre + np.array([-0.03, 0.0, 0.03]), 0.05)
                self._line_up(0)

            def _line_up(self, attempt):
                pre = tool_home - TOOL_PRE * X
                self._tool_to(pre - self.correction, 0.03 if attempt == 0 else 0.01)
                self.T.hold(hold_k(0.6))
                self.when_settled(lambda: self._line_up_check(attempt), tries=10, tol=0.006)

            def _line_up_check(self, attempt):
                pre = tool_home - TOOL_PRE * X
                if attempt == 0:
                    self.model = measure("left", tool_pose)
                    self._line_up(1)
                    return
                p, q = tool_pose()
                err = p - pre
                ang = 2 * math.degrees(math.acos(min(1.0, abs(q[3]))))
                log(f"TOOL LINE-UP {attempt}: pos error (mm) {np.round(err * 1000, 2)}, rot error {ang:.2f} deg",
                    flush=True)
                self.correction += err
                self.q_corr = qmul(conj(q), self.q_corr)
                self.q_corr /= np.linalg.norm(self.q_corr)
                if attempt < 6 and (np.abs(err).max() > 0.0003 or ang > 0.5):
                    self._line_up(attempt + 1)
                    return
                self.push_x, self.last_x, self.stalled = -TOOL_PRE, None, 0
                self._push(0)

            def _push(self, k):
                p, _ = tool_pose()
                x = float(p[0] - tool_home[0])  # 0 = fully in (shaft on the face); the blade enters at -BLADE_LEN
                lat = p - tool_home
                lat[0] = 0.0
                leg = leg_deg()
                log(f"TOOL PUSH {k}: tip x {x * 1000:.2f} mm (blade in {(x + props.BLADE_LEN) * 1000:.2f}), lateral "
                    f"(mm) {np.round(lat[1:] * 1000, 2)}, leg {leg:.1f} deg, hand lag "
                    f"{self.lag() * 1000:.1f} mm", flush=True)
                if x > -0.0008 and leg > TOOL_OPEN_DEG:
                    log(f"TOOL IN after {k} steps: leg {leg:.1f} deg", flush=True)
                    self._hold_in(x)
                    return
                if k > 60:
                    log("TOOL PUSH gave up", flush=True)
                    self._hold_in(x)
                    return
                blade_in = x + props.BLADE_LEN
                if (self.last_x is not None and blade_in < 0.002 and x - self.last_x < 0.0001
                        and self.push_x - x > 0.002):
                    self.stalled += 1
                else:
                    self.stalled = 0
                self.last_x = x
                if self.stalled >= 2 and self.retries < 6:  # on the slot's lip: back off, line up again
                    self.retries += 1
                    self.stalled = 0
                    log(f"TOOL RETRY {self.retries}: stalled on the mouth, lining up again", flush=True)
                    self.correction += 0.5 * lat
                    self._line_up(1)
                    return
                if blade_in < 0.002:  # steer the blade onto the slot's axis until it is inside
                    self.correction += 0.9 * lat
                    self.model = measure("left", tool_pose)
                lead = 0.0025 if blade_in < 0.002 else 0.045  # the ramp needs ~10 N: command well past it
                self.push_x = min(self.push_x + (PUSH_STEP if blade_in < 0.002 else 0.004), x + lead, TOOL_PUSH_PAST)
                self._tool_to(tool_home + self.push_x * X - self.correction + (self.correction @ X) * X, 0.01)
                self.T.hold(hold_k(0.3))
                self.soon(lambda: self._push(k + 1))

            def _hold_in(self, x):
                # Keep hold of the tool while B lands the wire. (The real tool would stay in by itself; in this
                # simulator it can slide out when the gripper opens on it.)
                T = self.T
                here, _ = grasp_point("left")
                # Command 2 cm past the face: the shaft rests on it, so this keeps the blade pressed fully home (off
                # the tab's ramp) while B works.
                T.move(here + 0.02 * X, speed=0.02, min_steps=hold_k(0.4))
                T.hold(hold_k(0.5))
                self.soon(self._held_check)

            def _held_check(self):
                p, _ = tool_pose()
                log(f"TOOL HELD IN: tip x {(p[0] - tool_home[0]) * 1000:.2f} mm, leg {leg_deg():.1f} deg, in a slot "
                    f"{bool(st.tool_in_slot[0])}", flush=True)
                flags["tool_in"] = True
                self.wait_for(lambda: flags["wire_home"], self.withdraw)

            def withdraw(self):
                here, _ = grasp_point("left")
                T = self.T
                log(f"TOOL WITHDRAW: leg {leg_deg():.1f} deg", flush=True)
                T.move(here, speed=0.02, min_steps=hold_k(0.2))
                T.move(here - 0.006 * X, speed=0.004)  # off the tab's flat and down its ramp, slowly
                T.move(T.pos + ROOT_OFFSET - 0.035 * X, speed=0.02)
                T.hold(hold_k(0.3))
                self.soon(self._out)

            def _out(self):
                p, _ = tool_pose()
                log(f"TOOL OUT: tip x {(p[0] - tool_home[0]) * 1000:.1f} mm, leg {leg_deg():.1f} deg, in a slot "
                    f"{bool(st.tool_in_slot[0])}", flush=True)
                flags["tool_out"] = True
                T = self.T
                T.move(T.pos + ROOT_OFFSET + np.array([-0.04, 0.06, 0.06]), speed=0.05)
                T.hold(hold_k(1.0))

        # -- arm B (right): the wire ------------------------------------------------------------------------
        class WireHand(Arm):
            def plan_pick(self):
                start, _ = end_pose()
                self.correction, self.q_corr, self.retries, self.bias = np.zeros(3), IDENT.copy(), 0, np.zeros(3)
                self.grasp, self.quat = GraspModel.nominal(WIRE_HAND, W_QUAT, GRASP_IN_WIRE).grasp_for(start, W_QUAT)
                T = self.T
                T.grip_to(OPEN, hold_k(0.2))
                T.move(self.grasp + np.array([-0.03, 0.0, 0.07]), self.quat, min_steps=hold_k(3.0))
                T.move(self.grasp + np.array([0.0, 0.0, 0.02]), speed=0.05)
                T.hold(hold_k(0.4))
                self.when_settled(self._final_approach)

            def _final_approach(self):
                T = self.T
                err = grasp_point("right")[0] - (T.pos + ROOT_OFFSET)
                T.move(self.grasp - err, speed=0.02)
                T.hold(hold_k(0.4))
                T.grip_to(CLOSED, hold_k(1.0))
                T.move(self.grasp - err + np.array([0.0, 0.0, 0.04]), speed=0.03)
                self.when_settled(self.carry)

            def _wire_to(self, root_pos, speed):
                g, q = self.model.grasp_for(root_pos, qmul(self.q_corr, W_QUAT))
                self.T.move(g, q, speed=speed)

            def carry(self):
                self.model = measure("right", end_pose)
                log("WIRE grasp rel_pos (mm)", np.round(self.model.rel_pos * 1000, 1), flush=True)
                pre = home - PRE_U * U
                if control:
                    self._wire_to(pre + np.array([-0.03, 0.0, 0.03]), 0.06)
                    self.T.hold(hold_k(0.3))
                    self._line_up(0)
                    return
                self._wire_to(np.array([0.17, -0.22, 0.20]), 0.06)  # parked well right of where A will work
                self.T.hold(hold_k(0.3))

                def parked():
                    flags["wire_parked"] = True
                    self.wait_for(lambda: flags["tool_in"], lambda: (
                        self._wire_to(pre + np.array([-0.03, 0.0, 0.04]), 0.04), self._line_up(0)))

                self.when_settled(parked, tries=20, tol=0.01)

            def _line_up(self, attempt):
                pre = home - PRE_U * U
                self._wire_to(pre - self.correction, 0.03 if attempt == 0 else 0.01)
                self.T.hold(hold_k(0.6))
                self.when_settled(lambda: self._line_up_check(attempt), tries=10, tol=0.006)

            def _line_up_check(self, attempt):
                pre = home - PRE_U * U
                if attempt == 0:  # the wire shifts in the fingers on the way (cable drag): measure it again here
                    self.model = measure("right", end_pose)
                    self._line_up(1)
                    return
                p, q = end_pose()
                err = p - pre
                # Only the wire's direction matters (it is round): correct the swing that turns its axis onto the entry's
                # axis and ignore any twist about it (the cable twists it in the fingers).
                u_now = rotate(q, (1.0, 0.0, 0.0))
                axis = np.cross(u_now, U)
                s_ = np.linalg.norm(axis)
                swing = math.atan2(s_, float(u_now @ U))
                rot_err = conj(q_axis(axis / s_, swing)) if s_ > 1e-9 else IDENT.copy()
                ang = math.degrees(swing)
                log(f"WIRE LINE-UP {attempt}: pos error (mm) {np.round(err * 1000, 2)}, rot error {ang:.2f} deg",
                    flush=True)
                self.correction += err
                self.q_corr = qmul(conj(rot_err), self.q_corr)
                self.q_corr /= np.linalg.norm(self.q_corr)
                if attempt < 6 and (np.abs(err).max() > 0.0003 or ang > 0.5):
                    self._line_up(attempt + 1)
                    return
                self.push_u, self.last_u, self.stalled = -PRE_U, None, 0
                self._push(0)

            def _push(self, k):
                p, _ = end_pose()
                u, lat = u_of(p)
                rep = task.episode_report(view, 0)
                log(f"WIRE PUSH {k}: insulation u {u * 1000:.2f} mm, lateral y/v (mm) {np.round(lat * 1000, 2)}, "
                    f"leg {leg_deg():.1f} deg, strands bunched {-float(wire.data.joint_pos.torch[0, st.slide_id]) * 1000:.2f}"
                    f" mm, milestone {rep['milestone']}, wrong {rep['semantic_events']}, hand lag "
                    f"{self.lag() * 1000:.1f} mm", flush=True)
                home_now = u > props.HOME_U - 0.0005
                stuck = self.last_u is not None and u - self.last_u < 0.0002 and u > -0.003
                self.stalled = self.stalled + 1 if stuck else 0
                if self.last_u is not None and u - self.last_u < 0.0002 and u <= -0.003:
                    self.mouth_stall = getattr(self, "mouth_stall", 0) + 1
                    if self.mouth_stall >= 3:  # caught on the mouth: back off 3 mm and come in again
                        self.mouth_stall = 0
                        log("WIRE caught on the mouth: backing off", flush=True)
                        self.push_u = u - 0.003
                self.last_u = u
                if home_now or self.stalled >= 4 or k > 50:
                    log(f"WIRE PUSH done after {k} steps: insulation u {u * 1000:.2f} mm, home {home_now}", flush=True)
                    self.T.hold(hold_k(0.5))
                    if control:
                        self.soon(self.release)
                    else:
                        flags["wire_home"] = True
                        self.wait_for(lambda: flags["tool_out"], lambda: (self.T.hold(hold_k(0.8)), self.release()))
                    return
                self.model = measure("right", end_pose)
                d = p - home
                perp = d - (d @ U) * U
                if u < -0.004:  # steer the strands onto the axis until they are well inside the entry
                    self.correction += 0.6 * perp
                # Bare strands catch on the mouth's lip and bunch a little; a firmer push (about 2 N) slips them in.
                lead = 0.006 if u < -0.004 else 0.012
                self.push_u = min(self.push_u + PUSH_STEP, u + lead, PUSH_PAST)
                aim = home + self.push_u * U - (self.correction - (self.correction @ U) * U)
                self._wire_to(aim, 0.01)
                self.T.hold(hold_k(0.3))
                self.soon(lambda: self._push(k + 1))

            def release(self):
                T = self.T
                here, _ = grasp_point("right")
                T.move(here - 0.002 * U, speed=0.02, min_steps=hold_k(0.5))
                T.hold(hold_k(0.3))
                T.grip_to(OPEN, hold_k(0.8))
                T.move(T.pos + ROOT_OFFSET - 0.04 * FINGER_DIR, speed=0.03)
                T.move(T.pos + ROOT_OFFSET + np.array([-0.08, -0.04, 0.08]), speed=0.06)
                if control:  # nothing clamped: tug it anyway once the hand is clear
                    self.at(len(T.steps) + hold_k(0.5), lambda: task.force_function_test(view))
                T.hold(hold_k(4.0))

        arms = {}
        for side in ("left", "right"):
            pos, quat = grasp_point(side)
            arms[side] = (ToolHand if side == "left" else WireHand)(side, ArmTimeline(pos - ROOT_OFFSET, quat, dt, speed))
            arms[side].T.hold(hold_k(1.0))
        # A lifts the tool off its stand and waits up and to the left; B picks the wire and parks it well to the
        # right; A puts the tool in and holds it there; B lands the wire; A pulls the tool out.
        if not control:
            arms["left"].at(hold_k(1.0), arms["left"].plan_pick)
        arms["right"].at(hold_k(1.0), lambda: arms["right"].wait_for(lambda: control or flags["tool_lifted"],
                                                                     arms["right"].plan_pick))

        q_rel = np.array(hb.Q_REL)
        tcp_extra = 0.1347 - 0.134  # our grasp point sits this much further out than their TCP
        q_arm = {s: view.scene[f"{s}_robot"].data.joint_pos.torch[0, :6].cpu().numpy().astype(float)
                 for s in ("left", "right")}
        worst = {"left": 0.0, "right": 0.0}

        def joint_action(side, pos_root, quat, grip):
            pos = pos_root + ROOT_OFFSET + origin  # world
            tcp = pos + rotate(quat, (0.0, 0.0, -tcp_extra))
            rot = qmul(quat, q_rel)
            q, pe, _ = ik.solve(tcp, rot, q_arm[side], base_pos=station.BASES[side], iters=80)
            if pe > 0.006:  # stuck in a poor branch: start again from where the arm actually is
                q_now = view.scene[f"{side}_robot"].data.joint_pos.torch[0, :6].cpu().numpy().astype(float)
                q2, pe2, _ = ik.solve(tcp, rot, q_now, base_pos=station.BASES[side], iters=200)
                if pe2 < pe:
                    q, pe = q2, pe2
            q_arm[side] = q
            if pe > 0.003 and pe > worst[side]:
                log(f"IK {side} k={state_k[0]} error {pe * 1000:.1f} mm at {np.round(pos - origin, 4)}")
            worst[side] = max(worst[side], pe)
            return np.r_[q, grip]

        self._arms, self._joint_action = arms, joint_action

    def next_action(self) -> np.ndarray:
        k = self.state_k[0]
        for arm in self._arms.values():
            arm.run_due(k)
        for arm in self._arms.values():
            if k >= len(arm.T.steps):
                arm.T.hold(k + 1 - len(arm.T.steps))
        self.state_k[0] = k + 1
        out = []
        for side in ("left", "right"):
            p, q, g = self._arms[side].T.steps[k]
            out.append(self._joint_action(side, p, q, g))
        return np.concatenate(out)
