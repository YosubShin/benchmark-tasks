"""The scripted demo, closed loop.

Arm A (left) grips the FEEDER 2 unit's bail cross-bar from the front (fingers 45 deg down, closing on the bar's top
and bottom) and lifts the handle through 90 deg, following the handle's measured angle and the unit's measured
position: the latch releases and the cam pushes the unit ~7 mm out. Arm A then pulls the unit out by the handle
until 12 cm of it shows in front of the panel. Arm B (right) slides its closed fingers sideways under the body there
and rises until the body lifts 1 mm; both arms then withdraw the unit together until it is clear of the case, hold it
in the air for 3 s, lower it onto the mat (B's fingers drop into the mat's gap), B slides out sideways, A lets go and
backs away. The task then runs its after-release check (off camera).

Load: yam_arena drives the arms with i2rt's soft joint gains (wrist 10 N m/rad) and compensates only the arms' own
weight. Carrying the 2.1 kg unit takes 2-5 N m at the wrists, which would sag them by 10-30 deg. Like a person
leaning into a load, the demo adds an integral term on each arm's joint error (joint target = IK solution + the
integral, bounded) while it carries; with no load the integral stays near zero.

``control=True``: both arms stay at the ready pose; at 2 s the task starts its control sequence (pull the unit with
its handle down, let go, lift the handle by a torque, pull again).

It reads the simulator's state (handle angle, unit pose), like any scripted demo with privileged state;
``--policy relay-demo`` (relay_sim.policy) plays it through Inspect Robots. Grasp-point targets become yam_arena's
14-D joint action through yam_arena's own IK every control step. Import after Isaac Sim has started.
"""

from __future__ import annotations

import math

import numpy as np

from yam_arena import ik, station

A_OPEN = 0.30
"""Arm A opens only ~28 mm around the 12 mm bar: wide open, its upper finger would hit the relay's front."""
A_BACK = 0.004
"""Arm A's grasp point sits this far short of the bar's centre along the fingers (the tips stop short of the front)."""
LIFT_DEG_PER_S = 18.0
LIFT_LEAD_DEG = 12.0
"""The lift's commanded angle runs at most this far ahead of the handle's measured angle."""
LIFTED_DEG = 84.0
S_REST = 0.0005
"""A seated unit rests with its frame against the panel, 0.5 mm in front of the nominal seated position."""
S_SUPPORT = 0.12
"""Arm A alone pulls the unit this far out before arm B goes under the body (B reaches no farther than x ~ 0.30)."""
S_OUT = 0.17
"""Withdrawal at which the unit is held, then lowered (body 2 cm clear of the case)."""
PULL_SPEED = 0.02
PULL_LEAD = 0.03
B_FROM_FRONT = 0.035
B_DEPTH = 0.022
"""Arm B's grasp point this far in from the body's -y side, B_FROM_FRONT behind its front face."""
B_UNDER = 0.018
B_RISE_PER_STEP = 0.001
B_LIFT_MM = 1.0
B_HOLD_UP = 0.003
HOLD_S = 3.0
LOWER_SPEED = 0.015
KI = 0.08
"""Integral gain per control step on the joint error while carrying."""
I_MAX = np.array([0.20, 0.20, 0.20, 0.60, 0.60, 0.60])
"""Bound on the integral term per joint (rad)."""


class Demo:
    """``next_action()`` gives the 14-D joint action for the next control step."""

    def __init__(self, STATION, control_hz: float = 20.0, control: bool = False, log=print):
        from relay_sim import station as hb
        from relay_sim.kit import CLOSED, FORWARD, OPEN, grasp_point, q_axis, qmul, rotate, slerp
        from relay_sim.usd import relay_props as props

        view, task = STATION.env, STATION.task
        st = task.state(view)
        origins = view.scene.env_origins
        origin = origins[0].cpu().numpy()
        robot = view.scene["robot"]
        unit = st.units[st.named]
        slot = st.named
        dt = 1.0 / control_hz
        self.finished = False
        self.log = log
        self.metrics: dict = {}
        steps = lambda s: max(1, round(s / dt))  # noqa: E731

        G0 = qmul(q_axis((1, 0, 0), math.pi / 2), FORWARD)  # fingers +x, closing along z (on a bar along y)

        def qa(phi):
            """Arm A at handle angle phi: fingers 45 deg down on the resting bar, straight down at 90 deg (the round
            bar turns in the closed fingers by the difference)."""
            return qmul(q_axis((0, 1, 0), math.radians(45.0) + phi / 2), G0)

        SIDE_B = qmul(q_axis((0, 0, 1), math.pi / 2), FORWARD)
        # Arm B: fingers level, closed fingers stacked vertically, pointing +y (QB0), turned back toward the robot
        # by 10 deg to go under the body and 20 deg by the end of the withdrawal (pointing straight +y, yam_arena's
        # IK misses the under-body points by 10-30 mm at x < 0.28; turned 20 deg it misses the approach point).
        QB0 = qmul(q_axis((0, 1, 0), -math.pi / 2), SIDE_B)
        QB10 = qmul(q_axis((0, 0, 1), math.radians(10.0)), QB0)
        QB20 = qmul(q_axis((0, 0, 1), math.radians(20.0)), QB0)
        B_WAIT = np.array([0.15, -0.14, 0.20])
        A_PARK = np.array([0.18, 0.20, 0.25])

        class Arm:
            def __init__(self, side):
                self.side = side
                self.pos, self.quat = grasp(side)
                self.grip = OPEN
                self.integrate = False
                self.I = np.zeros(6)

        def grasp(side):
            p, q = grasp_point(robot, side)
            return p - origin, q

        # -- measured state ----------------------------------------------------------------------------------------
        def phi():
            return float(unit.data.joint_pos.torch[0, st.handle_joint])

        def s_meas():
            return float(props.seated_origin(slot)[0] - (unit.data.root_pos_w.torch[0, 0].item() - origin[0]))

        def unit_pose():
            return unit.data.root_pos_w.torch[0].cpu().numpy() - origin, unit.data.root_quat_w.torch[0].cpu().numpy()

        def bar_world():
            pos = unit.data.body_pos_w.torch[0, st.handle_id].cpu().numpy() - origin
            quat = unit.data.body_quat_w.torch[0, st.handle_id].cpu().numpy()
            return pos + rotate(quat, (props.BAR_LOCAL[0], 0.0, props.BAR_LOCAL[1]))

        def pitch_deg():
            _, q = unit_pose()
            fwd = rotate(q, (1, 0, 0))
            return math.degrees(math.atan2(float(fwd[2]), float(fwd[0])))

        def status(label):
            g, _ = grasp("left")
            rep = task.episode_report(view, 0)
            log(f"{label}: s {s_meas() * 1000:.1f} mm, handle {math.degrees(phi()):.1f} deg, pitch {pitch_deg():.2f} deg, "
                f"unit z {unit_pose()[0][2]:.4f}, bar - graspA (mm) {np.round((bar_world() - g) * 1000, 1)}, "
                f"I_A {np.round(self._A.I, 3)}, I_B {np.round(self._B.I, 3)}, milestone {rep['milestone']} "
                f"events {rep['semantic_events']} {rep['contact_events']}", flush=True)

        def bar_target(phi_cmd, s):
            px, pz = props.pivot_world(slot)
            bx, bz = props.bar_local(phi_cmd)
            d = rotate(qa(phi_cmd), (0.0, 0.0, -1.0))  # finger direction
            return np.array([px + bx - s, props.SLOT_Y[slot], pz + bz]) - A_BACK * d

        # -- behaviours: generators that set an arm's target for one step per yield ---------------------------------
        def move(arm, pos, quat=None, speed=0.10, min_steps=3):
            pos = np.array(pos, float)
            quat = arm.quat if quat is None else np.array(quat, float)
            p0, q0 = arm.pos.copy(), arm.quat.copy()
            n = max(min_steps, int(math.ceil(np.linalg.norm(pos - p0) / (speed * dt))))
            for k in range(1, n + 1):
                a = k / n
                arm.pos, arm.quat = p0 + a * (pos - p0), slerp(q0, quat, a)
                yield

        def hold(arm, n):
            for _ in range(n):
                yield

        def grip(arm, value, n):
            arm.grip = value
            yield from hold(arm, n)

        def wait_until(cond, timeout):
            for _ in range(timeout):
                if cond():
                    return
                yield

        def settle(arm, tol=0.003, max_s=2.0):
            for _ in range(steps(max_s)):
                if np.linalg.norm(grasp(arm.side)[0] - arm.pos) < tol:
                    return
                yield

        flags = {"supported": False, "on_mat": False, "b_out": False}

        def script_a(arm):
            yield from hold(arm, steps(1.0))
            # Grip the bar.
            bar = bar_target(0.0, max(0.0, s_meas()))
            d = rotate(qa(0.0), (0.0, 0.0, -1.0))
            arm.grip = A_OPEN
            # First sideways at the ready height and well in front of the panel (straight down from the ready pose
            # would pass the other relay's front), then down to the approach point.
            yield from move(arm, np.array([0.20, bar[1], arm.pos[2]]), speed=0.12, min_steps=steps(1.5))
            yield from move(arm, bar - 0.08 * d, qa(0.0), speed=0.12, min_steps=steps(2.0))
            yield from settle(arm, max_s=3.0)
            yield from move(arm, bar, speed=0.03)
            yield from settle(arm, tol=0.002, max_s=3.0)
            log(f"GRIP: grasp point - bar target (mm) {np.round((grasp('left')[0] - bar) * 1000, 1)}", flush=True)
            yield from grip(arm, CLOSED, steps(0.8))
            status("LIFT start")
            # Lift: the commanded angle runs ahead of the measured one by at most LIFT_LEAD_DEG; the bar's target
            # follows the unit as the cam pushes it out.
            cmd = 0.0
            for _ in range(steps(15.0)):
                cmd = min(cmd + math.radians(LIFT_DEG_PER_S) * dt, phi() + math.radians(LIFT_LEAD_DEG),
                          math.radians(90.0))
                # Follow the cam: from where the tooth meets the cam face the unit is pushed out by TOOTH_R sin(phi)
                # - CAM_GAP (7 mm at 90 deg); holding the bar where the unit is now would hold the unit back.
                eject = S_REST + max(0.0, props.TOOTH_R * math.sin(cmd) - props.CAM_GAP)
                arm.pos, arm.quat = bar_target(cmd, max(s_meas(), eject)), qa(cmd)
                if math.degrees(phi()) >= LIFTED_DEG and cmd >= math.radians(90.0) - 1e-6:
                    break
                yield
            yield from hold(arm, steps(0.5))
            status("LIFTED")
            self.metrics["lift"] = {"handle_deg": round(math.degrees(phi()), 1), "eject_mm": round(s_meas() * 1000, 1)}
            # Pull alone (the handle rests on its up stop: the bar's weight and the pull keep it there).
            arm.integrate = True
            for _ in range(steps(20.0)):
                if s_meas() >= S_SUPPORT:
                    break
                if bar_world()[0] - arm.pos[0] < PULL_LEAD:
                    arm.pos = arm.pos + np.array([-PULL_SPEED * dt, 0.0, 0.0])
                yield
            yield from hold(arm, steps(0.5))
            status("PULLED (alone)")
            yield from wait_until(lambda: flags["supported"], steps(30.0))
            # Withdraw together with B (B mirrors A's x through flags["dx"]).
            yield from wait_until(lambda: flags["on_mat"], steps(60.0))
            yield from wait_until(lambda: flags["b_out"], steps(20.0))
            here, _ = grasp(arm.side)
            arm.integrate = False
            yield from grip(arm, OPEN, steps(1.0))
            yield from move(arm, arm.pos + np.array([0.0, 0.0, 0.05]), speed=0.04)
            yield from move(arm, A_PARK, speed=0.08)
            status("A parked")

        def script_b(arm):
            arm.grip = CLOSED
            yield from move(arm, B_WAIT, QB0, speed=0.10, min_steps=steps(2.0))
            A = self._A
            yield from wait_until(lambda: s_meas() >= S_SUPPORT - 0.002 and A.integrate, steps(90.0))
            yield from hold(arm, steps(0.5))
            x0, y0, z0 = props.seated_origin(slot)
            s = s_meas()
            pt = np.array([x0 - s + B_FROM_FRONT, y0 - props.BODY[1] / 2 + B_DEPTH, z0])
            under = pt + np.array([0.0, 0.0, -B_UNDER])
            yield from move(arm, under + np.array([0.0, -0.10, 0.02]), QB10, speed=0.10, min_steps=steps(1.5))
            yield from settle(arm, tol=0.006, max_s=3.0)
            yield from move(arm, under + np.array([0.0, -0.10, 0.0]), speed=0.04)
            yield from move(arm, under, speed=0.03)
            yield from settle(arm, tol=0.006, max_s=3.0)
            status("B under")
            z_ref = unit_pose()[0][2]
            arm.integrate = True
            n = 0
            for n in range(steps(6.0)):
                if (unit_pose()[0][2] - z_ref) * 1000 >= B_LIFT_MM:
                    break
                arm.pos = arm.pos + np.array([0.0, 0.0, B_RISE_PER_STEP])
                yield
            g, _ = grasp(arm.side)
            log(f"B CONTACT after {n} steps: unit lifted {(unit_pose()[0][2] - z_ref) * 1000:.1f} mm, B grasp z "
                f"{g[2]:.4f}, target z {arm.pos[2]:.4f}", flush=True)
            target = arm.pos.copy()
            target[2] = g[2] + B_HOLD_UP
            yield from move(arm, target, speed=0.02)
            yield from hold(arm, steps(1.0))
            flags["supported"] = True
            status("SUPPORTED")
            # Both withdraw to S_OUT at the same pace (A's target moves with B's).
            d = S_OUT - s_meas()
            n = max(3, int(math.ceil(d / (PULL_SPEED * dt))))
            a0, b0 = A.pos.copy(), arm.pos.copy()
            for k in range(1, n + 1):
                A.pos = a0 + np.array([-d * k / n, 0.0, 0.0])
                arm.pos = b0 + np.array([-d * k / n, 0.0, 0.0])
                arm.quat = slerp(QB10, QB20, k / n)
                yield
            yield from hold(arm, steps(1.0))
            status("HELD OUT")
            self.metrics["held_out"] = {"pitch_deg": round(pitch_deg(), 2), "s_mm": round(s_meas() * 1000, 1)}
            self._carry = True
            yield from hold(arm, steps(HOLD_S))
            self._carry = False
            status("AFTER HOLD")
            self.metrics["after_hold"] = {"pitch_deg": round(pitch_deg(), 2), "s_mm": round(s_meas() * 1000, 1)}
            # Lower both onto the mat.
            drop = props.BODY_Z - props.MAT_TOP + 0.003
            n = max(3, int(math.ceil(drop / (LOWER_SPEED * dt))))
            a0, b0 = A.pos.copy(), arm.pos.copy()
            for k in range(1, n + 1):
                A.pos = a0 + np.array([0.0, 0.0, -drop * k / n])
                arm.pos = b0 + np.array([0.0, 0.0, -drop * k / n])
                yield
            yield from hold(arm, steps(1.0))
            status("ON MAT")
            flags["on_mat"] = True
            arm.integrate = False
            yield from move(arm, arm.pos + np.array([0.0, -0.10, 0.0]), speed=0.03)
            flags["b_out"] = True
            yield from move(arm, B_WAIT, QB0, speed=0.08)
            status("B parked")

        def script_hold(arm):
            while True:
                yield

        self._A, self._B = Arm("left"), Arm("right")
        self._carry = False
        if control:
            self._gens = {"left": script_hold(self._A), "right": script_hold(self._B)}
        else:
            self._gens = {"left": script_a(self._A), "right": script_b(self._B)}
        self._alive = {"left": True, "right": True}
        self._control_at = steps(2.0) if control else None

        q_rel = np.array(hb.Q_REL)
        tcp_extra = 0.1347 - 0.134  # our grasp point sits this much further out than their TCP
        arms = {s: view.scene[f"{s}_robot"] for s in ("left", "right")}
        q_arm = {s: arms[s].data.joint_pos.torch[0, :6].cpu().numpy().astype(float) for s in ("left", "right")}
        self._q0 = {s: q.copy() for s, q in q_arm.items()}
        worst = {"left": 0.0, "right": 0.0}
        SEEDS = [np.array([0.0, 1.0, 1.0, -0.6, 0.0, 0.0])]
        k_step = [0]
        torque_log = {"left": [], "right": []}

        def joint_action(arm):
            if control:
                return np.r_[self._q0[arm.side], 1.0]
            pos = arm.pos + origin
            tcp = pos + rotate(arm.quat, (0.0, 0.0, -tcp_extra))
            rot = qmul(arm.quat, q_rel)
            q, pe, re = ik.solve(tcp, rot, q_arm[arm.side], base_pos=station.BASES[arm.side], iters=60)
            if pe > 0.002 or re > 0.02:  # stuck in a poor branch: try the seeds
                for seed in SEEDS:
                    q2, pe2, re2 = ik.solve(tcp, rot, seed, base_pos=station.BASES[arm.side], iters=150)
                    if pe2 + 0.05 * re2 < pe + 0.05 * re:
                        q, pe, re = q2, pe2, re2
            q_arm[arm.side] = q
            if pe > 0.003 and pe > worst[arm.side] + 0.002:
                worst[arm.side] = pe
                log(f"IK {arm.side} k={k_step[0]} error {pe * 1000:.1f} mm at {np.round(arm.pos, 4)}", flush=True)
            q_meas = arms[arm.side].data.joint_pos.torch[0, :6].cpu().numpy().astype(float)
            if arm.integrate:
                arm.I = np.clip(arm.I + KI * (q - q_meas), -I_MAX, I_MAX)
            else:
                arm.I = arm.I * 0.9
            if self._carry:
                torque_log[arm.side].append(arms[arm.side].data.applied_torque.torch[0, :6].abs().cpu().numpy())
            return np.r_[q + arm.I, arm.grip]

        self._joint_action = joint_action
        self._k = k_step
        self._torque_log = torque_log
        self._task, self._view = task, view

    def carry_torques(self) -> dict:
        out = {}
        for side, rows in self._torque_log.items():
            if rows:
                a = np.array(rows)
                out[side] = {"mean": np.round(a.mean(0), 2).tolist(), "peak": np.round(a.max(0), 2).tolist()}
        return out

    def next_action(self) -> np.ndarray:
        if self._control_at is not None and self._k[0] == self._control_at:
            self._task.force_control(self._view)
            self.log("CONTROL: force_control started", flush=True)
        for side in ("left", "right"):
            if self._alive[side]:
                try:
                    next(self._gens[side])
                except StopIteration:
                    self._alive[side] = False
                    if side == "right":
                        self.log(f"METRICS {self.metrics} carry torques {self.carry_torques()}", flush=True)
        if not any(self._alive.values()):
            self.finished = True
        self._k[0] += 1
        return np.concatenate([self._joint_action(self._A), self._joint_action(self._B)])
