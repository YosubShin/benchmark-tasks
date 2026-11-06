"""The scripted demo, closed loop.

Arm B (right) squeezes the ground clip's handles (fingers down, closing along y), lifts it off its insulating tab
with the jaws open, carries it over the earth bar, lowers it onto the bar's top edge (closed loop on the clip's
measured position) and lets go: the spring closes the pads on the bar. Arm A (left) then picks the stick out of its
cradle (fingers down, across the tube, just behind the hand guard) and holds it level beside C2's studs until the WAIT
lamp is out and the clip counts as on the bar. It slides the hook over both studs, lowers it onto them (re-measuring
the hook's position after each step), holds it there for the dwell time and lets go: the stick pivots on the near
stud, its butt drops to the table and the turned-down end catches behind the far stud. Both arms back off and the
task runs its after-release check (off camera).

``control=True``: arm B leaves the clip on its tab; arm A waits only for the lamp and lays the stick on C2's studs
(an unearthed stick). Once the hands are clear the check is forced. It must score 0.

It reads the simulator's state (stick and clip poses, the lamp, the clip-on-bar flag), like any scripted demo with
privileged state; ``--policy groundstick-demo`` (groundstick_sim.policy) plays it through Inspect Robots. Grasp-point
targets become yam_arena's 14-D joint action through yam_arena's own IK every control step. Import after Isaac Sim has
started.
"""

from __future__ import annotations

import math

import numpy as np

from yam_arena import ik, station

GRIPPER_OPEN_M = 0.04695
SQUEEZE = 0.006 / GRIPPER_OPEN_M
"""Gripper command for a 6 mm half-gap: the moving handle is pushed to the jaw's 29 deg stop and the clip is then held
rigid between the fingers (the tips open 27 mm)."""
GRIP_IN_CLIP = (-0.035, -0.009, 0.0)
"""Grasp point on the clip (clip frame): 35 mm behind the pivot and 9 mm toward the fixed half, so the squeeze turns the
moving half and does not press the fixed pad into the tab."""
CLIP_GAP = 0.0015
"""Pad A placed this far off the bar's face before release."""
CLIP_X = 0.22
"""Where along the earth bar (x) the clip goes."""
R_PARK = np.array([0.20, -0.26, 0.25])
STICK_GRASP_X = 0.06
"""Grasp the stick just behind the hand guard (stick frame x; the guard is at 70 mm)."""
SIDE_Z = 0.19
"""Stick axis height while hooking in: the turned-down end clears the case tops but not the studs."""
SIDE_DX = -0.025
"""Stick offset in x before sliding over the studs: the turned-down end waits beside C2's bushings."""
DWELL_HOLD_S = 3.0
A_PARK_DELTA = (np.array([-0.06, 0.0, 0.0]), np.array([0.0, 0.04, 0.12]))


class Demo:
    """``next_action()`` gives the 14-D joint action for the next control step."""

    def __init__(self, STATION, control_hz: float = 20.0, control: bool = False, probe: str = "", log=print):
        from groundstick_sim import station as hb
        from groundstick_sim.kit import CLOSED, DOWN, OPEN, GraspModel, conj, grasp_point, q_axis, qmul, rotate, slerp
        from groundstick_sim.spec import EARTH_BAR_Y
        from groundstick_sim.usd import stick_props as props

        view, task = STATION.env, STATION.task
        st = task.state(view)
        origin = view.scene.env_origins[0].cpu().numpy()
        robot = view.scene["robot"]
        art = st.art
        dt = 1.0 / control_hz
        self.finished = False
        steps = lambda s: max(1, round(s / dt))  # noqa: E731
        STICK_QUAT = np.array(props.STICK_QUAT)
        CLIP_QUAT = np.array(props.CLIP_QUAT)
        DOWN_ACROSS = qmul(q_axis((0, 0, 1), -math.pi / 2), DOWN)
        named = int(st.scorer.named[0])
        far = np.array(props.terminal_xyz(named, 1))
        far_local_x = props.PROBE_END_X - 0.018

        class Arm:
            def __init__(self, side):
                self.side = side
                self.pos, self.quat = grasp(side)
                self.grip = OPEN

        def grasp(side):
            p, q = grasp_point(robot, side)
            return p - origin, q

        def body(idx):
            return (art.data.body_pos_w.torch[0, idx].cpu().numpy() - origin,
                    art.data.body_quat_w.torch[0, idx].cpu().numpy())

        def measure(side, idx):
            g_pos, g_quat = grasp(side)
            o_pos, o_quat = body(idx)
            return GraspModel(rotate(conj(g_quat), o_pos - g_pos), qmul(conj(g_quat), o_quat))

        def flag(name):
            return bool(st.last[name][0]) if name in st.last else False

        # Behaviours: generators that set an arm's target for one step per yield ---------------------------------
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

        def settle(arm, tol=0.003, max_s=1.5):
            for _ in range(steps(max_s)):
                if np.linalg.norm(grasp(arm.side)[0] - arm.pos) < tol:
                    return
                yield
            log(f"SETTLE timeout ({arm.side}): lag {np.linalg.norm(grasp(arm.side)[0] - arm.pos) * 1000:.1f} mm",
                flush=True)

        # Arm B: the ground clip ---------------------------------------------------------------------------------
        def clip_target(gap):
            return np.array([CLIP_X, EARTH_BAR_Y - props.BAR_T / 2 - gap, props.BAR_Z[1] + props.THROAT_DEPTH])

        def clip_on_bar(arm):
            pos, quat = body(st.clip_id)
            g, gq = GraspModel.nominal(DOWN, quat, GRIP_IN_CLIP).grasp_for(pos, quat)
            log(f"CLIP start {np.round(pos, 4)} grasp {np.round(g, 4)}", flush=True)
            arm.grip = OPEN
            yield from move(arm, g + np.array([0.0, 0.0, 0.10]), gq, min_steps=steps(2.5))
            yield from move(arm, g + np.array([0.0, 0.0, 0.02]), speed=0.05)
            yield from move(arm, g, speed=0.02)
            yield from settle(arm, tol=0.002)
            yield from grip(arm, SQUEEZE, steps(1.4))
            yield from settle(arm)
            log(f"CLIP squeezed: jaw {float(st.last['jaw_deg'][0]):.1f} deg", flush=True)
            yield from move(arm, arm.pos + np.array([0.0, 0.0, 0.07]), speed=0.04)
            yield from settle(arm)
            model = measure(arm.side, st.clip_id)
            log(f"CLIP grasp rel_pos (mm) {np.round(model.rel_pos * 1000, 1)}", flush=True)
            yield from move(arm, model.grasp_for(clip_target(CLIP_GAP) + np.array([0.0, 0.0, 0.05]), CLIP_QUAT)[0],
                            speed=0.06)
            yield from settle(arm)
            final = clip_target(CLIP_GAP)
            correction = np.zeros(3)
            for attempt in range(4):
                pos, _ = body(st.clip_id)
                goal = final + np.array([0.0, 0.0, 0.05 if attempt == 0 else 0.0])
                err = pos - goal
                log(f"CLIP attempt {attempt}: error (mm) {np.round(err * 1000, 1)}", flush=True)
                if attempt > 0 and np.abs(err).max() <= 0.002:
                    break
                correction = correction + err
                if attempt == 0:
                    yield from move(arm, model.grasp_for(final - correction + np.array([0, 0, 0.012]), CLIP_QUAT)[0],
                                    speed=0.02)
                yield from move(arm, model.grasp_for(final - correction, CLIP_QUAT)[0], speed=0.01)
                yield from hold(arm, steps(0.3))
                yield from settle(arm, tol=0.002)
            yield from grip(arm, OPEN, steps(1.4))
            yield from move(arm, arm.pos + np.array([0.0, 0.0, 0.08]), speed=0.04)
            yield from move(arm, R_PARK, speed=0.08)
            yield from settle(arm, tol=0.005)
            log(f"CLIP released: grounded {flag('grounded')} jaw {float(st.last['jaw_deg'][0]):.1f} deg "
                f"clip {np.round(body(st.clip_id)[0], 4)}", flush=True)

        # Arm A: the stick ---------------------------------------------------------------------------------------
        def probe_point(local_x):
            pos, quat = body(st.stick_id)
            return pos + rotate(quat, (local_x, 0.0, 0.0))

        def stick_axis():
            return rotate(body(st.stick_id)[1], (1.0, 0.0, 0.0))

        def stick_pose_for(height):
            """Stick frame position that puts the probe's point far_local_x right above the far stud's top."""
            rod = far + np.array([0.0, 0.0, props.ROD_R + height])
            return rod - rotate(STICK_QUAT, (far_local_x, 0.0, 0.0))

        def stick_on_studs(arm):
            pos, quat = body(st.stick_id)
            g, gq = GraspModel.nominal(DOWN_ACROSS, quat, (STICK_GRASP_X, 0.0, 0.0)).grasp_for(pos, quat)
            log(f"STICK start {np.round(pos, 4)} grasp {np.round(g, 4)}", flush=True)
            arm.grip = OPEN
            yield from move(arm, g + np.array([0.0, 0.0, 0.10]), gq, min_steps=steps(2.5))
            yield from move(arm, g, speed=0.05)
            yield from settle(arm, tol=0.002)
            yield from grip(arm, CLOSED, steps(1.0))
            yield from move(arm, np.array([g[0], g[1], SIDE_Z]), speed=0.05)
            yield from settle(arm)
            model = measure(arm.side, st.stick_id)
            log(f"STICK grasp rel_pos (mm) {np.round(model.rel_pos * 1000, 1)}, axis {np.round(stick_axis(), 3)}",
                flush=True)
            side_h = SIDE_Z - props.STUD_TOP - props.ROD_R
            yield from move(arm, *model.grasp_for(stick_pose_for(side_h) + np.array([SIDE_DX, 0.0, 0.0]), STICK_QUAT),
                            speed=0.05)
            yield from settle(arm)
            while bool(st.lamp_shown[0]) or not (flag("grounded") or control):
                yield
            log(f"STICK lamp out, clip grounded={flag('grounded')}: hooking in", flush=True)
            model = measure(arm.side, st.stick_id)
            log(f"STICK axis {np.round(stick_axis(), 3)} (level: [0, -1, 0])", flush=True)
            yield from move(arm, *model.grasp_for(stick_pose_for(side_h), STICK_QUAT), speed=0.02)
            yield from hold(arm, steps(0.3))
            yield from settle(arm)
            correction = np.zeros(3)
            for attempt in range(3):
                h = 0.012 if attempt == 0 else 0.004
                yield from move(arm, *model.grasp_for(stick_pose_for(h) - correction, STICK_QUAT),
                                speed=0.03 if attempt == 0 else 0.01)
                yield from hold(arm, steps(0.3))
                yield from settle(arm, tol=0.002)
                err = probe_point(far_local_x) - (far + np.array([0.0, 0.0, props.ROD_R + h]))
                log(f"STICK at +{h * 1000:.0f} mm: probe error (mm) {np.round(err * 1000, 1)}, axis "
                    f"{np.round(stick_axis(), 3)}, arm lag (mm) {np.round((grasp(arm.side)[0] - arm.pos) * 1000, 1)}",
                    flush=True)
                correction = correction + err
            yield from move(arm, *model.grasp_for(stick_pose_for(-0.001) - correction, STICK_QUAT), speed=0.006)
            yield from hold(arm, steps(DWELL_HOLD_S))
            rep = task.episode_report(view, 0)
            log(f"STICK before release: milestone {rep['milestone']} discharged {rep['discharged']} "
                f"probe_on_named {st.last['probe_on_named'][0].tolist()} events {rep['semantic_events']}", flush=True)
            yield from grip(arm, OPEN, steps(1.2))
            for delta, speed in zip(A_PARK_DELTA, (0.03, 0.05)):
                yield from move(arm, arm.pos + delta, speed=speed)
            yield from settle(arm, tol=0.005)
            log(f"STICK released: probe_on_named {st.last['probe_on_named'][0].tolist()} "
                f"grounded {flag('grounded')}", flush=True)

        def script_b(arm):
            yield from hold(arm, steps(1.0))
            if control or probe == "stick":
                return
            yield from clip_on_bar(arm)

        def script_a(arm):
            yield from hold(arm, steps(1.0))
            if probe == "clip":
                return
            if not control:
                while self._alive["right"]:  # the stick is picked up once the clip is on the bar
                    yield
            yield from stick_on_studs(arm)
            if control:
                yield from hold(arm, steps(2.5))
                log("CONTROL: clip left on its tab; forcing the check", flush=True)
                task.force_function_test(view)

        self._A, self._B = Arm("left"), Arm("right")
        self._gens = {"left": script_a(self._A), "right": script_b(self._B)}
        self._alive = {"left": True, "right": True}

        q_rel = np.array(hb.Q_REL)
        tcp_extra = 0.1347 - 0.134  # our grasp point sits this much further out than their TCP
        q_arm = {s: view.scene[f"{s}_robot"].data.joint_pos.torch[0, :6].cpu().numpy().astype(float)
                 for s in ("left", "right")}
        worst = {"left": 0.0, "right": 0.0}
        SEEDS = [np.array([0.0, 1.0, 1.0, -0.6, 0.0, 0.0])]
        k_step = [0]

        def joint_action(arm):
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
            return np.r_[q, arm.grip]

        self._joint_action = joint_action
        self._k = k_step
        self.log = log

    def next_action(self) -> np.ndarray:
        for side in ("right", "left"):
            if self._alive[side]:
                try:
                    next(self._gens[side])
                except StopIteration:
                    self._alive[side] = False
        if not any(self._alive.values()):
            self.finished = True
        self._k[0] += 1
        return np.concatenate([self._joint_action(self._A), self._joint_action(self._B)])
