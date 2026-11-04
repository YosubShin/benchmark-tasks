"""The scripted demo, closed loop.

Arm B (right) opens the voltage poles on its side (6 black, 9 black, 10 white) and parks; then arm A (left) opens the
voltage pole 3 (black). (The two grippers' bodies would meet above the row, so the arms take turns.) Each pole: the
fingers, closed and pointing down (10 deg forward), come down the pole's lane into the gap between the handle and the
base, just above the blade's top, and pull the handle out toward the robot until the blade leaves its jaw; then the
blade swings down and hangs. Arm A then opens pole 1 (orange, the phase-A short-circuiting switch), then pole 2 (red,
the phase-A test-jack switch), picks the test plug off its rest (fingers straight down, across the handle's rear
half), carries it in front of pole 2's jack, measures and corrects the probe's position, pushes it in, lets go and
backs off. The task then runs its after-release check (off camera).

``control=True``: the red test-jack blade is left closed; the robot opens the rest and tries the plug anyway (the
closed blade in the jack's slot stops the probe short of seated), then the after-release check is forced. ``probe``:
development only ("plug": start with the poles written open and only insert the plug).

It reads the simulator's state (blade angles, the plug's pose), like any scripted demo with privileged state;
``--policy switch-demo`` (testswitch_sim.policy) plays it through Inspect Robots. Grasp-point targets become
yam_arena's 14-D joint action through yam_arena's own IK every control step. Import after Isaac Sim has started.
"""

from __future__ import annotations

import math

import numpy as np

from yam_arena import ik, station

HOOK_PITCH_DEG = 80.0
"""Closed fingers pointing down and 10 deg forward (straight down is out of the arm's reach there), closing along y:
their 7.8 mm slab goes down the pole's lane."""
HOOK_GAP = 0.0012
"""The fingers' robot-side face this far from the handle's inner face (the face toward the base) on the way down."""
HOOK_HALF_X = 0.008
"""Half the fingers' extent along x at their ends (15.5 mm across, tilted 15 deg)."""
HOOK_DEPTH = 0.008
"""The finger ends go this far below the handle's top: above the blade's top and its jaw (10 mm below the top)."""
PULL_PRESS_M = 0.004
PULL_PER_S = 0.015
PULL_LEAD_M = 0.05
"""The pull's command runs at most this far ahead of the hand (the arm's soft gains make the pulling force)."""
FINGER_END = 0.012
"""The closed fingers' ends are this far beyond the grasp point."""
RELEASE_DEG = 30.0
PLUG_GRASP_BACK = 0.022
"""Grasp point this far behind the handle's centre (toward its rear end, where the lead goes in)."""
SEAT_TIP_O = -0.015
"""Aim the probe tip 15 mm past the base face (it stops on the slot's bottom): the extra tracking error is the push
that spreads the jack's sprung leaves."""

class Demo:
    """``next_action()`` gives the 14-D joint action for the next control step."""

    def __init__(self, STATION, control_hz: float = 20.0, speed: float = 0.10, control: bool = False, probe: str = "",
                 log=print):
        import torch

        from testswitch_sim import station as hb
        from testswitch_sim.kit import (CLOSED, DOWN, FORWARD, OPEN, GraspModel, conj, grasp_point, q_axis, qmul,
                                        rotate, slerp)
        from testswitch_sim.spec import POLES, TARGET_JACK, TARGET_SHORT
        from testswitch_sim.task import VOLT_IDS
        from testswitch_sim.usd import switch_props as props

        view, task = STATION.env, STATION.task
        st = task.state(view)
        origins = view.scene.env_origins
        origin = origins[0].cpu().numpy()
        robot = view.scene["robot"]
        dt = 1.0 / control_hz
        self.finished = False
        self.log = log
        steps = lambda s: max(1, round(s / dt))  # noqa: E731

        PRESS = qmul(q_axis((0, 1, 0), math.radians(45.0)), FORWARD)
        A_PARK = np.array([0.20, 0.20, 0.25])
        B_PARK = np.array([0.12, -0.26, 0.20])

        class Arm:
            def __init__(self, side):
                self.side = side
                self.pos, self.quat = grasp(side)
                self.grip = OPEN

        def grasp(side):
            p, q = grasp_point(robot, side)
            return p - origin, q

        def open_rad(i):
            return -float(st.switch.data.joint_pos.torch[0, st.joint_ids[i]])

        def plug_pose():
            p, q = st.plug_pose(origins)
            return p[0].cpu().numpy(), q[0].cpu().numpy()

        def diag(arm, label):
            g, _ = grasp(arm.side)
            pads = st.tips.points(origins)[0].cpu().numpy()
            pads = pads[:2] if arm.side == "left" else pads[2:]
            log(f"DIAG {label}: target {np.round(arm.pos, 4)} grasp {np.round(g, 4)} err_mm "
                f"{np.round((g - arm.pos) * 1000, 1)} pads {np.round(pads, 4).tolist()}", flush=True)

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

        def flick(arm, i, verbose=False):
            """Pull pole i's handle open: the closed fingers, nearly upright, come down into the gap between the
            handle and the base, their ends just above the blade's top, and pull the handle out toward the robot."""
            y = props.pole_y(i)
            end = np.array([props.FACE_X - props.HANDLE_O[0] + HOOK_GAP + HOOK_HALF_X, y,
                            props.CENTER_Z + props.HANDLE_A[1] - HOOK_DEPTH])
            # Pole 6, near the midline, is at the edge of the right arm's reach with the fingers 10 deg from upright.
            pitch = 75.0 if i == 5 else HOOK_PITCH_DEG
            hook = qmul(q_axis((0, 1, 0), math.radians(pitch)), FORWARD)
            hook_dir = np.array([math.cos(math.radians(pitch)), 0.0, -math.sin(math.radians(pitch))])
            at = end - FINGER_END * hook_dir
            arm.grip = CLOSED
            yield from move(arm, at + np.array([-0.02, 0.0, 0.035]), hook, speed=0.12, min_steps=steps(2.0))
            yield from settle(arm, max_s=3.0)
            yield from move(arm, at + np.array([0.0, 0.0, 0.02]), speed=0.05)
            yield from settle(arm)
            if verbose:
                diag(arm, f"above pole {i + 1}")
            yield from move(arm, at, speed=0.02)
            yield from settle(arm)
            if verbose:
                diag(arm, f"hooked pole {i + 1}")
            # Pull with a light press down: the fingers' front faces lean back, so the handle's top edge would wedge
            # them up and out of the gap; pressing keeps the finger ends on the jaw's top.
            arm.pos = arm.pos + np.array([0.0, 0.0, -PULL_PRESS_M])
            n = 0
            for n in range(steps(8.0)):
                if math.degrees(open_rad(i)) > RELEASE_DEG:
                    break
                g, _ = grasp(arm.side)
                if g[0] - arm.pos[0] < PULL_LEAD_M:  # pull, at most PULL_LEAD_M ahead of the hand
                    arm.pos = arm.pos + np.array([-PULL_PER_S * dt, 0.0, 0.0])
                if verbose and n % 5 == 0:
                    diag(arm, f"pull {n} pole {i + 1} at {math.degrees(open_rad(i)):.1f} deg")
                yield
            log(f"FLICK pole {i + 1} ({POLES[i][1]}): {n} pull steps, angle {math.degrees(open_rad(i)):.1f} deg",
                flush=True)
            # Follow through toward the robot (the handle swings on over, out of the way), let the blade fall, then
            # lift clear. Lifting straight away would catch the handle from below and push it back toward closed.
            here, _ = grasp(arm.side)
            arm.pos = here
            yield from move(arm, arm.pos + np.array([-0.035, 0.0, 0.0]), speed=0.06)
            yield from hold(arm, steps(0.8))
            yield from move(arm, arm.pos + np.array([-0.01, 0.0, 0.05]), speed=0.08)

        def plug_pick_and_insert(arm, i):
            p_pos, p_quat = plug_pose()
            handle_c = p_pos + rotate(p_quat, (0.0, 0.0, props.PLUG_TIP_TO_HANDLE_C))
            target = handle_c + rotate(p_quat, (0.0, 0.0, PLUG_GRASP_BACK)) + np.array([0.0, 0.0, 0.009])
            arm.grip = OPEN
            yield from move(arm, np.array([arm.pos[0] - 0.03, arm.pos[1], 0.26]), speed=0.10)
            yield from move(arm, target + np.array([0.0, 0.0, 0.07]), DOWN, speed=0.12, min_steps=steps(2.0))
            yield from settle(arm)
            yield from move(arm, target, speed=0.03)
            yield from settle(arm)
            diag(arm, "plug grasp")
            yield from grip(arm, CLOSED, steps(1.2))
            yield from move(arm, arm.pos + np.array([0.0, 0.0, 0.02]), speed=0.03)
            yield from settle(arm)
            g_pos, g_quat = grasp(arm.side)
            o_pos, o_quat = plug_pose()
            model = GraspModel(rotate(conj(g_quat), o_pos - g_pos), qmul(conj(g_quat), o_quat))
            log(f"PLUG grasp rel {np.round(model.rel_pos * 1000, 1)} mm", flush=True)
            mouth = np.array(props.jack_mouth_world(i))
            quat = np.array(props.PLUG_QUAT)

            def go_tip(tip, speed):
                tgt, q = model.grasp_for(tip, quat)
                yield from move(arm, tgt, q, speed=speed)
                yield from settle(arm, tol=0.002)

            entry = mouth + np.array([-0.010, 0.0, 0.0])
            yield from go_tip(entry + np.array([-0.03, 0.0, 0.02]), 0.08)
            correction = np.zeros(3)
            for attempt in range(5):
                yield from go_tip(entry - correction, 0.02)
                yield from hold(arm, steps(0.4))
                err = plug_pose()[0] - entry
                log(f"PLUG entry attempt {attempt}: tip error (mm) {np.round(err * 1000, 1)}", flush=True)
                correction = correction + err
                if np.abs(err[1:]).max() < 0.0008:
                    break
            # One push to SEAT_TIP_O past the face (the probe stops on the slot's bottom first; the extra lag is the
            # push that spreads the jack's sprung leaves). If the probe is not seated (the lead tugs on the plug and
            # can tip it onto a leaf's edge), back off to the entry point, line up again and push again.
            seat = np.array([props.FACE_X - SEAT_TIP_O, mouth[1], mouth[2]])
            for attempt in range(3):
                yield from go_tip(seat - correction * np.array([0.0, 1.0, 1.0]), 0.01)
                yield from hold(arm, steps(0.6))
                if bool(st.seated_once[0]):
                    break
                tip, _ = plug_pose()
                log(f"PLUG not seated ({(props.FACE_X - tip[0]) * 1000:.1f} mm out, lateral error (mm) "
                    f"{np.round((tip[1:] - mouth[1:]) * 1000, 1)}): backing off, retry {attempt + 1}", flush=True)
                yield from go_tip(entry - correction, 0.02)
                for _ in range(3):
                    yield from hold(arm, steps(0.4))
                    err = plug_pose()[0] - entry
                    correction = correction + err
                    if np.abs(err[1:]).max() < 0.0006:
                        break
                    yield from go_tip(entry - correction, 0.02)
            yield from hold(arm, steps(0.6))
            log(f"PLUG pushed: tip {np.round(plug_pose()[0], 4)}, out from face "
                f"{(props.FACE_X - plug_pose()[0][0]) * 1000:.1f} mm, seated {bool(st.seated_once[0])}", flush=True)
            here, _ = grasp(arm.side)
            yield from move(arm, here, speed=0.02, min_steps=steps(0.5))
            yield from grip(arm, OPEN, steps(1.0))
            yield from move(arm, arm.pos + np.array([0.0, 0.0, 0.06]), speed=0.04)

        def write_open(ids):
            jid = [st.joint_ids[j] for j in ids]
            q = torch.full((1, len(jid)), -math.radians(160.0), device=origins.device)
            st.switch.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=jid)
            st.switch.set_joint_position_target(q, joint_ids=jid)

        b_done = [False]

        def script_a(arm):
            yield from hold(arm, steps(1.5))
            if probe == "plug":
                write_open(VOLT_IDS + [TARGET_SHORT, TARGET_JACK])
                yield from hold(arm, steps(1.0))
                yield from plug_pick_and_insert(arm, TARGET_JACK)
                yield from move(arm, A_PARK, speed=0.10)
                return
            # The two grippers' bodies (about 10 cm across) would meet above the row, so arm A waits for arm B.
            yield from wait_until(lambda: b_done[0], steps(90.0))
            yield from flick(arm, 2)
            yield from wait_until(lambda: all(math.degrees(open_rad(v)) > 60 for v in VOLT_IDS), steps(6.0))
            for pole in ([TARGET_SHORT] if control else [TARGET_SHORT, TARGET_JACK]):
                yield from flick(arm, pole)
            if not control:
                yield from wait_until(lambda: math.degrees(open_rad(TARGET_JACK)) > 60, steps(6.0))
            yield from plug_pick_and_insert(arm, TARGET_JACK)
            if control:
                yield from move(arm, A_PARK, speed=0.10)
                task.force_function_test(view)
                return
            yield from move(arm, A_PARK, speed=0.10)

        def script_b(arm):
            yield from hold(arm, steps(1.5))
            if probe:
                return
            for pole in (5, 8, 9):  # pole 6 first, from the ready pose (it is at the edge of the arm's reach)
                yield from flick(arm, pole)
            yield from move(arm, B_PARK, PRESS, speed=0.10)
            b_done[0] = True

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
            if pe > 0.002 or re > 0.02:  # stuck in a poor branch: try the seeds that reach the task's poses
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

    def next_action(self) -> np.ndarray:
        for side in ("left", "right"):
            if self._alive[side]:
                try:
                    next(self._gens[side])
                except StopIteration:
                    self._alive[side] = False
        if not any(self._alive.values()):
            self.finished = True
        self._k[0] += 1
        return np.concatenate([self._joint_action(self._A), self._joint_action(self._B)])
