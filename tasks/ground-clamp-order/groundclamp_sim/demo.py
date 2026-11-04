"""The scripted demo, closed loop.

Arm B (right) takes the ground clamp's stick by its square handle (fingers down), lifts the clamp a millimetre off its
parking plate, carries it out in front of the ground bar, lines it up, slides it onto the bar's front edge (pad 1 mm
above the bar) and lets go: the clamp hangs on its pad. It then tightens the eye screw in regrasp strokes: close on the
handle, push the stick's axis back over the bar (2.5 mm toward the board, 1 mm down), roll the wrist 110 deg
clockwise (seen from above), open, roll back, close again. It stops once the clamp force reaches SEAT_N (read from the
task: privileged state), or when a stroke no longer turns the screw, and backs off. Arm A (left) meanwhile picks the
line clamp off its plate and carries it in front of the tagged bar B, waits there until the ground clamp counts as
tight, then hangs and tightens the line clamp the same way and backs off. The task then runs its after-release tug
check (off camera).

``control=True``: the line clamp is hung on bar B but not tightened; once the hands are clear the check is forced.

It reads the simulator's state (clamp frames, handle poses, clamp force), like any scripted demo with privileged
state; ``--policy clamp-demo`` (groundclamp_sim.policy) plays it through Inspect Robots. Grasp-point targets become
yam_arena's 14-D joint action through yam_arena's own IK every control step. Import after Isaac Sim has started.
"""

from __future__ import annotations

import math

import numpy as np

from yam_arena import ik, station

BACK_OFF = 0.06
"""How far in front of the bar's edge (x) a clamp travels sideways."""
INSERT_LIFT = 0.0015
"""Pad this far above the bar's top while the clamp is slid on (the hook jaw is then 1.5 mm under the bar)."""
SEAT_N = 200.0
"""Stop tightening at this clamp force (0.66 N*m at the eye)."""
MAX_STROKES = 14
STALL_N = 150.0
STALL_TURNS = 0.1
"""Also stop once tight enough to count (> 120 N) and a stroke turned the screw less than 0.1 turn (the pinch on the
handle is slipping: what the YAM's fingers transmit, about 0.6-0.9 N*m here)."""
STROKE_DEG = 110.0
STROKE_S = 2.4
YAW_WINDOW = {"right": 25.0, "left": -5.0}
"""A stroke starts at a yaw in [w, w + 90) (deg about +z, fingers down; the square handle can be taken every 90 deg)
and ends STROKE_DEG lower: inside the wrist-roll range the fingers-down IK holds at each bar."""
APPROACH_DZ = {"right": 0.02, "left": 0.045}
RETREAT = {"right": np.array([-0.06, -0.02, 0.0]), "left": np.array([-0.05, 0.05, 0.0])}
PRESS = np.array([0.0025, 0.0, -0.001])
"""While turning, the stick is held pushed 2.5 mm toward the board and 1 mm down, keeping the bar in the throat (a
lineman also pushes the clamp onto the conductor while turning)."""
A_PARK = np.array([0.16, 0.24, 0.26])


class Demo:
    """``next_action()`` gives the 14-D joint action for the next control step."""

    def __init__(self, STATION, control_hz: float = 20.0, control: bool = False, probe: str = "", log=print):
        from groundclamp_sim import station as hb
        from groundclamp_sim.kit import CLOSED, DOWN, OPEN, GraspModel, conj, grasp_point, q_axis, qmul, rotate, slerp
        from groundclamp_sim.spec import GROUND_Y, PHASE_Y, TAGGED
        from groundclamp_sim.usd import clamp_props as props

        view, task = STATION.env, STATION.task
        st = task.state(view)
        origins = view.scene.env_origins
        origin = origins[0].cpu().numpy()
        robot = view.scene["robot"]
        art = st.art
        dt = 1.0 / control_hz
        self.finished = False
        self.log = log
        steps = lambda s: max(1, round(s / dt))  # noqa: E731
        IDENT = np.array([0.0, 0.0, 0.0, 1.0])

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

        def yaw_of(q):
            x = rotate(q, (1.0, 0.0, 0.0))
            return math.degrees(math.atan2(x[1], x[0]))

        def down_yaw(deg):
            return qmul(q_axis((0, 0, 1), math.radians(deg)), DOWN)

        def handle(c):
            pos, quat = body(st.screw_ids[c])
            return pos + rotate(quat, (0.0, 0.0, props.handle_center_local())), yaw_of(quat)

        def stroke_start(side, handle_yaw):
            w = YAW_WINDOW[side]
            return w + (handle_yaw - w) % 90.0

        def measure(side, idx):
            g_pos, g_quat = grasp(side)
            o_pos, o_quat = body(idx)
            return GraspModel(rotate(conj(g_quat), o_pos - g_pos), qmul(conj(g_quat), o_quat))

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

        def settle(arm, tol=0.003, max_s=2.0):
            for _ in range(steps(max_s)):
                if np.linalg.norm(grasp(arm.side)[0] - arm.pos) < tol:
                    return
                yield
            log(f"SETTLE timeout ({arm.side}): lag {np.linalg.norm(grasp(arm.side)[0] - arm.pos) * 1000:.1f} mm",
                flush=True)

        def pick(arm, c):
            """Take clamp c's stick by its handle, fingers down, and lift the clamp 1 mm off its plate. The fingers
            close across the handle along y (the square's side nearest yaw 0); if the grasp ends up off the handle's
            centre (the fingers nudged the stick), back off and try again."""
            for attempt in range(3):
                hpos, hyaw = handle(c)
                yaw = (hyaw + 45.0) % 90.0 - 45.0
                gq = down_yaw(yaw)
                log(f"PICK {arm.side} clamp {'gl'[c]} attempt {attempt}: handle {np.round(hpos, 4)} yaw {hyaw:.1f}",
                    flush=True)
                arm.grip = OPEN
                dz = APPROACH_DZ[arm.side]
                yield from move(arm, np.array([arm.pos[0], arm.pos[1], max(arm.pos[2], hpos[2] + dz + 0.01)]),
                                speed=0.10)
                yield from move(arm, hpos + np.array([0.0, 0.0, dz]), gq, speed=0.10, min_steps=steps(2.0))
                yield from settle(arm, tol=0.002, max_s=3.0)
                hpos, _ = handle(c)
                yield from move(arm, hpos, speed=0.03)
                yield from settle(arm, tol=0.002, max_s=2.0)
                g, _ = grasp(arm.side)
                off = np.linalg.norm((g - handle(c)[0])[:2])
                log(f"GRIP {arm.side}: grasp {np.round(g, 4)} vs handle {np.round(handle(c)[0], 4)}", flush=True)
                if off < 0.008:
                    break
                yield from move(arm, arm.pos + np.array([0.0, 0.0, 0.05]), speed=0.05)
                yield from hold(arm, steps(1.5))  # let the stick settle back upright on its plate
            yield from grip(arm, CLOSED, steps(1.0))
            yield from settle(arm)

        def carry(arm, c, bar_y):
            """Lift the clamp 1 mm and carry it out in front of the bar at bar_y (BACK_OFF in front of its edge)."""
            model = measure(arm.side, st.frame_ids[c])
            log(f"CARRY {arm.side}: frame in grasp frame (mm) {np.round(model.rel_pos * 1000, 1)}", flush=True)
            f0, _ = body(st.frame_ids[c])
            home = np.array(props.frame_home(bar_y))
            # Lift only 1 mm: the hook jaw hangs 3 mm under the plate, and a bigger lift clamps the plate.
            lift = f0 + np.array([0.0, 0.0, 0.001])
            out_here = np.array([home[0] - BACK_OFF, f0[1], f0[2] + 0.001])
            out_there = np.array([home[0] - BACK_OFF, bar_y, home[2] + INSERT_LIFT + 0.002])
            for target, speed in ((lift, 0.02), (out_here, 0.04), (out_there, 0.05)):
                g, q = model.grasp_for(target, IDENT)
                yield from move(arm, g, q, speed=speed)
                if target is lift:
                    yield from settle(arm, max_s=1.0)
            yield from hold(arm, steps(0.4))
            yield from settle(arm)
            f, fq = body(st.frame_ids[c])
            log(f"CARRIED {arm.side}: frame {np.round(f, 4)} (target {np.round(out_there, 4)}), yaw {yaw_of(fq):.1f}",
                flush=True)

        def hang(arm, c, bar_y):
            """Line the clamp up in front of the bar (closed loop on the frame's position), slide it on, let go."""
            model = measure(arm.side, st.frame_ids[c])  # the frame may have turned on its screw
            home = np.array(props.frame_home(bar_y))
            target_out = np.array([home[0] - BACK_OFF, bar_y, home[2] + INSERT_LIFT])
            correction = np.zeros(3)
            for attempt in range(3):
                f, fq = body(st.frame_ids[c])
                err = f - target_out
                log(f"HANG {arm.side} attempt {attempt}: frame error (mm) {np.round(err * 1000, 1)} "
                    f"yaw {yaw_of(fq):.1f}", flush=True)
                correction = correction + err
                g, q = model.grasp_for(target_out - correction, IDENT)
                yield from move(arm, g, q, speed=0.02)
                yield from hold(arm, steps(0.4))
                yield from settle(arm, tol=0.002)
            # Slide on in short increments, re-measuring the frame and correcting its height and side error after
            # each (the arm sags differently as it reaches out; the pad clears the bar's top by only INSERT_LIFT).
            for dx in (0.02, 0.035, 0.045, 0.05, 0.055, 0.06, 0.06):
                target = target_out + np.array([dx, 0.0, 0.0])
                g, q = model.grasp_for(target - correction, IDENT)
                yield from move(arm, g, q, speed=0.012)
                yield from hold(arm, steps(0.5))
                f, _ = body(st.frame_ids[c])
                err = f - target
                correction = correction + err * np.array([0.5, 1.0, 1.0])
                log(f"INSERT {arm.side} dx {dx * 1000:.0f} mm: frame error (mm) {np.round(err * 1000, 1)}", flush=True)
            f, _ = body(st.frame_ids[c])
            log(f"HUNG {arm.side}: frame vs home (mm) {np.round((f - home) * 1000, 1)}", flush=True)
            yield from grip(arm, OPEN, steps(0.8))
            yield from hold(arm, steps(0.8))

        def retreat(arm):
            """Open hand straight up off the handle, then back toward the robot, well clear of both sticks."""
            yield from move(arm, arm.pos + np.array([0.0, 0.0, 0.05]), speed=0.04)
            yield from move(arm, arm.pos + RETREAT[arm.side], speed=0.05)
            yield from settle(arm, tol=0.005, max_s=2.0)
            g, _ = grasp(arm.side)
            d = [float(np.linalg.norm(g - handle(c)[0])) for c in range(2)]
            log(f"RETREAT {arm.side}: grasp {np.round(g, 3)}, distance to handles (mm) {np.round(np.array(d) * 1000)}",
                flush=True)

        def tighten(arm, c, bar_y):
            strokes, last_turns = 0, None
            home = np.array(props.frame_home(bar_y))
            while True:
                force = float(st.force[0, c])
                turns = float(st.turn_acc[0, c]) / (2 * math.pi)
                hpos, hyaw = handle(c)
                log(f"STROKE {arm.side} #{strokes}: force {force:.0f} N, screw turns {turns:.3f}, handle yaw "
                    f"{hyaw:.1f}", flush=True)
                stalled = last_turns is not None and force >= STALL_N and abs(turns - last_turns) < STALL_TURNS
                if force >= SEAT_N or stalled or strokes >= MAX_STROKES:
                    break
                last_turns = turns
                strokes += 1
                yaw0 = stroke_start(arm.side, hyaw)
                arm.grip = OPEN
                yield from move(arm, hpos, down_yaw(yaw0), speed=0.03, min_steps=steps(0.8))
                yield from grip(arm, CLOSED, steps(0.8))
                # Re-seat: push the screw axis back over the bar (each stroke drags the clamp a little), then turn
                # about it.
                seat = np.array([home[0], home[1], hpos[2]]) + PRESS
                yield from move(arm, seat, down_yaw(yaw0), speed=0.01, min_steps=steps(0.4))
                yield from move(arm, seat, down_yaw(yaw0 - STROKE_DEG), min_steps=steps(STROKE_S))
                yield from hold(arm, steps(0.3))
                yield from grip(arm, OPEN, steps(0.6))
                yield from move(arm, hpos, down_yaw(yaw0), min_steps=steps(0.8))
            yield from grip(arm, OPEN, steps(0.6))
            yield from retreat(arm)
            log(f"DONE {arm.side} clamp {'gl'[c]}: force {float(st.force[0, c]):.0f} N after {strokes} strokes; "
                f"ground_attached {bool(st.last['ground_attached'][0])} line_attached "
                f"{bool(st.last['line_attached'][0])}", flush=True)

        def script_b(arm):  # right arm: the ground clamp
            yield from hold(arm, steps(1.0))
            if probe == "line":
                return
            yield from pick(arm, 0)
            yield from carry(arm, 0, GROUND_Y)
            yield from hang(arm, 0, GROUND_Y)
            yield from tighten(arm, 0, GROUND_Y)

        def script_a(arm):  # left arm: the line clamp
            bar_y = PHASE_Y[TAGGED]
            yield from hold(arm, steps(1.0))
            if probe == "ground":
                return
            yield from pick(arm, 1)
            yield from carry(arm, 1, bar_y)
            while not bool(st.last["ground_attached"][0]):
                yield
            log("LINE: ground clamp attached; hanging the line clamp", flush=True)
            yield from hang(arm, 1, bar_y)
            if control:
                yield from retreat(arm)
                yield from hold(arm, steps(1.0))
                log("CONTROL: line clamp left hung, not tightened; forcing the check", flush=True)
                task.force_function_test(view)
                return
            yield from tighten(arm, 1, bar_y)

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
