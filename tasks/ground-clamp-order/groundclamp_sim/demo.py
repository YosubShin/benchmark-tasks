"""The scripted demo, closed loop on what the robot itself observes.

What it reads, every control step, is only the policy's observation: the three camera images (``top_cam``,
``left_cam``, ``right_cam``, RGB) and the 14-D ``joint_pos`` (both arms' joint angles and gripper openings). It never
reads the simulator: no prop poses, prop joint angles, screw turns, clamp forces or scorer state. Besides the
observation it uses the documented setup layout (bar, plate and stick positions and heights, the clamp's published
and stated dimensions: task.json ``setup``), the station's camera calibration and the arm's URDF (both from
yam_arena), and time.

- Where things are. At the start both arms bring their open hands over where the layout puts the sticks, and the
  demo looks once with the top camera:
  each stick's red cap is found in the top image near where the layout puts it, and back-projected onto the cap's
  known height (groundclamp_sim.vision), which corrects the pick-up points; the yellow TESTED card is found on the
  label ledge, and the phase bar nearest to it is the one the line clamp goes on. Bars and plates are fixed to the
  bench, so their positions come from the layout.
- Where the held clamp is. When the fingers close on a stick's square handle, the handle's height is the layout's,
  its position across the fingers is the grasp point's (the fingers centre it) and along them the camera's
  estimate. From then on the clamp's frame is placed from the arm's own forward kinematics (measured joint angles)
  and that grasp, as a rigid hang below the handle (the clamp's stated dimensions). The hang-on is closed loop on
  this estimate, which takes out the arm's sag under the clamp's weight.
- Whether a grasp holds: the gripper's measured opening (it stops on the 26 mm handle instead of closing).
- How tight. The wrist turns the stick in regrasp strokes; the screw's turns are counted from the wrist's measured
  roll while the fingers hold the handle (about 1.2 turns take up the documented slack), and the screw's resistance
  is felt as the wrist's roll lagging its command at the end of a stroke (the wrist joint is a stiff PD; a resisting
  torque shows up as a steady lag). A clamp counts as tight to the demo when the counted turns exceed TURNS_TIGHT
  and the last stroke's lag has grown by LAG_LOADED_DEG over the first (unloaded) stroke.
- Order. Arm A (left) only hangs the line clamp once arm B (right) has finished tightening the ground clamp by that
  rule: the demo's own plan, not a reading of the task.

Arm B takes the ground clamp's stick by its square handle (fingers down), lifts the clamp off its parking plate,
carries it out in front of the ground bar, lines it up, slides it onto the bar's front edge (pad just above the bar)
and lets go: the clamp hangs on its pad. It then tightens the eye screw in strokes: close on the handle, push the
stick's axis back over the bar (2.5 mm toward the board, 1 mm down), roll the wrist 110 deg clockwise (seen from
above), open, roll back, close again. Arm A meanwhile picks the line clamp off its plate and carries it in front of
the tagged bar, waits, then hangs and tightens it the same way. Both hands back off; the task then runs its
after-release tug check (off camera).

``control=True``: the line clamp is hung on the tagged bar but not tightened; arm A backs off and holds.

Grasp-point targets become yam_arena's 14-D joint action through yam_arena's own IK every control step. Import after
Isaac Sim has started (the prop geometry module imports pxr).
"""

from __future__ import annotations

import math

import numpy as np

from yam_arena import ik, station

BACK_OFF = 0.06
"""How far in front of the bar's edge (x) a clamp travels sideways."""
INSERT_LIFT = 0.0015
"""Pad this far above the bar's top while the clamp is slid on (the hook jaw is then 1.5 mm under the bar)."""
TURNS_TIGHT = 1.40
"""Counted screw turns after which the clamp can be tight (the documented slack is about 1.2 turns)."""
LAG_LOADED_DEG = 1.5
"""Growth of the wrist's end-of-stroke roll lag over the first stroke's that shows the screw is resisting."""
TURNS_GIVE_UP = 2.2
MAX_STROKES = 9
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
HOLDING_MIN = 0.12
"""Gripper opening (0 closed, 1 open) above which the fingers have stopped on the 26 mm handle (about 0.28)."""
CAP_ABOVE_HANDLE = 0.0315
"""The red cap's top face above the handle's centre (stated stick dimensions)."""
ARM_KEYS = {"left": slice(0, 7), "right": slice(7, 14)}


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


class Demo:
    """``next_action(observation)`` gives the 14-D joint action for the next control step."""

    def __init__(self, width: int = 320, height: int = 176, control_hz: float = 20.0, control: bool = False,
                 log=print):
        from groundclamp_sim import vision
        from groundclamp_sim.kit import CLOSED, DOWN, OPEN, GraspModel, conj, q_axis, qmul, rotate, slerp
        from groundclamp_sim.spec import GROUND_Y, PARK_Y, PHASE_Y, PHASES
        from groundclamp_sim.usd import clamp_props as props

        dt = 1.0 / control_hz
        self.finished = False
        self.log = log
        self.obs = None
        self.debug: dict = {}
        """What the demo currently believes (for development logs; nothing here comes from the simulator)."""
        steps = lambda s: max(1, round(s / dt))  # noqa: E731
        IDENT = np.array([0.0, 0.0, 0.0, 1.0])
        cam = vision.TopCamera(width, height)
        HANDLE_Z = props.BAR_TOP + props.handle_center_local()
        """Handle centre height while a clamp hangs by its pad on a plate or bar top (layout)."""
        FRAME_BELOW_HANDLE = props.P0 + props.handle_center_local()
        """Clamp frame origin below the handle centre, on the screw axis, before the screw is turned (layout)."""
        park_handles = [np.array([props.frame_home(PARK_Y[c])[0], PARK_Y[c], HANDLE_Z]) for c in range(2)]
        shared = {"looked": False, "handles": [h.copy() for h in park_handles], "tagged_y": None,
                  "ground_tight": False, "waiting": set()}
        self.debug["shared"] = shared

        def joints(side):
            return np.asarray(self.obs.state["joint_pos"], float)[ARM_KEYS[side]]

        def grasp(side):
            return vision.grasp_pose(joints(side)[:6], side)

        def gripper_opening(side):
            return float(joints(side)[6])

        class Arm:
            def __init__(self, side):
                self.side = side
                self.pos, self.quat = grasp(side)
                self.grip = OPEN
                self.model = None

        def yaw_of(q):
            x = rotate(q, (1.0, 0.0, 0.0))
            return math.degrees(math.atan2(x[1], x[0]))

        def grip_yaw(q):
            """The yaw ``deg`` for which ``down_yaw(deg)`` is the fingers-down orientation q."""
            return _wrap(yaw_of(q) - 180.0)

        def down_yaw(deg):
            return qmul(q_axis((0, 0, 1), math.radians(deg)), DOWN)

        def stroke_start(side, handle_yaw):
            w = YAW_WINDOW[side]
            return w + (handle_yaw - w) % 90.0

        def frame_estimate(arm):
            """The held clamp's frame (desk frame), from the arm's measured joints and the grasp."""
            g_pos, g_quat = grasp(arm.side)
            pos = g_pos + rotate(g_quat, arm.model.rel_pos)
            self.debug[f"frame_est_{arm.side}"] = pos
            return pos

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

        def rise_and_look(arm, c):
            """Lift the hand, bring it (fingers open) over where the layout puts clamp c's stick, which keeps the stick's
            cap in the top camera's view (at the start pose the hand hides it), then wait until both hands are there and
            the demo has looked once."""
            z = HANDLE_Z + APPROACH_DZ[arm.side] + 0.01
            yield from move(arm, np.array([arm.pos[0], arm.pos[1], max(arm.pos[2], z)]), speed=0.10)
            yield from move(arm, park_handles[c] + np.array([0.0, 0.0, APPROACH_DZ[arm.side]]), down_yaw(0.0),
                            speed=0.10, min_steps=steps(2.0))
            yield from settle(arm, tol=0.003, max_s=2.0)
            shared["waiting"].add(arm.side)
            while not shared["looked"]:
                yield

        def look(img):
            """One look with the top camera: the two sticks' caps and the TESTED card."""
            for c in range(2):
                nominal = park_handles[c] + np.array([0.0, 0.0, CAP_ABOVE_HANDLE])
                seen = vision.find_cap(cam, img, nominal)
                if seen is None:
                    log(f"LOOK clamp {'gl'[c]}: cap not seen, using the layout position", flush=True)
                    continue
                off = (seen - nominal)[:2]
                off = np.clip(off, -0.02, 0.02)
                shared["handles"][c] = park_handles[c] + np.array([off[0], off[1], 0.0])
                log(f"LOOK clamp {'gl'[c]}: cap seen {np.round(seen[:2], 4)}, layout {np.round(nominal[:2], 4)}",
                    flush=True)
            ledge_z = props.BOARD_Z[1] + props.LEDGE_T
            tag = vision.find_tag(cam, img, props.LEDGE_X, props.BOARD_Y, ledge_z)
            if tag is not None:
                i = int(np.argmin([abs(tag[1] - y) for y in PHASE_Y]))
                if abs(tag[1] - PHASE_Y[i]) < 0.03:
                    shared["tagged_y"] = PHASE_Y[i]
                log(f"LOOK tag seen at y {tag[1]:.3f}: bar {PHASES[i]}", flush=True)
            else:
                log("LOOK tag not seen", flush=True)
            shared["looked"] = True

        def pick(arm, c):
            """Take clamp c's stick by its handle, fingers down (closing across the handle along y), and close. If the
            fingers close past the handle's width (missed), lift and try again."""
            hpos = shared["handles"][c]
            gq = down_yaw(0.0)  # the handle's square faces the axes on its plate (layout)
            dz = APPROACH_DZ[arm.side]
            for attempt in range(3):
                log(f"PICK {arm.side} clamp {'gl'[c]} attempt {attempt}: handle (estimate) {np.round(hpos, 4)}",
                    flush=True)
                arm.grip = OPEN
                yield from move(arm, hpos + np.array([0.0, 0.0, dz]), gq, speed=0.10, min_steps=steps(2.0))
                yield from settle(arm, tol=0.002, max_s=3.0)
                yield from move(arm, hpos, speed=0.03)
                yield from settle(arm, tol=0.002, max_s=2.0)
                yield from grip(arm, CLOSED, steps(1.0))
                yield from settle(arm)
                opening = gripper_opening(arm.side)
                log(f"GRIP {arm.side}: opening {opening:.3f}", flush=True)
                if opening > HOLDING_MIN:
                    break
                yield from grip(arm, OPEN, steps(0.6))
                yield from move(arm, arm.pos + np.array([0.0, 0.0, 0.05]), speed=0.05)
                yield from hold(arm, steps(1.0))
            # The grasp: handle across the fingers at the grasp point's y, along them at the camera's x, at the
            # layout's height; the frame hangs FRAME_BELOW_HANDLE under it, unturned (frame axes = desk axes).
            g_pos, g_quat = grasp(arm.side)
            handle = np.array([hpos[0], g_pos[1], HANDLE_Z])
            frame = handle - np.array([0.0, 0.0, FRAME_BELOW_HANDLE])
            arm.model = GraspModel(rotate(conj(g_quat), frame - g_pos), qmul(conj(g_quat), IDENT))
            log(f"HELD {arm.side}: frame in grasp frame (mm) {np.round(arm.model.rel_pos * 1000, 1)}", flush=True)

        def carry(arm, c, bar_y):
            """Lift the clamp 1 mm and carry it out in front of the bar at bar_y (BACK_OFF in front of its edge)."""
            f0 = frame_estimate(arm)
            home = np.array(props.frame_home(bar_y))
            lift = f0 + np.array([0.0, 0.0, 0.001])
            out_here = np.array([home[0] - BACK_OFF, f0[1], f0[2] + 0.001])
            out_there = np.array([home[0] - BACK_OFF, bar_y, home[2] + INSERT_LIFT + 0.002])
            for target, speed in ((lift, 0.02), (out_here, 0.04), (out_there, 0.05)):
                g, q = arm.model.grasp_for(target, IDENT)
                yield from move(arm, g, q, speed=speed)
                if target is lift:
                    yield from settle(arm, max_s=1.0)
            yield from hold(arm, steps(0.4))
            yield from settle(arm)
            log(f"CARRIED {arm.side}: frame (estimate) {np.round(frame_estimate(arm), 4)} "
                f"(target {np.round(out_there, 4)})", flush=True)

        def hang(arm, bar_y):
            """Line the clamp up in front of the bar (closed loop on the frame estimate), slide it on, let go."""
            home = np.array(props.frame_home(bar_y))
            target_out = np.array([home[0] - BACK_OFF, bar_y, home[2] + INSERT_LIFT])
            correction = np.zeros(3)
            for attempt in range(3):
                err = frame_estimate(arm) - target_out
                log(f"HANG {arm.side} attempt {attempt}: frame error (estimate, mm) {np.round(err * 1000, 1)}",
                    flush=True)
                correction = correction + err
                g, q = arm.model.grasp_for(target_out - correction, IDENT)
                yield from move(arm, g, q, speed=0.02)
                yield from hold(arm, steps(0.4))
                yield from settle(arm, tol=0.002)
            # Slide on in short increments, re-estimating the frame and correcting its height and side error after
            # each (the arm sags differently as it reaches out; the pad clears the bar's top by only INSERT_LIFT).
            for dx in (0.02, 0.035, 0.045, 0.05, 0.055, 0.06, 0.06):
                target = target_out + np.array([dx, 0.0, 0.0])
                g, q = arm.model.grasp_for(target - correction, IDENT)
                yield from move(arm, g, q, speed=0.012)
                yield from hold(arm, steps(0.5))
                err = frame_estimate(arm) - target
                correction = correction + err * np.array([0.5, 1.0, 1.0])
                log(f"INSERT {arm.side} dx {dx * 1000:.0f} mm: frame error (estimate, mm) {np.round(err * 1000, 1)}",
                    flush=True)
            f = frame_estimate(arm)
            log(f"HUNG {arm.side}: frame (estimate) vs home (mm) {np.round((f - home) * 1000, 1)}", flush=True)
            handle_yaw = grip_yaw(grasp(arm.side)[1])
            yield from grip(arm, OPEN, steps(0.8))
            yield from hold(arm, steps(0.8))
            return np.array([f[0], f[1], HANDLE_Z]), handle_yaw

        def retreat(arm):
            """Open hand straight up off the handle, then back toward the robot, well clear of both sticks."""
            yield from move(arm, arm.pos + np.array([0.0, 0.0, 0.05]), speed=0.04)
            yield from move(arm, arm.pos + RETREAT[arm.side], speed=0.05)
            yield from settle(arm, tol=0.005, max_s=2.0)
            log(f"RETREAT {arm.side}: grasp {np.round(grasp(arm.side)[0], 3)}", flush=True)

        def tighten(arm, bar_y, hpos, hyaw):
            """Turn the stick in strokes until the counted turns and the felt resistance say tight. hpos/hyaw: the
            handle's position and yaw as the arm last held it."""
            home = np.array(props.frame_home(bar_y))
            strokes, turns, lag0, lag = 0, 0.0, None, 0.0
            tight = False
            while True:
                if strokes:
                    loaded = lag0 is not None and lag - lag0 >= LAG_LOADED_DEG
                    log(f"STROKE {arm.side} #{strokes}: counted turns {turns:.3f}, end lag {lag:.2f} deg "
                        f"(first {lag0:.2f}), loaded {loaded}", flush=True)
                    if turns >= TURNS_TIGHT and loaded:
                        tight = True
                        break
                    if turns >= TURNS_GIVE_UP or strokes >= MAX_STROKES:
                        break
                strokes += 1
                yaw0 = stroke_start(arm.side, hyaw)
                yaw1 = yaw0 - STROKE_DEG
                arm.grip = OPEN
                yield from move(arm, hpos, down_yaw(yaw0), speed=0.03, min_steps=steps(0.8))
                yield from grip(arm, CLOSED, steps(0.8))
                y_closed = grip_yaw(grasp(arm.side)[1])
                # Re-seat: push the screw axis back over the bar (each stroke drags the clamp a little), then turn
                # about it.
                seat = np.array([home[0], home[1], hpos[2]]) + PRESS
                yield from move(arm, seat, down_yaw(yaw0), speed=0.01, min_steps=steps(0.4))
                yield from move(arm, seat, down_yaw(yaw1), min_steps=steps(STROKE_S))
                yield from hold(arm, steps(0.3))
                g_pos, g_quat = grasp(arm.side)
                y_end = grip_yaw(g_quat)
                lag = _wrap(y_end - yaw1)
                lag0 = lag if lag0 is None else lag0
                turns += _wrap(y_closed - y_end) / 360.0
                hyaw = y_end
                hpos = np.array([g_pos[0], g_pos[1], HANDLE_Z])
                self.debug[f"turns_{arm.side}"] = turns
                yield from grip(arm, OPEN, steps(0.6))
                yield from move(arm, hpos, down_yaw(yaw0), min_steps=steps(0.8))
            yield from grip(arm, OPEN, steps(0.6))
            yield from retreat(arm)
            log(f"DONE {arm.side}: {'tight' if tight else 'NOT judged tight'} after {strokes} strokes, counted turns "
                f"{turns:.3f}", flush=True)
            return tight

        def script_b(arm):  # right arm: the ground clamp
            yield from hold(arm, steps(1.0))
            yield from rise_and_look(arm, 0)
            yield from pick(arm, 0)
            yield from carry(arm, 0, GROUND_Y)
            hpos, hyaw = yield from hang(arm, GROUND_Y)
            shared["ground_tight"] = yield from tighten(arm, GROUND_Y, hpos, hyaw)

        def script_a(arm):  # left arm: the line clamp
            yield from hold(arm, steps(1.0))
            yield from rise_and_look(arm, 1)
            bar_y = shared["tagged_y"]
            if bar_y is None:
                log("LINE: no TESTED tag seen; the line clamp stays on its plate", flush=True)
                return
            yield from pick(arm, 1)
            yield from carry(arm, 1, bar_y)
            while not shared["ground_tight"]:
                yield
            log("LINE: ground clamp tight (by the right arm's own count and feel); hanging the line clamp", flush=True)
            hpos, hyaw = yield from hang(arm, bar_y)
            if control:
                yield from retreat(arm)
                log("CONTROL: line clamp left hung, not tightened", flush=True)
                return
            yield from tighten(arm, bar_y, hpos, hyaw)

        self._look = look
        self._shared = shared
        self._steps = steps
        self._make = lambda: (Arm("left"), Arm("right"))
        self._scripts = (script_a, script_b)
        self._started = False
        self._look_wait = 0

        tcp_extra = vision.TCP_EXTRA
        q_rel = np.array(vision.Q_REL)
        worst = {"left": 0.0, "right": 0.0}
        SEEDS = [np.array([0.0, 1.0, 1.0, -0.6, 0.0, 0.0])]
        k_step = [0]
        q_arm = {}

        def joint_action(arm):
            pos = arm.pos + np.array([0.0, 0.0, vision.Z0])
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
        self._q_arm = q_arm
        self._k = k_step

    def next_action(self, observation) -> np.ndarray:
        self.obs = observation
        if not self._started:
            self._A, self._B = self._make()
            for side in ("left", "right"):
                self._q_arm[side] = np.asarray(observation.state["joint_pos"], float)[ARM_KEYS[side]][:6].copy()
            script_a, script_b = self._scripts
            self._gens = {"left": script_a(self._A), "right": script_b(self._B)}
            self._alive = {"left": True, "right": True}
            self._started = True
        sh = self._shared
        if not sh["looked"] and sh["waiting"] >= {"left", "right"}:
            self._look_wait += 1
            if self._look_wait >= self._steps(0.3):  # hands still, then one look
                self._look(np.asarray(observation.images["top_cam"]))
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
