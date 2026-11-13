"""The scripted demo, closed loop on what the robot itself observes.

Inputs, every control step: the policy's observation only, i.e. the three camera images (``top_cam``, ``left_cam``,
``right_cam``) and the 14-D ``joint_pos`` (both arms' joint angles and gripper openings). It never reads the
simulator (no prop poses, joint angles, groove depths or scorer state). Besides the observation it uses the documented
setup (tube position, mark height, the cutter's stated dimensions: task.json ``setup``), the station's camera
calibration and the arm's URDF (both from yam_arena), and time.

- Where the cutter points. One look with the top camera at the start: the black knob's pixels in a ring round the
  tube, back-projected onto the knob's top; the direction from the tube's axis to their centroid is the cutter's
  angle about the tube (the setup says the knob points toward the left arm; the look corrects it).
- Where it points later. While a hand holds the cutter's body, the body's angle about the tube is the angle of the
  hand's grasp point about the tube's axis (forward kinematics of the measured joints). After each stroke that angle
  is where the next stroke starts.
- How far the knob has been turned: the wrist's measured roll about the knob's axis while the fingers hold the knob
  (the knob turns with the fingers; any slip shows as roll the knob did not follow, which is why it is measured, not
  commanded). Feed is the counted knob turns x the stated 1/4-turn-per-revolution guidance.
- Whether a grasp holds: the gripper's measured opening (it stops on the 4 cm body at about 0.43, on the knob at
  about 0.33, instead of closing).

The plan, per revolution (three in all): the left hand comes down over the knob with its fingers horizontal along the
knob's axis (the arm cannot reach a stand-off out along the axis), and turns it a quarter turn clockwise in two 45 deg
wrist strokes (close across the knob, roll, open, roll back), then lifts away; then the cutter is taken round the tube once in
four 90 deg strokes (clockwise seen from above), each a fingers-down grasp of the cutter's body, an arc about the
tube's axis, release and lift: the left hand for the strokes on its side, the right for the two on its side. The
wrist cannot turn a full turn, so the cutter changes hands every half turn. After the third revolution both hands
move back and wait.

``control=True``: the knob is turned half a turn (two quarter-turn grasps) before the first revolution, twice the
guidance; the tube is dented and the score must be 0.

Grasp-point targets become yam_arena's 14-D joint action through yam_arena's own IK every control step. Import after
Isaac Sim has started.
"""

from __future__ import annotations

import math

import numpy as np

from yam_arena import ik, station

ARM_KEYS = {"left": slice(0, 7), "right": slice(7, 14)}
REVOLUTIONS = 3
FEED_TURN = 0.25
"""Knob turn per revolution (the maker's guidance, Reed TC2Q)."""
STROKE_DEG = 90.0
STROKE_S = 2.0
HOVER_DZ = 0.065
HOLDING_MIN = 0.15
"""Gripper opening (0 closed, 1 open) above which the fingers have stopped on the body or knob."""
KNOB_ABOVE = 0.05
"""The fingers come down onto the knob from this far above it."""
KNOB_HIGH = 0.13
"""The hand turns between fingers-down and the side grasp this far above the knob (clear of the cutter)."""
FEED_STROKE_DEG = 45.0
FEED_AT_DEG = 85.0
"""After each revolution the cutter is brought round to this angle (knob toward the left arm, 5 deg past the start so
the revolution overlaps) before the next feed."""
STROKES = (("left", 90.0), ("right", 0.0), ("right", -90.0), ("left", -180.0))
"""(arm, nominal start angle of the cutter about the tube, deg) for the four strokes of one revolution."""


def _wrap(deg: float) -> float:
    return (deg + 180.0) % 360.0 - 180.0


class Demo:
    """``next_action(observation)`` gives the 14-D joint action for the next control step."""

    def __init__(self, width: int = 320, height: int = 176, control_hz: float = 20.0, control: bool = False,
                 log=print):
        from tubecutter_sim import vision
        from tubecutter_sim.kit import CLOSED, DOWN, FORWARD, OPEN, conj, q_axis, qmul, rotate, slerp
        from tubecutter_sim.spec import KNOB_START_DEG, MARK_Z, TUBE_X, TUBE_Y
        from tubecutter_sim.usd import cutter_props as props

        dt = 1.0 / control_hz
        self.finished = False
        self.log = log
        self.obs = None
        steps = lambda s: max(1, round(s / dt))  # noqa: E731
        cam = vision.TopCamera(width, height)
        tube = np.array([TUBE_X, TUBE_Y])
        GRASP_Z = MARK_Z
        shared = {"phi": KNOB_START_DEG, "knob_turns": 0.0}
        self.debug = shared

        def joints(side):
            return np.asarray(self.obs.state["joint_pos"], float)[ARM_KEYS[side]]

        def grasp(side):
            return vision.grasp_pose(joints(side)[:6], side)

        def opening(side):
            return float(joints(side)[6])

        class Arm:
            def __init__(self, side):
                self.side = side
                self.pos, self.quat = grasp(side)
                self.grip = OPEN
                self.override = None

        def down_yaw(deg):
            return qmul(q_axis((0, 0, 1), math.radians(deg)), DOWN)

        def side_grasp(phi, roll):
            """Fingers horizontal, pointing at the tube along the knob's axis (cutter angle phi), rolled by roll."""
            return qmul(q_axis((0, 0, 1), math.radians(phi + 180.0)), qmul(FORWARD, q_axis((0, 0, 1),
                                                                                           math.radians(roll))))

        def radial(phi, r, z):
            a = math.radians(phi)
            return np.array([tube[0] + r * math.cos(a), tube[1] + r * math.sin(a), z])

        def yaw_for(phi, sweep):
            """Wrist yaw (deg) closing the fingers across the body at angle phi, chosen (mod 180) so the whole stroke
            from phi to phi - sweep stays inside the wrist's range."""
            best = None
            for k in (-2, -1, 0, 1, 2):
                y0 = phi + 180.0 * k
                y1 = y0 - sweep
                worst = max(abs(y0), abs(y1))
                if best is None or worst < best[0]:
                    best = (worst, y0)
            return best[1]

        # Behaviours -------------------------------------------------------------------------------------------------
        def move(arm, pos, quat=None, speed=0.10, min_steps=3):
            pos = np.array(pos, float)
            quat = arm.quat if quat is None else np.array(quat, float)
            p0, q0 = arm.pos.copy(), arm.quat.copy()
            n = max(min_steps, int(math.ceil(np.linalg.norm(pos - p0) / (speed * dt))))
            for k in range(1, n + 1):
                a = k / n
                arm.pos, arm.quat = p0 + a * (pos - p0), slerp(q0, quat, a)
                yield

        def joint_move(arm, pos, quat, seconds):
            """Go to a pose by interpolating joint angles to an IK solution found once from several seeds (for a
            posture change, e.g. fingers down to the side grasp, that step-by-step Cartesian IK cannot follow)."""
            q_goal = solve_pose(arm.side, pos, quat)
            q0 = self._q_arm[arm.side].copy()
            n = steps(seconds)
            arm.override = None
            for k in range(1, n + 1):
                a = k / n
                arm.override = q0 + (q_goal - q0) * (0.5 - 0.5 * math.cos(math.pi * a))
                yield
            arm.override = None
            arm.pos, arm.quat = np.array(pos, float), np.array(quat, float)

        def hold(n):
            for _ in range(n):
                yield

        def grip(arm, value, n):
            arm.grip = value
            yield from hold(n)

        def settle(arm, tol=0.003, max_s=2.0):
            for _ in range(steps(max_s)):
                if np.linalg.norm(grasp(arm.side)[0] - arm.pos) < tol:
                    return
                yield

        def park(arm):
            """Out of the way: up and back toward the arm's own side."""
            y = 0.20 if arm.side == "left" else -0.20
            up = np.array([arm.pos[0], arm.pos[1], max(arm.pos[2], GRASP_Z + HOVER_DZ)])
            yield from move(arm, up, speed=0.15)
            yield from move(arm, np.array([0.16, y, GRASP_Z + HOVER_DZ + 0.03]), down_yaw(0.0), speed=0.25)

        def prehover(arm):
            """Wait out of the way of the other hand, 6 cm further out and 3 cm higher than the next grasp."""
            phi = shared["phi"]
            yaw0 = yaw_for(phi, STROKE_DEG)
            yield from move(arm, radial(phi, props.GRASP_U + 0.06, GRASP_Z + HOVER_DZ + 0.03), down_yaw(yaw0),
                            speed=0.25, min_steps=steps(0.6))

        def look(img):
            seen = vision.find_knob_angle(cam, img, tube, (props.KNOB_U0 - 0.01, props.KNOB_U0 + props.KNOB_L + 0.02),
                                          MARK_Z + props.KNOB_R)
            if seen is None or abs(_wrap(seen - shared["phi"])) > 40.0:
                log(f"LOOK cutter angle {seen}: not usable, keeping {shared['phi']:.1f}", flush=True)
                return
            log(f"LOOK cutter angle {seen:.1f} deg (was {shared['phi']:.1f})", flush=True)
            shared["phi"] = seen

        def stroke(arm, sweep=STROKE_DEG):
            """One stroke of ``sweep`` deg clockwise (seen from above) from the cutter's current angle."""
            phi0 = shared["phi"]
            yaw0 = yaw_for(phi0, sweep)
            r = props.GRASP_U
            for attempt in range(3):
                above = radial(phi0, r, GRASP_Z + HOVER_DZ)
                yield from move(arm, above, down_yaw(yaw0), speed=0.20, min_steps=steps(0.6))
                arm.grip = OPEN
                yield from settle(arm, tol=0.003, max_s=1.0)
                yield from move(arm, radial(phi0, r, GRASP_Z), speed=0.08)
                yield from settle(arm, tol=0.002, max_s=1.0)
                yield from grip(arm, CLOSED, steps(0.5))
                if opening(arm.side) > HOLDING_MIN:
                    break
                log(f"STROKE {arm.side}: missed the body (opening {opening(arm.side):.3f}), retry", flush=True)
                yield from grip(arm, OPEN, steps(0.5))
                yield from move(arm, radial(phi0, r, GRASP_Z + HOVER_DZ), speed=0.06)
            g = grasp(arm.side)[0]
            phi_held = math.degrees(math.atan2(g[1] - tube[1], g[0] - tube[0]))
            r_held = float(np.hypot(*(g[:2] - tube)))
            n = steps(STROKE_S)
            for k in range(1, n + 1):
                phi = phi_held - sweep * k / n
                arm.pos = radial(phi, r_held, arm.pos[2])
                arm.quat = down_yaw(yaw0 - (phi_held - phi))
                yield
            yield from hold(steps(0.4))
            g = grasp(arm.side)[0]
            shared["phi"] = math.degrees(math.atan2(g[1] - tube[1], g[0] - tube[0]))
            log(f"STROKE {arm.side}: cutter {phi_held:.1f} -> {shared['phi']:.1f} deg (r {r_held * 1000:.1f} mm, "
                f"opening {opening(arm.side):.3f})", flush=True)
            yield from grip(arm, OPEN, steps(0.4))
            yield from move(arm, arm.pos + np.array([0.0, 0.0, HOVER_DZ]), speed=0.12)

        def feed(arm, quarter_turns=1):
            """Turn the knob clockwise (seen from its end) by quarter turns, each in two 45 deg wrist strokes. Each
            stroke closes the fingers across the knob horizontally (so closing does not push the cutter along the
            tube), rolls 45 deg, opens, and rolls back with the fingers open (they clear the knob)."""
            phi = shared["phi"]
            knob = radial(phi, props.KNOB_CENTER_U, GRASP_Z)
            above = knob + np.array([0.0, 0.0, KNOB_ABOVE])
            high = knob + np.array([0.0, 0.0, KNOB_HIGH])
            total = 0.0
            arm.grip = OPEN
            # Over the knob, high and fingers down; turn the hand into the side grasp up there (fingers horizontal
            # along the knob's axis, closing across it horizontally); then down over the knob (the open fingers pass
            # either side of it: the arm cannot reach a stand-off out along the knob's axis).
            up = np.array([arm.pos[0], arm.pos[1], max(arm.pos[2], high[2])])
            yield from move(arm, up, speed=0.15)
            yield from move(arm, high, down_yaw(phi + 90.0), speed=0.20, min_steps=steps(0.8))
            yield from settle(arm, tol=0.004, max_s=1.0)
            yield from joint_move(arm, high, side_grasp(phi, 0.0), 1.5)
            yield from move(arm, above, speed=0.10)
            yield from settle(arm, tol=0.003, max_s=1.0)
            yield from move(arm, knob, speed=0.03)
            yield from settle(arm, tol=0.002, max_s=1.5)
            for _ in range(2 * quarter_turns):
                yield from grip(arm, CLOSED, steps(0.5))
                if opening(arm.side) < HOLDING_MIN:
                    log(f"FEED {arm.side}: knob not held (opening {opening(arm.side):.3f})", flush=True)
                q_start = grasp(arm.side)[1]
                n = steps(1.2)
                for k in range(1, n + 1):
                    arm.quat = side_grasp(phi, -FEED_STROKE_DEG * k / n)
                    yield
                yield from hold(steps(0.4))
                q_end = grasp(arm.side)[1]
                rel = qmul(conj(q_start), q_end)
                turned = -_wrap(math.degrees(2 * math.atan2(rel[2], rel[3]))) / 360.0
                total += turned
                log(f"FEED {arm.side}: wrist rolled {turned * 360:.1f} deg (knob turn {turned:.3f}), "
                    f"opening {opening(arm.side):.3f}", flush=True)
                yield from grip(arm, OPEN, steps(0.5))
                n = steps(0.8)
                for k in range(1, n + 1):
                    arm.quat = side_grasp(phi, -FEED_STROKE_DEG * (1 - k / n))
                    yield
                yield from settle(arm, tol=0.002, max_s=0.6)
            yield from move(arm, above, speed=0.06)
            yield from move(arm, high, speed=0.12)
            yield from joint_move(arm, high, down_yaw(phi + 90.0), 1.5)
            shared["knob_turns"] += total
            return total

        def plan(A, B):
            arms = {"left": A, "right": B}
            yield from hold(steps(0.5))
            yield from both(park(A), park(B))
            yield from hold(steps(0.5))
            self._want_look = True
            while self._want_look:
                yield
            first = 2 if control else 1
            for rev in range(REVOLUTIONS):
                yield from feed(A, first if rev == 0 else 1)
                log(f"REV {rev}: knob turns so far {shared['knob_turns']:.3f}", flush=True)
                for i, (side, _nominal) in enumerate(STROKES):
                    yield from stroke(arms[side])
                    nxt = STROKES[i + 1][0] if i + 1 < len(STROKES) else None
                    if nxt is not None and nxt != side:
                        yield from both(park(arms[side]), prehover(arms[nxt]))
                # The grip slips a few degrees each stroke: look where the knob is (both hands parked) and, if the
                # cutter has not yet come all the way round to the feed position, finish the turn with the left hand.
                if rev == REVOLUTIONS - 1:
                    break  # the last revolution needs no feed position afterwards
                yield from both(park(A), park(B))
                yield from hold(steps(0.3))
                self._want_look = True
                while self._want_look:
                    yield
                rest = _wrap(shared["phi"] - FEED_AT_DEG)
                if rest > 3.0:
                    log(f"REV {rev}: cutter at {shared['phi']:.1f} deg, {rest:.1f} deg short: finishing the turn",
                        flush=True)
                    yield from stroke(A, sweep=rest)
                    yield from park(A)
            yield from both(park(A), park(B))
            log("DONE: both hands parked", flush=True)
            while True:
                yield

        def both(g1, g2):
            alive = [True, True]
            gens = [g1, g2]
            while any(alive):
                for i in (0, 1):
                    if alive[i]:
                        try:
                            next(gens[i])
                        except StopIteration:
                            alive[i] = False
                yield

        self._look = look
        self._plan = plan
        self._make = lambda: (Arm("left"), Arm("right"))
        self._started = False
        self._want_look = False

        tcp_extra = vision.TCP_EXTRA
        q_rel = np.array(vision.Q_REL)
        SEEDS = [np.array([0.0, 1.0, 1.0, -0.6, 0.0, 0.0]), np.array([0.5, 1.0, 1.0, -0.6, 0.0, 1.0]),
                 np.array([-0.5, 1.0, 1.0, -0.6, 0.0, -1.0])]
        q_arm = {}
        worst = {"left": 0.0, "right": 0.0}

        SIDE_SEED = np.array([0.0, 2.0, 0.5, 1.5, 1.5, 0.0])

        def solve_pose(side, pos, quat):
            tcp = np.asarray(pos, float) + np.array([0.0, 0.0, vision.Z0]) + rotate(quat, (0.0, 0.0, -tcp_extra))
            rot = qmul(quat, q_rel)
            best = None
            for seed in [q_arm[side], SIDE_SEED, *SEEDS]:
                q, pe, re = ik.solve(tcp, rot, seed, base_pos=station.BASES[side], iters=200)
                if best is None or pe + 0.05 * re < best[1] + 0.05 * best[2]:
                    best = (q, pe, re)
            if best[1] > 0.003:
                log(f"IK {side}: posture change target error {best[1] * 1000:.1f} mm", flush=True)
            return best[0]

        def joint_action(arm):
            if arm.override is not None:
                q_arm[arm.side] = np.asarray(arm.override, float).copy()
                return np.r_[q_arm[arm.side], arm.grip]
            pos = arm.pos + np.array([0.0, 0.0, vision.Z0])
            tcp = pos + rotate(arm.quat, (0.0, 0.0, -tcp_extra))
            rot = qmul(arm.quat, q_rel)
            q, pe, re = ik.solve(tcp, rot, q_arm[arm.side], base_pos=station.BASES[arm.side], iters=60)
            if pe > 0.002 or re > 0.02:
                for seed in SEEDS:
                    q2, pe2, re2 = ik.solve(tcp, rot, seed, base_pos=station.BASES[arm.side], iters=150)
                    if pe2 + 0.05 * re2 < pe + 0.05 * re:
                        q, pe, re = q2, pe2, re2
            q_arm[arm.side] = q
            if pe > 0.003 and pe > worst[arm.side] + 0.002:
                worst[arm.side] = pe
                log(f"IK {arm.side} error {pe * 1000:.1f} mm at {np.round(arm.pos, 4)}", flush=True)
            return np.r_[q, arm.grip]

        self._joint_action = joint_action
        self._q_arm = q_arm

    def next_action(self, observation) -> np.ndarray:
        self.obs = observation
        if not self._started:
            self._A, self._B = self._make()
            for side in ("left", "right"):
                self._q_arm[side] = np.asarray(observation.state["joint_pos"], float)[ARM_KEYS[side]][:6].copy()
            self._gen = self._plan(self._A, self._B)
            self._started = True
        if self._want_look:
            self._look(np.asarray(observation.images["top_cam"]))
            self._want_look = False
        next(self._gen)
        return np.concatenate([self._joint_action(self._A), self._joint_action(self._B)])
