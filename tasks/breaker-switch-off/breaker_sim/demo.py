"""The scripted demo, closed loop.

The arm on the named breaker's side (way 4 is right of the midline: arm B, the right arm) closes its fingers,
holds them level pointing at the board, rests the underside of the fingers' ends on top of the handle's grip bar
and follows the handle's arc down past centre to the OFF stop, correcting y every step from its measured grasp
point (the neighbouring bars are 5 mm away). It then backs straight out and parks; once its fingers have been
4 cm clear for 1 s the after-release check runs.

Grasp-point targets become yam_arena's 14-D joint action through yam_arena's own IK (``yam_arena.ik``) every control
step. It reads the simulator's state (the handle angles) and knows which way is named, like any scripted demo with
privileged state; ``--policy breaker-demo`` (breaker_sim.policy) plays it through Inspect Robots. Import after Isaac
Sim has started. With BREAKER_CONTROL=1 it pushes the named handle only 30% of its throw (short of centre) and lets
go: the detent must spring it back to ON.
"""

from __future__ import annotations

import math
import os

import numpy as np

from yam_arena import ik, station

CONTROL = os.environ.get("BREAKER_CONTROL") == "1"

TIP_BEYOND_GRASP = 0.012
"""Grasp point (between the pads) to the fingers' ends."""
FINGER_HALF_WIDTH = 0.008
"""Grasp point to the closed fingers' underside (level fingers)."""
ARC_STEP_RAD = math.radians(3.0)


class Demo:
    """``next_action()`` gives the 14-D joint action for the next control step."""

    def __init__(self, STATION, control_hz: float = 20.0, speed: float = 0.10, log=print):
        from breaker_sim import station as hb
        from breaker_sim.kit import CLOSED, FORWARD, ROOT_OFFSET, ArmTimeline, qmul, rotate
        from breaker_sim.kit import grasp_point as grasp_point_world
        from breaker_sim.spec import FACE_X, RAIL_Z
        from breaker_sim.usd import board_props as props

        view = STATION.env
        task = STATION.task
        st = task.state(view)
        state_k = [0]
        self.state_k = state_k
        origin = view.scene.env_origins[0].cpu().numpy()
        robot = view.scene["robot"]
        dt = 1.0 / control_hz
        hold_k = lambda s: max(1, round(s / dt))  # noqa: E731
        Q_ON = props.Q_THROW
        named = int(st.named[0])
        fraction = 0.3 if CONTROL else 1.0
        L_PARK, R_PARK = np.array([0.17, 0.20, 0.22]), np.array([0.17, -0.20, 0.22])

        def grasp_point(side):
            p, q = grasp_point_world(robot, side)
            return p - origin, q

        def note(msg):
            log(msg, flush=True)

        def rot_y(q, v):
            c, s = math.cos(q), math.sin(q)
            return np.array([c * v[0] + s * v[2], v[1], -s * v[0] + c * v[2]])

        def pivot(slot):
            x, y, z = props.pivot_local(slot)
            return np.array([FACE_X + x, y, RAIL_Z + z])

        def handle_q(slot):
            return float(st.board.data.joint_pos.torch[0, st.handle_joints[slot]])

        def contact(slot, q, press):
            """A point on top of the grip bar, 1 mm beyond its centre, `press` m into it."""
            local = np.array([-(props.HANDLE_LEN + 0.001), 0.0, props.HANDLE_T / 2 - press])
            return pivot(slot) + rot_y(q, local)

        def away(q):
            return rot_y(q, np.array([0.0, 0.0, 1.0]))

        def tip_to_grasp(point):
            """Level fingers: their underside, 5 mm short of their ends, on the point."""
            return point + np.array([-(TIP_BEYOND_GRASP - 0.005), 0.0, FINGER_HALF_WIDTH])

        class Planner:
            def __init__(self, L, R):
                self.L, self.R, self.events = L, R, {}

            def at(self, step, fn):
                self.events.setdefault(step, []).append(fn)

            def run_due(self, k):
                for fn in self.events.pop(k, []):
                    fn()

            def sync(self):
                n = max(len(self.L.steps), len(self.R.steps))
                self.L.pad_to(n)
                self.R.pad_to(n)
                return n

            def arm(self, slot):
                return (self.L, "left") if pivot(slot)[1] > 0 else (self.R, "right")

            def when_settled(self, tl, side, fn, tries=30, tol=0.003):
                def check():
                    lag = np.linalg.norm(grasp_point(side)[0] - (tl.pos + ROOT_OFFSET))
                    if lag < tol or check.tries <= 0:
                        if lag >= tol:
                            note(f"SETTLE timeout ({side}): lag {lag * 1000:.1f} mm")
                        fn()
                    else:
                        check.tries -= 1
                        tl.hold(hold_k(0.1))
                        self.at(len(tl.steps), check)

                check.tries = tries
                self.at(len(tl.steps), check)

            def operate(self, attempt=0):
                T, side = self.arm(named)
                T.grip_to(CLOSED, hold_k(0.4))
                start = contact(named, Q_ON, -0.004) + 0.008 * away(Q_ON)
                T.move(tip_to_grasp(start + np.array([-0.06, 0.0, 0.0])), FORWARD, min_steps=hold_k(2.5))
                T.move(tip_to_grasp(start), speed=0.05)
                T.hold(hold_k(0.3))
                self.cy = 0.0
                self.when_settled(T, side, lambda: self.approach(attempt), tol=0.002)

            def approach(self, attempt):
                T, side = self.arm(named)
                err = grasp_point(side)[0] - (T.pos + ROOT_OFFSET)
                note(f"APPROACH attempt {attempt}: grasp-point error (mm) {np.round(err * 1000, 1)}")
                self.corr = np.array([0.0, -err[1], -err[2]])
                T.move(tip_to_grasp(contact(named, Q_ON, -0.001)) + self.corr, speed=0.02)
                T.hold(hold_k(0.2))
                self.at(self.sync(), lambda: self.arc(attempt))

            def arc(self, attempt):
                q_end = Q_ON - fraction * 2 * Q_ON
                if fraction >= 1.0:
                    q_end -= math.radians(3.0)
                n = max(3, int(math.ceil(abs(q_end - Q_ON) / ARC_STEP_RAD)))
                self.arc_step(attempt, q_end, n, 1)

            def arc_step(self, attempt, q_end, n, k):
                T, side = self.arm(named)
                self.corr[1] = float(np.clip(self.corr[1] + 0.8 * (pivot(named)[1] - grasp_point(side)[0][1]),
                                             -0.01, 0.01))
                if k > n:
                    T.hold(hold_k(0.15))
                    self.at(self.sync(), lambda: self.check(attempt, q_end))
                    return
                q = Q_ON + (q_end - Q_ON) * k / n
                press = 0.003 if fraction >= 1.0 else 0.0005
                # The position-controlled hand sags and lags a few mm under yam_arena's gains: integrate the height
                # error so the fingers really reach the bar (and press it with the detent's force).
                if k > 1:
                    want = tip_to_grasp(contact(named, Q_ON + (q_end - Q_ON) * (k - 1) / n, press))
                    gain = 0.5 if fraction >= 1.0 else 0.25
                    self.corr[2] = float(np.clip(self.corr[2] + gain * (want[2] - grasp_point(side)[0][2]),
                                                 -0.04, 0.015))
                T.move(tip_to_grasp(contact(named, q, press)) + self.corr, min_steps=3)
                self.at(self.sync(), lambda: self.arc_step(attempt, q_end, n, k + 1))

            def check(self, attempt, q_end):
                T, side = self.arm(named)
                q = handle_q(named)
                note(f"ARC attempt {attempt}: named handle at {math.degrees(q):.1f} deg (fraction {fraction}); "
                     f"level {task.episode_report(view, 0)['milestone']}")
                back = contact(named, q_end, -0.004) + 0.006 * away(q_end)
                if fraction >= 1.0 and q > -math.radians(25.0) and attempt < 2:
                    T.move(tip_to_grasp(back + np.array([-0.02, 0.0, 0.02])), speed=0.04)
                    self.operate(attempt + 1)
                    return
                T.move(tip_to_grasp(back) + self.corr, speed=0.02)
                T.move(tip_to_grasp(back + np.array([-0.03, 0.0, 0.0])), speed=0.015)
                T.move(tip_to_grasp(back + np.array([-0.08, 0.0, 0.0])), speed=0.05)
                T.move(R_PARK if side == "right" else L_PARK, speed=0.08)
                self.sync()
                T.hold(hold_k(4.0))
                self.at(len(T.steps), lambda: note(f"DONE: {task.episode_report(view, 0)}"))

        l_pos, l_quat = grasp_point("left")
        r_pos, r_quat = grasp_point("right")
        L = ArmTimeline(l_pos - ROOT_OFFSET, l_quat, dt, speed)
        R = ArmTimeline(r_pos - ROOT_OFFSET, r_quat, dt, speed)
        planner = Planner(L, R)
        settle = hold_k(1.0)
        L.hold(settle)
        R.hold(settle)
        planner.at(settle, planner.operate)

        q_rel = np.array(hb.Q_REL)
        tcp_extra = 0.1347 - 0.134
        q_arm = {s: view.scene[f"{s}_robot"].data.joint_pos.torch[0, :6].cpu().numpy().astype(float)
                 for s in ("left", "right")}
        worst = {"left": 0.0, "right": 0.0}

        def joint_action(side, pos_root, quat, grip):
            pos = pos_root + ROOT_OFFSET + origin
            tcp = pos + rotate(quat, (0.0, 0.0, -tcp_extra))
            rot = qmul(quat, q_rel)
            q, pe, _ = ik.solve(tcp, rot, q_arm[side], base_pos=station.BASES[side], iters=60)
            q_arm[side] = q
            if pe > 0.005 and pe > worst[side]:
                log(f"IK {side} k={state_k[0]} error {pe * 1000:.1f} mm at {np.round(pos - origin, 4)}")
            worst[side] = max(worst[side], pe)
            return np.r_[q, grip]

        self._L, self._R, self._planner, self._joint_action = L, R, planner, joint_action
        self.worst = worst

    def next_action(self) -> np.ndarray:
        k = self.state_k[0]
        L, R = self._L, self._R
        self._planner.run_due(k)
        L.pad_to(len(R.steps))
        R.pad_to(len(L.steps))
        if k >= len(L.steps):
            L.hold(1)
            R.hold(1)
        (lp, lq, lg), (rp, rq, rg) = L.steps[k], R.steps[k]
        self.state_k[0] = k + 1
        return np.concatenate([self._joint_action("left", lp, lq, lg), self._joint_action("right", rp, rq, rg)])
