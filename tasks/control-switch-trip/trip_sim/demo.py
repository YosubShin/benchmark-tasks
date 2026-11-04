"""The scripted demo, closed loop.

Arm B (the right arm; the named switch 52-2 is right of the midline) points its fingers level at the panel, opens
them around the pistol grip 40 mm below the shaft and closes them on it with a light squeeze (fingers commanded 2 mm
inside the grip's 19 mm width: about 10 N per finger under yam_arena's 5000 N/m finger gain). It then turns the
grip to TRIP by rolling the wrist about the switch's shaft while the hand follows the grip's arc (pinch-and-twist),
leading the handle by a few degrees so the grip is pressed into the TRIP stop. It holds there, watching the named
breaker's state (the lamps), and if the handle sags out of the TRIP contact it adds lead. 0.25 s after the green lamp
lights it opens the fingers and backs straight off; the spring returns the handle to NORMAL.

TRIP_DEMO=push uses the earlier fallback instead: closed fingers beside the grip's +y face push it round.
TRIP_CONTROL=1 is the control run: the demo lets go as soon as the handle reaches the TRIP contact, before the lamp
changes. The breaker then stays closed and the run must score below 1.

Grasp-point targets become yam_arena's 14-D joint action through yam_arena's own IK every control step. The demo
reads the simulator's state (handle angle, breaker state) and knows which switch is named, like any scripted demo
with privileged state; ``--policy trip-demo`` (trip_sim.policy) plays it through Inspect Robots. Import after Isaac
Sim has started.
"""

from __future__ import annotations

import math
import os

import numpy as np

from yam_arena import ik, station

CONTROL = os.environ.get("TRIP_CONTROL") == "1"
MODE = os.environ.get("TRIP_DEMO", "pinch")

ARC_STEP_RAD = math.radians(3.0)
HOLD_AFTER_GREEN_S = 0.25


class Demo:
    """``next_action()`` gives the 14-D joint action for the next control step."""

    def __init__(self, STATION, control_hz: float = 20.0, speed: float = 0.10, log=print):
        from trip_sim import station as hb
        from trip_sim.kit import CLOSED, FORWARD, OPEN, ROOT_OFFSET, ArmTimeline, q_axis, qmul, rotate
        from trip_sim.kit import grasp_point as grasp_point_world
        from trip_sim.spec import AXIS_Z, FACE_X, SWITCH_Y
        from trip_sim.task import CONTACT_RAD
        from trip_sim.usd import panel_props as props

        view = STATION.env
        task = STATION.task
        st = task.state(view)
        state_k = [0]
        self.state_k = state_k
        origin = view.scene.env_origins[0].cpu().numpy()
        robot = view.scene["robot"]
        dt = 1.0 / control_hz
        hold_k = lambda s: max(1, round(s / dt))  # noqa: E731
        slot = int(st.named[0])
        R_PARK = np.array([0.17, -0.20, 0.22])
        VIA = np.array([0.18, -0.21, 0.28])
        grip_x = FACE_X - props.HANDLE_DEPTH + props.GRIP_T / 2

        if MODE == "push":
            radius, off_y, turn_to, grip_cmd = 0.045, props.GRIP_W / 2 + 0.013 + 0.004, math.radians(-58.0), CLOSED
            x_in = 0.0
        else:
            # Each finger 2 mm inside the grip's half width: pads touch at 0, fully open at 0.04695 m per finger.
            radius, off_y, turn_to = 0.040, 0.0, math.radians(-48.0)
            grip_cmd = (props.GRIP_W / 2 - 0.002) / 0.04695
            x_in = 0.002
        max_lead = math.radians(-62.0)

        def note(msg):
            log(msg, flush=True)

        def grasp_point(side):
            p, q = grasp_point_world(robot, side)
            return p - origin, q

        def handle_q():
            return float(st.panel.data.joint_pos.torch[0, st.handle_joints[slot]])

        def tripped():
            return not bool(st.closed[0, slot])

        def hand_pose(q):
            """Grasp-point pose turning with the handle: level fingers (FORWARD) rolled by q about the shaft."""
            c, s = math.cos(q), math.sin(q)
            y_l, z_l = off_y, -radius
            pos = np.array([grip_x + x_in, SWITCH_Y[slot] + y_l * c - z_l * s, AXIS_Z + y_l * s + z_l * c])
            return pos, qmul(q_axis((1.0, 0.0, 0.0), q), FORWARD)

        class Planner:
            def __init__(self, L, R):
                self.L, self.R, self.events = L, R, {}
                self.corr = np.zeros(3)
                self.lead = turn_to

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

            def when_settled(self, fn, tries=30, tol=0.003):
                T = self.R

                def check():
                    lag = np.linalg.norm(grasp_point("right")[0] - (T.pos + ROOT_OFFSET))
                    if lag < tol or check.tries <= 0:
                        if lag >= tol:
                            note(f"SETTLE timeout: lag {lag * 1000:.1f} mm")
                        fn()
                    else:
                        check.tries -= 1
                        T.hold(hold_k(0.1))
                        self.at(len(T.steps), check)

                check.tries = tries
                self.at(len(T.steps), check)

            def start(self):
                T = self.R
                p0, r0 = hand_pose(0.0)
                # Arm B's ready pose has its hand beside the panel's right edge, level with the panel face. Go via a
                # point above and in front of switch 52-3, so the hand never sweeps across its grip or lamps. (Backing
                # straight off toward the robot puts yam_arena's IK on a branch that then stalls 5 cm short of the
                # grip, as it does offline.)
                T.move(VIA, r0, min_steps=hold_k(2.0))
                if MODE == "push":
                    T.grip_to(CLOSED, hold_k(0.4))
                T.move(p0 + np.array([-0.08, 0.0, 0.0]), r0, min_steps=hold_k(2.0))
                T.hold(hold_k(0.3))
                self.when_settled(self.approach)

            def approach(self):
                T = self.R
                err = grasp_point("right")[0] - (T.pos + ROOT_OFFSET)
                self.corr = np.clip(np.array([0.0, -err[1], -err[2]]), -0.01, 0.01)
                note(f"APPROACH: grasp-point error (mm) {np.round(err * 1000, 1)}")
                p0, _ = hand_pose(0.0)
                T.move(p0 + np.array([-0.01, 0.0, 0.0]) + self.corr, speed=0.05)
                T.move(p0 + self.corr, speed=0.02)
                T.hold(hold_k(0.2))
                self.when_settled(self.close, tol=0.002)

            def close(self):
                T = self.R
                err = grasp_point("right")[0] - (T.pos + ROOT_OFFSET)
                note(f"AT GRIP ({MODE}): grasp-point error (mm) {np.round(err * 1000, 1)}")
                if MODE != "push":
                    T.grip_to(grip_cmd, hold_k(0.6))
                self.at(self.sync(), lambda: self.turn(1))

            def turn(self, k):
                T = self.R
                n = int(math.ceil(abs(turn_to) / ARC_STEP_RAD))
                if CONTROL and handle_q() <= -CONTACT_RAD:
                    note(f"CONTROL: letting go at the TRIP contact, handle {math.degrees(handle_q()):.1f} deg")
                    self.release()
                    return
                if k > n:
                    T.hold(hold_k(0.1))
                    self.at(self.sync(), lambda: self.hold_trip(0))
                    return
                p, r = hand_pose(turn_to * k / n)
                T.move(p + self.corr, r, min_steps=2)
                self.at(self.sync(), lambda: self.turn(k + 1))

            def hold_trip(self, k):
                T = self.R
                q = handle_q()
                if CONTROL and q <= -CONTACT_RAD:
                    note(f"CONTROL: letting go at the TRIP contact, handle {math.degrees(q):.1f} deg")
                    self.release()
                    return
                if tripped():
                    note(f"GREEN lamp at hold step {k}, handle {math.degrees(q):.1f} deg, lead "
                         f"{math.degrees(self.lead):.1f} deg; holding {HOLD_AFTER_GREEN_S} s more")
                    T.hold(hold_k(HOLD_AFTER_GREEN_S))
                    self.at(self.sync(), self.release)
                    return
                if k > hold_k(8.0):
                    note(f"NO green lamp after 8 s; handle {math.degrees(q):.1f} deg")
                    self.release()
                    return
                if q > -CONTACT_RAD - math.radians(1.0) and self.lead > max_lead:
                    self.lead = max(max_lead, self.lead - math.radians(1.0))
                    p, r = hand_pose(self.lead)
                    T.move(p + self.corr, r, min_steps=1)
                else:
                    T.hold(1)
                if k % 10 == 0:
                    note(f"HOLD {k}: handle {math.degrees(q):.1f} deg, lead {math.degrees(self.lead):.1f} deg")
                self.at(self.sync(), lambda: self.hold_trip(k + 1))

            def release(self):
                """Let go: open the fingers and back straight off; the spring returns the handle to NORMAL."""
                T = self.R
                if MODE != "push":
                    T.grip_to(0.6, hold_k(0.25))
                T.move(T.pos + ROOT_OFFSET + np.array([-0.07, 0.0, 0.0]), speed=0.08)
                T.move(T.pos + ROOT_OFFSET + np.array([-0.02, 0.0, 0.04]), speed=0.08)
                T.move(R_PARK, FORWARD, speed=0.08)
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
        planner.at(settle, planner.start)
        note(f"demo mode {MODE}, control {CONTROL}, named slot {slot}, grip command {grip_cmd:.3f}")

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
        self._debug = lambda k: (f"DBG k={k} handle={math.degrees(handle_q()):.1f} "
                                 f"closed={st.closed[0].tolist()} lvl={task.episode_report(view, 0)['milestone']}")
        self._log = log
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
        if k % 20 == 0:
            self._log(self._debug(k), flush=True)
        (lp, lq, lg), (rp, rq, rg) = L.steps[k], R.steps[k]
        self.state_k[0] = k + 1
        return np.concatenate([self._joint_action("left", lp, lq, lg), self._joint_action("right", rp, rq, rg)])
