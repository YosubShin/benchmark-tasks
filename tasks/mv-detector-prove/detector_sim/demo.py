"""The scripted demo, closed loop.

Arm B (right) presses the PRX's ON button with its closed fingers (fingers down) while the detector lies in its
cradle, then hovers over the tester's TEST button. Arm A (left) takes the stick from above (fingers down, closing
across it at its balance point) and lifts it to working height; the demo then measures how the detector actually
sits in the hand. Prove: arm B presses and holds TEST; arm A brings the sensing end 15 mm in front of the end plate,
measures the sensing end's position, corrects sideways, and pushes along the stick until it touches (the plate stops
it). It reads the detector (lit or not) and withdraws. If it lit: the same approach on the bar (perpendicular to it),
then prove again. If not: withdraw the sensing end 13 cm from the plate and hold still (the correct stop).

DETECTOR_CONTROL=1 is the control run: on a failed prove the demo carries on to the bar anyway (it must score 0).

Grasp-point targets become yam_arena's 14-D joint action through yam_arena's own IK every control step. The demo
reads the simulator's state (the detector's pose, whether it is lit) like any scripted demo with privileged state;
``--policy detector-demo`` (detector_sim.policy) plays it through Inspect Robots. Import after Isaac Sim has started.
"""

from __future__ import annotations

import math
import os

import numpy as np

from yam_arena import ik, station

CONTROL = os.environ.get("DETECTOR_CONTROL") == "1"

STANDOFF = 0.015
PUSH = 0.002
BACKOFF = 0.05
FINGER_END_BELOW_GRASP = 0.012


class Demo:
    """``next_action()`` gives the 14-D joint action for the next control step."""

    def __init__(self, STATION, control_hz: float = 20.0, speed: float = 0.10, log=print):
        from detector_sim import station as hb
        from detector_sim.kit import CLOSED, DOWN, OPEN, ROOT_OFFSET, ArmTimeline, GraspModel, q_axis, qmul, rotate
        from detector_sim.kit import grasp_point as grasp_point_world
        from detector_sim.spec import STICK_YAW_DEG, TOOL_Z
        from detector_sim.task import TOOL_QUAT, tool_start_pose
        from detector_sim.usd import detector_props as props

        view = STATION.env
        task = STATION.task
        st = task.state(view)
        robot = view.scene["robot"]
        det = view.scene["detector"]
        origin = view.scene.env_origins[0].cpu().numpy()
        dt = 1.0 / control_hz
        hk = lambda s: max(1, round(s / dt))  # noqa: E731
        state_k = [0]
        self.state_k = state_k

        D = np.array(props.D3())
        Q_TOOL = np.array(TOOL_QUAT)
        GRIP_A = qmul(q_axis((0, 0, 1), math.radians(STICK_YAW_DEG)), DOWN)
        GRIP_B = DOWN

        def note(msg):
            log(msg, flush=True)

        def grasp_point(side):
            p, q = grasp_point_world(robot, side)
            return p - origin, q

        def tip():
            return det.data.root_pos_w.torch[0].cpu().numpy() - origin

        def now(key):
            return bool(st.now[key][0])

        def report():
            return task.episode_report(view, 0)

        class Planner:
            def __init__(self, L, R):
                self.L, self.R = L, R
                self.events, self.readings, self.k = {}, [], -1
                self.model = None

            def at(self, step, fn):
                step = max(step, self.k + 1)
                while step in self.events:
                    step += 1
                self.events[step] = fn

            def run_due(self, k):
                self.k = k
                if k in self.events:
                    self.events.pop(k)()

            def sync(self):
                n = max(len(self.L.steps), len(self.R.steps))
                self.L.pad_to(n)
                self.R.pad_to(n)
                return n

            def when_settled(self, side, fn, tries=15):
                tl = self.L if side == "left" else self.R

                def check():
                    lag = np.linalg.norm(grasp_point(side)[0] - (tl.pos + ROOT_OFFSET))
                    still = check.prev is not None and abs(lag - check.prev) < 0.0003
                    check.prev = lag
                    if lag < 0.003 or still or check.tries <= 0:
                        fn()
                    else:
                        check.tries -= 1
                        tl.hold(hk(0.2))
                        self.at(self.sync(), check)

                check.tries, check.prev = tries, None
                self.at(self.sync(), check)

            def tool_to(self, tip_pos, speed=None, min_steps=3):
                g, q = self.model.grasp_for(tip_pos, Q_TOOL)
                self.L.move(g, q, speed=speed, min_steps=min_steps)

            # Self-test --------------------------------------------------------------------------
            def press_on(self):
                start = np.array(tool_start_pose())
                top = start[:3] + rotate(start[3:], (props.BODY[0] / 2 + 0.002, props.ON_BUTTON[1], props.ON_BUTTON[0]))
                g = top + np.array([0.0, 0.0, FINGER_END_BELOW_GRASP - 0.0015])
                self.R.grip_to(CLOSED, hk(0.4))
                self.R.move(g + np.array([0.0, 0.0, 0.05]), GRIP_B, min_steps=hk(3.5))
                self.R.hold(hk(0.5))
                self.R.move(g, speed=0.02)
                self.R.hold(hk(1.2))
                self.R.move(g + np.array([0.0, 0.0, 0.05]), speed=0.04)
                top = np.array(props.test_button_top())
                self.test_g = top + np.array([0.0, 0.0, FINGER_END_BELOW_GRASP - 0.0015])
                self.R.move(self.test_g + np.array([0.0, 0.0, 0.04]), GRIP_B, min_steps=hk(2.0))
                self.plan_pick()

            # Pick ------------------------------------------------------------------------------
            def plan_pick(self):
                start = tool_start_pose()
                grasp, _ = GraspModel.nominal(GRIP_A, Q_TOOL, (0.0, 0.0, props.GRASP_Z)).grasp_for(start[:3], Q_TOOL)
                self.pick_grasp = grasp
                self.L.grip_to(OPEN, hk(0.2))
                self.L.move(self.L.pos + ROOT_OFFSET + np.array([0.0, 0.0, 0.05]), min_steps=hk(1.0))
                self.L.move(grasp + np.array([0.0, 0.0, 0.07]), GRIP_A, min_steps=hk(4.0))
                self.L.hold(hk(1.0))
                self.sync()
                self.when_settled("left", self.descend)

            def descend(self):
                self.L.move(self.pick_grasp, speed=0.02)
                self.L.hold(hk(0.6))
                self.sync()
                self.when_settled("left", self.close)

            def close(self):
                e = grasp_point("left")[0] - self.pick_grasp
                note(f"PICK: grasp error before closing (mm) {np.round(e * 1000, 1)}")
                self.L.grip_to(CLOSED, hk(1.4))
                self.L.move(self.pick_grasp + np.array([0.0, 0.0, TOOL_Z - props.CRADLE_Z + 0.01]), speed=0.02)
                self.L.hold(hk(1.0))
                self.sync()
                self.when_settled("left", self.after_pick)

            def after_pick(self):
                self.model = GraspModel.measure(robot, "left", det)
                note(f"GRASP rel_pos (mm) {np.round(self.model.rel_pos * 1000, 1)}; tip {np.round(tip(), 4)}; "
                     f"selftest_done {now('selftest_done')}")
                self.prove(self.after_prove1)

            # Touch a contact ----------------------------------------------------------------------
            def touch(self, point, then):
                """Sensing end to STANDOFF in front of `point` (along -D), measure and correct sideways, push in."""
                self.tool_to(point - BACKOFF * D, speed=0.06, min_steps=hk(1.5))
                self.sync()
                self.tool_to(point - STANDOFF * D, speed=0.03)
                self.L.hold(hk(0.4))
                self.sync()
                self.when_settled("left", lambda: self._correct(point, np.zeros(3), 0, then))

            def _correct(self, point, corr, attempt, then):
                target = point - STANDOFF * D
                err = tip() - target
                side = err - np.dot(err, D) * D
                note(f"  standoff attempt {attempt}: tip error (mm) {np.round(err * 1000, 1)}")
                corr = corr + side
                if attempt < 3 and np.abs(side).max() > 0.0008:
                    self.tool_to(target - corr, speed=0.01)
                    self.L.hold(hk(0.4))
                    self.sync()
                    self.when_settled("left", lambda: self._correct(point, corr, attempt + 1, then))
                    return
                self.tool_to(point + PUSH * D - corr, speed=0.008)
                self.L.hold(hk(0.6))
                self.sync()
                self.when_settled("left", then)

            def withdraw(self, point, then, back=BACKOFF):
                self.tool_to(tip() - 0.015 * D, speed=0.02)
                self.tool_to(point - back * D, speed=0.05)
                self.at(self.sync(), then)

            # Prove / test --------------------------------------------------------------------------
            def prove(self, then):
                plate = np.array(props.plate_point())
                self.R.move(self.test_g, speed=0.02)
                self.R.hold(hk(0.4))

                def read():
                    self.L.hold(hk(1.5))
                    self.sync()

                    def take():
                        lit = now("lit")
                        note(f"READ prove: on_plate {now('on_plate')} test_pressed {now('test_pressed')} "
                             f"along {float(st.now['along'][0]) * 1000:.1f} mm -> lit {lit}; "
                             f"milestone {report()['milestone']}")
                        self.readings.append(("prove", lit))
                        self.R.move(self.test_g + np.array([0.0, 0.0, 0.04]), speed=0.03)
                        self.withdraw(plate, lambda: then(lit))

                    self.at(self.sync(), take)

                self.touch(plate, read)

            def test_target(self, then):
                c = np.array(props.contact_point())

                def read():
                    self.L.hold(hk(1.5))
                    self.sync()

                    def take():
                        lit = now("lit")
                        note(f"READ bar: on_cond {now('on_cond')} along {float(st.now['c_along'][0]) * 1000:.1f} mm "
                             f"-> lit {lit}; milestone {report()['milestone']}")
                        self.readings.append(("bar", lit))
                        self.withdraw(c, lambda: then(lit))

                    self.at(self.sync(), take)

                self.touch(c, read)

            def after_prove1(self, lit):
                if lit or CONTROL:
                    if not lit:
                        note("CONTROL: the prove failed; carrying on to the bar anyway")
                    self.test_target(self.after_test)
                else:
                    note("STOP: the detector did not prove; not going to the bar.")
                    self.tool_to(np.array(props.plate_point()) - 0.13 * D, speed=0.04)
                    self.sync()
                    self.L.hold(hk(8.0))
                    self.sync()
                    self.at(len(self.L.steps), lambda: note(f"DONE (stopped): {report()}"))

            def after_test(self, lit):
                self.prove(self.after_prove2)

            def after_prove2(self, lit):
                note(f"DONE: readings {self.readings}; {report()}")
                self.L.hold(hk(3.0))
                self.sync()

        l_pos, l_quat = grasp_point("left")
        r_pos, r_quat = grasp_point("right")
        L = ArmTimeline(l_pos - ROOT_OFFSET, l_quat, dt, speed)
        R = ArmTimeline(r_pos - ROOT_OFFSET, r_quat, dt, speed)
        planner = Planner(L, R)
        settle = hk(1.0)
        L.hold(settle)
        R.hold(settle)
        planner.at(settle, planner.press_on)
        note(f"detector demo: control {CONTROL}, prove fails {bool(st.fails[0])}")

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

        def debug(k):
            nw = st.now
            if not nw:
                return f"DBG k={k}"
            lag = np.round((grasp_point("left")[0] - (L.steps[k][0] + ROOT_OFFSET)) * 1000, 0)
            lagb = np.round((grasp_point("right")[0] - (R.steps[k][0] + ROOT_OFFSET)) * 1000, 0)
            r = report()
            return (f"DBG k={k} t={k * dt:.1f} lagA={lag} lagB={lagb} ms={r['milestone']} tip={np.round(tip(), 4)} "
                    f"on={bool(nw['on_pressed'][0])} st={bool(nw['selftest_done'][0])} "
                    f"test={bool(nw['test_pressed'][0])} plate={bool(nw['on_plate'][0])} "
                    f"cond={bool(nw['on_cond'][0])} lit={bool(nw['lit'][0])} ev={r['semantic_events']}")

        self._L, self._R, self._planner, self._joint_action = L, R, planner, joint_action
        self._debug = debug
        self._log = log
        self.worst = worst

    def next_action(self) -> np.ndarray:
        k = self.state_k[0]
        L, R = self._L, self._R
        self._planner.run_due(k)
        self._planner.sync()
        if k >= len(L.steps):
            L.hold(1)
            R.hold(1)
        if k % 20 == 0:
            self._log(self._debug(k), flush=True)
        (lp, lq, lg), (rp, rq, rg) = L.steps[k], R.steps[k]
        self.state_k[0] = k + 1
        return np.concatenate([self._joint_action("left", lp, lq, lg), self._joint_action("right", rp, rq, rg)])
