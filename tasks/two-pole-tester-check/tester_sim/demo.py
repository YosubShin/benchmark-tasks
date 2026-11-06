"""The scripted demo, closed loop.

Arm A (left) takes the T150 main unit and arm B (right) the L1 probe from the tray: fingers forward, each hand rolled
90 degrees and coming in from behind, so the fingers close top and bottom on each tool's grip (closing across the
main unit's width let it pivot nose-down about the closing axis under yam_arena's gains). Both tools are then held level and yawed
toward each other, so the two tips meet each contact pair (PRV240: 24 mm apart; L1-N: 18.6 mm). For each pair, each
arm in turn brings its tip 15 mm in front of its recess, measures the tip and corrects, then again 1.5 mm in front,
and pushes straight in until the tip bottoms. With both seated it reads the tester (lit or not) and both withdraw.

Order: check on the PRV240; if the tester lit, test L1-N, then check again. If it did not light, withdraw 12 cm and
hold still (the correct stop). TESTER_CONTROL=1 is the control run: when nothing lights, the demo carries on to
L1-N anyway (it must score 0).

Grasp-point targets become yam_arena's 14-D joint action through yam_arena's own IK every control step. The demo
reads the simulator's state (tool poses, whether the tester is lit) like any scripted demo with privileged state;
``--policy tester-demo`` (tester_sim.policy) plays it through Inspect Robots. Import after Isaac Sim has started.
"""

from __future__ import annotations

import math
import os

import numpy as np

from yam_arena import ik, station

CONTROL = os.environ.get("TESTER_CONTROL") == "1"
GRIP_B = float(os.environ.get("TESTER_GRIP_B", "0.0"))
"""Arm B's closed gripper command (0 = fully closed). Above 0 the fingers stop short of the handle's 22 mm and squeeze
it only with the finger drive's stiffness times the remaining gap."""

ROLL_A = os.environ.get("TESTER_ROLL_A", "1") == "1"
"""Arm A takes the main unit with its hand rolled 90 degrees too (0: fingers closing across its width, from above)."""
YAW_A_DEG, YAW_B_DEG = 22.0, 16.0
STANDOFF = 0.015
PUSH = 0.0055
BACKOFF = 0.05
PRE_MOUTH = 0.0015


class Demo:
    """``next_action()`` gives the 14-D joint action for the next control step."""

    def __init__(self, STATION, control_hz: float = 20.0, speed: float = 0.10, log=print):
        from tester_sim import station as hb
        from tester_sim.kit import CLOSED, FORWARD, OPEN, ROOT_OFFSET, ArmTimeline, GraspModel, q_axis, qmul, rotate
        from tester_sim.kit import grasp_point as grasp_point_world
        from tester_sim.spec import TARGET
        from tester_sim.task import CONTACT_NAMES, LEVEL_QUAT, tool_start_pose
        from tester_sim.usd import tester_props as props

        view = STATION.env
        task = STATION.task
        st = task.state(view)
        robot = view.scene["robot"]
        tools = {"A": view.scene["tester_main"], "B": view.scene["probe_l1"]}
        origin = view.scene.env_origins[0].cpu().numpy()
        dt = 1.0 / control_hz
        hk = lambda s: max(1, round(s / dt))  # noqa: E731
        state_k = [0]
        self.state_k = state_k

        LEVEL = np.array(LEVEL_QUAT)
        TOOL_QUAT = {"A": qmul(q_axis((0, 0, 1), -math.radians(YAW_A_DEG)), LEVEL),
                     "B": qmul(q_axis((0, 0, 1), math.radians(YAW_B_DEG)), LEVEL)}
        GRIP_QUAT = {"A": qmul(FORWARD, q_axis((0, 0, 1), math.pi / 2)) if ROLL_A else FORWARD,
                     "B": qmul(FORWARD, q_axis((0, 0, 1), -math.pi / 2))}
        """Both hands fingers forward, rolled 90 degrees (wrist camera brackets outward), so the fingers close top and
        bottom on each tool's grip: a level tool held by side pads pivots nose-down about the closing axis."""
        CLOSE = {"A": CLOSED, "B": GRIP_B}
        dims = {"A": props.MAIN, "B": props.L1}
        side_of = {"A": "left", "B": "right"}

        def note(msg):
            log(msg, flush=True)

        def grasp_point(side):
            p, q = grasp_point_world(robot, side)
            return p - origin, q

        def tip(arm):
            return tools[arm].data.root_pos_w.torch[0].cpu().numpy() - origin

        def lit():
            return bool(st.now["lit"][0]) if st.now else False

        def report():
            return task.episode_report(view, 0)

        class Planner:
            def __init__(self, L, R):
                self.L, self.R = L, R
                self.arm = {"A": L, "B": R}
                self.events, self.readings, self.k = {}, [], -1
                self.model = {}
                self.retries = {}

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

            def when_settled(self, arm, fn, tries=15):
                tl, side = self.arm[arm], side_of[arm]

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

            def tool_to(self, arm, tip_pos, speed=None, min_steps=3):
                g, q = self.model[arm].grasp_for(tip_pos, TOOL_QUAT[arm])
                self.arm[arm].move(g, q, speed=speed, min_steps=min_steps)

            # Pick ------------------------------------------------------------------------------
            def plan_pick(self):
                self.pick_grasp = {}
                for arm, name in (("A", "main"), ("B", "l1")):
                    d, tl, q = dims[arm], self.arm[arm], GRIP_QUAT[arm]
                    start = tool_start_pose(name)
                    grasp, _ = GraspModel.nominal(q, LEVEL, (0.0, d.body_y, d.grasp_z)).grasp_for(start[:3], LEVEL)
                    self.pick_grasp[arm] = grasp
                    tl.grip_to(OPEN, hk(0.2))
                    tl.move(tl.pos + ROOT_OFFSET + np.array([0.0, 0.0, 0.05]), min_steps=hk(1.0))
                    from_above = arm == "A" and not ROLL_A
                    pre = grasp + (np.array([0.0, 0.0, 0.065]) if from_above else np.array([-0.06, 0.0, 0.0]))
                    tl.move(pre, q, min_steps=hk(4.0))
                    tl.hold(hk(1.0))
                self.sync()
                self.when_settled("A", lambda: self.when_settled("B", self.descend))

            def descend(self):
                for arm in "AB":
                    self.arm[arm].move(self.pick_grasp[arm], speed=0.02)
                    self.arm[arm].hold(hk(0.6))
                self.sync()
                self.when_settled("A", lambda: self.when_settled("B", self.close))

            def close(self):
                for arm in "AB":
                    e = grasp_point(side_of[arm])[0] - self.pick_grasp[arm]
                    note(f"PICK {arm}: grasp error before closing (mm) {np.round(e * 1000, 1)}")
                for arm in "AB":
                    tl = self.arm[arm]
                    tl.grip_to(CLOSE[arm], hk(1.2))
                    tl.move(self.pick_grasp[arm] + np.array([0.0, 0.0, 0.06]), speed=0.02)
                self.sync()
                self.when_settled("A", lambda: self.when_settled("B", self.after_pick))

            def after_pick(self):
                for arm in "AB":
                    self.model[arm] = GraspModel.measure(robot, side_of[arm], tools[arm])
                    note(f"GRASP {arm} rel_pos (mm) {np.round(self.model[arm].rel_pos * 1000, 1)}; tip "
                         f"{np.round(tip(arm), 4)}")
                self.do_pair("prv", self.after_prove1)

            # A contact pair ------------------------------------------------------------------------
            def pair_mouths(self, pair):
                if pair == "prv":
                    return np.array(props.prover_mouth(0)), np.array(props.prover_mouth(1))
                return np.array(props.well_mouth(TARGET[0])), np.array(props.well_mouth(TARGET[1]))

            def do_pair(self, pair, then):
                ma, mb = self.pair_mouths(pair)
                self.pair, self.then = pair, then
                for arm, m in (("A", ma), ("B", mb)):
                    self.tool_to(arm, m - np.array([BACKOFF, 0.0, 0.0]), speed=0.08, min_steps=hk(2.0))
                self.sync()
                self.insert("A", ma, lambda: self.insert("B", mb, self.read))

            def insert(self, arm, mouth, then):
                target = mouth - np.array([STANDOFF, 0.0, 0.0])
                self.tool_to(arm, target, speed=0.05)
                self.arm[arm].hold(hk(0.4))
                self.sync()
                self.when_settled(arm, lambda: self._correct(arm, mouth, STANDOFF, np.zeros(3), 0, then))

            def _correct(self, arm, mouth, gap, correction, attempt, then):
                target = mouth - np.array([gap, 0.0, 0.0])
                err = tip(arm) - target
                note(f"{self.pair} {arm} gap {gap * 1000:.1f} mm attempt {attempt}: tip error (mm) "
                     f"{np.round(err * 1000, 1)}")
                correction = correction + err
                tol = 0.0003 if gap <= PRE_MOUTH else 0.0015
                if attempt < (5 if gap <= PRE_MOUTH else 3) and max(abs(err[1]), abs(err[2])) > tol:
                    self.tool_to(arm, target - correction, speed=0.01)
                    self.arm[arm].hold(hk(0.4))
                    self.sync()
                    self.when_settled(arm, lambda: self._correct(arm, mouth, gap, correction, attempt + 1, then))
                    return
                if gap > PRE_MOUTH:
                    self.tool_to(arm, mouth - np.array([PRE_MOUTH, 0.0, 0.0]) - correction, speed=0.01)
                    self.arm[arm].hold(hk(0.4))
                    self.sync()
                    self.when_settled(arm, lambda: self._correct(arm, mouth, PRE_MOUTH, correction, 0, then))
                    return
                self.tool_to(arm, mouth + np.array([PUSH, 0.0, 0.0]) - correction, speed=0.008)
                self.arm[arm].hold(hk(0.6))
                self.sync()

                def seated():
                    name = self.contact_name(arm)
                    ok = bool(st.now["seated"][0, 0 if arm == "A" else 1, CONTACT_NAMES.index(name)]) if st.now else False
                    note(f"{self.pair} {arm} pushed: tip past mouth (mm) {np.round((tip(arm) - mouth) * 1000, 1)} "
                         f"seated in {name} {ok}")
                    key = (self.pair, arm)
                    if not ok and self.retries.get(key, 0) < 2:
                        self.retries[key] = self.retries.get(key, 0) + 1
                        note(f"  not seated: backing out and trying again ({self.retries[key]})")
                        self.tool_to(arm, tip(arm) - np.array([0.012, 0.0, 0.0]), speed=0.02)
                        self.arm[arm].hold(hk(0.3))
                        self.sync()
                        self.insert(arm, mouth, then)
                        return
                    self.retries.pop(key, None)
                    then()

                self.when_settled(arm, seated)

            def contact_name(self, arm):
                if self.pair == "prv":
                    return "prv_plus" if arm == "A" else "prv_minus"
                return TARGET[0] if arm == "A" else TARGET[1]

            def read(self):
                self.L.hold(hk(1.5))
                self.sync()

                def take():
                    on = lit()
                    note(f"READ {self.pair}: tester lit {on}, milestone {report()['milestone']}")
                    self.readings.append((self.pair, on))
                    self.withdraw(lambda: self.then(on))

                self.at(self.sync(), take)

            def withdraw(self, then):
                ma, mb = self.pair_mouths(self.pair)
                for arm in "AB":
                    self.tool_to(arm, tip(arm) - np.array([0.012, 0.0, 0.0]), speed=0.02)
                self.sync()
                for arm, m in (("A", ma), ("B", mb)):
                    self.tool_to(arm, m - np.array([BACKOFF, 0.0, 0.0]), speed=0.06)
                self.at(self.sync(), then)

            # Order ------------------------------------------------------------------------------------
            def after_prove1(self, on):
                if on or CONTROL:
                    if not on:
                        note("CONTROL: nothing lit on the proving unit; carrying on to the terminals anyway")
                    self.do_pair("target", self.after_test)
                else:
                    note("STOP: the tester did not light on the proving unit; not touching the terminals.")
                    ma, mb = self.pair_mouths("prv")
                    for arm, m in (("A", ma), ("B", mb)):
                        self.tool_to(arm, m - np.array([0.12, 0.0, -0.02]), speed=0.05)
                    self.sync()
                    self.L.hold(hk(8.0))
                    self.sync()
                    self.at(len(self.L.steps), lambda: note(f"DONE (stopped): {report()}"))

            def after_test(self, on):
                self.do_pair("prv", self.after_prove2)

            def after_prove2(self, on):
                note(f"DONE: readings {self.readings}; {report()}")
                self.L.hold(hk(4.0))
                self.sync()

        l_pos, l_quat = grasp_point("left")
        r_pos, r_quat = grasp_point("right")
        L = ArmTimeline(l_pos - ROOT_OFFSET, l_quat, dt, speed)
        R = ArmTimeline(r_pos - ROOT_OFFSET, r_quat, dt, speed)
        planner = Planner(L, R)
        settle = hk(1.5)
        L.hold(settle)
        R.hold(settle)
        planner.at(settle, planner.plan_pick)
        note(f"tester demo: control {CONTROL}, prover flat {bool(st.flat[0])}, arm B close command {GRIP_B}")

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
            lag = np.round((grasp_point("left")[0] - (L.steps[k][0] + ROOT_OFFSET)) * 1000, 0)
            lagb = np.round((grasp_point("right")[0] - (R.steps[k][0] + ROOT_OFFSET)) * 1000, 0)
            r = report()
            return (f"DBG k={k} t={k * dt:.1f} lagA={lag} lagB={lagb} ms={r['milestone']} tipA={np.round(tip('A'), 4)} "
                    f"tipB={np.round(tip('B'), 4)} lit={lit()} ev={r['semantic_events']}")

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
        if k % 40 == 0:
            self._log(self._debug(k), flush=True)
        (lp, lq, lg), (rp, rq, rg) = L.steps[k], R.steps[k]
        self.state_k[0] = k + 1
        return np.concatenate([self._joint_action("left", lp, lq, lg), self._joint_action("right", rp, rq, rg)])
