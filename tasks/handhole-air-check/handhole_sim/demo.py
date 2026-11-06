"""The scripted demo, closed loop.

Arm A (left) picks the sample hose's inline filter off its pad (fingers down, closing across it), lifts it so the
hose's last 100 mm hangs below on its gimbal, centres the hose's open end over the free half of the pull slot (it
measures the end and corrects, twice: above the slot and just above its mouth), and lowers it so the end hangs in the
box's air under the cover. It holds there for the sampling time plus 1.5 s while the meter's reading changes from
room air to the box's air, reads the meter (privileged: the task's display state, the same thing the alarm bars
show), lifts the hose out and lays it on the table.
Clean (seed 0): arm A takes out the far bolt and arm B the near bolt, each by pinching the hex head and turning it
counter-clockwise in regrasps of 180 degrees until the thread lets go, then pulling the bolt straight up out of the
cover. Both arms back away; the task's (off-camera) lift check then lifts the cover. Alarm (seed 1): both arms back
away and wait; the lift check must find the cover still bolted.

HANDHOLE_CONTROL=1 is the control run: the hose's end is held 35 mm beside the slot, 20 mm above the cover (never
inside) for 20 s; the meter keeps reading room air (20.9 % O2, zeros: it looks clean), and the bolts are taken out
anyway. It must score 0. Since no valid sample exists the task never starts its lift check by itself, so the demo
asks for it once the arms have backed away.

Grasp-point targets become yam_arena's 14-D joint action through yam_arena's own IK every control step. The demo
reads the simulator's state like any scripted demo with privileged state; ``--policy handhole-demo``
(handhole_sim.policy) plays it through Inspect Robots. Import after Isaac Sim has started.
"""

from __future__ import annotations

import math
import os

import numpy as np

from yam_arena import ik, station

CONTROL = os.environ.get("HANDHOLE_CONTROL") == "1"

PAD_ABOVE_GRASP = 0.006
"""The finger pads' centre is 6 mm below the grasp point: pads on the filter's axis keep the finger ends above the
pad it lies on."""
L_WAIT = (0.22, 0.26, 0.22)
R_WAIT = (0.22, -0.26, 0.22)
R_RETREAT = (0.20, -0.14, 0.24)
TIP_DEPTH = 0.060
"""How far below the cover's top face the hose end is aimed (it stops on the box's air side, under the cover)."""
YAW0, YAW1 = -90.0, 90.0
"""Each regrasp turns the head 180 deg (more is past the wrist joint's range with the fingers down)."""
SERVO_DT = 0.1
SERVO_GAIN = 0.15
SERVO_GAIN_SLOT = 0.05
SERVO_MAX = 0.003
DESCEND_SPEED = 0.006
LAY_DOWN = (0.25, 0.40, 0.05)
GRASP_ALONG = 0.013
"""Grasp the filter 13 mm from its hose end (7 mm short of its middle), so the fingers stay clear of the tail's hinge."""


class Demo:
    """``next_action()`` gives the 14-D joint action for the next control step."""

    def __init__(self, STATION, control_hz: float = 20.0, speed: float = 0.10, log=print):
        from handhole_sim import station as hb
        from handhole_sim.kit import CLOSED, DOWN, OPEN, ROOT_OFFSET, ArmTimeline, q_axis, qmul, rotate
        from handhole_sim.kit import grasp_point as grasp_point_world
        from handhole_sim.spec import SAMPLE_S
        from handhole_sim.usd import handhole_props as props

        view = STATION.env
        task = STATION.task
        st = task.state(view)
        robot = view.scene["robot"]
        mon = view.scene["monitor"]
        origin = view.scene.env_origins[0].cpu().numpy()
        dt = 1.0 / control_hz
        hk = lambda s: max(1, round(s / dt))  # noqa: E731
        state_k = [0]
        self.state_k = state_k
        tail_id = mon.find_bodies(props.TIP_BODY)[0][0]
        filter_id = mon.find_bodies("filter")[0][0]
        head_grasp_z = props.Z_TOP + props.WASHER[1] + props.HEAD[1] - 0.010 + 0.012

        def q_yaw(deg):
            return qmul(q_axis((0, 0, 1), math.radians(deg)), DOWN)

        QA = q_yaw(0)

        def note(msg):
            log(msg, flush=True)

        def grasp_point(side):
            p, q = grasp_point_world(robot, side)
            return p - origin, q

        def tip():
            pos = mon.data.body_pos_w.torch[0, tail_id].cpu().numpy() - origin
            quat = mon.data.body_quat_w.torch[0, tail_id].cpu().numpy()
            return pos + rotate(quat, (props.TAIL_L, 0.0, 0.0))

        def report():
            return task.episode_report(view, 0)

        class Planner:
            def __init__(self, L, R):
                self.L, self.R = L, R
                self.events, self.k = {}, -1
                self.cycles = 0

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

            # Arm A: the hose ---------------------------------------------------------------------
            def plan_pick(self):
                fx, fy = props.FILTER_XY
                g = np.array([fx + props.FILTER_DIR * GRASP_ALONG, fy, props.Z_HOSE + PAD_ABOVE_GRASP])
                self.pick_g = g
                L = self.L
                L.grip_to(OPEN, hk(0.2))
                L.move(g + np.array([0.0, 0.0, 0.08]), QA, min_steps=hk(3.0))
                L.hold(hk(0.5))
                self.sync()
                self.when_settled("left", self.descend)

            def descend(self):
                g = self.pick_g
                L = self.L
                L.move(g + np.array([0.0, 0.0, 0.01]), speed=0.04)
                L.move(g, speed=0.02)
                L.hold(hk(0.5))
                self.sync()
                self.when_settled("left", self.close)

            def close(self):
                e = grasp_point("left")[0] - self.pick_g
                note(f"PICK: grasp error before closing (mm) {np.round(e * 1000, 1)}")
                L = self.L
                L.grip_to(CLOSED, hk(1.0))
                L.move(self.pick_g + np.array([0.0, 0.0, 0.03]), speed=0.02)
                L.move(self.pick_g + np.array([-0.04, -0.08, 0.15]), speed=0.04)
                L.hold(hk(1.5))
                self.sync()
                self.when_still(self.carry)

            def when_still(self, then):
                """Once arm A has settled, wait (up to 6 s) until the hanging end moves less than 1 mm in 0.4 s."""
                self.when_settled("left", lambda: self._tail_still(None, 0, then))

            def _tail_still(self, prev, n, then):
                now = tip()
                if prev is not None and (np.linalg.norm(now - prev) < 0.001 or n >= 15):
                    then()
                    return
                self.L.hold(hk(0.4))
                self.at(self.sync(), lambda: self._tail_still(now, n + 1, then))

            def carry(self):
                g = grasp_point("left")[0]
                self.hang = g - tip()
                note(f"HOSE picked: grasp {np.round(g, 4)} tip {np.round(tip(), 4)} hang {np.round(self.hang, 4)}")
                for nm in ("filter", "tail_gimbal", props.TIP_BODY):
                    i = mon.find_bodies(nm)[0][0]
                    qq = mon.data.body_quat_w.torch[0, i].cpu().numpy()
                    pp = mon.data.body_pos_w.torch[0, i].cpu().numpy() - origin
                    note(f"  {nm}: pos {np.round(pp, 4)} x {np.round(rotate(qq, (1, 0, 0)), 2)} y {np.round(rotate(qq, (0, 1, 0)), 2)}")
                note(f"  grasp y-axis {np.round(rotate(grasp_point('left')[1], (0, 1, 0)), 2)}; tail joints "
                     f"{np.round(np.degrees(mon.data.joint_pos.torch[0, -2:].cpu().numpy()), 1)} ({mon.joint_names[-2:]})")
                self.tip_goal = np.array([props.SLOT_XY[0], props.SLOT_OPENING_Y, props.Z_TOP + 0.012])
                if CONTROL:
                    self.tip_goal = np.array([props.SLOT_XY[0] - 0.035, props.SLOT_OPENING_Y, props.Z_TOP + 0.02])
                L = self.L
                L.move(self.tip_goal + self.hang + np.array([0.0, 0.0, 0.010]), speed=0.06)
                L.move(self.tip_goal + self.hang, speed=0.03)
                L.hold(hk(1.0))
                self.sync()
                self.tries = 0
                self.when_still(lambda: self.servo_start("center"))

            # Closed-loop servo on the hanging end: every SERVO_DT, nudge the grasp target by a fraction of the end's
            # measured error (the held filter turns a little in the fingers as the hose drags, so a one-off
            # correction does not hold).
            def servo_start(self, mode):
                self.mode, self.good, self.ticks, self.low = mode, 0, 0, []
                self.target = self.L.pos + ROOT_OFFSET
                note(f"SERVO {mode}: tip error (mm) {np.round((tip() - self.tip_goal) * 1000, 1)}")
                self.at(self.sync(), self.servo_tick)

            def servo_tick(self):
                e = tip() - self.tip_goal
                self.ticks += 1
                inside_slot = tip()[2] < props.Z_TOP - 0.004
                gain = SERVO_GAIN_SLOT if inside_slot else SERVO_GAIN
                step = np.clip(gain * e[:2], -SERVO_MAX, SERVO_MAX)
                self.target[:2] -= step
                done = False
                if self.mode == "center":
                    self.good = self.good + 1 if np.abs(e[:2]).max() < 0.001 else 0
                    if self.good >= 6 or self.ticks > 300:
                        note(f"CENTRED after {self.ticks} ticks: tip error (mm) {np.round(e * 1000, 1)}")
                        if CONTROL:
                            note("CONTROL: holding the hose's end beside the slot for 20 s")
                            self.L.hold(hk(20.0))
                            self.at(self.sync(), self.after_sample)
                            return
                        self.mode, self.ticks = "descend", 0
                else:
                    z_min = props.Z_TOP - TIP_DEPTH + self.hang[2]
                    self.target[2] = max(z_min, self.target[2] - DESCEND_SPEED * SERVO_DT)
                    self.low.append(tip()[2])
                    if tip()[2] < props.Z_COVER_BOTTOM - 0.006:
                        done = True
                    elif len(self.low) > 30 and self.low[-30] - self.low[-1] < 0.002:
                        # The end has stopped going down: it caught on the slot's edge. Lift and centre again.
                        self.tries += 1
                        note(f"RETRY insertion {self.tries}: tip {np.round(tip(), 4)}")
                        if self.tries > 4:
                            done = True
                        else:
                            self.L.move(self.target + np.array([0.0, 0.0, 0.02]), speed=0.01)
                            self.L.hold(hk(0.5))
                            self.sync()
                            self.when_still(lambda: self.servo_start("center"))
                            return
                self.L.move(self.target, min_steps=hk(SERVO_DT))
                if done:
                    self.L.hold(hk(1.0))
                    self.at(self.sync(), self.sampling)
                    return
                self.at(self.sync(), self.servo_tick)

            def sampling(self):
                r = report()
                note(f"IN: tip {np.round(tip(), 4)} milestone {r['milestone']} inside {r['hose_end_inside_s']} s")
                self.L.hold(hk(SAMPLE_S + 1.5))
                self.at(self.sync(), self.after_sample)

            def after_sample(self):
                r = report()
                self.reading = r["display"]
                note(f"READ meter: {self.reading} (milestone {r['milestone']}, inside {r['hose_end_inside_s']} s, "
                     f"max {r['hose_end_inside_max_s']} s)")
                L = self.L
                up = L.pos + ROOT_OFFSET
                up[2] = min(up[2] + 0.12, 0.20)  # the end clears the slot; higher is outside the fingers-down reach
                L.move(up, speed=0.03)
                L.move(np.array(LAY_DOWN) + np.array([0.0, 0.0, 0.08]), speed=0.08)
                L.move(LAY_DOWN, speed=0.05)
                L.grip_to(OPEN, hk(0.8))
                L.move(np.array(LAY_DOWN) + np.array([0.0, 0.0, 0.10]), speed=0.06)
                self.sync()
                if self.reading in ("go", "room"):
                    self.at(self.sync(), self.bolts_start)
                else:
                    self.L.move(L_WAIT, speed=0.08)
                    self.sync()
                    note("ALARM: leaving the cover bolted")

            # Both arms: the bolts --------------------------------------------------------------------
            def bolts_start(self):
                self.arms = [(self.L, 0), (self.R, 1)]
                for tl, k in self.arms:
                    x, y = props.BOLTS[k]
                    tl.grip_to(OPEN, hk(0.2))
                    tl.move((x, y, head_grasp_z + 0.06), q_yaw(YAW0), min_steps=hk(3.0))
                self.at(self.sync(), self.bolt_cycle)

            def bolt_cycle(self):
                engaged = [bool(st.engaged[0, k]) for k in range(2)]
                turns = [round(float(st.turn[0, k]) / (2 * math.pi), 2) for k in range(2)]
                note(f"BOLT cycle {self.cycles}: turns {turns} engaged {engaged}")
                if not any(engaged) or self.cycles >= 10:
                    self.L.hold(hk(1.5))  # let the loose bolts drop back onto their seats before measuring them
                    self.at(self.sync(), self.pull_bolts)
                    return
                self.cycles += 1
                for tl, k in self.arms:
                    if not engaged[k]:
                        continue
                    x, y = props.BOLTS[k]
                    z = head_grasp_z + min(float(st.turn[0, k]), 2 * math.pi * props.ENGAGE_TURNS) / (2 * math.pi) * props.PITCH
                    tl.move((x, y, z + 0.03), q_yaw(YAW0), speed=0.05, min_steps=hk(0.6))
                    tl.move((x, y, z), speed=0.02)
                    tl.grip_to(CLOSED, hk(0.8))
                    n = hk(5.0)
                    for i in range(1, n + 1):
                        tl.move((x, y, z + props.PITCH * 0.5 * i / n), q_yaw(YAW0 + (YAW1 - YAW0) * i / n), min_steps=1)
                    tl.grip_to(0.45, hk(0.6))
                    tl.move((x, y, z + 0.03), speed=0.04)
                self.at(self.sync(), self.bolt_cycle)

            def pull_bolts(self, attempt=0):
                todo = [(tl, k) for tl, k in self.arms if not bool(st.out[0, k])]
                if todo and attempt < 4:
                    for tl, k in todo:
                        x, y = props.BOLTS[k]
                        z = head_grasp_z + float(st.box.data.joint_pos.torch[0, st.slider_j[k]])
                        tl.grip_to(OPEN, hk(0.4))
                        tl.move((x, y, z + 0.03), q_yaw(YAW0), speed=0.05)
                        tl.move((x, y, z), speed=0.02)
                        tl.grip_to(CLOSED, hk(1.0))
                        tl.move((x, y, z + props.SHANK_LEN + 0.015), speed=0.01)
                        tl.hold(hk(0.5))
                        tl.grip_to(OPEN, hk(0.6))
                        tl.move((x, y, z + props.SHANK_LEN + 0.10), speed=0.04)
                    self.at(self.sync(), lambda: self.pull_bolts(attempt + 1))
                    return
                self.L.move(L_WAIT, speed=0.08)
                self.R.move(R_RETREAT, speed=0.08)

                def done():
                    note(f"BOLTS out {[bool(v) for v in st.out[0]]} parked {[bool(v) for v in st.parked[0]]} "
                         f"(pull attempts {attempt})")
                    if CONTROL:  # no valid sample, so the lift check never arms itself: ask for it
                        task.force_lift_test(view)

                self.at(self.sync(), done)

        l_pos, l_quat = grasp_point("left")
        r_pos, r_quat = grasp_point("right")
        L = ArmTimeline(l_pos - ROOT_OFFSET, l_quat, dt, speed)
        R = ArmTimeline(r_pos - ROOT_OFFSET, r_quat, dt, speed)
        planner = Planner(L, R)
        R.move(R_WAIT, DOWN, min_steps=hk(3.0))
        L.move(L_WAIT, QA, min_steps=hk(3.0))
        L.hold(hk(0.5))
        planner.at(planner.sync(), planner.plan_pick)
        note(f"handhole demo: control {CONTROL}, box air {'clean' if bool(st.go[0]) else 'alarm'}")

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
            r = report()
            lag = np.round((grasp_point("left")[0] - (L.steps[k][0] + ROOT_OFFSET)) * 1000, 0)
            lagb = np.round((grasp_point("right")[0] - (R.steps[k][0] + ROOT_OFFSET)) * 1000, 0)
            fz = float(mon.data.body_pos_w.torch[0, filter_id, 2]) - origin[2]
            return (f"DBG k={k} t={k * dt:.1f} lagA={lag} lagB={lagb} ms={r['milestone']} tip={np.round(tip(), 4)} "
                    f"filter_z={fz:.3f} disp={r['display']} in={r['hose_end_inside_s']} turns={r['bolt_turns']} "
                    f"eng={r['bolts_engaged']} out={r['bolts_out']} phase={r['lift_test_phase']} "
                    f"ev={r['semantic_events']} {r['contact_events']}")

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
