"""The scripted demo, closed loop.

What it reads. Every control step it gets the observation Inspect Robots hands any policy: the three RGB images
(``top_cam``, ``left_cam``, ``right_cam``) and the 14-D ``joint_pos`` state. Grasp points come from forward kinematics
of the measured joints (``handhole_sim.vision``, the arm's URDF at the station's calibrated bases); the monitor's
alarm is read from the top-camera image; bolt holding is felt through the measured gripper opening. Fixed knowledge:
the documented layout (task.json ``setup``) and the station's published camera calibration.
ONE simulator read remains: while it puts the hose's open end into the pull slot, the demo reads the end's position
(the hose's last link) from the simulator to servo it into the 13 mm slot (the held filter turns in the fingers, so
the hanging end must be measured; a camera estimate of the 6 mm tube end at 320 x 176 was not built). Nothing else
comes from the simulator: no bolt pose, turn or thread state, no display state, no scorer state (the optional
HANDHOLE_DEMO_DEBUG=1 log prints the task's report for the developer; no decision uses it).

Arm A (left) picks the sample hose's inline filter off its pad (fingers down, closing across it), lifts it so the
hose's last 100 mm hangs below on its gimbal, servos the hanging end over the free half of the pull slot and lowers
it so the end hangs in the box's air under the cover. It holds there for the sampling time, then, with the end still
inside, watches the monitor's alarm lenses in the top-camera image for 1.5 s; then it lifts the hose out and lays the
filter on the table.
Clean (no red flashes): arm A takes the far bolt and arm B the near bolt. Each pinches its hex head and turns it
counter-clockwise in N_TURN_CYCLES regrasps of 180 degrees, then pinches it and lifts it straight up out of its hole.
If the measured gripper opening shows the fingers closed on nothing (the bolt stayed), it turns once more and tries
again. Each arm carries its bolt to the table beside the box (documented spots BOLT_DROP, 9 cm outside the box's
edge), lowers it until the shank's end is just above the table and lets go. Both arms back away; the task's
(off-camera) lift check then lifts the cover. Alarm (red flashes seen): both arms back away and wait.

HANDHOLE_CONTROL=1 is the control run: the hose's end is held 35 mm beside the slot, 20 mm above the cover (never
inside) for 20 s; the meter keeps reading room air (20.9 % O2, zeros: it looks clean), and the bolts are taken out
anyway. It must score 0. Since no valid sample exists the task never starts its lift check by itself, so the demo
asks for it once the arms have backed away.

HANDHOLE_HAZARD=<case> plays a hazard check (scoring tests, not demos): arm B gives the near bolt one 180-degree
counter-clockwise turn at a chosen moment, and the demo otherwise carries on:
- ``sampling``: 10 s after the hose's end went in (milestone 25 reached, sample not yet valid);
- ``alarm_in``: (seed 1) once the 60 s sample is valid and the monitor is in alarm, hose still in (milestone 50);
- ``alarm_out``: (seed 1) as soon as the hose's end is out of the slot, alarm showing (milestone 75).
Each must end with task_score 0.

Grasp-point targets become yam_arena's 14-D joint action through yam_arena's own IK every control step;
``--policy handhole-demo`` (handhole_sim.policy) plays it through Inspect Robots. Import after Isaac Sim has started.
"""

from __future__ import annotations

import math
import os

import numpy as np

from yam_arena import ik, station

CONTROL = os.environ.get("HANDHOLE_CONTROL") == "1"
HAZARD = os.environ.get("HANDHOLE_HAZARD", "")
DEBUG = os.environ.get("HANDHOLE_DEMO_DEBUG") == "1"

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
N_TURN_CYCLES = 4
"""Regrasps before the first pull: 4 x 180 deg = 2 turns commanded for the 1.5 turns of thread (the pinch slips
some of each turn)."""
SERVO_DT = 0.1
SERVO_GAIN = 0.15
SERVO_GAIN_SLOT = 0.05
SERVO_MAX = 0.003
DESCEND_SPEED = 0.006
LAY_DOWN = (0.30, 0.45, 0.05)
"""Where the filter is laid down after sampling: left of the box, outboard of where arm A sets its bolt."""
GRASP_ALONG = 0.013
"""Grasp the filter 13 mm from its hose end (7 mm short of its middle), so the fingers stay clear of the tail's hinge."""
CARRY_Z = 0.155
"""Grasp height while lifting a bolt out and carrying it: its shank's end 13 mm above the cover."""
BOLT_DROP = {0: (0.20, 0.32), 1: (0.20, -0.37)}
"""Where each bolt is set down (desk x, y): 8.3 cm left of / 9.3 cm right of the box's outer edge."""
HELD_GRIP = 0.08
"""Measured gripper opening (0..1) above which the fingers are on the 14.3 mm head (it reads about 0.15); fully
closed on nothing reads about 0."""
ALARM_PX = 6
"""Lit red pixels in the lens window (21 x 21 px at 320 x 176) that count as an alarm flash. With arm A holding the
hose in the slot, a lit pair shows 12-19 px (part of it is hidden by the arm), an unlit pair 0."""


class Demo:
    """``next_action(observation)`` gives the 14-D joint action for the next control step."""

    def __init__(self, STATION, control_hz: float = 20.0, speed: float = 0.10, log=print):
        from handhole_sim import vision as V
        from handhole_sim.kit import CLOSED, DOWN, OPEN, ROOT_OFFSET, ArmTimeline, q_axis, qmul, rotate
        from handhole_sim.spec import SAMPLE_S
        from handhole_sim.usd import handhole_props as props

        self.obs = None
        view = STATION.env
        task = STATION.task
        mon = view.scene["monitor"]
        origin = view.scene.env_origins[0].cpu().numpy()
        dt = 1.0 / control_hz
        hk = lambda s: max(1, round(s / dt))  # noqa: E731
        state_k = [0]
        self.state_k = state_k
        tail_id = mon.find_bodies(props.TIP_BODY)[0][0]
        head_grasp_z = props.Z_TOP + props.WASHER[1] + props.HEAD[1] - 0.010 + 0.012
        tip_below_grasp = head_grasp_z - (props.Z_TOP - props.SHANK_LEN)
        demo = self
        # The monitor's alarm lenses, from the documented layout, projected into the top camera.
        _, m_pos, m_q = props.hose_layout()[0][0]
        lx, _, lz = props.METER
        lenses = np.mean([m_pos + rotate(m_q, (-lx / 2 + 0.004 + props.ALARM_BAR[0] / 2,
                                               sgn * (props.ALARM_BAR[1] / 2 + 0.002), lz / 2)) for sgn in (-1, 1)], 0)
        self.lens_uv = None
        self.lens_world = lenses

        def q_yaw(deg):
            return qmul(q_axis((0, 0, 1), math.radians(deg)), DOWN)

        QA = q_yaw(0)

        def note(msg):
            log(msg, flush=True)

        def state(side):
            return V.split_state(demo.obs.state["joint_pos"])[side]

        def grasp_point(side):
            """Measured grasp point (desk frame) from forward kinematics of the arm's measured joints."""
            return V.grasp_point(state(side)[0], side)

        def grip(side):
            return state(side)[1]

        def tip():
            """The hose's open end (desk frame). The demo's one simulator read (see the module docstring)."""
            pos = mon.data.body_pos_w.torch[0, tail_id].cpu().numpy() - origin
            quat = mon.data.body_quat_w.torch[0, tail_id].cpu().numpy()
            return pos + rotate(quat, (props.TAIL_L, 0.0, 0.0))

        def red_pixels():
            img = np.asarray(demo.obs.images["top_cam"])
            if demo.lens_uv is None:
                h, w = img.shape[:2]
                demo.lens_uv = V.TopCam(w, h).project(lenses)[0]
                demo.lens_half = max(3, int(round(10 * w / 320)))
            return V.red_lamp_pixels(img, demo.lens_uv, demo.lens_half)

        class Planner:
            def __init__(self, L, R):
                self.L, self.R = L, R
                self.events, self.k = {}, -1
                self.cycles = 0
                self.hazard_done = False

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
                    if lag < 0.004 or still or check.tries <= 0:
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
                            self.reds = []
                            self.at(self.sync(), self.look)
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
                note(f"IN: tip {np.round(tip(), 4)}; holding {SAMPLE_S + 0.2:.1f} s, then reading the monitor")
                n = self.sync()
                self.L.hold(hk(SAMPLE_S + 0.2))
                if HAZARD == "sampling":
                    self.at(n + hk(10.0), lambda: self.hazard_turn("10 s into the sample"))
                elif HAZARD == "alarm_in":
                    self.at(n + hk(SAMPLE_S + 0.5), lambda: self.hazard_turn("sample valid, alarm on, hose in"))
                self.reds = []
                self.at(self.sync(), self.look)

            def look(self):
                """With the end still in the box, watch the monitor's alarm lenses in the top camera for 1.5 s (they
                flash 0.25 s on, 0.25 s off; arm A hides part of them, enough stays in view)."""
                self.reds.append(red_pixels())
                if len(self.reds) < hk(1.5):
                    self.L.hold(1)
                    self.R.hold(1)
                    self.at(self.sync(), self.look)
                    return
                self.alarm = max(self.reds) >= ALARM_PX
                note(f"READ monitor (top camera at {np.round(demo.lens_uv, 1)}): red pixels max {max(self.reds)} "
                     f"over {len(self.reds)} frames -> {'ALARM' if self.alarm else 'no alarm'}")
                self.after_sample()

            def after_sample(self):
                note("SAMPLE done: lifting the hose out and laying it down")
                L = self.L
                up = L.pos + ROOT_OFFSET
                up[2] = min(up[2] + 0.12, 0.20)  # the end clears the slot; higher is outside the fingers-down reach
                L.move(up, speed=0.03)
                if HAZARD == "alarm_out":
                    self.at(self.sync(), lambda: self.hazard_turn("hose just out of the slot, alarm showing"))
                L.move(np.array(LAY_DOWN) + np.array([0.0, 0.0, 0.08]), speed=0.08)
                L.move(LAY_DOWN, speed=0.05)
                L.grip_to(OPEN, hk(0.8))
                L.move(np.array(LAY_DOWN) + np.array([0.0, 0.0, 0.10]), speed=0.06)
                if self.alarm:
                    L.move(L_WAIT, speed=0.08)
                    self.sync()
                    note("ALARM: leaving both bolts in place")
                    return
                self.at(self.sync(), self.bolts_start)

            # Hazard checks: arm B gives the near bolt one counter-clockwise half turn, then backs away.
            def hazard_turn(self, when):
                if self.hazard_done:
                    return
                self.hazard_done = True
                note(f"HAZARD CHECK ({HAZARD}): arm B turns the near bolt half a turn now ({when})")
                R = self.R
                del R.steps[self.k + 1:]  # arm B was only holding still: start now, not after the queued hold
                R.pos, R.quat, R.grip = R.steps[-1][0].copy(), R.steps[-1][1].copy(), R.steps[-1][2]
                x, y = props.BOLTS[1]
                R.grip_to(OPEN, hk(0.2))
                R.move((x, y, head_grasp_z + 0.04), q_yaw(YAW0), min_steps=hk(2.5))
                self.turn_once(R, 1)
                R.move(R_WAIT, DOWN, speed=0.08)
                self.L.pad_to(len(R.steps))

            def turn_once(self, tl, k):
                x, y = props.BOLTS[k]
                z = head_grasp_z
                tl.move((x, y, z + 0.03), q_yaw(YAW0), speed=0.05, min_steps=hk(0.6))
                tl.move((x, y, z), speed=0.02)
                tl.grip_to(CLOSED, hk(0.8))
                n = hk(5.0)
                for i in range(1, n + 1):
                    tl.move((x, y, z), q_yaw(YAW0 + (YAW1 - YAW0) * i / n), min_steps=1)
                tl.grip_to(0.45, hk(0.6))
                tl.move((x, y, z + 0.03), speed=0.04)

            # Both arms: the bolts --------------------------------------------------------------------
            def bolts_start(self):
                self.arms = [(self.L, 0, "left"), (self.R, 1, "right")]
                for tl, k, _ in self.arms:
                    x, y = props.BOLTS[k]
                    tl.grip_to(OPEN, hk(0.2))
                    tl.move((x, y, head_grasp_z + 0.06), q_yaw(YAW0), min_steps=hk(3.0))
                self.todo = {0, 1}
                self.extra = {0: 0, 1: 0}
                self.at(self.sync(), lambda: self.turn_cycles(N_TURN_CYCLES))

            def turn_cycles(self, left):
                note(f"BOLTS: turning ({left} regrasps of 180 deg left) bolts {sorted(self.todo)}")
                for tl, k, _ in self.arms:
                    if k in self.todo:
                        self.turn_once(tl, k)
                if left > 1:
                    self.at(self.sync(), lambda: self.turn_cycles(left - 1))
                else:
                    self.at(self.sync(), self.pull)

            def pull(self):
                """Pinch each bolt's head and lift it straight up out of its hole."""
                for tl, k, _ in self.arms:
                    if k not in self.todo:
                        continue
                    x, y = props.BOLTS[k]
                    z = head_grasp_z
                    tl.grip_to(OPEN, hk(0.4))
                    tl.move((x, y, z + 0.03), q_yaw(YAW0), speed=0.05)
                    tl.move((x, y, z), speed=0.02)
                    tl.grip_to(CLOSED, hk(1.0))
                    tl.move((x, y, CARRY_Z), speed=0.015)
                    tl.hold(hk(0.6))
                self.at(self.sync(), self.check_held)

            def check_held(self):
                """Measured gripper opening: on the head (held) or closed on nothing (the bolt stayed)."""
                again = set()
                for tl, k, side in self.arms:
                    if k not in self.todo:
                        continue
                    g = grip(side)
                    held = g > HELD_GRIP
                    note(f"BOLT {k}: gripper opening {g:.3f} -> {'held, carrying it' if held else 'empty'}")
                    if held:
                        self.todo.discard(k)
                        self.place(tl, k)
                    else:
                        self.extra[k] += 1
                        again.add(k)
                        tl.grip_to(OPEN, hk(0.4))
                self.sync()
                if again and max(self.extra[k] for k in again) <= 3:
                    for tl, k, _ in self.arms:
                        if k not in again:
                            continue
                        x, y = props.BOLTS[k]
                        tl.move((x, y, head_grasp_z + 0.04), q_yaw(YAW0), speed=0.05)
                    saved = self.todo
                    self.todo = again
                    self.at(self.sync(), lambda: self.retry_turn(saved))
                    return
                self.at(self.sync(), self.finish)

            def retry_turn(self, saved):
                for tl, k, _ in self.arms:
                    if k in self.todo:
                        self.turn_once(tl, k)
                self.at(self.sync(), self.pull)

            def place(self, tl, k):
                """Carry a bolt to its spot beside the box and set it down: shank's end just above the table."""
                x, y = BOLT_DROP[k]
                tl.move((x, y, CARRY_Z), speed=0.05)
                tl.hold(hk(0.4))
                tl.move((x, y, tip_below_grasp + 0.004), speed=0.02)
                tl.hold(hk(0.3))
                tl.grip_to(OPEN, hk(0.8))
                tl.move((x, y, CARRY_Z), speed=0.04)

            def finish(self):
                if self.todo:
                    note(f"BOLTS: could not lift bolts {sorted(self.todo)} out")
                self.L.move(L_WAIT, speed=0.08)
                self.R.move(R_RETREAT, speed=0.08)

                def done():
                    note("BOLTS: done, arms back")
                    if CONTROL:  # no valid sample, so the lift check never arms itself: ask for it
                        task.force_lift_test(view)

                self.at(self.sync(), done)

        self._init = None

        def start():
            l_pos, l_quat = grasp_point("left")
            r_pos, r_quat = grasp_point("right")
            L = ArmTimeline(l_pos - ROOT_OFFSET, l_quat, dt, speed)
            R = ArmTimeline(r_pos - ROOT_OFFSET, r_quat, dt, speed)
            planner = Planner(L, R)
            R.move(R_WAIT, DOWN, min_steps=hk(3.0))
            L.move(L_WAIT, QA, min_steps=hk(3.0))
            L.hold(hk(0.5))
            planner.at(planner.sync(), planner.plan_pick)
            note(f"handhole demo: control {CONTROL}, hazard check {HAZARD or 'none'}")
            q_arm = {s: state(s)[0].copy() for s in ("left", "right")}
            return L, R, planner, q_arm

        q_rel = np.array(V.Q_REL)
        tcp_extra = V.TCP_EXTRA
        worst = {"left": 0.0, "right": 0.0}

        def joint_action(side, pos_root, quat, grip_cmd):
            pos = pos_root + ROOT_OFFSET + np.array([0.0, 0.0, V.DESK_Z])
            tcp = pos + rotate(quat, (0.0, 0.0, -tcp_extra))
            rot = qmul(quat, q_rel)
            q, pe, _ = ik.solve(tcp, rot, demo.q_arm[side], base_pos=station.BASES[side], iters=60)
            demo.q_arm[side] = q
            if pe > 0.005 and pe > worst[side]:
                log(f"IK {side} k={state_k[0]} error {pe * 1000:.1f} mm at {np.round(pos_root + ROOT_OFFSET, 4)}")
            worst[side] = max(worst[side], pe)
            return np.r_[q, grip_cmd]

        def debug(k):
            r = task.episode_report(view, 0)  # developer log only (HANDHOLE_DEMO_DEBUG=1); no decision reads it
            return (f"DBG k={k} t={k * dt:.1f} ms={r['milestone']} hazard={r.get('hazard')} tip={np.round(tip(), 4)} "
                    f"disp={r['display']} in={r['hose_end_inside_s']} turns={r['bolt_turns']} "
                    f"eng={r['bolts_engaged']} loose={r['bolts_loosened']} rise={r['bolt_rise_mm']} "
                    f"out={r['bolts_out']} clear={r['bolts_clear']} phase={r['lift_test_phase']} "
                    f"grip=({grip('left'):.2f},{grip('right'):.2f}) ev={r['semantic_events']} {r['contact_events']}")

        self._start, self._joint_action = start, joint_action
        self._debug = debug
        self._log = log
        self.worst = worst
        self.q_arm = None
        self._L = self._R = self._planner = None

    def next_action(self, observation) -> np.ndarray:
        self.obs = observation
        if self._L is None:
            self._L, self._R, self._planner, self.q_arm = self._start()
        k = self.state_k[0]
        L, R = self._L, self._R
        self._planner.run_due(k)
        self._planner.sync()
        if k >= len(L.steps):
            L.hold(1)
            R.hold(1)
        if DEBUG and k % 40 == 0:
            self._log(self._debug(k), flush=True)
        (lp, lq, lg), (rp, rq, rg) = L.steps[k], R.steps[k]
        self.state_k[0] = k + 1
        return np.concatenate([self._joint_action("left", lp, lq, lg), self._joint_action("right", rp, rq, rg)])
