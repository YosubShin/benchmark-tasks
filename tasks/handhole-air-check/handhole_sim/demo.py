"""The scripted demo, closed loop.

What it reads. Every control step it gets the observation Inspect Robots hands any policy: the three RGB images
(``top_cam``, ``left_cam``, ``right_cam``) and the 14-D ``joint_pos`` state. Grasp points come from forward kinematics
of the measured joints (``handhole_sim.vision``, the arm's URDF at the station's calibrated bases); the monitor's
alarm is read from the top-camera image; whether the probe went down into the slot or stopped on the cover, and
whether a bolt is held, are felt through the measured joints (the grasp point's measured height, the gripper's
measured opening). Fixed knowledge: the documented layout (task.json ``setup``) and the station's published camera
calibration. Nothing comes from the simulator: no prop pose, no probe or bolt state, no display state, no scorer
state (the optional HANDHOLE_DEMO_DEBUG=1 log prints the task's report for the developer; no decision uses it).

Arm A (left) grasps the sampling probe's handle where it stands in its holster (fingers down, closing across the
handle along x, so any tilt of the probe in the fingers swings its tip along the slot's length), lifts it out,
carries it over the free half of the pull slot and lowers it at 6 mm/s until the stop collar rests on the cover. If
the measured grasp height stops falling above the seated height (the tip landed on the cover beside the slot), it
lifts 15 mm and tries again a few mm to the side. Before descending, it finds the probe's blue tip band in the left
wrist camera and corrects the grasp so the tip is over the slot. It holds the probe seated for the sampling time,
lifts it out, moves it aside (READ_AT, where arm A no longer hides the monitor from the top camera) and watches the
monitor's alarm lenses in the top-camera image for 1.5 s (the alarm latches); then it stands the probe down on the
table left of the box.
Clean (no red flashes): arm A takes the far bolt and arm B the near bolt. Each pinches its hex head and turns it
counter-clockwise in N_TURN_CYCLES regrasps of 180 degrees, then pinches it and lifts it straight up out of its hole.
If the measured gripper opening shows the fingers closed on nothing (the bolt stayed), it turns once more and tries
again. Each arm carries its bolt to the table beside the box (documented spots BOLT_DROP, 8-9 cm outside the box's
edge), lowers it until the shank's end is just above the table and lets go. Both arms back away; the task's
(off-camera) lift check then lifts the cover. Alarm (red flashes seen): both arms back away and wait.

HANDHOLE_CONTROL=1 is the control run: the probe's tip is held 35 mm beside the slot, 20 mm above the cover (never
in) for 20 s; the meter keeps reading room air (20.9 % O2, zeros: it looks clean), and the bolts are taken out
anyway. It must score 0. Since no valid sample exists the task never starts its lift check by itself, so the demo
asks for it once the arms have backed away.

HANDHOLE_HAZARD=<case> plays a hazard check (scoring tests, not demos): arm B gives the near bolt one 180-degree
counter-clockwise turn at a chosen moment, and the demo otherwise carries on:
- ``sampling``: 10 s after the probe was seated (milestone 25 reached, sample not yet valid);
- ``alarm_in``: (seed 1) once the 60 s sample is valid and the monitor is in alarm, probe still in (milestone 50);
- ``alarm_out``: (seed 1) as soon as the probe's tip is out of the slot, alarm showing (milestone 75).
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

L_WAIT = (0.22, 0.26, 0.22)
R_WAIT = (0.22, -0.26, 0.22)
R_RETREAT = (0.20, -0.14, 0.24)
PROBE_YAW = 90.0
"""Fingers close along x on the probe's square handle: a tilt in the fingers then swings the tip along the slot's
51 mm length, not across its 13 mm width."""
PRESS_M = 0.0015
"""The grasp is commanded this far below the seated height, so the collar stays down on the cover."""
SEATED_TOL = 0.004
"""Measured grasp height within this of the seated height: the collar is on the cover."""
DESCEND_SPEED = 0.006
STALL_S = 1.0
"""No measured descent (under 1 mm) for this long above the seated height: the tip is on the cover, not in the slot."""
SEARCH = ((0.0, 0.0), (0.0, -0.004), (0.0, 0.004), (-0.002, 0.0), (0.002, 0.0), (0.0, -0.007), (0.0, 0.007))
"""Grasp offsets (x, y) tried in turn after a stall."""
YAW0, YAW1 = -90.0, 90.0
"""Each regrasp turns the head 180 deg (more is past the wrist joint's range with the fingers down)."""
N_TURN_CYCLES = 4
"""Regrasps before the first pull: 4 x 180 deg = 2 turns commanded for the 1.5 turns of thread (the pinch slips
some of each turn)."""
READ_AT = (0.30, 0.30)
"""Where arm A holds the probe (just lifted out of the slot) while it reads the monitor: from here its links leave the
monitor in the top camera's view (with the probe seated over the slot they hide it)."""
LAY_DOWN = (0.30, 0.45)
"""Where the probe is stood down after sampling (it falls over there): left of the box, outboard of where arm A sets
its bolt; farther from the box than the probe is long."""
CARRY_Z = 0.165
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
        dt = 1.0 / control_hz
        hk = lambda s: max(1, round(s / dt))  # noqa: E731
        state_k = [0]
        self.state_k = state_k
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
        self.wrist = None

        def q_yaw(deg):
            return qmul(q_axis((0, 0, 1), math.radians(deg)), DOWN)

        QA = q_yaw(0)
        QP = q_yaw(PROBE_YAW)
        hx, hy = props.HOLSTER_XY
        grasp_rest = np.array([hx, hy, props.Z_TOP + props.GRASP_Z])
        """Where the handle is grasped with the probe in its holster (collar on the holster's top, at Z_TOP)."""
        tip_below_grasp_p = props.GRASP_Z + props.PROBE_L
        seated_z = props.Z_TOP + props.GRASP_Z
        """Grasp height with the collar resting on the cover's top face (the cover's top is at Z_TOP too)."""

        def note(msg):
            log(msg, flush=True)

        def state(side):
            return V.split_state(demo.obs.state["joint_pos"])[side]

        def grasp_point(side):
            """Measured grasp point (desk frame) from forward kinematics of the arm's measured joints."""
            return V.grasp_point(state(side)[0], side)

        def grip(side):
            return state(side)[1]

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

            # Arm A: the sampling probe ----------------------------------------------------------------
            def plan_pick(self):
                L = self.L
                L.grip_to(OPEN, hk(0.2))
                L.move(grasp_rest + np.array([0.0, 0.0, 0.06]), QP, min_steps=hk(3.0))
                L.hold(hk(0.5))
                self.sync()
                self.when_settled("left", self.descend)

            def descend(self):
                L = self.L
                L.move(grasp_rest + np.array([0.0, 0.0, 0.01]), speed=0.04)
                L.move(grasp_rest, speed=0.02)
                L.hold(hk(0.5))
                self.sync()
                self.when_settled("left", self.close)

            def close(self):
                e = grasp_point("left")[0] - grasp_rest
                note(f"PICK: grasp error before closing (mm) {np.round(e * 1000, 1)}")
                L = self.L
                L.grip_to(CLOSED, hk(1.0))
                L.move(grasp_rest + np.array([0.0, 0.0, 0.09]), speed=0.03)  # the tip clears the holster's top
                self.over = np.array([props.SLOT_XY[0], props.SLOT_OPENING_Y, 0.0])
                if CONTROL:
                    self.over[0] -= 0.035
                self.over[2] = props.Z_TOP + tip_below_grasp_p + (0.020 if CONTROL else 0.012)
                self.tip_goal = self.over[:2].copy()
                L.move(self.over, speed=0.06)
                L.hold(hk(1.0))
                self.sync()
                self.tries = 0
                self.aligns = 0
                self.when_settled("left", self.align)

            def align(self):
                """Find the probe's blue tip band in the left wrist camera, place it at its known height (from the
                measured grasp height) and move the grasp so the tip is over the slot's free half."""
                g, _ = grasp_point("left")
                img = np.asarray(demo.obs.images["left_cam"])
                if demo.wrist is None:
                    h, w = img.shape[:2]
                    demo.wrist = V.WristCam("left", w, h)
                uv, npx = V.blue_blob(img)
                if uv is None:
                    note(f"ALIGN: tip band not seen ({npx} px); inserting from the documented position")
                    self.insert()
                    return
                z_band = g[2] - tip_below_grasp_p + props.TIP_BAND / 2
                tip = demo.wrist.ray_to_z(state("left")[0], uv, z_band)
                e = tip[:2] - self.tip_goal
                note(f"ALIGN {self.aligns}: band at px {np.round(uv, 1)} ({npx} px) -> tip (desk) {np.round(tip, 4)}, "
                     f"off the slot target by (mm) {np.round(e * 1000, 1)}")
                self.aligns += 1
                if np.abs(e).max() < 0.0008 or self.aligns > 6:
                    self.insert()
                    return
                self.over[:2] -= 0.9 * e
                self.L.move(self.over, speed=0.02)
                self.L.hold(hk(0.6))
                self.sync()
                self.when_settled("left", self.align)

            def insert(self):
                """Lower the probe at DESCEND_SPEED, watching the measured grasp height (joint states, FK)."""
                if CONTROL:
                    note("CONTROL: holding the probe's tip beside the slot for 20 s")
                    self.L.hold(hk(20.0))
                    self.alarm = False
                    self.at(self.sync(), self.after_sample)
                    return
                dx, dy = SEARCH[self.tries % len(SEARCH)]
                start = self.over + np.array([dx, dy, 0.0])
                note(f"INSERT try {self.tries}: offset (mm) {dx * 1000:.0f}, {dy * 1000:.0f}")
                L = self.L
                L.move(start, speed=0.03)
                L.hold(hk(0.5))
                n0 = self.sync()
                L.move(np.r_[start[:2], seated_z - PRESS_M], speed=DESCEND_SPEED)
                self.descent_end = self.sync()
                self.zs = []
                self.at(n0, self.watch_descent)

            def watch_descent(self):
                z = grasp_point("left")[0][2]
                self.zs.append(z)
                if z < seated_z + SEATED_TOL:
                    note(f"SEATED: grasp z {z:.4f} (seated {seated_z:.4f}); holding {SAMPLE_S + 0.2:.1f} s, then "
                         "reading the monitor")
                    del self.L.steps[self.k + 1:]
                    self.L.pos = self.L.steps[-1][0].copy()
                    self.L.move(np.r_[self.L.pos[:2] + ROOT_OFFSET[:2], seated_z - PRESS_M], min_steps=hk(0.5))
                    self.sampling()
                    return
                w = hk(STALL_S)
                if len(self.zs) > w and self.zs[-w] - z < 0.001 or self.k >= self.descent_end + hk(2.0):
                    self.tries += 1
                    note(f"STALL: grasp z {z:.4f}, {1000 * (z - seated_z):.1f} mm above seated (the tip is on the "
                         f"cover); retry {self.tries}")
                    del self.L.steps[self.k + 1:]
                    self.L.pos = self.L.steps[-1][0].copy()
                    if self.tries > 2 * len(SEARCH):
                        note("INSERT: giving up")
                        return
                    self.L.move(self.L.pos + ROOT_OFFSET + np.array([0.0, 0.0, 0.015]), speed=0.02)
                    self.sync()
                    self.when_settled("left", self.insert)
                    return
                self.L.pad_to(self.k + 2)
                self.at(self.k + 1, self.watch_descent)

            def sampling(self):
                n = self.sync()
                # (hazard check alarm_in only: keep the probe seated 12 s longer, so arm B's half turn lands while
                # the probe is still in)
                self.L.hold(hk(SAMPLE_S + 0.2 + (12.0 if HAZARD == "alarm_in" else 0.0)))
                if HAZARD == "sampling":
                    self.at(n + hk(10.0), lambda: self.hazard_turn("10 s into the sample"))
                elif HAZARD == "alarm_in":
                    self.at(n + hk(SAMPLE_S + 0.5), lambda: self.hazard_turn("sample valid, alarm on, probe in"))
                self.at(self.sync(), self.after_sample)

            def look(self):
                """With the probe just out of the slot and arm A moved aside (READ_AT), watch the monitor's alarm lenses
                in the top camera for 1.5 s (they flash 0.25 s on, 0.25 s off, and latch). While the probe is seated,
                arm A's upper link hides the monitor from the top camera."""
                self.reds.append(red_pixels())
                if len(self.reds) < hk(1.5):
                    self.L.hold(1)
                    self.R.hold(1)
                    self.at(self.sync(), self.look)
                    return
                self.alarm = max(self.reds) >= ALARM_PX
                note(f"READ monitor (top camera at {np.round(demo.lens_uv, 1)}): red pixels max {max(self.reds)} "
                     f"over {len(self.reds)} frames -> {'ALARM' if self.alarm else 'no alarm'}")
                self.stand_down()

            def after_sample(self):
                note("SAMPLE done: lifting the probe out, moving aside to read the monitor")
                L = self.L
                up = L.pos + ROOT_OFFSET
                up[2] = props.Z_TOP + tip_below_grasp_p + 0.015  # the tip clears the cover
                L.move(up, speed=0.03)
                if HAZARD == "alarm_out":
                    self.at(self.sync(), lambda: self.hazard_turn("probe just out of the slot, alarm showing"))
                L.move(np.r_[READ_AT, up[2]], speed=0.08)
                L.hold(hk(0.3))
                self.reds = []
                self.at(self.sync(), self.look)

            def stand_down(self):
                L = self.L
                down = np.array([LAY_DOWN[0], LAY_DOWN[1], tip_below_grasp_p + 0.006])
                L.move(down + np.array([0.0, 0.0, 0.08]), speed=0.08)
                L.move(down, speed=0.04)
                L.grip_to(OPEN, hk(0.8))
                L.move(down + np.array([0.0, 0.0, 0.08]), speed=0.06)
                if self.alarm:
                    L.move(L_WAIT, QA, speed=0.08)
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
            return (f"DBG k={k} t={k * dt:.1f} ms={r['milestone']} hazard={r.get('hazard')} tip={r['probe_tip']} "
                    f"collar={r['probe_collar']} tilt={r['probe_tilt_deg']} axis={r['probe_axis']} "
                    f"graspL={np.round(grasp_point('left')[0], 4)} "
                    f"depth_min={r['tip_below_underside_min_mm']} disp={r['display']} in={r['probe_in_s']} "
                    f"turns={r['bolt_turns']} "
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
