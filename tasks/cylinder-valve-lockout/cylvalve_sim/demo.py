"""The scripted demo, closed loop.

1. Arm B (right), fingers down, closes on the argon valve's handwheel and turns it clockwise (seen from above) by
   rolling its wrist, aiming a little ahead of the measured wheel angle, until the wheel seats; it lets go and parks.
2. Arm A (left), fingers pointing right (toward -y), closes across the cap's can near its top, lifts it off its stand
   (plate swung open), carries it over the argon valve with its rim above the handwheel, lines its axis up with the
   valve's from the measured can position and lowers it until the plate is level with the gap between the neck ring
   and the valve's wrench block, and holds it there.
3. Arm B swings the plate shut round the valve's neck: fingers down, it pushes the plate's tab round the rivet from
   behind the cap (aiming ahead of the measured plate angle); for the last part it comes in from the right with its
   fingers pointing left and pushes the tab against the can's tab. Arm A lets go of the cap, which settles on the
   plate.
4. Arm A picks the open padlock off its pedestal (as in the plug-lockout demo), slides it so the two tabs enter the
   gap under the shackle's short leg, lines the leg up with the holes from the measured tip position and pushes it
   through both tabs; arm B pushes the shackle's bend shut while arm A holds the body; both let go.

``control=True``: steps 1 to 3 only; once both hands are clear the after-release check is forced (the plate, shut
but not padlocked, must swing open and the cap lift off).

It reads the simulator's state (wheel and plate angles, the cap's and padlock's poses), like any scripted demo with
privileged state; ``--policy cylinder-lockout-demo`` (cylvalve_sim.policy) plays it through Inspect Robots.
Grasp-point targets become yam_arena's 14-D joint action through yam_arena's own IK every control step. Import after
Isaac Sim has started.
"""

from __future__ import annotations

import math
import os

import numpy as np

from yam_arena import ik, station

STOP_AFTER = os.environ.get("CYLVALVE_DEMO_STOP", "")
"""Development only: 'turn', 'cap' or 'plate' ends the demo (both arms park) after that stage."""

WHEEL_TCP_DZ = -0.004
"""Grasp point below the handwheel's top: the pads cover its rim, the fingertips clear the outlet."""
WHEEL_YAW0_DEG = 118.0
YAW_MIN_DEG = -76.0
TURN_LEAD0, TURN_LEAD_MAX = 6.0, 40.0
CAN_GRASP = (0.0, 0.018, 0.130)
"""Arm A's grasp point in the can's frame: 13 cm up the can (3 cm below its top), 18 mm toward the left of its axis
(the pads meet the can 80 mm apart)."""
CARRY_RIM_Z = 0.190
"""The can's rim height while it is carried over the valve (the handwheel's top is at 17.3 cm)."""
HOLD_RIM_ABOVE_RING = 0.009
"""Where arm A holds the can while the plate swings: the plate 4.5 mm above the neck ring, 5.5 mm under the wrench
block."""
PUSH_Z = -0.0025
"""Height of the plate's tab (its mid-plane, below the rim) where arm B pushes on its edge."""
PUSH_TIP_BELOW = 0.007
PUSH_STAND = 0.018
"""Arm B's closed fingertips' centre this far behind the tab's centre line along the push (the tab is 20 mm wide)."""
SWITCH_DEG = 50.0
"""Below this plate angle arm B pushes from the right with its fingers pointing left (and a little back): from
above, its gripper would meet the can."""
SIDE_YAW_DEG = 112.0
PUSH_YAW_DEG = 90.0
"""Fingers down, the gripper's long side along x: it stays clear of the can as the tab comes round its right side."""
SIDE_PITCH_DEG = 25.0
PLATE_LEAD0, PLATE_LEAD_MAX = 8.0, 30.0
LOCK_PITCH_DEG = 35.0
GRASP_IN_LOCK = (-0.010, 0.0, -0.008)
"""Arm A's grasp point in the padlock's frame: the outer (long-leg) half of the body, below its middle."""
PRESS_PITCH_DEG = 70.0
TIP_OVER = 0.002
"""The padlock comes in with the two tabs in the 12 mm gap between its body and the open shackle's short leg: the
leg's tip 2 mm over the can's tab, the body 2 mm under the plate's tab."""
PIERCE = 0.0115
"""... and then goes down this far, so the short leg passes down through both holes (its tip 1.5 mm under them)."""


class Demo:
    """``next_action()`` gives the 14-D joint action for the next control step."""

    def __init__(self, STATION, control_hz: float = 20.0, speed: float = 0.10, control: bool = False, log=print):
        from cylvalve_sim import station as hb
        from cylvalve_sim.kit import (CLOSED, DOWN, FORWARD, OPEN, ROOT_OFFSET, ArmTimeline, GraspModel, conj, q_axis,
                                      qmul, rotate)
        from cylvalve_sim.kit import grasp_point as grasp_point_world
        from cylvalve_sim.spec import TARGET_XY
        from cylvalve_sim.task import PV, lock_start_pose
        from cylvalve_sim.usd import cylvalve_props as props

        view, task = STATION.env, STATION.task
        st = task.state(view)
        state_k = [0]
        self.state_k = state_k
        origin = view.scene.env_origins[0].cpu().numpy()
        robot = view.scene["robot"]
        dt = 1.0 / control_hz
        hold_k = lambda s: max(1, round(s / dt))  # noqa: E731
        EX, EY, EZ = np.eye(3)
        TX, TY = TARGET_XY
        cyl, cap, lock = view.scene["cylinders"], view.scene["cap"], view.scene["padlock"]
        wheel_joint = cyl.find_joints("wheel_target_joint")[0][0]
        plate_joint = cap.find_joints("plate_joint")[0][0]
        can_id = cap.find_bodies("can")[0][0]
        lock_body = lock.find_bodies("body")[0][0]
        shackle_joint = lock.find_joints("shackle_joint")[0][0]
        SIDE_L = qmul(q_axis((0, 0, 1), -math.pi / 2), FORWARD)
        """Arm A on the cap: fingers pointing right (-y), closing along x."""
        SP = math.radians(SIDE_PITCH_DEG)
        SY = math.radians(SIDE_YAW_DEG)
        SIDE_R = qmul(q_axis((0, 0, 1), SY), qmul(q_axis((0, 1, 0), SP), FORWARD))
        SIDE_R_DIR = np.array([math.cos(SY) * math.cos(SP), math.sin(SY) * math.cos(SP), -math.sin(SP)])
        """Arm B's last push: fingers pointing left (+y), pitched down."""
        TIP_PAST = 0.013

        def grasp_point(side):
            p, q = grasp_point_world(robot, side)
            return p - origin, q

        def wheel_deg():
            return math.degrees(float(cyl.data.joint_pos.torch[0, wheel_joint]))

        def plate_deg():
            return math.degrees(float(cap.data.joint_pos.torch[0, plate_joint]))

        def can_pose():
            return (cap.data.body_pos_w.torch[0, can_id].cpu().numpy() - origin,
                    cap.data.body_quat_w.torch[0, can_id].cpu().numpy())

        def hand_down(yaw_deg):
            return qmul(q_axis((0, 0, 1), math.radians(yaw_deg)), DOWN)

        def tab_point(theta_deg, dz=PUSH_Z):
            """A point of the plate's tab (outer part, mid-plane) with the plate at theta, in the world."""
            px, py = props.PIVOT
            e0 = np.array([props.TAB_HOLE_X - 0.006 - px, -py])
            t = math.radians(theta_deg)
            c, s = math.cos(t), math.sin(t)
            e = np.array([px + c * e0[0] - s * e0[1], py + s * e0[0] + c * e0[1], dz])
            pos, quat = can_pose()
            return pos + rotate(quat, e)

        def push_dir(theta_deg):
            d = tab_point(theta_deg - 1.0) - tab_point(theta_deg)
            d[2] = 0.0
            return d / np.linalg.norm(d)

        R_WAIT = np.array((0.16, -0.20, 0.24))
        L_WAIT = np.array((0.16, 0.22, 0.26))
        LOCK_QUAT = np.array(lock_start_pose()[3:])
        PITCH_A = math.radians(LOCK_PITCH_DEG)
        QA = qmul(q_axis((0, 1, 0), PITCH_A), FORWARD)
        """Arm A on the upright padlock: fingers pointing forward (+x) and down, closing on the body's broad faces."""
        A_FINGER_DIR = np.array((math.cos(PITCH_A), 0.0, -math.sin(PITCH_A)))
        PB = math.radians(PRESS_PITCH_DEG)
        QB = qmul(q_axis((0, 1, 0), PB), FORWARD)
        """Arm B on the shackle's bend: fingers pointing steeply down and a little forward, its hand high above arm A's,
        leaning away from the can."""
        B_FINGER_DIR = np.array((math.cos(PB), 0.0, -math.sin(PB)))

        for side, name in (("left", "left_robot"), ("right", "right_robot")):
            qj = view.scene[name].data.joint_pos.torch[0, :6].cpu().numpy()
            tcp, _ = ik.tcp_pose(qj, station.BASES[side])
            log(f"CHECK {side}: their TCP {np.round(tcp, 4)} our grasp point {np.round(grasp_point(side)[0] + origin, 4)}",
                flush=True)

        class Planner:
            def __init__(self, L, R):
                self.L, self.R, self.events = L, R, {}

            def at(self, step, fn):
                self.events.setdefault(step, []).append(fn)

            def run_due(self, k):
                for fn in self.events.pop(k, []):
                    fn()

            def when_settled(self, tl, side, fn, tries=12, tol=0.003):
                def check():
                    lag = np.linalg.norm(grasp_point(side)[0] - (tl.pos + ROOT_OFFSET))
                    if lag < tol or check.tries <= 0:
                        if lag >= tol:
                            log(f"SETTLE timeout ({side}): lag {lag * 1000:.1f} mm", flush=True)
                        fn()
                    else:
                        check.tries -= 1
                        tl.hold(hold_k(0.1))
                        self.at(len(tl.steps), check)

                check.tries = tries
                self.at(len(tl.steps), check)

            def sync(self):
                n = max(len(self.L.steps), len(self.R.steps))
                self.L.pad_to(n)
                self.R.pad_to(n)
                return n

            def report(self, tag):
                rep = task.episode_report(view, 0)
                log(f"{tag}: milestone {rep['milestone']} wheel {rep['target_wheel_deg']} deg, decoy max "
                    f"{rep['decoy_wheel_max_deg']} deg, plate {rep['plate_deg']} deg, cap offset {rep['cap_offset_mm']} mm "
                    f"rim {rep['cap_rim_above_ring_mm']} mm, events {rep['semantic_events']}", flush=True)
                return rep

            def finish(self):
                self.sync()
                self.L.move(L_WAIT, speed=0.08)
                self.R.move(R_WAIT, speed=0.08)
                self.sync()
                self.L.hold(hold_k(3.0))
                self.R.hold(hold_k(3.0))

            # -- 1. close the valve -----------------------------------------------------------------
            def plan_turn(self):
                R = self.R
                self.wheel_top = np.array([TX, TY, props.WHEEL_Z[1] + WHEEL_TCP_DZ])
                R.grip_to(OPEN, hold_k(0.2))
                R.move(self.wheel_top + 0.03 * EZ, hand_down(WHEEL_YAW0_DEG), min_steps=hold_k(2.5))
                R.hold(hold_k(0.3))
                self.when_settled(R, "right", self._wheel_descend)

            def _wheel_descend(self):
                err = grasp_point("right")[0] - (self.R.pos + ROOT_OFFSET)
                self.wcorr = np.clip(-err * np.array([1.0, 1.0, 0.0]), -0.006, 0.006)
                log(f"WHEEL approach: lag (mm) {np.round(err * 1000, 1)}", flush=True)
                self.R.move(self.wheel_top + self.wcorr + 0.015 * EZ, speed=0.04)
                self.R.move(self.wheel_top + self.wcorr, speed=0.015)
                self.R.hold(hold_k(0.3))
                self.when_settled(self.R, "right", self._wheel_grip, tol=0.002)

            def _wheel_grip(self):
                g = grasp_point("right")[0]
                log(f"WHEEL grip: hand {np.round(g, 4)} target {np.round(self.wheel_top, 4)} wheel {wheel_deg():.1f}",
                    flush=True)
                self.R.grip_to(CLOSED, hold_k(0.8))
                self.yaw = self.yaw_base = WHEEL_YAW0_DEG
                self.at(len(self.R.steps), lambda: self._turn(0, TURN_LEAD0, wheel_deg(), wheel_deg()))

            def _turn(self, k, lead, q_grip, last_q):
                q = wheel_deg()
                if k % 5 == 0:
                    log(f"TURN {k}: wheel {q:.1f} deg open, hand yaw {self.yaw:.0f}, lead {lead:.0f}", flush=True)
                if q <= 0.5 or k > 300:
                    self.R.hold(hold_k(0.4))
                    self.at(len(self.R.steps), self._turned)
                    return
                rate = last_q - q
                if rate < 1.0:
                    lead = min(TURN_LEAD_MAX, lead + 2.0)
                elif rate > 3.0:
                    lead = max(TURN_LEAD0, lead - 2.0)
                target = self.yaw_base - (q_grip - q) - lead
                if target < YAW_MIN_DEG and q < 12.0:
                    target = YAW_MIN_DEG
                if target < YAW_MIN_DEG:
                    # Out of wrist roll: let go, turn the hand back, take the wheel again.
                    log(f"REGRASP at wheel {q:.1f} deg", flush=True)
                    R = self.R
                    R.grip_to(OPEN, hold_k(0.4))
                    R.move(self.wheel_top + self.wcorr + 0.012 * EZ, speed=0.03)
                    self.yaw_base = WHEEL_YAW0_DEG
                    self.yaw = WHEEL_YAW0_DEG
                    R.move(self.wheel_top + self.wcorr + 0.012 * EZ, hand_down(self.yaw), speed=0.03,
                           min_steps=hold_k(1.5))
                    R.move(self.wheel_top + self.wcorr, speed=0.015)
                    R.grip_to(CLOSED, hold_k(0.8))
                    self.at(len(R.steps), lambda: self._turn(1, TURN_LEAD0, wheel_deg(), wheel_deg()))
                    return
                self.yaw = target
                self.R.move(self.wheel_top + self.wcorr, hand_down(target), speed=0.05, min_steps=hold_k(0.15))
                self.at(len(self.R.steps), lambda: self._turn(k + 1, lead, q_grip, q))

            def _turned(self):
                self.report("TURNED")
                R = self.R
                R.grip_to(OPEN, hold_k(0.5))
                R.move(self.wheel_top + self.wcorr + 0.06 * EZ, speed=0.04)
                R.move(R_WAIT, hand_down(0.0), speed=0.08)
                if STOP_AFTER == "turn":
                    self.finish()
                    return
                self.sync()
                self.plan_cap()

            # -- 2. fit the cap -------------------------------------------------------------------
            def _can_to(self, pos, speed, quat=None):
                g, q = self.cmodel.grasp_for(pos, self.can_q if quat is None else quat)
                self.L.move(g, q, speed=speed)

            def plan_cap(self):
                p, q = can_pose()
                self.can_q = q
                self.cmodel = GraspModel.nominal(SIDE_L, q, CAN_GRASP)
                grasp, _ = self.cmodel.grasp_for(p, q)
                self.cap_grasp = grasp
                L = self.L
                L.grip_to(OPEN, hold_k(0.2))
                L.move(grasp + 0.06 * EZ, SIDE_L, min_steps=hold_k(2.5))
                L.hold(hold_k(0.3))
                self.when_settled(L, "left", self._cap_in)

            def _cap_in(self):
                err = grasp_point("left")[0] - (self.L.pos + ROOT_OFFSET)
                self.ccorr = np.clip(-err * np.array([1.0, 1.0, 0.0]), -0.006, 0.006)
                log(f"CAP approach: lag (mm) {np.round(err * 1000, 1)}", flush=True)
                self.L.move(self.cap_grasp + self.ccorr + 0.02 * EZ, speed=0.03)
                self.L.move(self.cap_grasp + self.ccorr, speed=0.015)
                self.L.hold(hold_k(0.4))
                self.when_settled(self.L, "left", self._cap_close, tol=0.002)

            def _cap_close(self):
                g = grasp_point("left")[0]
                log(f"CAP grasp: hand {np.round(g, 4)} planned {np.round(self.cap_grasp, 4)}", flush=True)
                self.L.grip_to(CLOSED, hold_k(1.0))
                self.at(len(self.L.steps), self._cap_lift)

            def _cap_lift(self):
                self.cmodel = GraspModel.measure(robot, "left", cap)
                p, q = can_pose()
                log(f"CAP held: grasp rel (mm) {np.round(self.cmodel.rel_pos * 1000, 1)}, plate {plate_deg():.1f}",
                    flush=True)
                self._can_to(p + 0.03 * EZ, 0.03)
                self._can_to(np.array([p[0], p[1], CARRY_RIM_Z]), 0.05)
                self._can_to(np.array([TX, TY, CARRY_RIM_Z]), 0.06)
                self.L.hold(hold_k(0.6))
                self.corr = np.zeros(3)
                self.qcmd = np.array(self.can_q, dtype=float)
                self.when_settled(self.L, "left", lambda: self._over(0))

            def _tilt(self, q):
                return math.degrees(math.acos(min(1.0, rotate(q, EZ)[2])))

            def _tilt_fix(self, q):
                """Turn the commanded can orientation against its measured tilt (the soft wrist lets it hang)."""
                err_q = qmul(q, conj(self.can_q))
                self.qcmd = qmul(conj(err_q), self.qcmd)
                self.qcmd = self.qcmd / np.linalg.norm(self.qcmd)

            def _over(self, attempt):
                p, q = can_pose()
                goal = np.array([TX, TY, CARRY_RIM_Z])
                err = p - goal
                log(f"OVER {attempt}: can error (mm) {np.round(err * 1000, 1)}, tilt "
                    f"{math.degrees(math.acos(min(1.0, rotate(q, EZ)[2]))):.1f} deg, plate {plate_deg():.1f}", flush=True)
                self._tilt_fix(q)
                if attempt < 4 and (np.abs(err[:2]).max() > 0.0012 or self._tilt(q) > 1.5):
                    self.corr = self.corr + err * np.array([1.0, 1.0, 0.0])
                    self._can_to(goal - self.corr, 0.02, self.qcmd)
                    self.L.hold(hold_k(0.4))
                    self.when_settled(self.L, "left", lambda: self._over(attempt + 1))
                    return
                self._lower(0)

            def _lower(self, stage):
                rim = [0.150, 0.120, props.RING_Z[1] + HOLD_RIM_ABOVE_RING]
                if stage >= len(rim):
                    log(f"HELD: tilt {self._tilt(can_pose()[1]):.1f} deg", flush=True)
                    self.report("CAP over valve")
                    if STOP_AFTER == "cap":
                        self.finish()
                        return
                    self.sync()
                    self.plan_plate()
                    return
                p, q = can_pose()
                err = p - np.array([TX, TY, p[2]])
                self.corr = self.corr + 0.8 * err * np.array([1.0, 1.0, 0.0])
                if self._tilt(q) > 1.0:
                    self._tilt_fix(q)
                log(f"LOWER {stage}: can {np.round(p, 4)} xy error (mm) {np.round(err[:2] * 1000, 1)} tilt "
                    f"{self._tilt(q):.1f} deg", flush=True)
                self._can_to(np.array([TX, TY, rim[stage]]) - self.corr, 0.02 if stage < 2 else 0.01, self.qcmd)
                self.L.hold(hold_k(0.5))
                self.when_settled(self.L, "left", lambda: self._lower_z(stage))

            def _lower_z(self, stage):
                p, _ = can_pose()
                rim = [0.150, 0.120, props.RING_Z[1] + HOLD_RIM_ABOVE_RING][stage]
                dz = p[2] - rim
                if abs(dz) > 0.0015 and not getattr(self, "_zfix", False):
                    self._zfix = True
                    self.corr[2] += dz
                    self._can_to(np.array([TX, TY, rim]) - self.corr, 0.01, self.qcmd)
                    self.L.hold(hold_k(0.4))
                    self.when_settled(self.L, "left", lambda: self._lower_z(stage))
                    return
                self._zfix = False
                self._lower(stage + 1)

            # -- 3. swing the plate shut ----------------------------------------------------------
            def _push_target_down(self, theta):
                e = tab_point(theta)
                d = push_dir(theta)
                return e - PUSH_STAND * d + (TIP_PAST - PUSH_TIP_BELOW) * EZ

            def plan_plate(self):
                th = plate_deg()
                R = self.R
                R.grip_to(CLOSED, hold_k(0.2))
                start = self._push_target_down(th) - 0.015 * push_dir(th)
                R.move(start + 0.06 * EZ, hand_down(PUSH_YAW_DEG), min_steps=hold_k(2.0))
                R.move(start, speed=0.03)
                R.hold(hold_k(0.3))
                self.when_settled(R, "right", lambda: self._push(0, PLATE_LEAD0, th))

            def _push(self, k, lead, last):
                th = plate_deg()
                if k % 4 == 0:
                    g = grasp_point("right")[0]
                    log(f"PUSH {k}: plate {th:.1f} deg lead {lead:.0f}, hand to tab (mm) "
                        f"{np.round((g - tab_point(th)) * 1000, 1)}", flush=True)
                self.stall = getattr(self, "stall", 0) + 1 if last - th < 0.3 else 0
                if th < SWITCH_DEG or k > 150 or (th < 75.0 and self.stall >= 6):
                    self._side_start()
                    return
                rate = last - th
                if rate < 1.0:
                    lead = min(PLATE_LEAD_MAX, lead + 2.0)
                elif rate > 4.0:
                    lead = max(PLATE_LEAD0, lead - 2.0)
                self.R.move(self._push_target_down(max(th - lead, SWITCH_DEG - 5.0)), speed=0.04,
                            min_steps=hold_k(0.15))
                self.at(len(self.R.steps), lambda: self._push(k + 1, lead, th))

            def _side_target(self, theta, back=0.0):
                e = tab_point(theta, dz=PUSH_Z)
                d = push_dir(theta)
                tip = e - (props.TAB_HW + 0.003 + back) * d
                return tip - TIP_PAST * SIDE_R_DIR

            def _side_start(self):
                th = plate_deg()
                self.report("PUSH switch")
                R = self.R
                g, _ = grasp_point("right")
                R.move(g + 0.06 * EZ - 0.04 * EY, speed=0.05)
                R.move(self._side_target(th, back=0.03) + 0.04 * EZ, SIDE_R, speed=0.06, min_steps=hold_k(1.5))
                R.move(self._side_target(th, back=0.03), speed=0.03)
                R.hold(hold_k(0.3))
                self.when_settled(R, "right", lambda: self._side_push(0, PLATE_LEAD0, th))

            def _side_push(self, k, lead, last):
                th = plate_deg()
                if k % 4 == 0:
                    log(f"SIDE {k}: plate {th:.1f} deg lead {lead:.0f}", flush=True)
                if th < 1.5 or k > 40:
                    self.R.move(self._side_target(0.0, back=-0.0015), speed=0.01)
                    self.R.hold(hold_k(0.5))
                    self.at(len(self.R.steps), self._plate_done)
                    return
                rate = last - th
                if rate < 0.8:
                    lead = min(PLATE_LEAD_MAX, lead + 2.0)
                elif rate > 4.0:
                    lead = max(PLATE_LEAD0, lead - 2.0)
                self.R.move(self._side_target(max(th - lead, -2.0)), speed=0.03, min_steps=hold_k(0.15))
                self.at(len(self.R.steps), lambda: self._side_push(k + 1, lead, th))

            def _plate_done(self):
                self.report("PLATE shut")
                R, L = self.R, self.L
                g, _ = grasp_point("right")
                R.move(g - 0.05 * EY + 0.02 * EZ, speed=0.04)
                R.move(R_WAIT, QB, speed=0.08)
                self.sync()
                L.grip_to(OPEN, hold_k(0.8))
                g, _ = grasp_point("left")
                L.move(g + 0.07 * EY, speed=0.04)
                L.move(g + 0.07 * EY + 0.06 * EZ, speed=0.05)
                L.hold(hold_k(1.0))
                self.sync()
                self.at(len(L.steps), self._cap_released)

            def _cap_released(self):
                rep = self.report("CAP released")
                p, q = can_pose()
                log(f"CAP at {np.round(p, 4)} quat {np.round(q, 4)}", flush=True)
                if STOP_AFTER == "plate":
                    self.finish()
                    return
                if control:
                    self.L.move(L_WAIT, speed=0.08)
                    self.sync()
                    self.L.hold(hold_k(1.5))
                    self.at(len(self.L.steps), lambda: task.force_function_test(view))
                    self.L.hold(hold_k(6.0))
                    self.sync()
                    return
                self.sync()
                self.plan_pick()

            # -- 4. padlock ------------------------------------------------------------------------
            def lock_pose(self):
                return (lock.data.body_pos_w.torch[0, lock_body].cpu().numpy() - origin,
                        lock.data.body_quat_w.torch[0, lock_body].cpu().numpy())

            def travel(self):
                return float(lock.data.joint_pos.torch[0, shackle_joint])

            def tip(self):
                pos, quat = self.lock_pose()
                return pos + rotate(quat, PV.short_tip(self.travel()))

            def hole(self):
                """The tabs' hole axis at the top of the can's tab, in the world."""
                p, q = can_pose()
                return p + rotate(q, (props.TAB_HOLE_X, 0.0, props.TAB_Z["can"][1]))

            def lock_center_for_tip(self, tip_target):
                return np.asarray(tip_target) - rotate(LOCK_QUAT, PV.short_tip(self.travel()))

            def _lock_to(self, center, speed):
                self.L.move(self.lmodel.grasp_for(center, LOCK_QUAT)[0], speed=speed)

            def plan_pick(self):
                start = np.array(lock_start_pose()[:3])
                grasp, _ = GraspModel.nominal(QA, LOCK_QUAT, GRASP_IN_LOCK).grasp_for(start, LOCK_QUAT)
                L = self.L
                L.grip_to(OPEN, hold_k(0.2))
                L.move(grasp - 0.06 * A_FINGER_DIR + 0.02 * EZ, QA, min_steps=hold_k(3.0))
                L.hold(hold_k(0.4))
                L.move(grasp - 0.012 * A_FINGER_DIR, speed=0.04)
                L.hold(hold_k(0.4))
                self.when_settled(L, "left", self._final_approach)

            def _final_approach(self):
                start = np.array(lock_start_pose()[:3])
                grasp, _ = GraspModel.nominal(QA, LOCK_QUAT, GRASP_IN_LOCK).grasp_for(start, LOCK_QUAT)
                err = grasp_point("left")[0] - (self.L.pos + ROOT_OFFSET)
                self.L.move(grasp - err, speed=0.02)
                self.L.hold(hold_k(0.4))
                self.L.grip_to(CLOSED, hold_k(1.0))
                self.L.move(grasp - err + np.array([0.0, 0.0, 0.03]), speed=0.03)
                self.when_settled(self.L, "left", self.lock_carry)

            def lock_carry(self):
                self.lmodel = GraspModel.measure(robot, "left", lock)
                log("LOCK grasp rel_pos (mm)", np.round(self.lmodel.rel_pos * 1000, 1), flush=True)
                self.tip_entry = self.hole() + TIP_OVER * EZ - 0.025 * EX
                c = self.lock_center_for_tip(self.tip_entry)
                here = self.lock_pose()[0]
                self._lock_to(here + 0.02 * EZ, 0.04)
                self._lock_to(np.array([c[0] - 0.03, c[1], here[2] + 0.02]), 0.06)
                self._lock_to(c - 0.03 * EX, 0.04)
                self._lock_to(c, 0.03)
                self.L.hold(hold_k(0.6))
                self.when_settled(self.L, "left", lambda: self._entry_correct(np.zeros(3), 0))

            def _entry_correct(self, correction, attempt):
                err = self.tip() - self.tip_entry
                log(f"ENTRY attempt {attempt}: tip error (mm) {np.round(err * 1000, 1)}, shackle out "
                    f"{self.travel() * 1000:.1f} mm", flush=True)
                correction = correction + err
                if attempt < 4 and np.abs(err).max() > 0.001:
                    self._lock_to(self.lock_center_for_tip(self.tip_entry - correction), 0.01)
                    self.L.hold(hold_k(0.6))
                    self.when_settled(self.L, "left", lambda: self._entry_correct(correction, attempt + 1))
                    return
                self.correction = correction
                self.tip_under = self.hole() + TIP_OVER * EZ
                self._lock_to(self.lock_center_for_tip(self.tip_under - correction), 0.01)
                self.L.hold(hold_k(0.6))
                self.when_settled(self.L, "left", lambda: self._under_correct(0))

            def _under_correct(self, attempt):
                self.tip_under = self.hole() + TIP_OVER * EZ
                err = self.tip() - self.tip_under
                log(f"OVER hole attempt {attempt}: tip error (mm) {np.round(err * 1000, 1)}, lock rot "
                    f"{np.round(self.lock_pose()[1], 3)}", flush=True)
                self.correction = self.correction + err
                if attempt < 5 and np.abs(err).max() > 0.0008:
                    self._lock_to(self.lock_center_for_tip(self.tip_under - self.correction), 0.008)
                    self.L.hold(hold_k(0.6))
                    self.when_settled(self.L, "left", lambda: self._under_correct(attempt + 1))
                    return
                self._pierce(0, 0.0)

            def _pierce(self, k, dz):
                """Lower the padlock so the short leg goes down through both tabs, in 2 mm steps, re-centring it on the
                holes from the measured tip position each step."""
                tip = self.tip()
                rise = float((tip - self.tip_under) @ EZ)
                err = tip - (self.tip_under + rise * EZ)
                if k % 2 == 0 or k < 4:
                    log(f"PIERCE {k}: tip down {-rise * 1000:.1f} mm, off the hole's axis (mm) {np.round(err * 1000, 1)}",
                        flush=True)
                if dz >= PIERCE or k > 25:
                    self.L.hold(hold_k(0.6))
                    self.when_settled(self.L, "left", self._threaded)
                    return
                self.correction = self.correction + 0.7 * (err - float(err @ EZ) * EZ)
                dz = min(dz + 0.002, PIERCE)
                self._lock_to(self.lock_center_for_tip(self.tip_under - dz * EZ - self.correction), 0.006)
                self.L.hold(hold_k(0.3))
                self.at(len(self.L.steps), lambda: self._pierce(k + 1, dz))

            def _threaded(self):
                log(f"PIERCED: tip {np.round(self.tip(), 4)} hole {np.round(self.hole(), 4)} through "
                    f"{bool(st.through_now[0])}", flush=True)
                if control:
                    self.release()
                    return
                self.press_start()

            def bend_top(self):
                pos, quat = self.lock_pose()
                return pos + rotate(quat, (0.0, 0.0, PV.bend_top(self.travel())))

            def press_start(self):
                R = self.R
                R.pad_to(len(self.L.steps))
                R.grip = CLOSED
                b = self.bend_top()
                start = b + 0.004 * EZ - TIP_PAST * B_FINGER_DIR
                R.move(start + 0.06 * EZ - 0.02 * EX, QB, min_steps=hold_k(2.5))
                R.move(start, speed=0.03)
                R.hold(hold_k(0.6))
                self.body0 = self.lock_pose()[0]
                self.when_settled(R, "right", lambda: self._press(0))

            def _press(self, k):
                latched = bool(st.latched[0])
                if latched or k > 30:
                    log(f"PRESS done after {k}: travel {self.travel() * 1000:.1f} mm, latched {latched}, through "
                        f"{bool(st.latched_through[0])}", flush=True)
                    self.R.hold(hold_k(0.4))
                    self.L.pad_to(len(self.R.steps))
                    self.release()
                    return
                drift = float((self.lock_pose()[0] - self.body0) @ EZ)
                self.L.pad_to(len(self.R.steps))
                self.L.move(self.L.pos + ROOT_OFFSET - drift * EZ, speed=0.02)
                self.R.move(self.R.pos + ROOT_OFFSET - 0.003 * EZ, speed=0.02)
                self.R.hold(hold_k(0.2))
                self.sync()
                self.at(len(self.R.steps), lambda: self._press(k + 1))

            def release(self):
                R, L = self.R, self.L
                R.move(R.pos + ROOT_OFFSET + 0.05 * EZ - 0.02 * EX, speed=0.04)
                R.move(R_WAIT, speed=0.10)
                L.pad_to(len(R.steps) - hold_k(1.2))
                L.grip_to(0.55, hold_k(1.0))
                L.move(L.pos + ROOT_OFFSET - 0.05 * A_FINGER_DIR, speed=0.03)
                L.move(L_WAIT, speed=0.06)
                L.hold(hold_k(1.0))
                self.sync()
                self.at(len(L.steps), lambda: self.report("RELEASED"))
                L.hold(hold_k(4.0))

        l_pos, l_quat = grasp_point("left")
        r_pos, r_quat = grasp_point("right")
        L = ArmTimeline(l_pos - ROOT_OFFSET, l_quat, dt, speed)
        R = ArmTimeline(r_pos - ROOT_OFFSET, r_quat, dt, speed)
        planner = Planner(L, R)
        settle = hold_k(1.0)
        R.hold(settle)
        L.hold(settle)
        start = os.environ.get("CYLVALVE_DEMO_START", "")  # development only: "cap", "plate" or "padlock"
        if start in ("cap", "plate", "padlock"):
            import torch

            z = torch.zeros(1, 1, device=view.device)
            cyl.write_joint_state_to_sim(z, z, joint_ids=[wheel_joint])
            cyl.set_joint_position_target(z, joint_ids=[wheel_joint])
        if start == "padlock":  # development only: the cap already shut on the valve
            import torch

            pose = torch.tensor([[TX, TY, props.RING_Z[1] + props.PLATE_T + props.PLATE_GAP + 0.0005, 0, 0, 0, 1.0]],
                                device=view.device)
            pose[:, :3] += view.scene.env_origins[:1]
            cap.write_root_pose_to_sim(pose)
            cap.write_root_velocity_to_sim(torch.zeros(1, 6, device=view.device))
            z = torch.zeros(1, 1, device=view.device)
            cap.write_joint_state_to_sim(z, z, joint_ids=[plate_joint])
            cap.set_joint_position_target(z, joint_ids=[plate_joint])
        planner.at(settle, {"cap": planner.plan_cap, "padlock": planner.plan_pick}.get(start, planner.plan_turn))

        q_rel = np.array(hb.Q_REL)
        tcp_extra = 0.1347 - 0.134
        q_arm = {s: view.scene[f"{s}_robot"].data.joint_pos.torch[0, :6].cpu().numpy().astype(float) for s in ("left", "right")}
        worst = {"left": 0.0, "right": 0.0}

        def joint_action(side, pos_root, quat, grip):
            pos = pos_root + ROOT_OFFSET + origin
            tcp = pos + rotate(quat, (0.0, 0.0, -tcp_extra))
            rot = qmul(quat, q_rel)
            q, pe, _ = ik.solve(tcp, rot, q_arm[side], base_pos=station.BASES[side], iters=60)
            q_arm[side] = q
            if pe > 0.005 and (pe > worst[side] + 0.002 or state_k[0] % 40 == 0):
                log(f"IK {side} k={state_k[0]} error {pe * 1000:.1f} mm at {np.round(pos - origin, 4)}", flush=True)
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
