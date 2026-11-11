"""The scripted demo, closed loop.

1. Arm B (right), fingers down, closes across the yellow-tagged plug's body in the top outlet and pulls it straight out
   toward the robot in small steps (re-measuring the plug each step), lifts it, turns it so its blades point along
   the lockout's long axis and lowers it into the open lower half until its fingertips are just above the rims, with
   the cord over the end notch; it opens the fingers and the plug drops the last few millimetres into the shell.
2. Arm B presses the cord into the end notch, closes on the open lid's tab and swings the lid up about the hinge,
   pitching the hand with it (the tab across the fingers). The YAM cannot pitch past 90 degrees, so near upright it lets
   go, takes the tab again from above (along the fingers) and carries the lid down until it falls shut on the plug;
   arm B then presses it down.
3. Arm A (left) picks the open padlock off its pedestal (as in the group-hasp demo), slides it so the two tabs enter
   the gap under the shackle's short leg, lines the leg up with the holes from the measured tip position and pushes
   it through both tabs; arm B pushes the shackle's bend shut while arm A holds the body; both let go.

``control=True``: steps 1 and 2 only; once both hands are clear the after-release check is forced (the lid, shut but
not padlocked, must open).

It reads the simulator's state (the plug's, lid's and padlock's poses), like any scripted demo with privileged state;
``--policy plug-lockout-demo`` (pluglock_sim.policy) plays it through Inspect Robots. Grasp-point targets become
yam_arena's 14-D joint action through yam_arena's own IK every control step. Import after Isaac Sim has started.
"""

from __future__ import annotations

import math
import os

import numpy as np

from yam_arena import ik, station

PULL_STEP = 0.003
PULL_LEAD = 0.08
"""Command the plug at most this far ahead of where it is (the soft arm gains turn the lead into pull force)."""
PULL_DONE_S = 0.032
PLACE_Y = 0.008
"""Plug origin (its front face) along the lockout's long axis, box frame: tail 4 mm short of the cord-hole wall."""
FINGER_CLEAR = 0.0012
TIP_BELOW_GRASP = 0.013
"""The fingertips' ends are 13 mm past the grasp point (yam_arena PAD_TIP_PAST_TCP)."""
GRIP_Z = 0.002
"""Grasp height on the plug (plug frame): the pads cover the body's upper part, so the fingertips stay above the
shell's rims when the plug is set in."""
LID_GRASP = (None, None, -0.005)
LID_RELEASE_DEG = 28.0
LID_REGRASP_DEG = 80.0
LID_GRASP2_X = -0.044
"""Second grasp on the tab (lid frame x): its outer part, so the fingertips stay above the lid's front edge."""
LID_PITCH_K = 1.0
LID_PITCH_MAX_DEG = 90.0
LID_GRIP = 0.0
LID_GRIP_SLIP = 0.012
"""Past the vertical the hand cannot pitch further: a lighter pinch (about 4 N per finger) lets the tab turn in the
fingers while the hand carries it on along the arc."""


class Demo:
    """``next_action()`` gives the 14-D joint action for the next control step."""

    def __init__(self, STATION, control_hz: float = 20.0, speed: float = 0.10, control: bool = False, log=print):
        from pluglock_sim import station as hb
        from pluglock_sim.kit import (CLOSED, DOWN, FORWARD, OPEN, ROOT_OFFSET, ArmTimeline, GraspModel, conj, q_axis,
                                      qmul, rotate)
        from pluglock_sim.kit import grasp_point as grasp_point_world
        from pluglock_sim.task import PV, lock_start_pose
        from pluglock_sim.usd import pluglock_props as props

        view, task = STATION.env, STATION.task
        st = task.state(view)
        state_k = [0]
        self.state_k = state_k
        origin = view.scene.env_origins[0].cpu().numpy()
        robot = view.scene["robot"]
        dt = 1.0 / control_hz
        hold_k = lambda s: max(1, round(s / dt))  # noqa: E731
        EX, EY, EZ = np.eye(3)
        IDENT = np.array([0.0, 0.0, 0.0, 1.0])
        YAW90 = q_axis((0, 0, 1), math.pi / 2)
        DOWN_X = qmul(YAW90, DOWN)
        BOX = np.array(props.BOX_CENTER)
        HINGE_W = BOX + np.array(props.HINGE)
        TAB_Y = props.TAB_GAP + props.TAB_T / 2
        lock = view.scene["padlock"]
        lock_body = lock.find_bodies("body")[0][0]
        shackle_joint = lock.find_joints("shackle_joint")[0][0]

        def grasp_point(side):
            p, q = grasp_point_world(robot, side)
            return p - origin, q

        for side, name in (("left", "left_robot"), ("right", "right_robot")):
            qj = view.scene[name].data.joint_pos.torch[0, :6].cpu().numpy()
            tcp, _ = ik.tcp_pose(qj, station.BASES[side])
            log(f"CHECK {side}: their TCP {np.round(tcp, 4)} our grasp point {np.round(grasp_point(side)[0] + origin, 4)}",
                flush=True)

        def plug_pose():
            pos = st.cord_t.data.body_pos_w.torch[0, st.plug_t].cpu().numpy() - origin
            return pos, st.cord_t.data.body_quat_w.torch[0, st.plug_t].cpu().numpy()

        def measure_plug():
            g_pos, g_quat = grasp_point("right")
            o_pos, o_quat = plug_pose()
            return GraspModel(rotate(conj(g_quat), o_pos - g_pos), qmul(conj(g_quat), o_quat))

        def hinge_deg():
            return math.degrees(float(st.box.data.joint_pos.torch[0, st.hinge]))

        def lid_point(p_closed, theta_deg):
            """A point of the lid (given in the box frame with the lid closed) with the lid at theta_deg."""
            t = math.radians(theta_deg)
            d = np.array(p_closed) - np.array(props.HINGE)
            x, z = d[0] * math.cos(t) + d[2] * math.sin(t), -d[0] * math.sin(t) + d[2] * math.cos(t)
            return HINGE_W + np.array([x, d[1], z])

        def lid_hand2(theta_deg):
            """Second grasp: the tab along the fingers, hand pitched with the lid from about upright down to shut."""
            return qmul(q_axis((0, 1, 0), -math.radians(max(0.0, 92.0 - theta_deg))), DOWN)

        def lid_hand(theta_deg):
            """Hand orientation on the lid's tab: fingers down at the start, then pitched with the lid (about the
            fingers' closing axis, so the grip turns with the tab) as far as LID_PITCH_MAX_DEG."""
            pitch = min(LID_PITCH_K * (180.0 - theta_deg), LID_PITCH_MAX_DEG)
            return qmul(q_axis((0, 1, 0), -math.radians(pitch)), DOWN)

        FINGER_END, GRASP_IN_LOCK, FACE_GAP, PIERCE = 0.012, (0.0, 0.0, -0.012), 0.0013, 0.0115
        LOCK_QUAT = np.array(lock_start_pose()[3:])
        # Arm A holds the lying padlock's body near its bottom end, fingers toward -y and 15 deg down (as the padlock lies), closing on
        # the body's top and bottom faces.
        SIDE_A = qmul(q_axis((0, 0, 1), -math.pi / 2), FORWARD)
        PITCH_A = math.radians(15)
        QA = qmul(q_axis((1, 0, 0), PITCH_A), qmul(SIDE_A, q_axis((0, 0, 1), math.pi / 2)))
        A_FINGER_DIR = np.array((0.0, -math.cos(PITCH_A), -math.sin(PITCH_A)))
        YAW_B = math.radians(60)
        QB = qmul(q_axis((-math.sin(YAW_B), math.cos(YAW_B), 0), math.radians(20)), qmul(q_axis((0, 0, 1), YAW_B), FORWARD))
        B_FINGER_DIR = np.array((math.cos(YAW_B) * math.cos(math.radians(20)), math.sin(YAW_B) * math.cos(math.radians(20)),
                                 -math.sin(math.radians(20))))
        R_WAIT = np.array((0.16, -0.16, 0.28))
        L_WAIT = np.array((0.16, 0.22, 0.28))

        class Planner:
            def __init__(self, L, R):
                self.L, self.R, self.events = L, R, {}

            def at(self, step, fn):
                self.events.setdefault(step, []).append(fn)

            def run_due(self, k):
                for fn in self.events.pop(k, []):
                    fn()

            def when_settled(self, tl, side, fn, tries=30, tol=0.003):
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
                log(f"{tag}: milestone {rep['milestone']} target out {rep['target_out_mm']} mm, decoy max "
                    f"{rep['decoy_out_max_mm']} mm, hinge {rep['hinge_deg']} deg, events {rep['semantic_events']}",
                    flush=True)
                return rep

            # -- 1. unplug -------------------------------------------------------------------------
            def plan_unplug(self):
                pos, q = plug_pose()
                self.grasp = pos + rotate(q, (props.GRIP_X, 0.0, GRIP_Z))
                R = self.R
                R.grip_to(OPEN, hold_k(0.2))
                R.move(self.grasp + np.array([-0.03, 0.0, 0.03]), DOWN, min_steps=hold_k(2.0))
                R.move(self.grasp + np.array([0.0, 0.0, 0.022]), min_steps=hold_k(1.0))
                R.hold(hold_k(0.3))
                self.when_settled(R, "right", self._unplug_descend)

            def _unplug_descend(self):
                err = grasp_point("right")[0] - (self.R.pos + ROOT_OFFSET)
                self.R.move(self.grasp - err * np.array([1.0, 1.0, 0.0]) + np.array([0.0, 0.0, 0.015]), speed=0.03)
                self.R.move(self.grasp - err * np.array([1.0, 1.0, 0.0]), speed=0.015)
                self.R.hold(hold_k(0.4))
                self.when_settled(self.R, "right", self._unplug_close)

            def _unplug_close(self):
                g = grasp_point("right")[0]
                log(f"GRASP plug: hand {np.round(g, 4)} planned {np.round(self.grasp, 4)} lag (mm) "
                    f"{np.round((g - self.R.pos - ROOT_OFFSET) * 1000, 1)}", flush=True)
                self.R.grip_to(CLOSED, hold_k(1.0))
                self.at(len(self.R.steps), self._pull_start)

            def _plug_to(self, pos, quat, speed):
                g, q = self.model.grasp_for(pos, quat)
                self.R.move(g, q, speed=speed)

            def _pull_start(self):
                self.model = measure_plug()
                log(f"PLUG grasp rel_pos (mm) {np.round(self.model.rel_pos * 1000, 1)}", flush=True)
                self.plug0, self.plug_q0 = plug_pose()
                self.pull_cmd = 0.0
                self.q_corr = IDENT.copy()
                self._pull(0)

            def _pull(self, k):
                s = float(st.s_t[0])
                if k % 3 == 0:
                    self.report(f"PULL {k} s {s * 1000:.1f} mm")
                    lag = grasp_point("right")[0] - (self.R.pos + ROOT_OFFSET)
                    m = measure_plug()
                    p, q = plug_pose()
                    ex = rotate(q, (1.0, 0.0, 0.0))
                    log(f"   hand lag (mm) {np.round(lag * 1000, 1)} plug in hand (mm) {np.round(m.rel_pos * 1000, 1)} "
                        f"plug axis {np.round(ex, 3)}", flush=True)
                if s > PULL_DONE_S or k > 60:
                    self.report("UNPLUGGED")
                    self.carry()
                    return
                self.pull_cmd = min(max(self.pull_cmd + PULL_STEP, s + 0.004), s + PULL_LEAD)
                # The soft wrist lets the plug pitch under the pull (its blades then bind in the slots): turn the
                # command against the measured tilt.
                _, q = plug_pose()
                err = qmul(q, conj(self.plug_q0))
                self.q_corr = qmul(conj(err), self.q_corr)
                self.q_corr /= np.linalg.norm(self.q_corr)
                self._plug_to(self.plug0 - self.pull_cmd * EX, qmul(self.q_corr, self.plug_q0), 0.02)
                self.R.hold(hold_k(0.15))
                self.at(len(self.R.steps), lambda: self._pull(k + 1))

            # -- set the plug in the lockout -------------------------------------------------------
            def place_target(self, dz=0.0):
                """Plug origin in the world for the drop: blades along +y, fingertips FINGER_CLEAR above the rims."""
                z = props.RIM_Z + FINGER_CLEAR + TIP_BELOW_GRASP - GRIP_Z
                return np.array([BOX[0], BOX[1] + PLACE_Y, z + dz])

            def carry(self):
                self.model = measure_plug()
                p, q = plug_pose()
                self.plug_q = qmul(YAW90, self.plug_q0)
                self._plug_to(p + np.array([-0.03, 0.0, 0.035]), self.plug_q0, 0.04)
                self._plug_to(self.place_target(0.05) + np.array([0.0, -0.04, 0.0]), self.plug_q, 0.06)
                self._plug_to(self.place_target(0.03), self.plug_q, 0.04)
                self.R.hold(hold_k(0.5))
                self.place_corr = np.zeros(3)
                self.when_settled(self.R, "right", lambda: self._place(0))

            def _place(self, attempt):
                p, q = plug_pose()
                goal = self.place_target(0.03 if attempt < 3 else 0.0)
                err = p - goal
                log(f"PLACE {attempt}: plug error (mm) {np.round(err * 1000, 1)}", flush=True)
                if attempt < 3:
                    # Correct the measured offset over the lockout (the arm sags under the cord's pull).
                    self.place_corr = self.place_corr + err
                    self._plug_to(goal - self.place_corr, self.plug_q, 0.02)
                    self.R.hold(hold_k(0.4))
                    self.when_settled(self.R, "right", lambda: self._place(attempt + 1))
                    return
                if attempt == 3:
                    self._plug_to(self.place_target(0.0) - self.place_corr, self.plug_q, 0.012)
                    self.R.hold(hold_k(0.5))
                    self.when_settled(self.R, "right", lambda: self._place(attempt + 1))
                    return
                self.R.grip_to(OPEN, hold_k(0.8))
                self.R.move(self.R.pos + ROOT_OFFSET + np.array([0.0, 0.0, 0.025]), speed=0.03)
                self.R.hold(hold_k(0.5))
                self.at(len(self.R.steps), self.seat_cord)

            def seat_cord(self):
                """Press the cord down into the end notch just outside the shell, as a hand settles it before closing."""
                R = self.R
                p = BOX + np.array([0.0, -props.BOX_HL - 0.010, TIP_BELOW_GRASP + 0.001])
                R.grip_to(CLOSED, hold_k(0.4))
                R.move(p + np.array([0.0, 0.0, 0.03]), DOWN_X, speed=0.05)
                R.move(p, speed=0.02)
                R.hold(hold_k(0.4))
                R.move(p + np.array([0.0, 0.0, 0.035]), speed=0.04)
                R.grip_to(OPEN, hold_k(0.3))
                self.at(len(R.steps), self.lid_start)

            # -- 2. close the lid -------------------------------------------------------------------
            def lid_start(self):
                self.report("PLACED")
                th = hinge_deg()
                self.lid_g = (props.TAB_HOLE_X, TAB_Y, LID_GRASP[2])
                g = lid_point(self.lid_g, th)
                R = self.R
                R.move(np.array([0.5 * (R.pos[0] + g[0]), g[1] - 0.01, 0.19]), DOWN, speed=0.05)
                R.move(g + np.array([0.0, 0.0, 0.012]), DOWN, speed=0.04)
                R.hold(hold_k(0.3))
                self.when_settled(R, "right", self._lid_descend)

            def _lid_descend(self):
                g = lid_point(self.lid_g, hinge_deg())
                err = grasp_point("right")[0] - (self.R.pos + ROOT_OFFSET)
                self.lid_err = err * np.array([1.0, 1.0, 0.0]) if np.linalg.norm(err) < 0.006 else np.zeros(3)
                log(f"LID grasp: hand lag (mm) {np.round(err * 1000, 1)}", flush=True)
                self.R.move(g - self.lid_err, speed=0.015)
                self.R.hold(hold_k(0.4))
                self.R.grip_to(LID_GRIP, hold_k(0.8))
                self.at(len(self.R.steps), lambda: self._lid_swing(0))

            def _lid_swing(self, k):
                """First grasp: the tab across the fingers, the hand pitching with the lid from flat (180 deg) to
                about upright (it cannot pitch past 90 deg), on to LID_REGRASP_DEG with the tab slipping a little."""
                th = hinge_deg()
                g, gq = grasp_point("right")
                tab = rotate(conj(gq), lid_point(self.lid_g, th) - g)
                log(f"LID {k}: hinge {th:.1f} deg, tab grasp point in hand (mm) {np.round(tab * 1000, 1)}", flush=True)
                if k == 0:
                    for target in np.arange(th - 3.0, LID_REGRASP_DEG - 4.0, -3.0):
                        if target < 180.0 - LID_PITCH_MAX_DEG and self.R.grip != LID_GRIP_SLIP:
                            self.R.grip_to(LID_GRIP_SLIP, 2)
                        self.R.move(lid_point(self.lid_g, target) - self.lid_err, lid_hand(target), speed=0.03,
                                    min_steps=2)
                    self.R.hold(hold_k(0.3))
                    self.at(len(self.R.steps), lambda: self._lid_swing(1))
                    return
                # Let go and take the tab again from above, along the fingers this time (the hand then turns the
                # other way and can carry the lid the rest of the way down).
                self.R.grip_to(OPEN, hold_k(0.4))
                self.R.move(g + 0.03 * rotate(gq, (0.0, 0.0, 1.0)) + np.array([0.0, 0.0, 0.02]), speed=0.04)
                self.at(len(self.R.steps), self._lid_regrasp)

            def _lid_regrasp(self):
                th = hinge_deg()
                log(f"LID regrasp at {th:.1f} deg", flush=True)
                self.lid_g2 = (LID_GRASP2_X, TAB_Y, 0.0)
                g = lid_point(self.lid_g2, th)
                R = self.R
                R.move(g + np.array([0.0, 0.0, 0.012]), lid_hand2(th), speed=0.06)
                R.hold(hold_k(0.2))
                self.at(len(R.steps), self._lid_regrasp_close)

            def _lid_regrasp_close(self):
                th = hinge_deg()
                self.R.move(lid_point(self.lid_g2, th), lid_hand2(th), speed=0.015)
                self.R.grip_to(CLOSED, hold_k(0.6))
                self.at(len(self.R.steps), lambda: self._lid_lower(0))

            def _lid_lower(self, k):
                th = hinge_deg()
                g, gq = grasp_point("right")
                tab = rotate(conj(gq), lid_point(self.lid_g2, th) - g)
                log(f"LID lower {k}: hinge {th:.1f} deg, tab grasp point in hand (mm) {np.round(tab * 1000, 1)}",
                    flush=True)
                if th < LID_RELEASE_DEG + 3.0 or k > 3:
                    self.R.grip_to(OPEN, hold_k(0.5))
                    self.R.move(g + 0.03 * rotate(gq, (0.0, 0.0, 1.0)) + np.array([0.0, 0.0, 0.02]), speed=0.04)
                    self.R.hold(hold_k(0.8))
                    self.at(len(self.R.steps), self._lid_press)
                    return
                for target in np.arange(th - 3.0, LID_RELEASE_DEG - 3.0, -3.0):
                    self.R.move(lid_point(self.lid_g2, target), lid_hand2(target), speed=0.03, min_steps=2)
                self.R.hold(hold_k(0.3))
                self.at(len(self.R.steps), lambda: self._lid_lower(k + 1))

            def _lid_press(self):
                """Press the shut lid down on its rim with the closed fingertips (it may rest a few degrees open on the
                cord in the notch)."""
                self.report("LID dropped")
                p = BOX + np.array([-0.008, -0.02, props.HALF_D + TIP_BELOW_GRASP])
                R = self.R
                R.grip_to(CLOSED, hold_k(0.3))
                R.move(p + np.array([0.0, 0.0, 0.03]), DOWN, speed=0.05)
                R.move(p + np.array([0.0, 0.0, 0.002]), speed=0.02)
                R.move(p - np.array([0.0, 0.0, 0.006]), speed=0.01)
                R.hold(hold_k(0.5))
                R.move(p + np.array([0.0, 0.0, 0.05]), speed=0.04)
                R.hold(hold_k(0.3))
                self.at(len(R.steps), self._lid_done)

            def _lid_done(self):
                rep = self.report("LID closed")
                R = self.R
                R.move(R_WAIT, QB, speed=0.08)
                if control:
                    R.hold(hold_k(1.5))
                    self.at(len(R.steps), lambda: task.force_function_test(view))
                    R.hold(hold_k(4.0))
                    return
                self.sync()
                self.plan_pick()

            # -- 3. padlock ------------------------------------------------------------------------
            def lock_pose(self):
                return (lock.data.body_pos_w.torch[0, lock_body].cpu().numpy() - origin,
                        lock.data.body_quat_w.torch[0, lock_body].cpu().numpy())

            def travel(self):
                return float(lock.data.joint_pos.torch[0, shackle_joint])

            def tip(self):
                pos, quat = self.lock_pose()
                return pos + rotate(quat, PV.short_tip(self.travel()))

            def hole(self):
                return BOX + np.array([props.TAB_HOLE_X, 0.0, 0.0])

            def face_point(self, hole):
                return hole - (props.TAB_GAP + props.TAB_T + FACE_GAP) * EY

            def lock_center_for_tip(self, tip_target):
                return np.asarray(tip_target) - rotate(LOCK_QUAT, PV.short_tip(self.travel()))

            def _lock_to(self, center, speed):
                self.L.move(self.lmodel.grasp_for(center, LOCK_QUAT)[0], speed=speed)

            def plan_pick(self):
                start = np.array(lock_start_pose()[:3])
                grasp, _ = GraspModel.nominal(QA, LOCK_QUAT, GRASP_IN_LOCK).grasp_for(start, LOCK_QUAT)
                L = self.L
                L.grip_to(OPEN, hold_k(0.2))
                L.move(grasp - 0.07 * A_FINGER_DIR + 0.03 * EZ, QA, min_steps=hold_k(3.0))
                L.hold(hold_k(0.4))
                L.move(grasp - 0.015 * A_FINGER_DIR, speed=0.04)
                L.hold(hold_k(0.4))
                self.when_settled(L, "left", self._final_approach)

            def _final_approach(self):
                start = np.array(lock_start_pose()[:3])
                grasp, _ = GraspModel.nominal(QA, LOCK_QUAT, GRASP_IN_LOCK).grasp_for(start, LOCK_QUAT)
                err = grasp_point("left")[0] - (self.L.pos + ROOT_OFFSET)
                self.L.move(grasp - err, speed=0.02)
                self.L.hold(hold_k(0.4))
                self.L.grip_to(CLOSED, hold_k(1.0))
                self.L.move(grasp - err + np.array([0.0, 0.0, 0.035]), speed=0.03)
                self.when_settled(self.L, "left", self.lock_carry)

            def lock_carry(self):
                self.lmodel = GraspModel.measure(robot, "left", lock)
                log("LOCK grasp rel_pos (mm)", np.round(self.lmodel.rel_pos * 1000, 1), flush=True)
                self.tip_entry = self.face_point(self.hole()) - 0.022 * EX
                c = self.lock_center_for_tip(self.tip_entry)
                here = self.lock_pose()[0]
                self._lock_to(here + 0.03 * EZ, 0.04)
                self._lock_to(c + 0.03 * EZ + 0.02 * EY, 0.06)
                self._lock_to(c, 0.03)
                self.L.hold(hold_k(0.6))
                self.when_settled(self.L, "left", lambda: self._entry_correct(np.zeros(3), 0))

            def _entry_correct(self, correction, attempt):
                err = self.tip() - self.tip_entry
                bp, bq = self.lock_pose()
                log(f"ENTRY attempt {attempt}: tip error (mm) {np.round(err * 1000, 1)}, shackle out "
                    f"{self.travel() * 1000:.1f} mm, body in box frame (mm) {np.round((bp - BOX) * 1000, 1)}",
                    flush=True)
                correction = correction + err
                if attempt < 4 and np.abs(err).max() > 0.001:
                    self._lock_to(self.lock_center_for_tip(self.tip_entry - correction), 0.01)
                    self.L.hold(hold_k(0.6))
                    self.when_settled(self.L, "left", lambda: self._entry_correct(correction, attempt + 1))
                    return
                self.correction = correction
                self.tip_over = self.face_point(self.hole())
                self._lock_to(self.lock_center_for_tip(self.tip_over - correction), 0.012)
                self.L.hold(hold_k(0.6))
                self.when_settled(self.L, "left", lambda: self._over_correct(0))

            def _over_correct(self, attempt):
                err = self.tip() - self.tip_over
                log(f"OVER hole attempt {attempt}: tip error (mm) {np.round(err * 1000, 1)}, lock rot vs plan "
                    f"{np.round(qmul(self.lock_pose()[1], conj(LOCK_QUAT)), 3)}", flush=True)
                self.correction = self.correction + err
                if attempt < 5 and np.abs(err).max() > 0.0008:
                    self._lock_to(self.lock_center_for_tip(self.tip_over - self.correction), 0.008)
                    self.L.hold(hold_k(0.6))
                    self.when_settled(self.L, "left", lambda: self._over_correct(attempt + 1))
                    return
                self._pierce(0, 0.0)

            def _pierce(self, k, dy):
                """Push the short leg through both tabs in 2 mm steps, re-centring it on the holes from the measured tip
                each step (the arm sags as it reaches across)."""
                tip = self.tip()
                err = tip - (self.tip_over + float((tip - self.tip_over) @ EY) * EY)
                if k % 2 == 0 or k < 4:
                    bq = self.lock_pose()[1]
                    rel = qmul(bq, conj(LOCK_QUAT))
                    g, gq = grasp_point("left")
                    hrel = qmul(gq, conj(QA))
                    log(f"PIERCE {k}: tip past start {float((tip - self.tip_over) @ EY) * 1000:.1f} mm, off the hole's "
                        f"axis (mm) {np.round(err * 1000, 1)}, lock rot vs plan {np.round(rel, 3)}, hand rot vs QA "
                        f"{np.round(hrel, 3)}, hand lag (mm) {np.round((g - self.L.pos - ROOT_OFFSET) * 1000, 1)}",
                        flush=True)
                if dy >= PIERCE or k > 25:
                    self.L.hold(hold_k(0.6))
                    self.when_settled(self.L, "left", self._threaded)
                    return
                self.correction = self.correction + 0.7 * (err - float(err @ EY) * EY)
                dy = min(dy + 0.002, PIERCE)
                self._lock_to(self.lock_center_for_tip(self.tip_over + dy * EY - self.correction), 0.006)
                self.L.hold(hold_k(0.3))
                self.at(len(self.L.steps), lambda: self._pierce(k + 1, dy))

            def _threaded(self):
                log(f"PIERCED: tip {np.round(self.tip(), 4)} hole {np.round(self.hole(), 4)}", flush=True)
                self.press_start()

            def bend_outer(self):
                pos, quat = self.lock_pose()
                return pos + rotate(quat, (0.0, 0.0, PV.bend_top(self.travel())))

            def press_start(self):
                R = self.R
                R.pad_to(len(self.L.steps))
                R.grip = CLOSED
                b = self.bend_outer()
                start = b - 0.012 * EY - FINGER_END * B_FINGER_DIR
                R.move(start - 0.06 * B_FINGER_DIR, QB, min_steps=hold_k(2.5))
                R.move(start, speed=0.04)
                R.hold(hold_k(0.6))
                self.body0 = self.lock_pose()[0]
                self.when_settled(R, "right", lambda: self._press(0))

            def _press(self, k):
                latched = bool(st.latched[0])
                if latched or k > 30:
                    log(f"PRESS done after {k}: travel {self.travel() * 1000:.1f} mm, latched {latched}, through "
                        f"{bool(st.latched_through[0])}", flush=True)
                    self.R.move(self.R.pos + ROOT_OFFSET + 0.004 * EY, speed=0.02)
                    self.R.hold(hold_k(0.4))
                    self.L.pad_to(len(self.R.steps))
                    self.release()
                    return
                drift = float((self.lock_pose()[0] - self.body0) @ EY)
                self.L.pad_to(len(self.R.steps))
                self.L.move(self.L.pos + ROOT_OFFSET - drift * EY, speed=0.02)
                self.R.move(self.R.pos + ROOT_OFFSET + 0.004 * EY, speed=0.02)
                self.R.hold(hold_k(0.2))
                self.sync()
                self.at(len(self.R.steps), lambda: self._press(k + 1))

            def release(self):
                R, L = self.R, self.L
                R.move(R.pos + ROOT_OFFSET - 0.04 * EY, speed=0.04)
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
        start = os.environ.get("PLUGLOCK_DEMO_START", "")  # development only: "lid" or "padlock"
        if start == "padlock":  # development only: start with the lid shut
            import torch

            z = torch.zeros(1, 1, device=view.device)
            st.box.write_joint_state_to_sim(z, z, joint_ids=[st.hinge])
            st.box.set_joint_position_target(z, joint_ids=[st.hinge])
        planner.at(settle, {"lid": planner.lid_start, "padlock": planner.plan_pick}.get(start, planner.plan_unplug))

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
