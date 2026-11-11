"""The scripted demo, closed loop.

1. Disconnect, both arms: arm A (left) closes across the yellow-tagged coupler's sleeve from above (fingers along x,
   wrist leaning toward its own side) and pulls the sleeve back toward the manifold to its stop; arm B (right) closes
   across the plug's hex and ferrule just in front of it and draws the plug straight out in small steps (re-measuring
   it each step). Arm A lets go (the sleeve springs forward) and moves away.
2. Arm B carries the plug over the lockout, tips it nipple down by 25 degrees (its hand upright) and lowers it into the lower half with
   its ferrule just inside the end wall at the height of the half hole, so the hose leaves it straight through the
   hole; it lets go and presses the hose down into the half hole outside the wall.
3. Arm A takes the open lid's tab (hand pitched forward, fingers on the tab's faces) and swings the lid toward the
   robot, pitching the hand with it, to about 60 degrees, lets go (the lid falls shut under its own weight) and
   presses it down.
4. Arm A picks the open padlock off its pedestal, slides it so the two tabs enter the gap under the shackle's short
   leg, lines the leg up with the holes from the measured tip position and pushes it through both tabs; arm B pushes
   the shackle's bend shut while arm A holds the body; both let go.

``control=True``: steps 1 to 3 only; once both hands are clear the after-release check is forced (the lid, shut but
not padlocked, must open).

It reads the simulator's state (the plug's, sleeve's, lid's and padlock's poses), like any scripted demo with
privileged state; ``--policy air-line-lockout-demo`` (airlock_sim.policy) plays it through Inspect Robots. Grasp-point
targets become yam_arena's 14-D joint action through yam_arena's own IK every control step. Import after Isaac Sim
has started.
"""

from __future__ import annotations

import math
import os

import numpy as np

from yam_arena import ik, station

PULL_STEP = 0.003
PULL_LEAD = 0.05
"""Command the plug at most this far ahead of where it is (the soft arm gains turn the lead into pull force)."""
PULL_DONE_M = 0.045
SLEEVE_GRASP_X = 0.019
"""Arm A's grasp on the sleeve (coupler frame x, from the mouth): its back half."""
PLUG_GRASP = (-0.036, 0.0, 0.004)
"""Arm B's grasp on the plug (plug frame): across the hex's back, the stem and the ferrule's front, 4 mm above the
axis (so the fingertips stay above the lockout's floor when the plug is set in)."""
SLEEVE_PULL = 0.016
"""Arm A's command beyond the sleeve's start (it stops at 6 mm; the rest becomes force)."""
TILT_DEG = 25.0
PLACE_TILT_DEG = 25.0
TAIL_GAP = 0.004
"""The ferrule's end this far inside the hose-hole wall when the plug is let go."""
"""The ferrule's end this far from the hose-hole wall inside the lockout."""
FINGER_CLEAR = 0.002
TIP_PAST_GRASP = 0.013
"""The fingertips' ends are 13 mm past the grasp point (yam_arena PAD_TIP_PAST_TCP)."""
LID_GRASP_R = 0.140
"""Tab grasp: this far from the hinge along the lid (the tab's middle, around its hole)."""
LID_PITCH0_DEG = 40.0
LID_RELEASE_DEG = 48.0
LID_PITCH_FOLLOW = 0.5
"""The hand pitches back by half the lid's turn (the arm cannot reach the lower tab positions with the hand straight
down); the tab turns in the lighter pinch for the rest."""
LID_GRIP_SLIP = 0.012


class Demo:
    """``next_action()`` gives the 14-D joint action for the next control step."""

    def __init__(self, STATION, control_hz: float = 20.0, speed: float = 0.10, control: bool = False, log=print):
        from airlock_sim import station as hb
        from airlock_sim.kit import (CLOSED, DOWN, FORWARD, OPEN, ROOT_OFFSET, ArmTimeline, GraspModel, conj, q_axis,
                                      qmul, rotate)
        from airlock_sim.kit import grasp_point as grasp_point_world
        from airlock_sim.task import PV, lock_start_pose
        from airlock_sim.usd import airlock_props as props

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
        DOWN_X = qmul(q_axis((0, 0, 1), math.pi / 2), DOWN)
        DOWN_XM = qmul(q_axis((0, 0, 1), -math.pi / 2), DOWN)
        QL_SLEEVE = qmul(q_axis((1, 0, 0), -math.radians(TILT_DEG)), DOWN_XM)
        QR_PLUG = qmul(q_axis((1, 0, 0), math.radians(TILT_DEG)), DOWN_X)
        BOX = np.array(props.BOX_CENTER)
        HINGE_W = BOX + np.array(props.HINGE)
        TAB_Y = props.TAB_GAP + props.TAB_T / 2
        C_T = props.coupler_root("target")
        lock = view.scene["padlock"]
        lock_body = lock.find_bodies("body")[0][0]
        shackle_joint = lock.find_joints("shackle_joint")[0][0]

        def grasp_point(side):
            p, q = grasp_point_world(robot, side)
            return p - origin, q

        def fdir(q):
            return rotate(q, (0.0, 0.0, -1.0))

        for side, name in (("left", "left_robot"), ("right", "right_robot")):
            qj = view.scene[name].data.joint_pos.torch[0, :6].cpu().numpy()
            tcp, _ = ik.tcp_pose(qj, station.BASES[side])
            log(f"CHECK {side}: their TCP {np.round(tcp, 4)} our grasp point {np.round(grasp_point(side)[0] + origin, 4)}",
                flush=True)

        def plug_pose():
            pos = st.hose_t.data.body_pos_w.torch[0, st.plug_t].cpu().numpy() - origin
            return pos, st.hose_t.data.body_quat_w.torch[0, st.plug_t].cpu().numpy()

        def measure_plug():
            g_pos, g_quat = grasp_point("right")
            o_pos, o_quat = plug_pose()
            return GraspModel(rotate(conj(g_quat), o_pos - g_pos), qmul(conj(g_quat), o_quat))

        def sleeve_mm():
            return float(st.manifold.data.joint_pos.torch[0, st.sleeve_t]) * 1000

        def hinge_deg():
            return math.degrees(float(st.box.data.joint_pos.torch[0, st.hinge]))

        def lid_point(p_closed, theta_deg):
            """A point of the lid (given in the box frame with the lid closed) with the lid at theta_deg."""
            t = math.radians(theta_deg)
            d = np.array(p_closed) - np.array(props.HINGE)
            x, z = d[0] * math.cos(t) + d[2] * math.sin(t), -d[0] * math.sin(t) + d[2] * math.cos(t)
            return HINGE_W + np.array([x, d[1], z])

        def lid_hand(theta_deg):
            """Arm A on the lid's tab: fingers forward and down (pitched LID_PITCH0_DEG from straight down at the
            start), pitching back toward straight down as the lid comes over (the tab turns with the fingers)."""
            pitch = max(0.0, LID_PITCH0_DEG - LID_PITCH_FOLLOW * (props.LID_OPEN_DEG - theta_deg))
            return qmul(q_axis((0, 1, 0), -math.radians(pitch)), DOWN)

        FINGER_END, GRASP_IN_LOCK, FACE_GAP, PIERCE = 0.012, (0.0, 0.0, -0.012), 0.0013, 0.0115
        LOCK_QUAT = np.array(lock_start_pose()[3:])
        SIDE_A = qmul(q_axis((0, 0, 1), -math.pi / 2), FORWARD)
        PITCH_A = math.radians(15)
        QA = qmul(q_axis((1, 0, 0), PITCH_A), qmul(SIDE_A, q_axis((0, 0, 1), math.pi / 2)))
        A_FINGER_DIR = np.array((0.0, -math.cos(PITCH_A), -math.sin(PITCH_A)))
        YAW_B = math.radians(60)
        QB = qmul(q_axis((-math.sin(YAW_B), math.cos(YAW_B), 0), math.radians(20)), qmul(q_axis((0, 0, 1), YAW_B), FORWARD))
        B_FINGER_DIR = np.array((math.cos(YAW_B) * math.cos(math.radians(20)), math.sin(YAW_B) * math.cos(math.radians(20)),
                                 -math.sin(math.radians(20))))
        R_WAIT = np.array((0.14, -0.22, 0.22))
        L_WAIT = np.array((0.14, 0.22, 0.20))
        L_DETOUR = np.array((0.21, -0.075, 0.20))

        class Planner:
            def __init__(self, L, R):
                self.L, self.R, self.events = L, R, {}

            k_now = -1

            def at(self, step, fn):
                step = max(step, self.k_now + 1)
                self.events.setdefault(step, []).append(fn)

            def run_due(self, k):
                self.k_now = k
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
                    f"{rep['decoy_out_max_mm']} mm, sleeve {sleeve_mm():.1f} mm (decoy max {rep['decoy_sleeve_max_mm']}), "
                    f"hinge {rep['hinge_deg']} deg, events {rep['semantic_events']}", flush=True)
                return rep

            # -- 1. disconnect ----------------------------------------------------------------------
            def plan_disconnect(self):
                L, R = self.L, self.R
                self.sleeve_g = C_T + SLEEVE_GRASP_X * EY
                pos, q = plug_pose()
                self.plug_g = pos + rotate(q, PLUG_GRASP)
                # Arm A passes in front of the lockout's open lid, then right of the box, on its way to the sleeve.
                L.grip_to(OPEN, hold_k(0.2))
                L.move(L_DETOUR, QL_SLEEVE, min_steps=hold_k(2.0))
                for tl, g, quat in ((L, self.sleeve_g, QL_SLEEVE), (R, self.plug_g, QR_PLUG)):
                    tl.grip_to(OPEN, hold_k(0.2))
                    tl.move(g - 0.06 * fdir(quat) + 0.03 * EZ, quat, min_steps=hold_k(2.5))
                    tl.move(g - 0.025 * fdir(quat), min_steps=hold_k(1.0))
                    tl.hold(hold_k(0.3))
                self.sync()
                self.when_settled(R, "right", lambda: self.when_settled(L, "left", self._grip_both))

            def _grip_both(self):
                L, R = self.L, self.R
                for tl, side, g in ((L, "left", self.sleeve_g), (R, "right", self.plug_g)):
                    err = grasp_point(side)[0] - (tl.pos + ROOT_OFFSET)
                    err = err if np.linalg.norm(err) < 0.008 else np.zeros(3)
                    tl.move(g - err, speed=0.02)
                    tl.hold(hold_k(0.4))
                self.sync()
                for tl in (L, R):
                    tl.grip_to(CLOSED, hold_k(1.0))
                self.sync()
                self.at(len(L.steps), self._retract)

            def _retract(self):
                gl, gr = grasp_point("left")[0], grasp_point("right")[0]
                log(f"GRASP sleeve hand {np.round(gl, 4)} planned {np.round(self.sleeve_g, 4)}; plug hand "
                    f"{np.round(gr, 4)} planned {np.round(self.plug_g, 4)}", flush=True)
                self.model = measure_plug()
                log(f"PLUG grasp rel_pos (mm) {np.round(self.model.rel_pos * 1000, 1)}", flush=True)
                self.L.move(self.L.pos + ROOT_OFFSET + SLEEVE_PULL * EY, speed=0.01)
                self.L.hold(hold_k(0.6))
                self.R.pad_to(len(self.L.steps))
                self.at(len(self.L.steps), lambda: self._retract_check(0))

            def _retract_check(self, k):
                s = sleeve_mm()
                log(f"SLEEVE {k}: back {s:.2f} mm, left hand lag (mm) "
                    f"{np.round((grasp_point('left')[0] - self.L.pos - ROOT_OFFSET) * 1000, 1)}", flush=True)
                if s < props.SLEEVE_RELEASE * 1000 + 0.6 and k < 3:
                    self.L.move(self.L.pos + ROOT_OFFSET + 0.004 * EY, speed=0.01)
                    self.L.hold(hold_k(0.5))
                    self.R.pad_to(len(self.L.steps))
                    self.at(len(self.L.steps), lambda: self._retract_check(k + 1))
                    return
                self._pull_start()

            def _plug_to(self, pos, quat, speed):
                g, q = self.model.grasp_for(pos, quat)
                self.R.move(g, q, speed=speed)

            def _pull_start(self):
                self.model = measure_plug()
                self.plug0, self.plug_q0 = plug_pose()
                self.pull_cmd = 0.0
                self.q_corr = IDENT.copy()
                self._pull(0)

            def _pull(self, k):
                s = float(st.s_t[0])
                if k % 3 == 0:
                    self.report(f"PULL {k} out {s * 1000:.1f} mm")
                    lag = grasp_point("right")[0] - (self.R.pos + ROOT_OFFSET)
                    m = measure_plug()
                    log(f"   hand lag (mm) {np.round(lag * 1000, 1)} plug in hand (mm) {np.round(m.rel_pos * 1000, 1)} "
                        f"balls (mm) {np.round(st.manifold.data.joint_pos.torch[0, st.ball_joints].cpu().numpy() * 1000, 2)}",
                        flush=True)
                if s > PULL_DONE_M or k > 60:
                    self.report("DISCONNECTED")
                    self._release_sleeve()
                    return
                self.pull_cmd = min(max(self.pull_cmd + PULL_STEP, s + 0.004), s + PULL_LEAD)
                _, q = plug_pose()
                err = qmul(q, conj(self.plug_q0))
                self.q_corr = qmul(conj(err), self.q_corr)
                self.q_corr /= np.linalg.norm(self.q_corr)
                self._plug_to(self.plug0 - self.pull_cmd * EY, qmul(self.q_corr, self.plug_q0), 0.02)
                self.R.hold(hold_k(0.15))
                self.L.pad_to(len(self.R.steps))
                self.at(len(self.R.steps), lambda: self._pull(k + 1))

            def _release_sleeve(self):
                L = self.L
                L.pad_to(len(self.R.steps))
                L.grip_to(OPEN, hold_k(0.5))
                L.move(L.pos + ROOT_OFFSET - 0.05 * fdir(QL_SLEEVE) + 0.04 * EZ, speed=0.05)
                L.move(L_DETOUR, speed=0.08)
                L.move(L_WAIT, DOWN, speed=0.10)
                self.carry()

            # -- 2. set the plug in the lockout -----------------------------------------------------
            def place_target(self, dz=0.0):
                """Plug origin for the drop (plug at self.plug_q, nipple down PLACE_TILT_DEG): on the box's centre line,
                its tail (the ferrule's end) just inside the hose-hole wall at the height of the hole, so the hose
                leaves it straight through the half hole."""
                d = rotate(self.plug_q, (-1.0, 0.0, 0.0))
                tail = np.array([BOX[0], BOX[1] - props.BOX_HL + props.SHELL_T + TAIL_GAP, props.RIM_Z + 0.001])
                return tail - props.PLUG_L * d + np.array([0.0, 0.0, dz])

            def carry(self):
                self.model = measure_plug()
                p, q = plug_pose()
                # Tip the plug nipple down on the way, its tail and hose pointing up at the hose hole.
                self.plug_q = qmul(q_axis((1, 0, 0), -math.radians(PLACE_TILT_DEG)), self.plug_q0)
                self._plug_to(p + np.array([0.0, -0.01, 0.10]), self.plug_q0, 0.05)
                hi = self.place_target(0.08)
                self._plug_to(hi, self.plug_q, 0.07)
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
                # Slide the open hand off along the plug toward its nipple (away from the hose), then lift.
                self.R.move(self.R.pos + ROOT_OFFSET + np.array([0.0, 0.03, 0.005]), speed=0.02)
                self.R.move(self.R.pos + ROOT_OFFSET + np.array([0.0, 0.0, 0.05]), speed=0.03)
                self.R.hold(hold_k(0.5))
                self.at(len(self.R.steps), self.seat_hose)

            def hose_points(self):
                """Centres of the target hose's links (desk frame)."""
                pos = st.hose_t.data.body_pos_w.torch[0].cpu().numpy() - origin
                quat = st.hose_t.data.body_quat_w.torch[0].cpu().numpy()
                out = []
                for k in range(props.CORD_N):
                    i = st.hose_t.body_names.index(f"cord_{k:02d}")
                    out.append(pos[i] + rotate(quat[i], (props.CORD_LINK / 2, 0.0, 0.0)))
                return np.array(out)

            def hose_at_wall(self, tag):
                """Log where the hose crosses the hose-hole wall (box frame), return the hose point nearest to
                WALL_OUT outside it."""
                pts = self.hose_points() - BOX
                wall = -props.BOX_HL
                k = int(np.argmin(np.abs(pts[:, 1] - (wall - 0.02))))
                near = pts[np.abs(pts[:, 1] - wall) < 0.03]
                log(f"HOSE {tag}: links near the end wall (box frame, mm) {np.round(near * 1000, 1).tolist()}",
                    flush=True)
                return pts[k] + BOX

            def seat_hose(self):
                """Press the hose down into the half hole just outside the end wall, as a hand settles it before
                closing."""
                self.report("PLACED")
                self.hose_at_wall("before")
                R = self.R
                p = BOX + np.array([0.0, -props.BOX_HL - 0.012, props.CORD_R + TIP_PAST_GRASP + 0.001])
                R.grip_to(CLOSED, hold_k(0.4))
                R.move(p + np.array([0.0, 0.0, 0.04]), DOWN_X, speed=0.05)
                R.move(p, speed=0.02)
                R.move(p - np.array([0.0, 0.0, 0.003]), speed=0.01)
                R.hold(hold_k(0.4))
                R.move(p + np.array([0.0, 0.0, 0.05]), speed=0.04)
                self.at(len(R.steps), lambda: self.hose_at_wall("seated"))
                R.move(R_WAIT, QB, speed=0.08)
                self.L.pad_to(len(R.steps) - hold_k(2.0))
                self.at(len(self.L.steps), self.lid_start)

            def release_hose(self):
                pass

            # -- 3. close the lid -------------------------------------------------------------------
            def lid_start(self):
                th = hinge_deg()
                self.lid_g = (props.HINGE[0] - LID_GRASP_R, TAB_Y, 0.0)
                g = lid_point(self.lid_g, th)
                q = lid_hand(th)
                L = self.L
                L.move(g - 0.07 * fdir(q) + 0.03 * EZ, q, speed=0.08)
                L.move(g - 0.025 * fdir(q), speed=0.04)
                L.hold(hold_k(0.3))
                self.when_settled(L, "left", self._lid_grip)

            def _lid_grip(self):
                th = hinge_deg()
                g = lid_point(self.lid_g, th)
                err = grasp_point("left")[0] - (self.L.pos + ROOT_OFFSET)
                self.lid_err = err if np.linalg.norm(err) < 0.008 else np.zeros(3)
                log(f"LID grasp at {th:.1f} deg: hand lag (mm) {np.round(err * 1000, 1)}", flush=True)
                self.L.move(g - self.lid_err, speed=0.015)
                self.L.hold(hold_k(0.4))
                self.L.grip_to(CLOSED, hold_k(0.8))
                self.at(len(self.L.steps), lambda: self._lid_swing(0))

            def _lid_swing(self, k):
                th = hinge_deg()
                g, gq = grasp_point("left")
                tab = rotate(conj(gq), lid_point(self.lid_g, th) - g)
                log(f"LID {k}: hinge {th:.1f} deg, tab grasp point in hand (mm) {np.round(tab * 1000, 1)}", flush=True)
                if th < LID_RELEASE_DEG + 2.0 or k > 3:
                    self.L.grip_to(OPEN, hold_k(0.4))
                    self.L.move(g - 0.03 * fdir(gq), speed=0.04)
                    self.at(len(self.L.steps), self._lid_push)
                    return
                if k == 0:
                    self.L.grip_to(LID_GRIP_SLIP, 2)
                for target in np.arange(th - 3.0, LID_RELEASE_DEG - 1.0, -3.0):
                    self.L.move(lid_point(self.lid_g, target) - self.lid_err, lid_hand(target), speed=0.03, min_steps=2)
                self.L.hold(hold_k(0.3))
                self.at(len(self.L.steps), lambda: self._lid_swing(k + 1))

            def _lid_push(self):
                """Near 63 degrees the lid balances on its hinge: push its outer face down with the closed fingertips,
                following it round, until it falls shut under its own weight."""
                th = hinge_deg()
                face = (props.HINGE[0] - 0.09, 0.0, props.HALF_D)
                up = np.array([0.0, 0.0, TIP_PAST_GRASP + 0.002])
                L = self.L
                L.grip_to(CLOSED, hold_k(0.3))
                L.move(lid_point(face, th) + up + 0.03 * EZ, DOWN, speed=0.05)
                L.move(lid_point(face, th) + up + 0.003 * EZ, speed=0.02)
                for t in np.arange(th - 2.0, 33.0, -2.0):
                    L.move(lid_point(face, t) + up - 0.004 * EZ, speed=0.02, min_steps=2)
                L.move(L.pos + ROOT_OFFSET + 0.04 * EZ - 0.02 * EX, speed=0.04)
                L.hold(hold_k(1.5))
                self.at(len(L.steps), self._lid_press)

            def _lid_press(self):
                """Press the shut lid down near its front with the closed fingertips (it may rest a few degrees open on
                the hose in the hole)."""
                self.report("LID dropped")
                p = BOX + np.array([-0.03, 0.0, props.HALF_D + TIP_PAST_GRASP])
                L = self.L
                L.grip_to(CLOSED, hold_k(0.3))
                L.move(p + np.array([0.0, 0.0, 0.04]), DOWN, speed=0.05)
                L.move(p + np.array([0.0, 0.0, 0.002]), speed=0.02)
                L.move(p - np.array([0.0, 0.0, 0.006]), speed=0.01)
                L.hold(hold_k(0.5))
                L.move(p + np.array([0.0, 0.0, 0.06]), speed=0.04)
                L.hold(hold_k(0.3))
                self.at(len(L.steps), self._lid_done)

            def _lid_done(self):
                self.report("LID closed")
                self.release_hose()
                L = self.L
                L.move(L_WAIT, DOWN, speed=0.08)
                if control:
                    self.R.pad_to(len(L.steps))
                    L.hold(hold_k(1.5))
                    self.at(len(L.steps), lambda: task.force_function_test(view))
                    L.hold(hold_k(4.0))
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
        start = os.environ.get("AIRLOCK_DEMO_START", "")  # development only: "lid" or "padlock"
        if start == "padlock":  # development only: start with the lid shut
            import torch

            z = torch.zeros(1, 1, device=view.device)
            st.box.write_joint_state_to_sim(z, z, joint_ids=[st.hinge])
            st.box.set_joint_position_target(z, joint_ids=[st.hinge])
        planner.at(settle, {"lid": planner.lid_start, "padlock": planner.plan_pick}.get(start, planner.plan_disconnect))

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
