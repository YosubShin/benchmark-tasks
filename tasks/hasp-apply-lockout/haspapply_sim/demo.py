"""The scripted demo, closed loop.

1. Arm A (left) pinches jaw A's ring on its free side (the side away from the switch wall), lifts the hasp out of its
   holder.
2. Arm B (right) presses a fingertip on jaw B's tail and drags it toward the robot, opening the jaws (as a thumb
   does); rivet friction keeps them open.
3. Arm A carries the open hasp level with the handle's lock-off hole and pushes it toward the wall, so jaw A's top
   bar threads through the handle's hole and the ear's hole (positions corrected from the measured hasp pose).
4. Arm B drags jaw B's tail back until the jaws are shut; arm A lowers the ring onto the holes' lower edges and
   lets go.
5. Arm A picks the open padlock off its pedestal, lines the short leg up with a tail hole (bottom row, free side)
   from the measured tip position and pushes it through both jaws; arm B pushes the shackle's bend shut while arm A
   holds the body; both let go.

Grasp-point targets become yam_arena's 14-D joint action through yam_arena's own IK every control step. It reads the
simulator's state (hasp, handle and padlock poses), like any scripted demo with privileged state;
``--policy hasp-apply-demo`` (haspapply_sim.policy) plays it. With HASP_CONTROL=1 it stops after step 4 (no padlock).
Import after Isaac Sim has started.
"""

from __future__ import annotations

import math
import os

import numpy as np

from yam_arena import ik, station


class Demo:
    """``next_action()`` gives the 14-D joint action for the next control step."""

    def __init__(self, STATION, control_hz: float = 20.0, speed: float = 0.10, log=print):
        from haspapply_sim import station as hb

        view = STATION.env
        task = STATION.task
        state_k = [0]
        self.state_k = state_k
        control = os.environ.get("HASP_CONTROL") == "1"
        from haspapply_sim.kit import CLOSED, DOWN, FORWARD, OPEN, ROOT_OFFSET, ArmTimeline, GraspModel, q_axis, qmul, rotate
        from haspapply_sim.kit import grasp_point as grasp_point_world
        from haspapply_sim.task import (EAR_HOLE_IN_SWITCH, PV, TAB_HOLE_IN_HANDLE, _YAW_Q, hasp_start_pose,
                                        layout_dir, lock_start_pose)
        from haspapply_sim.usd import hasp_props as props
        origin = view.scene.env_origins[0].cpu().numpy()
        robot = view.scene["robot"]

        def grasp_point(side):
            p, q = grasp_point_world(robot, side)
            return p - origin, q

        for side, name in (("left", "left_robot"), ("right", "right_robot")):
            qj = view.scene[name].data.joint_pos.torch[0, :6].cpu().numpy()
            tcp, _ = ik.tcp_pose(qj, station.BASES[side])
            log(f"CHECK {side}: their TCP {np.round(tcp, 4)} our grasp point {np.round(grasp_point(side)[0] + origin, 4)}",
                flush=True)

        turn = lambda v: np.array(layout_dir(v))  # noqa: E731
        turn_q = lambda q: qmul(_YAW_Q, q)  # noqa: E731
        EX, EY, EZ = turn((1.0, 0.0, 0.0)), turn((0.0, 1.0, 0.0)), np.array([0.0, 0.0, 1.0])
        HASP_QUAT = np.array(_YAW_Q)
        # Arm A on jaw A's free-side ring bar: fingers toward the wall and 40 deg down, closing across the hasp.
        PITCH_A = math.radians(40)
        QL_HASP = turn_q(qmul(q_axis((0, 1, 0), PITCH_A), FORWARD))
        L_DIR = math.cos(PITCH_A) * EX - math.sin(PITCH_A) * EZ
        BAR_IN_HASP = (-(props.JAW_R + props.JAW_W / 2), props.JAW_T / 2, -0.006)
        # Arm B's fingertip on jaw B's tail: fingers along +EY (into jaw B's face), 35 deg down, closing along EX.
        PITCH_B = math.radians(35)
        QR_DRAG = turn_q(qmul(q_axis((0, 0, 1), math.pi / 2), qmul(q_axis((0, 1, 0), PITCH_B), FORWARD)))
        R_DIR = math.cos(PITCH_B) * EY - math.sin(PITCH_B) * EZ
        TIP_PAST_PAD = 0.006
        B_FACE_Y = -props.JAW_T
        OPEN_IN = (0.008, 0.0)
        """Opening: arm B's closed fingertips go into the ring here, clear of jaw B's wall-side bar ..."""
        OPEN_P0 = (0.008, 0.0)
        """Opening: arm B's fingertip, inside the ring, pushes jaw B's wall-side bar outward from here."""
        PROBE_Y = 0.0
        """Pad-centre offset across the hasp that puts the fingertips' ends about 2 mm past jaw B's layer."""
        CLOSE_P0 = (-0.026, -0.065)
        """Closing: arm B's fingertips push jaw B's tail back from beyond its free-side edge."""
        LEAD_DEG = 6.0
        OPEN_DEG, THREAD_BACK, BAR_LIFT = 38.0, 0.040, 0.0003
        THREAD_STAGES = (THREAD_BACK, 0.016, 0.014, 0.012, 0.008, 0.004, 0.0)
        BAR_LIFT = 0.0018
        THREAD_LOW = 0.003
        # Padlock part (as in the group-hasp demo).
        LOCK_QUAT = np.array(lock_start_pose()[3:])
        # Arm A grips the lying padlock's body across its 19 mm thickness near its bottom end, fingers toward the
        # hasp and 65 deg down.
        SIDE_A = qmul(q_axis((0, 0, 1), -math.pi / 2), FORWARD)
        QA = turn_q(qmul(q_axis((1, 0, 0), math.radians(65)), SIDE_A))
        A_FINGER_DIR = turn((0.0, -math.cos(math.radians(65)), -math.sin(math.radians(65))))
        YAW_B = math.radians(60)
        QB_WAIT = qmul(q_axis((-math.sin(YAW_B), math.cos(YAW_B), 0), math.radians(20)), qmul(q_axis((0, 0, 1), YAW_B), FORWARD))
        QB = turn_q(QB_WAIT)
        B_FINGER_DIR = turn((math.cos(YAW_B) * math.cos(math.radians(20)), math.sin(YAW_B) * math.cos(math.radians(20)),
                             -math.sin(math.radians(20))))
        FINGER_END, GRASP_IN_LOCK, HOLE, FACE_GAP, PIERCE = 0.012, (0.0, 0.0, -0.012), (0, 0), 0.006, 0.010
        R_WAIT = np.array((0.18, -0.14, 0.30))
        L_WAIT = np.array((0.16, 0.20, 0.30))

        lock, hasp, switch = view.scene["padlock"], view.scene["hasp"], view.scene["switch"]
        body_id = lock.find_bodies("body")[0][0]
        shackle_joint = lock.find_joints("shackle_joint")[0][0]
        jaw_a = hasp.find_bodies("jaw_a")[0][0]
        jaw_joint = hasp.find_joints("jaw_joint")[0][0]
        handle_body = switch.find_bodies("handle")[0][0]
        enclosure_body = switch.find_bodies("enclosure")[0][0]
        dt = 1.0 / control_hz
        hold_k = lambda s: max(1, round(s / dt))  # noqa: E731

        def body_pose(asset, i):
            return (asset.data.body_pos_w.torch[0, i].cpu().numpy() - origin,
                    asset.data.body_quat_w.torch[0, i].cpu().numpy())

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
                            log(f"SETTLE timeout ({side}): lag {lag * 1000:.1f} mm")
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

            # -- hasp state --------------------------------------------------------------------------
            def hasp_pose(self):
                return body_pose(hasp, jaw_a)

            def jaw(self):
                return math.degrees(float(hasp.data.joint_pos.torch[0, jaw_joint]))

            def in_hasp(self, p):
                pos, quat = self.hasp_pose()
                return pos + rotate(quat, p)

            def holes_mid(self):
                h_pos, h_quat = body_pose(switch, handle_body)
                e_pos, e_quat = body_pose(switch, enclosure_body)
                tab = h_pos + rotate(h_quat, TAB_HOLE_IN_HANDLE)
                ear = e_pos + rotate(e_quat, EAR_HOLE_IN_SWITCH)
                return (tab + ear) / 2

            def threaded_ring(self):
                """Ring centre that puts jaw A's top bar 0.5 mm above the lock-off holes' lower edges."""
                return self.holes_mid() - (props.JAW_R + props.LOCKOFF_HOLE / 2 - 0.0005) * EZ

            def b_point(self, theta_deg, p0, y, pose=None):
                """World point that moves with jaw B: ``p0`` = (x, z) in the hasp frame with the jaws shut, turned
                with jaw B about the rivet to theta_deg; ``y`` across the hasp (jaw B's face is at -2 mm)."""
                t = math.radians(theta_deg)
                x0, z0 = p0[0], p0[1] - props.RIVET_Z
                x, z = x0 * math.cos(t) + z0 * math.sin(t), -x0 * math.sin(t) + z0 * math.cos(t)
                if pose is None:
                    return self.in_hasp((x, y, props.RIVET_Z + z))
                return pose[0] + rotate(pose[1], (x, y, props.RIVET_Z + z))

            def pads_in_hasp(self):
                from haspapply_sim.kit import FINGER_PAD_POINTS
                pos, quat = self.hasp_pose()
                out = []
                for name, off in (("right_tip_left", FINGER_PAD_POINTS[0]), ("right_tip_right", FINGER_PAD_POINTS[1])):
                    i = robot.body_names.index(name)
                    bp = robot.data.body_pos_w.torch[0, i].cpu().numpy() - origin
                    bq = robot.data.body_quat_w.torch[0, i].cpu().numpy()
                    w = bp + rotate(bq, off)
                    out.append(np.round(rotate(np.array([-quat[0], -quat[1], -quat[2], quat[3]]), w - pos) * 1000, 1))
                return out

            def hasp_to(self, center, speed, quat=None):
                pos, q = self.model.grasp_for(center, HASP_QUAT if quat is None else quat)
                self.L.move(pos, q, speed=speed)

            # -- 1. take the hasp out of its holder --------------------------------------------------------
            def plan_hasp_pick(self):
                start = np.array(hasp_start_pose()[:3])
                grasp = start + rotate(HASP_QUAT, BAR_IN_HASP)
                L = self.L
                L.grip_to(OPEN, hold_k(0.2))
                L.move(grasp - 0.06 * L_DIR + 0.02 * EZ, QL_HASP, min_steps=hold_k(3.0))
                L.move(grasp - 0.02 * L_DIR, speed=0.04)
                L.hold(hold_k(0.4))
                self.when_settled(L, "left", self._hasp_final_approach)

            def _hasp_final_approach(self):
                grasp = self.in_hasp(BAR_IN_HASP)
                err = grasp_point("left")[0] - (self.L.pos + ROOT_OFFSET)
                self.L.move(grasp - err, speed=0.02)
                self.L.hold(hold_k(0.4))
                self.L.grip_to(CLOSED, hold_k(1.0))
                self.when_settled(self.L, "left", self._hasp_lift)

            def _hasp_lift(self):
                self.model = GraspModel.measure(robot, "left", hasp)
                log(f"HASP grasp rel_pos (mm) {np.round(self.model.rel_pos * 1000, 1)} jaw {self.jaw():.1f}")
                c = self.hasp_pose()[0]
                self.hasp_to(c + 0.03 * EZ, 0.02)
                self.hasp_to(c + 0.06 * EZ, 0.04)
                pre = self.threaded_ring() - THREAD_BACK * EX - THREAD_LOW * EZ
                c2 = c + 0.06 * EZ
                self.hasp_to(c2 + float((pre - c2) @ EX) * EX, 0.05)
                self.hasp_to(pre + 0.025 * EZ, 0.06)
                self.hasp_to(pre, 0.04)
                self.L.hold(hold_k(0.6))
                self.when_settled(self.L, "left", self.open_start)

            # -- 2. open the jaws ----------------------------------------------------------------------
            def open_start(self):
                log(f"OPEN start: jaw {self.jaw():.1f} ring {np.round(self.hasp_pose()[0], 4)}")
                R = self.R
                R.pad_to(len(self.L.steps))
                c = self.b_point(self.jaw(), OPEN_IN, -0.006) - TIP_PAST_PAD * R_DIR
                R.grip = CLOSED
                R.move(c - 0.05 * R_DIR, QR_DRAG, min_steps=hold_k(2.5))
                R.move(c - 0.004 * EY, speed=0.04)
                R.hold(hold_k(0.4))
                self.when_settled(R, "right", self._open_insert)

            def _open_insert(self):
                c = self.b_point(self.jaw(), OPEN_IN, PROBE_Y) - TIP_PAST_PAD * R_DIR
                self.R.move(c, speed=0.01)
                self.R.hold(hold_k(0.4))
                self.when_settled(self.R, "right", lambda: self._drag(OPEN_P0, PROBE_Y, OPEN_DEG, 0, self.open_done))

            def _drag(self, p0, y, goal, k, done, tip_dir=None):  # noqa: C901
                """Arm B's closed fingertip pushes jaw B (at the point ``p0`` of jaw B) toward `goal` degrees, leading
                the measured jaw angle by a few degrees."""
                jaw = self.jaw()
                if k == 0:
                    self.drag_from = jaw
                    self.drag_pose0 = self.hasp_pose()
                sign = 1.0 if goal > self.drag_from else -1.0
                finished = (jaw >= goal) if sign > 0 else (jaw <= goal)
                hist = self.__dict__.setdefault("drag_hist", [])
                if k == 0:
                    hist.clear()
                hist.append(jaw)
                stalled = len(hist) > 8 and abs(hist[-1] - hist[-8]) < 0.3 and (jaw > 20.0 if sign > 0 else jaw < 1.5)
                if finished or stalled or k > 60:
                    log(f"DRAG done after {k}: jaw {jaw:.1f} (goal {goal})")
                    done()
                    return
                target = jaw + sign * (LEAD_DEG if sign > 0 or jaw > 6.0 else 9.0)
                target = min(target, goal + 3.0) if sign > 0 else max(target, goal - 3.0)
                p = self.b_point(target, p0, y)
                c = p - TIP_PAST_PAD * (R_DIR if tip_dir is None else tip_dir)
                self.R.move(c, speed=0.02, min_steps=hold_k(0.25))
                self.L.pad_to(len(self.R.steps))
                if k % 5 == 0:
                    log(f"DRAG {k}: jaw {jaw:.1f} -> {target:.1f} ring {np.round(self.hasp_pose()[0], 4)} "
                        f"arm B pads in hasp frame (mm) {self.pads_in_hasp()}")
                self.at(len(self.R.steps), lambda: self._drag(p0, y, goal, k + 1, done, tip_dir))

            def _r_back_off(self, away, tip_dir=None):
                R = self.R
                R.move(R.pos + ROOT_OFFSET + away, speed=0.03)
                R.move(R.pos + ROOT_OFFSET - 0.04 * (R_DIR if tip_dir is None else tip_dir), speed=0.04)
                R.move(R_WAIT, speed=0.10)

            def open_done(self):
                self.model = GraspModel.measure(robot, "left", hasp)
                log(f"OPENED: jaw {self.jaw():.1f}, hasp re-measured in the hand")
                self._r_back_off(-0.012 * EY)
                self.L.pad_to(len(self.R.steps) - hold_k(2.5))
                self.when_settled(self.L, "left", lambda: self._thread_correct(0))

            # -- 3. thread jaw A's top bar through the holes -------------------------------------------
            def _thread_correct(self, attempt, stage=0):
                """Threading in stages along EX (THREAD_STAGES, metres short of threaded), correcting the ring's
                height and sideways position from the measured hasp pose at each stage (the arm sags under way)."""
                back = THREAD_STAGES[stage]
                pre = self.threaded_ring() + BAR_LIFT * EZ - back * EX - (THREAD_LOW if stage == 0 else 0.0) * EZ
                err = self.hasp_pose()[0] - pre
                ey, ez = float(err @ EY), float(err[2])
                q = self.hasp_pose()[1]
                tilt = math.degrees(2 * math.asin(max(-1.0, min(1.0, float(np.linalg.norm(qmul(q, np.array([-HASP_QUAT[0], -HASP_QUAT[1], -HASP_QUAT[2], HASP_QUAT[3]]))[:3]))))))
                log(f"THREAD stage {stage} attempt {attempt}: ring error y {ey * 1000:.1f} z {ez * 1000:.1f} "
                    f"x {float(err @ EX) * 1000:.1f} mm, tilt {tilt:.1f} deg, jaw {self.jaw():.1f}")
                if not hasattr(self, "corr"):
                    self.corr = np.zeros(3)
                if tilt > 3.0:  # the hasp has turned in the hand: re-measure, so the hand levels it again
                    self.model = GraspModel.measure(robot, "left", hasp)
                self.corr = self.corr + ey * EY + ez * EZ
                if attempt < (4 if stage == 0 else (2 if stage <= 2 else 1)) and max(abs(ey), abs(ez)) > (0.0006 if stage < 3 else 0.001):
                    self.hasp_to(pre - self.corr, 0.008)
                    self.L.hold(hold_k(0.5))
                    self.when_settled(self.L, "left", lambda: self._thread_correct(attempt + 1, stage))
                    return
                if stage + 1 < len(THREAD_STAGES):
                    nxt = self.threaded_ring() + BAR_LIFT * EZ - THREAD_STAGES[stage + 1] * EX - self.corr
                    if stage == 0:
                        # Jaw B's open tip passes under the lock-off plates' lower edges on the way in: go in low,
                        # then lift the bar to the holes' height before it reaches the handle's plate.
                        self.hasp_to(nxt - THREAD_LOW * EZ, 0.01)
                        self.L.hold(hold_k(0.3))
                    self.hasp_to(nxt, 0.008)
                    self.L.hold(hold_k(0.5))
                    if stage > 0:
                        self.at(len(self.L.steps) - 1, self.trace)
                    self.when_settled(self.L, "left", lambda: self._thread_correct(0, stage + 1))
                    return
                self.trace()
                self.close_start()

            # -- 4. close the jaws, set the hasp down ------------------------------------------------------
            def trace(self):
                """Debug: jaw A's bar end and ring relative to the lock-off holes (switch frame), and the jaw."""
                e_pos, e_quat = body_pose(switch, enclosure_body)
                def sw(p):
                    return np.round(rotate(np.array([-e_quat[0], -e_quat[1], -e_quat[2], e_quat[3]]), p - e_pos) * 1000, 1)
                h_pos, h_quat = body_pose(switch, handle_body)
                tab = h_pos + rotate(h_quat, TAB_HOLE_IN_HANDLE)
                end = self.in_hasp((0.0085, 0.001, props.JAW_R + props.JAW_W / 2))
                log(f"TRACE k={state_k[0]} bar_end {sw(end)} ring {sw(self.hasp_pose()[0])} tab_hole {sw(tab)} "
                    f"ear_hole {np.round(np.array(EAR_HOLE_IN_SWITCH) * 1000, 1)} jaw {self.jaw():.1f} "
                    f"L grasp {np.round(grasp_point('left')[0], 4)}")

            def close_start(self):
                err = self.hasp_pose()[0] - self.threaded_ring()
                log(f"THREADED: ring vs threaded (mm) {np.round(err * 1000, 1)} jaw {self.jaw():.1f}")
                R = self.R
                R.pad_to(len(self.L.steps))
                c = self.b_point(self.jaw(), CLOSE_P0, -0.03) - TIP_PAST_PAD * R_DIR
                R.move(c - 0.04 * R_DIR, QR_DRAG, speed=0.08)
                R.move(c, speed=0.04)
                R.move(self.b_point(self.jaw(), CLOSE_P0, -0.001) - TIP_PAST_PAD * R_DIR, speed=0.02)
                R.hold(hold_k(0.4))
                self.when_settled(R, "right", lambda: self._drag(CLOSE_P0, -0.001, 0.3, 0, self.closed))

            def closed(self):
                self._r_back_off(-0.03 * EY)
                L = self.L
                L.pad_to(len(self.R.steps) - hold_k(2.0))
                self.hasp_to(self.threaded_ring() - self.corr - 0.0015 * EZ, 0.005)
                L.hold(hold_k(0.5))
                L.grip_to(OPEN, hold_k(0.8))
                L.move(L.pos + ROOT_OFFSET - 0.05 * L_DIR, speed=0.03)
                L.move(L.pos + ROOT_OFFSET + 0.04 * EZ - 0.03 * EX, speed=0.05)
                L.hold(hold_k(1.0))
                self.sync()
                self.at(len(L.steps), self.hasp_on_check)

            def hasp_on_check(self):
                rep = task.episode_report(view, 0)
                log(f"HASP ON: {rep['hasp_on_handle']} jaw {rep['jaw_deg']} milestone {rep['milestone']} "
                    f"ring {np.round(self.hasp_pose()[0], 4)}", flush=True)
                if control:
                    self.L.move(L_WAIT, speed=0.08)
                    self.L.hold(hold_k(8.0))
                    return
                self.plan_pick()

            # -- 5. padlock (as in the group-hasp demo) -------------------------------------------------
            def lock_pose(self):
                return body_pose(lock, body_id)

            def travel(self):
                return float(lock.data.joint_pos.torch[0, shackle_joint])

            def tip(self):
                pos, quat = self.lock_pose()
                return pos + rotate(quat, PV.short_tip(self.travel()))

            def hole(self):
                """The tail hole pair the shackle goes through: midway between jaw A's and jaw B's hole (they line up
                when the jaws are fully shut)."""
                hx, _, hz = props.hole_local(*HOLE)
                return 0.5 * (self.in_hasp(props.hole_local(*HOLE)) + self.b_point(self.jaw(), (hx, hz), 0.0))

            def face_point(self, hole):
                s = float((self.hasp_pose()[0] - hole) @ EY)
                return hole + (s - FACE_GAP) * EY

            def lock_center_for_tip(self, tip_target):
                return np.asarray(tip_target) - rotate(LOCK_QUAT, PV.short_tip(self.travel()))

            def _lock_to(self, center, speed):
                self.L.move(self.lmodel.grasp_for(center, LOCK_QUAT)[0], speed=speed)

            def plan_pick(self):
                start = np.array(lock_start_pose()[:3])
                grasp, _ = GraspModel.nominal(QA, LOCK_QUAT, GRASP_IN_LOCK).grasp_for(start, LOCK_QUAT)
                L = self.L
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
                self.when_settled(self.L, "left", self.carry)

            def carry(self):
                self.lmodel = GraspModel.measure(robot, "left", lock)
                log("LOCK grasp rel_pos (mm)", np.round(self.lmodel.rel_pos * 1000, 1))
                self.tip_entry = self.face_point(self.hole()) - 0.020 * EX
                c = self.lock_center_for_tip(self.tip_entry)
                # Over the empty hasp holder, then down beside the hasp's tail.
                here = self.lock_pose()[0]
                self._lock_to(here + 0.03 * EZ, 0.04)
                self._lock_to(c + 0.03 * EZ, 0.06)
                self._lock_to(c, 0.03)
                self.L.hold(hold_k(0.6))
                self.when_settled(self.L, "left", lambda: self._entry_correct(np.zeros(3), 0))

            def _entry_correct(self, correction, attempt):
                self.tip_entry = self.face_point(self.hole()) - 0.020 * EX
                err = self.tip() - self.tip_entry
                log(f"ENTRY attempt {attempt}: tip error (mm) {np.round(err * 1000, 1)}")
                correction = correction + err
                if attempt < 4 and np.abs(err).max() > 0.0015:
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
                self.tip_over = self.face_point(self.hole())
                err = self.tip() - self.tip_over
                log(f"OVER hole attempt {attempt}: tip error (mm) {np.round(err * 1000, 1)}")
                ex, ez = float(err @ EX), float(err[2])
                self.correction = self.correction + ex * EX + ez * EZ
                if attempt < 4 and max(abs(ex), abs(ez)) > 0.001:
                    self._lock_to(self.lock_center_for_tip(self.tip_over - self.correction), 0.008)
                    self.L.hold(hold_k(0.6))
                    self.when_settled(self.L, "left", lambda: self._over_correct(attempt + 1))
                    return
                through = self.tip_over + PIERCE * EY
                self._lock_to(self.lock_center_for_tip(through - self.correction), 0.006)
                self.L.hold(hold_k(0.8))
                self.when_settled(self.L, "left", self._threaded)

            def _threaded(self):
                rep = task.episode_report(view, 0)
                log(f"PIERCED: tip {np.round(self.tip(), 4)} hole {np.round(self.hole(), 4)} milestone {rep['milestone']}",
                    flush=True)
                self.press_start()

            def bend_outer(self):
                pos, quat = self.lock_pose()
                return pos + rotate(quat, (0.0, 0.0, PV.bend_top(self.travel())))

            def press_start(self):
                R = self.R
                R.pad_to(len(self.L.steps))
                b = self.bend_outer()
                start = b - 0.012 * EY - FINGER_END * B_FINGER_DIR
                R.move(start - 0.06 * B_FINGER_DIR, QB, min_steps=hold_k(2.5))
                R.move(start, speed=0.04)
                R.hold(hold_k(0.6))
                self.body0 = self.lock_pose()[0]
                self.when_settled(R, "right", lambda: self._press(0))

            def _press(self, k):
                latched = bool(view._task_state.latched[0])
                if latched or k > 30:
                    log(f"PRESS done after {k}: travel {self.travel() * 1000:.1f} mm, latched {latched}")
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
                L.move(L.pos + ROOT_OFFSET + turn((-0.05, 0.06, 0.04)), speed=0.06)
                L.hold(hold_k(4.0))

        l_pos, l_quat = grasp_point("left")
        r_pos, r_quat = grasp_point("right")
        L = ArmTimeline(l_pos - ROOT_OFFSET, l_quat, dt, speed)
        R = ArmTimeline(r_pos - ROOT_OFFSET, r_quat, dt, speed)
        R.grip = CLOSED
        R.move(R_WAIT, QR_DRAG, min_steps=hold_k(3.0))
        planner = Planner(L, R)
        settle = hold_k(1.0)
        L.hold(settle)
        planner.at(settle, planner.plan_hasp_pick)

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
            if pe > 0.005 and (pe > worst[side] or state_k[0] % 20 == 0):
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
