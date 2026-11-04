"""The scripted demo, closed loop.

1. Arm A (left), fingers level, pinches the sides of F2's grip tab from the front and pulls it toward the robot
   along its arc (pulling further while the carrier does not follow) until it leaves the latch, then lets go:
   gravity swings the carrier open to its stop. (Pressing the tab top down, from above or with level fingers, did
   not open it: the pad's friction on the tab top locks it, and the fingers' wide part rests on the holder's upper
   body.)
2. Arm B (right), fingers down and leaning 40 deg toward the holder, closes across the blown fuse's outer ferrule,
   lifts it toward the clip's open side in short steps, draws it out toward the robot, and drops it in the tray.
3. Arm B picks the spare of the matching rating off the stand by its near ferrule, lines it up with the open
   carrier's axis 9 mm above its seat (re-measuring how the fuse sits in its fingers), slides it in over the seat,
   lowers it into the clip and lets go.
4. Arm A, closed fingers level, lifts the carrier by the underside of its grip tab and follows it round its arc
   until it latches shut, then backs away. The task's hold test then pulls the carrier open with 4 N at the tab.

It reads the simulator's state (the carrier angle and the fuses' poses), like any scripted demo with privileged
state, and turns grasp-point targets into yam_arena's 14-D joint action through yam_arena's own IK every control
step. ``--policy fuse-demo`` (fuse_sim.policy) plays it through Inspect Robots. Import after Isaac Sim has started.

Control (``FUSE_CONTROL=1``): the demo is unchanged, but the task turns the carrier latch off (fuse_sim.task.NO_LATCH),
so nothing holds the shut carrier against the hold test's pull.
"""

from __future__ import annotations

import math
import os

import numpy as np

from yam_arena import ik, station


class Demo:
    """``next_action()`` gives the 14-D joint action for the next control step."""

    def __init__(self, STATION, control_hz: float = 20.0, speed: float = 0.10, log=print):
        from fuse_sim import station as hb
        from fuse_sim.kit import CLOSED, DOWN, FORWARD, ROOT_OFFSET, ArmTimeline, GraspModel, conj, q_axis, qmul, rotate
        from fuse_sim.kit import grasp_point as grasp_point_world
        from fuse_sim.spec import TRAY_XY
        from fuse_sim.task import FUSE_ROLES
        from fuse_sim.usd import fuse_props as props

        view, task = STATION.env, STATION.task
        state_k = [0]
        self.state_k = state_k
        origin = view.scene.env_origins[0].cpu().numpy()  # desk-frame origin in the world
        robot = view.scene["robot"]
        st = task.state(view)
        layout = st.next_layout

        def grasp_point(side):
            p, q = grasp_point_world(robot, side)
            return p - origin, q

        for side, name in (("left", "left_robot"), ("right", "right_robot")):
            qj = view.scene[name].data.joint_pos.torch[0, :6].cpu().numpy()
            tcp, _ = ik.tcp_pose(qj, station.BASES[side])
            log(f"CHECK {side}: their TCP {np.round(tcp, 4)} our grasp point {np.round(grasp_point(side)[0] + origin, 4)}",
                flush=True)

        k10 = control_hz / 10.0

        def n(steps):
            """A step count of the 10 Hz original, at this control rate."""
            return max(1, round(steps * k10))

        FINGER_END = 0.012
        TIP_HALF_W = 0.0075
        PRESS_DEG = 24.0
        PINCH_D = -0.0005 - FINGER_END
        """Pinch grasp point: the fingertips end 0.5 mm in front of the carrier face, so the pads cover the tab's
        front 8.5 mm and the fingers stay clear of the holder body's side walls behind it."""
        PINCH_Z = 0.5 * (props.TAB_Z[0] + props.TAB_Z[1])
        PINCH_OPEN = 0.30
        FUSE_GRIP = 0.0  # squeeze fully: the station gripper is a 5000 N/m spring capped at 50 N
        FUSE_OPEN_CMD = 0.35
        TOP_LEAN = math.radians(40.0)
        ALONG, BELOW = 0.002, 0.008
        Q_TOP = qmul(q_axis((0, 1, 0), -TOP_LEAN), DOWN)
        OPEN_RAD = math.radians(props.OPEN_DEG)
        HOVER = 0.009
        SPARE_VIRTUAL_Q = q_axis((0, 1, 0), -math.pi / 2)
        CLOSE_CONTACT = np.array([props.TAB_D[0], 0.0, props.TAB_Z[0] + 0.002])
        CLOSE_LEAD = 0.0015
        L_PARK = np.array([0.13, 0.20, 0.20])

        def grasp_in_fuse(tip_z, below):
            cq = q_axis((0, 1, 0), -OPEN_RAD)
            f_world = np.array([math.sin(TOP_LEAN), 0.0, -math.cos(TOP_LEAN)])
            f_c = rotate(conj(cq), f_world)
            tip = np.array([0.0, 0.0, tip_z]) + f_c * below
            return tip - f_c * FINGER_END, qmul(conj(cq), Q_TOP)

        GRASP_F, REL_Q = grasp_in_fuse(props.FUSE_L / 2 - ALONG, BELOW)

        def carrier_q(theta):
            return q_axis((0, 1, 0), -theta)

        def carrier_pt(theta, d, z, slot=1):
            h = np.array(props.hinge_world(slot))
            return h + rotate(carrier_q(theta), np.array([d - props.HINGE_D, 0.0, z - props.HINGE_Z]))

        holders = view.scene["holders"]
        joint = holders.find_joints("carrier1_joint")[0][0]
        carrier = holders.find_bodies("carrier1")[0][0]
        fuses = {r: view.scene[f"fuse_{r}"] for r in FUSE_ROLES}

        class Planner:
            def __init__(self, L, R):
                self.L, self.R, self.events = L, R, {}
                self.spare_role = FUSE_ROLES[1 + layout.matching_spare]

            def at(self, step, fn):
                self.events.setdefault(step, []).append(fn)

            def run_due(self, k):
                for fn in self.events.pop(k, []):
                    fn()

            def when_settled(self, tl, side, fn, tries=15, tol=0.003):
                def check():
                    lag = np.linalg.norm(grasp_point(side)[0] - (tl.pos + ROOT_OFFSET))
                    if lag < tol or check.tries <= 0:
                        if lag >= tol:
                            log(f"SETTLE timeout ({side}): lag {lag * 1000:.1f} mm", flush=True)
                        fn()
                    else:
                        check.tries -= 1
                        tl.hold(n(2))
                        self.at(len(tl.steps), check)
                check.tries = tries
                self.at(len(tl.steps), check)

            def sync(self):
                m = max(len(self.L.steps), len(self.R.steps))
                self.L.pad_to(m)
                self.R.pad_to(m)

            # state
            def theta(self):
                return -float(holders.data.joint_pos.torch[0, joint])

            def fuse_pose(self, role):
                f = fuses[role]
                return f.data.root_pos_w.torch[0].cpu().numpy() - origin, f.data.root_quat_w.torch[0].cpu().numpy()

            def carrier_pose(self):
                c_pos = holders.data.body_pos_w.torch[0, carrier].cpu().numpy() - origin
                return c_pos, holders.data.body_quat_w.torch[0, carrier].cpu().numpy()

            def lag(self, side, tl):
                return np.linalg.norm(grasp_point(side)[0] - (tl.pos + ROOT_OFFSET)) * 1000

            def report(self, tag):
                rep = task.episode_report(view, 0)
                log(f"{tag}: theta {math.degrees(self.theta()):.1f} deg milestone {rep['milestone']} "
                    f"seated {rep['fuse_seated_in_named']} sem {rep['semantic_events']} con {rep['contact_events']}",
                    flush=True)

            # 1. Open (arm A) -----------------------------------------------------------------------------
            def plan_open(self):
                """Pinch the grip tab's sides from the front (fingers level, closing across the rail, the fingertips
                ending 0.5 mm short of the carrier face) and pull it toward the robot along its arc, as a person
                pulls the tab; past the latch, let go and gravity swings the carrier open."""
                L = self.L
                self.ycorr = 0.0
                L.grip_to(PINCH_OPEN, n(2))
                g0 = self.pinch_target(0.0)
                L.move(g0 + np.array([-0.05, 0.0, 0.02]), FORWARD, min_steps=n(30))
                L.move(g0 + np.array([-0.02, 0.0, 0.0]), speed=0.04)
                L.move(g0, speed=0.015)
                L.hold(n(3))
                self.when_settled(L, "left", self.pinch)

            @staticmethod
            def pinch_target(th):
                return carrier_pt(th, PINCH_D, PINCH_Z)

            def pinch(self):
                log(f"ON TAB: lag {self.lag('left', self.L):.1f} mm, grasp {np.round(grasp_point('left')[0], 4)}",
                    flush=True)
                self.L.grip_to(0.0, n(8))
                self.at(len(self.L.steps), lambda: self.pull_open(0.0, 0.0, 0, 0.0))

            def pull_open(self, th_cmd, pull, k, th_prev):
                """Follow the tab's arc 4 deg ahead of the carrier, pulling 3 mm further whenever it stalls."""
                th = self.theta()
                if k:
                    log(f"PULL {k}: theta {math.degrees(th):.1f} deg, extra pull {pull * 1000:.0f} mm, "
                        f"lag {self.lag('left', self.L):.1f} mm", flush=True)
                if math.degrees(th) >= PRESS_DEG or k > 40:
                    self.let_go()
                    return
                if k and th - th_prev < math.radians(0.5):
                    pull = min(pull + 0.003, 0.030)
                th_cmd = min(max(th_cmd, th) + math.radians(4.0), math.radians(PRESS_DEG + 4.0))
                L = self.L
                start = L.pos + ROOT_OFFSET
                goal = self.pinch_target(th_cmd) + np.array([-pull, 0.0, 0.0])
                m = n(5)
                for i in range(1, m + 1):
                    L.path([(start + (goal - start) * i / m, FORWARD)])
                L.hold(n(2))
                self.at(len(L.steps), lambda: self.pull_open(th_cmd, pull, k + 1, th))

            def let_go(self):
                log(f"PULLED: theta {math.degrees(self.theta()):.1f} deg, lag {self.lag('left', self.L):.1f} mm",
                    flush=True)
                L = self.L
                L.grip_to(PINCH_OPEN, n(4))
                L.move(L.pos + ROOT_OFFSET + np.array([-0.03, 0.0, 0.03]), speed=0.05)
                L.move(L_PARK, FORWARD, speed=0.10, min_steps=n(15))
                self.sync()
                self.wait_open(0)

            def wait_open(self, tries):
                th = math.degrees(self.theta())
                if th > props.OPEN_DEG - 3.0 or tries > 80:
                    self.report(f"OPEN after {tries} checks")
                    if th < 35.0:  # it fell back shut: press again
                        self.plan_open()
                        return
                    self.plan_remove()
                    return
                self.R.hold(n(2))
                self.sync()
                self.at(len(self.R.steps), lambda: self.wait_open(tries + 1))

            # 2. Old fuse out (arm B) ----------------------------------------------------------------------
            @staticmethod
            def hand_for(pos, vq):
                return pos + rotate(vq, GRASP_F), qmul(vq, REL_Q)

            def plan_remove(self):
                R = self.R
                pos, _ = self.fuse_pose("old")
                _, c_quat = self.carrier_pose()
                g, q = self.hand_for(pos, c_quat)
                f_dir = rotate(q, (0.0, 0.0, -1.0))
                self.outward = rotate(c_quat, (0.0, 0.0, 1.0))
                log(f"OLD FUSE at {np.round(pos, 4)}, grasp {np.round(g, 4)}", flush=True)
                R.grip_to(FUSE_OPEN_CMD, n(2))
                R.move(g - f_dir * 0.06 + self.outward * 0.01, q, min_steps=n(35))
                R.move(g - f_dir * 0.03, speed=0.05)
                R.move(g, speed=0.02)
                R.hold(n(4))
                self.when_settled(R, "right", self.grab_old)

            def grab_old(self):
                log(f"AT OLD FUSE: lag {self.lag('right', self.R):.1f} mm", flush=True)
                self.R.grip_to(FUSE_GRIP, n(10))
                self.at(len(self.R.steps), self.draw_out)

            def draw_out(self, seg=0):
                R = self.R
                _, c_quat = self.carrier_pose()
                up = rotate(c_quat, (1.0, 0.0, 0.0))
                if seg:
                    pos, _ = self.fuse_pose("old")
                    log(f"LIFT {seg}: lag {self.lag('right', R):.1f} mm, fuse {np.round(pos, 4)}", flush=True)
                if seg < 4:
                    R.move(R.pos + ROOT_OFFSET + up * 0.006, speed=0.006)
                    R.hold(n(2))
                    self.when_settled(R, "right", lambda: self.draw_out(seg + 1), tries=10, tol=0.002)
                    return
                R.move(R.pos + ROOT_OFFSET + np.array([-0.03, 0.0, 0.02]), speed=0.03)
                self.when_settled(R, "right", self.to_tray, tries=8)

            def to_tray(self):
                self.report("OLD DRAWN OUT")
                R = self.R
                tx, ty = TRAY_XY
                R.move(np.array([tx, ty, 0.16]), speed=0.08, min_steps=n(25))
                R.hold(n(5))
                R.grip_to(0.6, n(15))
                R.move(np.array([tx - 0.02, ty + 0.03, 0.19]), speed=0.08)
                self.sync()
                self.at(len(R.steps), self.plan_pick)

            # 3. Spare in (arm B) ---------------------------------------------------------------------------
            def plan_pick(self):
                self.report("OLD DROPPED")
                R = self.R
                pos, quat = self.fuse_pose(self.spare_role)
                self.spare_d = qmul(conj(SPARE_VIRTUAL_Q), quat)
                g, q = self.hand_for(pos, SPARE_VIRTUAL_Q)
                f_dir = rotate(q, (0.0, 0.0, -1.0))
                log(f"SPARE {self.spare_role} at {np.round(pos, 4)}, grasp {np.round(g, 4)}", flush=True)
                R.grip_to(FUSE_OPEN_CMD, n(2))
                R.move(g - f_dir * 0.06, q, min_steps=n(30))
                R.move(g - f_dir * 0.03, speed=0.05)
                R.move(g, speed=0.02)
                R.hold(n(4))
                self.when_settled(R, "right", self.grab_spare)

            def grab_spare(self):
                log(f"AT SPARE: lag {self.lag('right', self.R):.1f} mm", flush=True)
                R = self.R
                R.grip_to(FUSE_GRIP, n(10))
                R.move(R.pos + ROOT_OFFSET + np.array([0.0, 0.0, 0.03]), speed=0.03)
                R.move(R.pos + ROOT_OFFSET + np.array([-0.03, 0.0, 0.01]), speed=0.05)
                self.when_settled(R, "right", self.to_carrier, tries=8)

            def to_carrier(self):
                self.grasp = GraspModel.measure(robot, "right", fuses[self.spare_role])
                c_pos, c_quat = self.carrier_pose()
                seat = c_pos + rotate(c_quat, props.seat_in_carrier())
                outward = rotate(c_quat, (0.0, 0.0, 1.0))
                quat = qmul(c_quat, self.spare_d)
                self.ins = (seat, quat, outward)
                self.up = rotate(c_quat, (1.0, 0.0, 0.0))
                self.corr = np.zeros(3)
                g, q = self.grasp.grasp_for(seat + self.up * HOVER + outward * 0.045, quat)
                log(f"SPARE HELD: rel {np.round(self.grasp.rel_pos * 1000, 1)} mm; seat {np.round(seat, 4)}",
                    flush=True)
                R = self.R
                R.move(g + np.array([-0.02, 0.0, 0.03]), q, speed=0.06, min_steps=n(30))
                R.move(g, q, speed=0.03)
                R.hold(n(5))
                self.when_settled(R, "right", lambda: self.insert(0))

            def insert(self, attempt):
                seat, quat, outward = self.ins
                fuse = fuses[self.spare_role]
                f_pos, f_quat = self.fuse_pose(self.spare_role)
                want = seat + self.up * HOVER + outward * 0.045
                inner = f_pos + rotate(f_quat, (0.0, 0.0, props.FUSE_L / 2))
                err = inner - (want + rotate(quat, (0.0, 0.0, props.FUSE_L / 2)))
                tilt = math.degrees(math.acos(min(1.0, abs(float(np.dot(rotate(f_quat, (0, 0, 1)),
                                                                        rotate(quat, (0, 0, 1))))))))
                log(f"PRE-PLACE {attempt}: inner end error {np.round(err * 1000, 1)} mm, axis off {tilt:.1f} deg",
                    flush=True)
                if attempt < 4 and (np.abs(err).max() > 0.0006 or tilt > 1.5):
                    self.grasp = GraspModel.measure(robot, "right", fuse)
                    if attempt:
                        self.corr = self.corr + err
                    g, q = self.grasp.grasp_for(want - self.corr, quat)
                    self.R.move(g, q, speed=0.01)
                    self.R.hold(n(4))
                    self.when_settled(self.R, "right", lambda: self.insert(attempt + 1), tol=0.0015)
                    return
                self.grasp = GraspModel.measure(robot, "right", fuse)
                self.slide(0)

            def slide(self, seg):
                seat, quat, outward = self.ins
                if seg == 3:
                    self.lower(0)
                    return
                g, q = self.grasp.grasp_for(seat + self.up * HOVER + outward * (0.030, 0.015, 0.0)[seg] - self.corr,
                                            quat)
                self.R.move(g, q, speed=0.006)
                self.R.hold(n(3))
                self.when_settled(self.R, "right", lambda: self.slide(seg + 1), tries=8, tol=0.0015)

            def lower(self, seg):
                seat, quat, outward = self.ins
                if seg:
                    f_pos, _ = self.fuse_pose(self.spare_role)
                    log(f"LOWER {seg}: fuse off seat {np.round((f_pos - seat) * 1000, 1)} mm", flush=True)
                if seg == 2:
                    self.release_spare()
                    return
                g, q = self.grasp.grasp_for(seat + self.up * (0.004, 0.0005)[seg] + outward * 0.001 - self.corr, quat)
                self.R.move(g, q, speed=0.004)
                self.R.hold(n(4))
                self.when_settled(self.R, "right", lambda: self.lower(seg + 1), tries=8, tol=0.0015)

            def release_spare(self):
                R = self.R
                g_quat = grasp_point("right")[1]
                f_dir = rotate(g_quat, (0.0, 0.0, -1.0))
                R.grip_to(FUSE_OPEN_CMD, n(8))
                R.move(R.pos + ROOT_OFFSET - f_dir * 0.05, speed=0.03)
                R.hold(n(5))
                self.at(len(R.steps), self.after_release)

            def after_release(self):
                f_pos, _ = self.fuse_pose(self.spare_role)
                log(f"INSERTED: fuse off seat {np.round((f_pos - self.ins[0]) * 1000, 1)} mm", flush=True)
                R = self.R
                R.move(np.array([0.15, -0.20, 0.22]), speed=0.08, min_steps=n(15))
                self.sync()
                self.at(len(R.steps), self.plan_close)

            # 4. Close (arm A) ------------------------------------------------------------------------------
            @staticmethod
            def close_target(th, lead=CLOSE_LEAD):
                edge = carrier_pt(th, CLOSE_CONTACT[0], CLOSE_CONTACT[2])
                edge = edge + np.array([math.cos(th), 0.0, math.sin(th)]) * lead
                return edge - np.array([FINGER_END, 0.0, TIP_HALF_W])

            def plan_close(self):
                self.report("SPARE RELEASED")
                L = self.L
                th0 = self.theta()
                start = self.close_target(th0)
                L.grip_to(CLOSED, n(2))
                L.move(start + np.array([-0.04, 0.0, -0.01]), FORWARD, min_steps=n(30))
                L.move(start + np.array([-0.01, 0.0, -0.006]), speed=0.04)
                L.move(start + np.array([0.0, 0.0, -0.004]), speed=0.02)
                L.hold(n(3))
                self.stop_rad = math.radians(-2.0)
                self.when_settled(L, "left", lambda: self.sweep(th0, CLOSE_LEAD, 0, th0))

            def sweep(self, th_cmd, lead, k, th_prev):
                """Lift the tab along the arc in 6-deg steps ahead of the carrier; push 2 mm harder if it stalls."""
                th = self.theta()
                log(f"SWEEP {k}: theta {math.degrees(th):.1f} deg, lead {lead * 1000:.1f} mm, "
                    f"lag {self.lag('left', self.L):.1f} mm", flush=True)
                done = th < math.radians(0.5)
                if done or k > 40:
                    self.L.hold(n(8))
                    self.at(len(self.L.steps), self.back_off)
                    return
                if k and th_prev - th < math.radians(0.5):
                    lead = min(lead + 0.002, 0.012)
                th_cmd = max(min(th_cmd, th) - math.radians(6.0), self.stop_rad)
                L = self.L
                start = L.pos + ROOT_OFFSET
                goal = self.close_target(th_cmd, lead)
                m = n(6)
                for i in range(1, m + 1):
                    L.path([(start + (goal - start) * i / m, FORWARD)])
                L.hold(n(2))
                self.at(len(L.steps), lambda: self.sweep(th_cmd, lead, k + 1, th))

            def back_off(self):
                self.report("SWEPT")
                L = self.L
                L.move(L.pos + ROOT_OFFSET + np.array([-0.07, 0.0, 0.0]), speed=0.06)
                L.move(L_PARK, speed=0.10)
                self.sync()
                L.hold(n(60))
                self.sync()

        l_pos, l_quat = grasp_point("left")
        r_pos, r_quat = grasp_point("right")
        dt = 1.0 / control_hz
        L = ArmTimeline(l_pos - ROOT_OFFSET, l_quat, dt, speed)
        R = ArmTimeline(r_pos - ROOT_OFFSET, r_quat, dt, speed)
        planner = Planner(L, R)
        settle = n(15)
        L.hold(settle)
        R.hold(settle)
        if os.environ.get("FUSE_HOLD_CHECK") != "1":  # hold check (fuse_sim.task.HOLD_CHECK): the arms stay at ready
            planner.at(settle, planner.plan_open)
        self.planner = planner

        q_rel = np.array(hb.Q_REL)
        tcp_extra = 0.1347 - 0.134  # our grasp point sits this much further out than their TCP
        q_arm = {s: view.scene[f"{s}_robot"].data.joint_pos.torch[0, :6].cpu().numpy().astype(float)
                 for s in ("left", "right")}
        worst = {"left": 0.0, "right": 0.0}

        def joint_action(side, pos_root, quat, grip):
            pos = pos_root + ROOT_OFFSET + origin  # world
            tcp = pos + rotate(quat, (0.0, 0.0, -tcp_extra))
            rot = qmul(quat, q_rel)
            q, pe, _ = ik.solve(tcp, rot, q_arm[side], base_pos=station.BASES[side], iters=60)
            q_arm[side] = q
            if pe > 0.005 and pe > worst[side]:
                log(f"IK {side} k={state_k[0]} error {pe * 1000:.1f} mm at {np.round(pos - origin, 4)}", flush=True)
            worst[side] = max(worst[side], pe)
            return np.r_[q, grip]

        self._L, self._R, self._joint_action = L, R, joint_action
        self.worst = worst
        self._last_ms = -1
        self._task, self._view = task, view
        self._log = log

    def next_action(self) -> np.ndarray:
        k = self.state_k[0]
        L, R = self._L, self._R
        self.planner.run_due(k)
        L.pad_to(len(R.steps))
        R.pad_to(len(L.steps))
        if k >= len(L.steps):
            L.hold(1)
            R.hold(1)
        rep = self._task.episode_report(self._view, 0)
        if rep["milestone"] != self._last_ms:
            self._log(f"MILESTONE {rep['milestone']} at step {k}", flush=True)
            self._last_ms = rep["milestone"]
        (lp, lq, lg), (rp, rq, rg) = L.steps[k], R.steps[k]
        self.state_k[0] = k + 1
        return np.concatenate([self._joint_action("left", lp, lq, lg), self._joint_action("right", rp, rq, rg)])
