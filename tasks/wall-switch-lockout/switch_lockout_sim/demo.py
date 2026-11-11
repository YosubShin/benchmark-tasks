"""The scripted demo, closed loop.

Arm B (right), fingers closed and pitched 45 deg down, puts its fingertips on top of S1's toggle near its tip and
pushes it down past the middle (it snaps to OFF). It then opens, takes the open lockout cover by its two side walls
near the hinge (fingers pitched 30 deg down, closing across the cover), pulls it about 15 deg forward past its
balance point, lets go and backs away: the cover swings down shut under its own weight (its slot passing over the
padlock ring). Arm A (left) picks the open padlock off its post (fingers pitched 30 deg down on the body's narrow sides), carries
it in front of S1 with the short leg's tip just above the ring's height, moves it in so that the ring slides into the
gap between the short leg's tip and the body, then lowers it so the short leg goes down through the ring's hole
(the short leg's measured tip is corrected onto each target). Arm B, fingers closed and pitched 50 deg down, comes
down on top of the shackle's bend and pushes it into the body while arm A holds the body, until it latches. Both let
go; the task then runs its after-release test (off camera).

``control=True``: the same up to the cover swinging shut, no padlock; the test is forced 1 s after the cover is shut.

It reads the simulator's state (toggle and cover angles, the cover's and padlock's poses), like any scripted demo
with privileged state, and sends yam_arena's 14-D joint action through yam_arena's IK (``yam_arena.ik``). Import after
Isaac Sim has started.
"""

from __future__ import annotations

import math
import os

import numpy as np

from yam_arena import ik, station

STOP_AFTER = os.environ.get("WSL_DEMO_STOP", "")
"""Debugging: 'toggle' or 'cover' ends the demo (arms park) after that stage."""
START_AT = os.environ.get("WSL_DEMO_START", "")
"""Debugging: 'pick' puts S1 at OFF with its cover shut at the start and begins with the padlock."""


def _log(*args, **kwargs):
    kwargs["flush"] = True
    print("[wall-switch-demo]", *args, **kwargs)


class Demo:
    """``next_action()`` gives the 14-D joint action for the next control step."""

    def __init__(self, STATION, control_hz: float = 20.0, speed: float = 0.10, control: bool = False, log=_log):
        from switch_lockout_sim import station as hb
        from switch_lockout_sim.kit import CLOSED, FORWARD, OPEN, ROOT_OFFSET, ArmTimeline, GraspModel, q_axis, qmul
        from switch_lockout_sim.kit import grasp_point as grasp_point_world
        from switch_lockout_sim.kit import rotate
        from switch_lockout_sim.task import LOCK_QUAT, PV, TARGET_SLOT, lock_start_pose
        from switch_lockout_sim.usd import switch_props as props

        view = STATION.env
        task = STATION.task
        state_k = [0]
        self.state_k = state_k
        origin = view.scene.env_origins[0].cpu().numpy()
        robot = view.scene["robot"]

        def grasp_point(side):
            p, q = grasp_point_world(robot, side)
            return p - origin, q

        def pitched(deg):
            return qmul(q_axis((0, 1, 0), math.radians(deg)), FORWARD)

        def finger_dir(deg):
            return np.array([math.cos(math.radians(deg)), 0.0, -math.sin(math.radians(deg))])

        LQ = np.array(LOCK_QUAT)
        L_PITCH = 30.0
        QL = pitched(L_PITCH)
        """Arm A holds the upright padlock with fingers pointing at the wall, pitched 30 deg down, closing on the body's
        narrow sides (level fingers cannot reach this far left of the arm's base)."""
        L_FINGER_DIR = finger_dir(L_PITCH)
        GRASP_IN_LOCK = (0.0, 0.0, -0.006)
        """A little below the body's centre, on its centre line."""
        FINGER_END = 0.012
        """Grasp point to the closed fingers' ends."""
        T_PITCH, C_PITCH, P_PITCH = 45.0, 30.0, 50.0
        QT, QC, QP = pitched(T_PITCH), pitched(C_PITCH), pitched(P_PITCH)
        Y0 = props.SLOT_Y[TARGET_SLOT]
        HOLE = np.array(props.hole_world(TARGET_SLOT))
        PINCH_IN_COVER = np.array([0.0005, 0.0, -0.035])
        """Where arm B takes the open cover (cover frame, shut coordinates: 3.5 cm down from the hinge, mid-depth)."""
        PULL_TO_DEG = 168.0
        TIP_ABOVE_RING = 0.004
        FRONT_DX = 0.06
        DROP_DZ = 0.022
        """After the tip is over the hole, the padlock drops this far (the short leg goes down through the ring)."""
        PRESS_STEP, PRESS_MAX = 0.002, 45
        R_WAIT = np.array([0.15, -0.24, 0.25])
        L_WAIT = np.array([0.15, 0.24, 0.25])

        sw, lock = view.scene["switches"], view.scene["padlock"]
        body_id = lock.find_bodies("body")[0][0]
        shackle_joint = lock.find_joints("shackle_joint")[0][0]
        toggle_joint = sw.find_joints(f"toggle_{TARGET_SLOT}_joint")[0][0]
        cover_joint = sw.find_joints(f"cover_{TARGET_SLOT}_joint")[0][0]
        cover_body = sw.find_bodies(f"cover_{TARGET_SLOT}")[0][0]
        dt = 1.0 / control_hz
        hold_k = lambda s: max(1, round(s / dt))  # noqa: E731

        class Planner:
            def __init__(self, L, R):
                self.L, self.R, self.events = L, R, {}

            def at(self, step, fn):
                self.events.setdefault(step, []).append(fn)

            def run_due(self, k):
                for fn in self.events.pop(k, []):
                    fn()

            def sync(self):
                n = max(len(self.L.steps), len(self.R.steps))
                self.L.pad_to(n)
                self.R.pad_to(n)
                return n

            def when_settled(self, tl, side, fn, tries=15, tol=0.003):
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

            def when(self, tl, cond, fn, tries=200):
                def check():
                    if cond() or check.tries <= 0:
                        fn()
                    else:
                        check.tries -= 1
                        tl.hold(1)
                        self.sync()
                        self.at(len(tl.steps), check)
                check.tries = tries
                self.at(len(tl.steps), check)

            # state
            def toggle_deg(self):
                return math.degrees(float(sw.data.joint_pos.torch[0, toggle_joint]))

            def cover_deg(self):
                return math.degrees(float(sw.data.joint_pos.torch[0, cover_joint]))

            def pinch_world(self):
                pos = sw.data.body_pos_w.torch[0, cover_body].cpu().numpy() - origin
                quat = sw.data.body_quat_w.torch[0, cover_body].cpu().numpy()
                return pos + rotate(quat, PINCH_IN_COVER)

            def body_pose(self):
                pos = lock.data.body_pos_w.torch[0, body_id].cpu().numpy() - origin
                return pos, lock.data.body_quat_w.torch[0, body_id].cpu().numpy()

            def travel(self):
                return float(lock.data.joint_pos.torch[0, shackle_joint])

            def tip(self):
                pos, quat = self.body_pose()
                return pos + rotate(quat, PV.short_tip(self.travel()))

            def bend_top(self):
                pos, quat = self.body_pose()
                return pos + rotate(quat, (-PV.leg_x * 0.0, 0.0, PV.bend_top(self.travel())))

            def lock_center_for_tip(self, tip_target):
                return np.asarray(tip_target) - rotate(LQ, PV.short_tip(self.travel()))

            def lock_to(self, center, speed):
                self.L.move(self.model.grasp_for(center, LQ)[0], speed=speed)

            # Arm B: the toggle ---------------------------------------------------------------------
            def plan_toggle(self):
                R = self.R
                R.grip = CLOSED
                d = finger_dir(T_PITCH)
                tip0 = np.array([props.PIVOT_X - 0.020, Y0, props.ZC + 0.025])
                R.move(np.array([0.18, -0.16, 0.30]), QT, min_steps=hold_k(1.5))
                R.move(tip0 - FINGER_END * d - 0.04 * d, QT, min_steps=hold_k(1.5))
                R.move(tip0 - FINGER_END * d, speed=0.04)
                R.hold(hold_k(0.3))
                self.when_settled(R, "right", lambda: self.push_toggle(0))

            def push_toggle(self, k):
                q = self.toggle_deg()
                log(f"TOGGLE {k}: {q:.1f} deg (ON +30, OFF -30)")
                R = self.R
                if q < -25.0 or k > 6:
                    R.move(R.pos + ROOT_OFFSET + np.array([-0.03, 0.0, 0.03]), speed=0.05)
                    if STOP_AFTER == "toggle":
                        self.finish()
                        return
                    self.plan_cover()
                    return
                d = finger_dir(T_PITCH)
                tip = np.array([props.PIVOT_X - 0.020, Y0, props.ZC - 0.006 - 0.004 * k])
                R.move(tip - FINGER_END * d, speed=0.02)
                R.hold(hold_k(0.4))
                self.at(len(R.steps), lambda: self.push_toggle(k + 1))

            # Arm B: shut the cover -----------------------------------------------------------------
            def plan_cover(self):
                R = self.R
                d = finger_dir(C_PITCH)
                p = self.pinch_world()
                log(f"COVER at {self.cover_deg():.1f} deg, pinch point {np.round(p, 4)}")
                R.grip_to(OPEN, hold_k(0.4))
                R.move(p - 0.06 * d + np.array([0.0, 0.0, 0.01]), QC, min_steps=hold_k(1.5))
                R.move(p - 0.06 * d, QC, min_steps=hold_k(0.5))
                R.hold(hold_k(0.3))
                self.when_settled(R, "right", self.cover_approach)

            def cover_approach(self):
                R = self.R
                p = self.pinch_world()
                err = grasp_point("right")[0] - (R.pos + ROOT_OFFSET)
                self.corr = np.clip(np.array([0.0, -err[1], -err[2]]), -0.01, 0.01)
                R.move(p + self.corr, speed=0.03)
                R.hold(hold_k(0.3))
                self.when_settled(R, "right", self.cover_pinch, tol=0.002)

            def cover_pinch(self):
                err = grasp_point("right")[0] - self.pinch_world()
                log(f"PINCH: grasp point vs pinch point (mm) {np.round(err * 1000, 1)}")
                self.R.grip_to(CLOSED, hold_k(0.6))
                self.at(len(self.R.steps), lambda: self.cover_pull(0))

            def cover_pull(self, k):
                """Pull the pinch point round the hinge, a few degrees at a time, past the balance point."""
                q = self.cover_deg()
                if k % 3 == 0:
                    log(f"PULL {k}: cover {q:.1f} deg")
                R = self.R
                if q < PULL_TO_DEG or k > 40:
                    log(f"PULLED: cover {q:.1f} deg")
                    R.grip_to(OPEN, hold_k(0.3))
                    d = finger_dir(C_PITCH)
                    R.move(R.pos + ROOT_OFFSET - 0.045 * d, speed=0.12)
                    R.move(R.pos + ROOT_OFFSET + np.array([-0.02, -0.10, -0.04]), speed=0.15)
                    R.move(R_WAIT, speed=0.10)
                    self.sync()
                    self.when(R, lambda: self.cover_deg() < 2.0, self.cover_shut, tries=hold_k(6.0))
                    return
                hx, _, hz = props.hinge(TARGET_SLOT)
                p = grasp_point("right")[0]
                ang = math.atan2(p[0] - hx, p[2] - hz) - math.radians(4.0)
                r = math.hypot(p[0] - hx, p[2] - hz)
                tgt = np.array([hx + r * math.sin(ang), p[1], hz + r * math.cos(ang)])
                R.move(tgt + (R.pos + ROOT_OFFSET - p) * 0.5, speed=0.02, min_steps=hold_k(0.2))
                self.at(len(R.steps), lambda: self.cover_pull(k + 1))

            def cover_shut(self):
                log(f"COVER SHUT: {self.cover_deg():.1f} deg, toggle {self.toggle_deg():.1f} deg")
                R = self.R
                R.move(self.r_home[0], self.r_home[1], speed=0.08)
                self.at(len(R.steps) - 1, reseed_ready)
                self.sync()
                if control:
                    R.hold(hold_k(1.0))
                    self.sync()
                    self.at(len(R.steps), lambda: task.force_function_test(view))
                elif STOP_AFTER == "cover":
                    self.finish()
                else:
                    self.plan_pick()

            # Arm A: the padlock -----------------------------------------------------------------------
            def plan_pick(self):
                start = np.array(lock_start_pose()[:3])
                grasp, _ = GraspModel.nominal(QL, LQ, GRASP_IN_LOCK).grasp_for(start, LQ)
                L = self.L
                L.grip_to(OPEN, hold_k(0.2))
                L.move(np.array([0.14, 0.20, 0.24]), QL, min_steps=hold_k(1.5))
                L.move(grasp - 0.07 * L_FINGER_DIR + np.array([0.0, 0.0, 0.02]), QL, min_steps=hold_k(2.0))
                L.move(grasp - 0.07 * L_FINGER_DIR, QL, min_steps=hold_k(0.8))
                L.move(grasp - 0.005 * L_FINGER_DIR, speed=0.04)
                L.hold(hold_k(0.4))
                self.when_settled(L, "left", self.pick_final)

            def pick_final(self):
                start = np.array(lock_start_pose()[:3])
                grasp, _ = GraspModel.nominal(QL, LQ, GRASP_IN_LOCK).grasp_for(start, LQ)
                err = grasp_point("left")[0] - (self.L.pos + ROOT_OFFSET)
                log(f"PICK: grasp-point error (mm) {np.round(err * 1000, 1)}, padlock at {np.round(self.body_pose()[0], 4)}")
                self.L.move(grasp - err, speed=0.02)
                self.L.hold(hold_k(0.4))
                self.L.grip_to(CLOSED, hold_k(1.0))
                self.L.move(grasp - err + np.array([0.0, 0.0, 0.03]), speed=0.03)
                self.when_settled(self.L, "left", self.carry)

            def carry(self):
                self.model = GraspModel.measure(robot, "left", lock)
                log("LOCK grasp rel_pos (mm)", np.round(self.model.rel_pos * 1000, 1), "padlock quat",
                    np.round(self.body_pose()[1], 3))
                self.tip_above = np.array([HOLE[0], HOLE[1], props.RING_Z[1] + TIP_ABOVE_RING])
                self.tip_front = self.tip_above - np.array([FRONT_DX, 0.0, 0.0])
                c_front = self.lock_center_for_tip(self.tip_front)
                self.lock_to(c_front + np.array([0.0, 0.0, 0.01]), 0.05)
                self.lock_to(c_front, 0.03)
                self.L.hold(hold_k(0.6))
                self.when_settled(self.L, "left", lambda: self.correct("FRONT", self.tip_front, np.zeros(3), 0,
                                                                       self.go_in))

            def correct(self, name, target, correction, attempt, then, tol=0.0015, tries=4):
                err = self.tip() - target
                log(f"{name} attempt {attempt}: tip error (mm) {np.round(err * 1000, 1)}")
                correction = correction + err
                if attempt < tries and np.abs(err).max() > tol:
                    self.lock_to(self.lock_center_for_tip(target - correction), 0.01)
                    self.L.hold(hold_k(0.6))
                    self.when_settled(self.L, "left", lambda: self.correct(name, target, correction, attempt + 1, then,
                                                                           tol, tries))
                    return
                self.correction = correction
                then()

            def go_in(self):
                # In along +x: the ring slides into the gap between the short leg's tip and the body's top.
                self.lock_to(self.lock_center_for_tip(self.tip_above - self.correction), 0.01)
                self.L.hold(hold_k(0.8))
                self.when_settled(self.L, "left", lambda: self.correct("ABOVE HOLE", self.tip_above, self.correction,
                                                                       0, self.go_down, tol=0.0008, tries=5))

            def go_down(self):
                self.tip_low = self.tip_above - np.array([0.0, 0.0, DROP_DZ])
                self.lock_to(self.lock_center_for_tip(self.tip_low - self.correction), 0.006)
                self.L.hold(hold_k(0.8))
                self.when_settled(self.L, "left", self.in_ring)

            def in_ring(self):
                through = bool(view._task_state.debug["through"][0])
                log(f"IN RING: tip error (mm) {np.round((self.tip() - self.tip_low) * 1000, 1)}, through {through}, "
                    f"cover {self.cover_deg():.1f} deg")
                self.sync()
                self.press_start()

            # Arm B: shut the shackle ---------------------------------------------------------------
            def press_start(self):
                """Arm B, fingers closed and pitched down, comes down on top of the shackle's bend."""
                R = self.R
                d = finger_dir(P_PITCH)
                b = self.bend_top()
                self.bend0 = b.copy()
                start = b - FINGER_END * d + np.array([0.0, 0.0, 0.012])
                R.grip = CLOSED
                R.move(np.array([0.17, -0.12, 0.28]), QP, min_steps=hold_k(2.0))
                R.move(start - 0.04 * d + np.array([0.0, 0.0, 0.02]), QP, min_steps=hold_k(1.5))
                R.move(start, speed=0.03)
                R.hold(hold_k(0.4))
                self.body0 = self.body_pose()[0]
                self.sync()
                self.when_settled(R, "right", lambda: self.press(0))

            def press(self, k):
                latched = bool(view._task_state.latched[0])
                if latched or k > PRESS_MAX:
                    log(f"PRESS done after {k}: travel {self.travel() * 1000:.1f} mm, latched {latched}, "
                        f"through {bool(view._task_state.debug['through'][0])}")
                    self.L.hold(hold_k(0.3))
                    self.R.move(self.R.pos + ROOT_OFFSET + np.array([-0.02, 0.0, 0.03]), speed=0.03)
                    self.sync()
                    self.release()
                    return
                if k % 3 == 0:
                    log(f"PRESS {k}: travel {self.travel() * 1000:.1f} mm, bend {np.round(self.bend_top(), 4)}, arm B "
                        f"{np.round(grasp_point('right')[0], 4)}, body {np.round(self.body_pose()[0], 4)}")
                d = finger_dir(P_PITCH)
                b = self.bend_top()
                # Arm B pushes the bend down toward the body, its fingertip kept on the bend's top.
                target_tip = np.array([b[0], b[1], self.bend0[2] - 0.012 - PRESS_STEP * (k + 1) * 1.5])
                tgt = target_tip - FINGER_END * d
                self.R.move(tgt, speed=0.02)
                self.R.hold(hold_k(0.15))
                # Arm A holds the body where it was (integral action against its soft gains).
                body = self.body_pose()[0]
                corr = 0.6 * (self.body0 - body)
                self.L.move(self.L.pos + ROOT_OFFSET + corr, speed=0.02)
                self.sync()
                self.at(len(self.R.steps), lambda: self.press(k + 1))

            def release(self):
                R, L = self.R, self.L
                R.move(R_WAIT, speed=0.10)
                L.grip_to(OPEN, hold_k(1.5))
                L.move(L.pos + ROOT_OFFSET + np.array([0.0, 0.0, -0.006]), speed=0.01)
                L.move(L.pos + ROOT_OFFSET - 0.06 * L_FINGER_DIR, speed=0.015)
                L.move(L_WAIT, QL, speed=0.08)
                self.sync()
                R.hold(hold_k(4.0))
                L.hold(hold_k(4.0))

            def finish(self):
                R = self.R
                R.move(R_WAIT, speed=0.08)
                self.sync()
                R.hold(hold_k(2.0))
                self.at(len(R.steps), lambda: log(f"STOP ({STOP_AFTER}): {task.episode_report(view, 0)}"))

        l_pos, l_quat = grasp_point("left")
        r_pos, r_quat = grasp_point("right")
        L = ArmTimeline(l_pos - ROOT_OFFSET, l_quat, dt, speed)
        R = ArmTimeline(r_pos - ROOT_OFFSET, r_quat, dt, speed)
        planner = Planner(L, R)
        planner.r_home = (r_pos.copy(), r_quat.copy())

        def reseed_ready():
            """Arm B is back at its start pose: restart its IK from the READY joints, so its next moves take the same
            joint branch as from the start."""
            from yam_arena.constants import READY

            q_arm["right"] = np.array(READY, dtype=float)
        settle = hold_k(1.0)
        L.hold(settle)
        R.hold(settle)
        if START_AT == "pick":
            import torch

            jq = torch.tensor([[-math.radians(props.THROW_DEG), 0.0]], device=sw.data.joint_pos.torch.device)
            ids = [toggle_joint, cover_joint]
            sw.write_joint_state_to_sim(jq, torch.zeros_like(jq), joint_ids=ids)
            sw.set_joint_position_target(jq, joint_ids=ids)
            planner.at(settle, planner.plan_pick)
        else:
            planner.at(settle, planner.plan_toggle)
        log(f"control {control}, stop after {STOP_AFTER or '-'}, start at {START_AT or '-'}")

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
