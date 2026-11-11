"""The scripted demo, closed loop.

Arm A (left) slides its closed fingers under the hem of AC-1's cover at the cover's left end and lifts: it follows
the cover's measured angle, pushing just behind the hem, until the cover stands out at about 72 degrees, and holds it
there on its fingers. Arm B (right), fingers forward and turned so they close up and down, grips the pull-out's grip
bar, pulls the block straight out of the clips and clear of the box, turns its wrist half a turn (the block turns
upside down, OFF to the top), lines the block up with the window from its measured pose and pushes it back in, then
lets go and backs off. Arm A lowers the cover until it rests shut, then picks the open padlock off its pedestal
(fingers pitched down, gripping the body's lower rear corner), carries it so the tab's end passes between the short
leg's tip and the body, lines the tip up with the tab's hole from the measured tip position and lowers it through;
arm B, fingers closed, pushes the shackle's bend down until it latches while arm A holds the body; both let go.
Grasp-point targets become yam_arena's 14-D joint action through yam_arena's own IK every control step.

It reads the simulator's state (the cover's angle, the block's, padlock's poses), like any scripted demo with
privileged state; ``--policy pullout-demo`` (pullout_sim.policy) plays it through Inspect Robots. Import after Isaac
Sim has started. Under PULLOUT_CONTROL=1 it skips the padlock (the control run).
"""

from __future__ import annotations

import math
import os

import numpy as np

from yam_arena import ik, station


class Demo:
    """``next_action()`` gives the 14-D joint action for the next control step."""

    def __init__(self, STATION, control_hz: float = 20.0, speed: float = 0.10, log=print):
        from pullout_sim import spec as S
        from pullout_sim import station as hb
        from pullout_sim.kit import CLOSED, FORWARD, OPEN, ROOT_OFFSET, ArmTimeline, GraspModel, q_axis, qmul, rotate
        from pullout_sim.kit import grasp_point as grasp_point_world
        from pullout_sim.task import PV, lock_start_pose

        view = STATION.env
        task = STATION.task
        state_k = [0]
        self.state_k = state_k
        control = os.environ.get("PULLOUT_CONTROL") == "1"
        stop_at = os.environ.get("PULLOUT_STOP", "")
        origin = view.scene.env_origins[0].cpu().numpy()
        robot = view.scene["robot"]
        st = task.state(view)
        box, block, lock = st.box, st.block, st.lock
        cover_joint = st.cover_joint
        body_id = lock.find_bodies("body")[0][0]
        shackle_joint = lock.find_joints("shackle_joint")[0][0]
        dt = 1.0 / control_hz
        hold_k = lambda s: max(1, round(s / dt))  # noqa: E731
        EZ = np.array([0.0, 0.0, 1.0])
        EX = np.array([1.0, 0.0, 0.0])
        Y0 = S.BOX_Y[0]

        def grasp_point(side):
            p, q = grasp_point_world(robot, side)
            return p - origin, q

        def fwd(pitch_deg, roll_deg=0.0):
            """Fingers along +x pitched down by pitch_deg, then turned roll_deg about the finger axis."""
            q = qmul(q_axis((0, 1, 0), math.radians(pitch_deg)), FORWARD)
            axis = rotate(q, (0.0, 0.0, -1.0))
            return qmul(q_axis(axis, math.radians(roll_deg)), q)

        def fdir(pitch_deg):
            a = math.radians(pitch_deg)
            return np.array([math.cos(a), 0.0, -math.sin(a)])

        def cover_angle():
            return float(box.data.joint_pos.torch[0, cover_joint])

        def on_cover(p0, ang):
            """A point of the cover (given where it is with the cover shut) at opening angle ang."""
            dx, dz = p0[0] - S.HINGE_X, p0[2] - S.HINGE_Z
            c, s = math.cos(ang), math.sin(ang)
            return np.array([S.HINGE_X + c * dx + s * dz, p0[1], S.HINGE_Z - s * dx + c * dz])

        def block_pose():
            return (block.data.root_pos_w.torch[0].cpu().numpy() - origin,
                    block.data.root_quat_w.torch[0].cpu().numpy())

        def body_pose():
            pos = lock.data.body_pos_w.torch[0, body_id].cpu().numpy() - origin
            return pos, lock.data.body_quat_w.torch[0, body_id].cpu().numpy()

        def travel():
            return float(lock.data.joint_pos.torch[0, shackle_joint])

        def lock_tip():
            pos, quat = body_pose()
            return pos + rotate(quat, PV.short_tip(travel()))

        def bend_top():
            pos, quat = body_pose()
            return pos + rotate(quat, (0.0, 0.0, PV.bend_top(travel())))

        # Arm A under the cover's hem, at the cover's left end.
        QC = fwd(8.0, 90.0)
        """Fingers forward, turned so they close up and down: the closed fingers are a 4 cm wide, 2 cm thick paddle."""
        HEM_Y = Y0 + 0.060
        HEM0 = np.array([S.COVER_FRONT + S.SHEET + 0.004, HEM_Y, S.COVER_BOTTOM])
        """The hem's inner part, cover shut; arm A's fingers push just under/behind it."""
        HEM_DX = 0.002
        """Arm A's grasp point this far toward the robot from the hem (the fingertips reach about 2 cm past it, and
        must stay clear of the body's bottom)."""
        COVER_HOLD = math.radians(92.0)
        """Held up just short of where the hood meets the wall (94 deg), so arm A's hand is well above arm B's turning wrist."""
        # Arm B on the block's grip bar.
        GRASP_IN_BLOCK = (-S.BLOCK_SIZE[0] / 2 - 0.013, -0.008, 0.0)
        """On the grip bar's right half (arm A's hand is under the cover's left end), the fingertips (12 mm past the grasp point) just short of the block's face."""
        Q_ON = np.array([0.0, 0.0, 0.0, 1.0])
        Q_OFF = np.array([1.0, 0.0, 0.0, 0.0])
        SEAT = np.array(S.block_seated_pose(0)[:3])
        BLOCK_OPEN = 0.45
        # Padlock.
        QA, FA = fwd(10), fdir(10)
        QP, FP = fwd(60), fdir(60)
        LOCK_QUAT = np.array([0.0, 0.0, 0.0, 1.0])
        GRASP_IN_LOCK = (-PV.body[0] / 2 + 0.010, 0.0, -PV.body[2] / 2 + 0.010)
        hx, hy = S.TAB_HOLE_X, Y0
        TAB_TOP = S.TAB_Z[1]
        FINGER_END = 0.012
        R_WAIT = np.array([0.18, -0.20, 0.22])
        L_WAIT = np.array([0.16, 0.20, 0.22])

        class Planner:
            def __init__(self, L, R):
                self.L, self.R, self.events = L, R, {}
                self.hz = 0.010
                self.press = 0.015
                """While lifting, aim this far above the contact (the soft arm then pushes up with a few newtons)."""

            def at(self, step, fn):
                self.events.setdefault(step, []).append(fn)

            def run_due(self, k):
                for fn in self.events.pop(k, []):
                    fn()

            def stop(self, name):
                if stop_at == name:
                    log(f"STOP at {name}")
                    return True
                return False

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

            # Arm A: lift the cover ------------------------------------------------------------------------
            def cover_start(self):
                L = self.L
                L.grip_to(CLOSED, hold_k(0.3))
                below = HEM0 + np.array([HEM_DX, 0.0, -0.035])
                L.move(np.array([0.26, HEM_Y + 0.05, 0.25]), QC, min_steps=hold_k(2.0))
                L.move(np.array([0.24, HEM_Y, 0.12]), speed=0.08)
                L.move(below - 0.05 * EX, speed=0.06)
                L.move(below, speed=0.03)
                L.hold(hold_k(0.3))
                self.when_settled(L, "left", self._cover_probe)

            def _cover_probe(self, k=0):
                """Rise slowly until the cover starts to turn: that height sets the fingers' half thickness."""
                ang = cover_angle()
                g = grasp_point("left")[0]
                if k % 5 == 0:
                    log(f"PROBE k={k} step={state_k[0]} grasp {np.round(g, 4)} target {np.round(self.L.pos + ROOT_OFFSET, 4)} "
                        f"angle {math.degrees(ang):.2f}")
                lag = float((self.L.pos + ROOT_OFFSET)[2] - g[2])
                if ang > math.radians(0.8) or lag > 0.005 or k > 40:
                    self.hz = float(HEM0[2] - g[2])
                    log(f"COVER contact at k={k}: angle {math.degrees(ang):.1f} deg, finger half thickness "
                        f"{self.hz * 1000:.1f} mm")
                    self._cover_lift()
                    return
                self.L.move(self.L.pos + ROOT_OFFSET + 0.002 * EZ, speed=0.02)
                self.at(len(self.L.steps), lambda: self._cover_probe(k + 1))

            def _cover_target(self, ang):
                return on_cover(HEM0, ang) + np.array([HEM_DX, 0.0, -self.hz + self.press])

            def _cover_lift(self, k=0):
                ang = cover_angle()
                if ang >= COVER_HOLD - math.radians(1.0) or k > 120:
                    log(f"COVER held at {math.degrees(ang):.1f} deg (k={k})")
                    self.L.hold(hold_k(0.5))
                    if self.stop("cover_open"):
                        return
                    self.at(len(self.L.steps), self.block_start)
                    return
                if k % 4 == 0:
                    log(f"LIFT k={k} step={state_k[0]} angle {math.degrees(ang):.1f} grasp {np.round(grasp_point('left')[0], 4)} "
                        f"target {np.round(self.L.pos + ROOT_OFFSET, 4)}")
                self.cmd = min(COVER_HOLD, ang + math.radians(8.0))
                target = self._cover_target(self.cmd)
                self.L.move(target, speed=0.04, min_steps=2)
                self.at(len(self.L.steps), lambda: self._cover_lift(k + 1))

            # Arm B: the block -----------------------------------------------------------------------------
            def block_start(self):
                R = self.R
                self.sync()
                bp, bq = block_pose()
                q0 = fwd(0.0, -90.0)
                self.bgrasp, _ = GraspModel.nominal(q0, bq, GRASP_IN_BLOCK).grasp_for(bp, bq)
                self.q_pull = q0
                R.grip_to(BLOCK_OPEN, hold_k(0.3))
                R.move(np.array([0.22, Y0 - 0.08, 0.20]), q0, min_steps=hold_k(2.0))
                R.move(self.bgrasp - 0.06 * EX, speed=0.06)
                R.move(self.bgrasp - 0.004 * EX, speed=0.03)
                R.hold(hold_k(0.3))
                self.when_settled(R, "right", self._block_close)

            def _block_close(self):
                R = self.R
                err = grasp_point("right")[0] - (R.pos + ROOT_OFFSET)
                g = self.bgrasp - 0.004 * EX - err
                R.move(g, speed=0.02)
                R.grip_to(CLOSED, hold_k(1.0))
                self.at(len(R.steps), self._block_pull)

            def _block_pull(self):
                R = self.R
                self.bmodel = GraspModel.measure(robot, "right", block)
                log("BLOCK grasp rel_pos (mm)", np.round(self.bmodel.rel_pos * 1000, 1))
                R.move(R.pos + ROOT_OFFSET - 0.035 * EX, speed=0.015)
                R.move(R.pos + ROOT_OFFSET - 0.055 * EX, speed=0.04)
                R.hold(hold_k(0.4))
                self.at(len(R.steps), self._block_turn)

            def _block_turn(self):
                bp, bq = block_pose()
                log(f"BLOCK out: pos {np.round(bp, 4)} quat {np.round(bq, 3)} level "
                    f"{task.episode_report(view, 0)['milestone']}")
                if self.stop("pulled"):
                    return
                R = self.R
                for roll in (-60, -30, 0, 30, 60, 90):
                    R.move(R.pos + ROOT_OFFSET, fwd(0.0, roll), min_steps=hold_k(0.5))
                R.hold(hold_k(0.5))
                self.when_settled(R, "right", self._block_align)

            def _block_target(self, pos):
                """Grasp point that puts the block at pos, OFF, from the measured grasp."""
                model = GraspModel.measure(robot, "right", block)
                g, q = model.grasp_for(pos, Q_OFF)
                return g, q

            def _block_align(self, attempt=0):
                bp, bq = block_pose()
                pre = SEAT - 0.045 * EX  # blades' tips 7 mm short of the dead front
                err = bp - pre
                log(f"BLOCK align {attempt}: pos {np.round(bp, 4)} quat {np.round(bq, 3)} err (mm) "
                    f"{np.round(err * 1000, 1)} q_right {np.round(view.scene['right_robot'].data.joint_pos.torch[0, :6].cpu().numpy(), 2)}")
                if attempt == 0:
                    self.bias = np.zeros(3)
                elif np.abs(err).max() <= 0.0012 or attempt >= 8:
                    self._block_insert()
                    return
                else:
                    # The soft arm sags short of its target: aim past it by the error left (integral correction).
                    self.bias += err
                g, q = self._block_target(pre)
                self.R.move(g - self.bias, q, speed=0.02, min_steps=hold_k(0.4))
                self.R.hold(hold_k(0.4))
                self.when_settled(self.R, "right", lambda: self._block_align(attempt + 1), tries=10, tol=0.004)

            def _block_insert(self, k=0):
                bp, _ = block_pose()
                depth = bp[0] - SEAT[0]
                if depth > -0.0015 or k > 30:
                    log(f"BLOCK seated (k={k}): pos {np.round(bp, 4)} depth {depth * 1000:.1f} mm, "
                        f"level {task.episode_report(view, 0)['milestone']}")
                    self._block_release()
                    return
                lat = bp - np.array([bp[0], SEAT[1], SEAT[2]])
                self.bias += 0.5 * np.array([0.0, lat[1], lat[2]])
                # Push past the seat by more each step while the block lags (the soft arm sags short of its target).
                self.push = min(0.025, (getattr(self, "push", 0.004) + 0.002) if k else 0.006)
                g, q = self._block_target(np.array([min(bp[0] + self.push, SEAT[0] + 0.015), SEAT[1], SEAT[2]]))
                self.R.move(g - self.bias, q, speed=0.01, min_steps=hold_k(0.2))
                self.at(len(self.R.steps), lambda: self._block_insert(k + 1))

            def _block_release(self):
                R = self.R
                R.grip_to(BLOCK_OPEN, hold_k(0.6))
                R.move(R.pos + ROOT_OFFSET - 0.06 * EX, speed=0.04)
                R.move(R_WAIT, fwd(20.0), speed=0.08)
                R.hold(hold_k(0.6))
                self.at(len(R.steps), self._block_done)

            def _block_done(self):
                bp, bq = block_pose()
                log(f"BLOCK done: pos {np.round(bp, 4)} quat {np.round(bq, 3)} level {task.episode_report(view, 0)['milestone']}")
                if self.stop("block"):
                    return
                self.sync()
                self._cover_lower()

            # Arm A: lower the cover ------------------------------------------------------------------------
            def _cover_lower(self, k=0):
                self.press = 0.002
                ang = cover_angle()
                if ang <= math.radians(8.0) or k > 120:
                    L = self.L
                    L.move(L.pos + ROOT_OFFSET - 0.04 * EZ - 0.02 * EX, speed=0.03)
                    L.move(L.pos + ROOT_OFFSET - 0.05 * EX, speed=0.05)
                    L.hold(hold_k(1.0))
                    self.at(len(L.steps), self._cover_shut)
                    return
                target = self._cover_target(max(0.0, ang - math.radians(6.0)))
                self.L.move(target, speed=0.03, min_steps=2)
                self.at(len(self.L.steps), lambda: self._cover_lower(k + 1))

            def _cover_shut(self):
                log(f"COVER lowered: angle {math.degrees(cover_angle()):.1f} deg, level {task.episode_report(view, 0)['milestone']}")
                if self.stop("cover_shut"):
                    return
                if control:
                    L = self.L
                    L.move(L_WAIT, speed=0.08)
                    L.hold(hold_k(6.0))
                    self.R.pad_to(len(L.steps))
                    return
                self.lock_pick()

            # Arm A: the padlock -----------------------------------------------------------------------------
            def lock_pick(self):
                start = np.array(lock_start_pose()[:3])
                self.lgrasp, _ = GraspModel.nominal(QA, LOCK_QUAT, GRASP_IN_LOCK).grasp_for(start, LOCK_QUAT)
                L = self.L
                L.grip_to(OPEN, hold_k(0.2))
                pre = self.lgrasp - 0.04 * FA
                L.move(np.array([0.24, 0.12, 0.27]), QA, speed=0.08, min_steps=hold_k(1.0))
                L.move(np.array([pre[0], pre[1], 0.27]), speed=0.08)
                L.move(pre, speed=0.05)
                L.hold(hold_k(0.4))
                L.move(self.lgrasp - 0.015 * FA, speed=0.04)
                L.hold(hold_k(0.4))
                self.when_settled(L, "left", self._lock_close)

            def _lock_close(self):
                err = grasp_point("left")[0] - (self.L.pos + ROOT_OFFSET)
                g = self.lgrasp - err
                self.L.move(g, speed=0.02)
                self.L.hold(hold_k(0.4))
                self.L.grip_to(CLOSED, hold_k(1.0))
                self.L.move(g + 0.035 * EZ, speed=0.03)
                self.when_settled(self.L, "left", self._lock_carry)

            def _lock_carry(self):
                self.lmodel = GraspModel.measure(robot, "left", lock)
                log("LOCK grasp rel_pos (mm)", np.round(self.lmodel.rel_pos * 1000, 1))
                self.tip_entry = np.array([hx - 0.024, hy, TAB_TOP + 0.0045])
                d = self.tip_entry - lock_tip()
                self.L.move(self.L.pos + ROOT_OFFSET + d - 0.03 * EX + 0.03 * EZ, speed=0.06)
                self.L.move(self.L.pos + ROOT_OFFSET + 0.03 * EX - 0.03 * EZ, speed=0.03)
                self.L.hold(hold_k(0.6))
                self.when_settled(self.L, "left", lambda: self._lock_correct(self.tip_entry, 0, self._lock_over, 0.0015))

            def _lock_correct(self, tip_target, attempt, then, tol):
                err = lock_tip() - tip_target
                log(f"LOCK attempt {attempt}: tip error (mm) {np.round(err * 1000, 1)}")
                if attempt < 5 and np.abs(err).max() > tol:
                    self.L.move(self.L.pos + ROOT_OFFSET - err, speed=0.01)
                    self.L.hold(hold_k(0.6))
                    self.when_settled(self.L, "left", lambda: self._lock_correct(tip_target, attempt + 1, then, tol))
                    return
                then()

            def _lock_over(self):
                self.tip_over = np.array([hx, hy, TAB_TOP + 0.0045])
                self.L.move(self.L.pos + ROOT_OFFSET + (self.tip_over - lock_tip()), speed=0.008)
                self.L.hold(hold_k(0.6))
                self.when_settled(self.L, "left", lambda: self._lock_correct(self.tip_over, 0, self._lock_down, 0.001))

            def _lock_down(self):
                self.L.move(self.L.pos + ROOT_OFFSET - 0.014 * EZ, speed=0.006)
                self.L.hold(hold_k(0.8))
                self.when_settled(self.L, "left", self._threaded)

            def _threaded(self):
                log(f"THREADED: tip {np.round(lock_tip(), 4)} hole ({hx:.4f}, {hy:.4f}) "
                    f"level {task.episode_report(view, 0)['milestone']}")
                if self.stop("threaded"):
                    return
                self._press_start()

            # Arm B presses the shackle shut ---------------------------------------------------------------
            def _press_start(self):
                R = self.R
                self.sync()
                R.grip_to(CLOSED, hold_k(0.5))
                b = bend_top()
                start = b + 0.004 * EZ - FINGER_END * FP
                R.move(start - 0.04 * FP + 0.015 * EZ, QP, speed=0.08)
                R.move(start, speed=0.03)
                R.hold(hold_k(0.6))
                self.body0 = body_pose()[0]
                self.when_settled(R, "right", lambda: self._press(0))

            def _press(self, k):
                latched = bool(st.latched[0])
                if latched or k > 30:
                    log(f"PRESS done after {k}: travel {travel() * 1000:.1f} mm, latched {latched}")
                    self.R.move(self.R.pos + ROOT_OFFSET + 0.004 * EZ, speed=0.02)
                    self.R.hold(hold_k(0.4))
                    self.L.pad_to(len(self.R.steps))
                    self.release()
                    return
                drift = float((body_pose()[0] - self.body0) @ EZ)
                self.L.pad_to(len(self.R.steps))
                self.L.move(self.L.pos + ROOT_OFFSET - drift * EZ, speed=0.02)
                self.R.move(self.R.pos + ROOT_OFFSET - 0.004 * EZ, speed=0.02)
                self.R.hold(hold_k(0.2))
                self.sync()
                self.at(len(self.R.steps), lambda: self._press(k + 1))

            def release(self):
                R, L = self.R, self.L
                R.move(R.pos + ROOT_OFFSET + 0.04 * EZ - 0.03 * EX, speed=0.04)
                R.move(R_WAIT, speed=0.10)
                L.pad_to(len(R.steps) - hold_k(1.2))
                L.grip_to(0.55, hold_k(1.0))
                L.move(L.pos + ROOT_OFFSET - 0.05 * FA, speed=0.03)
                L.move(L_WAIT, speed=0.06)
                L.hold(hold_k(4.0))

        l_pos, l_quat = grasp_point("left")
        r_pos, r_quat = grasp_point("right")
        L = ArmTimeline(l_pos - ROOT_OFFSET, l_quat, dt, speed)
        R = ArmTimeline(r_pos - ROOT_OFFSET, r_quat, dt, speed)
        planner = Planner(L, R)
        settle = hold_k(1.0)
        L.hold(settle)
        R.hold(settle)
        planner.at(settle, planner.cover_start)

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
        L.pad_to(k + 1)
        R.pad_to(k + 1)
        (lp, lq, lg), (rp, rq, rg) = L.steps[k], R.steps[k]
        self.state_k[0] = k + 1
        return np.concatenate([self._joint_action("left", lp, lq, lg), self._joint_action("right", rp, rq, rg)])
