"""The scripted demo, closed loop. Arm A (left) brings the tagged padlock to the handle; arm B (right) takes it over to
swing it down and snap it shut.

1. Arm A grips the open padlock's body (fingers level, pointing at the switch, closing on the body's 38 mm width) and
   lifts it off its column: legs along +x, long leg up, short leg down.
2. Tag: arm A lowers the lock over the standing tag so the tag's top edge goes into the gap between the body's top
   and the short leg's tip, lines the short leg up with the grommet (corrected from the measured tip and grommet),
   and draws the lock toward the robot: the tip passes through the grommet. Lifting the lock lifts the tag out of
   its pocket, hanging on the short leg; a slight tilt (bend lower) slides the grommet toward the bend.
3. Arm A rolls the lock a quarter turn about its legs (long leg to the right), carries it under the handle's lock-off
   plates, raises it so the plates go into the gap between the body's top and the short leg's tip (tip behind the
   ear), lines the short leg up with the holes (corrected), and draws the lock toward the robot until the shackle's
   bend is just behind the ear: the short leg has passed through the ear's and the handle's holes. Arm A then rolls
   the lock back a quarter turn about the short leg (long leg below), so the bend stands in the holes' plane.
4. Hand-over: arm B grips the body's front half from above (fingers down, closing on its 19 mm thickness); arm A
   lets go. Arm B draws the lock on until the bend is in the holes and swings it down about the bend's centre: the
   bend slides through the holes and the lock hangs from them, legs up.
5. Arm B pushes the body up along the legs until the shackle latches, then lets go and moves away.

With ``control=True`` it skips step 2 (no tag): a padlock without its tag.
Grasp-point targets become yam_arena's 14-D joint action through yam_arena's own IK every control step. Import after
Isaac Sim has started.
"""

from __future__ import annotations

import math
import os

import numpy as np

from yam_arena import ik, station

STOP = os.environ.get("LOCKTAG_STOP", "")
PULL_GAP = float(os.environ.get("LOCKTAG_PULL_GAP", "0.002"))
"""The pull stops with the shackle's bend this far behind the ear."""
SWING_DEG = 75.0
"""Arm B swings the lock this far down (its wrist reaches 75 degrees); gravity takes it the rest of the way."""


class Demo:
    """``next_action()`` gives the 14-D joint action for the next control step."""

    def __init__(self, STATION, control_hz: float = 20.0, speed: float = 0.10, control: bool = False, log=print):
        from locktag_sim import station as hb

        view = STATION.env
        task = STATION.task
        state_k = [0]
        self.state_k = state_k
        from locktag_sim.kit import CLOSED, DOWN, FORWARD, OPEN, ROOT_OFFSET, ArmTimeline, GraspModel, q_axis, qmul, rotate
        from locktag_sim.kit import grasp_point as grasp_point_world
        from locktag_sim.spec import HOLE_X, HOLE_Y, HOLE_Z
        from locktag_sim.task import LOCK_START_QUAT, PV, lock_start_pose, tag_start_pose
        from locktag_sim.usd import locktag_props as props
        from locktag_sim.usd import padlock_props as lock_props

        origin = view.scene.env_origins[0].cpu().numpy()
        robot = view.scene["robot"]

        def grasp_point(side):
            p, q = grasp_point_world(robot, side)
            return p - origin, q

        EX, EY, EZ = np.array([1.0, 0, 0]), np.array([0, 1.0, 0]), np.array([0, 0, 1.0])
        Q0 = np.array(LOCK_START_QUAT)
        """Start orientation (and for the tag): legs along +x, long leg up, short leg below, thickness along y."""
        Q0T = qmul(q_axis((0, 1, 0), math.radians(10)), Q0)
        """Tilted 10 deg bend-down: the tag's grommet slides along the short leg toward the bend."""
        TILT_INS = math.radians(float(os.environ.get("LOCKTAG_TILT_DEG", "4")))
        Q1 = qmul(q_axis((0, 1, 0), TILT_INS), qmul(q_axis((1, 0, 0), math.pi / 2), Q0))
        """At the handle: legs along +x tilted TILT_INS bend-down, long leg to the right (-y), thickness vertical."""
        Q1T = qmul(q_axis((0, 1, 0), math.radians(10)), qmul(q_axis((1, 0, 0), math.pi / 2), Q0))
        LEG_DIR = rotate(Q1, (0.0, 0.0, 1.0))
        QA0 = qmul(q_axis((1, 0, 0), -math.pi / 2), FORWARD)
        """Arm A's grasp on the lying lock: fingers level toward +x, closing vertically on the body's 38 mm width (the
        wrist then has room to roll the lock half a turn, a quarter for the handle and a quarter to let it hang)."""
        QA = np.array(FORWARD)
        """Arm A's grasp: fingers toward +x (down for the tag), closing along y on the body's 19 mm thickness."""
        GRASP_IN_LOCK = (0.0, 0.0, -0.012)
        B_GRASP_IN_LOCK = (0.0, 0.0, 0.016)
        """Arm B's hand-over grasp: the front of the body, 6 mm from its top face (arm A holds the rear half)."""
        """Grasp point 4 mm below the body's centre: a pinch grasp pivots about the finger axis, so grip near the CoM."""
        BODY_TOP = PV.body[2] / 2
        from locktag_sim.usd.padlock_410 import BEND_ZC
        TIP_Z = lambda tr: PV.short_tip(tr)[2]  # noqa: E731
        R_WAIT = np.array((0.10, -0.25, 0.30))

        lock, tag, switch = view.scene["padlock"], view.scene["tag"], view.scene["switch"]
        body_id = lock.find_bodies("body")[0][0]
        shackle_joint = lock.find_joints("shackle_joint")[0][0]
        grommet_id = tag.find_bodies("seg0")[0][0]
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

            def stop(self, name):
                if STOP == name:
                    log(f"STOP at {name}")
                    self.L.hold(hold_k(600))
                    return True
                return False

            # -- state ------------------------------------------------------------------------------
            def lock_pose(self):
                return body_pose(lock, body_id)

            def travel(self):
                return float(lock.data.joint_pos.torch[0, shackle_joint])

            def in_lock(self, p):
                pos, quat = self.lock_pose()
                return pos + rotate(quat, p)

            def tip(self):
                return self.in_lock(PV.short_tip(self.travel()))

            def grommet(self):
                return body_pose(tag, grommet_id)[0]

            def lock_center_for(self, point_in_lock, target, quat):
                return np.asarray(target) - rotate(quat, point_in_lock)

            def lock_to(self, center, quat, speed, min_steps=3, arm="L"):
                tl, model = (self.L, self.lmodel) if arm == "L" else (self.R, self.rmodel)
                g, q = model.grasp_for(center, quat)
                tl.move(g, q, speed=speed, min_steps=min_steps)

            def lock_path(self, poses, arm="L"):
                tl, model = (self.L, self.lmodel) if arm == "L" else (self.R, self.rmodel)
                tl.path([model.grasp_for(c, q) for c, q in poses])

            # -- 1. pick ------------------------------------------------------------------------------
            def plan_pick(self):
                start = np.array(lock_start_pose()[:3])
                grasp, _ = GraspModel.nominal(QA0, Q0, GRASP_IN_LOCK).grasp_for(start, Q0)
                self.grasp0 = grasp
                L = self.L
                L.move(grasp - 0.035 * EX + 0.05 * EZ, QA0, min_steps=hold_k(3.0))
                L.move(grasp - 0.035 * EX, speed=0.05)
                L.hold(hold_k(0.4))
                self.when_settled(L, "left", self._pick_in)

            def _pick_in(self):
                err = grasp_point("left")[0] - (self.L.pos + ROOT_OFFSET)
                self.L.move(self.grasp0 - err, speed=0.03)
                self.L.hold(hold_k(0.4))
                self.L.grip_to(CLOSED, hold_k(1.0))
                self.L.move(self.grasp0 - err + 0.03 * EZ, speed=0.03)
                self.when_settled(self.L, "left", self.after_pick)

            def after_pick(self):
                self.lmodel = GraspModel.measure(robot, "left", lock)
                log("LOCK grasp rel_pos (mm)", np.round(self.lmodel.rel_pos * 1000, 1),
                    "rel_quat", np.round(self.lmodel.rel_quat, 3))
                if self.stop("pick"):
                    return
                if control:
                    self.lift_tag_done()
                    return
                self.tag_start()

            # -- 2. tag -------------------------------------------------------------------------------
            def _tag_target(self, dx):
                """Lock centre that puts the short leg's axis through the grommet's centre, the body's top `dx`
                in front of the tag's mid-plane."""
                g = self.grommet()
                c = g - rotate(Q0, (PV.leg_x, 0.0, 0.0))
                return c + (g[0] - dx - (c[0] + BODY_TOP)) * EX

            def tag_start(self):
                c = self._tag_target(0.006)
                self.lock_to(c + 0.06 * EZ, Q0, 0.06, min_steps=hold_k(2.0))
                self.lock_to(c + 0.015 * EZ, Q0, 0.03)
                self.lock_to(c, Q0, 0.012)
                self.L.hold(hold_k(0.5))
                self.when_settled(self.L, "left", lambda: self._tag_correct(np.zeros(3), 0))

            def _tag_correct(self, corr, attempt):
                c = self._tag_target(0.006)
                g, tip = self.grommet(), self.tip()
                err = np.array([float(self.in_lock((0, 0, BODY_TOP))[0] - (g[0] - 0.006)), tip[1] - g[1], tip[2] - g[2]])
                log(f"TAG attempt {attempt}: body top x, tip-grommet y z errors (mm) {np.round(err * 1000, 1)}")
                corr = corr + err
                if attempt < 5 and np.abs(err).max() > 0.001:
                    self.lock_to(c - corr, Q0, 0.008)
                    self.L.hold(hold_k(0.5))
                    self.when_settled(self.L, "left", lambda: self._tag_correct(corr, attempt + 1))
                    return
                if self.stop("tag_align"):
                    return
                # Draw the lock toward the robot: the tip passes through the grommet, then on toward the bend.
                self.lock_to(c - corr - 0.012 * EX, Q0, 0.006)
                self.lock_to(c - corr - 0.034 * EX, Q0, 0.010)
                self.L.hold(hold_k(0.5))
                self.when_settled(self.L, "left", self._tag_lift)

            def _tag_lift(self):
                rep = task.episode_report(view, 0)
                log(f"TAG THREADED? on {rep['tag_on_shackle']} d {rep['tag_grommet_to_shackle_mm']} mm, "
                    f"milestone {rep['milestone']}")
                here = self.lock_pose()[0]
                # Lift the tag out of its pocket on the short leg, tilting bend-down so it slides toward the bend.
                self.lock_to(here + 0.03 * EZ, Q0, 0.015)
                self.lock_to(here + 0.11 * EZ, Q0T, 0.02, min_steps=hold_k(4.0))
                self.L.hold(hold_k(1.0))
                self.when_settled(self.L, "left", self.lift_tag_done)

            def lift_tag_done(self):
                rep = task.episode_report(view, 0)
                log(f"TAG LIFTED: on {rep['tag_on_shackle']} d {rep['tag_grommet_to_shackle_mm']} mm, grommet "
                    f"{np.round(self.grommet(), 3)}")
                if self.stop("tag"):
                    return
                self.to_handle()

            def _slerp(self, a, b, t):
                from locktag_sim.kit import slerp
                return slerp(a, b, t)

            # -- 3. handle ----------------------------------------------------------------------------
            def _hole_target(self, dz):
                """Lock centre (orientation Q1) with the short leg on the holes' axis, the body's top 2.5 mm in
                front of the handle's tab, `dz` below."""
                c = np.array([0.0, HOLE_Y, HOLE_Z]) - rotate(Q1, (PV.leg_x, 0.0, 0.0))
                c[0] = props.TAB_X[0] - 0.0025 - BODY_TOP
                return c + dz * EZ

            def to_handle(self):
                # A quarter turn about the (tilted) legs: long leg from above to the right. The tag stays on the short
                # leg, leaning on the long leg.
                q_from = Q0 if control else Q0T
                n = hold_k(2.5)
                axis_pt = self.in_lock((PV.leg_x, 0.0, 0.0))
                leg_axis = rotate(q_from, (0.0, 0.0, 1.0))
                poses = []
                for i in range(1, n + 1):
                    q = qmul(q_axis(leg_axis, math.pi / 2 * i / n), q_from)
                    poses.append((axis_pt - rotate(q, (PV.leg_x, 0.0, 0.0)), q))
                self.lock_path(poses)
                self.L.hold(hold_k(0.5))
                q_mid = poses[-1][1]
                self.when_settled(self.L, "left", lambda: self._carry(poses[-1][0], q_mid))

            def _carry(self, c_now, q_mid):
                self.lmodel = GraspModel.measure(robot, "left", lock)
                rep = task.episode_report(view, 0)
                log(f"ROLLED: tag on {rep['tag_on_shackle']} d {rep['tag_grommet_to_shackle_mm']} mm; grasp rel_quat "
                    f"{np.round(self.lmodel.rel_quat, 3)}")
                if self.stop("rolled"):
                    return
                below = self._hole_target(-0.035)
                # Up and over the tag pocket and the padlock's column, down left of the handle's plates, then in under
                # them. The tag hangs about 6 cm beyond the lock's centre (toward the switch).
                self.lock_to(np.array([0.21, c_now[1], 0.27]), q_mid, 0.03)
                self.lock_to(np.array([0.21, below[1] + 0.06, 0.27]), q_mid, 0.03)
                self.lock_to(np.array([0.21, below[1] + 0.06, below[2]]), Q1T, 0.03)
                self.lock_to(np.array([0.21, below[1], below[2]]), Q1T, 0.03)
                self.lock_to(below, Q1, 0.015, min_steps=hold_k(1.5))
                self.L.hold(hold_k(0.5))
                self.when_settled(self.L, "left", lambda: self._below_correct(np.zeros(3), 0))

            def _ref_errors(self):
                """(x error of the body's top vs its target, y and z errors of the tip vs the holes' axis)."""
                top = self.in_lock((0, 0, BODY_TOP))
                tip = self.tip()
                return np.array([top[0] - (props.TAB_X[0] - 0.0025), tip[1] - HOLE_Y, tip[2] - HOLE_Z])

            def _below_correct(self, corr, attempt):
                if attempt == 0:
                    self.lmodel = GraspModel.measure(robot, "left", lock)
                    pos, quat = self.lock_pose()
                    log("LOCK at below: rel_quat", np.round(self.lmodel.rel_quat, 3), "lock quat", np.round(quat, 3),
                        "target", np.round(Q1, 3))
                err = self._ref_errors()
                err[2] += 0.035
                log(f"BELOW attempt {attempt}: errors (mm) {np.round(err * 1000, 1)}")
                corr = corr + err
                if attempt < 4 and np.abs(err).max() > 0.0012:
                    self.lock_to(self._hole_target(-0.035) - corr, Q1, 0.01)
                    self.L.hold(hold_k(0.5))
                    self.when_settled(self.L, "left", lambda: self._below_correct(corr, attempt + 1))
                    return
                if self.stop("below"):
                    return
                self.corr = corr
                self.lock_to(self._hole_target(0.0) - corr, Q1, 0.012)
                self.L.hold(hold_k(0.6))
                self.when_settled(self.L, "left", lambda: self._up_correct(corr, 0))

            def _up_correct(self, corr, attempt):
                err = self._ref_errors()
                log(f"AT HOLES attempt {attempt}: errors (mm) {np.round(err * 1000, 1)}")
                corr = corr + err
                if attempt < 4 and np.abs(err).max() > 0.001:
                    self.lock_to(self._hole_target(0.0) - corr, Q1, 0.006)
                    self.L.hold(hold_k(0.5))
                    self.when_settled(self.L, "left", lambda: self._up_correct(corr, attempt + 1))
                    return
                if self.stop("holes"):
                    return
                self.corr = corr
                # Draw the lock toward the robot until the shackle's bend reaches the ear: the short leg passes
                # through both holes and the bend is in them, so the lock can swing down and hang when let go.
                junction = self.in_lock((PV.leg_x, 0.0, BODY_TOP + self.travel() + BEND_ZC))
                pull = (junction[0] - (props.EAR_X[1] + PV.rod / 2 + PULL_GAP)) / LEG_DIR[0]
                pull = float(np.clip(pull, 0.0, 0.05))
                log(f"PULL {pull * 1000:.1f} mm")
                self.lock_to(self._hole_target(0.0) - corr - pull * LEG_DIR, Q1, 0.006)
                self.L.hold(hold_k(0.6))
                self.when_settled(self.L, "left", self._pulled)

            def _pulled(self):
                rep = task.episode_report(view, 0)
                log(f"PULLED: holes d {rep['holes_to_shackle_mm']} mm through {rep['shackle_through_holes']}, tag on "
                    f"{rep['tag_on_shackle']}")
                if self.stop("pulled"):
                    return
                # Roll the lock back a quarter turn about the short leg (now in the holes): long leg below, so the
                # bend stands in the holes' plane and the lock swings down through them when let go.
                pos, q_from = self.lock_pose()
                axis_pt = self.in_lock((PV.leg_x, 0.0, 0.0))
                leg_axis = rotate(q_from, (0.0, 0.0, 1.0))
                n = hold_k(2.5)
                poses = []
                for i in range(1, n + 1):
                    q = qmul(q_axis(leg_axis, math.pi / 2 * i / n), q_from)
                    poses.append((axis_pt - rotate(q, (PV.leg_x, 0.0, 0.0)), q))
                self.lock_path(poses)
                self.L.hold(hold_k(0.6))
                self.when_settled(self.L, "left", self._let_go)

            def _let_go(self):
                rep = task.episode_report(view, 0)
                log(f"ROLLED BACK: holes d {rep['holes_to_shackle_mm']} mm, tag on {rep['tag_on_shackle']}, "
                    f"lock quat {np.round(self.lock_pose()[1], 3)}")
                if self.stop("rolled_back"):
                    return
                # Hand over: arm B (right) takes the body's front half from above (fingers down, closing on its
                # 19 mm thickness) while arm A still holds its rear half; then A lets go. The open lock is never
                # left on its own (it could slide back out of the holes).
                pos, quat = self.lock_pose()
                self.top_grasp = pos + rotate(quat, B_GRASP_IN_LOCK)
                R = self.R
                self.sync()
                QB = qmul(q_axis((0, 1, 0), math.radians(-10)), DOWN)
                R.move(np.array((0.16, -0.10, 0.24)), QB, speed=0.08, min_steps=hold_k(2.5))
                R.move(self.top_grasp + 0.04 * EZ, QB, speed=0.06)
                R.move(self.top_grasp + 0.02 * EZ, speed=0.02)
                R.hold(hold_k(0.4))
                self.when_settled(R, "right", self._top_in)

            def regrasp(self):
                pass

            def _top_in(self, attempt=0):
                R = self.R
                err = grasp_point("right")[0] - (R.pos + ROOT_OFFSET)
                R.move(self.top_grasp - err, speed=0.015)
                R.hold(hold_k(0.3))
                R.grip_to(CLOSED, hold_k(1.2))
                self.sync()
                self.at(len(R.steps), lambda: self._check_b(attempt))

            def _check_b(self, attempt):
                """Arm B's fingers must have stopped on the body (about 9.5 mm each from shut), not on arm A's finger
                (wider) or on nothing (shut)."""
                q = view.scene["right_robot"].data.joint_pos.torch[0, 6:8].abs().cpu().numpy()
                log(f"B GRIP attempt {attempt}: finger joints {np.round(q * 1000, 1)} mm")
                ok = 0.006 < float(q.mean()) < 0.0125
                R = self.R
                if not ok and attempt < 3:
                    R.grip_to(OPEN, hold_k(0.6))
                    up = R.pos + ROOT_OFFSET + 0.02 * EZ
                    R.move(up, speed=0.02)
                    pos, quat = self.lock_pose()
                    shift = 0.003 * (attempt + 1)
                    self.top_grasp = pos + rotate(quat, (B_GRASP_IN_LOCK[0], 0.0, B_GRASP_IN_LOCK[2] + shift)) + 0.0 * EZ
                    R.move(self.top_grasp + 0.02 * EZ, speed=0.02)
                    self.sync()
                    self.when_settled(R, "right", lambda: self._top_in(attempt + 1))
                    return
                L = self.L
                # Arm A lets go: its fingers close vertically here, so drop the lower one away first.
                L.pad_to(len(R.steps))
                L.grip_to(OPEN, hold_k(0.6))
                L.move(L.pos + ROOT_OFFSET - 0.03 * EZ, speed=0.02)
                L.move(L.pos + ROOT_OFFSET - 0.04 * EX, speed=0.03)
                L.move(np.array((0.12, 0.22, 0.28)), speed=0.06)
                self.sync()
                R.hold(hold_k(0.5))
                self.sync()
                self.when_settled(R, "right", self._swing_down)

            def _swing_down(self):
                """Swing the lock down a quarter turn about the centre of its shackle's bend, which is in the holes:
                the bend slides through them and the lock ends hanging from them, legs up."""
                self.rmodel = GraspModel.measure(robot, "right", lock)
                pos, quat = self.lock_pose()
                rep = task.episode_report(view, 0)
                log(f"HANDED OVER: lock {np.round(pos, 4)} holes d {rep['holes_to_shackle_mm']} tag on "
                    f"{rep['tag_on_shackle']}")
                # First draw the lock on until the bend starts in the middle of the plate stack.
                junction = self.in_lock((PV.leg_x, 0.0, BODY_TOP + self.travel() + BEND_ZC))
                legs = rotate(quat, (0.0, 0.0, 1.0))
                extra = float(np.clip((junction[0] - HOLE_X) / legs[0], 0.0, 0.02))
                log(f"EXTRA PULL {extra * 1000:.1f} mm")
                pos = pos - extra * legs
                self.lock_to(pos, quat, 0.005, arm="R")
                self.R.hold(hold_k(0.4))
                centre = self.in_lock((0.0, 0.0, BODY_TOP + self.travel() + BEND_ZC)) - extra * legs
                n = hold_k(4.0)
                poses = []
                for i in range(1, n + 1):
                    r = q_axis((0, 1, 0), -math.radians(SWING_DEG) * i / n)
                    poses.append((centre + rotate(r, pos - centre), qmul(r, quat)))
                self.lock_path(poses, arm="R")
                self.R.hold(hold_k(0.8))
                self.when_settled(self.R, "right", lambda: self._push(0))

            def _push(self, k):
                latched = bool(view._task_state.latched[0])
                if k == 0:
                    rep = task.episode_report(view, 0)
                    log(f"SWUNG DOWN: lock {np.round(self.lock_pose()[0], 4)} holes d {rep['holes_to_shackle_mm']} "
                        f"tag on {rep['tag_on_shackle']}")
                    if self.stop("swung"):
                        return
                if latched or k > 30:
                    log(f"PUSH done after {k}: travel {self.travel() * 1000:.1f} mm, latched {latched}")
                    self.release()
                    return
                legs = rotate(self.lock_pose()[1], (0.0, 0.0, 1.0))
                self.R.move(self.R.pos + ROOT_OFFSET + 0.003 * legs, speed=0.015)
                self.R.hold(hold_k(0.15))
                self.at(len(self.R.steps), lambda: self._push(k + 1))

            def release(self):
                R = self.R
                R.grip_to(OPEN, hold_k(0.8))
                R.move(R.pos + ROOT_OFFSET - 0.06 * EX, speed=0.03)
                R.move(np.array((0.12, -0.22, 0.28)), speed=0.06)
                R.hold(hold_k(6.0))
                self.sync()

        l_pos, l_quat = grasp_point("left")
        r_pos, r_quat = grasp_point("right")
        L = ArmTimeline(l_pos - ROOT_OFFSET, l_quat, dt, speed)
        R = ArmTimeline(r_pos - ROOT_OFFSET, r_quat, dt, speed)
        L.grip = OPEN
        R.grip = OPEN
        planner = Planner(L, R)
        settle = hold_k(0.5)
        L.hold(settle)
        planner.at(settle, planner.plan_pick)
        control = bool(control)

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
        del tag_start_pose, lock_props, TIP_Z, R_WAIT, switch, QA

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
