"""Close a named gas cylinder's valve and lock it out with a cylinder lockout: Arena props and TaskBase.

Mechanism (cylvalve_sim/usd/cylvalve_props.py):
- Cylinders: fixed-base articulation; each handwheel turns on a revolute joint about its valve's axis (0 = seated
  shut, positive = open, anticlockwise from above). Turning it takes WHEEL_TORQUE_NM: a position drive whose target
  follows the wheel each step and whose effort is capped at that torque (Coulomb-like friction). The seat is the
  joint's lower limit. Both start open WHEEL_OPEN_DEG.
- Cap (Brady LM023E): floating articulation, the can and its locking plate on a vertical pin at the rim (0 = shut,
  positive = open). The plate turns against PLATE_FRICTION_NM the same way. It starts swung open, the can standing
  on its stand.
- Padlock (Master Lock 410): body and shackle on a prismatic joint with a weak spring; once pushed within
  LATCH_TRAVEL_M of shut it latches (a joint-limit rule; the real lock's latch is internal).

Measured each step: both handwheel angles; the can's pose against each valve (axis offset, tilt, how high its rim
stands above the neck ring); the plate angle; whether the shackle's short leg passes through both tabs' holes (each
tab in its own body's frame).

Function test (the success check): once the shackle has latched through both tabs with the plate shut round the
argon valve's neck and both grasp points have been clear of the padlock and the cap for TEST_WAIT_S, a reviewer's
checks are applied as forces for TEST_PHASE_S each: (1) holding the can, the plate's tab pushed toward open with
PUSH_N (an equal and opposite torque on the can); (2) still pushing, the cap twisted anticlockwise (the valve's opening
direction) with TWIST_NM through up to TWIST_DEG; (3) still pushing, the cap lifted with LIFT_N (stopped once it has
clearly come off). Passed if the plate stays within PLATE_HELD_DEG of shut, the cap rises less than LIFT_HELD_M and
the handwheel stays closed. ``force_function_test(env)`` starts the same test at once (the control run).
"""

from __future__ import annotations

import math
from dataclasses import MISSING
from pathlib import Path

import torch

import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import EventTermCfg
from isaaclab.utils.configclass import configclass

from isaaclab_arena.assets.object import Object
from isaaclab_arena.assets.object_type import ObjectType
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.utils.pose import Pose

from cylvalve_sim.kit import GRASP_OFFSET_M, q_axis, reset_robot_home
from cylvalve_sim.scorer import MILESTONES, CylinderLockoutScorer, Step
from cylvalve_sim.spec import DECOY_XY, PEDESTAL_XY, TARGET_XY, Layout, make_seed
from cylvalve_sim.usd import cylvalve_props as props
from cylvalve_sim.usd import padlock_props as lock_props

_ASSETS = Path(__file__).resolve().parent / "assets"

PV = lock_props.PADLOCKS["ml410"]
LOCK_QUAT = (0.0, 0.0, 0.0, 1.0)
"""The padlock standing upright, shackle up, its short leg on the far side (+x): as it hangs from the cap's tabs."""
SPRING_N_PER_M = 150.0
LATCH_TRAVEL_M = 0.002

WHEEL_TORQUE_NM = 1.1
"""Turning torque of the handwheel, breakaway = running (Sherwood GSV datasheet: operating torque 1.1-2.2 N m,
closing torque at the seat 2.2-3.3 N m). The sim uses the low end of the operating range all the way to the seat."""
WHEEL_STIFFNESS = 3000.0
WHEEL_DAMPING = 0.05
WHEEL_OPEN_RAD = math.radians(props.WHEEL_OPEN_DEG)
PLATE_FRICTION_NM = 0.02
"""The plate's rivet friction (inferred): it stays where it is left, a light finger push swings it."""
PLATE_DAMPING = 0.3
"""The plate's rivet damping, N m s/rad (inferred: a stiff moulded rivet; a light plate let go swings slowly)."""
PLATE_OPEN_RAD = math.radians(props.PLATE_OPEN_DEG)

CLOSED_DEG = 5.0
REOPEN_DEG = 30.0
DECOY_TURNED_DEG = 15.0
SEAT_OFFSET_M = 0.010
"""The can's axis within this of the valve's axis."""
SEAT_TILT_DEG = 10.0
SEAT_Z = (props.PLATE_GAP + props.PLATE_T - 0.001, props.COLLAR_Z[0] - props.RING_Z[1] + props.PLATE_GAP)
"""The can's rim above the neck ring's top: the plate in the gap between the ring and the wrench block."""
PLATE_SHUT_DEG = 8.0
RELEASE_CLEAR_M = 0.06
CAP_CLEAR_M = props.CAN_R + 0.03
"""Released: both grasp points this far from the can's axis (horizontally) and RELEASE_CLEAR_M from the padlock."""
DROP_Z_M = 0.03

TEST_WAIT_S = 1.0
TEST_PHASE_S = 1.5
TEST_RAMP_S = 0.5
PUSH_N = 10.0
"""The plate's tab pushed toward open (a firm thumb; 0.94 N m about the rivet)."""
TWIST_NM = 1.0
"""The cap twisted anticlockwise, the valve's opening direction (a firm wrist on an 88 mm can)."""
TWIST_DEG = 90.0
"""... through a quarter turn (it turns freely on the valve; the handwheel inside must not turn with it)."""
LIFT_N = 20.0
"""The cap pulled straight up (about ten times its weight)."""
PLATE_HELD_DEG = 15.0
LIFT_HELD_M = 0.015

CYLINDERS, CAP, PADLOCK, STAND = "cylinders", "cap", "padlock", "stand"


def _pose(p) -> Pose:
    return Pose(position_xyz=tuple(p[:3]), rotation_xyzw=tuple(p[3:]))


def lock_start_pose() -> tuple:
    """Body centre over the pedestal's column, standing on its bottom face 0.5 mm above the column's top."""
    x, y = PEDESTAL_XY
    return (x, y, props.PEDESTAL_TOP + PV.body[2] / 2 + 0.0005) + LOCK_QUAT


class Cylinders(Object):
    def __init__(self):
        super().__init__(name=CYLINDERS, usd_path=str(_ASSETS / "cylinders.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {"wheels": ImplicitActuatorCfg(joint_names_expr=["wheel_.*_joint"], stiffness=WHEEL_STIFFNESS,
                                                       damping=WHEEL_DAMPING, effort_limit_sim=WHEEL_TORQUE_NM,
                                                       armature=1e-4)}
        cfg.init_state.joint_pos = {"wheel_.*_joint": WHEEL_OPEN_RAD}
        return cfg


class Cap(Object):
    def __init__(self):
        super().__init__(name=CAP, usd_path=str(_ASSETS / "cap.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=_pose(props.cap_start_pose()))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {"plate": ImplicitActuatorCfg(joint_names_expr=["plate_joint"], stiffness=50.0, damping=PLATE_DAMPING,
                                                      effort_limit_sim=PLATE_FRICTION_NM, armature=1e-5)}
        cfg.init_state.joint_pos = {"plate_joint": PLATE_OPEN_RAD}
        return cfg


class Padlock(Object):
    """The blue Master Lock 410 (sprung shackle; the task latches it)."""

    def __init__(self):
        super().__init__(name=PADLOCK, usd_path=str(_ASSETS / "padlock_ml410.usda"),
                         object_type=ObjectType.ARTICULATION, initial_pose=_pose(lock_start_pose()))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {"shackle": ImplicitActuatorCfg(joint_names_expr=["shackle_joint"], stiffness=SPRING_N_PER_M,
                                                        damping=3.0)}
        cfg.init_state.joint_pos = {"shackle_joint": lock_props.OPEN_TRAVEL}
        return cfg


class Stand(Object):
    def __init__(self):
        super().__init__(name=STAND, usd_path=str(_ASSETS / "stand.usda"), object_type=ObjectType.BASE,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))


def _write_pose(asset, pose, env_ids, origins):
    dev, n = origins.device, len(env_ids)
    p = torch.tensor(pose, device=dev, dtype=torch.float32).unsqueeze(0).repeat(n, 1)
    p[:, :3] += origins[env_ids]
    asset.write_root_pose_to_sim(p, env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)


class _TaskState:
    def __init__(self, env: ManagerBasedEnv, task: "CylinderLockoutTask"):
        n, dev = env.num_envs, env.device
        self.cyl = env.scene[CYLINDERS]
        self.cap = env.scene[CAP]
        self.lock = env.scene[PADLOCK]
        self.robot = env.scene["robot"]
        self.wheel_t = self.cyl.find_joints("wheel_target_joint")[0][0]
        self.wheel_d = self.cyl.find_joints("wheel_decoy_joint")[0][0]
        self.plate_joint = self.cap.find_joints("plate_joint")[0][0]
        self.can_id = self.cap.find_bodies("can")[0][0]
        self.plate_id = self.cap.find_bodies("plate")[0][0]
        self.shackle_joint = self.lock.find_joints("shackle_joint")[0][0]
        self.lock_body = self.lock.find_bodies("body")[0][0]
        self.shackle_id = self.lock.find_bodies("shackle")[0][0]
        self.flange_ids, _ = self.robot.find_bodies(["left_gripper", "right_gripper"], preserve_order=True)
        self.scorer = CylinderLockoutScorer(n, dev)
        self.next_layout = make_seed(task.default_seed)
        self.layouts: dict[int, Layout] = {}
        self.last_reports: dict[int, dict] = {}
        zb = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
        zl = lambda: torch.zeros(n, dtype=torch.long, device=dev)  # noqa: E731
        zf = lambda: torch.zeros(n, device=dev)  # noqa: E731
        self.latched, self.latched_through, self.released, self.force_test = zb(), zb(), zb(), zb()
        self.test_passed, self.closed_once, self.seated_once, self.shut_once = zb(), zb(), zb(), zb()
        self.plate_held, self.lift_held, self.wheel_held = zb(), zb(), zb()
        self.clear_steps, self.phase, self.phase_steps = zl(), zl(), zl()
        self.decoy_max, self.reopen_max, self.plate_max, self.lift_max, self.wheel_max = zf(), zf(), zf(), zf(), zf()
        self.test_q0 = torch.zeros(n, 4, device=dev)
        self.test_z0, self.q_t, self.q_d, self.plate_q = zf(), zf(), zf(), zf()
        self.offset, self.rim_z = zf(), zf()
        self.wait_steps = max(1, round(TEST_WAIT_S / env.step_dt))
        self.phase_len = max(1, round(TEST_PHASE_S / env.step_dt))
        self.ramp_len = max(1, round(TEST_RAMP_S / env.step_dt))


def through_tab(tip, top, tab_z, half_t, slack=0.0):
    """The short leg (tip -> top, in a tab's body frame) crosses that flat tab (mid-plane z = tab_z) inside its
    vertical hole."""
    denom = tip[:, 2] - top[:, 2]
    denom = torch.where(denom.abs() > 1e-6, denom, torch.full_like(denom, 1e-6))
    a = ((tip[:, 2] - tab_z) / denom).clamp(0.0, 1.0)
    cross = tip + a.unsqueeze(1) * (top - tip)
    h = props.TAB_HOLE / 2 + slack
    in_hole = ((cross[:, 0] - props.TAB_HOLE_X).abs() < h) & (cross[:, 1].abs() < h)
    spans = (((tip[:, 2] < tab_z - half_t) & (top[:, 2] > tab_z + half_t))
             | ((tip[:, 2] > tab_z + half_t) & (top[:, 2] < tab_z - half_t)))
    return in_hole & spans


class CylinderLockoutTask(TaskBase):
    def __init__(self, default_seed: int = 0, episode_length_s: float = 180.0):
        super().__init__(episode_length_s=episode_length_s, task_description=make_seed(default_seed).instruction)
        self.default_seed = default_seed

    def state(self, env: ManagerBasedEnv) -> _TaskState:
        if getattr(env, "_task_state", None) is None:
            env._task_state = _TaskState(env, self)
        return env._task_state

    def force_function_test(self, env: ManagerBasedEnv) -> None:
        """Control runs: start the check now, padlock or not."""
        self.state(env).force_test[:] = True

    def episode_report(self, env: ManagerBasedEnv, env_id: int = 0, ended: bool = False) -> dict:
        st = self.state(env)
        if ended and env_id in st.last_reports:
            return st.last_reports.pop(env_id)
        return self._report(st, env_id)

    def _report(self, st: _TaskState, env_id: int) -> dict:
        r = st.scorer.report(env_id)
        d = lambda t: round(math.degrees(float(t[env_id])), 1)  # noqa: E731
        r.update(
            target_wheel_deg=d(st.q_t), decoy_wheel_max_deg=d(st.decoy_max), plate_deg=d(st.plate_q),
            cap_offset_mm=round(float(st.offset[env_id]) * 1000, 1), cap_rim_above_ring_mm=round(float(st.rim_z[env_id]) * 1000, 1),
            closed_once=bool(st.closed_once[env_id]), seated_once=bool(st.seated_once[env_id]),
            plate_shut_once=bool(st.shut_once[env_id]), shackle_latched=bool(st.latched[env_id]),
            latched_through_tabs=bool(st.latched_through[env_id]), released=bool(st.released[env_id]),
            check_phase=int(st.phase[env_id]), check_plate_max_deg=d(st.plate_max),
            check_lift_max_mm=round(float(st.lift_max[env_id]) * 1000, 1), check_wheel_max_deg=d(st.wheel_max),
            plate_held=bool(st.plate_held[env_id]), lift_held=bool(st.lift_held[env_id]),
            wheel_held=bool(st.wheel_held[env_id]), check_passed=bool(st.test_passed[env_id]),
        )
        return r

    def get_scene_cfg(self):
        return None

    def get_events_cfg(self):
        return EventsCfg(self)

    def get_termination_cfg(self) -> TaskTerminationCfg:
        sequence = [(self._milestone_predicate(m), 0.25) for m in MILESTONES]
        return TaskTerminationCfg(timeout_s=self.episode_length_s,
                                  success=[CompletionCriteria(name="milestones", predicate_sequence=sequence)])

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def _milestone_predicate(self, milestone: int):
        def predicate(env: ManagerBasedEnv) -> torch.Tensor:
            return self.state(env).scorer.milestone >= milestone

        predicate.__name__ = f"milestone_{milestone}"
        return predicate

    # Events ------------------------------------------------------------------------------------
    def reset(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        reset_robot_home(env, env_ids)
        for env_id in env_ids.tolist():
            if int(st.scorer.step[env_id]) > 0:
                st.last_reports[env_id] = self._report(st, env_id)
        layout = st.next_layout
        dev, n = env.device, len(env_ids)
        origins = env.scene.env_origins
        st.scorer.reset(env_ids)
        for buf in (st.latched, st.latched_through, st.released, st.force_test, st.test_passed, st.closed_once,
                    st.seated_once, st.shut_once, st.plate_held, st.lift_held, st.wheel_held):
            buf[env_ids] = False
        for buf in (st.clear_steps, st.phase, st.phase_steps):
            buf[env_ids] = 0
        for buf in (st.decoy_max, st.reopen_max, st.plate_max, st.lift_max, st.wheel_max):
            buf[env_ids] = 0.0
        for asset in (st.cap, st.lock):
            asset.permanent_wrench_composer.reset(env_ids=env_ids)
        # Both valves open half a turn.
        w = torch.full((n, 2), WHEEL_OPEN_RAD, device=dev)
        wj = [st.wheel_t, st.wheel_d]
        st.cyl.write_joint_state_to_sim(w, torch.zeros_like(w), joint_ids=wj, env_ids=env_ids)
        st.cyl.set_joint_position_target(w, joint_ids=wj, env_ids=env_ids)
        # The cap on its stand, plate swung open.
        _write_pose(st.cap, props.cap_start_pose(), env_ids, origins)
        p = torch.full((n, 1), PLATE_OPEN_RAD, device=dev)
        st.cap.write_joint_state_to_sim(p, torch.zeros_like(p), joint_ids=[st.plate_joint], env_ids=env_ids)
        st.cap.set_joint_position_target(p, joint_ids=[st.plate_joint], env_ids=env_ids)
        # Padlock open on its pedestal.
        joint = [st.shackle_joint]
        limits = torch.tensor([[[0.0, lock_props.OPEN_TRAVEL]]], device=dev).expand(n, 1, 2).contiguous()
        travel = torch.full((n, 1), lock_props.OPEN_TRAVEL, device=dev)
        st.lock.write_joint_position_limit_to_sim(limits, joint_ids=joint, env_ids=env_ids, warn_limit_violation=False)
        st.lock.write_joint_state_to_sim(travel, torch.zeros_like(travel), joint_ids=joint, env_ids=env_ids)
        st.lock.set_joint_position_target(travel + 0.005, joint_ids=joint, env_ids=env_ids)
        _write_pose(st.lock, lock_start_pose(), env_ids, origins)
        for env_id in env_ids.tolist():
            st.layouts[env_id] = layout

    def _seat(self, can_pos, can_quat, xy, dev):
        """(seated over the valve at xy, horizontal axis offset, rim height above the neck ring)."""
        n = can_pos.shape[0]
        off = (can_pos[:, :2] - torch.tensor(xy, device=dev)).norm(dim=1)
        up = math_utils.quat_apply(can_quat, torch.tensor([[0.0, 0.0, 1.0]], device=dev).expand(n, 3))
        tilt_ok = up[:, 2] > math.cos(math.radians(SEAT_TILT_DEG))
        rim_z = can_pos[:, 2] - props.RING_Z[1]
        seated = (off < SEAT_OFFSET_M) & tilt_ok & (rim_z > SEAT_Z[0]) & (rim_z < SEAT_Z[1])
        return seated, off, rim_z

    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:  # noqa: C901
        st = self.state(env)
        origins = env.scene.env_origins
        n, dev = env.num_envs, env.device
        live = st.phase == 0

        # Friction: hold both handwheels and the plate where they are (the drives' effort caps are the friction).
        jp = st.cyl.data.joint_pos.torch
        wj = [st.wheel_t, st.wheel_d]
        st.cyl.set_joint_position_target(jp[:, wj], joint_ids=wj)
        pq = st.cap.data.joint_pos.torch[:, st.plate_joint]
        st.cap.set_joint_position_target(pq.unsqueeze(1), joint_ids=[st.plate_joint])
        st.q_t, st.q_d, st.plate_q = jp[:, st.wheel_t], jp[:, st.wheel_d], pq

        closed = st.q_t < math.radians(CLOSED_DEG)
        st.closed_once |= closed & live
        st.decoy_max = torch.where(live, torch.maximum(st.decoy_max, (st.q_d - WHEEL_OPEN_RAD).abs()), st.decoy_max)
        reopened = st.closed_once & (st.q_t > math.radians(REOPEN_DEG)) & live & ~st.latched_through

        # The cap against each valve.
        can_pos = st.cap.data.body_pos_w.torch[:, st.can_id] - origins
        can_quat = st.cap.data.body_quat_w.torch[:, st.can_id]
        seated, st.offset, st.rim_z = self._seat(can_pos, can_quat, TARGET_XY, dev)
        seated_d, _, _ = self._seat(can_pos, can_quat, DECOY_XY, dev)
        plate_shut = pq < math.radians(PLATE_SHUT_DEG)
        st.seated_once |= seated & closed & live
        st.shut_once |= seated & closed & plate_shut & live
        wrong = seated_d & plate_shut & live

        # Padlock: the short leg through both tabs (each in its own body's frame).
        travel = st.lock.data.joint_pos.torch[:, st.shackle_joint]
        lock_pos = st.lock.data.body_pos_w.torch[:, st.lock_body] - origins
        sh_pos = st.lock.data.body_pos_w.torch[:, st.shackle_id] - origins
        sh_quat = st.lock.data.body_quat_w.torch[:, st.shackle_id]
        pl_pos = st.cap.data.body_pos_w.torch[:, st.plate_id] - origins
        pl_quat = st.cap.data.body_quat_w.torch[:, st.plate_id]

        def on_shackle(p):
            return sh_pos + math_utils.quat_apply(sh_quat, torch.tensor([p], device=dev).expand(n, 3))

        tip_w = on_shackle((PV.leg_x, 0.0, lock_props.SHORT_LEG_BOTTOM))
        top_w = on_shackle((PV.leg_x, 0.0, PV.bend_z))
        zc, zp = props.TAB_Z["can"], props.TAB_Z["plate"]
        through_can = through_tab(math_utils.quat_apply_inverse(can_quat, tip_w - can_pos),
                                  math_utils.quat_apply_inverse(can_quat, top_w - can_pos),
                                  (zc[0] + zc[1]) / 2, (zc[1] - zc[0]) / 2)
        through_plate = through_tab(math_utils.quat_apply_inverse(pl_quat, tip_w - pl_pos),
                                    math_utils.quat_apply_inverse(pl_quat, top_w - pl_pos),
                                    (zp[0] + zp[1]) / 2, (zp[1] - zp[0]) / 2)
        through = through_can & through_plate
        st.through_now = through
        # Still hanging from the can's tab (the leg round it, within 3 mm of the hole: the hanging lock tilts).
        on_tab = through_tab(math_utils.quat_apply_inverse(can_quat, tip_w - can_pos),
                             math_utils.quat_apply_inverse(can_quat, top_w - can_pos),
                             (zc[0] + zc[1]) / 2, (zc[1] - zc[0]) / 2, slack=0.003)

        newly = ~st.latched & (travel < LATCH_TRAVEL_M)
        if newly.any():
            ids = newly.nonzero().flatten()
            limits = torch.tensor([[[0.0, 0.0005]]], device=dev).expand(len(ids), 1, 2).contiguous()
            st.lock.write_joint_position_limit_to_sim(limits, joint_ids=[st.shackle_joint], env_ids=ids,
                                                      warn_limit_violation=False)
            st.lock.set_joint_position_target(torch.zeros(len(ids), 1, device=dev), joint_ids=[st.shackle_joint],
                                              env_ids=ids)
            st.latched |= newly
            st.latched_through |= newly & through

        # Released: both grasp points clear of the padlock and of the cap.
        flange_pos = st.robot.data.body_pos_w.torch[:, st.flange_ids]
        flange_quat = st.robot.data.body_quat_w.torch[:, st.flange_ids]
        down = torch.tensor([0.0, 0.0, -GRASP_OFFSET_M], device=dev).expand_as(flange_pos)
        grasp = flange_pos + math_utils.quat_apply(flange_quat, down) - origins[:, None]
        clear = (((grasp - lock_pos[:, None]).norm(dim=-1) > RELEASE_CLEAR_M).all(dim=1)
                 & ((grasp[..., :2] - can_pos[:, None, :2]).norm(dim=-1) > CAP_CLEAR_M).all(dim=1))
        locked = st.latched_through & through & seated & plate_shut & closed
        st.released |= locked & clear

        # Function test: push the plate's tab toward open; then also twist the cap; then also lift it.
        ready = live & ((locked & clear) | st.force_test)
        st.clear_steps = torch.where(ready, st.clear_steps + 1, torch.zeros_like(st.clear_steps))
        start = ready & ((st.clear_steps >= st.wait_steps) | st.force_test)
        if start.any():
            ids = start.nonzero().flatten()
            st.phase[ids] = 1
            st.phase_steps[ids] = 0
            st.test_z0[ids] = can_pos[ids, 2]
            st.test_q0[ids] = can_quat[ids]
            st.plate_max[ids] = pq[ids]
            st.wheel_max[ids] = st.q_t[ids]
            st.lift_max[ids] = 0.0
        testing = st.phase > 0
        if (testing & (st.phase < 4)).any():
            ids = (testing & (st.phase < 4)).nonzero().flatten()
            k = len(ids)
            ramp = ((st.phase_steps[ids].float() + 1) / st.ramp_len).clamp(max=1.0)
            zaxis = math_utils.quat_apply(can_quat[ids], torch.tensor([[0.0, 0.0, 1.0]], device=dev).expand(k, 3))
            lever = abs(props.TAB_HOLE_X - props.PIVOT[0])
            # (1) the tab pushed toward open: a torque about the rivet's (vertical) axis on the plate.
            push = torch.where(st.phase[ids] == 1, ramp, torch.ones_like(ramp))
            # A reviewer stops twisting and pulling once the cap has clearly come off (risen LIFT_HELD_M + 5 mm):
            # nothing more is learned and the cap would be flung.
            gave_way = can_pos[ids, 2] - st.test_z0[ids] > LIFT_HELD_M + 0.005
            plate_torque = (PUSH_N * lever * push).unsqueeze(1) * zaxis
            st.cap.permanent_wrench_composer.reset(env_ids=ids)
            st.cap.permanent_wrench_composer.add_forces_and_torques_index(
                forces=torch.zeros_like(plate_torque).unsqueeze(1), torques=plate_torque.unsqueeze(1),
                body_ids=[st.plate_id], env_ids=ids, is_global=True)
            # (2) the can twisted anticlockwise; (3) lifted.
            twist = torch.where(st.phase[ids] == 2, ramp, (st.phase[ids] > 2).float())
            lift = torch.where(st.phase[ids] == 3, ramp, torch.zeros_like(ramp))
            twist = torch.where(gave_way, torch.zeros_like(twist), twist)
            # The cap turns freely on the valve: the reviewer twists it through a quarter turn, then stops.
            dq = math_utils.quat_mul(can_quat[ids], math_utils.quat_conjugate(st.test_q0[ids]))
            turned = 2.0 * torch.atan2(dq[:, :3].norm(dim=1), dq[:, 3].abs())  # xyzw: the cap's turn since the start
            twist = torch.where(turned > math.radians(TWIST_DEG), torch.zeros_like(twist), twist)
            lift = torch.where(gave_way, torch.zeros_like(lift), lift)
            # Holding the can, the reviewer pushes the plate's tab: an equal and opposite torque on the can.
            can_torque = (TWIST_NM * twist).unsqueeze(1) * zaxis - plate_torque
            can_force = torch.zeros(k, 3, device=dev)
            can_force[:, 2] = LIFT_N * lift
            st.cap.permanent_wrench_composer.add_forces_and_torques_index(
                forces=can_force.unsqueeze(1), torques=can_torque.unsqueeze(1), body_ids=[st.can_id], env_ids=ids,
                is_global=True)
            st.plate_max[ids] = torch.maximum(st.plate_max[ids], pq[ids])
            st.lift_max[ids] = torch.maximum(st.lift_max[ids], can_pos[ids, 2] - st.test_z0[ids])
            st.wheel_max[ids] = torch.maximum(st.wheel_max[ids], st.q_t[ids])
        st.phase_steps = torch.where(testing, st.phase_steps + 1, st.phase_steps)
        end = testing & (st.phase < 4) & (st.phase_steps >= st.phase_len)
        if end.any():
            ids = end.nonzero().flatten()
            st.phase[ids] += 1
            st.phase_steps[ids] = 0
            done = ids[st.phase[ids] >= 4]
            if len(done):
                st.cap.permanent_wrench_composer.reset(env_ids=done)
                st.plate_held[done] = st.plate_max[done] < math.radians(PLATE_HELD_DEG)
                st.lift_held[done] = st.lift_max[done] < LIFT_HELD_M
                st.wheel_held[done] = st.wheel_max[done] < math.radians(CLOSED_DEG)
                st.test_passed[done] = (st.plate_held[done] & st.lift_held[done] & st.wheel_held[done]
                                        & st.latched_through[done] & on_tab[done])
                st.phase[done] = 3 + 2  # 5: finished
                for i in done.tolist():
                    print(f"[cylinder-lockout] after-release check: plate pushed open to "
                          f"{math.degrees(float(st.plate_max[i])):.1f} deg (held: {bool(st.plate_held[i])}), cap lifted "
                          f"{float(st.lift_max[i]) * 1000:.1f} mm (held: {bool(st.lift_held[i])}), handwheel "
                          f"{math.degrees(float(st.wheel_max[i])):.1f} deg from shut (held: {bool(st.wheel_held[i])}), "
                          f"padlock still on the can's tab: {bool(st.latched_through[i] & on_tab[i])}, passed: "
                          f"{bool(st.test_passed[i])}", flush=True)

        lock_dropped = lock_pos[:, 2] < DROP_Z_M
        cap_dropped = can_pos[:, 2] < DROP_Z_M
        st.scorer.update(Step(closed & live, seated & live, plate_shut & live, st.test_passed.clone(),
                              (st.decoy_max > math.radians(DECOY_TURNED_DEG)) & live, reopened, wrong,
                              lock_dropped, cap_dropped))


@configclass
class EventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: CylinderLockoutTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))
