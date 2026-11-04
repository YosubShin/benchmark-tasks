"""Ball-valve lockout: close the named quarter-turn ball valve and padlock its locking handle.

Mechanism (see valve_lockout_sim/usd/valve_props.py; Milwaukee BA-100 "LD" / Apollo "-27" latch-lock):
- Each valve's lever turns on a revolute joint (0 = open, -90 deg = closed, clockwise from above), with hard stops
  (joint limits, as the real stop plate's stops). Turning resistance: a stiff drive re-targeted every step to the
  current angle, with its effort capped at VALVE_TORQUE_NM (Coulomb-like breakaway and running friction).
- The latch sleeve slides along the lever (prismatic joint, light friction the same way). Its tongue can only go in
  where the stop plate has a notch (open or closed); in the notch it blocks the lever (collision).
- The padlock's shackle through the lever's lock hole, just outboard of the latched sleeve, stops the sleeve from
  sliding back out (collision). Nothing in the code holds the lever: geometry does.
- The robot's Master Lock 410: the shackle on a prismatic joint with a weak spring; the task latches it in software
  once it is pushed within 2 mm of shut.

Function test (the success check): once the shackle has latched through the named valve's lock hole and both grippers
have been clear of the padlock for TEST_WAIT_S, the scorer pulls the sleeve outward with SLEEVE_TEST_N for the whole
test and, after TEST_PHASE_S, adds TEST_NM on the lever toward open for TEST_PHASE_S. Passed if the lever stayed
within LEVER_HELD_DEG of closed and the shackle is still through the hole. `force_function_test(env)` starts the
same test at once, lock or no lock (the control run).
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
from isaaclab.sensors import CameraCfg
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

from valve_lockout_sim.kit import GRASP_OFFSET_M, q_axis, reset_robot_home, reviewer_camera
from valve_lockout_sim.scorer import MILESTONES, ValveScorer, ValveStep
from valve_lockout_sim.spec import ValveSeed, make_seed
from valve_lockout_sim.usd import padlock_props as lock_props
from valve_lockout_sim.usd import valve_props as props

_ASSETS = Path(__file__).resolve().parent / "assets"
SPRING_N_PER_M = 150.0
"""Shackle spring (inferred): 2.4 N at full 16 mm travel."""
LATCH_TRAVEL_M = 0.002
"""The shackle latches once pushed within this distance of shut."""

PV = lock_props.PADLOCKS["ml410"]
LOCK_QUAT = tuple(float(v) for v in q_axis((0, 0, 1), -math.pi / 2))
"""Upright, the shackle's legs along y with the short leg at -y (it goes through the hole of a lever along x)."""

VALVE_TORQUE_NM = 1.5
"""Sim turning torque (breakaway = running). Real: Milwaukee's chart gives 40 in-lb (4.5 N*m) for its 1/2 in
two-piece threaded valve at full rated pressure (no BA-100 row; proxy). See the report for why the sim is lower."""
SLEEVE_FRICTION_N = 0.5
STEM_STIFFNESS = 3000.0
STEM_DAMPING = 0.3

TURNED_DEG = 10.0
CLOSED_DEG = 85.0
REOPEN_DEG = 60.0
DECOY_MOVED_DEG = 10.0
LATCHED_M = 0.003
"""The sleeve counts as in (tongue in the notch) within this distance of its inner end."""
RELEASE_CLEAR_M = 0.06
DROP_Z_M = 0.05
CONTACT_MOVE_M = 0.01

TEST_WAIT_S = 2.0
TEST_PHASE_S = 2.0
TEST_RAMP_S = 1.0
SLEEVE_TEST_N = 10.0
"""Pulling the latch sleeve back out (inferred finger effort)."""
TEST_NM = 3.0
"""Toward open on the lever: 2x the sim turning torque (about 32 N at the grip, 95 mm from the stem)."""
LEVER_HELD_DEG = 15.0

TARGET_SLOT = 0


def lock_start_pose() -> tuple:
    x, y = props.PEDESTAL_XY
    return (x, y, props.PEDESTAL_TOP + PV.body[2] / 2 + 0.001) + LOCK_QUAT


class PipeRun(Object):
    def __init__(self):
        super().__init__(name="pipe_run", usd_path=str(_ASSETS / "pipe_run.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {
            "levers": ImplicitActuatorCfg(joint_names_expr=["lever_.*_joint"], stiffness=STEM_STIFFNESS,
                                          damping=STEM_DAMPING, effort_limit_sim=VALVE_TORQUE_NM),
            "sleeves": ImplicitActuatorCfg(joint_names_expr=["sleeve_.*_joint"], stiffness=2000.0, damping=1.0,
                                           effort_limit_sim=SLEEVE_FRICTION_N),
        }
        cfg.init_state.joint_pos = {"lever_.*_joint": 0.0, "sleeve_.*_joint": -props.SLEEVE_TRAVEL}
        return cfg


class ValvePadlock(Object):
    """The robot's Master Lock 410PRP (sprung shackle; the task latches it)."""

    def __init__(self):
        p = lock_start_pose()
        super().__init__(name="padlock", usd_path=str(_ASSETS / "padlock_ml410.usda"),
                         object_type=ObjectType.ARTICULATION, initial_pose=Pose(position_xyz=p[:3], rotation_xyzw=p[3:]))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {"shackle": ImplicitActuatorCfg(joint_names_expr=["shackle_joint"], stiffness=SPRING_N_PER_M,
                                                        damping=3.0)}
        cfg.init_state.joint_pos = {"shackle_joint": lock_props.OPEN_TRAVEL}
        return cfg


class Pedestal(Object):
    def __init__(self):
        super().__init__(name="pedestal", usd_path=str(_ASSETS / "pedestal.usda"), object_type=ObjectType.BASE,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))


class _TaskState:
    def __init__(self, env: ManagerBasedEnv, task: "ValveLockoutTask"):
        n, dev = env.num_envs, env.device
        self.lock = env.scene["padlock"]
        self.pipe = env.scene["pipe_run"]
        self.robot = env.scene["robot"]
        self.shackle_joint = self.lock.find_joints("shackle_joint")[0][0]
        self.body_id = self.lock.find_bodies("body")[0][0]
        self.shackle_id = self.lock.find_bodies("shackle")[0][0]
        self.lever_joints = [self.pipe.find_joints(f"lever_{k}_joint")[0][0] for k in range(2)]
        self.sleeve_joints = [self.pipe.find_joints(f"sleeve_{k}_joint")[0][0] for k in range(2)]
        self.lever_bodies = [self.pipe.find_bodies(f"lever_{k}")[0][0] for k in range(2)]
        self.sleeve_bodies = [self.pipe.find_bodies(f"sleeve_{k}")[0][0] for k in range(2)]
        self.flange_ids, _ = self.robot.find_bodies(["left_gripper", "right_gripper"], preserve_order=True)
        self.scorer = ValveScorer(n, dev)
        self.next_layout = make_seed(task.default_seed)
        self.layouts: dict[int, ValveSeed] = {}
        self.last_reports: dict[int, dict] = {}
        zb = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
        zl = lambda: torch.zeros(n, dtype=torch.long, device=dev)  # noqa: E731
        zf = lambda: torch.zeros(n, device=dev)  # noqa: E731
        self.latched, self.latched_through, self.released, self.force_test = zb(), zb(), zb(), zb()
        self.was_closed, self.held, self.test_passed = zb(), zb(), zb()
        self.clear_steps, self.phase, self.phase_steps = zl(), zl(), zl()
        self.open_max, self.sleeve_out_max = zf(), zf()
        self.start_pos = torch.zeros(n, 3, device=dev)
        self.wait_steps = max(1, round(TEST_WAIT_S / env.step_dt))
        self.phase_len = max(1, round(TEST_PHASE_S / env.step_dt))


def _write_pose(asset, pose, env_ids, origins, velocity=True):
    dev, n = origins.device, len(env_ids)
    p = torch.tensor(pose, device=dev, dtype=torch.float32).unsqueeze(0).repeat(n, 1)
    p[:, :3] += origins[env_ids]
    asset.write_root_pose_to_sim(p, env_ids=env_ids)
    if velocity:
        asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)


class ValveLockoutTask(TaskBase):
    def __init__(self, default_seed: int = 0, episode_length_s: float = 90.0, reviewer_camera: bool = False):
        super().__init__(episode_length_s=episode_length_s, task_description=make_seed(default_seed).instruction)
        self.default_seed = default_seed
        self.reviewer_camera = reviewer_camera

    def state(self, env: ManagerBasedEnv) -> _TaskState:
        if getattr(env, "_task_state", None) is None:
            env._task_state = _TaskState(env, self)
        return env._task_state

    def set_next_seed(self, env: ManagerBasedEnv, seed: int) -> None:
        self.state(env).next_layout = make_seed(seed)

    def force_function_test(self, env: ManagerBasedEnv) -> None:
        """Control runs: start the function test now, lock or no lock."""
        self.state(env).force_test[:] = True

    def episode_report(self, env: ManagerBasedEnv, env_id: int = 0, ended: bool = False) -> dict:
        st = self.state(env)
        if ended and env_id in st.last_reports:
            return st.last_reports.pop(env_id)
        return self._report(st, env_id)

    def _report(self, st: _TaskState, env_id: int) -> dict:
        r = st.scorer.report(env_id)
        layout = st.layouts.get(env_id, st.next_layout)
        r.update(
            target_tag=layout.target_tag, valve_torque_nm=VALVE_TORQUE_NM, test_nm=TEST_NM,
            shackle_latched=bool(st.latched[env_id]), shackle_latched_through_hole=bool(st.latched_through[env_id]),
            function_test_phase=int(st.phase[env_id]),
            test_lever_max_open_deg=round(math.degrees(float(st.open_max[env_id])), 1),
            test_sleeve_max_out_mm=round(1000 * float(st.sleeve_out_max[env_id]), 1),
            test_lever_held=bool(st.held[env_id]), function_test_passed=bool(st.test_passed[env_id]),
        )
        return r

    def get_scene_cfg(self):
        return ReviewerCameraSceneCfg() if self.reviewer_camera else None

    def get_events_cfg(self):
        return ValveEventsCfg(self)

    def get_termination_cfg(self) -> TaskTerminationCfg:
        sequence = [(self._milestone_predicate(m), 0.25) for m in MILESTONES]
        return TaskTerminationCfg(timeout_s=self.episode_length_s,
                                  success=[CompletionCriteria(name="valve_milestones", predicate_sequence=sequence)])

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
        for buf in (st.latched, st.latched_through, st.released, st.force_test, st.was_closed, st.held,
                    st.test_passed):
            buf[env_ids] = False
        for buf in (st.clear_steps, st.phase, st.phase_steps):
            buf[env_ids] = 0
        st.open_max[env_ids] = 0.0
        st.sleeve_out_max[env_ids] = 0.0
        for asset in (st.lock, st.pipe):
            asset.permanent_wrench_composer.reset(env_ids=env_ids)

        # The robot's padlock: open, unlatched, upright on its pedestal.
        joint = [st.shackle_joint]
        limits = torch.tensor([[[0.0, lock_props.OPEN_TRAVEL]]], device=dev).expand(n, 1, 2).contiguous()
        travel = torch.full((n, 1), lock_props.OPEN_TRAVEL, device=dev)
        st.lock.write_joint_position_limit_to_sim(limits, joint_ids=joint, env_ids=env_ids, warn_limit_violation=False)
        st.lock.write_joint_state_to_sim(travel, torch.zeros_like(travel), joint_ids=joint, env_ids=env_ids)
        st.lock.set_joint_position_target(travel + 0.005, joint_ids=joint, env_ids=env_ids)
        _write_pose(st.lock, lock_start_pose(), env_ids, origins)
        st.start_pos[env_ids] = torch.tensor(lock_start_pose()[:3], device=dev)

        # Both valves open, sleeves out.
        joints = st.lever_joints + st.sleeve_joints
        q = torch.tensor([[0.0, 0.0, -props.SLEEVE_TRAVEL, -props.SLEEVE_TRAVEL]], device=dev).repeat(n, 1)
        st.pipe.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=joints, env_ids=env_ids)
        st.pipe.set_joint_position_target(q, joint_ids=joints, env_ids=env_ids)
        for env_id in env_ids.tolist():
            st.layouts[env_id] = layout

    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        origins = env.scene.env_origins
        n, dev = env.num_envs, env.device
        t_slot = TARGET_SLOT
        jp = st.pipe.data.joint_pos.torch
        # Friction: hold every lever and sleeve where it is (the drive's effort cap is the friction).
        joints = st.lever_joints + st.sleeve_joints
        st.pipe.set_joint_position_target(jp[:, joints], joint_ids=joints)
        q_t = jp[:, st.lever_joints[t_slot]]
        q_d = jp[:, st.lever_joints[1 - t_slot]]
        closed_deg = torch.rad2deg(-q_t)
        sleeve_q = jp[:, st.sleeve_joints[t_slot]]

        # The padlock's short leg against the named lever's lock hole, in the lever frame.
        travel = st.lock.data.joint_pos.torch[:, st.shackle_joint]
        body_pos = st.lock.data.body_pos_w.torch[:, st.body_id] - origins
        sh_pos = st.lock.data.body_pos_w.torch[:, st.shackle_id] - origins
        sh_quat = st.lock.data.body_quat_w.torch[:, st.shackle_id]
        lv_pos = st.pipe.data.body_pos_w.torch[:, st.lever_bodies[t_slot]] - origins
        lv_quat = st.pipe.data.body_quat_w.torch[:, st.lever_bodies[t_slot]]

        def on_shackle(p):
            return sh_pos + math_utils.quat_apply(sh_quat, torch.tensor([p], device=dev).expand(n, 3))

        def to_lever(p):
            return math_utils.quat_apply_inverse(lv_quat, p - lv_pos)

        tip = to_lever(on_shackle((PV.leg_x, 0.0, lock_props.SHORT_LEG_BOTTOM)))
        top = to_lever(on_shackle((PV.leg_x, 0.0, PV.bend_z - PV.rod / 2)))
        half = props.LEVER_T / 2
        denom = top[:, 2] - tip[:, 2]
        denom = torch.where(denom.abs() > 1e-6, denom, torch.full_like(denom, 1e-6))
        a = (-tip[:, 2] / denom).clamp(0.0, 1.0)
        cross = tip + a.unsqueeze(1) * (top - tip)
        in_hole = (cross[:, 0].abs() < props.HOLE / 2) & ((cross[:, 1] + props.HOLE_S).abs() < props.HOLE / 2)
        spans = ((tip[:, 2] < -half) & (top[:, 2] > half)) | ((tip[:, 2] > half) & (top[:, 2] < -half))
        through = in_hole & spans

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

        flange_pos = st.robot.data.body_pos_w.torch[:, st.flange_ids]
        flange_quat = st.robot.data.body_quat_w.torch[:, st.flange_ids]
        down = torch.tensor([0.0, 0.0, -GRASP_OFFSET_M], device=dev).expand_as(flange_pos)
        grasp = flange_pos + math_utils.quat_apply(flange_quat, down) - origins[:, None]
        clear = ((grasp - body_pos[:, None]).norm(dim=-1) > RELEASE_CLEAR_M).all(dim=1)
        st.released |= st.latched & clear
        moved = (body_pos - st.start_pos).norm(dim=-1) > CONTACT_MOVE_M

        # Function test.
        ready = (st.latched_through & st.released & (st.phase == 0)) | (st.force_test & (st.phase == 0))
        st.clear_steps = torch.where(ready, st.clear_steps + 1, torch.zeros_like(st.clear_steps))
        start = ready & ((st.clear_steps >= st.wait_steps) | st.force_test)
        sleeve_id, lever_id = st.sleeve_bodies[t_slot], st.lever_bodies[t_slot]
        if start.any():
            ids = start.nonzero().flatten()
            st.phase[ids] = 1
            st.phase_steps[ids] = 0
        in_test = st.phase > 0
        st.phase_steps = torch.where(in_test, st.phase_steps + 1, st.phase_steps)
        st.sleeve_out_max = torch.where(in_test & (st.phase < 3), torch.maximum(st.sleeve_out_max, -sleeve_q),
                                        st.sleeve_out_max)
        open_from_closed = torch.deg2rad(90.0 - closed_deg)
        st.open_max = torch.where(st.phase == 2, torch.maximum(st.open_max, open_from_closed), st.open_max)
        to_lever_phase = (st.phase == 1) & (st.phase_steps >= st.phase_len)
        st.phase_steps = torch.where(to_lever_phase, torch.zeros_like(st.phase_steps), st.phase_steps)
        st.phase = torch.where(to_lever_phase, torch.full_like(st.phase, 2), st.phase)
        # The test loads ramp up over TEST_RAMP_S (a hand pulls and pushes; a step load would fling the 10 g sleeve
        # into the shackle at ~1000 m/s^2 and knock the lock out of the hole).
        active = (st.phase == 1) | (st.phase == 2)
        if active.any():
            ids = active.nonzero().flatten()
            st.pipe.permanent_wrench_composer.reset(env_ids=ids)
            ramp_steps = max(1, round(TEST_RAMP_S / env.step_dt))
            t_sleeve = torch.where(st.phase[ids] == 1, st.phase_steps[ids], st.phase_steps[ids] + st.phase_len)
            a_sleeve = (t_sleeve.float() / ramp_steps).clamp(max=1.0)
            out_dir = math_utils.quat_apply(lv_quat[ids], torch.tensor([[0.0, -1.0, 0.0]], device=dev).expand(len(ids), 3))
            force = (SLEEVE_TEST_N * a_sleeve.unsqueeze(1) * out_dir).unsqueeze(1)
            st.pipe.permanent_wrench_composer.add_forces_and_torques_index(
                forces=force, torques=torch.zeros_like(force), body_ids=[sleeve_id], env_ids=ids, is_global=True)
            a_lever = torch.where(st.phase[ids] == 2, (st.phase_steps[ids].float() / ramp_steps).clamp(max=1.0),
                                  torch.zeros_like(a_sleeve))
            torque = torch.zeros(len(ids), 1, 3, device=dev)
            torque[:, 0, 2] = TEST_NM * a_lever  # +z = counterclockwise from above = toward open
            st.pipe.permanent_wrench_composer.add_forces_and_torques_index(
                forces=torch.zeros_like(torque), torques=torque, body_ids=[lever_id], env_ids=ids, is_global=True)
        end = (st.phase == 2) & (st.phase_steps >= st.phase_len)
        if end.any():
            ids = end.nonzero().flatten()
            st.pipe.permanent_wrench_composer.reset(env_ids=ids)
            st.held[ids] = st.open_max[ids] < math.radians(LEVER_HELD_DEG)
            st.test_passed[ids] = st.held[ids] & st.latched_through[ids] & through[ids]
            st.phase[ids] = 3

        before_test = st.phase == 0
        closed = closed_deg > CLOSED_DEG
        st.was_closed |= closed & before_test
        reopened = before_test & st.was_closed & (closed_deg < REOPEN_DEG)
        decoy_moved = q_d.abs() > math.radians(DECOY_MOVED_DEG)
        dropped = body_pos[:, 2] < DROP_Z_M
        st.scorer.update(ValveStep(closed_deg > TURNED_DEG, closed, through & before_test & (sleeve_q > -LATCHED_M),
                                  st.test_passed.clone(), decoy_moved, reopened, dropped))
        st.debug = dict(closed_deg=closed_deg, sleeve_q=sleeve_q, through=through, moved=moved, travel=travel)


REVIEWER_EYE = (0.48, -0.26, 0.38)
REVIEWER_TARGET = (0.22, 0.02, 0.15)


@configclass
class ReviewerCameraSceneCfg:
    reviewer_cam: CameraCfg = reviewer_camera(REVIEWER_EYE, REVIEWER_TARGET)


@configclass
class ValveEventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: ValveLockoutTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))
