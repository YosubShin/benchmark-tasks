"""Ball-valve clamp lockout: close the named quarter-turn ball valve and lock it closed with a clamp-on ball valve
lockout (Master Lock S3080) and a padlock.

Mechanism (see clamp_valve_sim/usd/clamp_props.py for the device, usd/valve_props.py for the valve):
- Each valve's lever turns on a revolute joint (0 = open, -90 deg = closed, clockwise from above) with hard stops
  (joint limits). Turning resistance: stick-slip friction, a stiff drive whose effort is capped at VALVE_TORQUE_NM
  and whose target follows the joint only once the joint has moved past what that effort holds.
- The lockout: half A (channel over the lever, legs on the robot's side of the pipe, a tongue with one hole) is a free
  rigid body; half B (legs on the far side, housing with four holes) slides on it along x (prismatic joint, light
  friction the same way, capped at SLIDE_FRICTION_N). Fitted and pushed together, the legs straddle the pipe beside the valve body, so the device
  cannot turn about the stem, and the lever in the channel cannot turn (collision). The padlock's shackle through B's
  wall and A's tongue stops the halves sliding apart (collision). Nothing in the code holds the lever or the halves.
- The robot's Master Lock 410: the shackle on a prismatic joint with a weak spring; the task latches it in software
  once it is pushed within 2 mm of shut.

Function test (the success check): once the shackle has latched through the aligned holes and both grippers have been
clear of the padlock and the valve for TEST_WAIT_S, the scorer pulls the two halves apart (A toward the robot, B away,
PULL_N each) for the whole test and, after TEST_PHASE_S, adds TEST_NM on the lever toward open for TEST_PHASE_S.
Passed if the lever stayed within LEVER_HELD_DEG of closed and the shackle is still through the holes.
`force_function_test(env)` starts the same test at once, lock or no lock (the control run).
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

from clamp_valve_sim.kit import GRASP_OFFSET_M, reset_robot_home
from clamp_valve_sim.scorer import MILESTONES, ValveScorer, ValveStep
from clamp_valve_sim.spec import ValveSeed, make_seed
from clamp_valve_sim.usd import clamp_props as dev
from clamp_valve_sim.usd import padlock_props as lock_props
from clamp_valve_sim.usd import valve_props as props

_ASSETS = Path(__file__).resolve().parent / "assets"
SPRING_N_PER_M = 150.0
"""Shackle spring (inferred): 2.4 N at full 16 mm travel."""
LATCH_TRAVEL_M = 0.002

PV = lock_props.PADLOCKS["ml410"]
LOCK_QUAT = (-0.5, 0.5, 0.5, 0.5)
"""The padlock on its side: legs along +y (toward the device's holes), short leg below the long one, body thickness
along x."""

VALVE_TORQUE_NM = 1.0
"""Sim turning torque (breakaway = running). Real: Milwaukee's chart gives 40 in-lb (4.5 N*m) for its 1/2 in
two-piece threaded valve at full rated pressure (proxy). Lowered so one soft-wristed arm can turn it to its stop."""
SLIDE_FRICTION_N = 1.0
SLIDE_STIFFNESS = 2000.0
STEM_STIFFNESS = 3000.0
STEM_DAMPING = 0.3

TURNED_DEG = 10.0
CLOSED_DEG = 85.0
REOPEN_DEG = 60.0
DECOY_MOVED_DEG = 10.0
FIT_XY_M = 0.004
FIT_Z_M = 0.004
FIT_YAW_DEG = 8.0
PUSHED_M = 0.002
"""The halves count as pushed together within this distance of the first setting."""
RELEASE_CLEAR_M = 0.06
DROP_Z_M = 0.05

TEST_WAIT_S = 2.0
TEST_PHASE_S = 2.0
TEST_RAMP_S = 1.0
PULL_N = 15.0
"""Pulling the halves apart: A toward the robot, B away from it (a spring scale on each)."""
TEST_NM = 3.0
"""Toward open on the lever: 3x the sim turning torque (about 42 N at the grip's centre, 71 mm from the stem)."""
LEVER_HELD_DEG = 15.0

TARGET_SLOT = 0
DEVICE_START = (0.15, 0.15)
"""Half A's origin (x, y) when the device stands on the table (its channel's end on the rest block)."""


def lock_start_pose() -> tuple:
    x, y = props.CRADLE_XY
    return (x, y, props.CRADLE_TOP + PV.body[0] / 2 + 0.0005) + LOCK_QUAT


def device_start_pose() -> tuple:
    return (*DEVICE_START, dev.DEVICE_START_Z, 0.0, 0.0, 0.0, 1.0)


def device_fitted_pos(slot: int = TARGET_SLOT) -> tuple:
    """Half A's origin when the device is fitted on the closed valve."""
    return (props.VALVE_X, props.VALVE_Y[slot], props.LEVER_Z)


class PipeRun(Object):
    def __init__(self):
        super().__init__(name="pipe_run", usd_path=str(_ASSETS / "pipe_run.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {
            "levers": ImplicitActuatorCfg(joint_names_expr=["lever_.*_joint"], stiffness=STEM_STIFFNESS,
                                          damping=STEM_DAMPING, effort_limit_sim=VALVE_TORQUE_NM),
        }
        cfg.init_state.joint_pos = {"lever_.*_joint": 0.0}
        return cfg


class Lockout(Object):
    """The Master Lock S3080 (two halves on a slide)."""

    def __init__(self):
        p = device_start_pose()
        super().__init__(name="lockout", usd_path=str(_ASSETS / "s3080.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=p[:3], rotation_xyzw=p[3:]))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {"slide": ImplicitActuatorCfg(joint_names_expr=["slide_joint"], stiffness=SLIDE_STIFFNESS, damping=1.0,
                                                      effort_limit_sim=SLIDE_FRICTION_N)}
        cfg.init_state.joint_pos = {"slide_joint": dev.B_START}
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


class Stands(Object):
    def __init__(self):
        super().__init__(name="stands", usd_path=str(_ASSETS / "stands.usda"), object_type=ObjectType.BASE,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))


class _TaskState:
    def __init__(self, env: ManagerBasedEnv, task: "ClampLockoutTask"):
        n, dev_ = env.num_envs, env.device
        self.lock = env.scene["padlock"]
        self.pipe = env.scene["pipe_run"]
        self.device = env.scene["lockout"]
        self.robot = env.scene["robot"]
        self.shackle_joint = self.lock.find_joints("shackle_joint")[0][0]
        self.body_id = self.lock.find_bodies("body")[0][0]
        self.shackle_id = self.lock.find_bodies("shackle")[0][0]
        self.lever_joints = [self.pipe.find_joints(f"lever_{k}_joint")[0][0] for k in range(2)]
        self.lever_bodies = [self.pipe.find_bodies(f"lever_{k}")[0][0] for k in range(2)]
        self.slide_joint = self.device.find_joints("slide_joint")[0][0]
        self.a_id = self.device.find_bodies("half_a")[0][0]
        self.b_id = self.device.find_bodies("half_b")[0][0]
        self.flange_ids, _ = self.robot.find_bodies(["left_gripper", "right_gripper"], preserve_order=True)
        self.scorer = ValveScorer(n, dev_)
        self.next_layout = make_seed(task.default_seed)
        self.layouts: dict[int, ValveSeed] = {}
        self.last_reports: dict[int, dict] = {}
        zb = lambda: torch.zeros(n, dtype=torch.bool, device=dev_)  # noqa: E731
        zl = lambda: torch.zeros(n, dtype=torch.long, device=dev_)  # noqa: E731
        zf = lambda: torch.zeros(n, device=dev_)  # noqa: E731
        self.latched, self.latched_through, self.released, self.force_test = zb(), zb(), zb(), zb()
        self.was_closed, self.held, self.test_passed, self.fitted_ever = zb(), zb(), zb(), zb()
        self.clear_steps, self.phase, self.phase_steps = zl(), zl(), zl()
        self.open_max, self.slide_max = zf(), zf()
        self.lever_target = torch.zeros(n, 2, device=dev_)
        self.slide_target = torch.full((n,), dev.B_START, device=dev_)
        self.wait_steps = max(1, round(TEST_WAIT_S / env.step_dt))
        self.phase_len = max(1, round(TEST_PHASE_S / env.step_dt))


def _write_pose(asset, pose, env_ids, origins):
    dev_, n = origins.device, len(env_ids)
    p = torch.tensor(pose, device=dev_, dtype=torch.float32).unsqueeze(0).repeat(n, 1)
    p[:, :3] += origins[env_ids]
    asset.write_root_pose_to_sim(p, env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev_), env_ids=env_ids)


class ClampLockoutTask(TaskBase):
    def __init__(self, default_seed: int = 0, episode_length_s: float = 90.0):
        super().__init__(episode_length_s=episode_length_s, task_description=make_seed(default_seed).instruction)
        self.default_seed = default_seed

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
            target_tag=layout.target_tag, valve_torque_nm=VALVE_TORQUE_NM, test_nm=TEST_NM, pull_n=PULL_N,
            lockout_fitted=bool(st.fitted_ever[env_id]),
            shackle_latched=bool(st.latched[env_id]), shackle_latched_through_holes=bool(st.latched_through[env_id]),
            function_test_phase=int(st.phase[env_id]),
            test_lever_max_open_deg=round(math.degrees(float(st.open_max[env_id])), 1),
            test_halves_max_apart_mm=round(1000 * float(st.slide_max[env_id]), 1),
            test_lever_held=bool(st.held[env_id]), function_test_passed=bool(st.test_passed[env_id]),
        )
        return r

    def get_scene_cfg(self):
        return None

    def get_events_cfg(self):
        return ClampEventsCfg(self)

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
        dev_, n = env.device, len(env_ids)
        origins = env.scene.env_origins
        st.scorer.reset(env_ids)
        for buf in (st.latched, st.latched_through, st.released, st.force_test, st.was_closed, st.held,
                    st.test_passed, st.fitted_ever):
            buf[env_ids] = False
        for buf in (st.clear_steps, st.phase, st.phase_steps):
            buf[env_ids] = 0
        st.open_max[env_ids] = 0.0
        st.slide_max[env_ids] = 0.0
        for asset in (st.lock, st.pipe, st.device):
            asset.permanent_wrench_composer.reset(env_ids=env_ids)

        # The robot's padlock: open, unlatched, on its side in its cradle.
        joint = [st.shackle_joint]
        limits = torch.tensor([[[0.0, lock_props.OPEN_TRAVEL]]], device=dev_).expand(n, 1, 2).contiguous()
        travel = torch.full((n, 1), lock_props.OPEN_TRAVEL, device=dev_)
        st.lock.write_joint_position_limit_to_sim(limits, joint_ids=joint, env_ids=env_ids, warn_limit_violation=False)
        st.lock.write_joint_state_to_sim(travel, torch.zeros_like(travel), joint_ids=joint, env_ids=env_ids)
        st.lock.set_joint_position_target(travel + 0.005, joint_ids=joint, env_ids=env_ids)
        _write_pose(st.lock, lock_start_pose(), env_ids, origins)

        # The lockout: standing on the table, opened to 40 mm between its leg pairs.
        q = torch.full((n, 1), dev.B_START, device=dev_)
        st.device.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=[st.slide_joint], env_ids=env_ids)
        st.device.set_joint_position_target(q, joint_ids=[st.slide_joint], env_ids=env_ids)
        st.slide_target[env_ids] = dev.B_START
        _write_pose(st.device, device_start_pose(), env_ids, origins)

        # Both valves open.
        q = torch.zeros(n, 2, device=dev_)
        st.pipe.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=st.lever_joints, env_ids=env_ids)
        st.pipe.set_joint_position_target(q, joint_ids=st.lever_joints, env_ids=env_ids)
        st.lever_target[env_ids] = 0.0
        for env_id in env_ids.tolist():
            st.layouts[env_id] = layout

    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        origins = env.scene.env_origins
        n, dev_ = env.num_envs, env.device
        t_slot = TARGET_SLOT
        # Friction (stick-slip): each drive's target stays put while the joint is within effort-cap / stiffness of it
        # (static friction holds), and is dragged along once a load moves the joint further (sliding at the cap).
        jp = st.pipe.data.joint_pos.torch
        lever_q = jp[:, st.lever_joints]
        win = VALVE_TORQUE_NM / STEM_STIFFNESS
        st.lever_target = torch.minimum(torch.maximum(st.lever_target, lever_q - win), lever_q + win)
        st.pipe.set_joint_position_target(st.lever_target, joint_ids=st.lever_joints)
        slide_q = st.device.data.joint_pos.torch[:, st.slide_joint]
        win = SLIDE_FRICTION_N / SLIDE_STIFFNESS
        st.slide_target = torch.minimum(torch.maximum(st.slide_target, slide_q - win), slide_q + win)
        st.device.set_joint_position_target(st.slide_target.unsqueeze(1), joint_ids=[st.slide_joint])
        q_t = jp[:, st.lever_joints[t_slot]]
        q_d = jp[:, st.lever_joints[1 - t_slot]]
        closed_deg = torch.rad2deg(-q_t)

        # The device's pose against its fitted pose on the named valve.
        a_pos = st.device.data.body_pos_w.torch[:, st.a_id] - origins
        a_quat = st.device.data.body_quat_w.torch[:, st.a_id]
        fit = torch.tensor(device_fitted_pos(t_slot), device=dev_)
        d = a_pos - fit
        x_axis = math_utils.quat_apply(a_quat, torch.tensor([[1.0, 0.0, 0.0]], device=dev_).expand(n, 3))
        z_axis = math_utils.quat_apply(a_quat, torch.tensor([[0.0, 0.0, 1.0]], device=dev_).expand(n, 3))
        yaw_err = torch.rad2deg(torch.atan2(x_axis[:, 1], x_axis[:, 0]).abs())
        seated = ((d[:, :2].abs() < FIT_XY_M).all(dim=1) & (d[:, 2].abs() < FIT_Z_M) & (yaw_err < FIT_YAW_DEG)
                  & (z_axis[:, 2] > math.cos(math.radians(FIT_YAW_DEG))))
        pushed = slide_q < PUSHED_M
        fitted = seated & pushed & (closed_deg > CLOSED_DEG)
        st.fitted_ever |= fitted

        # The padlock's short leg against A's tongue hole, in A's frame.
        travel = st.lock.data.joint_pos.torch[:, st.shackle_joint]
        body_pos = st.lock.data.body_pos_w.torch[:, st.body_id] - origins
        sh_pos = st.lock.data.body_pos_w.torch[:, st.shackle_id] - origins
        sh_quat = st.lock.data.body_quat_w.torch[:, st.shackle_id]

        def on_shackle(p):
            return sh_pos + math_utils.quat_apply(sh_quat, torch.tensor([p], device=dev_).expand(n, 3))

        def to_a(p):
            return math_utils.quat_apply_inverse(a_quat, p - a_pos)

        tip = to_a(on_shackle((PV.leg_x, 0.0, lock_props.SHORT_LEG_BOTTOM)))
        top = to_a(on_shackle((PV.leg_x, 0.0, PV.bend_z - PV.rod / 2)))
        y_mid = (dev.TONGUE_Y[0] + dev.TONGUE_Y[1]) / 2
        denom = top[:, 1] - tip[:, 1]
        denom = torch.where(denom.abs() > 1e-6, denom, torch.full_like(denom, 1e-6))
        a = ((y_mid - tip[:, 1]) / denom).clamp(0.0, 1.0)
        cross = tip + a.unsqueeze(1) * (top - tip)
        in_hole = (cross[:, 0].abs() < dev.HOLE / 2) & ((cross[:, 2] - dev.HOLE_Z).abs() < dev.HOLE / 2)
        spans = (tip[:, 1] < dev.WALL_Y[0]) & (top[:, 1] > dev.TONGUE_Y[1])
        through = in_hole & spans

        newly = ~st.latched & (travel < LATCH_TRAVEL_M)
        if newly.any():
            ids = newly.nonzero().flatten()
            limits = torch.tensor([[[0.0, 0.0005]]], device=dev_).expand(len(ids), 1, 2).contiguous()
            st.lock.write_joint_position_limit_to_sim(limits, joint_ids=[st.shackle_joint], env_ids=ids,
                                                      warn_limit_violation=False)
            st.lock.set_joint_position_target(torch.zeros(len(ids), 1, device=dev_), joint_ids=[st.shackle_joint],
                                              env_ids=ids)
            st.latched |= newly
            st.latched_through |= newly & through

        flange_pos = st.robot.data.body_pos_w.torch[:, st.flange_ids]
        flange_quat = st.robot.data.body_quat_w.torch[:, st.flange_ids]
        down = torch.tensor([0.0, 0.0, -GRASP_OFFSET_M], device=dev_).expand_as(flange_pos)
        grasp = flange_pos + math_utils.quat_apply(flange_quat, down) - origins[:, None]
        clear = (((grasp - body_pos[:, None]).norm(dim=-1) > RELEASE_CLEAR_M).all(dim=1)
                 & ((grasp - fit).norm(dim=-1) > RELEASE_CLEAR_M).all(dim=1))
        st.released |= st.latched & clear

        # Function test.
        ready = (st.latched_through & st.released & (st.phase == 0)) | (st.force_test & (st.phase == 0))
        st.clear_steps = torch.where(ready, st.clear_steps + 1, torch.zeros_like(st.clear_steps))
        start = ready & ((st.clear_steps >= st.wait_steps) | st.force_test)
        if start.any():
            ids = start.nonzero().flatten()
            st.phase[ids] = 1
            st.phase_steps[ids] = 0
        in_test = st.phase > 0
        st.phase_steps = torch.where(in_test, st.phase_steps + 1, st.phase_steps)
        st.slide_max = torch.where(in_test & (st.phase < 3), torch.maximum(st.slide_max, slide_q), st.slide_max)
        open_from_closed = torch.deg2rad(90.0 - closed_deg)
        st.open_max = torch.where(st.phase == 2, torch.maximum(st.open_max, open_from_closed), st.open_max)
        to_lever_phase = (st.phase == 1) & (st.phase_steps >= st.phase_len)
        st.phase_steps = torch.where(to_lever_phase, torch.zeros_like(st.phase_steps), st.phase_steps)
        st.phase = torch.where(to_lever_phase, torch.full_like(st.phase, 2), st.phase)
        # The loads ramp up over TEST_RAMP_S (a hand pulls and pushes; a step load would fling the light halves).
        active = (st.phase == 1) | (st.phase == 2)
        if active.any():
            ids = active.nonzero().flatten()
            k = len(ids)
            st.pipe.permanent_wrench_composer.reset(env_ids=ids)
            st.device.permanent_wrench_composer.reset(env_ids=ids)
            ramp_steps = max(1, round(TEST_RAMP_S / env.step_dt))
            t_pull = torch.where(st.phase[ids] == 1, st.phase_steps[ids], st.phase_steps[ids] + st.phase_len)
            a_pull = (t_pull.float() / ramp_steps).clamp(max=1.0)
            x_dir = x_axis[ids]
            forces = torch.stack([-PULL_N * a_pull.unsqueeze(1) * x_dir, PULL_N * a_pull.unsqueeze(1) * x_dir], dim=1)
            st.device.permanent_wrench_composer.add_forces_and_torques_index(
                forces=forces, torques=torch.zeros_like(forces), body_ids=[st.a_id, st.b_id], env_ids=ids,
                is_global=True)
            a_lever = torch.where(st.phase[ids] == 2, (st.phase_steps[ids].float() / ramp_steps).clamp(max=1.0),
                                  torch.zeros_like(a_pull))
            torque = torch.zeros(k, 1, 3, device=dev_)
            torque[:, 0, 2] = TEST_NM * a_lever  # +z = counterclockwise from above = toward open
            st.pipe.permanent_wrench_composer.add_forces_and_torques_index(
                forces=torch.zeros_like(torque), torques=torque, body_ids=[st.lever_bodies[t_slot]], env_ids=ids,
                is_global=True)
        end = (st.phase == 2) & (st.phase_steps >= st.phase_len)
        if end.any():
            ids = end.nonzero().flatten()
            st.pipe.permanent_wrench_composer.reset(env_ids=ids)
            st.device.permanent_wrench_composer.reset(env_ids=ids)
            st.held[ids] = st.open_max[ids] < math.radians(LEVER_HELD_DEG)
            st.test_passed[ids] = st.held[ids] & st.latched_through[ids] & through[ids]
            st.phase[ids] = 3

        before_test = st.phase == 0
        closed = closed_deg > CLOSED_DEG
        st.was_closed |= closed & before_test
        reopened = before_test & st.was_closed & (closed_deg < REOPEN_DEG)
        decoy_moved = q_d.abs() > math.radians(DECOY_MOVED_DEG)
        st.scorer.update(ValveStep(closed_deg > TURNED_DEG, closed, st.fitted_ever.clone(), st.test_passed.clone(),
                                   decoy_moved, reopened, body_pos[:, 2] < DROP_Z_M, a_pos[:, 2] < DROP_Z_M))
        st.debug = dict(closed_deg=closed_deg, slide_q=slide_q, through=through, travel=travel, a_pos=a_pos,
                        yaw_err=yaw_err, seated=seated, tip=tip, top=top)


@configclass
class ClampEventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: ClampLockoutTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))
