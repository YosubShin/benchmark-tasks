"""Handwheel valve lockout: close the named handwheel gate valve and lock it closed with a rotating gate valve lockout
(Master Lock 480) and a padlock.

Mechanism (see wheel_lockout_sim/usd/cover_props.py for the cover, usd/valve_props.py for the valve):
- Each valve's handwheel turns on a revolute joint about its stem (clockwise from above = decreasing q), from
  WHEEL_START (one turn open) to WHEEL_SEAT, the closed stop (joint limit). Turning resistance: stick-slip friction, a
  stiff drive whose effort is capped at VALVE_TORQUE_NM and whose target follows the joint only once the joint has
  moved past what that effort holds.
- The cover: half A (the fixed piece, with the open sector and the stem opening) is a free rigid body; half B (the
  rotating piece) turns on it about the drum axis (revolute joint, light stick-slip friction, SWING_FRICTION_NM).
  Fitted, the wheel lies between the two faces and the stem in the back face's opening; closed, the drum encloses
  the wheel all round, so it cannot be gripped, and the back face behind the rim stops the cover being pulled off.
  Closed, B's padlock ear lies on A's ear; the padlock's shackle through both ears' holes stops B swinging open
  (collision). Nothing in the code holds the cover. The cover rests on the wheel nut by A's pivot boss.
- The robot's Master Lock 410: the shackle on a prismatic joint with a weak spring; the task latches it in software
  once it is pushed within 2 mm of shut.

Function test (the success check): once the shackle has latched through both ears and both grippers have been clear of
the padlock and the stem for TEST_WAIT_S, the scorer pries the two halves apart (PRY_NM between them, B toward open)
for the whole test, in three TEST_PHASE_S phases: 1 the pry alone; 2 the cover turned TURN_DEG toward open about the
stem (a soft, damped pull, at most TURN_NM); 3 the cover lifted (LIFT_N). Passed if B stayed within SWING_HELD_DEG of
closed, the wheel within WHEEL_HELD_DEG of its seat, the cover on the wheel (COVER_HELD_*), and the shackle is still
through the ears. `force_function_test(env)` starts the same test at once (control run).
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

from wheel_lockout_sim.kit import GRASP_OFFSET_M, reset_robot_home
from wheel_lockout_sim.scorer import MILESTONES, ValveScorer, ValveStep
from wheel_lockout_sim.spec import ValveSeed, make_seed
from wheel_lockout_sim.usd import cover_props as cov
from wheel_lockout_sim.usd import padlock_props as lock_props
from wheel_lockout_sim.usd import valve_props as props

_ASSETS = Path(__file__).resolve().parent / "assets"
SPRING_N_PER_M = 150.0
"""Shackle spring (inferred): 2.4 N at full 16 mm travel."""
LATCH_TRAVEL_M = 0.002

PV = lock_props.PADLOCKS["ml410"]
LOCK_QUAT = (0.0, 0.0, -math.sqrt(0.5), math.sqrt(0.5))
"""The padlock upright (shackle up), its legs side by side along y with the short leg on the right (-y), body thickness
along x."""

VALVE_TORQUE_NM = 0.3
"""Sim turning torque at the handwheel (breakaway = running). Real: not published for this valve; see README."""
SWING_FRICTION_NM = 0.0005
WHEEL_STIFFNESS = 3000.0
WHEEL_DAMPING = 0.3
SWING_STIFFNESS = float(__import__("os").environ.get("WL_SWING_K", "50.0"))

WHEEL_START = math.radians(props.WHEEL_START_DEG)
WHEEL_SEAT = math.radians(props.WHEEL_SEAT_DEG)
TURNED_DEG = 90.0
CLOSED_DEG = 10.0
"""Closed: within this of the seat."""
REOPEN_DEG = 30.0
DECOY_MOVED_DEG = 10.0
FIT_XY_M = 0.006
FIT_Z_M = 0.005
FIT_TILT_DEG = 8.0
B_CLOSED_DEG = 3.0
RELEASE_CLEAR_M = 0.06
DROP_Z_M = 0.05

DONE_PHASE = 4
TEST_DEBUG = bool(int(__import__("os").environ.get("WL_TEST_DEBUG", "0")))
TEST_WAIT_S = 2.0
TEST_PHASE_S = 2.0
TEST_RAMP_S = 1.0
PRY_NM = 0.83
"""Prying the halves apart: 15 N on B's ear toward open (and on A's ear the other way), 55 mm from the axis."""
LIFT_N = 10.0
TURN_NM = 0.66
"""Reported spring-scale reading for turning the cover by its ear (12 N at 55 mm); the simulated turn is a steady
velocity (see step), which the free-turning cover follows with far less."""
TURN_DEG = 30.0
"""The cover is turned this far toward open (a twelfth of a turn; further, the hanging padlock would meet the pipe)."""
SWING_HELD_DEG = 10.0
WHEEL_HELD_DEG = 10.0
COVER_HELD_XY_M = 0.015
COVER_HELD_Z_M = 0.025
"""The cover stays on the wheel: centre within 15 mm of the stem, lifted no more than 25 mm (the lift stops when its
back face meets the underside of the wheel's rim, about 17 mm up)."""

TARGET_SLOT = 0


def lock_start_pose() -> tuple:
    x, y = props.PEDESTAL_XY
    return (x, y, props.PEDESTAL_TOP + PV.body[2] / 2 + 0.001) + LOCK_QUAT


def cover_start_pose() -> tuple:
    x, y = props.COVER_PED_XY
    return (x, y, props.COVER_PED_TOP + cov.A_FACE_Z + cov.FACE_T + 0.0005, 0.0, 0.0, 0.0, 1.0)


def cover_fitted_pos(slot: int = TARGET_SLOT) -> tuple:
    """Half A's origin when the cover rests on the named valve's wheel nut (A's pivot boss on the nut)."""
    return (props.VALVE_X, props.VALVE_Y[slot], props.NUT_TOP - cov.BOSS_Z)


class PipeRun(Object):
    def __init__(self):
        super().__init__(name="pipe_run", usd_path=str(_ASSETS / "pipe_run.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {
            "wheels": ImplicitActuatorCfg(joint_names_expr=["wheel_.*_joint"], stiffness=WHEEL_STIFFNESS,
                                          damping=WHEEL_DAMPING, effort_limit_sim=VALVE_TORQUE_NM),
        }
        cfg.init_state.joint_pos = {"wheel_.*_joint": WHEEL_START}
        return cfg


class Cover(Object):
    """The Master Lock 480 (two halves on a pivot)."""

    def __init__(self):
        p = cover_start_pose()
        super().__init__(name="cover", usd_path=str(_ASSETS / "ml480.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=p[:3], rotation_xyzw=p[3:]))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {"swing": ImplicitActuatorCfg(joint_names_expr=["swing_joint"], stiffness=SWING_STIFFNESS,
                                                      damping=0.0001, effort_limit_sim=SWING_FRICTION_NM)}
        cfg.init_state.joint_pos = {"swing_joint": math.radians(cov.Q_OPEN)}
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
    def __init__(self, env: ManagerBasedEnv, task: "WheelLockoutTask"):
        n, dev_ = env.num_envs, env.device
        self.lock = env.scene["padlock"]
        self.pipe = env.scene["pipe_run"]
        self.cover = env.scene["cover"]
        self.robot = env.scene["robot"]
        self.shackle_joint = self.lock.find_joints("shackle_joint")[0][0]
        self.body_id = self.lock.find_bodies("body")[0][0]
        self.shackle_id = self.lock.find_bodies("shackle")[0][0]
        self.wheel_joints = [self.pipe.find_joints(f"wheel_{k}_joint")[0][0] for k in range(2)]
        self.wheel_bodies = [self.pipe.find_bodies(f"wheel_{k}")[0][0] for k in range(2)]
        self.swing_joint = self.cover.find_joints("swing_joint")[0][0]
        self.a_id = self.cover.find_bodies("half_a")[0][0]
        self.b_id = self.cover.find_bodies("half_b")[0][0]
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
        self.swing_max, self.wheel_max, self.cover_xy_max, self.cover_rise_max = zf(), zf(), zf(), zf()
        self.yaw_prev, self.yaw_turned, self.yaw_at_turn = zf(), zf(), zf()
        self.test_done = zb()
        self.wheel_target = torch.full((n, 2), WHEEL_START, device=dev_)
        self.swing_target = torch.full((n,), math.radians(cov.Q_OPEN), device=dev_)
        self.wait_steps = max(1, round(TEST_WAIT_S / env.step_dt))
        self.phase_len = max(1, round(TEST_PHASE_S / env.step_dt))


def _write_pose(asset, pose, env_ids, origins):
    dev_, n = origins.device, len(env_ids)
    p = torch.tensor(pose, device=dev_, dtype=torch.float32).unsqueeze(0).repeat(n, 1)
    p[:, :3] += origins[env_ids]
    asset.write_root_pose_to_sim(p, env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev_), env_ids=env_ids)


class WheelLockoutTask(TaskBase):
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
            target_tag=layout.target_tag, valve_torque_nm=VALVE_TORQUE_NM, pry_nm=PRY_NM, lift_n=LIFT_N,
            turn_nm=TURN_NM, cover_fitted=bool(st.fitted_ever[env_id]),
            shackle_latched=bool(st.latched[env_id]), shackle_latched_through_ears=bool(st.latched_through[env_id]),
            function_test_phase=int(st.phase[env_id]),
            test_swing_max_open_deg=round(math.degrees(float(st.swing_max[env_id])), 1),
            test_wheel_max_open_deg=round(math.degrees(float(st.wheel_max[env_id])), 1),
            test_cover_turned_deg=round(math.degrees(float(st.yaw_turned[env_id] - st.yaw_at_turn[env_id])), 1),
            test_cover_off_axis_mm=round(1000 * float(st.cover_xy_max[env_id]), 1),
            test_cover_rise_mm=round(1000 * float(st.cover_rise_max[env_id]), 1),
            test_held=bool(st.held[env_id]), function_test_passed=bool(st.test_passed[env_id]),
        )
        return r

    def get_scene_cfg(self):
        return None

    def get_events_cfg(self):
        return WheelEventsCfg(self)

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
        st.test_done[env_ids] = False
        for buf in (st.swing_max, st.wheel_max, st.cover_xy_max, st.cover_rise_max, st.yaw_prev, st.yaw_turned,
                    st.yaw_at_turn):
            buf[env_ids] = 0.0
        for asset in (st.lock, st.pipe, st.cover):
            asset.permanent_wrench_composer.reset(env_ids=env_ids)

        # The robot's padlock: open, unlatched, upright on its column.
        joint = [st.shackle_joint]
        limits = torch.tensor([[[0.0, lock_props.OPEN_TRAVEL]]], device=dev_).expand(n, 1, 2).contiguous()
        travel = torch.full((n, 1), lock_props.OPEN_TRAVEL, device=dev_)
        st.lock.write_joint_position_limit_to_sim(limits, joint_ids=joint, env_ids=env_ids, warn_limit_violation=False)
        st.lock.write_joint_state_to_sim(travel, torch.zeros_like(travel), joint_ids=joint, env_ids=env_ids)
        st.lock.set_joint_position_target(travel + 0.005, joint_ids=joint, env_ids=env_ids)
        _write_pose(st.lock, lock_start_pose(), env_ids, origins)

        # The cover: lying open on its pedestal (B nested in A).
        q = torch.full((n, 1), math.radians(cov.Q_OPEN), device=dev_)
        st.cover.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=[st.swing_joint], env_ids=env_ids)
        st.cover.set_joint_position_target(q, joint_ids=[st.swing_joint], env_ids=env_ids)
        st.swing_target[env_ids] = math.radians(cov.Q_OPEN)
        _write_pose(st.cover, cover_start_pose(), env_ids, origins)

        # Both valves one turn open.
        q = torch.full((n, 2), WHEEL_START, device=dev_)
        st.pipe.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=st.wheel_joints, env_ids=env_ids)
        st.pipe.set_joint_position_target(q, joint_ids=st.wheel_joints, env_ids=env_ids)
        st.wheel_target[env_ids] = WHEEL_START
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
        wheel_q = jp[:, st.wheel_joints]
        win = VALVE_TORQUE_NM / WHEEL_STIFFNESS
        st.wheel_target = torch.minimum(torch.maximum(st.wheel_target, wheel_q - win), wheel_q + win)
        st.pipe.set_joint_position_target(st.wheel_target, joint_ids=st.wheel_joints)
        swing_q = st.cover.data.joint_pos.torch[:, st.swing_joint]
        if SWING_STIFFNESS > 0:
            win = SWING_FRICTION_NM / SWING_STIFFNESS
            st.swing_target = torch.minimum(torch.maximum(st.swing_target, swing_q - win), swing_q + win)
            st.cover.set_joint_position_target(st.swing_target.unsqueeze(1), joint_ids=[st.swing_joint])
        q_t = wheel_q[:, t_slot]
        q_d = wheel_q[:, 1 - t_slot]
        turned_deg = torch.rad2deg(WHEEL_START - q_t)
        from_seat_deg = torch.rad2deg(q_t - WHEEL_SEAT)
        swing_deg = torch.rad2deg(swing_q)

        # The cover's pose against its fitted pose on the named valve (any turn about the stem).
        a_pos = st.cover.data.body_pos_w.torch[:, st.a_id] - origins
        a_quat = st.cover.data.body_quat_w.torch[:, st.a_id]
        fit = torch.tensor(cover_fitted_pos(t_slot), device=dev_)
        d = a_pos - fit
        x_axis = math_utils.quat_apply(a_quat, torch.tensor([[1.0, 0.0, 0.0]], device=dev_).expand(n, 3))
        z_axis = math_utils.quat_apply(a_quat, torch.tensor([[0.0, 0.0, 1.0]], device=dev_).expand(n, 3))
        yaw = torch.atan2(x_axis[:, 1], x_axis[:, 0])
        seated = ((d[:, :2].norm(dim=1) < FIT_XY_M) & (d[:, 2].abs() < FIT_Z_M)
                  & (z_axis[:, 2] > math.cos(math.radians(FIT_TILT_DEG))))
        b_closed = swing_deg > -B_CLOSED_DEG
        closed = from_seat_deg < CLOSED_DEG
        fitted = seated & b_closed & closed
        st.fitted_ever |= fitted

        # The padlock's short leg against the ears' inner holes, in A's frame.
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
        hx, hy, hz = cov.hole_in_cover(0)
        denom = top[:, 2] - tip[:, 2]
        denom = torch.where(denom.abs() > 1e-6, denom, torch.full_like(denom, 1e-6))
        a = ((hz - tip[:, 2]) / denom).clamp(0.0, 1.0)
        cross = tip + a.unsqueeze(1) * (top - tip)
        in_hole = ((cross[:, 0] - hx).abs() < cov.HOLE / 2) & ((cross[:, 1] - hy).abs() < cov.HOLE / 2)
        spans = (tip[:, 2] < cov.STACK_Z[0]) & (top[:, 2] > cov.STACK_Z[1])
        through = in_hole & spans & (swing_deg > -2 * B_CLOSED_DEG)

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
            st.yaw_prev[ids] = yaw[ids]
            st.yaw_turned[ids] = 0.0
        # Phases (TEST_PHASE_S each, the pry on throughout): 1 pry; 2 pry + turn the cover; 3 pry + lift the cover.
        in_test = (st.phase > 0) & (st.phase < DONE_PHASE)
        st.phase_steps = torch.where(in_test, st.phase_steps + 1, st.phase_steps)
        st.phase = torch.where(in_test, torch.clamp(1 + st.phase_steps // st.phase_len, max=DONE_PHASE), st.phase)
        in_test = (st.phase > 0) & (st.phase < DONE_PHASE)
        # The cover's turn about the stem since the test began (unwrapped).
        dyaw = torch.remainder(yaw - st.yaw_prev + math.pi, 2 * math.pi) - math.pi
        st.yaw_turned = torch.where(in_test, st.yaw_turned + dyaw, st.yaw_turned)
        st.yaw_prev = yaw
        st.swing_max = torch.where(in_test, torch.maximum(st.swing_max, -swing_q), st.swing_max)
        st.wheel_max = torch.where(in_test, torch.maximum(st.wheel_max, q_t - WHEEL_SEAT), st.wheel_max)
        st.cover_xy_max = torch.where(in_test, torch.maximum(st.cover_xy_max, d[:, :2].norm(dim=1)), st.cover_xy_max)
        st.cover_rise_max = torch.where(in_test, torch.maximum(st.cover_rise_max, d[:, 2]), st.cover_rise_max)
        # The loads ramp up over TEST_RAMP_S (a hand pulls and turns; a step load would fling the light cover).
        if in_test.any():
            ids = in_test.nonzero().flatten()
            k = len(ids)
            st.cover.permanent_wrench_composer.reset(env_ids=ids)
            ramp_steps = max(1, round(TEST_RAMP_S / env.step_dt))
            t_all = st.phase_steps[ids]
            t_phase = t_all - (st.phase[ids] - 1) * st.phase_len
            a_all = (t_all.float() / ramp_steps).clamp(max=1.0)
            a_phase = (t_phase.float() / ramp_steps).clamp(max=1.0)
            zA = z_axis[ids]
            # Pry: B toward open (clockwise about A's axis from above, -z), A the other way.
            pry = PRY_NM * a_all.unsqueeze(1) * zA
            forces = torch.zeros(k, 2, 3, device=dev_)
            torques = torch.stack([pry, -pry], dim=1)
            # Turn (phase 2): a hand turns the cover by its ear toward open (counterclockwise from above) at a steady
            # TURN_DEG over the phase, as far as TURN_DEG. A torque servo at the 20 Hz control rate is unstable on the
            # light cover, so the hand is a velocity: A's spin about the vertical is set each control step.
            turned = st.yaw_turned[ids] - st.yaw_at_turn[ids]
            turning = (st.phase[ids] == 2) & (turned < math.radians(TURN_DEG))
            if turning.any():
                tid = ids[turning]
                vel = st.cover.data.root_vel_w.torch[tid].clone()
                vel[:, 5] = math.radians(TURN_DEG) / TEST_PHASE_S
                st.cover.write_root_velocity_to_sim(vel, env_ids=tid)
            # Lift (phase 3): A pulled straight up.
            forces[:, 0, 2] = torch.where(st.phase[ids] == 3, LIFT_N * a_phase, torch.zeros_like(a_phase))
            st.cover.permanent_wrench_composer.add_forces_and_torques_index(
                forces=forces, torques=torques, body_ids=[st.a_id, st.b_id], env_ids=ids, is_global=True)
            if TEST_DEBUG and int(ids[0]) == 0:
                print(f"[wheel-test] t={int(t_all[0])} turned {math.degrees(float(turned[0])):.1f}", flush=True)
            st.yaw_at_turn[ids] = torch.where(st.phase[ids] == 1, st.yaw_turned[ids], st.yaw_at_turn[ids])
        if TEST_DEBUG and bool(in_test[0]) and int(st.phase_steps[0]) % 4 == 0:
            print(f"[wheel-test] t={int(st.phase_steps[0])} phase {int(st.phase[0])}: swing {float(swing_deg[0]):.1f} deg, "
                  f"A {[round(float(v), 4) for v in a_pos[0]]} yaw {math.degrees(float(yaw[0])):.1f}, wheel "
                  f"{math.degrees(float(q_t[0] - WHEEL_SEAT)):.1f} deg, lock z {float(body_pos[0, 2]):.4f}, through "
                  f"{bool(through[0])}", flush=True)
        end = (st.phase == DONE_PHASE) & ~st.test_done
        if end.any():
            ids = end.nonzero().flatten()
            st.cover.permanent_wrench_composer.reset(env_ids=ids)
            st.held[ids] = ((st.swing_max[ids] < math.radians(SWING_HELD_DEG))
                            & (st.wheel_max[ids] < math.radians(WHEEL_HELD_DEG))
                            & (st.cover_xy_max[ids] < COVER_HELD_XY_M) & (st.cover_rise_max[ids] < COVER_HELD_Z_M))
            st.test_passed[ids] = st.held[ids] & st.latched_through[ids] & through[ids]
            st.test_done[ids] = True

        before_test = st.phase == 0
        st.was_closed |= closed & before_test
        reopened = before_test & st.was_closed & (from_seat_deg > REOPEN_DEG)
        decoy_moved = (q_d - WHEEL_START).abs() > math.radians(DECOY_MOVED_DEG)
        st.scorer.update(ValveStep(turned_deg > TURNED_DEG, closed, st.fitted_ever.clone(), st.test_passed.clone(),
                                   decoy_moved, reopened, body_pos[:, 2] < DROP_Z_M, a_pos[:, 2] < DROP_Z_M))
        st.debug = dict(turned_deg=turned_deg, swing_deg=swing_deg, through=through, travel=travel, a_pos=a_pos,
                        seated=seated, tip=tip, top=top, yaw=yaw)


@configclass
class WheelEventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: WheelLockoutTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))
