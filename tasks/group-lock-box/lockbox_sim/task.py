"""Group lock box: drop a job lock's key into a Master Lock S601 group lock box and add your padlock to the box.

Mechanism (see lockbox_sim/usd/lockbox_props.py and lockbox_sim/spec.py):
- The lock box is a fixed-base articulation: the body and the lid on its hinge (revolute, 0 = shut, opening negative
  about +x). The task holds the lid shut with a joint limit, as the box's latch does, until the after-release check
  releases it. A shackle through a lid-flange hole and the body-flange hole under it stops the lid opening further
  than the shackle's play (collision).
- The keys are rigid bodies hung on pegs by their bow holes. The key slot is a gap in the lid's top.
- The robot's Master Lock 410 has a sprung shackle: prismatic joint, spring, software latch.
- The coworker's lock is a rigid yellow 410, shut, hung through front hole 5 at reset.

After-release check (the success check): once the key is inside the box, the robot's shackle has latched through a
free hole pair and both grippers have been clear of its padlock for TEST_WAIT_S, the task (1) parks the coworker's
lock under the desk, (2) releases the latch and applies LIFT_N at the lid's free end (as a torque about the hinge) for
TEST_S. Passed if the lid opened less than LID_HELD_DEG, the robot's lock is still through and the key still inside.
``force_function_test`` starts the same check without a robot lock (the control run).
"""

from __future__ import annotations

import math
import os
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

from lockbox_sim import spec as S
from lockbox_sim.kit import GRASP_OFFSET_M, reset_robot_home
from lockbox_sim.scorer import MILESTONES, LockBoxScorer, LockBoxStep
from lockbox_sim.usd import padlock_props as lock_props

_ASSETS = Path(__file__).resolve().parent / "assets"

PV = lock_props.PADLOCKS["ml410"]
CW = lock_props.PRELOCKED
SPRING_N_PER_M = 150.0
"""Shackle spring (N/m), as the other lockout tasks."""
LATCH_TRAVEL_M = 0.002
"""The shackle latches once pushed within this travel of shut."""

LID_DAMPING = 0.2
LATCH_LIMIT_RAD = 0.002
"""While latched the lid can move this little."""
LID_OPEN_LIMIT_DEG = 100.0

KEY_MOVED_M = 0.02
COWORKER_MOVE_M = 0.01
COWORKER_SETTLE_S = 2.0
RELEASE_CLEAR_M = 0.06
DROP_Z_M = 0.03

TEST_WAIT_S = 2.0
TEST_S = 2.0
LIFT_N = 20.0
"""The reviewer lifts the lid's free end with this force (spring scale)."""
LID_LEVER_M = S.HINGE_Y - (S.BOX_Y - S.BODY_L / 2)
LID_HELD_DEG = 15.0
CONTROL_TEST_WAIT_S = 2.0

PARK_COWORKER = (0.30, 0.10, -0.30 + 0.03, 0.0, 0.0, 0.0, 1.0)


def lock_start_pose() -> tuple:
    x, y = S.PEDESTAL_XY
    return (x, y, S.PEDESTAL_TOP + PV.body[2] / 2 + 0.001, 0.0, 0.0, 0.0, 1.0)


def coworker_start_pose() -> tuple:
    """Shut, upright, short leg (+x) through front hole 5, bend resting on the lid flange."""
    hx, hy = S.hole_xy(S.COWORKER_HOLE)
    return (hx - PV.leg_x, hy, S.Z_LID_FLANGE[1] - CW.bend_bottom(0.0) + 0.0005, 0.0, 0.0, 0.0, 1.0)


def key_start_pose(panel: int) -> tuple:
    return tuple(S.key_hang_pos(panel)) + S.KEY_QUAT


def _pose(p) -> Pose:
    return Pose(position_xyz=tuple(p[:3]), rotation_xyzw=tuple(p[3:]))


class LockBox(Object):
    def __init__(self):
        super().__init__(name="lockbox", usd_path=str(_ASSETS / "lockbox.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=_pose((S.BOX_X, S.BOX_Y, 0.0, 0.0, 0.0, 0.0, 1.0)))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {"lid": ImplicitActuatorCfg(joint_names_expr=["lid_joint"], stiffness=0.0,
                                                    damping=LID_DAMPING)}
        cfg.init_state.joint_pos = {"lid_joint": 0.0}
        return cfg


class Fixtures(Object):
    def __init__(self):
        super().__init__(name="fixtures", usd_path=str(_ASSETS / "fixtures.usda"), object_type=ObjectType.BASE,
                         initial_pose=_pose((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)))


class Key(Object):
    def __init__(self, name: str, panel: int):
        super().__init__(name=name, usd_path=str(_ASSETS / "key.usda"), object_type=ObjectType.RIGID,
                         initial_pose=_pose(key_start_pose(panel)))


class Padlock(Object):
    """The robot's purple Master Lock 410 (sprung shackle; the task latches it)."""

    def __init__(self):
        super().__init__(name="padlock", usd_path=str(_ASSETS / "padlock_ml410.usda"),
                         object_type=ObjectType.ARTICULATION, initial_pose=_pose(lock_start_pose()))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {"shackle": ImplicitActuatorCfg(joint_names_expr=["shackle_joint"], stiffness=SPRING_N_PER_M,
                                                        damping=3.0)}
        cfg.init_state.joint_pos = {"shackle_joint": lock_props.OPEN_TRAVEL}
        return cfg


class CoworkerLock(Object):
    def __init__(self):
        super().__init__(name="coworker_lock", usd_path=str(_ASSETS / "coworker.usda"), object_type=ObjectType.RIGID,
                         initial_pose=_pose(coworker_start_pose()))


def _holes(dev) -> tuple[torch.Tensor, torch.Tensor]:
    """(18, 2) hole centres (front 1-9, back 1-9) and (18,) free mask."""
    xy = [S.hole_xy(k, True) for k in range(1, 10)] + [S.hole_xy(k, False) for k in range(1, 10)]
    free = torch.ones(18, dtype=torch.bool, device=dev)
    free[S.COWORKER_HOLE - 1] = False
    return torch.tensor(xy, device=dev), free


class _TaskState:
    def __init__(self, env: ManagerBasedEnv, task: "LockBoxTask"):
        n, dev = env.num_envs, env.device
        self.box = env.scene["lockbox"]
        self.lock = env.scene["padlock"]
        self.key = env.scene["key"]
        self.decoy = env.scene["other_key"]
        self.coworker = env.scene["coworker_lock"]
        self.robot = env.scene["robot"]
        self.lid_joint = self.box.find_joints("lid_joint")[0][0]
        self.lid_body = self.box.find_bodies("lid")[0][0]
        self.shackle_joint = self.lock.find_joints("shackle_joint")[0][0]
        self.body_id = self.lock.find_bodies("body")[0][0]
        self.shackle_id = self.lock.find_bodies("shackle")[0][0]
        self.flange_ids, _ = self.robot.find_bodies(["left_gripper", "right_gripper"], preserve_order=True)
        self.scorer = LockBoxScorer(n, dev)
        self.last_reports: dict[int, dict] = {}
        self.holes, self.free = _holes(dev)
        zb = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
        zl = lambda: torch.zeros(n, dtype=torch.long, device=dev)  # noqa: E731
        self.latched, self.latched_through, self.released, self.force_test = zb(), zb(), zb(), zb()
        self.test_passed, self.lid_held, self.cw_ref_set = zb(), zb(), zb()
        self.hole_used = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.clear_steps, self.phase, self.phase_steps, self.inside_steps = zl(), zl(), zl(), zl()
        self.lid_max = torch.zeros(n, device=dev)
        self.cw_ref = torch.zeros(n, 3, device=dev)
        self.wait_steps = max(1, round(TEST_WAIT_S / env.step_dt))
        self.test_len = max(1, round(TEST_S / env.step_dt))
        self.settle_steps = max(1, round(COWORKER_SETTLE_S / env.step_dt))
        self.control_wait = max(1, round(CONTROL_TEST_WAIT_S / env.step_dt))
        self.control = task.control


def _write_pose(asset, pose, env_ids, origins):
    dev, n = origins.device, len(env_ids)
    p = torch.tensor(pose, device=dev, dtype=torch.float32).unsqueeze(0).repeat(n, 1)
    p[:, :3] += origins[env_ids]
    asset.write_root_pose_to_sim(p, env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)


def _set_lid_limit(st: _TaskState, ids, lower: float, upper: float) -> None:
    limits = torch.tensor([[[lower, upper]]], device=ids.device).expand(len(ids), 1, 2).contiguous()
    st.box.write_joint_position_limit_to_sim(limits, joint_ids=[st.lid_joint], env_ids=ids, warn_limit_violation=False)


class LockBoxTask(TaskBase):
    def __init__(self, default_seed: int = 0, episode_length_s: float = S.EPISODE_SECONDS, control: bool = False):
        super().__init__(episode_length_s=episode_length_s, task_description=S.instruction())
        self.default_seed = default_seed
        self.control = control

    def state(self, env: ManagerBasedEnv) -> _TaskState:
        if getattr(env, "_task_state", None) is None:
            env._task_state = _TaskState(env, self)
        return env._task_state

    def set_next_seed(self, env: ManagerBasedEnv, seed: int) -> None:
        pass

    def force_function_test(self, env: ManagerBasedEnv) -> None:
        self.state(env).force_test[:] = True

    def episode_report(self, env: ManagerBasedEnv, env_id: int = 0, ended: bool = False) -> dict:
        st = self.state(env)
        if ended and env_id in st.last_reports:
            return st.last_reports.pop(env_id)
        return self._report(st, env_id)

    def _report(self, st: _TaskState, env_id: int) -> dict:
        r = st.scorer.report(env_id)
        if st.scorer.hazard[env_id]:
            r["milestone_before_hazard"] = r["milestone"]
        r.update(
            shackle_latched=bool(st.latched[env_id]), shackle_latched_through_free_hole=bool(st.latched_through[env_id]),
            hole_used=int(st.hole_used[env_id]), check_phase=int(st.phase[env_id]),
            check_lid_max_deg=round(math.degrees(float(st.lid_max[env_id])), 1), check_lid_held=bool(st.lid_held[env_id]),
            check_passed=bool(st.test_passed[env_id]),
        )
        return r

    def get_scene_cfg(self):
        if os.environ.get("LOCKBOX_DEBUG_CAM") == "1":
            return _debug_cams()
        return None

    def get_events_cfg(self):
        return LockBoxEventsCfg(self)

    def get_termination_cfg(self) -> TaskTerminationCfg:
        sequence = [(self._milestone_predicate(m), 0.25) for m in MILESTONES]
        return TaskTerminationCfg(timeout_s=self.episode_length_s,
                                  success=[CompletionCriteria(name="lockbox_milestones", predicate_sequence=sequence)])

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def _milestone_predicate(self, milestone: int):
        def predicate(env: ManagerBasedEnv) -> torch.Tensor:
            st = self.state(env)
            return (st.scorer.milestone >= milestone) & ~st.scorer.hazard

        predicate.__name__ = f"milestone_{milestone}"
        return predicate

    # Events ------------------------------------------------------------------------------------
    def reset(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        reset_robot_home(env, env_ids)
        for env_id in env_ids.tolist():
            if int(st.scorer.step[env_id]) > 0:
                st.last_reports[env_id] = self._report(st, env_id)
        dev, n = env.device, len(env_ids)
        origins = env.scene.env_origins
        st.scorer.reset(env_ids)
        for buf in (st.latched, st.latched_through, st.released, st.force_test, st.test_passed, st.lid_held,
                    st.cw_ref_set):
            buf[env_ids] = False
        for buf in (st.clear_steps, st.phase, st.phase_steps, st.inside_steps):
            buf[env_ids] = 0
        st.lid_max[env_ids] = 0.0
        st.hole_used[env_ids] = -1
        st.box.permanent_wrench_composer.reset(env_ids=env_ids)

        # Lid shut and latched.
        zero = torch.zeros(n, 1, device=dev)
        _set_lid_limit(st, env_ids, -LATCH_LIMIT_RAD, 0.0)
        st.box.write_joint_state_to_sim(zero, zero, joint_ids=[st.lid_joint], env_ids=env_ids)
        st.box.set_joint_position_target(zero, joint_ids=[st.lid_joint], env_ids=env_ids)

        # The robot's padlock: open, unlatched, on its pedestal.
        joint = [st.shackle_joint]
        limits = torch.tensor([[[0.0, lock_props.OPEN_TRAVEL]]], device=dev).expand(n, 1, 2).contiguous()
        travel = torch.full((n, 1), lock_props.OPEN_TRAVEL, device=dev)
        st.lock.write_joint_position_limit_to_sim(limits, joint_ids=joint, env_ids=env_ids, warn_limit_violation=False)
        st.lock.write_joint_state_to_sim(travel, torch.zeros_like(travel), joint_ids=joint, env_ids=env_ids)
        st.lock.set_joint_position_target(travel + 0.005, joint_ids=joint, env_ids=env_ids)
        _write_pose(st.lock, lock_start_pose(), env_ids, origins)

        _write_pose(st.key, key_start_pose(S.TARGET_PANEL), env_ids, origins)
        _write_pose(st.decoy, key_start_pose(1 - S.TARGET_PANEL), env_ids, origins)
        _write_pose(st.coworker, coworker_start_pose(), env_ids, origins)

    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        origins = env.scene.env_origins
        n, dev = env.num_envs, env.device
        lid_angle = -st.box.data.joint_pos.torch[:, st.lid_joint]

        # Keys.
        key_pos = st.key.data.root_pos_w.torch - origins
        decoy_pos = st.decoy.data.root_pos_w.torch - origins
        home = torch.tensor(S.key_hang_pos(S.TARGET_PANEL), device=dev)
        decoy_home = torch.tensor(S.key_hang_pos(1 - S.TARGET_PANEL), device=dev)
        key_off = (key_pos - home).norm(dim=-1) > KEY_MOVED_M
        decoy_moved = (decoy_pos - decoy_home).norm(dim=-1) > KEY_MOVED_M
        rel = key_pos - torch.tensor([S.BOX_X, S.BOX_Y, 0.0], device=dev)
        key_inside = ((rel[:, 0].abs() < S.BODY_W / 2 - S.WALL_T) & (rel[:, 1].abs() < S.BODY_L / 2 - S.WALL_T)
                      & (key_pos[:, 2] > S.Z_BASE) & (key_pos[:, 2] < S.Z_TOP - 0.005))

        # The robot's padlock: its short leg against the flange holes (box fixed; the lid is latched until the check).
        travel = st.lock.data.joint_pos.torch[:, st.shackle_joint]
        body_pos = st.lock.data.body_pos_w.torch[:, st.body_id] - origins
        sh_pos = st.lock.data.body_pos_w.torch[:, st.shackle_id] - origins
        sh_quat = st.lock.data.body_quat_w.torch[:, st.shackle_id]

        def on_shackle(p):
            return sh_pos + math_utils.quat_apply(sh_quat, torch.tensor([p], device=dev).expand(n, 3))

        tip = on_shackle((PV.leg_x, 0.0, lock_props.SHORT_LEG_BOTTOM))
        top = on_shackle((PV.leg_x, 0.0, PV.bend_z - PV.rod / 2))
        z_lo, z_hi = S.Z_BODY_FLANGE[0], S.Z_LID_FLANGE[1]
        zm = (z_lo + z_hi) / 2
        denom = top[:, 2] - tip[:, 2]
        denom = torch.where(denom.abs() > 1e-6, denom, torch.full_like(denom, 1e-6))
        a = ((zm - tip[:, 2]) / denom).clamp(0.0, 1.0)
        cross = tip + a.unsqueeze(1) * (top - tip)
        d = cross[:, None, :2] - st.holes[None]  # (N, 18, 2)
        in_hole = (d[..., 0].abs() < S.HOLE_X / 2) & (d[..., 1].abs() < S.HOLE_Y / 2) & st.free[None]
        spans = (tip[:, 2] < z_lo) & (top[:, 2] > z_hi)
        through_any = in_hole.any(dim=1) & spans

        newly = ~st.latched & (travel < LATCH_TRAVEL_M)
        if newly.any():
            ids = newly.nonzero().flatten()
            limits = torch.tensor([[[0.0, 0.0005]]], device=dev).expand(len(ids), 1, 2).contiguous()
            st.lock.write_joint_position_limit_to_sim(limits, joint_ids=[st.shackle_joint], env_ids=ids,
                                                      warn_limit_violation=False)
            st.lock.set_joint_position_target(torch.zeros(len(ids), 1, device=dev), joint_ids=[st.shackle_joint],
                                              env_ids=ids)
            st.latched |= newly
            st.latched_through |= newly & through_any
            st.hole_used = torch.where(newly & through_any, in_hole.float().argmax(dim=1), st.hole_used)

        flange_pos = st.robot.data.body_pos_w.torch[:, st.flange_ids]
        flange_quat = st.robot.data.body_quat_w.torch[:, st.flange_ids]
        down = torch.tensor([0.0, 0.0, -GRASP_OFFSET_M], device=dev).expand_as(flange_pos)
        grasp = flange_pos + math_utils.quat_apply(flange_quat, down) - origins[:, None]
        clear = ((grasp - body_pos[:, None]).norm(dim=-1) > RELEASE_CLEAR_M).all(dim=1)
        st.released = st.latched & clear

        # Coworker's lock: reference once settled; moved before the robot's shackle shut.
        cw_pos = st.coworker.data.root_pos_w.torch - origins
        settle = ~st.cw_ref_set & (st.scorer.step + 1 >= st.settle_steps)
        if settle.any():
            st.cw_ref[settle] = cw_pos[settle]
            st.cw_ref_set |= settle
        cw_moved = st.cw_ref_set & ~st.latched & (st.phase == 0) & ((cw_pos - st.cw_ref).norm(dim=-1) > COWORKER_MOVE_M)

        # After-release check.
        st.inside_steps = torch.where(key_inside, st.inside_steps + 1, torch.zeros_like(st.inside_steps))
        if st.control:
            st.force_test |= st.inside_steps >= st.control_wait
        ready = ((st.latched_through & st.released & key_inside) | st.force_test) & (st.phase == 0)
        st.clear_steps = torch.where(ready, st.clear_steps + 1, torch.zeros_like(st.clear_steps))
        start = ready & ((st.clear_steps >= st.wait_steps) | st.force_test)
        if start.any():
            ids = start.nonzero().flatten()
            _write_pose(st.coworker, PARK_COWORKER, ids, origins)
            _set_lid_limit(st, ids, -math.radians(LID_OPEN_LIMIT_DEG), 0.0)
            torque = torch.zeros(len(ids), 1, 3, device=dev)
            torque[..., 0] = -LIFT_N * LID_LEVER_M
            st.box.permanent_wrench_composer.add_forces_and_torques_index(
                forces=torch.zeros_like(torque), torques=torque, body_ids=[st.lid_body], env_ids=ids, is_global=True)
            st.phase[ids] = 1
            st.phase_steps[ids] = 0
        st.phase_steps = torch.where(st.phase > 0, st.phase_steps + 1, st.phase_steps)
        st.lid_max = torch.where(st.phase == 1, torch.maximum(st.lid_max, lid_angle), st.lid_max)
        end = (st.phase == 1) & (st.phase_steps >= st.test_len)
        if end.any():
            ids = end.nonzero().flatten()
            st.box.permanent_wrench_composer.reset(env_ids=ids)
            st.lid_held[ids] = st.lid_max[ids] < math.radians(LID_HELD_DEG)
            st.test_passed[ids] = (st.lid_held[ids] & st.latched_through[ids] & through_any[ids] & key_inside[ids]
                                   & ~st.force_test[ids])
            st.phase[ids] = 2

        dropped = body_pos[:, 2] < DROP_Z_M
        st.scorer.update(LockBoxStep(key_off, key_inside, through_any & (st.phase == 0), st.test_passed.clone(),
                                     decoy_moved, cw_moved, dropped))


@configclass
class LockBoxEventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: LockBoxTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))


DEBUG_VIEWS = {
    "debug_cam": ((0.06, 0.12, 0.26), (0.27, 0.02, 0.11)),
    "debug_cam2": ((0.22, -0.13, 0.17), (0.308, -0.227, 0.10)),
    "debug_cam3": ((0.20, -0.30, 0.20), (0.27, 0.03, 0.12)),
}
"""Development-only views (LOCKBOX_DEBUG_CAM=1), desk frame: the box's front, the key panels, the flange from the side."""


def _debug_cams():
    """Extra 640 x 360 cameras for development; not part of the policy's observations."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg
    from yam_arena.embodiment import RENDERER

    from lockbox_sim.kit import look_at_xyzw
    from lockbox_sim.station import Z0

    fields = {}
    for name, (eye, target) in DEBUG_VIEWS.items():
        e = (eye[0], eye[1], eye[2] + Z0)
        t = (target[0], target[1], target[2] + Z0)
        fields[name] = CameraCfg(
            prim_path=f"{{ENV_REGEX_NS}}/{name}", update_period=0.0, width=640, height=360, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, horizontal_aperture=20.955, clipping_range=(0.02, 10.0)),
            offset=CameraCfg.OffsetCfg(pos=e, rot=look_at_xyzw(e, t), convention="world"), renderer_cfg=RENDERER)

    @configclass
    class DebugCamsCfg:
        debug_cam: CameraCfg = fields["debug_cam"]
        debug_cam2: CameraCfg = fields["debug_cam2"]
        debug_cam3: CameraCfg = fields["debug_cam3"]

    return DebugCamsCfg()
