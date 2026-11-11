"""Pull-out disconnect lockout: isolate AC-1 at its Milbank U3802 pull-out disconnect and padlock the box.

Mechanism (see pullout_sim/usd/disconnect_props.py and pullout_sim/spec.py):
- Each disconnect is a fixed-base articulation: the body (with the pull-out's recess, four live clips, four dummy
  pockets and the padlock tab) and the cover on its hinge (revolute about y, 0 = shut, opening positive, stop at
  120 degrees). Nothing holds the cover shut but its weight; a shackle through the tab's hole, which sticks out
  through the slot in the cover's front, stops the cover's bottom from swinging out (collision).
- The pull-outs are free rigid bodies. Seated ON their blades sit in the live clips, whose fingers grip them by
  compliant contact (the pull needed to take the block out comes from the clip spring and friction); turned over
  (OFF) the blades sit loosely in the dummy pockets.
- The robot's Master Lock 410 has a sprung shackle: prismatic joint, spring, software latch.

After-release check (the success check): once AC-1's block is seated OFF, its cover is shut, the robot's shackle
has latched through the tab's hole and both grippers have been clear for TEST_WAIT_S, the task pulls the cover's
bottom edge outward with LIFT_N (as a torque about the hinge) for TEST_S. Passed if the cover opened less than
COVER_HELD_DEG, the shackle is still through the hole and the block still seated OFF. Under ``control`` the same
check starts without a padlock once the block is OFF and the cover shut (the control run).
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

from pullout_sim import spec as S
from pullout_sim.kit import GRASP_OFFSET_M, reset_robot_home
from pullout_sim.scorer import MILESTONES, PulloutScorer, PulloutStep
from pullout_sim.usd import padlock_props as lock_props

_ASSETS = Path(__file__).resolve().parent / "assets"

PV = lock_props.PADLOCKS["ml410"]
SPRING_N_PER_M = 150.0
"""Shackle spring (N/m), as the other lockout tasks."""
LATCH_TRAVEL_M = 0.002
"""The shackle latches once pushed within this travel of shut."""

COVER_DAMPING = 0.02
COVER_SHUT_DEG = 4.0
DECOY_COVER_DEG = 5.0
DECOY_MOVED_M = 0.004
DECOY_TURNED_DEG = 8.0
RELEASE_CLEAR_M = 0.06
DROP_Z_M = 0.03

# The block's state relative to its seat.
SEAT_TOL_M = 0.003
"""Seated: within this of the seat in x (depth), y and z."""
ALIGN_COS = math.cos(math.radians(8.0))
CLOSED_DEPTH_M = 0.0145
"""The circuit is closed while the blades (block ON) are more than 3 mm into the live clips: the block's centre
less than this short of its seat."""

TEST_WAIT_S = 2.0
TEST_S = 2.0
LIFT_N = 20.0
"""The reviewer pulls the cover's bottom edge outward with this force (spring scale)."""
COVER_LEVER_M = math.hypot(S.COVER_FRONT - S.HINGE_X, S.COVER_BOTTOM - S.HINGE_Z)
COVER_HELD_DEG = 10.0
CONTROL_TEST_WAIT_S = 2.0


def lock_start_pose() -> tuple:
    x, y = S.PEDESTAL_XY
    return (x, y, S.PEDESTAL_TOP + PV.body[2] / 2 + 0.001, 0.0, 0.0, 0.0, 1.0)


def _pose(p) -> Pose:
    return Pose(position_xyz=tuple(p[:3]), rotation_xyzw=tuple(p[3:]))


class Disconnect(Object):
    def __init__(self, k: int):
        super().__init__(name=f"disconnect_{k}", usd_path=str(_ASSETS / f"disconnect_ac{k + 1}.usda"),
                         object_type=ObjectType.ARTICULATION,
                         initial_pose=_pose((0.0, S.BOX_Y[k], 0.0, 0.0, 0.0, 0.0, 1.0)))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {"cover": ImplicitActuatorCfg(joint_names_expr=["cover_joint"], stiffness=0.0,
                                                      damping=COVER_DAMPING)}
        cfg.init_state.joint_pos = {"cover_joint": 0.0}
        return cfg


class Block(Object):
    def __init__(self, k: int):
        super().__init__(name=f"block_{k}", usd_path=str(_ASSETS / "block.usda"), object_type=ObjectType.RIGID,
                         initial_pose=_pose(S.block_seated_pose(k)))


class Fixtures(Object):
    def __init__(self):
        super().__init__(name="fixtures", usd_path=str(_ASSETS / "fixtures.usda"), object_type=ObjectType.BASE,
                         initial_pose=_pose((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)))


class Padlock(Object):
    """The robot's red Master Lock 410 (sprung shackle; the task latches it)."""

    def __init__(self):
        super().__init__(name="padlock", usd_path=str(_ASSETS / "padlock_ml410.usda"),
                         object_type=ObjectType.ARTICULATION, initial_pose=_pose(lock_start_pose()))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {"shackle": ImplicitActuatorCfg(joint_names_expr=["shackle_joint"], stiffness=SPRING_N_PER_M,
                                                        damping=3.0)}
        cfg.init_state.joint_pos = {"shackle_joint": lock_props.OPEN_TRAVEL}
        return cfg


def block_state(pos, quat, k: int):
    """(circuit_closed, seated_on, seated_off, offset (N, 3)) of box k's block from its env-frame pose."""
    dev, n = pos.device, pos.shape[0]
    seat = torch.tensor(S.block_seated_pose(k)[:3], device=dev)
    off = pos - seat
    ex = math_utils.quat_apply(quat, torch.tensor([[1.0, 0.0, 0.0]], device=dev).expand(n, 3))
    ez = math_utils.quat_apply(quat, torch.tensor([[0.0, 0.0, 1.0]], device=dev).expand(n, 3))
    aligned = ex[:, 0] > ALIGN_COS
    on, upside = ez[:, 2] > ALIGN_COS, ez[:, 2] < -ALIGN_COS
    lateral = (off[:, 1].abs() < 0.004) & (off[:, 2].abs() < 0.004)
    closed = aligned & on & lateral & (off[:, 0] > -CLOSED_DEPTH_M)
    seated = aligned & (off.abs() < SEAT_TOL_M).all(dim=1)
    return closed, seated & on, seated & upside, off


class _TaskState:
    def __init__(self, env: ManagerBasedEnv, task: "PulloutTask"):
        n, dev = env.num_envs, env.device
        self.box = env.scene["disconnect_0"]
        self.decoy_box = env.scene["disconnect_1"]
        self.block = env.scene["block_0"]
        self.decoy_block = env.scene["block_1"]
        self.lock = env.scene["padlock"]
        self.robot = env.scene["robot"]
        self.cover_joint = self.box.find_joints("cover_joint")[0][0]
        self.decoy_joint = self.decoy_box.find_joints("cover_joint")[0][0]
        self.cover_body = self.box.find_bodies("cover")[0][0]
        self.shackle_joint = self.lock.find_joints("shackle_joint")[0][0]
        self.body_id = self.lock.find_bodies("body")[0][0]
        self.shackle_id = self.lock.find_bodies("shackle")[0][0]
        self.flange_ids, _ = self.robot.find_bodies(["left_gripper", "right_gripper"], preserve_order=True)
        self.scorer = PulloutScorer(n, dev)
        self.last_reports: dict[int, dict] = {}
        zb = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
        zl = lambda: torch.zeros(n, dtype=torch.long, device=dev)  # noqa: E731
        self.latched, self.latched_through, self.released, self.force_test = zb(), zb(), zb(), zb()
        self.test_passed, self.cover_held, self.was_pulled = zb(), zb(), zb()
        self.clear_steps, self.phase, self.phase_steps = zl(), zl(), zl()
        self.cover_max = torch.zeros(n, device=dev)
        self.wait_steps = max(1, round(TEST_WAIT_S / env.step_dt))
        self.test_len = max(1, round(TEST_S / env.step_dt))
        self.control_wait = max(1, round(CONTROL_TEST_WAIT_S / env.step_dt))
        self.control = task.control


def _write_pose(asset, pose, env_ids, origins):
    dev, n = origins.device, len(env_ids)
    p = torch.tensor(pose, device=dev, dtype=torch.float32).unsqueeze(0).repeat(n, 1)
    p[:, :3] += origins[env_ids]
    asset.write_root_pose_to_sim(p, env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)


class PulloutTask(TaskBase):
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
            shackle_latched=bool(st.latched[env_id]), shackle_latched_through_tab=bool(st.latched_through[env_id]),
            check_phase=int(st.phase[env_id]), check_cover_max_deg=round(math.degrees(float(st.cover_max[env_id])), 1),
            check_cover_held=bool(st.cover_held[env_id]), check_passed=bool(st.test_passed[env_id]),
        )
        return r

    def get_scene_cfg(self):
        if os.environ.get("PULLOUT_DEBUG_CAM") == "1":
            return _debug_cams()
        return None

    def get_events_cfg(self):
        return PulloutEventsCfg(self)

    def get_termination_cfg(self) -> TaskTerminationCfg:
        sequence = [(self._milestone_predicate(m), 0.25) for m in MILESTONES]
        return TaskTerminationCfg(timeout_s=self.episode_length_s,
                                  success=[CompletionCriteria(name="pullout_levels", predicate_sequence=sequence)])

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
        for buf in (st.latched, st.latched_through, st.released, st.force_test, st.test_passed, st.cover_held,
                    st.was_pulled):
            buf[env_ids] = False
        for buf in (st.clear_steps, st.phase, st.phase_steps):
            buf[env_ids] = 0
        st.cover_max[env_ids] = 0.0
        st.box.permanent_wrench_composer.reset(env_ids=env_ids)

        zero = torch.zeros(n, 1, device=dev)
        for box, joint in ((st.box, st.cover_joint), (st.decoy_box, st.decoy_joint)):
            box.write_joint_state_to_sim(zero, zero, joint_ids=[joint], env_ids=env_ids)
            box.set_joint_position_target(zero, joint_ids=[joint], env_ids=env_ids)
        _write_pose(st.block, S.block_seated_pose(0), env_ids, origins)
        _write_pose(st.decoy_block, S.block_seated_pose(1), env_ids, origins)

        # The robot's padlock: open, unlatched, on its pedestal.
        joint = [st.shackle_joint]
        limits = torch.tensor([[[0.0, lock_props.OPEN_TRAVEL]]], device=dev).expand(n, 1, 2).contiguous()
        travel = torch.full((n, 1), lock_props.OPEN_TRAVEL, device=dev)
        st.lock.write_joint_position_limit_to_sim(limits, joint_ids=joint, env_ids=env_ids, warn_limit_violation=False)
        st.lock.write_joint_state_to_sim(travel, torch.zeros_like(travel), joint_ids=joint, env_ids=env_ids)
        st.lock.set_joint_position_target(travel + 0.005, joint_ids=joint, env_ids=env_ids)
        _write_pose(st.lock, lock_start_pose(), env_ids, origins)

    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        origins = env.scene.env_origins
        n, dev = env.num_envs, env.device
        cover = st.box.data.joint_pos.torch[:, st.cover_joint]
        decoy_cover = st.decoy_box.data.joint_pos.torch[:, st.decoy_joint]

        # Blocks.
        bpos = st.block.data.root_pos_w.torch - origins
        bquat = st.block.data.root_quat_w.torch
        closed, _, seated_off, _ = block_state(bpos, bquat, 0)
        st.was_pulled |= ~closed
        re_energized = st.was_pulled & closed
        dpos = st.decoy_block.data.root_pos_w.torch - origins
        dquat = st.decoy_block.data.root_quat_w.torch
        _, _, _, doff = block_state(dpos, dquat, 1)
        dex = math_utils.quat_apply(dquat, torch.tensor([[1.0, 0.0, 0.0]], device=dev).expand(n, 3))
        dez = math_utils.quat_apply(dquat, torch.tensor([[0.0, 0.0, 1.0]], device=dev).expand(n, 3))
        turned = (dex[:, 0] < math.cos(math.radians(DECOY_TURNED_DEG))) | (
            dez[:, 2] < math.cos(math.radians(DECOY_TURNED_DEG)))
        decoy_moved = ((doff.norm(dim=-1) > DECOY_MOVED_M) | turned
                       | (decoy_cover > math.radians(DECOY_COVER_DEG)))
        cover_shut = cover < math.radians(COVER_SHUT_DEG)

        # The robot's padlock: its short leg through the tab's hole (the tab is fixed to the body).
        travel = st.lock.data.joint_pos.torch[:, st.shackle_joint]
        body_pos = st.lock.data.body_pos_w.torch[:, st.body_id] - origins
        sh_pos = st.lock.data.body_pos_w.torch[:, st.shackle_id] - origins
        sh_quat = st.lock.data.body_quat_w.torch[:, st.shackle_id]

        def on_shackle(p):
            return sh_pos + math_utils.quat_apply(sh_quat, torch.tensor([p], device=dev).expand(n, 3))

        tip = on_shackle((PV.leg_x, 0.0, lock_props.SHORT_LEG_BOTTOM))
        top = on_shackle((PV.leg_x, 0.0, PV.bend_z - PV.rod / 2))
        z_lo, z_hi = S.TAB_Z
        zm = (z_lo + z_hi) / 2
        denom = top[:, 2] - tip[:, 2]
        denom = torch.where(denom.abs() > 1e-6, denom, torch.full_like(denom, 1e-6))
        a = ((zm - tip[:, 2]) / denom).clamp(0.0, 1.0)
        cross = tip + a.unsqueeze(1) * (top - tip)
        hole = torch.tensor([S.TAB_HOLE_X, S.BOX_Y[0]], device=dev)
        in_hole = ((cross[:, :2] - hole).abs() < S.TAB_HOLE / 2 + 0.0005).all(dim=1)
        through = in_hole & (tip[:, 2] < z_lo) & (top[:, 2] > z_hi)

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
        edge = torch.tensor(S.cover_edge(0, 0.0), device=dev)
        clear_lock = ((grasp - body_pos[:, None]).norm(dim=-1) > RELEASE_CLEAR_M).all(dim=1)
        clear_cover = ((grasp - edge).norm(dim=-1) > RELEASE_CLEAR_M).all(dim=1) & (grasp[..., 0] < S.COVER_FRONT
                                                                                   - 0.02).all(dim=1)
        st.released = st.latched & clear_lock & clear_cover

        # After-release check.
        ready_lock = st.latched_through & st.released
        if st.control:
            ready_lock = ready_lock | (clear_cover & ~st.latched)
        ready = ((ready_lock & seated_off & cover_shut) | st.force_test) & (st.phase == 0)
        st.clear_steps = torch.where(ready, st.clear_steps + 1, torch.zeros_like(st.clear_steps))
        start = ready & ((st.clear_steps >= (st.control_wait if st.control else st.wait_steps)) | st.force_test)
        if start.any():
            ids = start.nonzero().flatten()
            torque = torch.zeros(len(ids), 1, 3, device=dev)
            torque[..., 1] = LIFT_N * COVER_LEVER_M
            st.box.permanent_wrench_composer.add_forces_and_torques_index(
                forces=torch.zeros_like(torque), torques=torque, body_ids=[st.cover_body], env_ids=ids, is_global=True)
            st.phase[ids] = 1
            st.phase_steps[ids] = 0
        st.phase_steps = torch.where(st.phase > 0, st.phase_steps + 1, st.phase_steps)
        st.cover_max = torch.where(st.phase == 1, torch.maximum(st.cover_max, cover), st.cover_max)
        end = (st.phase == 1) & (st.phase_steps >= st.test_len)
        if end.any():
            ids = end.nonzero().flatten()
            st.box.permanent_wrench_composer.reset(env_ids=ids)
            st.cover_held[ids] = st.cover_max[ids] < math.radians(COVER_HELD_DEG)
            st.test_passed[ids] = (st.cover_held[ids] & st.latched_through[ids] & through[ids] & seated_off[ids]
                                   & ~st.force_test[ids])
            st.phase[ids] = 2

        # During and after the check the cover is pulled by the check itself: score the shut state from before it
        # (a failed check leaves the level at 75; only a pass raises it to 100).
        shut_for_score = torch.where(st.phase > 0, torch.ones_like(cover_shut), cover_shut)
        dropped = body_pos[:, 2] < DROP_Z_M
        st.scorer.update(PulloutStep(~closed, seated_off, shut_for_score, st.test_passed.clone(), decoy_moved,
                                     re_energized, dropped))


@configclass
class PulloutEventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: PulloutTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))


DEBUG_VIEWS = {
    "debug_cam": ((0.02, 0.26, 0.36), (0.36, 0.0, 0.20)),
    "debug_cam2": ((0.20, -0.32, 0.26), (0.36, 0.0, 0.19)),
    "debug_cam3": ((0.10, 0.05, 0.50), (0.36, 0.0, 0.18)),
}
"""Development-only views (PULLOUT_DEBUG_CAM=1), desk frame: AC-1 from the front left, from the right, from above."""


def _debug_cams():
    """Extra 640 x 360 cameras for development; not part of the policy's observations."""
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg
    from yam_arena.embodiment import RENDERER

    from pullout_sim.kit import look_at_xyzw
    from pullout_sim.station import Z0

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
