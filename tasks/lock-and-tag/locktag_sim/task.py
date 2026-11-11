"""Lock and tag: lock out and tag out a disconnect handle.

Mechanism (see locktag_sim/usd/locktag_props.py):
- The switch's operating handle turns on a revolute joint (0 = OFF, 90 deg = ON) with an over-centre detent (a
  spring drive whose target is the nearer end). At OFF its tab's lock-off hole lines up with the fixed ear's hole; a
  shackle through both holes stops the handle turning (collision).
- The danger tag is a free articulation of three thin hinged plates; its grommet is a ring of colliders around a
  9.5 mm hole. Nothing but collision keeps it on a shackle.
- The robot's Master Lock 410 has a sprung shackle: prismatic joint, spring, a latch rule within 2 mm of shut.

Threaded tests (geometry, from simulator state): a point is on the shackle when it lies within ON_SHACKLE_M of the
shackle's centre line (legs above the body's top, and the bend). The tag is on the shackle when its grommet's centre
is; the padlock is through the lock-off holes when both holes' centres are, at the moment it latches.

After-release check (the success check): once the shackle has latched through both holes and both grasp points have
been at least RELEASE_CLEAR_M from the padlock for TEST_WAIT_S, the task (1) applies HANDLE_TEST_NM to the handle
toward ON for TEST_PHASE_S (held if it turns less than HANDLE_HELD_DEG), then (2) pulls the tag's bottom down with
TAG_TUG_N for TEST_PHASE_S (the tag stays if its grommet is still on the shackle at the end).
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

from locktag_sim.kit import GRASP_OFFSET_M, q_axis, reset_robot_home
from locktag_sim.scorer import MILESTONES, LockTagScorer, LockTagStep
from locktag_sim.spec import HOLE_Y, HOLE_Z, LOCK_PEDESTAL_TOP, LOCK_START_XY, TAG_GROMMET, WALL_X, make_seed
from locktag_sim.usd import locktag_props as props
from locktag_sim.usd import padlock_410
from locktag_sim.usd import padlock_props as lock_props

_ASSETS = Path(__file__).resolve().parent / "assets"

PV = props.PADLOCK
SPRING_N_PER_M = 150.0
"""Shackle spring (N/m)."""
LATCH_TRAVEL_M = 0.002
LOCK_START_QUAT = tuple(float(v) for v in q_axis((0, 1, 0), math.pi / 2))
"""The padlock on its narrow side: legs (local z) along +x, short leg (local x) down, thickness (local y) along y."""
TAG_START_QUAT = (0.0, 0.0, 0.0, 1.0)
"""The tag standing in its pocket, grommet up, printed front (local -x) toward the robot."""

DETENT_PEAK_NM = 1.0
"""Torque to push the handle over centre (inferred; to measure on a real switch)."""
DETENT_K = DETENT_PEAK_NM / math.radians(props.HANDLE_ON_DEG / 2)

ON_SHACKLE_M = 0.004
"""A point within this of the shackle's centre line is on it (grommet hole 9.5 mm, 12 mm lock-off holes, 6.35 mm
rod: a centre that close can only be there with the rod through the hole)."""
RELEASE_CLEAR_M = 0.06
DROP_Z_M = 0.03
HANDLE_MOVED_DEG = 10.0

TEST_WAIT_S = 2.0
TEST_PHASE_S = 1.5
HANDLE_TEST_NM = 1.5
"""1.5 times the detent's peak torque, toward ON: 15 N at the lever's tip, 10 cm from the pivot."""
HANDLE_HELD_DEG = 25.0
TAG_TUG_N = 3.0
"""A 3 N downward pull on the tag's bottom edge (spring scale clipped to it): far below the tag's 50 lb rating."""

TAB_HOLE_IN_HANDLE = ((props.TAB_X[0] + props.TAB_X[1]) / 2 - props.PIVOT_X, 0.0, HOLE_Z - props.PIVOT_Z)
"""The handle's lock-off hole centre in the handle body's frame (origin at the pivot)."""
EAR_HOLE_IN_SWITCH = ((props.EAR_X[0] + props.EAR_X[1]) / 2 - WALL_X, 0.0, HOLE_Z)
"""The fixed ear's hole centre in the switch's (enclosure's) frame."""

# Shackle centre line, in the shackle body's frame (origin at the body's top centre when shut; legs along +z).
LEG_X = PV.leg_x
BEND_R = padlock_410.BEND_R
BEND_ZC = padlock_410.BEND_ZC
SHORT_LO = lock_props.SHORT_LEG_BOTTOM


def lock_start_pose() -> tuple:
    x, y = LOCK_START_XY
    return (x, y, LOCK_PEDESTAL_TOP + PV.body[0] / 2 + 0.0005) + LOCK_START_QUAT


def tag_start_pose() -> tuple:
    x, y, z = TAG_GROMMET
    return (x, y, z + 0.0005) + TAG_START_QUAT


PARK = (props.PARK_X, 0.10, props.PARK_Z + 0.03, 0.0, 0.0, 0.0, 1.0)


def _pose(p) -> Pose:
    return Pose(position_xyz=p[:3], rotation_xyzw=p[3:])


def shackle_distance(p: torch.Tensor, travel: torch.Tensor) -> torch.Tensor:
    """Distance (N,) from points p (N, 3, shackle frame) to the exposed shackle centre line (legs from the body's
    top, which is at z = -travel in this frame, up to the bend; the bend's semicircle)."""
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    lo_short = torch.clamp(-travel, min=SHORT_LO)
    d_short = torch.sqrt((x - LEG_X) ** 2 + y**2 + (z - z.clamp(min=lo_short, max=torch.full_like(z, BEND_ZC))) ** 2)
    d_long = torch.sqrt((x + LEG_X) ** 2 + y**2 + (z - z.clamp(min=-travel, max=torch.full_like(z, BEND_ZC))) ** 2)
    rho = torch.sqrt(x**2 + (z - BEND_ZC).clamp(min=0.0) ** 2)
    d_bend = torch.where(z >= BEND_ZC, torch.sqrt((rho - BEND_R) ** 2 + y**2), torch.full_like(z, 1.0))
    return torch.minimum(torch.minimum(d_short, d_long), d_bend)


class Switch(Object):
    def __init__(self):
        super().__init__(name="switch", usd_path=str(_ASSETS / "switch.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=_pose((WALL_X, HOLE_Y, 0.0, 0.0, 0.0, 0.0, 1.0)))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {"handle": ImplicitActuatorCfg(joint_names_expr=["handle_joint"], stiffness=DETENT_K,
                                                       damping=0.05 * DETENT_K)}
        cfg.init_state.joint_pos = {"handle_joint": 0.0}
        return cfg


class Tag(Object):
    def __init__(self):
        super().__init__(name="tag", usd_path=str(_ASSETS / "tag.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=_pose(tag_start_pose()))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {"hinges": ImplicitActuatorCfg(joint_names_expr=["hinge.*"], stiffness=props.TAG_HINGE_K,
                                                       damping=props.TAG_HINGE_D)}
        cfg.init_state.joint_pos = {"hinge.*": 0.0}
        return cfg


class Padlock(Object):
    """The robot's red Master Lock 410 (sprung shackle, the task latches it)."""

    def __init__(self):
        super().__init__(name="padlock", usd_path=str(_ASSETS / "padlock_ml410.usda"),
                         object_type=ObjectType.ARTICULATION, initial_pose=_pose(lock_start_pose()))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {"shackle": ImplicitActuatorCfg(joint_names_expr=["shackle_joint"], stiffness=SPRING_N_PER_M,
                                                        damping=3.0)}
        cfg.init_state.joint_pos = {"shackle_joint": lock_props.OPEN_TRAVEL}
        return cfg


class Stands(Object):
    """The padlock's column and the tag pocket (one fixed asset)."""

    def __init__(self):
        super().__init__(name="stands", usd_path=str(_ASSETS / "stands.usda"), object_type=ObjectType.BASE,
                         initial_pose=_pose((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0)))


class _TaskState:
    def __init__(self, env: ManagerBasedEnv, task: "LockTagTask"):
        n, dev = env.num_envs, env.device
        self.lock = env.scene["padlock"]
        self.tag = env.scene["tag"]
        self.switch = env.scene["switch"]
        self.robot = env.scene["robot"]
        self.shackle_joint = self.lock.find_joints("shackle_joint")[0][0]
        self.body_id = self.lock.find_bodies("body")[0][0]
        self.shackle_id = self.lock.find_bodies("shackle")[0][0]
        self.tag_top = self.tag.find_bodies("seg0")[0][0]
        self.tag_bottom = self.tag.find_bodies("seg2")[0][0]
        self.hinge_joints = self.tag.find_joints("hinge.*")[0]
        self.handle_joint = self.switch.find_joints("handle_joint")[0][0]
        self.handle_body = self.switch.find_bodies("handle")[0][0]
        self.enclosure_body = self.switch.find_bodies("enclosure")[0][0]
        self.flange_ids, _ = self.robot.find_bodies(["left_gripper", "right_gripper"], preserve_order=True)
        self.scorer = LockTagScorer(n, dev)
        self.last_reports: dict[int, dict] = {}
        zb = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
        zl = lambda: torch.zeros(n, dtype=torch.long, device=dev)  # noqa: E731
        zf = lambda: torch.zeros(n, device=dev)  # noqa: E731
        self.latched, self.latched_through, self.tag_at_latch, self.released, self.force_test = zb(), zb(), zb(), zb(), zb()
        self.handle_held, self.tag_held, self.test_passed = zb(), zb(), zb()
        self.tag_on, self.through = zb(), zb()
        self.clear_steps, self.phase, self.phase_steps = zl(), zl(), zl()
        self.handle_max, self.tag_dist, self.hole_dist = zf(), zf(), zf()
        self.wait_steps = max(1, round(TEST_WAIT_S / env.step_dt))
        self.phase_len = max(1, round(TEST_PHASE_S / env.step_dt))


def _write_pose(asset, pose, env_ids, origins, velocity=True):
    dev, n = origins.device, len(env_ids)
    p = torch.tensor(pose, device=dev, dtype=torch.float32).unsqueeze(0).repeat(n, 1)
    p[:, :3] += origins[env_ids]
    asset.write_root_pose_to_sim(p, env_ids=env_ids)
    if velocity:
        asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)


class LockTagTask(TaskBase):
    def __init__(self, default_seed: int = 0, episode_length_s: float = 240.0, control: bool = False):
        super().__init__(episode_length_s=episode_length_s, task_description=make_seed(default_seed).instruction)
        self.default_seed = default_seed
        self.control = control

    def state(self, env: ManagerBasedEnv) -> _TaskState:
        if getattr(env, "_task_state", None) is None:
            env._task_state = _TaskState(env, self)
        return env._task_state

    def set_next_seed(self, env: ManagerBasedEnv, seed: int) -> None:
        make_seed(seed)

    def force_function_test(self, env: ManagerBasedEnv) -> None:
        self.state(env).force_test[:] = True

    def episode_report(self, env: ManagerBasedEnv, env_id: int = 0, ended: bool = False) -> dict:
        st = self.state(env)
        if ended and env_id in st.last_reports:
            return st.last_reports.pop(env_id)
        return self._report(st, env_id)

    def _report(self, st: _TaskState, env_id: int) -> dict:
        r = st.scorer.report(env_id)
        r.update(
            tag_on_shackle=bool(st.tag_on[env_id]), tag_grommet_to_shackle_mm=round(1000 * float(st.tag_dist[env_id]), 1),
            shackle_through_holes=bool(st.through[env_id]), holes_to_shackle_mm=round(1000 * float(st.hole_dist[env_id]), 1),
            shackle_travel_mm=round(1000 * float(st.lock.data.joint_pos.torch[env_id, st.shackle_joint]), 1),
            shackle_latched=bool(st.latched[env_id]), latched_through_holes=bool(st.latched_through[env_id]),
            tag_on_at_latch=bool(st.tag_at_latch[env_id]), function_test_phase=int(st.phase[env_id]),
            test_handle_max_deg=round(math.degrees(float(st.handle_max[env_id])), 1),
            test_handle_held=bool(st.handle_held[env_id]), test_tag_held=bool(st.tag_held[env_id]),
            function_test_passed=bool(st.test_passed[env_id]),
        )
        return r

    def get_scene_cfg(self):
        return None

    def get_events_cfg(self):
        return LockTagEventsCfg(self)

    def get_termination_cfg(self) -> TaskTerminationCfg:
        sequence = [(self._milestone_predicate(m), 0.25) for m in MILESTONES]
        return TaskTerminationCfg(timeout_s=self.episode_length_s,
                                  success=[CompletionCriteria(name="locktag_milestones", predicate_sequence=sequence)])

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
        dev, n = env.device, len(env_ids)
        origins = env.scene.env_origins
        st.scorer.reset(env_ids)
        for buf in (st.latched, st.latched_through, st.tag_at_latch, st.released, st.force_test, st.handle_held,
                    st.tag_held, st.test_passed, st.tag_on, st.through):
            buf[env_ids] = False
        for buf in (st.clear_steps, st.phase, st.phase_steps):
            buf[env_ids] = 0
        st.handle_max[env_ids] = 0.0
        for asset in (st.lock, st.tag, st.switch):
            asset.permanent_wrench_composer.reset(env_ids=env_ids)

        # The padlock: open, unlatched, on its column.
        joint = [st.shackle_joint]
        limits = torch.tensor([[[0.0, lock_props.OPEN_TRAVEL]]], device=dev).expand(n, 1, 2).contiguous()
        travel = torch.full((n, 1), lock_props.OPEN_TRAVEL, device=dev)
        st.lock.write_joint_position_limit_to_sim(limits, joint_ids=joint, env_ids=env_ids, warn_limit_violation=False)
        st.lock.write_joint_state_to_sim(travel, torch.zeros_like(travel), joint_ids=joint, env_ids=env_ids)
        st.lock.set_joint_position_target(travel + 0.005, joint_ids=joint, env_ids=env_ids)
        _write_pose(st.lock, lock_start_pose(), env_ids, origins)

        # The tag: flat, standing in its pocket. The handle: OFF.
        hz = torch.zeros(n, len(st.hinge_joints), device=dev)
        st.tag.write_joint_state_to_sim(hz, hz, joint_ids=st.hinge_joints, env_ids=env_ids)
        st.tag.set_joint_position_target(hz, joint_ids=st.hinge_joints, env_ids=env_ids)
        _write_pose(st.tag, tag_start_pose(), env_ids, origins)
        zero = torch.zeros(n, 1, device=dev)
        st.switch.write_joint_state_to_sim(zero, zero, joint_ids=[st.handle_joint], env_ids=env_ids)
        st.switch.set_joint_position_target(zero, joint_ids=[st.handle_joint], env_ids=env_ids)

    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        origins = env.scene.env_origins
        n, dev = env.num_envs, env.device

        # Handle detent: spring toward the nearer end.
        q_handle = st.switch.data.joint_pos.torch[:, st.handle_joint]
        target = torch.where(q_handle > math.radians(props.HANDLE_ON_DEG / 2), math.radians(props.HANDLE_ON_DEG), 0.0)
        st.switch.set_joint_position_target(target.unsqueeze(1), joint_ids=[st.handle_joint])

        # Shackle frame.
        travel = st.lock.data.joint_pos.torch[:, st.shackle_joint]
        body_pos = st.lock.data.body_pos_w.torch[:, st.body_id] - origins
        sh_pos = st.lock.data.body_pos_w.torch[:, st.shackle_id] - origins
        sh_quat = st.lock.data.body_quat_w.torch[:, st.shackle_id]

        def to_shackle(p):
            return math_utils.quat_apply_inverse(sh_quat, p - sh_pos)

        # Lock-off holes (world, desk frame).
        h_pos = st.switch.data.body_pos_w.torch[:, st.handle_body] - origins
        h_quat = st.switch.data.body_quat_w.torch[:, st.handle_body]
        e_pos = st.switch.data.body_pos_w.torch[:, st.enclosure_body] - origins
        e_quat = st.switch.data.body_quat_w.torch[:, st.enclosure_body]
        tab_hole = h_pos + math_utils.quat_apply(h_quat, torch.tensor([TAB_HOLE_IN_HANDLE], device=dev).expand(n, 3))
        ear_hole = e_pos + math_utils.quat_apply(e_quat, torch.tensor([EAR_HOLE_IN_SWITCH], device=dev).expand(n, 3))
        hole_dist = torch.maximum(shackle_distance(to_shackle(tab_hole), travel),
                                  shackle_distance(to_shackle(ear_hole), travel))
        st.hole_dist = hole_dist
        st.through = hole_dist < ON_SHACKLE_M

        # The tag's grommet on the shackle.
        grommet = st.tag.data.body_pos_w.torch[:, st.tag_top] - origins
        tag_dist = shackle_distance(to_shackle(grommet), travel)
        st.tag_dist = tag_dist
        st.tag_on = tag_dist < ON_SHACKLE_M

        # Latch: once pushed home the shackle stays shut. It counts only if it shut while through both holes.
        newly = ~st.latched & (travel < LATCH_TRAVEL_M)
        if newly.any():
            ids = newly.nonzero().flatten()
            limits = torch.tensor([[[0.0, 0.0005]]], device=dev).expand(len(ids), 1, 2).contiguous()
            st.lock.write_joint_position_limit_to_sim(limits, joint_ids=[st.shackle_joint], env_ids=ids,
                                                      warn_limit_violation=False)
            st.lock.set_joint_position_target(torch.zeros(len(ids), 1, device=dev), joint_ids=[st.shackle_joint],
                                              env_ids=ids)
            if self.control:
                # Control: the tag was never threaded, it only lies against the lock. Take it off the shackle and
                # lay it against the side of the padlock's body as the shackle latches.
                b_quat = st.lock.data.body_quat_w.torch[ids, st.body_id]
                side = math_utils.quat_apply(b_quat, torch.tensor([[0.0, 1.0, 0.0]], device=dev).expand(len(ids), 3))
                pose = torch.zeros(len(ids), 7, device=dev)
                pose[:, :3] = body_pos[ids] + 0.012 * side + origins[ids]
                pose[:, 2] += 0.03
                pose[:, 6] = 1.0
                st.tag.write_root_pose_to_sim(pose, env_ids=ids)
                st.tag.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=dev), env_ids=ids)
                st.tag_on[ids] = False
            st.latched |= newly
            st.latched_through |= newly & st.through
            st.tag_at_latch |= newly & st.tag_on

        # Release: both grasp points clear of the padlock.
        flange_pos = st.robot.data.body_pos_w.torch[:, st.flange_ids]
        flange_quat = st.robot.data.body_quat_w.torch[:, st.flange_ids]
        down = torch.tensor([0.0, 0.0, -GRASP_OFFSET_M], device=dev).expand_as(flange_pos)
        grasp = flange_pos + math_utils.quat_apply(flange_quat, down) - origins[:, None]
        clear_lock = ((grasp - body_pos[:, None]).norm(dim=-1) > RELEASE_CLEAR_M).all(dim=1)
        clear_tag = ((grasp - grommet[:, None]).norm(dim=-1) > RELEASE_CLEAR_M).all(dim=1)
        st.released |= st.latched & clear_lock & clear_tag

        # After-release check.
        ready = (st.latched & st.released & (st.phase == 0)) | (st.force_test & (st.phase == 0))
        st.clear_steps = torch.where(ready, st.clear_steps + 1, torch.zeros_like(st.clear_steps))
        start = ready & (st.clear_steps >= st.wait_steps)
        if start.any():
            ids = start.nonzero().flatten()
            torque = torch.zeros(len(ids), 1, 3, device=dev)
            torque[..., 0] = HANDLE_TEST_NM
            st.switch.permanent_wrench_composer.add_forces_and_torques_index(
                forces=torch.zeros_like(torque), torques=torque, body_ids=[st.handle_body], env_ids=ids, is_global=True)
            st.phase[ids] = 1
            st.phase_steps[ids] = 0
        in_test = st.phase > 0
        st.phase_steps = torch.where(in_test, st.phase_steps + 1, st.phase_steps)
        st.handle_max = torch.where(st.phase == 1, torch.maximum(st.handle_max, q_handle), st.handle_max)
        end_handle = (st.phase == 1) & (st.phase_steps >= st.phase_len)
        if end_handle.any():
            ids = end_handle.nonzero().flatten()
            st.switch.permanent_wrench_composer.reset(env_ids=ids)
            st.handle_held[ids] = (st.handle_max[ids] < math.radians(HANDLE_HELD_DEG)) & st.through[ids]
            force = torch.zeros(len(ids), 1, 3, device=dev)
            force[..., 2] = -TAG_TUG_N
            # The pull acts at the tag's bottom edge (the bottom plate's lower end).
            st.tag.permanent_wrench_composer.add_forces_and_torques_index(
                forces=force, torques=torch.zeros_like(force), body_ids=[st.tag_bottom], env_ids=ids, is_global=True)
            st.phase[ids] = 2
            st.phase_steps[ids] = 0
        end_tag = (st.phase == 2) & (st.phase_steps >= st.phase_len)
        if end_tag.any():
            ids = end_tag.nonzero().flatten()
            st.tag.permanent_wrench_composer.reset(env_ids=ids)
            st.tag_held[ids] = st.tag_on[ids]
            st.test_passed[ids] = st.handle_held[ids] & st.latched_through[ids]
            st.phase[ids] = 3
            for i in ids.tolist():
                print(f"[lock-and-tag] after-release check: handle turned {math.degrees(float(st.handle_max[i])):.1f} "
                      f"deg (held: {bool(st.handle_held[i])}), tag on the shackle after the pull: "
                      f"{bool(st.tag_held[i])} (grommet {1000 * float(st.tag_dist[i]):.1f} mm from the shackle), "
                      f"latched through the holes: {bool(st.latched_through[i])}", flush=True)

        before_test = st.phase == 0
        handle_moved = before_test & (q_handle > math.radians(HANDLE_MOVED_DEG))
        lock_dropped = body_pos[:, 2] < DROP_Z_M
        st.scorer.update(LockTagStep(
            tag_on=st.tag_on & before_test,
            locked_tagged=st.latched_through & st.tag_at_latch & st.tag_on & before_test,
            locked_untagged_passed=st.test_passed & ~st.tag_held,
            passed_tagged=st.test_passed & st.tag_held & st.tag_at_latch,
            handle_moved=handle_moved, lock_dropped=lock_dropped))


@configclass
class LockTagEventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: LockTagTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))
