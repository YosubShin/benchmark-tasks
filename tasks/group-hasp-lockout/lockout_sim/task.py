"""Group hasp: add your padlock to a group-lockout hasp that holds a switch handle OFF.

Mechanism (see lockout_sim/usd/hasp_props.py):
- The switch's operating handle turns on a revolute joint (0 = OFF, 90 deg = ON). The task supplies an
  over-centre detent (a spring drive whose target is the nearer end), as task 11 does.
- The Master Lock 420 hasp is a free articulation (jaw A + jaw B on the rivet joint). Jaw A's ring bar
  passes through the handle's lock-off hole and the fixed ear behind it, so while the ring is closed the
  handle cannot turn (collision). A shackle through an aligned hole pair of both tails stops jaw B from
  turning (collision). A weak spring (0.02 N*m/rad) keeps an empty hasp from flopping open; it does not
  hold against the test.
- The robot's Master Lock 410 has a sprung shackle: prismatic joint, weak spring, software latch.
- The coworker's lock is a rigid yellow 410 hung through one tail hole at reset.

Function test (the success check): once the robot's shackle has latched through a free hole and both
grippers have been clear of the padlock for TEST_WAIT_S, the scorer (1) parks the coworker's lock under
the desk, (2) applies JAW_TEST_NM to open the jaws for TEST_PHASE_S; if they open past JAW_HELD_DEG the
hasp counts as removable (an open hasp slides off the handle by itself), (3) applies HANDLE_TEST_NM toward ON for TEST_PHASE_S.
Passed if the jaws stayed shut, the handle stayed below HANDLE_HELD_DEG and the robot's lock is still
through. `force_function_test(env)` starts the same test without a lock (the control run).
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

from lockout_sim.kit import GRASP_OFFSET_M
from lockout_sim.kit import q_axis, qmul, rotate
from lockout_sim.kit import reset_robot_home, reviewer_camera
from lockout_sim.scorer import MILESTONES, GroupHaspScorer, GroupHaspStep
from lockout_sim.spec import HASP_Y, PEDESTAL_XY, RING_X, WALL_X, GroupHaspSeed, make_seed
SPRING_N_PER_M = 150.0
"""Shackle spring (N/m), the padlock task's value."""
LATCH_TRAVEL_M = 0.002
"""The shackle latches once pushed within this travel of shut."""
from lockout_sim.kit import Fingertips
from lockout_sim.usd import hasp_props as props
from lockout_sim.usd import padlock_props as lock_props

_ASSETS = Path(__file__).resolve().parent / "assets"

PV = lock_props.PADLOCKS["ml410"]
CW = lock_props.PRELOCKED
LOCK_LYING_QUAT = tuple(float(v) for v in qmul(q_axis((1, 0, 0), math.pi / 2), q_axis((0, 0, 1), math.pi / 2)))
"""The robot's padlock on its side: width (local x) up, thickness (local y) along -x, shackle (local z) along -y."""
COWORKER_QUAT = tuple(float(v) for v in q_axis((0, 0, 1), math.pi / 2))
"""Hanging: the shackle's legs straddle the tail (local x along world y)."""

DETENT_PEAK_NM = 1.0
"""Torque to push the handle over centre (inferred; to measure on a real switch)."""
DETENT_K = DETENT_PEAK_NM / math.radians(props.HANDLE_ON_DEG / 2)
JAW_SPRING = 0.02
JAW_DAMPING = 0.005

CONTACT_MOVE_M = 0.01
AT_HOLE_M = 0.012
RELEASE_CLEAR_M = 0.06
DROP_Z_M = 0.05
COWORKER_SETTLE_S = 2.0
COWORKER_MOVE_M = 0.01
TOUCH_MARGIN_M = 0.006
HASP_OPEN_DEG = 8.0
HANDLE_MOVED_DEG = 10.0

TEST_WAIT_S = 2.0
TEST_PHASE_S = 1.5
JAW_TEST_NM = 0.3
"""Prying the tails apart: about 6 N at the tail holes (inferred hand effort; to measure)."""
JAW_HELD_DEG = 10.0
HANDLE_TEST_NM = 1.5
"""1.5 times the detent's peak torque, toward ON."""
HANDLE_HELD_DEG = 25.0

LAYOUT_YAW_DEG = 30.0
"""The whole prop assembly (switch, hasp, both padlocks, pedestal) is turned this much counterclockwise (seen from
above) about the vertical through the hasp's ring, so the station's top camera sees the hasp's face, not its edge."""
PIVOT_XY = (RING_X, HASP_Y)
_YAW_Q = q_axis((0, 0, 1), math.radians(LAYOUT_YAW_DEG))


def layout_dir(v) -> tuple:
    """A direction of the unturned layout, turned with the assembly."""
    return tuple(float(c) for c in rotate(_YAW_Q, v))


def layout_pose(pose) -> tuple:
    """(x, y, z, qx, qy, qz, qw) of the unturned layout -> the turned layout (desk frame)."""
    x, y, z = pose[:3]
    dx, dy, _ = rotate(_YAW_Q, (x - PIVOT_XY[0], y - PIVOT_XY[1], 0.0))
    q = qmul(_YAW_Q, pose[3:])
    return (PIVOT_XY[0] + float(dx), PIVOT_XY[1] + float(dy), float(z)) + tuple(float(c) for c in q)


PARK_COWORKER = layout_pose((props.PARK_X, 0.10, props.PARK_Z + 0.03, 0.0, 0.0, 0.0, 1.0))


def lock_start_pose() -> tuple:
    x, y = PEDESTAL_XY
    return layout_pose((x, y, props.PEDESTAL_TOP + PV.body[0] / 2 + 0.001) + LOCK_LYING_QUAT)


def hasp_start_pose() -> tuple:
    return layout_pose(props.nominal_hasp_pos() + (0.0, 0.0, 0.0, 1.0))


def coworker_start_pose(layout: GroupHaspSeed) -> tuple:
    """Shackle bend resting on the lower edge of the coworker's tail hole."""
    hx, _, hz = props.hole_local(*layout.coworker_hole)
    rx, ry, rz = props.nominal_hasp_pos()
    z = rz + hz - props.HOLE / 2 - CW.bend_bottom(0.0) + 0.0005
    return layout_pose((rx + hx, ry, z) + COWORKER_QUAT)


def _pose(p) -> Pose:
    return Pose(position_xyz=p[:3], rotation_xyzw=p[3:])


class Switch(Object):
    def __init__(self):
        super().__init__(name="switch", usd_path=str(_ASSETS / "switch.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=_pose(layout_pose((WALL_X, HASP_Y, 0.0, 0.0, 0.0, 0.0, 1.0))))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {"handle": ImplicitActuatorCfg(joint_names_expr=["handle_joint"], stiffness=DETENT_K,
                                                       damping=0.05 * DETENT_K)}
        cfg.init_state.joint_pos = {"handle_joint": 0.0}
        return cfg


class Hasp(Object):
    def __init__(self):
        p = hasp_start_pose()
        super().__init__(name="hasp", usd_path=str(_ASSETS / "hasp.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=p[:3], rotation_xyzw=p[3:]))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {"jaw": ImplicitActuatorCfg(joint_names_expr=["jaw_joint"], stiffness=JAW_SPRING,
                                                    damping=JAW_DAMPING)}
        cfg.init_state.joint_pos = {"jaw_joint": 0.0}
        return cfg


class GroupPadlock(Object):
    """The robot's Master Lock 410 (padlock asset: sprung shackle, the task latches it)."""

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


class CoworkerLock(Object):
    def __init__(self, layout: GroupHaspSeed):
        p = coworker_start_pose(layout)
        super().__init__(name="coworker_lock", usd_path=str(_ASSETS / "coworker.usda"), object_type=ObjectType.RIGID,
                         initial_pose=Pose(position_xyz=p[:3], rotation_xyzw=p[3:]))


class Pedestal(Object):
    def __init__(self):
        super().__init__(name="pedestal", usd_path=str(_ASSETS / "pedestal.usda"), object_type=ObjectType.BASE,
                         initial_pose=_pose(layout_pose((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))))


class _TaskState:
    def __init__(self, env: ManagerBasedEnv, task: "GroupHaspTask"):
        n, dev = env.num_envs, env.device
        self.lock = env.scene["padlock"]
        self.hasp = env.scene["hasp"]
        self.switch = env.scene["switch"]
        self.coworker = env.scene["coworker_lock"]
        self.robot = env.scene["robot"]
        self.tips = Fingertips(self.robot)
        self.shackle_joint = self.lock.find_joints("shackle_joint")[0][0]
        self.body_id = self.lock.find_bodies("body")[0][0]
        self.shackle_id = self.lock.find_bodies("shackle")[0][0]
        self.jaw_joint = self.hasp.find_joints("jaw_joint")[0][0]
        self.jaw_a = self.hasp.find_bodies("jaw_a")[0][0]
        self.jaw_b = self.hasp.find_bodies("jaw_b")[0][0]
        self.handle_joint = self.switch.find_joints("handle_joint")[0][0]
        self.handle_body = self.switch.find_bodies("handle")[0][0]
        self.flange_ids, _ = self.robot.find_bodies(["left_gripper", "right_gripper"], preserve_order=True)
        self.scorer = GroupHaspScorer(n, dev)
        self.next_layout = make_seed(task.default_seed)
        self.layouts: dict[int, GroupHaspSeed] = {}
        self.last_reports: dict[int, dict] = {}
        self.holes_local = torch.tensor([props.hole_local(c, r) for c in (0, 1) for r in range(3)], device=dev)
        self.free = torch.ones(n, 6, dtype=torch.bool, device=dev)
        zb = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
        zl = lambda: torch.zeros(n, dtype=torch.long, device=dev)  # noqa: E731
        zf = lambda: torch.zeros(n, device=dev)  # noqa: E731
        self.latched, self.latched_through, self.released, self.force_test = zb(), zb(), zb(), zb()
        self.jaw_held, self.handle_held, self.test_passed, self.hasp_removed = zb(), zb(), zb(), zb()
        self.cw_ref_set = zb()
        self.hole_used = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.clear_steps, self.phase, self.phase_steps = zl(), zl(), zl()
        self.jaw_max, self.handle_max = zf(), zf()
        self.start_pos = torch.zeros(n, 3, device=dev)
        self.cw_ref = torch.zeros(n, 7, device=dev)
        self.wait_steps = max(1, round(TEST_WAIT_S / env.step_dt))
        self.phase_len = max(1, round(TEST_PHASE_S / env.step_dt))
        self.settle_steps = max(1, round(COWORKER_SETTLE_S / env.step_dt))


def _write_pose(asset, pose, env_ids, origins, velocity=True):
    dev, n = origins.device, len(env_ids)
    p = torch.tensor(pose, device=dev, dtype=torch.float32).unsqueeze(0).repeat(n, 1)
    p[:, :3] += origins[env_ids]
    asset.write_root_pose_to_sim(p, env_ids=env_ids)
    if velocity:
        asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)


class GroupHaspTask(TaskBase):
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
        r.update(
            shackle_latched=bool(st.latched[env_id]), shackle_latched_through_free_hole=bool(st.latched_through[env_id]),
            hole_used=int(st.hole_used[env_id]), function_test_phase=int(st.phase[env_id]),
            test_jaw_max_deg=round(math.degrees(float(st.jaw_max[env_id])), 1),
            test_handle_max_deg=round(math.degrees(float(st.handle_max[env_id])), 1),
            test_jaw_held=bool(st.jaw_held[env_id]), test_handle_held=bool(st.handle_held[env_id]),
            test_hasp_removable=bool(st.hasp_removed[env_id]), function_test_passed=bool(st.test_passed[env_id]),
        )
        return r

    def get_scene_cfg(self):
        return ReviewerCameraSceneCfg() if self.reviewer_camera else None

    def get_events_cfg(self):
        return GroupHaspEventsCfg(self)

    def get_termination_cfg(self) -> TaskTerminationCfg:
        sequence = [(self._milestone_predicate(m), 0.25) for m in MILESTONES]
        return TaskTerminationCfg(timeout_s=self.episode_length_s,
                                  success=[CompletionCriteria(name="grouphasp_milestones", predicate_sequence=sequence)])

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
        for buf in (st.latched, st.latched_through, st.released, st.force_test, st.jaw_held, st.handle_held,
                    st.test_passed, st.hasp_removed, st.cw_ref_set):
            buf[env_ids] = False
        for buf in (st.clear_steps, st.phase, st.phase_steps):
            buf[env_ids] = 0
        st.jaw_max[env_ids] = 0.0
        st.handle_max[env_ids] = 0.0
        st.hole_used[env_ids] = -1
        cw_index = layout.coworker_hole[0] * 3 + layout.coworker_hole[1]
        st.free[env_ids] = True
        st.free[env_ids, cw_index] = False
        for asset in (st.lock, st.hasp, st.switch):
            asset.permanent_wrench_composer.reset(env_ids=env_ids)

        # The robot's padlock: open, unlatched, on its pedestal.
        joint = [st.shackle_joint]
        limits = torch.tensor([[[0.0, lock_props.OPEN_TRAVEL]]], device=dev).expand(n, 1, 2).contiguous()
        travel = torch.full((n, 1), lock_props.OPEN_TRAVEL, device=dev)
        st.lock.write_joint_position_limit_to_sim(limits, joint_ids=joint, env_ids=env_ids, warn_limit_violation=False)
        st.lock.write_joint_state_to_sim(travel, torch.zeros_like(travel), joint_ids=joint, env_ids=env_ids)
        st.lock.set_joint_position_target(travel + 0.005, joint_ids=joint, env_ids=env_ids)
        _write_pose(st.lock, lock_start_pose(), env_ids, origins)
        st.start_pos[env_ids] = torch.tensor(lock_start_pose()[:3], device=dev)

        # Hasp closed on the handle, coworker's lock in its hole, handle OFF.
        zero = torch.zeros(n, 1, device=dev)
        st.hasp.write_joint_state_to_sim(zero, zero, joint_ids=[st.jaw_joint], env_ids=env_ids)
        st.hasp.set_joint_position_target(zero, joint_ids=[st.jaw_joint], env_ids=env_ids)
        _write_pose(st.hasp, hasp_start_pose(), env_ids, origins)
        st.switch.write_joint_state_to_sim(zero, zero, joint_ids=[st.handle_joint], env_ids=env_ids)
        st.switch.set_joint_position_target(zero, joint_ids=[st.handle_joint], env_ids=env_ids)
        _write_pose(st.coworker, coworker_start_pose(layout), env_ids, origins)
        for env_id in env_ids.tolist():
            st.layouts[env_id] = layout

    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        origins = env.scene.env_origins
        n, dev = env.num_envs, env.device
        all_ids = torch.arange(n, device=dev)

        # Handle detent: spring toward the nearer end.
        q_handle = st.switch.data.joint_pos.torch[:, st.handle_joint]
        target = torch.where(q_handle > math.radians(props.HANDLE_ON_DEG / 2), math.radians(props.HANDLE_ON_DEG), 0.0)
        st.switch.set_joint_position_target(target.unsqueeze(1), joint_ids=[st.handle_joint])
        q_jaw = st.hasp.data.joint_pos.torch[:, st.jaw_joint]

        # The robot's padlock: short leg against the hasp's free holes, in the hasp (jaw A) frame.
        travel = st.lock.data.joint_pos.torch[:, st.shackle_joint]
        body_pos = st.lock.data.body_pos_w.torch[:, st.body_id] - origins
        sh_pos = st.lock.data.body_pos_w.torch[:, st.shackle_id] - origins
        sh_quat = st.lock.data.body_quat_w.torch[:, st.shackle_id]
        a_pos = st.hasp.data.body_pos_w.torch[:, st.jaw_a] - origins
        a_quat = st.hasp.data.body_quat_w.torch[:, st.jaw_a]

        def on_shackle(p):
            return sh_pos + math_utils.quat_apply(sh_quat, torch.tensor([p], device=dev).expand(n, 3))

        def to_hasp(p):
            return math_utils.quat_apply_inverse(a_quat, p - a_pos)

        tip = to_hasp(on_shackle((PV.leg_x, 0.0, lock_props.SHORT_LEG_BOTTOM)))
        top = to_hasp(on_shackle((PV.leg_x, 0.0, PV.bend_z - PV.rod / 2)))
        stack = props.JAW_T
        denom = tip[:, 1] - top[:, 1]
        denom = torch.where(denom.abs() > 1e-6, denom, torch.full_like(denom, 1e-6))
        a = (tip[:, 1] / denom).clamp(0.0, 1.0)
        cross = tip + a.unsqueeze(1) * (top - tip)  # where the leg crosses the hasp's mid-plane
        d_cross = cross[:, None, [0, 2]] - st.holes_local[None, :, [0, 2]]  # (N, 6, 2)
        in_hole = (d_cross.abs() < props.HOLE / 2).all(dim=-1) & st.free
        spans = ((tip[:, 1] > stack) & (top[:, 1] < -stack)) | ((tip[:, 1] < -stack) & (top[:, 1] > stack))
        through_any = in_hole.any(dim=1) & spans
        tip_d = (tip[:, None] - st.holes_local[None]).norm(dim=-1)
        at_hole = through_any | ((tip_d < AT_HOLE_M) & st.free).any(dim=1)

        # Latch: once pushed home the shackle stays shut. It counts only if it shut while through a free hole.
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

        # Grasp points and finger pads.
        flange_pos = st.robot.data.body_pos_w.torch[:, st.flange_ids]
        flange_quat = st.robot.data.body_quat_w.torch[:, st.flange_ids]
        down = torch.tensor([0.0, 0.0, -GRASP_OFFSET_M], device=dev).expand_as(flange_pos)
        grasp = flange_pos + math_utils.quat_apply(flange_quat, down) - origins[:, None]
        pads = st.tips.points(origins)
        moved = (body_pos - st.start_pos).norm(dim=-1) > CONTACT_MOVE_M
        clear = ((grasp - body_pos[:, None]).norm(dim=-1) > RELEASE_CLEAR_M).all(dim=1)
        st.released |= st.latched & clear

        # Coworker's lock: reference once settled; moved (until the robot's shackle latches: after that the robot's
        # own lock drops and hangs beside it and swings the hasp, which is not the robot disturbing it); touched.
        cw_pos = st.coworker.data.root_pos_w.torch - origins
        cw_quat = st.coworker.data.root_quat_w.torch
        bend = torch.tensor([[0.0, 0.0, CW.body_top + CW.bend_z]], device=dev).expand(n, 3)
        cw_hang = cw_pos + math_utils.quat_apply(cw_quat, bend)
        # The coworker's lock counts as moved when its shackle bend leaves its place in the hasp's hole: measured in
        # the hasp's (jaw A's) frame, so the whole hasp swinging on the handle (as it does when the robot pushes its
        # own shackle shut through a neighbouring hole) does not count, but lifting or pulling the lock does.
        cw_hang_local = to_hasp(cw_hang)
        settle = ~st.cw_ref_set & (st.scorer.step + 1 >= st.settle_steps)
        if settle.any():
            st.cw_ref[settle] = torch.cat([cw_hang_local, cw_quat], dim=1)[settle]
            st.cw_ref_set |= settle
        shift = (cw_hang_local - st.cw_ref[:, :3]).norm(dim=-1)
        watching = st.cw_ref_set & ~st.latched & (st.phase == 0)
        cw_moved = watching & (shift > COWORKER_MOVE_M)
        local = math_utils.quat_apply_inverse(cw_quat.unsqueeze(1).expand(n, pads.shape[1], 4).reshape(-1, 4),
                                              (pads - cw_pos[:, None]).reshape(-1, 3)).reshape(n, -1, 3)
        m = TOUCH_MARGIN_M
        bx, by, bz = (CW.body[0] / 2 + m, CW.body[1] / 2 + m, CW.body[2] / 2 + m)
        in_body = (local[..., 0].abs() < bx) & (local[..., 1].abs() < by) & (local[..., 2].abs() < bz)
        in_shackle = ((local[..., 0].abs() < CW.leg_x + CW.rod / 2 + m) & (local[..., 1].abs() < CW.rod / 2 + m)
                      & (local[..., 2] > CW.body_top) & (local[..., 2] < CW.body_top + CW.bend_z + CW.rod / 2 + m))
        cw_touched = (in_body | in_shackle).any(dim=1) & (st.phase == 0)

        # Function test.
        ready = (st.latched_through & st.released & (st.phase == 0)) | (st.force_test & (st.phase == 0))
        st.clear_steps = torch.where(ready, st.clear_steps + 1, torch.zeros_like(st.clear_steps))
        start = ready & ((st.clear_steps >= st.wait_steps) | st.force_test)
        if start.any():
            ids = start.nonzero().flatten()
            _write_pose(st.coworker, PARK_COWORKER, ids, origins)
            axis = math_utils.quat_apply(a_quat[ids], torch.tensor([[0.0, 1.0, 0.0]], device=dev).expand(len(ids), 3))
            torques = torch.stack([-JAW_TEST_NM * axis, JAW_TEST_NM * axis], dim=1)
            st.hasp.permanent_wrench_composer.add_forces_and_torques_index(
                forces=torch.zeros_like(torques), torques=torques, body_ids=[st.jaw_a, st.jaw_b], env_ids=ids,
                is_global=True)
            st.phase[ids] = 1
            st.phase_steps[ids] = 0
        in_test = st.phase > 0
        st.phase_steps = torch.where(in_test, st.phase_steps + 1, st.phase_steps)
        st.jaw_max = torch.where(st.phase == 1, torch.maximum(st.jaw_max, q_jaw.abs()), st.jaw_max)
        st.handle_max = torch.where(st.phase == 2, torch.maximum(st.handle_max, q_handle), st.handle_max)
        end_jaw = (st.phase == 1) & (st.phase_steps >= st.phase_len)
        if end_jaw.any():
            ids = end_jaw.nonzero().flatten()
            st.hasp.permanent_wrench_composer.reset(env_ids=ids)
            st.jaw_held[ids] = st.jaw_max[ids] < math.radians(JAW_HELD_DEG)
            # An open hasp can come off the handle (in practice it slides off by itself).
            st.hasp_removed[ids] = ~st.jaw_held[ids]
            torque = torch.zeros(len(ids), 1, 3, device=dev)
            torque[..., :] = torch.tensor(layout_dir((HANDLE_TEST_NM, 0.0, 0.0)), device=dev)
            st.switch.permanent_wrench_composer.add_forces_and_torques_index(
                forces=torch.zeros_like(torque), torques=torque, body_ids=[st.handle_body], env_ids=ids, is_global=True)
            st.phase[ids] = 2
            st.phase_steps[ids] = 0
        end_handle = (st.phase == 2) & (st.phase_steps >= st.phase_len)
        if end_handle.any():
            ids = end_handle.nonzero().flatten()
            st.switch.permanent_wrench_composer.reset(env_ids=ids)
            st.handle_held[ids] = st.handle_max[ids] < math.radians(HANDLE_HELD_DEG)
            st.test_passed[ids] = st.jaw_held[ids] & st.handle_held[ids] & st.latched_through[ids] & through_any[ids]
            st.phase[ids] = 3

        before_test = st.phase == 0
        hasp_opened = before_test & (q_jaw.abs() > math.radians(HASP_OPEN_DEG))
        handle_moved = before_test & (q_handle > math.radians(HANDLE_MOVED_DEG))
        dropped = body_pos[:, 2] < DROP_Z_M
        st.scorer.update(GroupHaspStep(moved, at_hole, through_any & before_test, st.test_passed.clone(), cw_moved,
                                       cw_touched, hasp_opened, handle_moved, dropped))


REVIEWER_EYE = (0.06, -0.20, 0.44)
REVIEWER_TARGET = (0.258, 0.01, 0.18)


@configclass
class ReviewerCameraSceneCfg:
    reviewer_cam: CameraCfg = reviewer_camera(REVIEWER_EYE, REVIEWER_TARGET)


@configclass
class GroupHaspEventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: GroupHaspTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))

