"""Hasp apply: put a group-lockout hasp on a disconnect handle and padlock it.

Mechanism (see haspapply_sim/usd/hasp_props.py):
- The switch's operating handle turns on a revolute joint (0 = OFF, 90 deg = ON) with an over-centre detent (a
  spring drive whose target is the nearer end).
- The Master Lock 420 hasp is a free articulation (jaw A + jaw B on the rivet joint). It starts closed, standing in a
  slotted holder. Rivet friction is a rule: the jaw joint's drive target follows the jaw with a
  JAW_SLIP_RAD dead band and JAW_HOLD_K stiffness, so the jaw stays where it is put against up to about
  JAW_HOLD_K * JAW_SLIP_RAD (0.01 N*m) and moves freely past that.
- Closed through the handle's lock-off hole and the fixed ear behind it (jaw A's top bar in both holes), the ring
  stops the handle turning (collision). A shackle through an aligned hole pair of both tails stops jaw B from
  turning (collision); nothing else keeps the jaws shut against the after-release check.
- The robot's Master Lock 410 has a sprung shackle: prismatic joint, spring, a latch rule within 2 mm of shut.

After-release check (the success check): once the shackle has latched through a hole pair and both grasp points
have been at least RELEASE_CLEAR_M from the padlock and from the hasp's ring for TEST_WAIT_S, the task (1) applies
JAW_TEST_NM to pry the jaws open for TEST_PHASE_S; held if they open less than JAW_HELD_DEG; (2) applies
HANDLE_TEST_NM to the handle toward ON for TEST_PHASE_S; held if it stays below HANDLE_HELD_DEG. In control mode
(``control=True``) the check also starts once the hasp is on the handle and released, with no padlock.
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

from haspapply_sim.kit import GRASP_OFFSET_M, Fingertips, q_axis, qmul, reset_robot_home, rotate
from haspapply_sim.scorer import MILESTONES, HaspApplyScorer, HaspApplyStep
from haspapply_sim.spec import HASP_START_DXY, HASP_START_DZ, HASP_Y, PEDESTAL_XY, RING_X, WALL_X, make_seed
from haspapply_sim.usd import hasp_props as props
from haspapply_sim.usd import padlock_props as lock_props

_ASSETS = Path(__file__).resolve().parent / "assets"

PV = lock_props.PADLOCKS["ml410"]
SPRING_N_PER_M = 150.0
"""Shackle spring (N/m)."""
LATCH_TRAVEL_M = 0.002
"""The shackle latches once pushed within this travel of shut."""
LOCK_LYING_QUAT = tuple(float(v) for v in qmul(q_axis((1, 0, 0), math.pi / 2), q_axis((0, 0, 1), math.pi / 2)))
"""The padlock on its side: width (local x) up, thickness (local y) along -x, shackle (local z) along -y."""

DETENT_PEAK_NM = 1.0
"""Torque to push the handle over centre (inferred; to measure on a real switch)."""
DETENT_K = DETENT_PEAK_NM / math.radians(props.HANDLE_ON_DEG / 2)
JAW_HOLD_K = 1.0
"""Rivet friction rule: stiffness (N*m/rad) of the drive that holds the jaw where it was left."""
JAW_SLIP_RAD = 0.01
"""Rivet friction rule: the drive's target follows the jaw once it is more than this away (holds about 0.01 N*m)."""
JAW_DAMPING = 0.02

STAGED_JAW_DEG = 15.0
STAGED_NEAR_M = 0.06
"""0.25: the jaws open at least STAGED_JAW_DEG with the ring's centre within STAGED_NEAR_M of the lock-off holes (the
hasp opened and brought up to the handle; a knocked-over or dropped hasp is far from the holes)."""
AT_HOLE_M = 0.012
RELEASE_CLEAR_M = 0.06
DROP_Z_M = 0.05
HASP_CLOSED_DEG = 5.0
"""On the handle: jaws within this of shut."""
HANDLE_MOVED_DEG = 10.0

TEST_WAIT_S = 2.0
TEST_PHASE_S = 1.5
JAW_TEST_NM = 0.3
"""Prying the tails apart: 6 N on a spring scale through a free tail hole 5 cm from the rivet (inferred hand effort)."""
JAW_HELD_DEG = 10.0
HANDLE_TEST_NM = 1.5
"""1.5 times the detent's peak torque, toward ON: 15 N at the lever's tip, 10 cm from the pivot."""
HANDLE_HELD_DEG = 25.0

LAYOUT_YAW_DEG = -15.0
"""The prop assembly (switch, hasp holder, padlock pedestal) is turned this much counterclockwise (seen from
above) about the vertical through the hasp's ring on the handle, so the top camera sees jaw B's face."""
PIVOT_XY = (RING_X, HASP_Y)
_YAW_Q = q_axis((0, 0, 1), math.radians(LAYOUT_YAW_DEG))

TAB_HOLE_IN_HANDLE = ((props.TAB_X[0] + props.TAB_X[1]) / 2 - props.PIVOT_X, 0.0, props.LOCKOFF_Z - props.PIVOT_Z)
"""The handle's lock-off hole centre in the handle body's frame (origin at the pivot)."""
EAR_HOLE_IN_SWITCH = ((props.EAR_X[0] + props.EAR_X[1]) / 2 - WALL_X, 0.0, props.LOCKOFF_Z)
"""The fixed ear's hole centre in the switch's (enclosure's) frame."""


def layout_dir(v) -> tuple:
    return tuple(float(c) for c in rotate(_YAW_Q, v))


def layout_pose(pose) -> tuple:
    """(x, y, z, qx, qy, qz, qw) of the unturned layout -> the turned layout (desk frame)."""
    x, y, z = pose[:3]
    dx, dy, _ = rotate(_YAW_Q, (x - PIVOT_XY[0], y - PIVOT_XY[1], 0.0))
    q = qmul(_YAW_Q, pose[3:])
    return (PIVOT_XY[0] + float(dx), PIVOT_XY[1] + float(dy), float(z)) + tuple(float(c) for c in q)


PARK = layout_pose((props.PARK_X, 0.10, props.PARK_Z + 0.03, 0.0, 0.0, 0.0, 1.0))


def lock_start_pose() -> tuple:
    x, y = PEDESTAL_XY
    return layout_pose((x, y, props.PEDESTAL_TOP + PV.body[0] / 2 + 0.001) + LOCK_LYING_QUAT)


def hasp_on_handle_pose() -> tuple:
    return layout_pose(props.nominal_hasp_pos() + (0.0, 0.0, 0.0, 1.0))


def hasp_start_pose() -> tuple:
    x, y, z = props.nominal_hasp_pos()
    return layout_pose((x + HASP_START_DXY[0], y + HASP_START_DXY[1], z + HASP_START_DZ + 0.0005, 0.0, 0.0, 0.0, 1.0))


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
        super().__init__(name="hasp", usd_path=str(_ASSETS / "hasp.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=_pose(hasp_start_pose()))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {"jaw": ImplicitActuatorCfg(joint_names_expr=["jaw_joint"], stiffness=JAW_HOLD_K,
                                                    damping=JAW_DAMPING)}
        cfg.init_state.joint_pos = {"jaw_joint": 0.0}
        return cfg


class Padlock(Object):
    """The robot's Master Lock 410 (sprung shackle, the task latches it)."""

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
    """The padlock's pedestal and the hasp's holder (one fixed asset)."""

    def __init__(self):
        super().__init__(name="stands", usd_path=str(_ASSETS / "pedestal.usda"), object_type=ObjectType.BASE,
                         initial_pose=_pose(layout_pose((0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0))))


class _TaskState:
    def __init__(self, env: ManagerBasedEnv, task: "HaspApplyTask"):
        n, dev = env.num_envs, env.device
        self.lock = env.scene["padlock"]
        self.hasp = env.scene["hasp"]
        self.switch = env.scene["switch"]
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
        self.enclosure_body = self.switch.find_bodies("enclosure")[0][0]
        self.flange_ids, _ = self.robot.find_bodies(["left_gripper", "right_gripper"], preserve_order=True)
        self.scorer = HaspApplyScorer(n, dev)
        self.last_reports: dict[int, dict] = {}
        self.holes_local = torch.tensor([props.hole_local(c, r) for c in (0, 1) for r in range(3)], device=dev)
        zb = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
        zl = lambda: torch.zeros(n, dtype=torch.long, device=dev)  # noqa: E731
        zf = lambda: torch.zeros(n, device=dev)  # noqa: E731
        self.latched, self.latched_through, self.released, self.force_test = zb(), zb(), zb(), zb()
        self.jaw_held, self.handle_held, self.test_passed, self.hasp_removed = zb(), zb(), zb(), zb()
        self.hasp_on, self.hasp_released = zb(), zb()
        self.hole_used = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.clear_steps, self.phase, self.phase_steps = zl(), zl(), zl()
        self.jaw_max, self.handle_max, self.jaw_target = zf(), zf(), zf()
        self.hasp_start = torch.zeros(n, 3, device=dev)
        self.lock_start = torch.zeros(n, 3, device=dev)
        self.wait_steps = max(1, round(TEST_WAIT_S / env.step_dt))
        self.phase_len = max(1, round(TEST_PHASE_S / env.step_dt))
        self.control = task.control


def _write_pose(asset, pose, env_ids, origins, velocity=True):
    dev, n = origins.device, len(env_ids)
    p = torch.tensor(pose, device=dev, dtype=torch.float32).unsqueeze(0).repeat(n, 1)
    p[:, :3] += origins[env_ids]
    asset.write_root_pose_to_sim(p, env_ids=env_ids)
    if velocity:
        asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)


class HaspApplyTask(TaskBase):
    def __init__(self, default_seed: int = 0, episode_length_s: float = 180.0, control: bool = False):
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
            hasp_on_handle=bool(st.hasp_on[env_id]), hasp_released=bool(st.hasp_released[env_id]),
            jaw_deg=round(math.degrees(float(st.hasp.data.joint_pos.torch[env_id, st.jaw_joint])), 1),
            shackle_latched=bool(st.latched[env_id]), shackle_latched_through_hole=bool(st.latched_through[env_id]),
            hole_used=int(st.hole_used[env_id]), function_test_phase=int(st.phase[env_id]),
            test_jaw_max_deg=round(math.degrees(float(st.jaw_max[env_id])), 1),
            test_handle_max_deg=round(math.degrees(float(st.handle_max[env_id])), 1),
            test_jaw_held=bool(st.jaw_held[env_id]), test_handle_held=bool(st.handle_held[env_id]),
            test_hasp_removable=bool(st.hasp_removed[env_id]), function_test_passed=bool(st.test_passed[env_id]),
        )
        return r

    def get_scene_cfg(self):
        return None

    def get_events_cfg(self):
        return HaspApplyEventsCfg(self)

    def get_termination_cfg(self) -> TaskTerminationCfg:
        sequence = [(self._milestone_predicate(m), 0.25) for m in MILESTONES]
        return TaskTerminationCfg(timeout_s=self.episode_length_s,
                                  success=[CompletionCriteria(name="haspapply_milestones", predicate_sequence=sequence)])

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
        for buf in (st.latched, st.latched_through, st.released, st.force_test, st.jaw_held, st.handle_held,
                    st.test_passed, st.hasp_removed, st.hasp_on, st.hasp_released):
            buf[env_ids] = False
        for buf in (st.clear_steps, st.phase, st.phase_steps):
            buf[env_ids] = 0
        st.jaw_max[env_ids] = 0.0
        st.handle_max[env_ids] = 0.0
        st.jaw_target[env_ids] = 0.0
        st.hole_used[env_ids] = -1
        for asset in (st.lock, st.hasp, st.switch):
            asset.permanent_wrench_composer.reset(env_ids=env_ids)

        # The padlock: open, unlatched, on its pedestal.
        joint = [st.shackle_joint]
        limits = torch.tensor([[[0.0, lock_props.OPEN_TRAVEL]]], device=dev).expand(n, 1, 2).contiguous()
        travel = torch.full((n, 1), lock_props.OPEN_TRAVEL, device=dev)
        st.lock.write_joint_position_limit_to_sim(limits, joint_ids=joint, env_ids=env_ids, warn_limit_violation=False)
        st.lock.write_joint_state_to_sim(travel, torch.zeros_like(travel), joint_ids=joint, env_ids=env_ids)
        st.lock.set_joint_position_target(travel + 0.005, joint_ids=joint, env_ids=env_ids)
        _write_pose(st.lock, lock_start_pose(), env_ids, origins)
        st.lock_start[env_ids] = torch.tensor(lock_start_pose()[:3], device=dev)

        # The hasp: closed, in its holder. The handle: OFF.
        zero = torch.zeros(n, 1, device=dev)
        st.hasp.write_joint_state_to_sim(zero, zero, joint_ids=[st.jaw_joint], env_ids=env_ids)
        st.hasp.set_joint_position_target(zero, joint_ids=[st.jaw_joint], env_ids=env_ids)
        _write_pose(st.hasp, hasp_start_pose(), env_ids, origins)
        st.hasp_start[env_ids] = torch.tensor(hasp_start_pose()[:3], device=dev)
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

        # Rivet friction: the jaw drive's target follows the jaw with a dead band.
        q_jaw = st.hasp.data.joint_pos.torch[:, st.jaw_joint]
        st.jaw_target = torch.minimum(torch.maximum(st.jaw_target, q_jaw - JAW_SLIP_RAD), q_jaw + JAW_SLIP_RAD)
        st.hasp.set_joint_position_target(st.jaw_target.unsqueeze(1), joint_ids=[st.jaw_joint])

        a_pos = st.hasp.data.body_pos_w.torch[:, st.jaw_a] - origins
        a_quat = st.hasp.data.body_quat_w.torch[:, st.jaw_a]

        def to_hasp(p):
            return math_utils.quat_apply_inverse(a_quat, p - a_pos)

        # Hasp on the handle: closed, its top bar through the handle's and the ear's holes.
        h_pos = st.switch.data.body_pos_w.torch[:, st.handle_body] - origins
        h_quat = st.switch.data.body_quat_w.torch[:, st.handle_body]
        e_pos = st.switch.data.body_pos_w.torch[:, st.enclosure_body] - origins
        e_quat = st.switch.data.body_quat_w.torch[:, st.enclosure_body]
        tab_hole = h_pos + math_utils.quat_apply(h_quat, torch.tensor([TAB_HOLE_IN_HANDLE], device=dev).expand(n, 3))
        ear_hole = e_pos + math_utils.quat_apply(e_quat, torch.tensor([EAR_HOLE_IN_SWITCH], device=dev).expand(n, 3))

        def bar_through(hole):
            h = to_hasp(hole)
            return ((h[:, 0].abs() < 0.0085) & (h[:, 1].abs() < props.LOCKOFF_HOLE / 2)
                    & (h[:, 2] > props.JAW_R - 0.0015) & (h[:, 2] < props.JAW_R + props.JAW_W + 0.0015))

        closed = q_jaw.abs() < math.radians(HASP_CLOSED_DEG)
        st.hasp_on = closed & bar_through(tab_hole) & bar_through(ear_hole)
        holes_mid = 0.5 * (tab_hole + ear_hole)
        staged = (q_jaw > math.radians(STAGED_JAW_DEG)) & ((a_pos - holes_mid).norm(dim=-1) < STAGED_NEAR_M)

        # The padlock: short leg against the hasp's holes, in the hasp (jaw A) frame.
        travel = st.lock.data.joint_pos.torch[:, st.shackle_joint]
        body_pos = st.lock.data.body_pos_w.torch[:, st.body_id] - origins
        sh_pos = st.lock.data.body_pos_w.torch[:, st.shackle_id] - origins
        sh_quat = st.lock.data.body_quat_w.torch[:, st.shackle_id]

        def on_shackle(p):
            return sh_pos + math_utils.quat_apply(sh_quat, torch.tensor([p], device=dev).expand(n, 3))

        tip = to_hasp(on_shackle((PV.leg_x, 0.0, lock_props.SHORT_LEG_BOTTOM)))
        top = to_hasp(on_shackle((PV.leg_x, 0.0, PV.bend_z - PV.rod / 2)))
        stack = props.JAW_T
        denom = tip[:, 1] - top[:, 1]
        denom = torch.where(denom.abs() > 1e-6, denom, torch.full_like(denom, 1e-6))
        a = (tip[:, 1] / denom).clamp(0.0, 1.0)
        cross = tip + a.unsqueeze(1) * (top - tip)  # where the leg crosses the hasp's mid-plane
        d_cross = cross[:, None, [0, 2]] - st.holes_local[None, :, [0, 2]]  # (N, 6, 2)
        in_hole = (d_cross.abs() < props.HOLE / 2).all(dim=-1)
        spans = ((tip[:, 1] > stack) & (top[:, 1] < -stack)) | ((tip[:, 1] < -stack) & (top[:, 1] > stack))
        through_any = in_hole.any(dim=1) & spans & closed
        tip_d = (tip[:, None] - st.holes_local[None]).norm(dim=-1)
        at_hole = through_any | (tip_d < AT_HOLE_M).any(dim=1)

        # Latch: once pushed home the shackle stays shut. It counts only if it shut while through a hole pair.
        newly = ~st.latched & (travel < LATCH_TRAVEL_M)
        if newly.any():
            ids = newly.nonzero().flatten()
            limits = torch.tensor([[[0.0, 0.0005]]], device=dev).expand(len(ids), 1, 2).contiguous()
            st.lock.write_joint_position_limit_to_sim(limits, joint_ids=[st.shackle_joint], env_ids=ids,
                                                      warn_limit_violation=False)
            st.lock.set_joint_position_target(torch.zeros(len(ids), 1, device=dev), joint_ids=[st.shackle_joint],
                                              env_ids=ids)
            st.latched |= newly
            st.latched_through |= newly & through_any & st.hasp_on
            st.hole_used = torch.where(newly & through_any, in_hole.float().argmax(dim=1), st.hole_used)

        # Release: both grasp points clear of the padlock and of the hasp's ring.
        flange_pos = st.robot.data.body_pos_w.torch[:, st.flange_ids]
        flange_quat = st.robot.data.body_quat_w.torch[:, st.flange_ids]
        down = torch.tensor([0.0, 0.0, -GRASP_OFFSET_M], device=dev).expand_as(flange_pos)
        grasp = flange_pos + math_utils.quat_apply(flange_quat, down) - origins[:, None]
        clear_lock = ((grasp - body_pos[:, None]).norm(dim=-1) > RELEASE_CLEAR_M).all(dim=1)
        clear_hasp = ((grasp - a_pos[:, None]).norm(dim=-1) > RELEASE_CLEAR_M).all(dim=1)
        st.released |= st.latched & clear_lock & clear_hasp
        st.hasp_released = st.hasp_on & clear_hasp & clear_lock

        # After-release check.
        ready = (st.latched_through & st.released & (st.phase == 0)) | (st.force_test & (st.phase == 0))
        if st.control:
            ready = ready | (st.hasp_released & (st.phase == 0))
        st.clear_steps = torch.where(ready, st.clear_steps + 1, torch.zeros_like(st.clear_steps))
        start = ready & (st.clear_steps >= st.wait_steps)
        if start.any():
            ids = start.nonzero().flatten()
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
            st.test_passed[ids] = (st.jaw_held[ids] & st.handle_held[ids] & st.latched_through[ids]
                                   & through_any[ids])
            st.phase[ids] = 3
            for i in ids.tolist():
                print(f"[hasp-apply] after-release check: jaws opened {math.degrees(float(st.jaw_max[i])):.1f} deg "
                      f"(held: {bool(st.jaw_held[i])}), handle turned {math.degrees(float(st.handle_max[i])):.1f} deg "
                      f"(held: {bool(st.handle_held[i])}), passed: {bool(st.test_passed[i])}", flush=True)

        before_test = st.phase == 0
        handle_moved = before_test & (q_handle > math.radians(HANDLE_MOVED_DEG))
        lock_dropped = body_pos[:, 2] < DROP_Z_M
        hasp_dropped = a_pos[:, 2] < DROP_Z_M
        st.scorer.update(HaspApplyStep(staged & (st.phase == 0), st.hasp_on & before_test, through_any & st.hasp_on & before_test,
                                       st.test_passed.clone(), handle_moved, lock_dropped, hasp_dropped, at_hole))


@configclass
class HaspApplyEventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: HaspApplyTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))
