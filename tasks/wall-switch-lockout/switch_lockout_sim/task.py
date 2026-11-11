"""Wall switch lockout: switch the named wall toggle switch OFF, shut its Master Lock 496B cover and padlock it.

Mechanism (see switch_lockout_sim/usd/switch_props.py):
- Each toggle turns on a revolute joint about y at its pivot (+30 deg = up = ON, -30 deg = down = OFF; end stops are
  joint limits). The snap: a stiff drive aimed at whichever end the toggle is nearer, its effort capped at
  TOGGLE_SNAP_NM, so it takes a small push over the middle and then snaps to the other end.
- Each 496B cover hangs on a revolute hinge at the top of its base (0 = shut, about 192 deg = open, leaning on the
  wall), with light damping. Gravity holds it shut; nothing in the code holds it.
- Lock-out by geometry: shut, the cover's inner face stops the toggle about 9 deg from its end (it cannot be pushed
  through the middle), and a shackle through the base's ring sits in front of the cover's face, so the cover can
  swing out only about 4 deg.
- The robot's Master Lock 410: the shackle on a prismatic joint with a weak spring; the task latches it in software
  once it is pushed within 2 mm of shut.

After-release test (the success check): once the shackle has latched through the named switch's ring and both
grippers have been clear of the padlock for TEST_WAIT_S, the task pulls the cover's bottom away from the wall with
COVER_TEST_N for the whole test and, after TEST_PHASE_S, adds TOGGLE_TEST_NM on the toggle toward ON for TEST_PHASE_S.
Passed if the cover stayed within COVER_HELD_DEG of shut, the toggle never reached the middle of its throw, and the
shackle is still through the ring. `force_function_test(env)` starts the same test at once (the control run).
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

from switch_lockout_sim.kit import GRASP_OFFSET_M, reset_robot_home, reviewer_camera
from switch_lockout_sim.scorer import MILESTONES, SwitchScorer, SwitchStep
from switch_lockout_sim.spec import SwitchSeed, make_seed
from switch_lockout_sim.usd import padlock_props as lock_props
from switch_lockout_sim.usd import switch_props as props

_ASSETS = Path(__file__).resolve().parent / "assets"
SPRING_N_PER_M = 150.0
"""Shackle spring (inferred): 2.4 N at full 16 mm travel."""
LATCH_TRAVEL_M = 0.002

PV = lock_props.PADLOCKS["ml410"]
LOCK_QUAT = (0.0, 0.0, -math.sqrt(0.5), math.sqrt(0.5))
"""Upright, shackle up, its plane parallel to the wall: the short leg toward -y (the right, as the robot sees it),
the body's thickness along x. (Padlock frame x -> world -y, y -> world x, z -> world z.)"""

THROW = math.radians(props.THROW_DEG)
TOGGLE_SNAP_NM = 0.06
"""Snap torque (inferred; not published for wall switches): about 3 N at the middle of the toggle, 20 mm out."""
TOGGLE_K = 3.0
COVER_DAMPING = 0.01
"""Hinge damping, N m s/rad (inferred): the cover falls shut in about 2 s once past its balance point."""
COVER_OPEN = math.radians(props.COVER_OPEN_DEG)

OFF_DEG = -25.0
"""The toggle counts as OFF within 5 deg of its lower stop."""
OFF_SIDE_DEG = -15.0
COVER_SHUT_DEG = 3.0
RELEASE_CLEAR_M = 0.06
DROP_Z_M = 0.05

TEST_WAIT_S = 2.0
TEST_PHASE_S = 2.0
TEST_RAMP_S = 1.0
COVER_TEST_N = 10.0
COVER_TEST_ARM_M = 0.115
"""The pull acts at the cover's bottom edge, 11.5 cm below the hinge (applied as the equivalent torque)."""
TOGGLE_TEST_NM = 0.20
"""Toward ON on the toggle: over three times the snap (10 N at the middle of the toggle)."""
COVER_HELD_DEG = 10.0

TARGET_SLOT = 0


def lock_start_pose() -> tuple:
    x, y = props.LOCK_START
    return (x, y, props.LOCK_START_Z + 0.001) + LOCK_QUAT


class Switches(Object):
    def __init__(self):
        super().__init__(name="switches", usd_path=str(_ASSETS / "switches.usda"),
                         object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {
            "toggles": ImplicitActuatorCfg(joint_names_expr=["toggle_.*_joint"], stiffness=TOGGLE_K, damping=0.002,
                                           effort_limit_sim=TOGGLE_SNAP_NM),
            "covers": ImplicitActuatorCfg(joint_names_expr=["cover_.*_joint"], stiffness=0.0, damping=COVER_DAMPING),
        }
        cfg.init_state.joint_pos = {"toggle_.*_joint": THROW, "cover_.*_joint": COVER_OPEN}
        return cfg


class SwitchPadlock(Object):
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
    def __init__(self, env: ManagerBasedEnv, task: "WallSwitchTask"):
        n, dev = env.num_envs, env.device
        self.lock = env.scene["padlock"]
        self.sw = env.scene["switches"]
        self.robot = env.scene["robot"]
        self.shackle_joint = self.lock.find_joints("shackle_joint")[0][0]
        self.body_id = self.lock.find_bodies("body")[0][0]
        self.shackle_id = self.lock.find_bodies("shackle")[0][0]
        self.toggle_joints = [self.sw.find_joints(f"toggle_{k}_joint")[0][0] for k in range(2)]
        self.cover_joints = [self.sw.find_joints(f"cover_{k}_joint")[0][0] for k in range(2)]
        self.toggle_bodies = [self.sw.find_bodies(f"toggle_{k}")[0][0] for k in range(2)]
        self.cover_bodies = [self.sw.find_bodies(f"cover_{k}")[0][0] for k in range(2)]
        self.flange_ids, _ = self.robot.find_bodies(["left_gripper", "right_gripper"], preserve_order=True)
        self.scorer = SwitchScorer(n, dev)
        self.next_layout = make_seed(task.default_seed)
        self.layouts: dict[int, SwitchSeed] = {}
        self.last_reports: dict[int, dict] = {}
        zb = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
        zl = lambda: torch.zeros(n, dtype=torch.long, device=dev)  # noqa: E731
        zf = lambda: torch.zeros(n, device=dev)  # noqa: E731
        self.latched, self.latched_through, self.released, self.force_test = zb(), zb(), zb(), zb()
        self.held, self.test_passed = zb(), zb()
        self.clear_steps, self.phase, self.phase_steps = zl(), zl(), zl()
        self.cover_max, self.toggle_max = zf(), zf()
        self.wait_steps = max(1, round(TEST_WAIT_S / env.step_dt))
        self.phase_len = max(1, round(TEST_PHASE_S / env.step_dt))


def _write_pose(asset, pose, env_ids, origins, velocity=True):
    dev, n = origins.device, len(env_ids)
    p = torch.tensor(pose, device=dev, dtype=torch.float32).unsqueeze(0).repeat(n, 1)
    p[:, :3] += origins[env_ids]
    asset.write_root_pose_to_sim(p, env_ids=env_ids)
    if velocity:
        asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)


class WallSwitchTask(TaskBase):
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
        """Control runs: start the after-release test now, lock or no lock."""
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
            target_tag=layout.target_tag, toggle_snap_nm=TOGGLE_SNAP_NM, cover_test_n=COVER_TEST_N,
            toggle_test_nm=TOGGLE_TEST_NM,
            shackle_latched=bool(st.latched[env_id]), shackle_latched_through_ring=bool(st.latched_through[env_id]),
            function_test_phase=int(st.phase[env_id]),
            test_cover_max_open_deg=round(math.degrees(float(st.cover_max[env_id])), 1),
            test_toggle_max_deg=round(math.degrees(float(st.toggle_max[env_id])), 1),
            test_held=bool(st.held[env_id]), function_test_passed=bool(st.test_passed[env_id]),
        )
        return r

    def get_scene_cfg(self):
        if os.environ.get("WSL_DEBUG_CAM") == "1":
            return _debug_cams()
        return ReviewerCameraSceneCfg() if self.reviewer_camera else None

    def get_events_cfg(self):
        return SwitchEventsCfg(self)

    def get_termination_cfg(self) -> TaskTerminationCfg:
        sequence = [(self._milestone_predicate(m), 0.25) for m in MILESTONES]
        return TaskTerminationCfg(timeout_s=self.episode_length_s,
                                  success=[CompletionCriteria(name="switch_milestones", predicate_sequence=sequence)])

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
        for buf in (st.latched, st.latched_through, st.released, st.force_test, st.held, st.test_passed):
            buf[env_ids] = False
        for buf in (st.clear_steps, st.phase, st.phase_steps):
            buf[env_ids] = 0
        st.cover_max[env_ids] = 0.0
        st.toggle_max[env_ids] = -THROW
        for asset in (st.lock, st.sw):
            asset.permanent_wrench_composer.reset(env_ids=env_ids)

        # The robot's padlock: open, unlatched, standing on its post.
        joint = [st.shackle_joint]
        limits = torch.tensor([[[0.0, lock_props.OPEN_TRAVEL]]], device=dev).expand(n, 1, 2).contiguous()
        travel = torch.full((n, 1), lock_props.OPEN_TRAVEL, device=dev)
        st.lock.write_joint_position_limit_to_sim(limits, joint_ids=joint, env_ids=env_ids, warn_limit_violation=False)
        st.lock.write_joint_state_to_sim(travel, torch.zeros_like(travel), joint_ids=joint, env_ids=env_ids)
        st.lock.set_joint_position_target(travel + 0.005, joint_ids=joint, env_ids=env_ids)
        _write_pose(st.lock, lock_start_pose(), env_ids, origins)

        # Both switches ON, both covers open.
        joints = st.toggle_joints + st.cover_joints
        q = torch.tensor([[THROW, THROW, COVER_OPEN, COVER_OPEN]], device=dev).repeat(n, 1)
        st.sw.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=joints, env_ids=env_ids)
        st.sw.set_joint_position_target(q, joint_ids=joints, env_ids=env_ids)
        for env_id in env_ids.tolist():
            st.layouts[env_id] = layout

    def through_ring(self, st: _TaskState, origins, slot: int = TARGET_SLOT) -> torch.Tensor:
        """The padlock's short leg passes down through the slot's ring hole (its tip below the ring, the leg's top
        above it, crossing the ring's mid-plane inside the hole)."""
        n, dev = origins.shape[0], origins.device
        sh_pos = st.lock.data.body_pos_w.torch[:, st.shackle_id] - origins
        sh_quat = st.lock.data.body_quat_w.torch[:, st.shackle_id]

        def on_shackle(p):
            return sh_pos + math_utils.quat_apply(sh_quat, torch.tensor([p], device=dev).expand(n, 3))

        tip = on_shackle((PV.leg_x, 0.0, lock_props.SHORT_LEG_BOTTOM))
        top = on_shackle((PV.leg_x, 0.0, PV.bend_z - PV.rod / 2))
        hx, hy, hz = props.hole_world(slot)
        denom = top[:, 2] - tip[:, 2]
        denom = torch.where(denom.abs() > 1e-6, denom, torch.full_like(denom, 1e-6))
        a = ((hz - tip[:, 2]) / denom).clamp(0.0, 1.0)
        cross = tip + a.unsqueeze(1) * (top - tip)
        in_hole = ((cross[:, 0] - hx).abs() < props.HOLE / 2) & ((cross[:, 1] - hy).abs() < props.HOLE / 2)
        spans = (tip[:, 2] < props.RING_Z[0]) & (top[:, 2] > props.RING_Z[1])
        self._dbg = dict(tip=tip, top=top, cross=cross)
        return in_hole & spans

    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        origins = env.scene.env_origins
        dev = env.device
        t_slot = TARGET_SLOT
        jp = st.sw.data.joint_pos.torch
        # The toggles' snap: each drive aims at the end its toggle is nearer.
        tq = jp[:, st.toggle_joints]
        st.sw.set_joint_position_target(torch.where(tq >= 0, THROW, -THROW), joint_ids=st.toggle_joints)
        q_t = tq[:, t_slot]
        q_d = tq[:, 1 - t_slot]
        cover_q = jp[:, st.cover_joints[t_slot]]
        at_off = q_t < math.radians(OFF_DEG)
        off_side = q_t < math.radians(OFF_SIDE_DEG)
        shut = cover_q < math.radians(COVER_SHUT_DEG)

        travel = st.lock.data.joint_pos.torch[:, st.shackle_joint]
        body_pos = st.lock.data.body_pos_w.torch[:, st.body_id] - origins
        through = self.through_ring(st, origins)

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

        # After-release test.
        ready = (st.latched_through & st.released & (st.phase == 0)) | (st.force_test & (st.phase == 0))
        st.clear_steps = torch.where(ready, st.clear_steps + 1, torch.zeros_like(st.clear_steps))
        start = ready & ((st.clear_steps >= st.wait_steps) | st.force_test)
        if start.any():
            ids = start.nonzero().flatten()
            st.phase[ids] = 1
            st.phase_steps[ids] = 0
            st.toggle_max[ids] = q_t[ids]
        in_test = (st.phase == 1) | (st.phase == 2)
        st.phase_steps = torch.where(st.phase > 0, st.phase_steps + 1, st.phase_steps)
        st.cover_max = torch.where(in_test, torch.maximum(st.cover_max, cover_q), st.cover_max)
        st.toggle_max = torch.where(in_test, torch.maximum(st.toggle_max, q_t), st.toggle_max)
        to_toggle_phase = (st.phase == 1) & (st.phase_steps >= st.phase_len)
        st.phase_steps = torch.where(to_toggle_phase, torch.zeros_like(st.phase_steps), st.phase_steps)
        st.phase = torch.where(to_toggle_phase, torch.full_like(st.phase, 2), st.phase)
        # Test loads ramp up over TEST_RAMP_S (a hand pulls and pushes; a step load would fling the light parts).
        if in_test.any():
            ids = in_test.nonzero().flatten()
            st.sw.permanent_wrench_composer.reset(env_ids=ids)
            ramp_steps = max(1, round(TEST_RAMP_S / env.step_dt))
            t_cover = torch.where(st.phase[ids] == 1, st.phase_steps[ids], st.phase_steps[ids] + st.phase_len)
            a_cover = (t_cover.float() / ramp_steps).clamp(max=1.0)
            a_toggle = torch.where(st.phase[ids] == 2, (st.phase_steps[ids].float() / ramp_steps).clamp(max=1.0),
                                   torch.zeros_like(a_cover))
            # +y torque opens the cover and pushes the toggle toward ON.
            for body, mag in ((st.cover_bodies[t_slot], COVER_TEST_N * COVER_TEST_ARM_M * a_cover),
                              (st.toggle_bodies[t_slot], TOGGLE_TEST_NM * a_toggle)):
                torque = torch.zeros(len(ids), 1, 3, device=dev)
                torque[:, 0, 1] = mag
                st.sw.permanent_wrench_composer.add_forces_and_torques_index(
                    forces=torch.zeros_like(torque), torques=torque, body_ids=[body], env_ids=ids, is_global=True)
        end = (st.phase == 2) & (st.phase_steps >= st.phase_len)
        if end.any():
            ids = end.nonzero().flatten()
            st.sw.permanent_wrench_composer.reset(env_ids=ids)
            st.held[ids] = (st.cover_max[ids] < math.radians(COVER_HELD_DEG)) & (st.toggle_max[ids] < 0.0)
            st.test_passed[ids] = st.held[ids] & st.latched_through[ids] & through[ids]
            st.phase[ids] = 3

        before_test = st.phase == 0
        decoy_off = q_d < 0.0
        dropped = body_pos[:, 2] < DROP_Z_M
        st.scorer.update(SwitchStep(at_off, shut & off_side, through & before_test & shut & off_side,
                                    st.test_passed.clone(), decoy_off, dropped))
        st.debug = dict(toggle=q_t, decoy=q_d, cover=cover_q, through=through, travel=travel)


REVIEWER_EYE = (0.05, -0.30, 0.40)
REVIEWER_TARGET = (0.29, 0.0, 0.22)


@configclass
class ReviewerCameraSceneCfg:
    reviewer_cam: CameraCfg = reviewer_camera(REVIEWER_EYE, REVIEWER_TARGET)


@configclass
class SwitchEventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: WallSwitchTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))


DEBUG_VIEWS = {
    "debug_cam": ((0.10, -0.16, 0.32), (0.29, -0.03, 0.22)),
    "debug_cam2": ((0.20, -0.26, 0.22), (0.29, -0.03, 0.21)),
    "debug_cam3": ((0.17, 0.07, 0.14), (0.285, -0.03, 0.18)),
}
"""Development-only views (WSL_DEBUG_CAM=1), desk frame: S1 from the front right, from the right, the ring from below."""


def _debug_cams():
    """Extra 640 x 360 cameras for development; not part of the policy's observations."""
    import isaaclab.sim as sim_utils
    from yam_arena.embodiment import RENDERER

    from switch_lockout_sim.kit import look_at_xyzw
    from switch_lockout_sim.station import Z0

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
