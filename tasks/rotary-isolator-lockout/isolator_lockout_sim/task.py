"""Rotary isolator lockout: switch the named enclosed rotary isolator OFF and padlock its handle there.

Mechanism (see isolator_lockout_sim/usd/isolator_props.py; ABB OT16 switch-disconnector with an OHB45J6 handle):
- Each handle turns on a revolute joint about its axis (0 = OFF, +90 deg = ON, clockwise as you face it), with hard
  end stops (joint limits). Turning resistance: a stiff drive re-targeted every step to the current angle, with its
  effort capped at ISOLATOR_TORQUE_NM (Coulomb-like; ABB gives 1 N m for the OT16; the real switch is snap-action).
- The red padlock slide moves in the handle (prismatic joint, light friction the same way). Its tail can only come
  out where the handle's fixed stop has its slot, at OFF; out, the tail in the slot stops the handle turning
  either way (collision).
- A shackle through the slide's hole meets the grip's lower edge if the slide is pushed back in, before the tail
  leaves the slot (collision). Nothing in the code holds the handle: geometry does.
- The robot's Master Lock 410: the shackle on a prismatic joint with a weak spring; the task latches it in software
  once it is pushed within 2 mm of shut.

After-release test (the success check): once the shackle has latched through the named isolator's slide and both
grippers have been clear of the padlock for TEST_WAIT_S, the task pushes the slide back into the handle with
SLIDE_TEST_N for the whole test and, after TEST_PHASE_S, adds TEST_NM on the handle toward ON for TEST_PHASE_S.
Passed if the handle stayed within HELD_DEG of OFF and the shackle is still through the hole.
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

from isolator_lockout_sim.kit import GRASP_OFFSET_M, reset_robot_home, reviewer_camera
from isolator_lockout_sim.scorer import MILESTONES, IsolatorScorer, IsolatorStep
from isolator_lockout_sim.spec import IsolatorSeed, make_seed
from isolator_lockout_sim.usd import isolator_props as props
from isolator_lockout_sim.usd import padlock_props as lock_props

_ASSETS = Path(__file__).resolve().parent / "assets"
SPRING_N_PER_M = 150.0
"""Shackle spring (inferred): 2.4 N at full 16 mm travel."""
LATCH_TRAVEL_M = 0.002
"""The shackle latches once pushed within this distance of shut."""

PV = lock_props.PADLOCKS["ml410"]
LOCK_QUAT = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
"""Lying flat: the shackle's legs point -y (back along the grip toward the hub), the short leg away from the robot
(+x), the body's thickness vertical. (Padlock frame x -> world x, y -> world z, z -> world -y.)"""

ISOLATOR_TORQUE_NM = 0.4
"""Turning torque in the sim (breakaway = running). ABB lists 1 N m operating torque for the OT16 (31 N at the grip,
32 mm out); the station arm's soft joint gains (shoulder 80 N m/rad, about 650 N/m at the hand) give about 15 N there,
so the sim uses 0.4 N m (12.5 N). The after-release test still applies twice the real 1 N m."""
SLIDE_FRICTION_N = 2.0
"""Padlock slide friction (inferred): holds the slide where it is against gravity and rubbing; a light finger pull."""
STIFFNESS = 3000.0
DAMPING = 0.3
ON_RAD = math.pi / 2

TURNED_DEG = 10.0
OFF_DEG = 5.0
BACK_ON_DEG = 30.0
DECOY_MOVED_DEG = 10.0
SLIDE_OUT_TOL_M = 0.003
"""The slide counts as out within this distance of fully out."""
RELEASE_CLEAR_M = 0.06
DROP_Z_M = 0.05

TEST_WAIT_S = 2.0
TEST_PHASE_S = 2.0
TEST_RAMP_S = 1.0
SLIDE_TEST_N = 10.0
"""Pushing the slide back into the handle (a firm thumb)."""
TEST_NM = 2.0
"""Toward ON on the handle: twice the real operating torque (about 67 N at the middle of the grip, 30 mm out)."""
HELD_DEG = 20.0

TARGET_SLOT = 0


def lock_start_pose() -> tuple:
    x, y = props.LOCK_START
    return (x, y, props.LOCK_START_Z + 0.001) + LOCK_QUAT


class Isolators(Object):
    def __init__(self):
        super().__init__(name="isolators", usd_path=str(_ASSETS / "isolators.usda"),
                         object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {
            "handles": ImplicitActuatorCfg(joint_names_expr=["handle_.*_joint"], stiffness=STIFFNESS,
                                           damping=DAMPING, effort_limit_sim=ISOLATOR_TORQUE_NM),
            "slides": ImplicitActuatorCfg(joint_names_expr=["slide_.*_joint"], stiffness=2000.0, damping=1.0,
                                          effort_limit_sim=SLIDE_FRICTION_N),
        }
        cfg.init_state.joint_pos = {"handle_.*_joint": ON_RAD, "slide_.*_joint": 0.0}
        return cfg


class IsolatorPadlock(Object):
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
    def __init__(self, env: ManagerBasedEnv, task: "IsolatorLockoutTask"):
        n, dev = env.num_envs, env.device
        self.lock = env.scene["padlock"]
        self.iso = env.scene["isolators"]
        self.robot = env.scene["robot"]
        self.shackle_joint = self.lock.find_joints("shackle_joint")[0][0]
        self.body_id = self.lock.find_bodies("body")[0][0]
        self.shackle_id = self.lock.find_bodies("shackle")[0][0]
        self.handle_joints = [self.iso.find_joints(f"handle_{k}_joint")[0][0] for k in range(2)]
        self.slide_joints = [self.iso.find_joints(f"slide_{k}_joint")[0][0] for k in range(2)]
        self.handle_bodies = [self.iso.find_bodies(f"handle_{k}")[0][0] for k in range(2)]
        self.slide_bodies = [self.iso.find_bodies(f"slide_{k}")[0][0] for k in range(2)]
        self.flange_ids, _ = self.robot.find_bodies(["left_gripper", "right_gripper"], preserve_order=True)
        self.scorer = IsolatorScorer(n, dev)
        self.next_layout = make_seed(task.default_seed)
        self.layouts: dict[int, IsolatorSeed] = {}
        self.last_reports: dict[int, dict] = {}
        zb = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
        zl = lambda: torch.zeros(n, dtype=torch.long, device=dev)  # noqa: E731
        zf = lambda: torch.zeros(n, device=dev)  # noqa: E731
        self.latched, self.latched_through, self.released, self.force_test = zb(), zb(), zb(), zb()
        self.was_off, self.held, self.test_passed = zb(), zb(), zb()
        self.clear_steps, self.phase, self.phase_steps = zl(), zl(), zl()
        self.on_max, self.slide_in_max = zf(), zf()
        self.wait_steps = max(1, round(TEST_WAIT_S / env.step_dt))
        self.phase_len = max(1, round(TEST_PHASE_S / env.step_dt))


def _write_pose(asset, pose, env_ids, origins, velocity=True):
    dev, n = origins.device, len(env_ids)
    p = torch.tensor(pose, device=dev, dtype=torch.float32).unsqueeze(0).repeat(n, 1)
    p[:, :3] += origins[env_ids]
    asset.write_root_pose_to_sim(p, env_ids=env_ids)
    if velocity:
        asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)


class IsolatorLockoutTask(TaskBase):
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
            target_tag=layout.target_tag, isolator_torque_nm=ISOLATOR_TORQUE_NM, test_nm=TEST_NM,
            shackle_latched=bool(st.latched[env_id]), shackle_latched_through_hole=bool(st.latched_through[env_id]),
            function_test_phase=int(st.phase[env_id]),
            test_handle_max_toward_on_deg=round(math.degrees(float(st.on_max[env_id])), 1),
            test_slide_max_in_mm=round(1000 * float(st.slide_in_max[env_id]), 1),
            test_handle_held=bool(st.held[env_id]), function_test_passed=bool(st.test_passed[env_id]),
            test_end_shackle=getattr(st, "end_debug", None), test_start_padlock=getattr(st, "start_debug", None),
        )
        return r

    def get_scene_cfg(self):
        return ReviewerCameraSceneCfg() if self.reviewer_camera else None

    def get_events_cfg(self):
        return IsolatorEventsCfg(self)

    def get_termination_cfg(self) -> TaskTerminationCfg:
        sequence = [(self._milestone_predicate(m), 0.25) for m in MILESTONES]
        return TaskTerminationCfg(timeout_s=self.episode_length_s,
                                  success=[CompletionCriteria(name="isolator_milestones", predicate_sequence=sequence)])

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
        for buf in (st.latched, st.latched_through, st.released, st.force_test, st.was_off, st.held, st.test_passed):
            buf[env_ids] = False
        for buf in (st.clear_steps, st.phase, st.phase_steps):
            buf[env_ids] = 0
        st.on_max[env_ids] = 0.0
        st.slide_in_max[env_ids] = 0.0
        for asset in (st.lock, st.iso):
            asset.permanent_wrench_composer.reset(env_ids=env_ids)

        # The robot's padlock: open, unlatched, lying on its stand.
        joint = [st.shackle_joint]
        limits = torch.tensor([[[0.0, lock_props.OPEN_TRAVEL]]], device=dev).expand(n, 1, 2).contiguous()
        travel = torch.full((n, 1), lock_props.OPEN_TRAVEL, device=dev)
        st.lock.write_joint_position_limit_to_sim(limits, joint_ids=joint, env_ids=env_ids, warn_limit_violation=False)
        st.lock.write_joint_state_to_sim(travel, torch.zeros_like(travel), joint_ids=joint, env_ids=env_ids)
        st.lock.set_joint_position_target(travel + 0.005, joint_ids=joint, env_ids=env_ids)
        _write_pose(st.lock, lock_start_pose(), env_ids, origins)

        # Both isolators ON, slides in.
        joints = st.handle_joints + st.slide_joints
        q = torch.tensor([[ON_RAD, ON_RAD, 0.0, 0.0]], device=dev).repeat(n, 1)
        st.iso.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=joints, env_ids=env_ids)
        st.iso.set_joint_position_target(q, joint_ids=joints, env_ids=env_ids)
        for env_id in env_ids.tolist():
            st.layouts[env_id] = layout

    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        origins = env.scene.env_origins
        n, dev = env.num_envs, env.device
        t_slot = TARGET_SLOT
        jp = st.iso.data.joint_pos.torch
        # Friction: hold every handle and slide where it is (the drive's effort cap is the friction).
        joints = st.handle_joints + st.slide_joints
        st.iso.set_joint_position_target(jp[:, joints], joint_ids=joints)
        q_t = jp[:, st.handle_joints[t_slot]]
        q_d = jp[:, st.handle_joints[1 - t_slot]]
        off_deg = torch.rad2deg(q_t)  # degrees from OFF toward ON
        slide_q = jp[:, st.slide_joints[t_slot]]
        slide_out = slide_q < -(props.SLIDE_TRAVEL - SLIDE_OUT_TOL_M)

        # The padlock's short leg against the named slide's hole, in the slide frame.
        travel = st.lock.data.joint_pos.torch[:, st.shackle_joint]
        body_pos = st.lock.data.body_pos_w.torch[:, st.body_id] - origins
        sh_pos = st.lock.data.body_pos_w.torch[:, st.shackle_id] - origins
        sh_quat = st.lock.data.body_quat_w.torch[:, st.shackle_id]
        sl_pos = st.iso.data.body_pos_w.torch[:, st.slide_bodies[t_slot]] - origins
        sl_quat = st.iso.data.body_quat_w.torch[:, st.slide_bodies[t_slot]]

        def on_shackle(p):
            return sh_pos + math_utils.quat_apply(sh_quat, torch.tensor([p], device=dev).expand(n, 3))

        def to_slide(p):
            return math_utils.quat_apply_inverse(sl_quat, p - sl_pos)

        tip = to_slide(on_shackle((PV.leg_x, 0.0, lock_props.SHORT_LEG_BOTTOM)))
        top = to_slide(on_shackle((PV.leg_x, 0.0, PV.bend_z - PV.rod / 2)))
        half = props.SLIDE_T / 2
        tip[:, 1] -= props.TAB_YC  # measure across the tab's plane
        top[:, 1] -= props.TAB_YC
        denom = top[:, 1] - tip[:, 1]
        denom = torch.where(denom.abs() > 1e-6, denom, torch.full_like(denom, 1e-6))
        a = (-tip[:, 1] / denom).clamp(0.0, 1.0)
        cross = tip + a.unsqueeze(1) * (top - tip)
        in_hole = (((cross[:, 0] + props.HOLE_D).abs() < props.HOLE / 2)
                   & ((cross[:, 2] + props.HOLE_R).abs() < props.HOLE / 2))
        spans = ((tip[:, 1] < -half) & (top[:, 1] > half)) | ((tip[:, 1] > half) & (top[:, 1] < -half))
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

        # After-release test.
        ready = (st.latched_through & st.released & (st.phase == 0)) | (st.force_test & (st.phase == 0))
        st.clear_steps = torch.where(ready, st.clear_steps + 1, torch.zeros_like(st.clear_steps))
        start = ready & ((st.clear_steps >= st.wait_steps) | st.force_test)
        slide_id, handle_id = st.slide_bodies[t_slot], st.handle_bodies[t_slot]
        if start.any():
            ids = start.nonzero().flatten()
            st.phase[ids] = 1
            st.phase_steps[ids] = 0
            st.start_debug = dict(body_mm=[round(1000 * float(v), 1) for v in body_pos[0]],
                                  quat=[round(float(v), 3) for v in st.lock.data.body_quat_w.torch[0, st.body_id]],
                                  through=bool(through[0]), slide_mm=round(1000 * float(slide_q[0]), 1))
        in_test = st.phase > 0
        st.phase_steps = torch.where(in_test, st.phase_steps + 1, st.phase_steps)
        slide_in_from_out = slide_q + props.SLIDE_TRAVEL
        st.slide_in_max = torch.where(in_test & (st.phase < 3), torch.maximum(st.slide_in_max, slide_in_from_out),
                                      st.slide_in_max)
        st.on_max = torch.where(st.phase == 2, torch.maximum(st.on_max, q_t), st.on_max)
        to_handle_phase = (st.phase == 1) & (st.phase_steps >= st.phase_len)
        st.phase_steps = torch.where(to_handle_phase, torch.zeros_like(st.phase_steps), st.phase_steps)
        st.phase = torch.where(to_handle_phase, torch.full_like(st.phase, 2), st.phase)
        # The test loads ramp up over TEST_RAMP_S (a hand pushes and turns; a step load would fling the 10 g slide
        # into the shackle).
        active = (st.phase == 1) | (st.phase == 2)
        if active.any():
            ids = active.nonzero().flatten()
            st.iso.permanent_wrench_composer.reset(env_ids=ids)
            ramp_steps = max(1, round(TEST_RAMP_S / env.step_dt))
            t_slide = torch.where(st.phase[ids] == 1, st.phase_steps[ids], st.phase_steps[ids] + st.phase_len)
            a_slide = (t_slide.float() / ramp_steps).clamp(max=1.0)
            in_dir = math_utils.quat_apply(sl_quat[ids], torch.tensor([[0.0, 0.0, 1.0]], device=dev).expand(len(ids), 3))
            force = (SLIDE_TEST_N * a_slide.unsqueeze(1) * in_dir).unsqueeze(1)
            st.iso.permanent_wrench_composer.add_forces_and_torques_index(
                forces=force, torques=torch.zeros_like(force), body_ids=[slide_id], env_ids=ids, is_global=True)
            a_handle = torch.where(st.phase[ids] == 2, (st.phase_steps[ids].float() / ramp_steps).clamp(max=1.0),
                                   torch.zeros_like(a_slide))
            torque = torch.zeros(len(ids), 1, 3, device=dev)
            torque[:, 0, 0] = TEST_NM * a_handle  # +x = clockwise as you face the handle = toward ON
            st.iso.permanent_wrench_composer.add_forces_and_torques_index(
                forces=torch.zeros_like(torque), torques=torque, body_ids=[handle_id], env_ids=ids, is_global=True)
        end = (st.phase == 2) & (st.phase_steps >= st.phase_len)
        if end.any():
            ids = end.nonzero().flatten()
            st.iso.permanent_wrench_composer.reset(env_ids=ids)
            st.held[ids] = st.on_max[ids] < math.radians(HELD_DEG)
            st.test_passed[ids] = st.held[ids] & st.latched_through[ids] & through[ids]
            st.phase[ids] = 3
            st.end_debug = dict(through=bool(through[0]), in_hole=bool(in_hole[0]), spans=bool(spans[0]),
                                tip_mm=[round(1000 * float(v), 1) for v in tip[0]],
                                top_mm=[round(1000 * float(v), 1) for v in top[0]],
                                cross_mm=[round(1000 * float(v), 1) for v in cross[0]])

        before_test = st.phase == 0
        at_off = off_deg < OFF_DEG
        st.was_off |= at_off & before_test
        back_on = before_test & st.was_off & (off_deg > BACK_ON_DEG)
        decoy_moved = (q_d - ON_RAD).abs() > math.radians(DECOY_MOVED_DEG)
        dropped = body_pos[:, 2] < DROP_Z_M
        st.scorer.update(IsolatorStep(off_deg < 90.0 - TURNED_DEG, at_off, through & before_test & slide_out & at_off,
                                      st.test_passed.clone(), decoy_moved, back_on, dropped))
        st.debug = dict(off_deg=off_deg, slide_q=slide_q, through=through, travel=travel, tip=tip, top=top)


REVIEWER_EYE = (0.05, -0.30, 0.40)
REVIEWER_TARGET = (0.28, 0.02, 0.17)


@configclass
class ReviewerCameraSceneCfg:
    reviewer_cam: CameraCfg = reviewer_camera(REVIEWER_EYE, REVIEWER_TARGET)


@configclass
class IsolatorEventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: IsolatorLockoutTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))
