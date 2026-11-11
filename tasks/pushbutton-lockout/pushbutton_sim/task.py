"""Lock out a START push button with a push-button lockout cover: Arena props and TaskBase.

Mechanism (pushbutton_sim/usd/pb_props.py):
- Station: fixed-base articulation; each push button's plunger slides on a prismatic joint (0 up, -4.3 mm fully in)
  pushed up by this task's preloaded joint drive (2.0 N at rest, 3.8 N at the 2.6 mm contact point).
- Cover (Master Lock S2151): free articulation, two half-shells on the vertical ``hinge`` (0 shut), light damping
  and nothing else holding it; the bottom lips go under the install base's flange when shut around it.
- Padlock (Master Lock 410): body and shackle on a prismatic joint with a weak spring; once pushed within
  LATCH_TRAVEL_M of shut it latches (a joint-limit rule; the real lock's latch is internal).

Measured each step: each plunger's travel; the cover's pose (the back half's frame: origin on the axis at the bottom
edge) against the buttons; the hinge angle; whether the shackle's short leg passes through both ears' holes (each ear
in its own half's frame).

Function test (the success check): once the shackle has latched through both ears with the cover shut on the START
button and both grasp points have been clear of the padlock and the cover for TEST_WAIT_S, three loads are applied
in turn for TEST_PHASE_S each, as a reviewer would with a spring scale: (1) PRESS_N down on the cover's top over the
button: passed if the START plunger moves less than PRESS_HELD_M; (2) the cover pried open, PRY_NM about the hinge
(the two halves turned apart): passed if it opens less than PRY_HELD_DEG; (3) LIFT_N straight up on the cover: passed
if it rises less than LIFT_HELD_M. ``force_function_test(env)`` starts the same test at once (the control run: cover
shut, no padlock).
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

from pushbutton_sim.kit import GRASP_OFFSET_M, q_axis, reset_robot_home
from pushbutton_sim.scorer import MILESTONES, PushButtonLockoutScorer, Step
from pushbutton_sim.spec import BUTTON_XY, COVER_OPEN_DEG, PANEL_Z, PEDESTAL_XY, Layout, make_seed
from pushbutton_sim.usd import padlock_props as lock_props
from pushbutton_sim.usd import pb_props as props

_ASSETS = Path(__file__).resolve().parent / "assets"

PV = lock_props.PADLOCKS["ml410"]
LOCK_LYING_QUAT = tuple(float(v) for v in q_axis((1, 0, 0), math.pi / 2))
"""The padlock lying level on its 38 x 44 mm face: width (local x, short leg at +x) along +x, shackle (local z) along
-y toward the cover's ears, thickness (local y) up. The arm's fingers close on its two broad faces."""
SPRING_N_PER_M = 150.0
LATCH_TRAVEL_M = 0.002

PRESSED_M = props.CONTACT_TRAVEL
"""A button counts as pressed once its plunger is in this far (its contact changes state)."""
CENTRE_TOL_M = 0.0025
SEAT_TOL_M = 0.003
TILT_TOL_DEG = 5.0
CLOSED_DEG = 3.0
COVERED_R_M = 0.015
COVERED_Z_M = 0.015
DROP_Z_M = 0.03
RELEASE_CLEAR_M = 0.06
COVER_CLEAR_M = 0.07

TEST_WAIT_S = 1.0
TEST_PHASE_S = 1.5
TEST_RAMP_S = 0.3
PRESS_N = 20.0
"""Pressing on the cover's top over the button (a firm thumb, 2 kgf)."""
PRESS_HELD_M = 0.001
PRY_NM = 0.35
"""Prying the cover open: 0.35 N*m about the hinge (about 5 N on a spring scale hooked through one ear, 7 cm from the
hinge, the other half held)."""
PRY_HELD_DEG = 10.0
LIFT_N = 10.0
"""Lifting the cover straight up (1 kgf on a spring scale)."""
LIFT_HELD_M = 0.003

STATION, COVER, PADLOCK, STAND = "station", "cover", "padlock", "stand"


def _pose(p) -> Pose:
    return Pose(position_xyz=tuple(p[:3]), rotation_xyzw=tuple(p[3:]))


def lock_start_pose() -> tuple:
    """Body centre on the pedestal (its broad face 0.5 mm above the slab's top), level with the ear holes."""
    x, y = PEDESTAL_XY
    return (x, y, props.LOCK_Z) + LOCK_LYING_QUAT


class Station(Object):
    def __init__(self):
        super().__init__(name=STATION, usd_path=str(_ASSETS / "station.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {"springs": ImplicitActuatorCfg(joint_names_expr=["press_.*"], stiffness=props.SPRING_K,
                                                        damping=props.PLUNGER_DAMPING, armature=1e-4)}
        cfg.init_state.joint_pos = {"press_.*": 0.0}
        return cfg


class Cover(Object):
    """The Master Lock S2151 cover, open on its block."""

    def __init__(self):
        super().__init__(name=COVER, usd_path=str(_ASSETS / "cover.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=_pose(props.cover_start_pose()))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {"hinge": ImplicitActuatorCfg(joint_names_expr=["hinge"], stiffness=0.0,
                                                      damping=props.HINGE_DAMPING, armature=1e-5)}
        cfg.init_state.joint_pos = {"hinge": math.radians(COVER_OPEN_DEG)}
        return cfg


class Padlock(Object):
    """The blue Master Lock 410 (sprung shackle; the task latches it)."""

    def __init__(self):
        super().__init__(name=PADLOCK, usd_path=str(_ASSETS / "padlock_ml410.usda"),
                         object_type=ObjectType.ARTICULATION, initial_pose=_pose(lock_start_pose()))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {"shackle": ImplicitActuatorCfg(joint_names_expr=["shackle_joint"], stiffness=SPRING_N_PER_M,
                                                        damping=3.0)}
        cfg.init_state.joint_pos = {"shackle_joint": lock_props.OPEN_TRAVEL}
        return cfg


class Stand(Object):
    def __init__(self):
        super().__init__(name=STAND, usd_path=str(_ASSETS / "stand.usda"), object_type=ObjectType.BASE,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))


def _write_pose(asset, pose, env_ids, origins):
    dev, n = origins.device, len(env_ids)
    p = torch.tensor(pose, device=dev, dtype=torch.float32).unsqueeze(0).repeat(n, 1)
    p[:, :3] += origins[env_ids]
    asset.write_root_pose_to_sim(p, env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)


class _TaskState:
    def __init__(self, env: ManagerBasedEnv, task: "PushButtonLockoutTask"):
        n, dev = env.num_envs, env.device
        self.station = env.scene[STATION]
        self.cover = env.scene[COVER]
        self.lock = env.scene[PADLOCK]
        self.robot = env.scene["robot"]
        self.press_start = self.station.find_joints("press_start")[0][0]
        self.press_stop = self.station.find_joints("press_stop")[0][0]
        self.hinge = self.cover.find_joints("hinge")[0][0]
        self.half_b = self.cover.find_bodies("half_b")[0][0]
        self.half_f = self.cover.find_bodies("half_f")[0][0]
        self.shackle_joint = self.lock.find_joints("shackle_joint")[0][0]
        self.lock_body = self.lock.find_bodies("body")[0][0]
        self.shackle_id = self.lock.find_bodies("shackle")[0][0]
        self.flange_ids, _ = self.robot.find_bodies(["left_gripper", "right_gripper"], preserve_order=True)
        self.scorer = PushButtonLockoutScorer(n, dev)
        self.next_layout = make_seed(task.default_seed)
        self.layouts: dict[int, Layout] = {}
        self.last_reports: dict[int, dict] = {}
        self.start_xy = torch.tensor(BUTTON_XY["start"], device=dev)
        self.stop_xy = torch.tensor(BUTTON_XY["stop"], device=dev)
        zb = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
        zl = lambda: torch.zeros(n, dtype=torch.long, device=dev)  # noqa: E731
        zf = lambda: torch.zeros(n, device=dev)  # noqa: E731
        self.latched, self.latched_through, self.released, self.force_test = zb(), zb(), zb(), zb()
        self.press_held, self.pry_held, self.lift_held, self.test_passed = zb(), zb(), zb(), zb()
        self.on_target_now, self.closed_on_now = zb(), zb()
        self.clear_steps, self.phase, self.phase_steps = zl(), zl(), zl()
        self.start_max, self.stop_max, self.hinge_deg, self.off_mm, self.seat_mm, self.tilt_deg = (
            zf(), zf(), zf(), zf(), zf(), zf())
        self.press_max, self.pry_max, self.lift_max, self.phase_z0 = zf(), zf(), zf(), zf()
        self.wait_steps = max(1, round(TEST_WAIT_S / env.step_dt))
        self.phase_len = max(1, round(TEST_PHASE_S / env.step_dt))
        self.ramp_len = max(1, round(TEST_RAMP_S / env.step_dt))


def through_ear(tip, top, ear_y, half_t):
    """The short leg (tip -> top, in a half's frame) crosses that half's ear (mid-plane y = ear_y) inside its hole."""
    denom = tip[:, 1] - top[:, 1]
    denom = torch.where(denom.abs() > 1e-6, denom, torch.full_like(denom, 1e-6))
    a = ((tip[:, 1] - ear_y) / denom).clamp(0.0, 1.0)
    cross = tip + a.unsqueeze(1) * (top - tip)
    h = props.EAR_HOLE / 2
    in_hole = (cross[:, 0].abs() < h) & ((cross[:, 2] - props.EAR_HOLE_Z).abs() < h)
    spans = (((tip[:, 1] < ear_y - half_t) & (top[:, 1] > ear_y + half_t))
             | ((tip[:, 1] > ear_y + half_t) & (top[:, 1] < ear_y - half_t)))
    return in_hole & spans


class PushButtonLockoutTask(TaskBase):
    def __init__(self, default_seed: int = 0, episode_length_s: float = 180.0):
        super().__init__(episode_length_s=episode_length_s, task_description=make_seed(default_seed).instruction)
        self.default_seed = default_seed

    def state(self, env: ManagerBasedEnv) -> _TaskState:
        if getattr(env, "_task_state", None) is None:
            env._task_state = _TaskState(env, self)
        return env._task_state

    def force_function_test(self, env: ManagerBasedEnv) -> None:
        """Control runs: start the check now, padlock or not."""
        self.state(env).force_test[:] = True

    def episode_report(self, env: ManagerBasedEnv, env_id: int = 0, ended: bool = False) -> dict:
        st = self.state(env)
        if ended and env_id in st.last_reports:
            return st.last_reports.pop(env_id)
        return self._report(st, env_id)

    def _report(self, st: _TaskState, env_id: int) -> dict:
        r = st.scorer.report(env_id)
        r.update(
            start_travel_max_mm=round(float(st.start_max[env_id]) * 1000, 2),
            stop_travel_max_mm=round(float(st.stop_max[env_id]) * 1000, 2),
            cover_off_axis_mm=round(float(st.off_mm[env_id]), 1), cover_seat_mm=round(float(st.seat_mm[env_id]), 1),
            cover_tilt_deg=round(float(st.tilt_deg[env_id]), 1), hinge_deg=round(float(st.hinge_deg[env_id]), 1),
            on_start=bool(st.on_target_now[env_id]), closed_on_start=bool(st.closed_on_now[env_id]),
            shackle_latched=bool(st.latched[env_id]), latched_through_ears=bool(st.latched_through[env_id]),
            released=bool(st.released[env_id]), check_phase=int(st.phase[env_id]),
            press_max_mm=round(float(st.press_max[env_id]) * 1000, 2), press_held=bool(st.press_held[env_id]),
            pry_max_deg=round(math.degrees(float(st.pry_max[env_id])), 1), pry_held=bool(st.pry_held[env_id]),
            lift_max_mm=round(float(st.lift_max[env_id]) * 1000, 1), lift_held=bool(st.lift_held[env_id]),
            check_passed=bool(st.test_passed[env_id]),
        )
        return r

    def get_scene_cfg(self):
        return None

    def get_events_cfg(self):
        return EventsCfg(self)

    def get_termination_cfg(self) -> TaskTerminationCfg:
        sequence = [(self._milestone_predicate(m), 0.25) for m in MILESTONES]
        return TaskTerminationCfg(timeout_s=self.episode_length_s,
                                  success=[CompletionCriteria(name="milestones", predicate_sequence=sequence)])

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
        for buf in (st.latched, st.latched_through, st.released, st.force_test, st.press_held, st.pry_held,
                    st.lift_held, st.test_passed, st.on_target_now, st.closed_on_now):
            buf[env_ids] = False
        for buf in (st.clear_steps, st.phase, st.phase_steps):
            buf[env_ids] = 0
        for buf in (st.start_max, st.stop_max, st.press_max, st.pry_max, st.lift_max):
            buf[env_ids] = 0.0
        for asset in (st.cover, st.station, st.lock):
            asset.permanent_wrench_composer.reset(env_ids=env_ids)
        # Buttons out, springs preloaded.
        j = [st.press_start, st.press_stop]
        z = torch.zeros(n, 2, device=dev)
        st.station.write_joint_state_to_sim(z, z, joint_ids=j, env_ids=env_ids)
        st.station.set_joint_position_target(z + props.SPRING_PRELOAD_M, joint_ids=j, env_ids=env_ids)
        # Cover open on its block.
        h = torch.full((n, 1), math.radians(COVER_OPEN_DEG), device=dev)
        st.cover.write_joint_state_to_sim(h, torch.zeros_like(h), joint_ids=[st.hinge], env_ids=env_ids)
        st.cover.set_joint_position_target(h, joint_ids=[st.hinge], env_ids=env_ids)
        _write_pose(st.cover, props.cover_start_pose(), env_ids, origins)
        # Padlock open on its pedestal.
        joint = [st.shackle_joint]
        limits = torch.tensor([[[0.0, lock_props.OPEN_TRAVEL]]], device=dev).expand(n, 1, 2).contiguous()
        travel = torch.full((n, 1), lock_props.OPEN_TRAVEL, device=dev)
        st.lock.write_joint_position_limit_to_sim(limits, joint_ids=joint, env_ids=env_ids, warn_limit_violation=False)
        st.lock.write_joint_state_to_sim(travel, torch.zeros_like(travel), joint_ids=joint, env_ids=env_ids)
        st.lock.set_joint_position_target(travel + 0.005, joint_ids=joint, env_ids=env_ids)
        _write_pose(st.lock, lock_start_pose(), env_ids, origins)
        for env_id in env_ids.tolist():
            st.layouts[env_id] = layout

    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:  # noqa: C901
        st = self.state(env)
        origins = env.scene.env_origins
        n, dev = env.num_envs, env.device
        live = st.phase == 0

        # Buttons.
        jp = st.station.data.joint_pos.torch
        t_start, t_stop = -jp[:, st.press_start], -jp[:, st.press_stop]
        st.start_max = torch.where(live, torch.maximum(st.start_max, t_start), st.start_max)
        st.stop_max = torch.where(live, torch.maximum(st.stop_max, t_stop), st.stop_max)

        # Cover: the back half's frame (origin on the axis at the bottom edge).
        bpos = st.cover.data.body_pos_w.torch[:, st.half_b] - origins
        bquat = st.cover.data.body_quat_w.torch[:, st.half_b]
        fpos = st.cover.data.body_pos_w.torch[:, st.half_f] - origins
        fquat = st.cover.data.body_quat_w.torch[:, st.half_f]
        axis = math_utils.quat_apply(bquat, torch.tensor([[0.0, 0.0, 1.0]], device=dev).expand(n, 3))
        tilt = torch.rad2deg(torch.acos(axis[:, 2].clamp(-1.0, 1.0)))
        off = (bpos[:, :2] - st.start_xy).norm(dim=1)
        seat = bpos[:, 2] - PANEL_Z
        st.off_mm, st.seat_mm, st.tilt_deg = off * 1000, seat * 1000, tilt
        on_target = (off < CENTRE_TOL_M) & (seat.abs() < SEAT_TOL_M) & (tilt < TILT_TOL_DEG)
        hinge = st.cover.data.joint_pos.torch[:, st.hinge]
        st.hinge_deg = torch.rad2deg(hinge)
        closed_on = on_target & (hinge < math.radians(CLOSED_DEG))
        st.on_target_now, st.closed_on_now = on_target, closed_on
        stop_covered = ((bpos[:, :2] - st.stop_xy).norm(dim=1) < COVERED_R_M) & (seat < COVERED_Z_M)

        # Padlock: the short leg through both ears (each in its own half's frame).
        travel = st.lock.data.joint_pos.torch[:, st.shackle_joint]
        lock_pos = st.lock.data.body_pos_w.torch[:, st.lock_body] - origins
        sh_pos = st.lock.data.body_pos_w.torch[:, st.shackle_id] - origins
        sh_quat = st.lock.data.body_quat_w.torch[:, st.shackle_id]

        def on_shackle(p):
            return sh_pos + math_utils.quat_apply(sh_quat, torch.tensor([p], device=dev).expand(n, 3))

        tip_w = on_shackle((PV.leg_x, 0.0, lock_props.SHORT_LEG_BOTTOM))
        top_w = on_shackle((PV.leg_x, 0.0, PV.bend_z - PV.rod / 2))
        half_t = props.EAR_T / 2
        y_in = sum(props.EAR_Y_IN) / 2
        y_out = sum(props.EAR_Y_OUT) / 2
        through_b = through_ear(math_utils.quat_apply_inverse(bquat, tip_w - bpos),
                                math_utils.quat_apply_inverse(bquat, top_w - bpos), y_in, half_t)
        through_f = through_ear(math_utils.quat_apply_inverse(fquat, tip_w - fpos),
                                math_utils.quat_apply_inverse(fquat, top_w - fpos), y_out, half_t)
        through = through_b & through_f

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
        locked = st.latched_through & through & closed_on

        # Released: both grasp points clear of the padlock and of the cover.
        flange_pos = st.robot.data.body_pos_w.torch[:, st.flange_ids]
        flange_quat = st.robot.data.body_quat_w.torch[:, st.flange_ids]
        down = torch.tensor([0.0, 0.0, -GRASP_OFFSET_M], device=dev).expand_as(flange_pos)
        grasp = flange_pos + math_utils.quat_apply(flange_quat, down) - origins[:, None]
        cover_mid = bpos + axis * (props.HEIGHT / 2)
        clear = (((grasp - lock_pos[:, None]).norm(dim=-1) > RELEASE_CLEAR_M).all(dim=1)
                 & ((grasp - cover_mid[:, None]).norm(dim=-1) > COVER_CLEAR_M).all(dim=1))
        st.released |= locked & clear

        # Function test: press the cover, pry it open, lift it.
        ready = live & ((locked & clear) | st.force_test)
        st.clear_steps = torch.where(ready, st.clear_steps + 1, torch.zeros_like(st.clear_steps))
        start = ready & ((st.clear_steps >= st.wait_steps) | st.force_test)
        if start.any():
            ids = start.nonzero().flatten()
            st.phase[ids] = 1
            st.phase_steps[ids] = 0
            st.press_max[ids] = t_start[ids]
        scale = ((st.phase_steps.float() + 1) / st.ramp_len).clamp(max=1.0)
        bodies = [st.half_b, st.half_f]
        p1, p2, p3 = st.phase == 1, st.phase == 2, st.phase == 3
        if p1.any():
            ids = p1.nonzero().flatten()
            f = torch.zeros(len(ids), 2, 3, device=dev)
            f[:, :, 2] = -(PRESS_N / 2) * scale[ids].unsqueeze(1)
            st.cover.permanent_wrench_composer.reset(env_ids=ids)
            st.cover.permanent_wrench_composer.add_forces_and_torques_index(
                forces=f, torques=torch.zeros_like(f), body_ids=bodies, env_ids=ids, is_global=True)
            st.press_max[ids] = torch.maximum(st.press_max[ids], t_start[ids])
        if p2.any():
            ids = p2.nonzero().flatten()
            t = torch.zeros(len(ids), 2, 3, device=dev)
            tq = (PRY_NM * scale[ids]).unsqueeze(1) * axis[ids]
            t[:, 0] = -tq
            t[:, 1] = tq
            st.cover.permanent_wrench_composer.reset(env_ids=ids)
            st.cover.permanent_wrench_composer.add_forces_and_torques_index(
                forces=torch.zeros_like(t), torques=t, body_ids=bodies, env_ids=ids, is_global=True)
            st.pry_max[ids] = torch.maximum(st.pry_max[ids], hinge[ids])
        if p3.any():
            ids = p3.nonzero().flatten()
            f = torch.zeros(len(ids), 2, 3, device=dev)
            f[:, :, 2] = (LIFT_N / 2) * scale[ids].unsqueeze(1)
            st.cover.permanent_wrench_composer.reset(env_ids=ids)
            st.cover.permanent_wrench_composer.add_forces_and_torques_index(
                forces=f, torques=torch.zeros_like(f), body_ids=bodies, env_ids=ids, is_global=True)
            st.lift_max[ids] = torch.maximum(st.lift_max[ids], torch.maximum(bpos[ids, 2], fpos[ids, 2])
                                             - st.phase_z0[ids])
        in_test = st.phase > 0
        st.phase_steps = torch.where(in_test, st.phase_steps + 1, st.phase_steps)
        end = in_test & (st.phase < 4) & (st.phase_steps >= st.phase_len)
        if end.any():
            ids = end.nonzero().flatten()
            st.cover.permanent_wrench_composer.reset(env_ids=ids)
            for i in ids.tolist():
                ph = int(st.phase[i])
                if ph == 1:
                    st.press_held[i] = st.press_max[i] < PRESS_HELD_M
                elif ph == 2:
                    st.pry_held[i] = st.pry_max[i] < math.radians(PRY_HELD_DEG)
                    st.phase_z0[i] = bpos[i, 2]
                elif ph == 3:
                    st.lift_held[i] = st.lift_max[i] < LIFT_HELD_M
                    st.test_passed[i] = bool(st.press_held[i] & st.pry_held[i] & st.lift_held[i]
                                             & st.latched_through[i] & through[i] & closed_on[i])
                    print(f"[pushbutton-lockout] after-release check: START plunger in {float(st.press_max[i]) * 1000:.2f}"
                          f" mm under the press (held: {bool(st.press_held[i])}), cover pried open to "
                          f"{math.degrees(float(st.pry_max[i])):.1f} deg (held: {bool(st.pry_held[i])}), lifted "
                          f"{float(st.lift_max[i]) * 1000:.1f} mm (held: {bool(st.lift_held[i])}), padlock through both "
                          f"ears: {bool(st.latched_through[i] & through[i])}, passed: {bool(st.test_passed[i])}",
                          flush=True)
                st.phase[i] = ph + 1
                st.phase_steps[i] = 0

        dropped = (bpos[:, 2] < DROP_Z_M) & live
        st.scorer.update(Step(on_target & live, closed_on & live, locked & live, st.test_passed.clone(),
                              (t_start > PRESSED_M) & live, (t_stop > PRESSED_M) & live, stop_covered & live, dropped))


@configclass
class EventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: PushButtonLockoutTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))
