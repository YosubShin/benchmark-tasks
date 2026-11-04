"""Swap the fuse in the named flip-open DIN fuse holder: Arena props and TaskBase.

Mechanism (see fuse_sim/usd/fuse_props.py):
- Each CHM1DU carrier is a rigid body on a revolute joint at its lower end (opening is negative about +y, the
  top swings toward the robot; the open stop is the joint limit at 72 deg). An unlatched carrier swings open by
  gravity (its centre of mass is in front of the hinge), as the real one does.
- Latch (rule): inside LATCH_DEG of shut, the joint drive pulls the carrier shut with a preload (LATCH_K and
  LATCH_PRELOAD_RAD; the moulded snap and the contacts' grip on the ferrules, TM). Within OPEN_DETENT_DEG of the
  open stop a weaker spring (DETENT_K) holds it open (an open detent, without it, pushing the new fuse in turned the
  carrier shut). In between the drive's stiffness is zero, so it adds no torque.
- The fuse is a free rigid body. In the carrier its inner ferrule sits in a U clip open toward the body on a bottom
  stop and its outer ferrule on a saddle; with the carrier shut the body's upper contact block stands behind the
  outer ferrule. Opening tilts it out with the carrier; it is lifted out of the clip's open side (and laid back in)
  across its axis. Nothing holds it but that geometry and gravity.

Function test (the success check): once the named carrier is shut (< CLOSED_DEG) with the matching spare seated and both
grasp points have been more than RELEASE_CLEAR_M from its grip tab for TEST_WAIT_S (and only after milestone 75), the scorer applies
TEST_NM of opening torque to the carrier for TEST_PHASE_S. Passed if it stays below HELD_DEG and, at the end,
the seated fuse is the spare of the matching rating and the old fuse lies in the tray.
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

from fuse_sim.kit import GRASP_OFFSET_M, Fingertips, reset_robot_home, reviewer_camera, write_pose
from fuse_sim.scorer import MILESTONES, SwapScorer, SwapStep
from fuse_sim.spec import SPARE_RATINGS, TRAY_XY, Layout, make_seed
from fuse_sim.usd import fuse_props as props

_ASSETS = Path(__file__).resolve().parent / "assets"
IDENTITY = (0.0, 0.0, 0.0, 1.0)
FUSE_ROLES = ("old", "spare_a", "spare_b")

LATCH_DEG = props.LATCH_DEG
LATCH_K = 0.60
"""N*m/rad. With the preload the shut carrier is held by 0.14 N*m (5 N pulling straight out at the grip tab, 28 mm
from the hinge), rising to 0.20 N*m (7.3 N) at the edge of the latch zone (not published; to be measured)."""
LATCH_PRELOAD_RAD = 0.233
DETENT_K = LATCH_K
DETENT_PRELOAD_RAD = LATCH_PRELOAD_RAD
"""Open detent: the same spring, 0.14 N*m at the open stop (a weaker detent, 0.042 N*m, let the swinging carrier
unseat the fuse at the stop)."""
NO_LATCH = os.environ.get("FUSE_CONTROL") == "1"
"""Control runs: the named carrier's latch spring is off (a broken snap); nothing holds it shut."""
HOLD_CHECK = os.environ.get("FUSE_HOLD_CHECK") == "1"
"""Mechanism check without the robot: start with the swap already done (milestone 75) so the hold test runs as soon
as the arms, idle at their ready pose, have been clear for 1 s. With and without FUSE_CONTROL it isolates the latch."""
CARRIER_DAMPING = 0.001

OPENED_DEG = 30.0
DECOY_OPENED_DEG = 5.0
DECOY_TOUCH_DEG = 1.0
CLOSED_DEG = 1.0
SEAT_M = 0.002
SEAT_AXIS_COS = math.cos(math.radians(6.0))
OUT_M = 0.025
TOUCH_MARGIN_M = 0.002
TRAY_Z_M = 0.03
DROP_Z_M = 0.06
RELEASE_CLEAR_M = 0.06
TEST_WAIT_S = 1.0
TEST_PHASE_S = 1.5
TEST_NM = 0.11
"""Opening torque in the test: 4 N pulling straight out at the grip tab, 28 mm from the hinge (0.8 of the latch's
hold)."""
HELD_DEG = 3.0


def fuse_usd(role: str) -> str:
    if role == "old":
        return str(_ASSETS / "fuse_6_blown.usda")
    return str(_ASSETS / f"fuse_{SPARE_RATINGS[FUSE_ROLES.index(role) - 1]}.usda")


def fuse_start_pose(role: str) -> tuple:
    if role == "old":
        return props.fuse_seat_world(1) + IDENTITY
    return props.spare_pose(FUSE_ROLES.index(role) - 1)


class Holders(Object):
    def __init__(self):
        super().__init__(name="holders", usd_path=str(_ASSETS / "holders.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=IDENTITY))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {"latch": ImplicitActuatorCfg(joint_names_expr=["carrier.*_joint"], stiffness=LATCH_K,
                                                      damping=CARRIER_DAMPING)}
        cfg.init_state.joint_pos = {"carrier.*_joint": 0.0}
        return cfg


class Fuse(Object):
    def __init__(self, role: str):
        p = fuse_start_pose(role)
        super().__init__(name=f"fuse_{role}", usd_path=fuse_usd(role), object_type=ObjectType.RIGID,
                         initial_pose=Pose(position_xyz=p[:3], rotation_xyzw=p[3:]))


class Stands(Object):
    def __init__(self):
        super().__init__(name="stands", usd_path=str(_ASSETS / "stands.usda"), object_type=ObjectType.BASE,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=IDENTITY))


class _TaskState:
    def __init__(self, env: ManagerBasedEnv, task: "FuseHolderTask"):
        n, dev = env.num_envs, env.device
        self.holders = env.scene["holders"]
        self.fuses = [env.scene[f"fuse_{r}"] for r in FUSE_ROLES]
        self.robot = env.scene["robot"]
        self.tips = Fingertips(self.robot)
        self.joint_ids = [self.holders.find_joints(nm)[0][0] for nm in props.joint_names()]
        self.carrier_ids = [self.holders.find_bodies(nm)[0][0] for nm in props.carrier_names()]
        self.flange_ids, _ = self.robot.find_bodies(["left_gripper", "right_gripper"], preserve_order=True)
        self.scorer = SwapScorer(n, dev)
        self.next_layout = make_seed(task.default_seed)
        self.layouts: dict[int, Layout] = {}
        self.last_reports: dict[int, dict] = {}
        self.named = torch.full((n,), 1, dtype=torch.long, device=dev)
        self.match = torch.full((n,), 2, dtype=torch.long, device=dev)
        """Index into FUSE_ROLES of the spare with the matching rating."""
        self.seat = torch.tensor(props.seat_in_carrier(), device=dev)
        zb = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
        zl = lambda: torch.zeros(n, dtype=torch.long, device=dev)  # noqa: E731
        self.force_test, self.test_passed, self.test_held, self.was_open = zb(), zb(), zb(), zb()
        self.clear_steps, self.phase, self.phase_steps = zl(), zl(), zl()
        self.test_max = torch.zeros(n, device=dev)
        self.max_open = torch.zeros(n, 3, device=dev)
        self.seated_role = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.zone = torch.ones(n, 3, dtype=torch.long, device=dev)
        self.wait_steps = max(1, round(TEST_WAIT_S / env.step_dt))
        self.phase_len = max(1, round(TEST_PHASE_S / env.step_dt))


class FuseHolderTask(TaskBase):
    def __init__(self, default_seed: int = 0, episode_length_s: float = 180.0, reviewer_camera: bool = False):
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
        """Control runs: start the function test now, shut or not."""
        self.state(env).force_test[:] = True

    def episode_report(self, env: ManagerBasedEnv, env_id: int = 0, ended: bool = False) -> dict:
        st = self.state(env)
        if ended and env_id in st.last_reports:
            return st.last_reports.pop(env_id)
        return self._report(st, env_id)

    def _report(self, st: _TaskState, env_id: int) -> dict:
        r = st.scorer.report(env_id)
        q = self.open_deg(st)[env_id].tolist()
        role = int(st.seated_role[env_id])
        r.update(named_holder=f"F{int(st.named[env_id]) + 1}",
                 carrier_open_deg=[round(v, 1) for v in q],
                 carrier_max_open_deg=[round(math.degrees(v), 1) for v in st.max_open[env_id].tolist()],
                 fuse_seated_in_named=FUSE_ROLES[role] if role >= 0 else None,
                 matching_spare=FUSE_ROLES[int(st.match[env_id])],
                 function_test_phase=int(st.phase[env_id]),
                 test_max_open_deg=round(math.degrees(float(st.test_max[env_id])), 2),
                 test_held=bool(st.test_held[env_id]), function_test_passed=bool(st.test_passed[env_id]))
        return r

    def open_deg(self, st: _TaskState) -> torch.Tensor:
        return torch.rad2deg(-st.holders.data.joint_pos.torch[:, st.joint_ids])

    def get_scene_cfg(self):
        return ReviewerCameraSceneCfg() if self.reviewer_camera else None

    def get_events_cfg(self):
        return SwapEventsCfg(self)

    def get_termination_cfg(self) -> TaskTerminationCfg:
        sequence = [(self._milestone_predicate(m), 0.25) for m in MILESTONES]
        return TaskTerminationCfg(timeout_s=self.episode_length_s,
                                  success=[CompletionCriteria(name="fuse_swap_milestones", predicate_sequence=sequence)])

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
        st.named[env_ids] = layout.named
        st.match[env_ids] = 1 + layout.matching_spare
        for buf in (st.force_test, st.test_passed, st.test_held, st.was_open):
            buf[env_ids] = False
        for buf in (st.clear_steps, st.phase, st.phase_steps):
            buf[env_ids] = 0
        st.test_max[env_ids] = 0.0
        st.max_open[env_ids] = 0.0
        st.seated_role[env_ids] = -1
        st.holders.permanent_wrench_composer.reset(env_ids=env_ids)
        zero = torch.zeros(n, 3, device=dev)
        st.holders.write_joint_state_to_sim(zero, zero, joint_ids=st.joint_ids, env_ids=env_ids)
        st.holders.set_joint_position_target(zero + LATCH_PRELOAD_RAD, joint_ids=st.joint_ids, env_ids=env_ids)
        stiff0 = torch.full((n, 3), LATCH_K, device=dev)
        if NO_LATCH:
            stiff0[:, layout.named] = 0.0
        st.holders.write_joint_stiffness_to_sim_index(stiffness=stiff0,
                                                      joint_ids=st.joint_ids, env_ids=env_ids)
        st.zone[env_ids] = 1
        for role, fuse in zip(FUSE_ROLES, st.fuses):
            write_pose(fuse, fuse_start_pose(role), env_ids, origins[env_ids])
        if HOLD_CHECK:  # the swap already done: the matching spare seated in the shut carrier, the old fuse in the tray
            s = math.sqrt(0.5)
            write_pose(st.fuses[0], (TRAY_XY[0], TRAY_XY[1], 0.012, s, 0.0, 0.0, s), env_ids, origins[env_ids])
            write_pose(st.fuses[1 + layout.matching_spare], props.fuse_seat_world(layout.named) + IDENTITY, env_ids,
                       origins[env_ids])
            st.scorer.milestone[env_ids] = 75
        for env_id in env_ids.tolist():
            st.layouts[env_id] = layout

    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        origins = env.scene.env_origins
        n, dev = env.num_envs, env.device
        ar = torch.arange(n, device=dev)

        # Latch rule: pull shut inside the latch zone; elsewhere the drive follows the joint (no torque).
        jp = st.holders.data.joint_pos.torch[:, st.joint_ids]
        q_open = -jp
        latch = q_open < math.radians(LATCH_DEG)
        detent = q_open > math.radians(props.OPEN_DEG - props.OPEN_DETENT_DEG)
        zone = latch.long() - detent.long()  # 1 latch, -1 open detent, 0 free
        if (zone != st.zone).any():  # a target that trails the joint by a step acts as damping, so in the free zone
            # the spring is switched off rather than retargeted (it crept open at 3 deg/s)
            stiff = torch.where(zone > 0, LATCH_K, torch.where(zone < 0, DETENT_K, 0.0))
            if NO_LATCH:
                stiff[ar, st.named] = torch.where(zone[ar, st.named] > 0, 0.0, stiff[ar, st.named])
            st.holders.write_joint_stiffness_to_sim_index(stiffness=stiff.float(), joint_ids=st.joint_ids)
            st.zone = zone.clone()
        open_target = -(math.radians(props.OPEN_DEG) + DETENT_PRELOAD_RAD)
        st.holders.set_joint_position_target(torch.where(detent, torch.full_like(jp, open_target),
                                                         torch.full_like(jp, LATCH_PRELOAD_RAD)),
                                             joint_ids=st.joint_ids)
        st.max_open = torch.maximum(st.max_open, q_open)

        # Named carrier frame; every fuse in it.
        c_pos = st.holders.data.body_pos_w.torch[:, st.carrier_ids] - origins[:, None]  # (N, 3, 3)
        c_quat = st.holders.data.body_quat_w.torch[:, st.carrier_ids]  # (N, 3, 4)
        nc_pos, nc_quat = c_pos[ar, st.named], c_quat[ar, st.named]
        f_pos = torch.stack([f.data.root_pos_w.torch - origins for f in st.fuses], dim=1)  # (N, 3, 3)
        f_quat = torch.stack([f.data.root_quat_w.torch for f in st.fuses], dim=1)
        z_axis = torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n * 3, 3)
        f_axis = math_utils.quat_apply(f_quat.reshape(-1, 4), z_axis).reshape(n, 3, 3)
        q_rep = nc_quat[:, None].expand(n, 3, 4).reshape(-1, 4)
        in_c = math_utils.quat_apply_inverse(q_rep, (f_pos - nc_pos[:, None]).reshape(-1, 3)).reshape(n, 3, 3)
        ax_c = math_utils.quat_apply_inverse(q_rep, f_axis.reshape(-1, 3)).reshape(n, 3, 3)
        off = (in_c - st.seat).norm(dim=-1)  # (N, 3)
        seated = (off < SEAT_M) & (ax_c[..., 2].abs() > SEAT_AXIS_COS)
        st.seated_role = torch.where(seated.any(dim=1), seated.float().argmax(dim=1), torch.full_like(st.seated_role, -1))
        right_seated = seated[ar, st.match]
        wrong_role = 3 - st.match  # the other spare
        wrong_installed = seated[ar, wrong_role]
        old_out = off[:, 0] > OUT_M

        q_named = q_open[ar, st.named]
        opened = q_named > math.radians(OPENED_DEG)
        decoy = torch.ones(n, 3, dtype=torch.bool, device=dev)
        decoy[ar, st.named] = False
        decoy_opened = ((q_open > math.radians(DECOY_OPENED_DEG)) & decoy).any(dim=1)

        # Finger pads near a decoy carrier's grip tab (in that carrier's frame), or a decoy moved at all.
        pads = st.tips.points(origins)  # (N, 4, 3)
        tab_lo = torch.tensor([props.TAB_D[0] - props.HINGE_D, -props.TAB_HALF_Y, props.TAB_Z[0] - props.HINGE_Z],
                              device=dev) - TOUCH_MARGIN_M
        tab_hi = torch.tensor([props.TAB_D[1] - props.HINGE_D, props.TAB_HALF_Y, props.TAB_Z[1] - props.HINGE_Z],
                              device=dev) + TOUCH_MARGIN_M
        touched = torch.zeros(n, dtype=torch.bool, device=dev)
        for s in range(3):
            rel = math_utils.quat_apply_inverse(c_quat[:, s, None].expand(n, 4, 4).reshape(-1, 4),
                                                (pads - c_pos[:, s, None]).reshape(-1, 3)).reshape(n, 4, 3)
            inside = ((rel > tab_lo) & (rel < tab_hi)).all(dim=-1).any(dim=1)
            touched |= decoy[:, s] & (inside | (q_open[:, s] > math.radians(DECOY_TOUCH_DEG)))

        # Tray and drops.
        tx, ty = TRAY_XY
        in_xy = ((f_pos[..., 0] - tx).abs() < 0.045) & ((f_pos[..., 1] - ty).abs() < 0.045)
        old_in_tray = in_xy[:, 0] & (f_pos[:, 0, 2] < TRAY_Z_M)
        fuse_dropped = ((f_pos[..., 2] < DROP_Z_M) & ~in_xy).any(dim=1)

        # Function test.
        flange_pos = st.robot.data.body_pos_w.torch[:, st.flange_ids]
        flange_quat = st.robot.data.body_quat_w.torch[:, st.flange_ids]
        down = torch.tensor([0.0, 0.0, -GRASP_OFFSET_M], device=dev).expand_as(flange_pos)
        grasp = flange_pos + math_utils.quat_apply(flange_quat, down) - origins[:, None]
        tab_c = torch.tensor([(props.TAB_D[0] + props.TAB_D[1]) / 2 - props.HINGE_D, 0.0,
                              (props.TAB_Z[0] + props.TAB_Z[1]) / 2 - props.HINGE_Z], device=dev).expand(n, 3)
        tab_w = nc_pos + math_utils.quat_apply(nc_quat, tab_c)
        clear = ((grasp - tab_w[:, None]).norm(dim=-1) > RELEASE_CLEAR_M).all(dim=1)
        shut = q_named < math.radians(CLOSED_DEG)
        armed = shut & right_seated & (st.scorer.milestone >= 75)  # only once the swap is done
        ready = (armed & clear & (st.phase == 0)) | (st.force_test & (st.phase == 0))
        st.clear_steps = torch.where(ready, st.clear_steps + 1, torch.zeros_like(st.clear_steps))
        start = ready & ((st.clear_steps >= st.wait_steps) | st.force_test)
        if start.any():
            ids = start.nonzero().flatten()
            torque = torch.zeros(len(ids), 1, 3, device=dev)
            torque[..., 1] = -TEST_NM  # opening is negative about +y
            body = [st.carrier_ids[int(st.named[ids[0]])]]
            st.holders.permanent_wrench_composer.add_forces_and_torques_index(
                forces=torch.zeros_like(torque), torques=torque, body_ids=body, env_ids=ids, is_global=True)
            st.phase[ids] = 1
            st.phase_steps[ids] = 0
        st.phase_steps = torch.where(st.phase == 1, st.phase_steps + 1, st.phase_steps)
        st.test_max = torch.where(st.phase == 1, torch.maximum(st.test_max, q_named), st.test_max)
        end = (st.phase == 1) & (st.phase_steps >= st.phase_len)
        if end.any():
            ids = end.nonzero().flatten()
            st.holders.permanent_wrench_composer.reset(env_ids=ids)
            st.test_held[ids] = st.test_max[ids] < math.radians(HELD_DEG)
            st.test_passed[ids] = st.test_held[ids] & right_seated[ids] & old_in_tray[ids] & shut[ids]
            st.phase[ids] = 2

        st.scorer.update(SwapStep(opened, old_out, right_seated, st.test_passed.clone(), decoy_opened,
                                  wrong_installed, touched, fuse_dropped))


REVIEWER_EYE = (0.10, 0.17, 0.30)
REVIEWER_TARGET = (0.27, -0.03, 0.14)


@configclass
class ReviewerCameraSceneCfg:
    reviewer_cam: CameraCfg = reviewer_camera(REVIEWER_EYE, REVIEWER_TARGET)


@configclass
class SwapEventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: FuseHolderTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))
