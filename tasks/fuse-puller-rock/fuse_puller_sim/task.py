"""Pull the named midget fuse from its open-clip block with a tong puller: Arena props and TaskBase.

Mechanism (see fuse_puller_sim/usd/puller_props.py):
- Each clip is two fixed fingers standing 0.7 mm inside the ferrule, with a compliant-contact material: the contact
  is a spring on the penetration, which stands in for the finger's spring deflection. The squeeze and the clip
  friction hold the ferrule; no force is scripted on any fuse. A straight pull has to slide both ferrules out of
  their clips at once; levered out one end first (the far clip as the pivot) it only has to overcome one clip.
- The puller is a free articulation: a hinge body (the loop) and two arms on revolute joints, sprung open (TONG_K).
  The gripper squeezing the handle ends closes the curved jaw shells round the fuse body. Nothing attaches the fuse
  to the jaws.

Success (the function): the named fuse out of both clips while held in the puller's jaws (the circuit is open),
then lying in the tray, with neither other fuse moved from its seat.
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

from fuse_puller_sim.kit import Fingertips, reset_robot_home, reviewer_camera, write_pose
from fuse_puller_sim.scorer import MILESTONES, PullScorer, PullStep
from fuse_puller_sim.spec import TRAY_IN, TRAY_XY, Layout, make_seed
from fuse_puller_sim.usd import puller_props as props

_ASSETS = Path(os.environ.get("FPR_ASSETS", Path(__file__).resolve().parent / "assets"))
IDENTITY = (0.0, 0.0, 0.0, 1.0)

TONG_K_NM = 5.0
"""Nylon flex of each arm about its pin at the loop (N*m/rad, I). A light pinch 85 mm from the hinge that leaves the
jaws about 1.5 mm open presses the handles with about 4 N; squeezed shut on a fuse, the spring takes 0.7 N*m of the
gripper's 4.25 N*m, so the jaws clamp the fuse body with about 175 N each."""
TONG_DAMPING = 0.004

OUT_M = 0.007
"""A ferrule counts as out of its clip once its centre is this far from the clip's centre (its equator clears the
finger tops after about 4.5 mm of lift)."""
LIFT_M = 0.02
JAWS_ON_M = 0.003
JAWS_CLOSED_DEG = 3.0
DECOY_MOVED_M = 0.002
DECOY_TILT_RAD = math.radians(4.0)
TOUCH_MARGIN_M = 0.003
TRAY_Z_M = 0.03
TRAY_SPEED = 0.05
DROP_Z_M = 0.015
PULLER_DROP_Z_M = 0.012
HAZARD_TEST = os.environ.get("FPR_HAZARD_TEST", "")
"""Scorer checks only (README, "Hazard checks"); empty in every scored run. The test harness moves a fuse by writing
its pose (a stand-in for a hand): ``decoy_move_start`` slides F1 4 mm along its clips at 1 s, before any milestone;
``decoy_move_75`` slides F3 4 mm as soon as F2 is out of its clips in the jaws (milestone 75); ``out_by_hand_25``
lifts F2 25 mm out of its clips (no jaws on it) as soon as the puller is picked up (milestone 25) and puts it back
in its clips on the next step, after which the demo goes on and pulls it with the puller."""
HAZARD_SLIDE_M = 0.004
HAZARD_LIFT_M = 0.025


class Blocks(Object):
    def __init__(self):
        super().__init__(name="blocks", usd_path=str(_ASSETS / "blocks.usda"), object_type=ObjectType.BASE,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=IDENTITY))


class Fuse(Object):
    def __init__(self, slot: int):
        super().__init__(name=f"fuse_{slot}", usd_path=str(_ASSETS / "fuse.usda"), object_type=ObjectType.RIGID,
                         initial_pose=Pose(position_xyz=props.fuse_seat(slot), rotation_xyzw=props.FUSE_QUAT))


def open_angles() -> dict:
    return {"arm_a_joint": props.arm_open_sign(0) * math.radians(props.OPEN_DEG),
            "arm_b_joint": props.arm_open_sign(1) * math.radians(props.OPEN_DEG)}


class Puller(Object):
    def __init__(self):
        p = props.puller_rest_pose()
        super().__init__(name="puller", usd_path=str(_ASSETS / "puller.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=p[:3], rotation_xyzw=p[3:]))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {"flex": ImplicitActuatorCfg(joint_names_expr=["arm_.*_joint"], stiffness=TONG_K_NM,
                                                     damping=TONG_DAMPING)}
        cfg.init_state.joint_pos = open_angles()
        return cfg


class Stands(Object):
    def __init__(self):
        super().__init__(name="stands", usd_path=str(_ASSETS / "stands.usda"), object_type=ObjectType.BASE,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=IDENTITY))


class _TaskState:
    def __init__(self, env: ManagerBasedEnv, task: "FusePullerTask"):
        n, dev = env.num_envs, env.device
        self.fuses = [env.scene[f"fuse_{s}"] for s in range(3)]
        self.puller = env.scene["puller"]
        self.robot = env.scene["robot"]
        self.tips = Fingertips(self.robot)
        self.arm_joints = [self.puller.find_joints(nm)[0][0] for nm in ("arm_a_joint", "arm_b_joint")]
        self.hinge_id = self.puller.find_bodies("hinge")[0][0]
        self.scorer = PullScorer(n, dev)
        self.next_layout = make_seed(task.default_seed)
        self.layouts: dict[int, Layout] = {}
        self.last_reports: dict[int, dict] = {}
        self.named = torch.full((n,), 1, dtype=torch.long, device=dev)
        self.seats = torch.tensor([props.fuse_seat(s) for s in range(3)], device=dev)
        self.clips = torch.tensor([[props.clip_centre(s, e) for e in range(2)] for s in range(3)], device=dev)
        self.tool_start = torch.zeros(n, 3, device=dev)
        self.out_time = torch.full((n,), -1.0, device=dev)
        self.first_end = torch.full((n,), -1, dtype=torch.long, device=dev)
        """Which ferrule left its clip first (0 near the robot, 1 far; -1 not yet, 2 both in the same step)."""
        self.tilt_first = torch.zeros(n, device=dev)
        self.max_tilt = torch.zeros(n, device=dev)
        self.was_jaws_on = torch.zeros(n, dtype=torch.bool, device=dev)
        self.inject_steps = torch.zeros(n, dtype=torch.long, device=dev)


class FusePullerTask(TaskBase):
    def __init__(self, default_seed: int = 0, episode_length_s: float = 150.0, reviewer_camera: bool = False):
        super().__init__(episode_length_s=episode_length_s, task_description=make_seed(default_seed).instruction)
        self.default_seed = default_seed
        self.reviewer_camera = reviewer_camera

    def state(self, env: ManagerBasedEnv) -> _TaskState:
        if getattr(env, "_task_state", None) is None:
            env._task_state = _TaskState(env, self)
        return env._task_state

    def set_next_seed(self, env: ManagerBasedEnv, seed: int) -> None:
        self.state(env).next_layout = make_seed(seed)

    def episode_report(self, env: ManagerBasedEnv, env_id: int = 0, ended: bool = False) -> dict:
        st = self.state(env)
        if ended and env_id in st.last_reports:
            return st.last_reports.pop(env_id)
        return self._report(env, st, env_id)

    def _report(self, env, st: _TaskState, env_id: int) -> dict:
        r = st.scorer.report(env_id)
        named = int(st.named[env_id])
        pos = st.fuses[named].data.root_pos_w.torch[env_id] - env.scene.env_origins[env_id]
        first = int(st.first_end[env_id])
        r.update(named_fuse=f"F{named + 1}", named_fuse_pos=[round(float(v), 4) for v in pos],
                 named_fuse_out_s=round(float(st.out_time[env_id]), 2),
                 first_ferrule_out={-1: None, 0: "near", 1: "far", 2: "both at once"}[first],
                 tilt_when_first_ferrule_out_deg=round(math.degrees(float(st.tilt_first[env_id])), 1),
                 max_tilt_deg=round(math.degrees(float(st.max_tilt[env_id])), 1))
        return r

    def get_scene_cfg(self):
        return ReviewerCameraSceneCfg() if self.reviewer_camera else None

    def get_events_cfg(self):
        return PullerEventsCfg(self)

    def get_termination_cfg(self) -> TaskTerminationCfg:
        sequence = [(self._milestone_predicate(m), 0.25) for m in MILESTONES]
        return TaskTerminationCfg(timeout_s=self.episode_length_s,
                                  success=[CompletionCriteria(name="fuse_puller_milestones", predicate_sequence=sequence)])

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def _milestone_predicate(self, milestone: int):
        def predicate(env: ManagerBasedEnv) -> torch.Tensor:
            return self.state(env).scorer.milestone >= milestone

        predicate.__name__ = f"milestone_{milestone}"
        return predicate

    # Geometry -------------------------------------------------------------------------------------------------
    @staticmethod
    def fuse_axes(st: _TaskState, origins):
        n = origins.shape[0]
        dev = origins.device
        fpos = torch.stack([f.data.root_pos_w.torch - origins for f in st.fuses], dim=1)  # (N, 3, 3)
        fquat = torch.stack([f.data.root_quat_w.torch for f in st.fuses], dim=1)
        z = torch.tensor([[0.0, 0.0, 1.0]], device=dev).expand(n * 3, 3)
        axis = math_utils.quat_apply(fquat.reshape(-1, 4), z).reshape(n, 3, 3)
        return fpos, axis

    @staticmethod
    def jaw_centre(st: _TaskState, origins):
        n = origins.shape[0]
        h_pos = st.puller.data.body_pos_w.torch[:, st.hinge_id] - origins
        h_quat = st.puller.data.body_quat_w.torch[:, st.hinge_id]
        off = torch.tensor([[props.JAW_X, 0.0, 0.0]], device=origins.device).expand(n, 3)
        return h_pos + math_utils.quat_apply(h_quat, off), h_pos

    # Events ------------------------------------------------------------------------------------------------------
    def reset(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        reset_robot_home(env, env_ids)
        for env_id in env_ids.tolist():
            if int(st.scorer.step[env_id]) > 0:
                st.last_reports[env_id] = self._report(env, st, env_id)
        layout = st.next_layout
        dev, n = env.device, len(env_ids)
        origins = env.scene.env_origins
        st.scorer.reset(env_ids)
        st.named[env_ids] = layout.named
        st.out_time[env_ids] = -1.0
        st.first_end[env_ids] = -1
        st.tilt_first[env_ids] = 0.0
        st.max_tilt[env_ids] = 0.0
        st.was_jaws_on[env_ids] = False
        st.inject_steps[env_ids] = 0
        for s, fuse in enumerate(st.fuses):
            fuse.permanent_wrench_composer.reset(env_ids=env_ids)
            write_pose(fuse, props.fuse_seat(s) + props.FUSE_QUAT, env_ids, origins[env_ids])
        oa = open_angles()
        open_q = torch.tensor([[oa["arm_a_joint"], oa["arm_b_joint"]]], device=dev).repeat(n, 1)
        st.puller.write_joint_state_to_sim(open_q, torch.zeros_like(open_q), joint_ids=st.arm_joints, env_ids=env_ids)
        st.puller.set_joint_position_target(open_q, joint_ids=st.arm_joints, env_ids=env_ids)
        write_pose(st.puller, props.puller_rest_pose(), env_ids, origins[env_ids])
        st.tool_start[env_ids] = torch.tensor(props.puller_rest_pose()[:3], device=dev)
        for env_id in env_ids.tolist():
            st.layouts[env_id] = layout

    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        origins = env.scene.env_origins
        n, dev = env.num_envs, env.device
        ar = torch.arange(n, device=dev)

        fpos, axis = self.fuse_axes(st, origins)
        named_pos, named_axis = fpos[ar, st.named], axis[ar, st.named]

        # Ferrules out of their clips (ferrule k=0 is the one at -z in the fuse frame: near the robot when seated).
        cap_off = props.FUSE_L / 2 - props.CAP_L / 2
        caps = named_pos[:, None] + torch.stack([-cap_off * named_axis, cap_off * named_axis], dim=1)  # (N, 2, 3)
        clips = st.clips[st.named]  # (N, 2, 3)
        cap_out = torch.cdist(caps, clips).min(dim=2).values > OUT_M  # (N, 2)
        out = cap_out.all(dim=1)
        tilt = torch.acos(named_axis[:, 0].abs().clamp(-1.0, 1.0))
        st.max_tilt = torch.maximum(st.max_tilt, tilt)
        any_out = cap_out.any(dim=1) & (st.first_end < 0)
        if any_out.any():  # the fuse's -z ferrule sits in the near clip
            left = torch.where(cap_out[:, 0] & cap_out[:, 1], torch.full_like(st.first_end, 2),
                               torch.where(cap_out[:, 0], torch.zeros_like(st.first_end),
                                           torch.ones_like(st.first_end)))
            st.first_end = torch.where(any_out, left, st.first_end)
            st.tilt_first = torch.where(any_out, tilt, st.tilt_first)
        newly = out & (st.out_time < 0)
        st.out_time = torch.where(newly, (st.scorer.step.float() + 1) * env.step_dt, st.out_time)

        # Puller: jaw centre on the named fuse's body, jaws closed.
        jaw, h_pos = self.jaw_centre(st, origins)
        q_arm = st.puller.data.joint_pos.torch[:, st.arm_joints]
        opening = props.arm_open_sign(0) * q_arm[:, 0] + props.arm_open_sign(1) * q_arm[:, 1]
        closed = opening < math.radians(JAWS_CLOSED_DEG)
        rel = jaw - named_pos
        along = (rel * named_axis).sum(-1)
        radial = (rel - along[:, None] * named_axis).norm(dim=-1)
        jaws_on = closed & (radial < JAWS_ON_M) & (along.abs() < props.FUSE_L / 2 - props.CAP_L)
        st.was_jaws_on |= jaws_on
        lifted = (h_pos - st.tool_start).norm(dim=-1) > LIFT_M
        held_out = out & jaws_on
        out_by_hand = out & ~jaws_on & (st.scorer.progress < 75)

        # Tray.
        tx, ty = TRAY_XY
        a, b, _ = TRAY_IN
        speed = torch.zeros(n, device=dev)
        for s, f in enumerate(st.fuses):
            speed = torch.where(st.named == s, f.data.root_lin_vel_w.torch.norm(dim=-1), speed)
        in_xy = ((named_pos[:, 0] - tx).abs() < a / 2) & ((named_pos[:, 1] - ty).abs() < b / 2)
        in_tray = out & in_xy & (named_pos[:, 2] < TRAY_Z_M) & (speed < TRAY_SPEED)
        fuse_dropped = (named_pos[:, 2] < DROP_Z_M) & ~in_xy

        # Decoys: moved or tilted; touched by a finger pad or the jaws.
        decoy = torch.ones(n, 3, dtype=torch.bool, device=dev)
        decoy[ar, st.named] = False
        moved = (fpos - st.seats[None]).norm(dim=-1) > DECOY_MOVED_M
        tilted = torch.acos(axis[..., 0].clamp(-1.0, 1.0)) > DECOY_TILT_RAD
        decoy_moved = ((moved | tilted) & decoy).any(dim=1)
        pts = torch.cat([st.tips.points(origins), jaw[:, None]], dim=1)  # (N, 5, 3)
        rel_p = pts[:, :, None] - fpos[:, None]  # (N, 5, 3, 3)
        al = (rel_p * axis[:, None]).sum(-1)
        rad = (rel_p - al[..., None] * axis[:, None]).norm(dim=-1)
        near_p = (rad < props.FUSE_R + TOUCH_MARGIN_M) & (al.abs() < props.FUSE_L / 2 + TOUCH_MARGIN_M)
        decoy_touched = (near_p & decoy[:, None]).any(dim=2).any(dim=1)
        puller_dropped = h_pos[:, 2] < PULLER_DROP_Z_M

        st.scorer.update(PullStep(lifted, jaws_on, held_out, in_tray, decoy_moved, decoy_touched, out_by_hand,
                                  fuse_dropped, puller_dropped))
        if HAZARD_TEST:
            self._inject_hazard(env, st)

    @staticmethod
    def _inject_hazard(env: ManagerBasedEnv, st: _TaskState) -> None:
        """Scorer checks only (HAZARD_TEST); env 0. Runs after the scorer's update, so a pose written here is scored
        on the next step."""
        k = int(st.inject_steps[0])
        named = int(st.named[0])
        ids = torch.tensor([0], device=env.device)
        origins = env.scene.env_origins[ids]
        if HAZARD_TEST in ("decoy_move_start", "decoy_move_75"):
            due = int(st.scorer.step[0]) >= 20 if HAZARD_TEST == "decoy_move_start" else int(st.scorer.progress[0]) >= 75
            if k == 0 and due:
                slot = 0 if HAZARD_TEST == "decoy_move_start" else 2
                x, y, z = props.fuse_seat(slot)
                write_pose(st.fuses[slot], (x - HAZARD_SLIDE_M, y, z) + props.FUSE_QUAT, ids, origins)
                st.inject_steps[0] = 1
        elif HAZARD_TEST == "out_by_hand_25":
            if k == 0 and int(st.scorer.progress[0]) >= 25:
                x, y, z = props.fuse_seat(named)
                write_pose(st.fuses[named], (x, y, z + HAZARD_LIFT_M) + props.FUSE_QUAT, ids, origins)
                st.inject_steps[0] = 1
            elif k == 1:
                write_pose(st.fuses[named], props.fuse_seat(named) + props.FUSE_QUAT, ids, origins)
                st.inject_steps[0] = 2


REVIEWER_EYE = (0.12, 0.18, 0.25)
REVIEWER_TARGET = (0.30, -0.02, 0.03)


@configclass
class ReviewerCameraSceneCfg:
    reviewer_cam: CameraCfg = reviewer_camera(REVIEWER_EYE, REVIEWER_TARGET)


@configclass
class PullerEventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: FusePullerTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))
