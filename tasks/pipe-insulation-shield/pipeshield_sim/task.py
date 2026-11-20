"""Pipe insulation shield task: Arena props and TaskBase (props: pipeshield_sim/usd/props.py, layout: spec.py).

Mechanism: geometry, contact and gravity only, no rules. The pipe (1.24 m of insulated 1/2 in copper, full of water,
11 N) is a free rigid body resting in two fixed clevis hangers: A (no shield; the insulation lies on the strap) and
B (its shield already in). Its centre of mass lies between them, near A, so A carries 10.2 N. The loose shield is a
free rigid body (a half-round of 18 ga sheet) balanced on a wooden post. Nothing is attached, welded, hidden or
moved by the task: the robot has to lift the pipe off A's strap, get the shield in under it and lower it.

Release check (the success check), once both milestones 33 and 67 hold: every finger pad must stay RELEASE_CLEAR from
the pipe and the shield for HOLD_S without a break, with the shield seated on the strap and the pipe resting on the
shield the whole time (the count restarts whenever a pad comes closer or a condition fails).
"""

from __future__ import annotations

import math
import os
from dataclasses import MISSING
from pathlib import Path

import torch

import isaaclab.utils.math as math_utils
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

from pipeshield_sim import spec as S
from pipeshield_sim.kit import Fingertips, reset_robot_home
from pipeshield_sim.scorer import MILESTONES, Scorer, Step

_ASSETS = Path(__file__).resolve().parent / "assets"
IDENTITY = (0.0, 0.0, 0.0, 1.0)
HAZARD_TEST = os.environ.get("PIS_HAZARD_TEST", "")
DEBUG_EVERY = int(os.environ.get("PIS_DEBUG", "0") or 0)


class Rig(Object):
    def __init__(self):
        super().__init__(name="rig", usd_path=str(_ASSETS / "rig.usda"), object_type=ObjectType.BASE,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=IDENTITY))


class Pipe(Object):
    def __init__(self):
        super().__init__(name="pipe", usd_path=str(_ASSETS / "pipe.usda"), object_type=ObjectType.RIGID,
                         initial_pose=Pose(position_xyz=S.pipe_start(), rotation_xyzw=IDENTITY))


class Shield(Object):
    def __init__(self):
        super().__init__(name="shield", usd_path=str(_ASSETS / "shield.usda"), object_type=ObjectType.RIGID,
                         initial_pose=Pose(position_xyz=S.shield_start(), rotation_xyzw=IDENTITY))


def all_props() -> list:
    return [Rig(), Pipe(), Shield()]


def _axis(quat, v):
    vv = torch.tensor(v, device=quat.device, dtype=quat.dtype).expand(quat.shape[0], 3)
    return math_utils.quat_apply(quat, vv)


def _seg_dist(pts, a, d, half):
    """Distance from points (N, P, 3) to segments centred at a (N, 3) along unit d (N, 3), half-length half."""
    rel = pts - a[:, None]
    t = (rel * d[:, None]).sum(-1).clamp(-half, half)
    return (rel - t[..., None] * d[:, None]).norm(dim=-1)


class _TaskState:
    def __init__(self, env: ManagerBasedEnv, task: "PipeShieldTask"):
        n, dev = env.num_envs, env.device
        self.robot = env.scene["robot"]
        self.pipe = env.scene["pipe"]
        self.shield = env.scene["shield"]
        self.tips = Fingertips(self.robot)
        self.scorer = Scorer(n, dev)
        self.last_reports: dict[int, dict] = {}
        self.hold_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.hold_len = max(1, round(S.HOLD_S / env.step_dt))
        zf = lambda: torch.zeros(n, device=dev)  # noqa: E731
        self.peak_lift, self.peak_shift = zf(), zf()
        self.m: dict[str, torch.Tensor] = {}
        self.hazard_done = False
        self.restore = None


class PipeShieldTask(TaskBase):
    def __init__(self, episode_length_s: float = S.EPISODE_SECONDS):
        super().__init__(episode_length_s=episode_length_s, task_description=S.instruction())

    def state(self, env: ManagerBasedEnv) -> _TaskState:
        if getattr(env, "_task_state", None) is None:
            env._task_state = _TaskState(env, self)
        return env._task_state

    def episode_report(self, env: ManagerBasedEnv, env_id: int = 0, ended: bool = False) -> dict:
        st = self.state(env)
        if ended and env_id in st.last_reports:
            return st.last_reports.pop(env_id)
        return self._report(st, env_id)

    def _report(self, st: _TaskState, env_id: int) -> dict:
        r = st.scorer.report(env_id)
        for k, v in st.m.items():
            val = float(v[env_id])
            r[k] = round(val * 1000, 2) if k.endswith("_mm") else round(val, 3)
        r.update(hold_steps=int(st.hold_steps[env_id]), peak_lift_mm=round(float(st.peak_lift[env_id]) * 1000, 2),
                 peak_shift_mm=round(float(st.peak_shift[env_id]) * 1000, 2))
        return r

    def get_scene_cfg(self):
        return None

    def get_events_cfg(self):
        return EventsCfg(self)

    def get_termination_cfg(self) -> TaskTerminationCfg:
        sequence = [(self._milestone_predicate(m), 1.0 / len(MILESTONES)) for m in MILESTONES]
        return TaskTerminationCfg(timeout_s=self.episode_length_s,
                                  success=[CompletionCriteria(name="milestones", predicate_sequence=sequence)])

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def _milestone_predicate(self, milestone: int):
        def predicate(env: ManagerBasedEnv) -> torch.Tensor:
            sc = self.state(env).scorer
            return (sc.milestone >= milestone) & ~sc.hazard

        predicate.__name__ = f"milestone_{milestone}"
        return predicate

    # Events ------------------------------------------------------------------------------------------------------
    def reset(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        reset_robot_home(env, env_ids)
        for env_id in env_ids.tolist():
            if int(st.scorer.step[env_id]) > 0:
                st.last_reports[env_id] = self._report(st, env_id)
        dev, n = env.device, len(env_ids)
        origins = env.scene.env_origins
        st.scorer.reset(env_ids)
        st.hold_steps[env_ids] = 0
        st.peak_lift[env_ids] = 0.0
        st.peak_shift[env_ids] = 0.0
        st.hazard_done = False
        st.restore = None
        st.lift_release_step = None
        st.pipe.permanent_wrench_composer.reset(env_ids=env_ids)
        for asset, p in ((st.pipe, S.pipe_start()), (st.shield, S.shield_start())):
            pose = torch.tensor([[*p, *IDENTITY]], device=dev).repeat(n, 1)
            pose[:, :3] += origins[env_ids]
            asset.write_root_pose_to_sim(pose, env_ids=env_ids)
            asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)

    def _hazard_hooks(self, env, st: _TaskState) -> None:
        """Test harness (PIS_HAZARD_TEST=<case>_<when>, empty in scored runs): move a part as a hand would, once;
        the next step puts it back where it was. when = start (1 s in) or a milestone (33, 67)."""
        if getattr(st, "lift_release_step", None) is not None and int(st.scorer.step[0]) >= st.lift_release_step:
            st.pipe.permanent_wrench_composer.reset(env_ids=torch.arange(env.num_envs, device=env.device))
            st.lift_release_step = None
        if not HAZARD_TEST or st.hazard_done:
            return
        case, _, when = HAZARD_TEST.partition("_")
        step = int(st.scorer.step[0])
        m = int(st.scorer.milestone[0])
        due = (when == "start" and step == round(1.0 / env.step_dt)) or (when.isdigit() and m >= int(when))
        if not due:
            return
        st.hazard_done = True
        ids = torch.arange(env.num_envs, device=env.device)
        if case == "lift":
            # A hand pulling the pipe up for 1 s at hanger A (twice its weight upward), then letting go.
            forces = torch.zeros(env.num_envs, 1, 3, device=env.device)
            forces[:, 0, 2] = 2.0 * S.PIPE_MASS * 9.81
            pos = (env.scene.env_origins + torch.tensor([S.PIPE_X, S.HANGER_A_Y, S.PIPE_Z_BARE], device=env.device))
            st.pipe.permanent_wrench_composer.set_forces_and_torques_index(
                forces=forces, torques=torch.zeros_like(forces), positions=pos[:, None], body_ids=[0], env_ids=ids,
                is_global=True)
            st.lift_release_step = step + round(1.0 / env.step_dt)
        else:
            asset = st.pipe if case == "shift" else st.shield
            pose = torch.cat([asset.data.root_pos_w.torch, asset.data.root_quat_w.torch], dim=-1).clone()
            moved = pose.clone()
            if case == "shift":
                moved[:, 1] += 0.020          # the pipe pulled 20 mm along its run
            elif case == "drop":
                moved[:, 2] = env.scene.env_origins[:, 2] + 0.10   # the shield 10 cm above the bench, falling
            asset.write_root_pose_to_sim(moved, env_ids=ids)
            st.restore = (asset, pose)
        print(f"[pipe-shield] HAZARD TEST {HAZARD_TEST} applied at step {step} (milestone {m})", flush=True)

    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        origins = env.scene.env_origins
        n, dev = env.num_envs, env.device
        restore, st.restore = st.restore, None

        p_pos = st.pipe.data.root_pos_w.torch - origins
        p_q = st.pipe.data.root_quat_w.torch
        p_ax = _axis(p_q, (0.0, 1.0, 0.0))
        s_pos = st.shield.data.root_pos_w.torch - origins
        s_q = st.shield.data.root_quat_w.torch
        s_up = _axis(s_q, (0.0, 0.0, 1.0))
        s_ax = _axis(s_q, (0.0, 1.0, 0.0))

        # The pipe at hanger A: its axis where it crosses the strap's centre line.
        t_a = (S.HANGER_A_Y - p_pos[:, 1]) / p_ax[:, 1].clamp(min=0.5)
        at_a = p_pos + t_a[:, None] * p_ax
        lift = at_a[:, 2] - S.PIPE_Z_BARE
        shift = torch.maximum((p_pos[:, 1] - S.PIPE_CY).abs(), (at_a[:, 0] - S.PIPE_X).abs())
        st.peak_lift = torch.maximum(st.peak_lift, lift)
        st.peak_shift = torch.maximum(st.peak_shift, shift)

        # The shield: open side up, under the pipe, across the strap, between the strap and the pipe.
        tilt = torch.rad2deg(torch.acos(s_up[:, 2].clamp(-1.0, 1.0)))
        spine = s_pos[:, 2] - S.SH_RO * s_up[:, 2]
        t_s = (s_pos[:, 1] - p_pos[:, 1]) / p_ax[:, 1].clamp(min=0.5)
        pipe_at_s = p_pos + t_s[:, None] * p_ax
        side_err = s_pos[:, 0] - pipe_at_s[:, 0]
        along = s_pos[:, 1] - S.HANGER_A_Y
        under = (tilt <= S.ARC_DOWN_DEG) & (side_err.abs() <= S.UNDER_X_TOL) & \
            (along.abs() <= S.SH_L / 2 - S.STRAP_W / 2) & (spine >= S.Z_CB - 0.002) & (spine <= S.Z_CB + 0.025) & \
            (s_ax[:, 1].abs() >= math.cos(math.radians(10.0)))
        centred = along.abs() <= S.CENTRE_TOL
        seated = (spine - S.Z_CB).abs() <= S.SEAT_TOL
        resting = (at_a[:, 2] - S.PIPE_Z_ON_SHIELD).abs() <= S.SEAT_TOL

        # Finger pads: clear of the pipe (its insulation) and of the shield.
        pads = st.tips.points(origins)
        L = S.PIPE_Y[1] - S.PIPE_Y[0]
        d_pipe = (_seg_dist(pads, p_pos, p_ax, L / 2) - S.INS_R).min(dim=1).values
        d_shield = (_seg_dist(pads, s_pos, s_ax, S.SH_L / 2) - S.SH_RO).min(dim=1).values
        clear = (d_pipe > S.RELEASE_CLEAR) & (d_shield > S.RELEASE_CLEAR)
        ok = under & centred & seated & resting & clear
        st.hold_steps = torch.where(ok, st.hold_steps + 1, torch.zeros_like(st.hold_steps))
        released_ok = st.hold_steps >= st.hold_len

        dropped = spine < S.DROP_Z
        st.m = {"lift_at_a_mm": lift, "shift_mm": shift, "shield_tilt_deg": tilt,
                "shield_roll_deg": torch.rad2deg(torch.atan2(s_up[:, 0], s_up[:, 2])),
                "shield_pitch_deg": torch.rad2deg(torch.atan2(-s_up[:, 1], s_up[:, 2])), "shield_spine_mm": spine,
                "shield_side_err_mm": side_err, "shield_from_hanger_mm": along, "pipe_at_a_z_mm": at_a[:, 2],
                "pad_to_pipe_mm": d_pipe, "pad_to_shield_mm": d_shield,
                "shield_x_mm": s_pos[:, 0], "shield_y_mm": s_pos[:, 1], "pipe_y_mm": p_pos[:, 1]}
        st.scorer.update(Step(under, centred, released_ok, lift > S.LIFT_MAX, shift > S.SHIFT_MAX, dropped))
        if restore is not None:
            asset, pose = restore
            asset.write_root_pose_to_sim(pose, env_ids=torch.arange(n, device=dev))
        self._hazard_hooks(env, st)
        k = int(st.scorer.step[0])
        flag = (int(st.scorer.milestone[0]) >= 100, bool(st.scorer.hazard[0]))
        if flag != getattr(st, "_printed", (False, False)) and any(flag):
            st._printed = flag
            print(f"[pipe-shield] REPORT k={k} {self._report(st, 0)}", flush=True)
        if DEBUG_EVERY and k % DEBUG_EVERY == 0:
            print(f"[pipe-shield] k={k} {self._report(st, 0)}", flush=True)


@configclass
class EventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: PipeShieldTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))
