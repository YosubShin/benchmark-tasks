"""Capacitor ground stick: Arena props and TaskBase. Clip the stick's ground lead to the earth bar, lay the probe across
the named capacitor's terminals after the WAIT lamp, leave the stick hanging on them.

Mechanism (groundstick_sim/usd/stick_props.py):
- The stick, its 18-link lead and the Mueller BU-21C clip are one floating articulation. The lead is a
  real tether: the stick can only go as far from the clip as the cable reaches.
- The clip's jaw is a revolute joint with a spring (joint drive toward slightly past closed, so it is
  preloaded against its stop). Squeezing the handles opens it; released over the bar, the spring closes the
  pads on the bar and friction holds it. "Clipped to ground" is read from geometry: a point between the pads
  lies inside the copper bar and the jaw is held open by it.
- The probe is a rod with its end turned down. Laid across both studs it touches both; left alone, the stick
  pivots on the near stud, its butt drops to the desk, it slides down-slope until the turned-down end
  catches the far stud. Contact with a terminal is a distance test between the probe's rod and the stud/nut.
- The WAIT lamp is a timer (rule), as on the real bench (a battery lamp behind an on-delay timer's break contact). "Discharged" and "earthed" are rules on top of these contacts: nothing
  electrical is simulated.

Function test (success check), once the stick has hung on both terminals with both grippers clear for
TEST_WAIT_S: (1) tug the clip up with TUG_N for TEST_PHASE_S (someone catches the lead); (2) pull the stick
along +y (away from the capacitor) with PULL_N for TEST_PHASE_S. Passed if the clip is still on the bar
after (1) and (2) and the probe is still on both terminals after (2).
"""

from __future__ import annotations

import json
import os
import math
from dataclasses import MISSING
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
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

from groundstick_sim.kit import GRASP_OFFSET_M, Fingertips, reset_robot_home, set_visible
from groundstick_sim.scorer import MILESTONES, StickScorer, StickStep
from groundstick_sim.spec import DWELL_SECONDS, EARTH_BAR_Y, EPISODE_SECONDS, WAIT_SECONDS, Layout, make_seed
from groundstick_sim.usd import stick_props as props

_ASSETS = Path(__file__).resolve().parent / "assets"
BENCH, STICK = "bench", "stick"

JAW_SPRING = 1.0
"""N*m/rad. With the target JAW_PRELOAD_RAD past closed: about 0.5 N*m (15 N at the handles) to open the jaws
22 mm, and 0.31 N*m (10 N at the pad edge) on the 6.35 mm bar. Inferred; to measure on a BU-21C."""
JAW_PRELOAD_RAD = -0.10
LEAD_STIFFNESS = 0.002
LEAD_DAMPING = 0.004

CONTACT_TOL_M = 0.002
FINGER_NEAR_M = 0.015
CLEAR_M = 0.08
BAR_MARGIN_M = 0.0015
DROP_Z_M = 0.05

_DEBUG = bool(os.environ.get("GROUNDSTICK_DEBUG"))
TEST_WAIT_S = 2.0
TEST_PHASE_S = 1.5
TUG_N = 8.0
"""Upward pull on the clip (a lead caught by a foot or a sleeve; inferred, about 10 times the clip's weight)."""
PULL_N = 2.0
"""Pull on the stick away from the capacitor along the hanging direction (inferred)."""


def _load_init() -> dict:
    return json.loads((_ASSETS / "init_joints.json").read_text())


class Bench(Object):
    def __init__(self):
        super().__init__(name=BENCH, usd_path=str(_ASSETS / "bench.usda"), object_type=ObjectType.BASE,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))


class DischargeStick(Object):
    """Stick + lead + clip, one floating articulation."""

    def __init__(self):
        p = props.stick_rest_pose()
        super().__init__(name=STICK, usd_path=str(_ASSETS / "stick.usda"),
                         object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=p[:3], rotation_xyzw=p[3:]))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {
            "jaw": ImplicitActuatorCfg(joint_names_expr=["jaw_joint"], stiffness=JAW_SPRING, damping=0.01),
            "lead": ImplicitActuatorCfg(joint_names_expr=["lead_j.*"], stiffness=LEAD_STIFFNESS, damping=LEAD_DAMPING),
        }
        cfg.init_state.joint_pos = _load_init()
        return cfg


class _TaskState:
    def __init__(self, env: ManagerBasedEnv, task: "StickTask"):
        n, dev = env.num_envs, env.device
        self.art = env.scene[STICK]
        self.robot = env.scene["robot"]
        self.tips = Fingertips(self.robot)
        self.stick_id = self.art.find_bodies("stick")[0][0]
        self.clip_id = self.art.find_bodies("clip")[0][0]
        self.jaw_joint = self.art.find_joints("jaw_joint")[0][0]
        self.flange_ids, _ = self.robot.find_bodies(["left_gripper", "right_gripper"], preserve_order=True)
        init = _load_init()
        self.init_q = torch.tensor([[init[name] for name in self.art.joint_names]], device=dev)
        self.scorer = StickScorer(n, max(1, round(DWELL_SECONDS / env.step_dt)), dev)
        self.next_layout = make_seed(task.default_seed)
        self.layouts: dict[int, Layout] = {}
        self.last_reports: dict[int, dict] = {}
        self.bench_paths = sim_utils.find_matching_prim_paths(f"/World/envs/env_.*/{BENCH}")
        self.lamp_shown = torch.ones(n, dtype=torch.bool, device=dev)
        # Probe sample points (stick frame) and terminal axis points with their radii.
        pts = [(props.HEAD_X[1] - 0.004 + i * (props.PROBE_END_X - props.HEAD_X[1] + 0.004) / 23, 0.0, 0.0)
               for i in range(24)]
        pts += [(props.PROBE_END_X, 0.0, -props.CURL_DROP * i / 9) for i in range(1, 10)]
        self.probe_pts = torch.tensor(pts, device=dev)
        zs = [(props.NUT_BOTTOM + props.NUT_H / 2, props.NUT_R)]
        zs += [(props.NUT_BOTTOM + props.NUT_H + props.STUD_H * f, props.STUD_R) for f in (0.2, 0.6, 1.0)]
        term = []
        for cap in range(2):
            for k in range(2):
                x, y, _ = props.terminal_xyz(cap, k)
                term.append([(x, y, z) for z, _ in zs])
        self.term_pts = torch.tensor(term, device=dev).reshape(2, 2, len(zs), 3)
        self.term_r = torch.tensor([r for _, r in zs], device=dev)
        self.studs_top = torch.tensor([[props.terminal_xyz(c, k) for k in range(2)] for c in range(2)], device=dev)
        self.bar_lo = torch.tensor([props.BAR_X[0], EARTH_BAR_Y - props.BAR_T / 2, props.BAR_Z[0]], device=dev) \
            - BAR_MARGIN_M
        self.bar_hi = torch.tensor([props.BAR_X[1], EARTH_BAR_Y + props.BAR_T / 2, props.BAR_Z[1]], device=dev) \
            + BAR_MARGIN_M
        zb = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
        zl = lambda: torch.zeros(n, dtype=torch.long, device=dev)  # noqa: E731
        self.force_test, self.tug_held, self.pull_held, self.test_passed = zb(), zb(), zb(), zb()
        self.hang_steps, self.phase, self.phase_steps = zl(), zl(), zl()
        self.wait_steps = max(1, round(TEST_WAIT_S / env.step_dt))
        self.phase_len = max(1, round(TEST_PHASE_S / env.step_dt))
        self.last = {}


class StickTask(TaskBase):
    def __init__(self, default_seed: int = 0, episode_length_s: float = EPISODE_SECONDS):
        super().__init__(episode_length_s=episode_length_s, task_description=make_seed(default_seed).instruction)
        self.default_seed = default_seed

    def state(self, env: ManagerBasedEnv) -> _TaskState:
        if getattr(env, "_task_state", None) is None:
            env._task_state = _TaskState(env, self)
        return env._task_state

    def set_next_seed(self, env: ManagerBasedEnv, seed: int) -> None:
        self.state(env).next_layout = make_seed(seed)

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
            wait_lamp_on=bool(st.lamp_shown[env_id]), function_test_phase=int(st.phase[env_id]),
            test_clip_held_tug=bool(st.tug_held[env_id]), test_stick_held_pull=bool(st.pull_held[env_id]),
            function_test_passed=bool(st.test_passed[env_id]),
            **{k: (v[env_id].tolist() if v.dim() > 1 else (bool(v[env_id]) if v.dtype == torch.bool
                                                            else round(float(v[env_id]), 4)))
               for k, v in st.last.items()},
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
        st.scorer.reset(env_ids)
        st.scorer.set_named(env_ids, layout.named)
        for buf in (st.force_test, st.tug_held, st.pull_held, st.test_passed):
            buf[env_ids] = False
        for buf in (st.hang_steps, st.phase, st.phase_steps):
            buf[env_ids] = 0
        st.art.permanent_wrench_composer.reset(env_ids=env_ids)
        pose = torch.tensor(props.stick_rest_pose(), device=dev).unsqueeze(0).repeat(n, 1)
        pose[:, :3] += env.scene.env_origins[env_ids]
        st.art.write_root_pose_to_sim(pose, env_ids=env_ids)
        st.art.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)
        q = st.init_q.expand(n, -1).clone()
        st.art.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
        target = q.clone()
        target[:, st.jaw_joint] = JAW_PRELOAD_RAD
        st.art.set_joint_position_target(target, env_ids=env_ids)
        stage = sim_utils.get_current_stage()
        for env_id in env_ids.tolist():
            st.layouts[env_id] = layout
            set_visible(stage, f"{st.bench_paths[env_id]}/lamp_on", True)
            set_visible(stage, f"{st.bench_paths[env_id]}/lamp_off", False)
        st.lamp_shown[env_ids] = True

    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        origins = env.scene.env_origins
        n, dev = env.num_envs, env.device
        lamp_on = st.scorer.step.float() * env.step_dt < WAIT_SECONDS
        self._show_lamp(st, lamp_on)

        body_pos = st.art.data.body_pos_w.torch - origins[:, None]
        body_quat = st.art.data.body_quat_w.torch
        s_pos, s_quat = body_pos[:, st.stick_id], body_quat[:, st.stick_id]
        c_pos, c_quat = body_pos[:, st.clip_id], body_quat[:, st.clip_id]
        q_jaw = st.art.data.joint_pos.torch[:, st.jaw_joint]

        # Clip on the earth bar: the point between the pads inside the bar, the jaws closed on it (not squeezed open).
        bite_local = torch.stack([torch.full_like(q_jaw, props.BITE[0]), props.BITE[0] * torch.sin(q_jaw) / 2,
                                  torch.zeros_like(q_jaw)], dim=1)
        bite = c_pos + math_utils.quat_apply(c_quat, bite_local)
        in_bar = ((bite > st.bar_lo) & (bite < st.bar_hi)).all(dim=1)
        clamp = props.clamp_angle(props.BAR_T)
        grounded = in_bar & (q_jaw > clamp - math.radians(3.0)) & (q_jaw < clamp + math.radians(4.0))

        # Probe against each terminal's stud and nut.
        m = st.probe_pts.shape[0]
        probe = s_pos[:, None] + math_utils.quat_apply(s_quat[:, None].expand(n, m, 4).reshape(-1, 4),
                                                       st.probe_pts.repeat(n, 1)).reshape(n, m, 3)
        d = (probe[:, :, None, None, None, :] - st.term_pts[None, None]).norm(dim=-1)  # (N, m, 2, 2, k)
        limit = props.ROD_R + st.term_r + CONTACT_TOL_M
        probe_on = (d < limit).any(dim=-1).any(dim=1)  # (N, 2, 2)

        # Finger pads near a named terminal.
        rows = torch.arange(n, device=dev)
        pads = st.tips.points(origins)
        named_tops = st.studs_top[st.scorer.named]  # (N, 2, 3)
        finger_on = ((pads[:, :, None] - named_tops[:, None]).norm(dim=-1) < FINGER_NEAR_M).any(dim=(1, 2))

        flange_pos = st.robot.data.body_pos_w.torch[:, st.flange_ids]
        flange_quat = st.robot.data.body_quat_w.torch[:, st.flange_ids]
        down = torch.tensor([0.0, 0.0, -GRASP_OFFSET_M], device=dev).expand_as(flange_pos)
        grasp = flange_pos + math_utils.quat_apply(flange_quat, down) - origins[:, None]
        clear = ((grasp - s_pos[:, None]).norm(dim=-1) > CLEAR_M).all(dim=1)
        clip_clear = ((grasp - c_pos[:, None]).norm(dim=-1) > CLEAR_M).all(dim=1)
        both_named = probe_on[rows, st.scorer.named].all(dim=1)
        hanging = both_named & clear & clip_clear

        # Function test.
        ready = (hanging & (st.phase == 0)) | (st.force_test & (st.phase == 0))
        st.hang_steps = torch.where(ready, st.hang_steps + 1, torch.zeros_like(st.hang_steps))
        start = ready & ((st.hang_steps >= st.wait_steps) | st.force_test)
        if start.any():
            ids = start.nonzero().flatten()
            f = torch.zeros(len(ids), 1, 3, device=dev)
            f[..., 2] = TUG_N
            st.art.permanent_wrench_composer.add_forces_and_torques_index(
                forces=f, torques=torch.zeros_like(f), body_ids=[st.clip_id], env_ids=ids, is_global=True)
            st.phase[ids] = 1
            st.phase_steps[ids] = 0
        st.phase_steps = torch.where(st.phase > 0, st.phase_steps + 1, st.phase_steps)
        end_tug = (st.phase == 1) & (st.phase_steps >= st.phase_len)
        if end_tug.any():
            ids = end_tug.nonzero().flatten()
            st.art.permanent_wrench_composer.reset(env_ids=ids)
            st.tug_held[ids] = grounded[ids]
            f = torch.zeros(len(ids), 1, 3, device=dev)
            f[..., 1] = PULL_N
            st.art.permanent_wrench_composer.add_forces_and_torques_index(
                forces=f, torques=torch.zeros_like(f), body_ids=[st.stick_id], env_ids=ids, is_global=True)
            st.phase[ids] = 2
            st.phase_steps[ids] = 0
        end_pull = (st.phase == 2) & (st.phase_steps >= st.phase_len)
        if end_pull.any():
            ids = end_pull.nonzero().flatten()
            st.art.permanent_wrench_composer.reset(env_ids=ids)
            st.pull_held[ids] = both_named[ids]
            st.test_passed[ids] = st.tug_held[ids] & st.pull_held[ids] & grounded[ids]
            st.phase[ids] = 3

        if _DEBUG and (bool(hanging[0]) or int(st.phase[0]) > 0):
            ax = math_utils.quat_apply(s_quat[:1], torch.tensor([[1.0, 0.0, 0.0]], device=dev))[0]
            up = math_utils.quat_apply(s_quat[:1], torch.tensor([[0.0, 0.0, 1.0]], device=dev))[0]
            print(f"DBG phase {int(st.phase[0])} k {int(st.phase_steps[0])} stick {[round(v, 4) for v in s_pos[0].tolist()]} "
                  f"axis {[round(v, 3) for v in ax.tolist()]} up {[round(v, 3) for v in up.tolist()]} "
                  f"on {probe_on[0, st.scorer.named[0]].tolist()} grounded {bool(grounded[0])}", flush=True)
        dropped = ~probe_on.flatten(1).any(dim=1) & (probe[:, -1, 2] < DROP_Z_M) & (s_pos[:, 2] < 0.04)
        st.last = {"grounded": grounded, "jaw_deg": torch.rad2deg(q_jaw), "probe_on_named": probe_on[rows, st.scorer.named],
                   "probe_on_decoy": probe_on[rows, 1 - st.scorer.named]}
        st.scorer.update(StickStep(lamp_on, grounded, probe_on, finger_on, hanging & (st.phase == 0),
                                  st.test_passed.clone(), dropped))

    def _show_lamp(self, st: _TaskState, lamp_on: torch.Tensor) -> None:
        changed = lamp_on != st.lamp_shown
        if not changed.any():
            return
        stage = sim_utils.get_current_stage()
        for env_id in changed.nonzero().flatten().tolist():
            on = bool(lamp_on[env_id])
            set_visible(stage, f"{st.bench_paths[env_id]}/lamp_on", on)
            set_visible(stage, f"{st.bench_paths[env_id]}/lamp_off", not on)
        st.lamp_shown = lamp_on.clone()


@configclass
class EventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: StickTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))
