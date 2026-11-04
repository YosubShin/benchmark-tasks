"""Ground clamp order: Arena props and TaskBase. Hang the ground clamp on the ground bar and tighten it, then the line
clamp on the bar tagged TESTED.

Mechanism (groundclamp_sim/usd/clamp_props.py):
- The grounding set is one floating articulation: ground clamp, 18-link cable, line clamp. Each clamp is a frame
  (C body with the hook jaw under the bar), a pad carrier on a prismatic joint along the screw axis, and the eye
  screw with its stick on a revolute joint about that axis.
- The eye screw is a screw-coupled joint (rule-driven drives, updated every control step):
  * the pad's prismatic drive (stiff, CLAMP_K) targets p = P0 + PITCH * turn / (2 pi): turning the screw clockwise
    (seen from the stick) runs the pad toward the jaw. Free, the pad follows. Once the bar is between pad and jaw,
    the bar's geometry stops the pad and the drive presses it: that is the clamp force F = CLAMP_K * (p - target).
  * thread friction on the revolute is a Coulomb drive: a stiff spring whose anchor slips once it
    carries T(F) = RUN_NM + NUT_K * D * F (the bolt formula T = K F d). So the more the clamp is tightened, the
    harder the screw turns, and a pull on the clamp never turns the screw back (self-locking).
- Holding on the bar is friction between the serrated jaw, the pad and the copper bar under F (geometry +
  friction). A clamp hung on its pad but not tightened is held only by its weight.

Function test (success check), once both clamps are attached with both grippers clear for TEST_WAIT_S: tug the
line clamp toward the robot (-x, off the bar's edge) with TUG_N for TEST_PHASE_S, then the ground clamp. Passed if
each moved less than HOLD_M and is still on its bar.
"""

from __future__ import annotations

import json
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

from groundclamp_sim.kit import GRASP_OFFSET_M, reset_robot_home
from groundclamp_sim.scorer import MILESTONES, ClampScorer, ClampStep
from groundclamp_sim.spec import EPISODE_SECONDS, Layout, make_seed
from groundclamp_sim.usd import clamp_props as props

_ASSETS = Path(__file__).resolve().parent / "assets"
BENCH, SET = "bench", "ground_set"

CLAMP_K = 2.0e5
"""N/m: clamp compliance (frame spring, pad, bar): one 90 deg stroke (0.79 mm) past contact gives 160 N (inferred)."""
CLAMP_D = 400.0
CLAMP_MAX_N = 1500.0
TURN_K = 30.0
TURN_D = 0.02
TURN_ARMATURE = 1e-4
RUN_NM = 0.015
"""Running torque of the Acme thread and swivel pad with no load (inferred; to measure). At 0.2 N*m the hung (not yet
tight) clamp turned with the screw and was levered off the bar's edge in the first run."""
NUT_K = 0.2
SCREW_D = 0.0159
"""T = NUT_K * F * d with d = 5/8 in (the bolt formula; inferred, the screw size is not published)."""
CABLE_K = 0.01
CABLE_D = 0.01

ATTACH_N = 120.0
"""Clamp force at which a clamp counts as attached (tight): 0.58 N*m at the eye (the real 250 in-lb = 28 N*m is past
what the YAM's pinch on the stick transmits; see the report)."""
TOUCH_M = 0.002
BITE_MARGIN_M = 0.001
CLEAR_M = 0.05

TEST_WAIT_S = 1.0
TEST_PHASE_S = 2.5
"""Per clamp: 1 s ramp, then 1.5 s at full tug."""
TUG_N = 40.0
"""Pull on each clamp toward the robot, off the bar's edge (a cable yanked; inferred)."""
HOLD_M = 0.003
RUNAWAY_M = 0.005
TUG_RAMP_S = 1.0

# Clamp-frame sample points: the jaw's top face, the spine's inner face; pad-frame points: the pad's bottom.
_JAW_PTS = [(x, y, 0.0) for x in (-0.019, -0.005, 0.012) for y in (-0.010, 0.010)]
_JAW_PTS += [(props.SPINE_X[1], 0.0, z) for z in (0.004, 0.02)]
_PAD_PTS = [(x, y, 0.0) for x in (-0.009, 0.009) for y in (-0.009, 0.009)]


def _load_init() -> dict:
    return json.loads((_ASSETS / "init_joints.json").read_text())


class Bench(Object):
    def __init__(self):
        super().__init__(name=BENCH, usd_path=str(_ASSETS / "bench.usda"), object_type=ObjectType.BASE,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))


class GroundSet(Object):
    """Ground clamp + cable + line clamp, one floating articulation rooted at the ground clamp's frame."""

    def __init__(self):
        p = props.park_pose(0)
        super().__init__(name=SET, usd_path=str(_ASSETS / "set.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=p[:3], rotation_xyzw=p[3:]))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {
            "slide": ImplicitActuatorCfg(joint_names_expr=["[gl]_slide"], stiffness=CLAMP_K, damping=CLAMP_D,
                                         effort_limit_sim=CLAMP_MAX_N),
            "turn": ImplicitActuatorCfg(joint_names_expr=["[gl]_turn"], stiffness=TURN_K, damping=TURN_D,
                                        effort_limit_sim=RUN_NM, armature=TURN_ARMATURE),
            "cable": ImplicitActuatorCfg(joint_names_expr=["cable_j.*"], stiffness=CABLE_K, damping=CABLE_D),
        }
        cfg.init_state.joint_pos = _load_init()
        return cfg


class _TaskState:
    def __init__(self, env: ManagerBasedEnv, task: "ClampTask"):
        n, dev = env.num_envs, env.device
        self.art = env.scene[SET]
        self.robot = env.scene["robot"]
        self.frame_ids = [self.art.find_bodies(f"{c}_frame")[0][0] for c in "gl"]
        self.pad_ids = [self.art.find_bodies(f"{c}_pad")[0][0] for c in "gl"]
        self.screw_ids = [self.art.find_bodies(f"{c}_screw")[0][0] for c in "gl"]
        self.slide_j = [self.art.find_joints(f"{c}_slide")[0][0] for c in "gl"]
        self.turn_j = [self.art.find_joints(f"{c}_turn")[0][0] for c in "gl"]
        self.flange_ids, _ = self.robot.find_bodies(["left_gripper", "right_gripper"], preserve_order=True)
        init = _load_init()
        self.init_q = torch.tensor([[init[name] for name in self.art.joint_names]], device=dev)
        self.scorer = ClampScorer(n, max(1, round(20.0 / env.step_dt)), dev)
        self.next_layout = make_seed(task.default_seed)
        self.layouts: dict[int, Layout] = {}
        self.last_reports: dict[int, dict] = {}
        boxes = [props.bar_box(i) for i in range(len(props.bars()))]
        self.bar_lo = torch.tensor([b[0] for b in boxes], device=dev)  # (B, 3)
        self.bar_hi = torch.tensor([b[1] for b in boxes], device=dev)
        self.jaw_pts = torch.tensor(_JAW_PTS, device=dev)
        self.pad_pts = torch.tensor(_PAD_PTS, device=dev)
        zf = lambda: torch.zeros(n, 2, device=dev)  # noqa: E731
        self.turn_acc, self.turn_prev, self.turn_hold, self.force = zf(), zf(), zf(), zf()
        self.peak_force = zf()
        zb = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
        self.force_test, self.test_passed = zb(), zb()
        self.held = torch.zeros(n, 2, dtype=torch.bool, device=dev)
        zl = lambda: torch.zeros(n, dtype=torch.long, device=dev)  # noqa: E731
        self.ready_steps, self.phase, self.phase_steps = zl(), zl(), zl()
        self.tug_start = torch.zeros(n, 3, device=dev)
        self.wait_steps = max(1, round(TEST_WAIT_S / env.step_dt))
        self.phase_len = max(1, round(TEST_PHASE_S / env.step_dt))
        self.ramp_steps = max(1, round(TUG_RAMP_S / env.step_dt))
        self.tug_on = False
        self.last = {}


class ClampTask(TaskBase):
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
            function_test_phase=int(st.phase[env_id]), test_line_held=bool(st.held[env_id, 1]),
            test_ground_held=bool(st.held[env_id, 0]), function_test_passed=bool(st.test_passed[env_id]),
            clamp_force_n=[round(float(v), 1) for v in st.force[env_id]],
            peak_clamp_force_n=[round(float(v), 1) for v in st.peak_force[env_id]],
            screw_turns=[round(float(v) / (2 * math.pi), 3) for v in st.turn_acc[env_id]],
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
        st.scorer.set_tagged(env_ids, layout.tagged)
        for buf in (st.force_test, st.test_passed, st.held):
            buf[env_ids] = False
        for buf in (st.ready_steps, st.phase, st.phase_steps):
            buf[env_ids] = 0
        for buf in (st.turn_acc, st.turn_prev, st.turn_hold, st.force, st.peak_force):
            buf[env_ids] = 0.0
        st.art.permanent_wrench_composer.reset(env_ids=env_ids)
        st.tug_on = False
        pose = torch.tensor(props.park_pose(0), device=dev).unsqueeze(0).repeat(n, 1)
        pose[:, :3] += env.scene.env_origins[env_ids]
        st.art.write_root_pose_to_sim(pose, env_ids=env_ids)
        st.art.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)
        q = st.init_q.expand(n, -1).clone()
        st.art.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
        st.art.set_joint_position_target(q.clone(), env_ids=env_ids)
        st.art.write_joint_effort_limit_to_sim_index(limits=torch.full((n, 2), RUN_NM, device=dev),
                                                     joint_ids=st.turn_j, env_ids=env_ids)
        for env_id in env_ids.tolist():
            st.layouts[env_id] = layout

    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        origins = env.scene.env_origins
        n, dev = env.num_envs, env.device
        jp = st.art.data.joint_pos.torch

        # The eye screws: screw coupling and thread friction, per clamp (0 = ground, 1 = line).
        for c in range(2):
            q_raw = jp[:, st.turn_j[c]]
            d = torch.remainder(q_raw - st.turn_prev[:, c] + math.pi, 2 * math.pi) - math.pi
            st.turn_prev[:, c] = q_raw
            st.turn_acc[:, c] += d
            turn = st.turn_acc[:, c]
            p = jp[:, st.slide_j[c]]
            target = (props.P0 + props.PITCH * turn / (2 * math.pi)).clamp(props.P_LIMITS[0], props.P_LIMITS[1])
            force = (CLAMP_K * (p - target)).clamp(min=0.0, max=CLAMP_MAX_N)
            st.force[:, c] = force
            st.peak_force[:, c] = torch.maximum(st.peak_force[:, c], force)
            limit = RUN_NM + NUT_K * SCREW_D * force
            st.turn_hold[:, c] = torch.minimum(torch.maximum(st.turn_hold[:, c], turn - limit / TURN_K),
                                               turn + limit / TURN_K)
            st.art.write_joint_effort_limit_to_sim_index(limits=limit.unsqueeze(1), joint_ids=[st.turn_j[c]])
            st.art.set_joint_position_target((q_raw + st.turn_hold[:, c] - turn).unsqueeze(1), joint_ids=[st.turn_j[c]])
            st.art.set_joint_position_target(target.unsqueeze(1), joint_ids=[st.slide_j[c]])

        body_pos = st.art.data.body_pos_w.torch - origins[:, None]
        body_quat = st.art.data.body_quat_w.torch
        nb = st.bar_lo.shape[0]
        touch = torch.zeros(n, 2, nb, dtype=torch.bool, device=dev)
        on_bar = torch.zeros(n, 2, nb, dtype=torch.bool, device=dev)
        frame_pos = []
        for c in range(2):
            f_pos, f_quat = body_pos[:, st.frame_ids[c]], body_quat[:, st.frame_ids[c]]
            p_pos, p_quat = body_pos[:, st.pad_ids[c]], body_quat[:, st.pad_ids[c]]
            frame_pos.append(f_pos)
            pts = [f_pos[:, None] + math_utils.quat_apply(f_quat[:, None].expand(-1, len(_JAW_PTS), 4),
                                                         st.jaw_pts[None].expand(n, -1, 3)),
                   p_pos[:, None] + math_utils.quat_apply(p_quat[:, None].expand(-1, len(_PAD_PTS), 4),
                                                         st.pad_pts[None].expand(n, -1, 3))]
            pts = torch.cat(pts, dim=1)  # (N, k, 3)
            inside = ((pts[:, :, None] > st.bar_lo[None, None] - TOUCH_M)
                      & (pts[:, :, None] < st.bar_hi[None, None] + TOUCH_M)).all(dim=-1)  # (N, k, B)
            touch[:, c] = inside.any(dim=1)
            bite_local = torch.zeros(n, 3, device=dev)
            bite_local[:, 2] = jp[:, st.slide_j[c]] / 2
            bite = f_pos + math_utils.quat_apply(f_quat, bite_local)
            on_bar[:, c] = ((bite[:, None] > st.bar_lo[None] - BITE_MARGIN_M)
                            & (bite[:, None] < st.bar_hi[None] + BITE_MARGIN_M)).all(dim=-1)
        tight = st.force >= ATTACH_N
        attached = on_bar & tight[:, :, None]  # (N, 2, B)
        tagged = st.scorer.tagged
        rows = torch.arange(n, device=dev)
        tag_bar = (tagged + 1).clamp(min=1)  # bar index of the tagged phase bar (bars: GND, A, B, C, parks)
        ground_attached = attached[:, 0, 0]
        line_on_tagged = on_bar[rows, 1, tag_bar] & (tagged >= 0)
        line_attached = attached[rows, 1, tag_bar] & (tagged >= 0)

        flange_pos = st.robot.data.body_pos_w.torch[:, st.flange_ids]
        flange_quat = st.robot.data.body_quat_w.torch[:, st.flange_ids]
        down = torch.tensor([0.0, 0.0, -GRASP_OFFSET_M], device=dev).expand_as(flange_pos)
        grasp = flange_pos + math_utils.quat_apply(flange_quat, down) - origins[:, None]  # (N, 2, 3)
        handles = []
        for c in range(2):
            s_pos, s_quat = body_pos[:, st.screw_ids[c]], body_quat[:, st.screw_ids[c]]
            up = torch.zeros(n, 3, device=dev)
            up[:, 2] = props.handle_center_local()
            handles.append(s_pos + math_utils.quat_apply(s_quat, up))
        handles = torch.stack(handles, dim=1)  # (N, 2, 3)
        clear = ((grasp[:, :, None] - handles[:, None]).norm(dim=-1) > CLEAR_M).all(dim=-1).all(dim=-1)

        # Function test: phase 1 tugs the line clamp, phase 2 the ground clamp, each along -x (off the bar's edge).
        # The tug ramps up over TUG_RAMP_S (a step to 40 N on a loose 0.75 kg clamp threw the whole set and the solver
        # went to NaN), and a phase ends early once its clamp has moved RUNAWAY_M (it let go).
        ready = ((ground_attached & line_attached & clear) | st.force_test) & (st.phase == 0)
        st.ready_steps = torch.where(ready, st.ready_steps + 1, torch.zeros_like(st.ready_steps))
        start = ready & ((st.ready_steps >= st.wait_steps) | st.force_test)
        fp = torch.stack(frame_pos, dim=1)  # (N, 2, 3)
        on_own = torch.stack([on_bar[:, 0, 0], line_on_tagged], dim=1)  # (N, 2)
        st.tug_start = torch.where(start[:, None], fp[:, 1], st.tug_start)
        st.phase = torch.where(start, torch.ones_like(st.phase), st.phase)
        st.phase_steps = torch.where(start, torch.zeros_like(st.phase_steps), st.phase_steps)
        for phase, c in ((1, 1), (2, 0)):
            active = st.phase == phase
            if not active.any():
                continue
            moved = (fp[:, c] - st.tug_start).norm(dim=-1)
            done = active & ((st.phase_steps >= st.phase_len) | (moved > RUNAWAY_M))
            st.held[:, c] = torch.where(done, (moved < HOLD_M) & on_own[:, c], st.held[:, c])
            st.phase = torch.where(done, torch.full_like(st.phase, phase + 1), st.phase)
            st.phase_steps = torch.where(done, torch.zeros_like(st.phase_steps), st.phase_steps)
            if phase == 1:
                st.tug_start = torch.where(done[:, None], fp[:, 0], st.tug_start)
        f = torch.zeros(n, len(st.art.body_names), 3, device=dev)
        ramp = ((st.phase_steps + 1).float() / st.ramp_steps).clamp(max=1.0) * TUG_N
        f[:, st.frame_ids[1], 0] = torch.where(st.phase == 1, -ramp, 0.0)
        f[:, st.frame_ids[0], 0] = torch.where(st.phase == 2, -ramp, 0.0)
        if (st.phase > 0).any() or st.tug_on:
            st.art.permanent_wrench_composer.set_forces_and_torques_index(
                forces=f, torques=torch.zeros_like(f), body_ids=list(range(len(st.art.body_names))),
                env_ids=torch.arange(n, device=dev), is_global=True)
            st.tug_on = bool(((st.phase == 1) | (st.phase == 2)).any())
        st.phase_steps = torch.where((st.phase == 1) | (st.phase == 2), st.phase_steps + 1, st.phase_steps)
        end = st.phase == 3
        st.test_passed = torch.where(end, st.held[:, 0] & st.held[:, 1], st.test_passed)

        st.last = {"ground_attached": ground_attached, "line_on_tagged": line_on_tagged,
                   "line_attached": line_attached, "hands_clear": clear,
                   "touch_bars_ground_clamp": touch[:, 0], "touch_bars_line_clamp": touch[:, 1],
                   "slide_mm": torch.stack([jp[:, j] for j in st.slide_j], dim=1) * 1000}
        st.scorer.update(ClampStep(ground_attached, touch[:, :, 1:4], line_on_tagged, line_attached, clear,
                                st.test_passed.clone()))


@configclass
class EventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: ClampTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))
