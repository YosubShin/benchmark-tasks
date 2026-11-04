"""Trip the named breaker from its pistol-grip control switch: the panel articulation, the task (reset, per-step
breaker logic and scoring) and the after-release check.

Mechanism (usd/panel_props.py):
- Each handle turns on a revolute joint with stops at +-45 deg (TRIP < 0 < CLOSE). Its return spring is the joint
  drive (a torsion spring toward NORMAL, 0.9 N m at the stop); the task never sets the handle's target.
- Each target flag is a coaxial revolute body driven only by contact: a pin on the handle's shaft pushes it through
  lugs with lost motion. TRIP leaves it green, CLOSE leaves it red, and the spring return does not move it.
- The breaker behind each switch is a rule, the same logic as the real-world stand-in circuit (an on-delay timer and
  a latching relay on a 24 V battery, see task.json): while the switch's TRIP contact is made (handle within 5 deg of
  the TRIP stop) for TRIP_DELAY_S without a break, the breaker opens. Release earlier and the timer resets, so the
  breaker stays closed. The CLOSE contact likewise closes an open breaker after CLOSE_DELAY_S. The lamps follow the
  breaker (red = closed, green = open), shown by toggling lit and dark lenses.

After-release check (the success check): once the named breaker is open and all four finger pads have been more than
RELEASE_CLEAR_M from the named grip for WAIT_S, the check watches for CHECK_S more (under the station harness it runs
off camera, the robot holding its last action): pass if the named handle is back at NORMAL (within NORMAL_DEG) by its
own spring, its target shows green, its breaker is open, the other two breakers are closed with their handles at
NORMAL, and the pads are still clear.
"""

from __future__ import annotations

import math
from dataclasses import MISSING
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
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

from trip_sim.kit import Fingertips, reset_robot_home, set_visible
from trip_sim.scorer import MILESTONES, TripScorer, TripStep
from trip_sim.spec import AXIS_Z, BREAKERS, FACE_X, SWITCH_Y, TripSeed, make_seed
from trip_sim.usd import panel_props as props

_ASSETS = Path(__file__).resolve().parent / "assets"
NS = len(BREAKERS)

HANDLE_DAMPING = 0.02
TARGET_DAMPING = 2e-4
TRIP_DELAY_S = 1.0
"""TRIP contact time until the breaker opens and the green lamp lights (the stand-in's timer setting)."""
CLOSE_DELAY_S = 1.0
CONTACT_RAD = math.radians(props.CONTACT_DEG)
MOVED_RAD = math.radians(10.0)
NORMAL_DEG = 5.0
NORMAL_RAD = math.radians(NORMAL_DEG)
OTHER_RAD = math.radians(10.0)
CLOSE_TURN_RAD = math.radians(20.0)
TARGET_GREEN_RAD = -math.radians(props.TARGET_SHOWN_DEG / 2)
RELEASE_CLEAR_M = 0.06
WAIT_S = 1.0
CHECK_S = 1.0
GRIP_LOCAL = (-props.HANDLE_DEPTH + props.GRIP_T / 2, 0.0, -0.035)
"""A point on the grip, 35 mm below the shaft (handle frame): where a hand takes it."""


def grip_point(slot: int, q: float = 0.0, r: float = 0.035) -> tuple[float, float, float]:
    """Desk-frame point on grip `slot`'s centre line, r below the shaft, with the handle at angle q."""
    x = GRIP_LOCAL[0]
    return (FACE_X + x, SWITCH_Y[slot] + r * math.sin(q), AXIS_Z - r * math.cos(q))


class Panel(Object):
    def __init__(self):
        super().__init__(name="panel", usd_path=str(_ASSETS / "panel.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=(FACE_X, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {
            "handles": ImplicitActuatorCfg(joint_names_expr=["handle_.*_joint"], stiffness=props.SPRING_K,
                                           damping=HANDLE_DAMPING),
            "targets": ImplicitActuatorCfg(joint_names_expr=["target_.*_joint"], stiffness=0.0,
                                           damping=TARGET_DAMPING),
        }
        cfg.init_state.joint_pos = {"handle_.*_joint": 0.0, "target_.*_joint": math.radians(props.TARGET_SHOWN_DEG)}
        return cfg


class _TaskState:
    def __init__(self, env: ManagerBasedEnv, task: "TripTask"):
        n, dev = env.num_envs, env.device
        self.panel = env.scene["panel"]
        self.robot = env.scene["robot"]
        self.tips = Fingertips(self.robot)
        self.handle_joints = self.panel.find_joints([f"handle_{i}_joint" for i in range(NS)], preserve_order=True)[0]
        self.target_joints = self.panel.find_joints([f"target_{i}_joint" for i in range(NS)], preserve_order=True)[0]
        self.scorer = TripScorer(n, dev)
        self.next_layout = make_seed(task.default_seed)
        self.layouts: dict[int, TripSeed] = {}
        self.last_reports: dict[int, dict] = {}
        self.panel_paths = sim_utils.find_matching_prim_paths("/World/envs/env_.*/panel")
        self.named = torch.zeros(n, dtype=torch.long, device=dev)
        zb = lambda: torch.zeros(n, NS, dtype=torch.bool, device=dev)  # noqa: E731
        self.closed, self.lamp_shown, self.prev_trip = zb(), zb(), zb()
        self.trip_time = torch.zeros(n, NS, device=dev)
        self.close_time = torch.zeros(n, NS, device=dev)
        self.min_q = torch.zeros(n, NS, device=dev)
        self.max_q = torch.zeros(n, NS, device=dev)
        zl = lambda: torch.zeros(n, dtype=torch.long, device=dev)  # noqa: E731
        self.clear_steps, self.phase, self.phase_steps = zl(), zl(), zl()
        self.passed = torch.zeros(n, dtype=torch.bool, device=dev)
        self.tripped_at = torch.full((n,), -1.0, device=dev)
        self.trip_contact_s = torch.zeros(n, device=dev)
        self.wait_len = max(1, round(WAIT_S / env.step_dt))
        self.check_len = max(1, round(CHECK_S / env.step_dt))
        self.dt = env.step_dt


class TripTask(TaskBase):
    def __init__(self, default_seed: int = 0, episode_length_s: float = 40.0):
        super().__init__(episode_length_s=episode_length_s, task_description=make_seed(default_seed).instruction)
        self.default_seed = default_seed

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
        return self._report(st, env_id)

    def _report(self, st: _TaskState, env_id: int) -> dict:
        r = st.scorer.report(env_id)
        layout = st.layouts.get(env_id, st.next_layout)
        q = st.panel.data.joint_pos.torch[env_id, st.handle_joints]
        qt = st.panel.data.joint_pos.torch[env_id, st.target_joints]
        deg = lambda t: [round(math.degrees(float(v)), 1) for v in t.tolist()]  # noqa: E731
        r.update(
            named_breaker=layout.device,
            handle_deg=deg(q), handle_min_deg=deg(st.min_q[env_id]), handle_max_deg=deg(st.max_q[env_id]),
            target=["green" if v < TARGET_GREEN_RAD else ("red" if v > -TARGET_GREEN_RAD else "-") for v in qt.tolist()],
            target_deg=deg(qt),
            breaker=["closed" if c else "open" for c in st.closed[env_id].tolist()],
            lamps=["red" if c else "green" for c in st.closed[env_id].tolist()],
            named_trip_contact_s=round(float(st.trip_contact_s[env_id]), 2),
            named_tripped_at_s=round(float(st.tripped_at[env_id]), 2),
            check_phase=int(st.phase[env_id]),
            check_passed=bool(st.passed[env_id]),
        )
        return r

    def get_scene_cfg(self):
        return None

    def get_events_cfg(self):
        return TripEventsCfg(self)

    def get_termination_cfg(self) -> TaskTerminationCfg:
        sequence = [(self._milestone_predicate(m), 1.0 / len(MILESTONES)) for m in MILESTONES]
        return TaskTerminationCfg(timeout_s=self.episode_length_s,
                                  success=[CompletionCriteria(name="trip_milestones", predicate_sequence=sequence)])

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
        st.closed[env_ids] = True
        st.prev_trip[env_ids] = False
        for buf in (st.trip_time, st.close_time, st.min_q, st.max_q):
            buf[env_ids] = 0.0
        for buf in (st.clear_steps, st.phase, st.phase_steps):
            buf[env_ids] = 0
        st.passed[env_ids] = False
        st.tripped_at[env_ids] = -1.0
        st.trip_contact_s[env_ids] = 0.0
        st.named[env_ids] = layout.named_slot
        zero = torch.zeros(n, NS, device=dev)
        st.panel.write_joint_state_to_sim(zero, zero, joint_ids=st.handle_joints, env_ids=env_ids)
        st.panel.set_joint_position_target(zero, joint_ids=st.handle_joints, env_ids=env_ids)
        red = zero + math.radians(props.TARGET_SHOWN_DEG)
        st.panel.write_joint_state_to_sim(red, zero, joint_ids=st.target_joints, env_ids=env_ids)
        st.panel.set_joint_position_target(red, joint_ids=st.target_joints, env_ids=env_ids)
        for env_id in env_ids.tolist():
            st.layouts[env_id] = layout
        st.lamp_shown[env_ids] = ~st.closed[env_ids]  # force a redraw
        self._show_lamps(st)

    def _show_lamps(self, st: _TaskState) -> None:
        changed = st.closed != st.lamp_shown
        if not changed.any():
            return
        stage = sim_utils.get_current_stage()
        for env_id, i in changed.nonzero().tolist():
            base = f"{st.panel_paths[env_id]}/base"
            closed = bool(st.closed[env_id, i])
            set_visible(stage, f"{base}/lamp_red_on_{i}", closed)
            set_visible(stage, f"{base}/lamp_red_off_{i}", not closed)
            set_visible(stage, f"{base}/lamp_green_on_{i}", not closed)
            set_visible(stage, f"{base}/lamp_green_off_{i}", closed)
        st.lamp_shown = st.closed.clone()

    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        n, dev = env.num_envs, env.device
        origins = env.scene.env_origins
        q = st.panel.data.joint_pos.torch[:, st.handle_joints]  # (N, 3)
        qt = st.panel.data.joint_pos.torch[:, st.target_joints]
        st.min_q = torch.minimum(st.min_q, q)
        st.max_q = torch.maximum(st.max_q, q)
        named = torch.nn.functional.one_hot(st.named, NS).bool()

        # Breakers (the stand-in circuit): trip / close timer runs while the contact is made, resets when it breaks.
        at_trip = q <= -CONTACT_RAD
        at_close = q >= CONTACT_RAD
        st.trip_time = torch.where(at_trip, st.trip_time + st.dt, torch.zeros_like(st.trip_time))
        st.close_time = torch.where(at_close, st.close_time + st.dt, torch.zeros_like(st.close_time))
        st.trip_contact_s = torch.maximum(st.trip_contact_s, (st.trip_time * named).sum(dim=1))
        trips = st.closed & (st.trip_time >= TRIP_DELAY_S - 1e-6)
        released_early = (st.prev_trip & ~at_trip & st.closed & named).any(dim=1)
        newly = (trips & named).any(dim=1) & (st.tripped_at < 0)
        st.tripped_at = torch.where(newly, st.scorer.step.float() * st.dt, st.tripped_at)
        st.closed = (st.closed & ~trips) | (st.close_time >= CLOSE_DELAY_S - 1e-6)
        st.prev_trip = at_trip
        self._show_lamps(st)

        # Release and the after-release check.
        pads = st.tips.points(origins)  # (N, 4, 3)
        gx, _, gz = GRIP_LOCAL
        ys = torch.tensor(SWITCH_Y, device=dev)
        grips = torch.stack([torch.full_like(q, FACE_X + gx), ys - gz * torch.sin(q), AXIS_Z + gz * torch.cos(q)],
                            dim=-1)  # (N, 3, 3)
        d = (pads[:, :, None] - grips[:, None]).norm(dim=-1)  # (N, 4, 3)
        d_named = d.gather(2, st.named.view(n, 1, 1).expand(n, 4, 1)).squeeze(2)
        clear = (d_named > RELEASE_CLEAR_M).all(dim=1)
        q_named = (q * named).sum(dim=1)
        qt_named = (qt * named).sum(dim=1)
        named_open = (~st.closed & named).any(dim=1)
        others_ok = ((st.closed & (q.abs() < NORMAL_RAD)) | named).all(dim=1)
        ok = named_open & (q_named.abs() < NORMAL_RAD) & (qt_named < TARGET_GREEN_RAD) & others_ok & clear

        ready = (st.phase == 0) & named_open & clear
        st.clear_steps = torch.where(ready, st.clear_steps + 1, torch.zeros_like(st.clear_steps))
        start = ready & (st.clear_steps >= st.wait_len)
        st.phase = torch.where(start, torch.ones_like(st.phase), st.phase)
        st.phase_steps = torch.where(st.phase == 1, st.phase_steps + 1, st.phase_steps)
        end = (st.phase == 1) & (st.phase_steps >= st.check_len)
        st.passed |= end & ok
        st.phase = torch.where(end, torch.full_like(st.phase, 2), st.phase)

        st.scorer.update(TripStep(
            named_moved=q_named <= -MOVED_RAD,
            named_at_trip=q_named <= -CONTACT_RAD,
            named_tripped=named_open,
            check_passed=st.passed.clone(),
            other_turned=((q.abs() > OTHER_RAD) & ~named).any(dim=1),
            close_turned=(q > CLOSE_TURN_RAD).any(dim=1),
            released_early=released_early,
        ))


@configclass
class TripEventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: TripTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))
