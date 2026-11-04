"""Meter test switch: Arena props and TaskBase.

Mechanism (testswitch_sim/usd/switch_props.py):
- Ten knife blades on revolute joints (0 = closed, -160 deg = open, swung over toward the robot and hanging).
- Spring jaw detent (rule): while a blade is within JAW_RELEASE_DEG of closed, a spring drive pulls it into the jaw
  (breakout JAW_BREAKOUT_NM, inferred); out of the jaw the drive holds it where it is (hinge friction) with a small
  bias toward open (OPEN_BIAS_NM), so a released blade swings down and hangs, and a blade pushed back within
  JAW_RELEASE_DEG of closed is pulled home.
- The test jack (geometry + springs): a mid-pole clip of two sprung leaves (prismatic joints, preloaded spring drives)
  whose slot opens outward; the leaves pinch the probe, which is what holds a released plug. The closed test-jack
  blade lies in the bottom of the slot, so the probe gets in only ~9 mm, short of seated.
- The test lead (geometry + weak springs): a chain of capsule links from the test-set box to the plug's handle.
- CT shorting (rule): a phase's CT is shorted while its short-circuiting blade is SHORT_MAKE_DEG or more from closed
  (make-before-break: before it leaves the jaw at JAW_RELEASE_DEG), and carried by the plug while a plug is seated in
  that phase's jack. A test-jack blade out of its jaw with neither is the open-CT hazard.

After-release check (success): once the plug is seated in the red pole's jack and both grippers have been clear of
the plug for TEST_WAIT_S, the task drives the red blade back toward closed with TEST_CLOSE_NM for TEST_PHASE_S. Passed
if the blade is stopped by the plug (stays more than TEST_HELD_DEG open), the probe is still in the jack and the
voltage poles are still open.
``force_function_test(env)`` starts the same test at once (control runs).
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

from testswitch_sim.kit import Fingertips, reset_robot_home
from testswitch_sim.scorer import MILESTONES, SwitchScorer, SwitchStep
from testswitch_sim.spec import CENTER_Z, FACE_X, POLES, TARGET_JACK, TARGET_SHORT, make_seed
from testswitch_sim.usd import switch_props as props

_ASSETS = Path(__file__).resolve().parent / "assets"

JAW_RELEASE_DEG = 8.0
"""The blade end has risen ~9 mm (its own height) out of the 25 mm jaw at about 8 deg."""
JAW_BREAKOUT_NM = 0.20
"""Torque to pull a blade out of its spring jaw (inferred: about 4 N at the handle; to measure). Our first guess was
0.35 N*m; the station arm (soft PD gains) pulls only about 4-5 N before its fingers ride over the handle."""
JAW_K = JAW_BREAKOUT_NM / math.radians(JAW_RELEASE_DEG)
JAW_DAMPING = 0.02
OPEN_BIAS_NM = 0.10
"""Out of the jaw, gravity and the handle's weight win over hinge friction (open blades hang)."""
SHORT_MAKE_DEG = 3.0
"""The short-circuiting pole's shorting contact is made this far from closed (make-before-break)."""
OPEN_DEG = 60.0
"""A pole counts as open past this angle (well out of the jaw)."""
MOVED_DEG = 5.0

LEAF_K = 2500.0
LEAF_PRELOAD_M = 0.0009
"""Each jack leaf is pulled 0.9 mm past its inner stop: 2.3 N at rest, ~2.9 N on the probe (inferred; to measure)."""
LEAF_DAMPING = 5.0
SEATED_O_M = 0.020
"""Probe tip closer than this to the base face (14 mm into the 30 mm slot; a closed blade stops it at 25 mm)."""
PLUG_NEAR_M = 0.006
DROP_Z_M = 0.02
RELEASE_CLEAR_M = 0.06
TEST_WAIT_S = 1.0
TEST_PHASE_S = 1.5
TEST_CLOSE_NM = 0.3
"""6 N at the handle (about 48 mm above the hinge)."""
TEST_HELD_DEG = 30.0

VOLT_IDS = [i for i, (f, _) in enumerate(POLES) if f == "volt"]
SHORT_IDS = [i for i, (f, _) in enumerate(POLES) if f == "short"]
JACK_IDS = [i for i, (f, _) in enumerate(POLES) if f == "jack"]
NAMED = set(VOLT_IDS) | {TARGET_SHORT, TARGET_JACK}
OTHER_IDS = [i for i in range(len(POLES)) if i not in NAMED]

SWITCH, LEAD, POST = "switch", "lead", "plug_post"


class MeterTestSwitch(Object):
    def __init__(self):
        super().__init__(name=SWITCH, usd_path=str(_ASSETS / "switch.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=(FACE_X, 0.0, CENTER_Z),
                                           rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {
            "poles": ImplicitActuatorCfg(joint_names_expr=["pole.*"], stiffness=JAW_K, damping=JAW_DAMPING),
            "leaves": ImplicitActuatorCfg(joint_names_expr=["leaf.*"], stiffness=LEAF_K, damping=LEAF_DAMPING),
        }
        cfg.init_state.joint_pos = {"pole.*": 0.0, "leaf.*": 0.0}
        return cfg


class TestLead(Object):
    """The test-set box (fixed), the lead and the test plug (body ``plug``)."""

    def __init__(self):
        super().__init__(name=LEAD, usd_path=str(_ASSETS / "lead.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {"lead": ImplicitActuatorCfg(joint_names_expr=["lead_j.*"], stiffness=props.LEAD_K,
                                                     damping=props.LEAD_DAMPING)}
        cfg.init_state.joint_pos = {"lead_j.*": 0.0}
        return cfg


class PlugPost(Object):
    def __init__(self):
        super().__init__(name=POST, usd_path=str(_ASSETS / "post.usda"), object_type=ObjectType.BASE,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))


class _TaskState:
    def __init__(self, env: ManagerBasedEnv, task: "SwitchTask"):
        n, dev = env.num_envs, env.device
        self.switch = env.scene[SWITCH]
        self.lead = env.scene[LEAD]
        self.robot = env.scene["robot"]
        self.tips = Fingertips(self.robot)
        self.plug_id = self.lead.find_bodies("plug")[0][0]
        self.joint_ids = [self.switch.find_joints(f"pole{i}")[0][0] for i in range(len(POLES))]
        self.blade_ids = [self.switch.find_bodies(f"blade{i}")[0][0] for i in range(len(POLES))]
        self.leaf_ids, leaf_names = self.switch.find_joints("leaf.*")
        sign = torch.tensor([1.0 if n.endswith("l") else -1.0 for n in leaf_names], device=dev)
        self.leaf_targets = (sign * LEAF_PRELOAD_M).unsqueeze(0).repeat(n, 1)
        self.scorer = SwitchScorer(n, dev)
        self.next_layout = make_seed(task.default_seed)
        self.last_reports: dict[int, dict] = {}
        self.jacks = torch.tensor([props.jack_mouth_world(i) for i in range(len(POLES))], device=dev)
        self.is_jack = torch.tensor([f == "jack" for f, _ in POLES], device=dev)
        zb = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
        zl = lambda: torch.zeros(n, dtype=torch.long, device=dev)  # noqa: E731
        self.ever_open = torch.zeros(n, len(POLES), dtype=torch.bool, device=dev)
        self.open_order = [[] for _ in range(n)]
        self.force_test, self.test_passed, self.seated_once = zb(), zb(), zb()
        self.clear_steps, self.phase, self.phase_steps = zl(), zl(), zl()
        self.test_min_deg = torch.full((n,), 999.0, device=dev)
        self.wait_steps = max(1, round(TEST_WAIT_S / env.step_dt))
        self.phase_len = max(1, round(TEST_PHASE_S / env.step_dt))

    def plug_pose(self, origins):
        return (self.lead.data.body_pos_w.torch[:, self.plug_id] - origins,
                self.lead.data.body_quat_w.torch[:, self.plug_id])


class SwitchTask(TaskBase):
    def __init__(self, default_seed: int = 0, episode_length_s: float = 160.0):
        super().__init__(episode_length_s=episode_length_s, task_description=make_seed(default_seed).instruction)
        self.default_seed = default_seed

    def state(self, env: ManagerBasedEnv) -> _TaskState:
        if getattr(env, "_task_state", None) is None:
            env._task_state = _TaskState(env, self)
        return env._task_state

    def force_function_test(self, env: ManagerBasedEnv) -> None:
        self.state(env).force_test[:] = True

    def episode_report(self, env: ManagerBasedEnv, env_id: int = 0, ended: bool = False) -> dict:
        st = self.state(env)
        if ended and env_id in st.last_reports:
            return st.last_reports.pop(env_id)
        return self._report(st, env_id)

    def _report(self, st: _TaskState, env_id: int) -> dict:
        r = st.scorer.report(env_id)
        q = st.switch.data.joint_pos.torch[env_id, st.joint_ids]
        r.update(
            opening_order=[i + 1 for i in st.open_order[env_id]],
            pole_open_deg=[round(-math.degrees(float(v)), 1) for v in q],
            plug_seated_once=bool(st.seated_once[env_id]),
            function_test_phase=int(st.phase[env_id]), test_jack_min_open_deg=round(float(st.test_min_deg[env_id]), 1),
            function_test_passed=bool(st.test_passed[env_id]),
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
            st.open_order[env_id] = []
        n, dev = len(env_ids), env.device
        st.scorer.reset(env_ids)
        st.ever_open[env_ids] = False
        for buf in (st.force_test, st.test_passed, st.seated_once):
            buf[env_ids] = False
        for buf in (st.clear_steps, st.phase, st.phase_steps):
            buf[env_ids] = 0
        st.test_min_deg[env_ids] = 999.0
        st.switch.permanent_wrench_composer.reset(env_ids=env_ids)
        zero = torch.zeros(n, len(POLES), device=dev)
        st.switch.write_joint_state_to_sim(zero, zero, joint_ids=st.joint_ids, env_ids=env_ids)
        st.switch.set_joint_position_target(zero, joint_ids=st.joint_ids, env_ids=env_ids)
        zl = torch.zeros(n, len(st.leaf_ids), device=dev)
        st.switch.write_joint_state_to_sim(zl, zl, joint_ids=st.leaf_ids, env_ids=env_ids)
        st.switch.set_joint_position_target(st.leaf_targets[env_ids], joint_ids=st.leaf_ids, env_ids=env_ids)
        zj = torch.zeros(n, st.lead.num_joints, device=dev)
        st.lead.write_joint_state_to_sim(zj, zj, env_ids=env_ids)
        st.lead.set_joint_position_target(zj, env_ids=env_ids)

    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        origins = env.scene.env_origins
        n, dev = env.num_envs, env.device

        # Jaw detent: in the jaw, a spring pulls the blade closed; out of it, hold where it is.
        q = st.switch.data.joint_pos.torch[:, st.joint_ids]
        open_deg = -torch.rad2deg(q)
        in_jaw = open_deg < JAW_RELEASE_DEG
        bias = OPEN_BIAS_NM / JAW_K
        target = torch.where(in_jaw, torch.zeros_like(q), q - bias)
        st.switch.set_joint_position_target(target, joint_ids=st.joint_ids)

        is_open = open_deg > OPEN_DEG
        for env_id, pole in (is_open & ~st.ever_open).nonzero().tolist():
            st.open_order[env_id].append(pole)
        st.ever_open |= is_open
        volts_open = is_open[:, VOLT_IDS].all(dim=1)
        currents_open = is_open[:, TARGET_SHORT] & is_open[:, TARGET_JACK]
        other_moved = (open_deg[:, OTHER_IDS] > MOVED_DEG).any(dim=1)

        # Plug probe vs the jacks (plug origin = probe tip; its local +z runs tip -> handle).
        p_pos, p_quat = st.plug_pose(origins)
        axis = math_utils.quat_apply(p_quat, torch.tensor([[0.0, 0.0, 1.0]], device=dev).expand(n, 3))
        d = p_pos[:, None] - st.jacks[None]  # (N, 10, 3); d.x < 0 is outside the mouth
        tip_o = FACE_X - p_pos[:, 0]
        in_slot_yz = (d[..., 1].abs() < props.JACK_GAP / 2 + 0.0005) & (d[..., 2].abs() < props.JACK_X / 2 + 0.004)
        # The probe's flat stays in the jack's plane; only its out-of-plane tilt is limited.
        in_plane = axis[:, 1].abs() < math.sin(math.radians(15))
        seated_any = in_slot_yz & (tip_o < SEATED_O_M)[:, None] & in_plane[:, None] & st.is_jack[None] & ~in_jaw
        seated = seated_any[:, TARGET_JACK]
        st.seated_once |= seated
        near = in_slot_yz & (d[..., 0] > -PLUG_NEAR_M) & st.is_jack[None]
        plug_at_closed = (near & in_jaw).any(dim=1) & (st.phase == 0)
        dropped = p_pos[:, 2] < DROP_Z_M

        # CT circuit: a test-jack blade out of its jaw, its phase's short not made and no plug seated in that jack.
        ct_opened = torch.zeros(n, dtype=torch.bool, device=dev)
        for s, j in zip(SHORT_IDS, JACK_IDS):
            ct_opened |= ~in_jaw[:, j] & (open_deg[:, s] < SHORT_MAKE_DEG) & ~seated_any[:, j]
        ct_opened &= st.phase == 0

        # After-release check.
        pads = st.tips.points(origins)
        handle_c = p_pos + axis * props.PLUG_TIP_TO_HANDLE_C
        clear = ((pads - handle_c[:, None]).norm(dim=-1) > RELEASE_CLEAR_M).all(dim=1)
        ready = ((seated & volts_open & clear) | st.force_test) & (st.phase == 0)
        st.clear_steps = torch.where(ready, st.clear_steps + 1, torch.zeros_like(st.clear_steps))
        start = ready & ((st.clear_steps >= st.wait_steps) | st.force_test)
        if start.any():
            ids = start.nonzero().flatten()
            torque = torch.zeros(len(ids), 1, 3, device=dev)
            torque[..., 1] = TEST_CLOSE_NM  # +y turns the blade back toward closed
            st.switch.permanent_wrench_composer.add_forces_and_torques_index(
                forces=torch.zeros_like(torque), torques=torque, body_ids=[st.blade_ids[TARGET_JACK]], env_ids=ids,
                is_global=True)
            st.phase[ids] = 1
            st.phase_steps[ids] = 0
        in_test = st.phase == 1
        st.phase_steps = torch.where(in_test, st.phase_steps + 1, st.phase_steps)
        st.test_min_deg = torch.where(in_test, torch.minimum(st.test_min_deg, open_deg[:, TARGET_JACK]), st.test_min_deg)
        end = in_test & (st.phase_steps >= st.phase_len)
        if end.any():
            ids = end.nonzero().flatten()
            st.switch.permanent_wrench_composer.reset(env_ids=ids)
            held = st.test_min_deg[ids] > TEST_HELD_DEG
            st.test_passed[ids] = held & seated_any[ids, TARGET_JACK] & volts_open[ids]
            st.phase[ids] = 2

        st.scorer.update(SwitchStep(volts_open, currents_open, seated, st.test_passed.clone(), ct_opened,
                                    other_moved & (st.phase == 0), plug_at_closed, dropped))


@configclass
class EventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: SwitchTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))
