"""Plug a twist-lock cord cap into the named receptacle, turn it to lock, pass a pull test. Arena props and TaskBase.

Mechanism (twistlock_sim/usd/twistlock_props.py): the panel is a fixed-base articulation (plate, two receptacles,
and per receptacle three contact springs on radial slides, preloaded by this task's joint drives). The cord and cap
are a second fixed-base articulation: anchored under the desk clamp, a 60 cm chain of capsule links (bending springs
toward the routed shape) and the cap. Entry only in the entry orientation, the turn's end stop and the lock against
pull-out are geometry: feet on the blade tips, entry notches in the face, solid face the feet turn behind.

Measured each step, in the named receptacle's frame M (world-aligned, origin on the face, +x inward): s, how far the
cap's face is short of the receptacle face; its lateral offset; its tilt (angle between the cap's axis and +x); its
turn phi (clockwise as seen from the front, about +x); and the same s and lateral offset against the other one.

Hazard events (score 0, stop progress):
- wrong_receptacle: the blade tips within WRONG_TIP_M of the other receptacle's face, within WRONG_LATERAL_M of its
  axis.
- crooked_insertion: while a hand holds the cap (a grasp point at its body, fingers closed on it and commanded
  closed), its blades partly in the named receptacle (tips at least PARTLY_IN[0] past the face, its face at least
  PARTLY_IN[1] short of home), and its axis tilted more than TILT_LIMIT_DEG from the receptacle's axis. A reviewer
  judges it from the video or a photo, against the square of the receptacle's face.

Function test (the success check): once the cap is home and turned (locked) and both grasp points have been clear of
it for TEST_WAIT_S, the cap's body is pulled straight out along the receptacle's axis, the pull ramping to TUG_N over
TUG_RAMP_S and held to TEST_PHASE_S (a spring-scale pull). Passed if the cap moved out less than TUG_HELD_M and is
still turned past LOCKED_MIN_DEG. ``force_function_test(env)`` starts the same pull at once (control runs).
"""

from __future__ import annotations

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

from twistlock_sim.kit import GRASP_OFFSET_M, reset_robot_home
from twistlock_sim.scorer import MILESTONES, Step, TwistLockScorer
from twistlock_sim.spec import Layout, make_seed
from twistlock_sim.usd import twistlock_props as props

_ASSETS = Path(__file__).resolve().parent / "assets"

LIFT_M = 0.01
MOUTH_TIP_M = 0.003
MOUTH_LATERAL_M = 0.002
MOUTH_TILT_DEG = 3.0
MOUTH_PHI_DEG = 4.0
PARTLY_IN = (0.003, 0.001)
"""Blades partly in: tips at least 3 mm past the face, the cap's face at least 1 mm short of home."""
ENGAGED_LATERAL_M = 0.006
TILT_LIMIT_DEG = 10.0
"""Crooked: more than 10 deg off square while the blades are in the slots. A straight blade (1.78 mm) in its slot band
(2.58 mm) through the 6 mm face has about 7.6 deg of free play, so beyond 10 deg a real cap is bending blades or
springing contacts."""
HELD_DIST_M = 0.05
HELD_OPENING_M = 0.085
HELD_TARGET_M = 0.005
WRONG_TIP_M = 0.005
WRONG_LATERAL_M = 0.015
SEATED_S = 0.0012
SEATED_LATERAL_M = 0.0015
SEATED_TILT_DEG = 3.0
RELEASE_CLEAR_M = 0.06
DROP_Z_M = 0.04

TEST_WAIT_S = 1.0
TUG_RAMP_S = 0.5
TUG_HOLD_S = 1.5
TEST_PHASE_S = TUG_RAMP_S + TUG_HOLD_S
TUG_N = 60.0
"""Straight pull on the cap's body (a firm hand tug, ~13.5 lbf; UL 498's latch test for straight-blade plugs uses
89 N)."""
TUG_HELD_M = 0.004

PANEL, CORD_CAP, CRADLE = "panel", "cord_cap", "cradle"


class Panel(Object):
    def __init__(self):
        super().__init__(name=PANEL, usd_path=str(_ASSETS / "panel.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {"contacts": ImplicitActuatorCfg(joint_names_expr=["contact_.*_joint"],
                                                         stiffness=props.CONTACT_K, damping=props.CONTACT_DAMPING,
                                                         armature=0.01)}
        cfg.init_state.joint_pos = {"contact_.*_joint": 0.0}
        return cfg


class CordCap(Object):
    """Anchor under the desk clamp, the flexible cord, the cap (body ``cap``)."""

    def __init__(self):
        super().__init__(name=CORD_CAP, usd_path=str(_ASSETS / "cord_cap.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        # Armature on the light cord joints keeps the chain stable when the cap at its end is pressed into the
        # receptacle (contact impulses on a link 30 times heavier than its neighbours).
        cfg.actuators = {"cord": ImplicitActuatorCfg(joint_names_expr=["cord_j.*"], stiffness=props.CORD_K,
                                                     damping=props.CORD_DAMPING, armature=2e-5)}
        cfg.init_state.joint_pos = {"cord_j.*": 0.0}
        return cfg


class Cradle(Object):
    def __init__(self):
        super().__init__(name=CRADLE, usd_path=str(_ASSETS / "cradle.usda"), object_type=ObjectType.BASE,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))


def cap_state(pos: torch.Tensor, quat: torch.Tensor, rec_root: tuple):
    """(s, lateral, tilt_deg, phi_deg) of the cap against one receptacle (pos in env frame)."""
    n, dev = pos.shape[0], pos.device
    rel = pos - torch.tensor(rec_root, device=dev)
    s = -rel[:, 0]
    lateral = rel[:, 1:].norm(dim=1)
    ex = math_utils.quat_apply(quat, torch.tensor([[1.0, 0.0, 0.0]], device=dev).expand(n, 3))
    ey = math_utils.quat_apply(quat, torch.tensor([[0.0, 1.0, 0.0]], device=dev).expand(n, 3))
    tilt = torch.rad2deg(torch.acos(ex[:, 0].clamp(-1.0, 1.0)))
    phi = torch.rad2deg(torch.atan2(ey[:, 2], ey[:, 1]))
    return s, lateral, tilt, phi


class _TaskState:
    def __init__(self, env: ManagerBasedEnv, task: "TwistLockTask"):
        n, dev = env.num_envs, env.device
        self.panel = env.scene[PANEL]
        self.cord = env.scene[CORD_CAP]
        self.robot = env.scene["robot"]
        self.cap_id = self.cord.find_bodies("cap")[0][0]
        self.contact_joints = self.panel.find_joints("contact_.*_joint")[0]
        self.flange_ids, _ = self.robot.find_bodies(["left_gripper", "right_gripper"], preserve_order=True)
        self.finger_ids = [[self.robot.find_joints(f"{side}_joint{j}")[0][0] for j in (7, 8)]
                           for side in ("left", "right")]
        self.scorer = TwistLockScorer(n, dev)
        self.next_layout = make_seed(task.default_seed)
        self.layouts: dict[int, Layout] = {}
        self.last_reports: dict[int, dict] = {}
        zb = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
        zl = lambda: torch.zeros(n, dtype=torch.long, device=dev)  # noqa: E731
        zf = lambda: torch.zeros(n, device=dev)  # noqa: E731
        self.seated_once, self.locked_once, self.released, self.force_test = zb(), zb(), zb(), zb()
        self.test_held, self.test_passed = zb(), zb()
        self.clear_steps, self.phase, self.phase_steps = zl(), zl(), zl()
        self.test_max_s, self.max_tilt_engaged, self.max_phi, self.min_s, self.phi_now, self.s_now = (
            zf(), zf(), zf(), zf(), zf(), zf())
        self.test_s0 = zf()
        self.locked_time = torch.full((n,), -1.0, device=dev)
        self.start_pos = torch.zeros(n, 3, device=dev)
        self.wait_steps = max(1, round(TEST_WAIT_S / env.step_dt))
        self.phase_len = max(1, round(TEST_PHASE_S / env.step_dt))

    def cap_pose(self, origins):
        pos = self.cord.data.body_pos_w.torch[:, self.cap_id] - origins
        return pos, self.cord.data.body_quat_w.torch[:, self.cap_id]


class TwistLockTask(TaskBase):
    def __init__(self, default_seed: int = 0, episode_length_s: float = 90.0):
        super().__init__(episode_length_s=episode_length_s, task_description=make_seed(default_seed).instruction)
        self.default_seed = default_seed

    def state(self, env: ManagerBasedEnv) -> _TaskState:
        if getattr(env, "_task_state", None) is None:
            env._task_state = _TaskState(env, self)
        return env._task_state

    def force_function_test(self, env: ManagerBasedEnv) -> None:
        """Control runs: start the pull now, locked or not."""
        self.state(env).force_test[:] = True

    def episode_report(self, env: ManagerBasedEnv, env_id: int = 0, ended: bool = False) -> dict:
        st = self.state(env)
        if ended and env_id in st.last_reports:
            return st.last_reports.pop(env_id)
        return self._report(st, env_id)

    def _report(self, st: _TaskState, env_id: int) -> dict:
        r = st.scorer.report(env_id)
        r.update(
            seated=bool(st.seated_once[env_id]), locked=bool(st.locked_once[env_id]),
            locked_time_s=round(float(st.locked_time[env_id]), 2), released=bool(st.released[env_id]),
            turn_deg_now=round(float(st.phi_now[env_id]), 1), max_turn_deg=round(float(st.max_phi[env_id]), 1),
            min_s_mm=round(float(st.min_s[env_id]) * 1000, 2), tug_phase=int(st.phase[env_id]), tug_n=TUG_N,
            tug_max_out_mm=round(float(st.test_max_s[env_id]) * 1000, 2), tug_held=bool(st.test_held[env_id]),
            tug_passed=bool(st.test_passed[env_id]),
            max_tilt_while_engaged_deg=round(float(st.max_tilt_engaged[env_id]), 2),
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
        for buf in (st.seated_once, st.locked_once, st.released, st.force_test, st.test_held, st.test_passed):
            buf[env_ids] = False
        for buf in (st.clear_steps, st.phase, st.phase_steps):
            buf[env_ids] = 0
        for buf in (st.test_max_s, st.max_tilt_engaged, st.max_phi, st.phi_now, st.test_s0):
            buf[env_ids] = 0.0
        st.min_s[env_ids] = 1.0
        st.locked_time[env_ids] = -1.0
        st.cord.permanent_wrench_composer.reset(env_ids=env_ids)
        nc = len(st.contact_joints)
        zc = torch.zeros(n, nc, device=dev)
        st.panel.write_joint_state_to_sim(zc, zc, joint_ids=st.contact_joints, env_ids=env_ids)
        st.panel.set_joint_position_target(zc + props.CONTACT_PRELOAD_M, joint_ids=st.contact_joints,
                                           env_ids=env_ids)
        nj = st.cord.num_joints
        zj = torch.zeros(n, nj, device=dev)
        st.cord.write_joint_state_to_sim(zj, zj, env_ids=env_ids)
        st.cord.set_joint_position_target(zj, env_ids=env_ids)
        st.start_pos[env_ids] = torch.tensor(props.plug_start_root(), device=dev)
        for env_id in env_ids.tolist():
            st.layouts[env_id] = layout

    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        origins = env.scene.env_origins
        n, dev = env.num_envs, env.device
        t = st.scorer.step.float() * env.step_dt

        pos, quat = st.cap_pose(origins)
        s, lateral, tilt, phi = cap_state(pos, quat, props.rec_root("target"))
        s_d, lateral_d, _, _ = cap_state(pos, quat, props.rec_root("decoy"))
        st.phi_now, st.s_now = phi, s
        st.min_s = torch.minimum(st.min_s, s)
        live = st.phase == 0

        wrong = (s_d < props.BLADE_L + WRONG_TIP_M) & (lateral_d < WRONG_LATERAL_M)

        # Held: a hand's grasp point at the cap's body, its fingers closed on it and commanded closed.
        flange_pos = st.robot.data.body_pos_w.torch[:, st.flange_ids]
        flange_quat = st.robot.data.body_quat_w.torch[:, st.flange_ids]
        down = torch.tensor([0.0, 0.0, -GRASP_OFFSET_M], device=dev).expand_as(flange_pos)
        grasp = flange_pos + math_utils.quat_apply(flange_quat, down) - origins[:, None]
        grip_c = pos + math_utils.quat_apply(quat, torch.tensor([[props.GRIP_X, 0.0, 0.0]], device=dev).expand(n, 3))
        jp = st.robot.data.joint_pos.torch
        jt = st.robot.data.joint_pos_target.torch
        closed = torch.stack([(jp[:, ids].sum(dim=1) < HELD_OPENING_M) & (jt[:, ids].sum(dim=1) < HELD_TARGET_M)
                              for ids in st.finger_ids], dim=1)
        held = (((grasp - grip_c[:, None]).norm(dim=-1) < HELD_DIST_M) & closed).any(dim=1)
        partly_in = (s < props.BLADE_L - PARTLY_IN[0]) & (s > PARTLY_IN[1]) & (lateral < ENGAGED_LATERAL_M)
        engaged = partly_in & held & live
        st.max_tilt_engaged = torch.where(engaged, torch.maximum(st.max_tilt_engaged, tilt), st.max_tilt_engaged)
        crooked = engaged & (tilt > TILT_LIMIT_DEG)

        seated = (s < SEATED_S) & (s > -0.001) & (lateral < SEATED_LATERAL_M) & (tilt < SEATED_TILT_DEG)
        st.seated_once |= seated & live
        locked = seated & (phi > props.LOCKED_MIN_DEG)
        st.max_phi = torch.where(seated & live, torch.maximum(st.max_phi, phi), st.max_phi)
        newly = locked & ~st.locked_once & live
        st.locked_time = torch.where(newly, t, st.locked_time)
        st.locked_once |= locked & live
        body_c = pos + math_utils.quat_apply(quat, torch.tensor([props.plug_body_center_offset()], device=dev)
                                             .expand(n, 3))
        clear = ((grasp - body_c[:, None]).norm(dim=-1) > RELEASE_CLEAR_M).all(dim=1)
        st.released = st.released | (locked & clear)

        # Pull test: a spring-scale pull on the cap's body, straight out along the receptacle's axis (-x).
        ready = live & ((locked & clear) | st.force_test)
        st.clear_steps = torch.where(ready, st.clear_steps + 1, torch.zeros_like(st.clear_steps))
        start = ready & ((st.clear_steps >= st.wait_steps) | st.force_test)
        if start.any():
            ids = start.nonzero().flatten()
            st.phase[ids] = 1
            st.phase_steps[ids] = 0
            st.test_s0[ids] = s[ids]
        in_test = st.phase == 1
        if in_test.any():
            ids = in_test.nonzero().flatten()
            scale = ((st.phase_steps[ids].float() + 1) * env.step_dt / TUG_RAMP_S).clamp(max=1.0)
            force = torch.zeros(len(ids), 1, 3, device=dev)
            force[:, 0, 0] = -TUG_N * scale
            st.cord.permanent_wrench_composer.reset(env_ids=ids)
            st.cord.permanent_wrench_composer.add_forces_and_torques_index(
                forces=force, torques=torch.zeros_like(force), body_ids=[st.cap_id], env_ids=ids, is_global=True)
        st.phase_steps = torch.where(in_test, st.phase_steps + 1, st.phase_steps)
        st.test_max_s = torch.where(in_test, torch.maximum(st.test_max_s, s - st.test_s0), st.test_max_s)
        end = in_test & (st.phase_steps >= st.phase_len)
        if end.any():
            ids = end.nonzero().flatten()
            st.cord.permanent_wrench_composer.reset(env_ids=ids)
            st.test_held[ids] = st.test_max_s[ids] < TUG_HELD_M
            st.test_passed[ids] = st.test_held[ids] & st.locked_once[ids] & (phi[ids] > props.LOCKED_MIN_DEG)
            st.phase[ids] = 2

        lifted = (pos - st.start_pos).norm(dim=-1) > LIFT_M
        at_mouth = ((s - props.BLADE_L < MOUTH_TIP_M) & (lateral < MOUTH_LATERAL_M) & (tilt < MOUTH_TILT_DEG)
                    & (phi.abs() < MOUTH_PHI_DEG))
        dropped = pos[:, 2] < DROP_Z_M
        st.scorer.update(Step(lifted, at_mouth, locked & live, st.test_passed.clone(), wrong & live, crooked,
                              dropped))


@configclass
class EventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: TwistLockTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))
