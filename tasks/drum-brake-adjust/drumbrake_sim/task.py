"""Drum brake adjust task: Arena props and TaskBase (props: drumbrake_sim/usd/props.py, layout: drumbrake_sim/spec.py).

Mechanism (geometry and contact, plus two rules):
- Star wheel and pawl (joint ``star``, about x): the wheel's 24 teeth are colliders; the spoon's tip turns it only by
  pushing a tooth. The self-adjuster pawl is a rule on the joint's drive: its target is the notch the pawl sits in
  (``notch`` x the tooth pitch). Turning forward (expanding) the drive resists with the wheel's running torque
  (SW_RUN_NM plus the screw's load from the shoes' contact force); once the wheel has turned a whole tooth past the
  notch the pawl drops into the next one (a click: ``notch`` + 1). Released short of that, the pawl's slope turns it
  back to its notch. Backward the pawl holds it (SW_BACK_NM): an overshoot cannot be undone with the spoon alone.
- Drum drag (joint ``drum``, about y): a Coulomb friction drive (a stiff spring whose anchor slips once it carries
  T), with T = S.drag_nm(e): the hub's own turning torque, plus DRAG_PER_NOTCH per notch of shoe expansion past the
  linings' first contact with the drum. e is the wheel's turned angle in teeth (expansion follows the wheel).

After-release check (the success check), once the spoon's tip is out of the hole and every finger pad has stayed
RELEASE_CLEAR from the brake for TEST_WAIT_S without a break: a torque about the axle is ramped 0 -> TEST_MAX_NM over
TEST_S on the drum (as a spring scale on a stud); the torque at which the drum has turned TEST_TURN_DEG is the measured
turning torque; it passes if within BAND.
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

from drumbrake_sim import spec as S
from drumbrake_sim.kit import Fingertips, reset_robot_home
from drumbrake_sim.scorer import MILESTONES, Scorer, Step

_ASSETS = Path(__file__).resolve().parent / "assets"
IDENTITY = (0.0, 0.0, 0.0, 1.0)
HAZARD_TEST = os.environ.get("DBA_HAZARD_TEST", "")
DEBUG_EVERY = int(os.environ.get("DBA_DEBUG", "0") or 0)


class Bench(Object):
    def __init__(self):
        super().__init__(name="bench", usd_path=str(_ASSETS / "bench.usda"), object_type=ObjectType.BASE,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=IDENTITY))


class Brake(Object):
    def __init__(self):
        super().__init__(name="brake", usd_path=str(_ASSETS / "brake.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=IDENTITY))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {
            "drum": ImplicitActuatorCfg(joint_names_expr=["drum"], stiffness=S.DRUM_K, damping=S.DRUM_D,
                                        effort_limit_sim=S.HUB_NM),
            "star": ImplicitActuatorCfg(joint_names_expr=["star"], stiffness=S.SW_K, damping=S.SW_D,
                                        effort_limit_sim=S.SW_BACK_NM, armature=S.SW_ARMATURE),
        }
        cfg.init_state.joint_pos = {"drum": 0.0, "star": 0.0}
        return cfg


class Spoon(Object):
    def __init__(self):
        super().__init__(name="spoon", usd_path=str(_ASSETS / "spoon.usda"), object_type=ObjectType.RIGID,
                         initial_pose=Pose(position_xyz=S.spoon_start(), rotation_xyzw=IDENTITY))


def all_props() -> list:
    return [Bench(), Brake(), Spoon()]


def _axis(quat, v):
    vv = torch.tensor(v, device=quat.device, dtype=quat.dtype).expand(quat.shape[0], 3)
    return math_utils.quat_apply(quat, vv)


class _TaskState:
    def __init__(self, env: ManagerBasedEnv, task: "DrumBrakeTask"):
        n, dev = env.num_envs, env.device
        self.robot = env.scene["robot"]
        self.brake = env.scene["brake"]
        self.spoon = env.scene["spoon"]
        self.tips = Fingertips(self.robot)
        self.j_drum = self.brake.find_joints("drum")[0][0]
        self.j_star = self.brake.find_joints("star")[0][0]
        self.b_drum = self.brake.find_bodies("drum")[0][0]
        self.scorer = Scorer(n, dev)
        self.last_reports: dict[int, dict] = {}
        zf = lambda: torch.zeros(n, device=dev)  # noqa: E731
        zb = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
        self.notch = torch.zeros(n, dtype=torch.long, device=dev)
        self.drum_prev, self.drum_acc, self.drum_hold, self.drag, self.e = zf(), zf(), zf(), zf(), zf()
        self.force_test, self.test_passed, self.ready_at_start = zb(), zb(), zb()
        self.clear_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.phase = torch.zeros(n, dtype=torch.long, device=dev)
        self.phase_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.test_start, self.test_turn, self.test_break_nm, self.test_nm = zf(), zf(), zf(), zf()
        self.wait_steps = max(1, round(S.TEST_WAIT_S / env.step_dt))
        self.test_len = max(1, round(S.TEST_S / env.step_dt))
        self.m: dict[str, torch.Tensor] = {}
        self.hazard_done = False
        self.max_drag = zf()
        self.th_ref, self.e_ref = zf(), zf()
        self.eng_prev = zb()


class DrumBrakeTask(TaskBase):
    def __init__(self, episode_length_s: float = S.EPISODE_SECONDS):
        super().__init__(episode_length_s=episode_length_s, task_description=S.instruction())

    def state(self, env: ManagerBasedEnv) -> _TaskState:
        if getattr(env, "_task_state", None) is None:
            env._task_state = _TaskState(env, self)
        return env._task_state

    def force_test(self, env: ManagerBasedEnv) -> None:
        self.state(env).force_test[:] = True

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
        r.update(notches=int(st.notch[env_id]), drag_nm=round(float(st.drag[env_id]), 3),
                 max_drag_nm=round(float(st.max_drag[env_id]), 3), test_phase=int(st.phase[env_id]),
                 test_break_nm=round(float(st.test_break_nm[env_id]), 3), test_passed=bool(st.test_passed[env_id]))
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
        for buf in (st.force_test, st.test_passed, st.ready_at_start):
            buf[env_ids] = False
        for buf in (st.clear_steps, st.phase, st.phase_steps, st.notch):
            buf[env_ids] = 0
        for buf in (st.drum_prev, st.drum_acc, st.drum_hold, st.drag, st.e, st.test_start, st.test_turn,
                    st.test_break_nm, st.test_nm, st.max_drag):
            buf[env_ids] = 0.0
        st.hazard_done = False
        st.eng_prev[env_ids] = False
        st.brake.permanent_wrench_composer.reset(env_ids=env_ids)
        st.spoon.permanent_wrench_composer.reset(env_ids=env_ids)
        q = torch.zeros(n, st.brake.num_joints, device=dev)
        st.brake.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
        st.brake.set_joint_position_target(q.clone(), env_ids=env_ids)
        pose = torch.tensor([[*S.spoon_start(), *IDENTITY]], device=dev).repeat(n, 1)
        pose[:, :3] += origins[env_ids]
        st.spoon.write_root_pose_to_sim(pose, env_ids=env_ids)
        st.spoon.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)

    def _mechanism(self, st: _TaskState, origins: torch.Tensor) -> None:
        jp = st.brake.data.joint_pos.torch
        dev = jp.device
        # Star wheel and pawl.
        e = S.EXPAND_SIGN * jp[:, st.j_star] / S.PITCH_ANG
        st.notch = torch.maximum(st.notch, torch.floor(e + 0.02).to(torch.long))
        st.e = e
        drag = S.drag_nm(e)
        st.drag = drag
        st.max_drag = torch.maximum(st.max_drag, drag)
        fwd = S.star_fwd_nm(drag)
        # Spoon tip on the teeth (the tooth-contact rule): while the tip is in among the teeth, the wheel follows the
        # tip's angle about its axis the way the tip moves the tooth in front of it (expanding only; moved back, the
        # tip slips back over the teeth and takes a new hold). The wheel's running torque pushes back on the tip.
        tip = st.spoon.data.root_pos_w.torch - origins
        rel = tip - torch.tensor(S.SW, device=dev)
        r = torch.sqrt(rel[:, 1] ** 2 + rel[:, 2] ** 2)
        th = torch.atan2(rel[:, 2], -rel[:, 1])
        eng = (r <= S.SW_RT) & (rel[:, 0].abs() <= S.ENGAGE_X) & (tip[:, 1] > S.BP_Y + S.BP_T)
        grab = eng & ~st.eng_prev
        st.th_ref = torch.where(grab, th, st.th_ref)
        st.e_ref = torch.where(grab, e, st.e_ref)
        want = st.e_ref + (th - st.th_ref) / S.PITCH_ANG
        slip = eng & (want < e - 0.02)
        st.th_ref = torch.where(slip, th, st.th_ref)
        st.e_ref = torch.where(slip, e, st.e_ref)
        want = torch.where(slip, e, want)
        drive = eng & (want > e - 0.02)
        st.eng_prev = eng
        tgt = torch.where(drive, want, st.notch.to(e.dtype))
        back = torch.full_like(fwd, S.SW_BACK_NM)
        limit = torch.where(drive, back, torch.where(e >= st.notch.to(e.dtype) - 0.05, fwd, back))
        st.brake.write_joint_effort_limit_to_sim_index(limits=limit.unsqueeze(1), joint_ids=[st.j_star])
        st.brake.set_joint_position_target((S.EXPAND_SIGN * tgt * S.PITCH_ANG).unsqueeze(1), joint_ids=[st.j_star])
        n = jp.shape[0]
        f = torch.zeros(n, 1, 3, device=dev)
        mag = torch.where(drive, fwd / r.clamp(min=0.005), torch.zeros_like(fwd))
        f[:, 0, 1] = -mag * torch.sin(th)
        f[:, 0, 2] = -mag * torch.cos(th)
        st.spoon.permanent_wrench_composer.set_forces_and_torques_index(
            forces=f, torques=torch.zeros_like(f), positions=(tip + origins)[:, None], body_ids=[0],
            env_ids=torch.arange(n, device=dev), is_global=True)
        # Drum drag: Coulomb friction drive.
        q = jp[:, st.j_drum]
        d = torch.remainder(q - st.drum_prev + math.pi, 2 * math.pi) - math.pi
        st.drum_prev = q.clone()
        st.drum_acc = st.drum_acc + d
        st.drum_hold = torch.minimum(torch.maximum(st.drum_hold, st.drum_acc - drag / S.DRUM_K),
                                     st.drum_acc + drag / S.DRUM_K)
        st.brake.write_joint_effort_limit_to_sim_index(limits=drag.unsqueeze(1), joint_ids=[st.j_drum])
        st.brake.set_joint_position_target((q + st.drum_hold - st.drum_acc).unsqueeze(1), joint_ids=[st.j_drum])

    def _hazard_hooks(self, env, st: _TaskState) -> None:
        """Test harness (DBA_HAZARD_TEST, empty in scored runs): cause a hazard as a slip or a hand would, once."""
        if not HAZARD_TEST or st.hazard_done:
            return
        case, _, when = HAZARD_TEST.partition("_")
        step = int(st.scorer.step[0])
        m = int(st.scorer.milestone[0])
        due = (when == "start" and step == round(1.0 / env.step_dt)) or (when.isdigit() and m >= int(when))
        if not due:
            return
        st.hazard_done = True
        origins = env.scene.env_origins
        ids = torch.arange(env.num_envs, device=env.device)
        if case == "over":
            # The star wheel is turned on past light drag (as if clicked too far): to 5 notches past first contact
            # (drag 1.35 N m), from wherever it is.
            st.notch[:] = torch.clamp(st.notch, min=int(math.ceil(S.CONTACT_NOTCHES)) + 4)
            jp = st.brake.data.joint_pos.torch.clone()
            jp[:, st.j_star] = S.EXPAND_SIGN * st.notch.to(jp.dtype) * S.PITCH_ANG
            st.brake.write_joint_state_to_sim(jp, torch.zeros_like(jp), env_ids=ids)
        elif case == "drop":
            # The spoon falls onto the bench; at the start it is put back on its rest the next step.
            pose = torch.cat([st.spoon.data.root_pos_w.torch, st.spoon.data.root_quat_w.torch], dim=-1).clone()
            low = pose.clone()
            low[:, 0] = origins[:, 0] + 0.26
            low[:, 1] = origins[:, 1] - 0.15
            low[:, 2] = origins[:, 2] + 0.004
            low[:, 3:] = torch.tensor(IDENTITY, device=env.device)
            st.spoon.write_root_pose_to_sim(low, env_ids=ids)
            st.spoon.write_root_velocity_to_sim(torch.zeros(env.num_envs, 6, device=env.device), env_ids=ids)
            if when == "start":
                st._restore_spoon = pose
        print(f"[drum-brake] HAZARD TEST {HAZARD_TEST} applied at step {step} (milestone {m})", flush=True)

    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        origins = env.scene.env_origins
        n, dev = env.num_envs, env.device
        restore = getattr(st, "_restore_spoon", None)
        self._mechanism(st, origins)
        idle = st.phase == 0

        sp = st.spoon.data.root_pos_w.torch - origins
        sq = st.spoon.data.root_quat_w.torch
        tip = sp
        com = sp + _axis(sq, (0.0, -0.150, 0.0))
        rel = tip - torch.tensor(S.SW, device=dev)
        r_yz = torch.sqrt(rel[:, 1] ** 2 + rel[:, 2] ** 2)
        engaged = (r_yz <= S.ENGAGE_R) & (rel[:, 0].abs() <= S.ENGAGE_X) & (tip[:, 1] > S.BP_Y + S.BP_T)
        depth = tip[:, 1] - S.BP_Y
        clicked = st.notch >= S.CLICKS_MIN
        in_band = (st.drag >= S.BAND[0]) & (st.drag <= S.BAND[1])
        over = st.drag > S.BAND[1]
        dropped = com[:, 2] < S.DROP_Z

        # Finger pads clear of the brake (drum, studs, plate): outside a cylinder round the axle grown by the margin.
        pads = st.tips.points(origins)
        c = S.RELEASE_CLEAR
        rr = torch.sqrt((pads[..., 0] - S.AXLE[0]) ** 2 + (pads[..., 2] - S.AXLE[2]) ** 2)
        inside = (rr < S.DRUM_RO + c) & (pads[..., 1] > S.BP_Y - c) & (pads[..., 1] < S.DRUM_FACE_Y + S.STUD_L + c)
        clear = ~inside.any(dim=1)
        tool_out = depth < -S.TOOL_OUT

        go = idle & (((st.scorer.milestone >= 75) & clear & tool_out) | st.force_test)
        st.clear_steps = torch.where(go, st.clear_steps + 1, torch.zeros_like(st.clear_steps))
        start = go & ((st.clear_steps >= st.wait_steps) | st.force_test)
        if start.any():
            st.phase = torch.where(start, torch.ones_like(st.phase), st.phase)
            st.phase_steps = torch.where(start, torch.zeros_like(st.phase_steps), st.phase_steps)
            st.test_start = torch.where(start, st.drum_acc, st.test_start)
            st.test_break_nm = torch.where(start, torch.full_like(st.test_break_nm, float("inf")), st.test_break_nm)
        testing = st.phase == 1
        if testing.any():
            ids = testing.nonzero().flatten()
            turned = (st.drum_acc - st.test_start).abs()
            broke = testing & (turned >= math.radians(S.TEST_TURN_DEG)) & torch.isinf(st.test_break_nm)
            st.test_break_nm = torch.where(broke, st.test_nm, st.test_break_nm)
            ramp = ((st.phase_steps[ids].to(torch.float32) + 1) / st.test_len).clamp(max=1.0)
            st.test_nm[ids] = S.TEST_MAX_NM * ramp
            torques = torch.zeros(len(ids), 1, 3, device=dev)
            torques[:, 0, 1] = st.test_nm[ids]
            st.brake.permanent_wrench_composer.set_forces_and_torques_index(
                forces=torch.zeros_like(torques), torques=torques, body_ids=[st.b_drum], env_ids=ids, is_global=True)
            st.phase_steps = torch.where(testing, st.phase_steps + 1, st.phase_steps)
            end = testing & ((st.phase_steps > st.test_len) | ~torch.isinf(st.test_break_nm))
            if end.any():
                eids = end.nonzero().flatten()
                st.brake.permanent_wrench_composer.reset(env_ids=eids)
                b = st.test_break_nm[eids]
                st.test_passed[eids] = (b >= S.BAND[0]) & (b <= S.BAND[1])
                st.phase[eids] = 2

        st.m = {"tip_depth_mm": depth, "tip_r_mm": r_yz, "tip_x_mm": rel[:, 0], "tip_z_mm": rel[:, 2],
                "e_notches": st.e, "drum_turn_deg": torch.rad2deg(st.drum_acc), "spoon_com_z_mm": com[:, 2]}
        st.scorer.update(Step(engaged & idle, clicked & idle, in_band & idle, st.test_passed.clone(), over & idle,
                              dropped & idle))
        if restore is not None:
            st.spoon.write_root_pose_to_sim(restore, env_ids=torch.arange(n, device=dev))
            st.spoon.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=torch.arange(n, device=dev))
            st._restore_spoon = None
        self._hazard_hooks(env, st)
        k = int(st.scorer.step[0])
        if DEBUG_EVERY and k % DEBUG_EVERY == 0:
            print(f"[drum-brake] k={k} {self._report(st, 0)}", flush=True)


@configclass
class EventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: DrumBrakeTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))
