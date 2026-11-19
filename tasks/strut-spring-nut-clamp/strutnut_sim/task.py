"""Strut spring-nut task: Arena props and TaskBase (props: strutnut_sim/usd/props.py, layout: strutnut_sim/spec.py).

Mechanism (geometry and contact, plus one screw rule):
- The channel's lips are fixed colliders as the P1000 section shows them; the nut is a free body whose outline lets
  it through the slot only edge-on (19 mm wide in the 22.2 mm slot; 34.5 mm long, more than the slot) and turn only
  clockwise inside the channel (two leading corners rounded, two square ones stop it just past a quarter turn); its
  top has two grooves the lip returns drop into. Its spring is a sprung prismatic foot (SPRING_K) standing on the
  channel's back: it pushes the nut up against the lips once it is turned under them, and back out of the slot if
  it is not.
- The bolt is screwed into the nut (kit articulation joints ``thread``, prismatic, and ``turn``, revolute), coupled
  every control step like a screw: the thread's stiff drive targets START + PITCH x turns (clockwise seen from the
  head runs the bolt into the nut). Blocked, the drive's pull is the bolt's tension F = CLAMP_K (p - target). The
  turn joint carries thread friction as a Coulomb drive (a stiff spring whose anchor slips once it carries
  T = RUN_NM + NUT_K d F, the bolt formula): the free nut below the lips turns with the bolt; once its square
  corners stop against the walls (or its grooves hold it on the lips) the bolt turns in it.
- The fitting (P1064 square washer) is a free body with a clearance hole round the shank, held by geometry only.

After-release check (the success check), once every finger pad has stayed RELEASE_CLEAR from the fitting and the
bolt for TEST_WAIT_S without a break (the count restarts on any approach): the 75 conditions, the fitting seated and
square and the head down on it must hold; then the fitting is pulled along the channel with TEST_PULL_N (ramped over
TEST_RAMP_S, then held to TEST_S); it passes if it moved less than TEST_MAX_MOVE. ``force_test(env)`` starts the
same check at once (control runs).
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

from strutnut_sim import spec as S
from strutnut_sim.kit import Fingertips, reset_robot_home
from strutnut_sim.scorer import MILESTONES, Scorer, Step

_ASSETS = Path(__file__).resolve().parent / "assets"
IDENTITY = (0.0, 0.0, 0.0, 1.0)
HAZARD_TEST = os.environ.get("SNC_HAZARD_TEST", "")
DEBUG_EVERY = int(os.environ.get("SNC_DEBUG", "0") or 0)


class Stand(Object):
    def __init__(self):
        super().__init__(name="stand", usd_path=str(_ASSETS / "stand.usda"), object_type=ObjectType.BASE,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=IDENTITY))


class Kit(Object):
    def __init__(self):
        k = S.kit_start()
        super().__init__(name="kit", usd_path=str(_ASSETS / "kit.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=k["nut"], rotation_xyzw=IDENTITY))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {
            "spring": ImplicitActuatorCfg(joint_names_expr=["spring"], stiffness=S.SPRING_K, damping=S.SPRING_D),
            "thread": ImplicitActuatorCfg(joint_names_expr=["thread"], stiffness=S.CLAMP_K, damping=S.CLAMP_D,
                                          effort_limit_sim=S.CLAMP_MAX_N),
            "turn": ImplicitActuatorCfg(joint_names_expr=["turn"], stiffness=S.TURN_K, damping=S.TURN_D,
                                        effort_limit_sim=S.RUN_NM, armature=S.TURN_ARMATURE),
        }
        cfg.init_state.joint_pos = {"spring": 0.0, "thread": S.BOLT_L - S.kit_start()["e0"], "turn": 0.0}
        return cfg


class Fitting(Object):
    def __init__(self):
        k = S.kit_start()
        super().__init__(name="fitting", usd_path=str(_ASSETS / "fitting.usda"), object_type=ObjectType.RIGID,
                         initial_pose=Pose(position_xyz=k["fit"], rotation_xyzw=IDENTITY))


def all_props() -> list:
    return [Stand(), Kit(), Fitting()]


def _axis(quat, v):
    vv = torch.tensor(v, device=quat.device, dtype=quat.dtype).expand(quat.shape[0], 3)
    return math_utils.quat_apply(quat, vv)


class _TaskState:
    def __init__(self, env: ManagerBasedEnv, task: "StrutNutTask"):
        n, dev = env.num_envs, env.device
        self.robot = env.scene["robot"]
        self.kit = env.scene["kit"]
        self.fit = env.scene["fitting"]
        self.tips = Fingertips(self.robot)
        self.j_spring = self.kit.find_joints("spring")[0][0]
        self.j_thread = self.kit.find_joints("thread")[0][0]
        self.j_turn = self.kit.find_joints("turn")[0][0]
        self.b_nut = self.kit.find_bodies("nut")[0][0]
        self.b_bolt = self.kit.find_bodies("bolt")[0][0]
        self.b_foot = self.kit.find_bodies("foot")[0][0]
        self.scorer = Scorer(n, dev)
        self.last_reports: dict[int, dict] = {}
        zf = lambda: torch.zeros(n, device=dev)  # noqa: E731
        zb = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
        self.turn_prev, self.turn_acc, self.turn_hold, self.force, self.peak_force = zf(), zf(), zf(), zf(), zf()
        self.force_test, self.test_passed, self.ready_at_start = zb(), zb(), zb()
        self.clear_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.phase = torch.zeros(n, dtype=torch.long, device=dev)
        self.phase_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.test_start_y, self.test_move = zf(), zf()
        self.wait_steps = max(1, round(S.TEST_WAIT_S / env.step_dt))
        self.test_len = max(1, round(S.TEST_S / env.step_dt))
        self.ramp_len = max(1, round(S.TEST_RAMP_S / env.step_dt))
        self.m: dict[str, torch.Tensor] = {}
        self.hazard_done = False


class StrutNutTask(TaskBase):
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
        r.update(test_phase=int(st.phase[env_id]), ready_at_test_start=bool(st.ready_at_start[env_id]),
                 test_move_mm=round(float(st.test_move[env_id]) * 1000, 3), test_passed=bool(st.test_passed[env_id]),
                 bolt_tension_n=round(float(st.force[env_id]), 1), peak_tension_n=round(float(st.peak_force[env_id]), 1))
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
        for buf in (st.clear_steps, st.phase, st.phase_steps):
            buf[env_ids] = 0
        for buf in (st.turn_prev, st.turn_acc, st.turn_hold, st.force, st.peak_force, st.test_start_y, st.test_move):
            buf[env_ids] = 0.0
        st.hazard_done = False
        st.fit.permanent_wrench_composer.reset(env_ids=env_ids)
        k = S.kit_start()
        pose = torch.tensor([[*k["nut"], *IDENTITY]], device=dev).repeat(n, 1)
        pose[:, :3] += origins[env_ids]
        st.kit.write_root_pose_to_sim(pose, env_ids=env_ids)
        st.kit.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)
        q = torch.zeros(n, st.kit.num_joints, device=dev)
        q[:, st.j_thread] = S.BOLT_L - k["e0"]
        st.kit.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
        st.kit.set_joint_position_target(q.clone(), env_ids=env_ids)
        st.kit.write_joint_effort_limit_to_sim_index(limits=torch.full((n, 1), S.RUN_NM, device=dev),
                                                     joint_ids=[st.j_turn], env_ids=env_ids)
        fpose = torch.tensor([[*k["fit"], *IDENTITY]], device=dev).repeat(n, 1)
        fpose[:, :3] += origins[env_ids]
        st.fit.write_root_pose_to_sim(fpose, env_ids=env_ids)
        st.fit.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)

    def _screw(self, st: _TaskState) -> None:
        jp = st.kit.data.joint_pos.torch
        q_raw = jp[:, st.j_turn]
        d = torch.remainder(q_raw - st.turn_prev + math.pi, 2 * math.pi) - math.pi
        st.turn_prev = q_raw.clone()
        st.turn_acc = st.turn_acc + d
        turn = st.turn_acc
        p = jp[:, st.j_thread]
        q0 = S.BOLT_L - S.START_TURNS * S.PITCH
        lo, hi = S.BOLT_L - S.NUT_T - 0.004, S.BOLT_L
        # Right-hand thread: clockwise seen from the head (negative about the bolt's +z) runs the bolt into the nut.
        target = (q0 + S.PITCH * turn / (2 * math.pi)).clamp(lo, hi)
        force = (S.CLAMP_K * (p - target)).clamp(min=0.0, max=S.CLAMP_MAX_N)
        st.force = force
        st.peak_force = torch.maximum(st.peak_force, force)
        limit = S.RUN_NM + S.NUT_K * S.BOLT_D * force
        st.turn_hold = torch.minimum(torch.maximum(st.turn_hold, turn - limit / S.TURN_K), turn + limit / S.TURN_K)
        st.kit.write_joint_effort_limit_to_sim_index(limits=limit.unsqueeze(1), joint_ids=[st.j_turn])
        st.kit.set_joint_position_target((q_raw + st.turn_hold - turn).unsqueeze(1), joint_ids=[st.j_turn])
        st.kit.set_joint_position_target(target.unsqueeze(1), joint_ids=[st.j_thread])

    def _hazard_hooks(self, env, st: _TaskState) -> None:
        """Test harness (SNC_HAZARD_TEST, empty in scored runs): move a part as a hand would, once."""
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
        if case == "drop":
            # The fitting falls off the stand onto the table in front of it, then is put back where it was.
            pose = torch.cat([st.fit.data.root_pos_w.torch, st.fit.data.root_quat_w.torch], dim=-1).clone()
            low = pose.clone()
            low[:, 0] = origins[:, 0] + 0.16
            low[:, 2] = origins[:, 2] + 0.004
            st.fit.write_root_pose_to_sim(low, env_ids=ids)
            st._restore_fit = pose
        elif case == "nutoff":
            # The bolt is backed out of the nut to its last thread (as if turned anticlockwise four turns).
            st.turn_acc[:] = 2 * math.pi * (S.START_TURNS - 0.2)
        print(f"[strut-nut] HAZARD TEST {HAZARD_TEST} applied at step {step} (milestone {m})", flush=True)

    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        origins = env.scene.env_origins
        n, dev = env.num_envs, env.device
        restore = getattr(st, "_restore_fit", None)  # a test drop is measured on this step, then undone
        self._screw(st)
        idle = st.phase == 0

        body_pos = st.kit.data.body_pos_w.torch - origins[:, None]
        body_quat = st.kit.data.body_quat_w.torch
        nut_p, nut_q = body_pos[:, st.b_nut], body_quat[:, st.b_nut]
        bolt_p, bolt_q = body_pos[:, st.b_bolt], body_quat[:, st.b_bolt]
        fit_p = st.fit.data.root_pos_w.torch - origins
        fit_q = st.fit.data.root_quat_w.torch
        jp = st.kit.data.joint_pos.torch
        engaged = (S.BOLT_L - jp[:, st.j_thread]).clamp(min=0.0)
        turns = engaged / S.PITCH

        # Nut seated: long axis across the channel (along x), top at the seated height, centred, level.
        long_ax = _axis(nut_q, (0.0, 1.0, 0.0))
        up_n = _axis(nut_q, (0.0, 0.0, 1.0))
        across = torch.rad2deg(torch.acos(long_ax[:, 0].abs().clamp(max=1.0)))
        nut_tilt = torch.rad2deg(torch.acos(up_n[:, 2].clamp(-1.0, 1.0)))
        seat_err = nut_p[:, 2] - S.SEATED_TOP
        centre_err = (nut_p[:, 0] - S.CH_X).abs()
        seated = (across <= S.NUT_ACROSS_DEG) & (seat_err.abs() <= S.SEAT_TOL) & (centre_err <= S.CENTRE_TOL) & \
            (nut_tilt <= 5.0)
        mark_err = bolt_p[:, 1] - S.MARK_Y
        at_mark = mark_err.abs() <= S.MARK_TOL
        engaged_ok = turns >= S.ENGAGED_TURNS

        # Fitting: flat on the channel, square to it, the head down on it.
        up_f = _axis(fit_q, (0.0, 0.0, 1.0))
        fx = _axis(fit_q, (1.0, 0.0, 0.0))
        fit_tilt = torch.rad2deg(torch.acos(up_f[:, 2].clamp(-1.0, 1.0)))
        yaw = torch.rad2deg(torch.atan2(fx[:, 1], fx[:, 0]))
        square_err = (torch.remainder(yaw + 45.0, 90.0) - 45.0).abs()
        fit_gap = fit_p[:, 2] - S.CH_TOP
        head_gap = bolt_p[:, 2] - (fit_p[:, 2] + S.FIT_T)
        fit_ok = (fit_gap.abs() <= S.FIT_FLAT_TOL) & (fit_tilt <= S.FIT_TILT_DEG) & (square_err <= S.FIT_SQUARE_DEG) & \
            (head_gap <= S.HEAD_GAP_TOL)
        ready = seated & at_mark & engaged_ok & fit_ok

        # Hazards.
        hx, hy = S.HOLDER_XY

        def fell(p):
            outside = ((p[:, 0] - hx).abs() > 0.035) | ((p[:, 1] - hy).abs() > 0.035)
            off_channel = ((p[:, 0] - S.CH_X).abs() > S.CH_W / 2 + 0.004) | (p[:, 1] < S.CH_Y[0]) | (p[:, 1] > S.CH_Y[1])
            return outside & off_channel & (p[:, 2] < S.DROP_Z)

        dropped = fell(fit_p) | fell(nut_p - torch.tensor([0.0, 0.0, S.NUT_T], device=dev)) | fell(bolt_p)
        nut_off = turns < S.NUT_OFF_TURNS

        # Finger pads: distance to the fitting (its plate) and to the bolt's head.
        pads = st.tips.points(origins)
        d_head = (pads - (bolt_p + _axis(bolt_q, (0.0, 0.0, S.HEAD_H / 2)))[:, None]).norm(dim=-1).min(dim=1).values
        rel = math_utils.quat_apply_inverse(fit_q[:, None].expand(n, 4, 4).reshape(-1, 4),
                                            (pads - fit_p[:, None]).reshape(-1, 3)).reshape(n, 4, 3)
        lo = torch.tensor([-S.FIT_W / 2, -S.FIT_W / 2, 0.0], device=dev)
        hi = torch.tensor([S.FIT_W / 2, S.FIT_W / 2, S.FIT_T], device=dev)
        d_fit = torch.maximum(lo - rel, rel - hi).clamp(min=0.0).norm(dim=-1).min(dim=1).values
        clear = (d_head > S.RELEASE_CLEAR) & (d_fit > S.RELEASE_CLEAR)

        # After-release check.
        go = idle & ((st.scorer.milestone >= 75) | st.force_test) & (clear | st.force_test)
        st.clear_steps = torch.where(go, st.clear_steps + 1, torch.zeros_like(st.clear_steps))
        start = go & ((st.clear_steps >= st.wait_steps) | st.force_test)
        if start.any():
            st.ready_at_start = torch.where(start, ready, st.ready_at_start)
            st.phase = torch.where(start, torch.ones_like(st.phase), st.phase)
            st.phase_steps = torch.where(start, torch.zeros_like(st.phase_steps), st.phase_steps)
            st.test_start_y = torch.where(start, fit_p[:, 1], st.test_start_y)
        testing = st.phase == 1
        if testing.any():
            ids = testing.nonzero().flatten()
            ramp = ((st.phase_steps[ids].to(torch.float32) + 1) / st.ramp_len).clamp(max=1.0)
            forces = torch.zeros(len(ids), 1, 3, device=dev)
            forces[:, 0, 1] = S.TEST_PULL_N * ramp
            centre = fit_p[ids] + _axis(fit_q[ids], (0.0, 0.0, S.FIT_T / 2)) + origins[ids]
            st.fit.permanent_wrench_composer.set_forces_and_torques_index(
                forces=forces, torques=torch.zeros_like(forces), positions=centre[:, None], body_ids=[0], env_ids=ids,
                is_global=True)
            st.test_move = torch.where(testing, torch.maximum(st.test_move, (fit_p[:, 1] - st.test_start_y).abs()),
                                       st.test_move)
            st.phase_steps = torch.where(testing, st.phase_steps + 1, st.phase_steps)
            end = testing & ((st.phase_steps >= st.test_len) | (st.test_move > 0.02))
            if end.any():
                eids = end.nonzero().flatten()
                st.fit.permanent_wrench_composer.reset(env_ids=eids)
                st.test_passed[eids] = (st.test_move[eids] < S.TEST_MAX_MOVE) & st.ready_at_start[eids]
                st.phase[eids] = 2

        st.m = {"nut_across_err_deg": across, "nut_seat_err_mm": seat_err, "nut_centre_err_mm": centre_err,
                "nut_tilt_deg": nut_tilt, "bolt_from_mark_mm": mark_err, "engaged_turns": turns,
                "fit_gap_mm": fit_gap, "fit_tilt_deg": fit_tilt, "fit_square_err_deg": square_err,
                "head_gap_mm": head_gap, "pad_to_head_mm": d_head, "pad_to_fitting_mm": d_fit,
                "nut_x_mm": nut_p[:, 0], "nut_y_mm": nut_p[:, 1], "nut_z_mm": nut_p[:, 2],
                "spring_mm": jp[:, st.j_spring], "turn_acc_rev": st.turn_acc / (2 * math.pi)}
        st.scorer.update(Step(seated & idle, at_mark & idle, engaged_ok & idle, st.test_passed.clone(),
                              dropped & idle, nut_off & idle))
        if restore is not None:
            st.fit.write_root_pose_to_sim(restore, env_ids=torch.arange(n, device=dev))
            st._restore_fit = None
        self._hazard_hooks(env, st)
        k = int(st.scorer.step[0])
        if DEBUG_EVERY and k % DEBUG_EVERY == 0:
            print(f"[strut-nut] k={k} {self._report(st, 0)}", flush=True)


@configclass
class EventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: StrutNutTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))
