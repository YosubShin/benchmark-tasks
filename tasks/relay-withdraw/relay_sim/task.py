"""Withdraw the named drawout relay's plug-in unit: props, mechanism, hazard watch and after-release check.

Mechanism (geometry unless marked RULE; see relay_sim/usd/relay_props.py):
- Each plug-in unit is a free articulation (body + bail handle on a revolute joint, 0-92 deg) sliding in a static
  case. Handle down: a lug on each hub hangs just behind a fixed catch, so a pull on the unit drives the lug into the
  catch below the pivot, which presses the handle onto its down stop: the unit stays in (latched, geometry, 5 mm of
  play). Lifting the handle swings the lug over the catch and a tooth onto a fixed cam face behind the pivot, which
  pushes the unit ~7 mm out (geometry).
- RULE: rear connector retention. While a unit is within CONN_TRAVEL of seated and its connectors are engaged, a
  force pulls it back in (CONN_PRELOAD_N rising to CONN_N at CONN_TRAVEL); past CONN_TRAVEL the connectors are parted
  and the force is gone until the unit is pushed home again. The real force is not published.
- The handle has no spring: gravity holds it down; the robot holds it up.
- Nothing stops the unit once out: it slides on the case floor and drops off the front edge unless held.

After-release check (the success test, run off camera by the station harness): once the named unit rests upright on
the mat, wholly out of the case, with both grippers clear for TEST_WAIT_S, the task pulls the other unit outward with
TEST_PULL_N for TEST_PHASE_S. Passed if that unit moved less than DECOY_HELD_M (still latched) and the named unit is
still on the mat. ``force_control(env)`` runs the control instead: with the robot idle, pull the named unit with its
handle down (latched: it must not come out), let go, lift its handle by a torque (the cam ejects it), pull again.
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

from relay_sim.kit import Fingertips, reset_robot_home, reviewer_camera, write_pose
from relay_sim.scorer import MILESTONES, RelayScorer, RelayStep
from relay_sim.spec import BODY_Z, EPISODE_SECONDS, NAMED_SLOT, PANEL_X, SLOT_Y, instruction
from relay_sim.usd import relay_props as props

_ASSETS = Path(__file__).resolve().parent / "assets"

CONN_TRAVEL = 0.006
"""Connector engagement: parted beyond this withdrawal (the cam stroke is ~7 mm; ABB: 'about 7 mm')."""
CONN_N = 30.0
"""Connector retention at full engagement travel (RULE; inferred, to measure on a real unit)."""
CONN_PRELOAD_N = 10.0
"""Retention at the start of travel, so a unit let go part-way draws back home against the case-floor friction
(~6 N at 2.1 kg) instead of sticking proud (RULE; inferred)."""
HANDLE_DAMPING = 0.02

LIFTED_DEG = 45.0
OUT_MARGIN_M = 0.003
DECOY_LIFT_DEG = 15.0
DECOY_MOVE_M = 0.003
TOUCH_MARGIN_M = 0.006
DROP_TILT_RAD = math.radians(35.0)
DROP_Z_M = props.MAT_TOP - 0.02
REST_TILT_RAD = math.radians(8.0)
REST_Z_TOL = (0.004, 0.010)
CLEAR_M = 0.015

TEST_WAIT_S = 2.0
TEST_PHASE_S = 1.5
TEST_PULL_N = 30.0
"""A firm tug on the decoy: as much as the connector retention (inferred)."""
DECOY_HELD_M = 0.0058
"""The decoy counts as still latched if the test pull moved it less than this: the latch's 5 mm of play plus the
0.5 mm seat gap, short of the 6 mm at which its connectors part."""

CTL_PULL_N = 40.0
CTL_LIFT_NM = 1.5
CTL_PULL_OUT_N = 15.0
CTL_OUT_M = 0.10
CTL_PHASE_S = 2.0

UNIT_FRONT_X = -(props.BEZEL[0] + 0.045)
"""Unit-frame bounding box of the bezel and the handle at rest, for touch checks."""


def unit_name(slot: int) -> str:
    return f"relay_unit_{slot}"


def seated_pose(slot: int) -> tuple:
    return props.seated_origin(slot) + (0.0, 0.0, 0.0, 1.0)


class Panel(Object):
    def __init__(self):
        super().__init__(name="relay_panel", usd_path=str(_ASSETS / "panel.usda"), object_type=ObjectType.BASE,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))


class Unit(Object):
    def __init__(self, slot: int):
        p = seated_pose(slot)
        super().__init__(name=unit_name(slot), usd_path=str(_ASSETS / "unit.usda"),
                         object_type=ObjectType.ARTICULATION, initial_pose=Pose(position_xyz=p[:3], rotation_xyzw=p[3:]))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {"handle": ImplicitActuatorCfg(joint_names_expr=["handle_joint"], stiffness=0.0,
                                                       damping=HANDLE_DAMPING)}
        cfg.init_state.joint_pos = {"handle_joint": 0.0}
        return cfg


class _TaskState:
    def __init__(self, env: ManagerBasedEnv, task: "RelayTask"):
        n, dev = env.num_envs, env.device
        self.units = [env.scene[unit_name(s)] for s in (0, 1)]
        self.robot = env.scene["robot"]
        self.tips = Fingertips(self.robot)
        u = self.units[0]
        self.handle_joint = u.find_joints("handle_joint")[0][0]
        self.body_id = u.find_bodies("body")[0][0]
        self.handle_id = u.find_bodies("handle")[0][0]
        self.scorer = RelayScorer(n, dev)
        self.last_reports: dict[int, dict] = {}
        zb = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
        zl = lambda: torch.zeros(n, dtype=torch.long, device=dev)  # noqa: E731
        zf = lambda: torch.zeros(n, device=dev)  # noqa: E731
        self.engaged = torch.ones(n, 2, dtype=torch.bool, device=dev)
        self.parted_by_handle = zb()
        self.test_passed, self.decoy_held, self.force_ctl = zb(), zb(), zb()
        self.clear_steps, self.phase, self.phase_steps, self.ctl_phase, self.ctl_steps = zl(), zl(), zl(), zl(), zl()
        self.decoy_max, self.max_s, self.max_phi = zf(), zf(), zf()
        self.ctl_max_s = torch.zeros(n, 4, device=dev)
        self.decoy_ref = torch.zeros(n, 3, device=dev)
        self.wait_steps = max(1, round(TEST_WAIT_S / env.step_dt))
        self.phase_len = max(1, round(TEST_PHASE_S / env.step_dt))
        self.ctl_len = max(1, round(CTL_PHASE_S / env.step_dt))
        self.named = 0
        self.debug: dict[str, torch.Tensor] = {}


class RelayTask(TaskBase):
    def __init__(self, default_seed: int = 0, episode_length_s: float = EPISODE_SECONDS, reviewer_camera: bool = False):
        super().__init__(episode_length_s=episode_length_s, task_description=instruction())
        self.default_seed = default_seed
        self.reviewer_camera = reviewer_camera

    def state(self, env: ManagerBasedEnv) -> _TaskState:
        if getattr(env, "_relay_state", None) is None:
            env._relay_state = _TaskState(env, self)
        return env._relay_state

    def force_control(self, env: ManagerBasedEnv) -> None:
        """Control demo: pull the named unit by its body with its handle down (40 N, 2 s), let go (2 s), lift the
        handle by a 1.5 N m torque (2 s), then keep it lifted and pull again (15 N until the unit is 10 cm out; 2 s)."""
        self.state(env).force_ctl[:] = True

    def episode_report(self, env: ManagerBasedEnv, env_id: int = 0, ended: bool = False) -> dict:
        st = self.state(env)
        if ended and env_id in st.last_reports:
            return st.last_reports.pop(env_id)
        return self._report(st, env_id)

    def _report(self, st: _TaskState, env_id: int) -> dict:
        r = st.scorer.report(env_id)
        r.update(
            named_slot=st.named, connectors_parted_by_handle=bool(st.parted_by_handle[env_id]),
            max_withdrawal_m=round(float(st.max_s[env_id]), 4),
            max_handle_deg=round(math.degrees(float(st.max_phi[env_id])), 1),
            function_test_phase=int(st.phase[env_id]), decoy_test_max_move_mm=round(1000 * float(st.decoy_max[env_id]), 2),
            decoy_held=bool(st.decoy_held[env_id]), function_test_passed=bool(st.test_passed[env_id]),
        )
        if bool(st.force_ctl[env_id]):
            r["control_max_withdrawal_mm"] = {
                k: round(1000 * float(st.ctl_max_s[env_id, i]), 2)
                for i, k in enumerate(("pull_handle_down", "lift_handle", "pull_handle_up"))
            }
        return r

    def get_scene_cfg(self):
        return ReviewerCameraSceneCfg() if self.reviewer_camera else None

    def get_events_cfg(self):
        return RelayEventsCfg(self)

    def get_termination_cfg(self) -> TaskTerminationCfg:
        sequence = [(self._milestone_predicate(m), 0.25) for m in MILESTONES]
        return TaskTerminationCfg(timeout_s=self.episode_length_s,
                                  success=[CompletionCriteria(name="relay_milestones", predicate_sequence=sequence)])

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def _milestone_predicate(self, milestone: int):
        def predicate(env: ManagerBasedEnv) -> torch.Tensor:
            return self.state(env).scorer.milestone >= milestone

        predicate.__name__ = f"milestone_{milestone}"
        return predicate

    # Events ----------------------------------------------------------------------------------------
    def reset(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        reset_robot_home(env, env_ids)
        for env_id in env_ids.tolist():
            if int(st.scorer.step[env_id]) > 0:
                st.last_reports[env_id] = self._report(st, env_id)
        st.named = NAMED_SLOT
        dev, n = env.device, len(env_ids)
        origins = env.scene.env_origins
        st.scorer.reset(env_ids)
        for buf in (st.parted_by_handle, st.test_passed, st.decoy_held, st.force_ctl):
            buf[env_ids] = False
        for buf in (st.clear_steps, st.phase, st.phase_steps, st.ctl_phase, st.ctl_steps):
            buf[env_ids] = 0
        for buf in (st.decoy_max, st.max_s, st.max_phi):
            buf[env_ids] = 0.0
        st.ctl_max_s[env_ids] = 0.0
        st.engaged[env_ids] = True
        zero = torch.zeros(n, 1, device=dev)
        for s, unit in enumerate(st.units):
            unit.permanent_wrench_composer.reset(env_ids=env_ids)
            unit.write_joint_state_to_sim(zero, zero, joint_ids=[st.handle_joint], env_ids=env_ids)
            unit.set_joint_position_target(zero, joint_ids=[st.handle_joint], env_ids=env_ids)
            write_pose(unit, seated_pose(s), env_ids, origins)
        st.decoy_ref[env_ids] = torch.tensor(seated_pose(1 - st.named)[:3], device=dev)

    def _unit_state(self, st: _TaskState, unit, origins):
        pos = unit.data.root_pos_w.torch - origins
        quat = unit.data.root_quat_w.torch
        phi = unit.data.joint_pos.torch[:, st.handle_joint]
        return pos, quat, phi

    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        origins = env.scene.env_origins
        n, dev = env.num_envs, env.device
        named, decoy = st.units[st.named], st.units[1 - st.named]
        seat_x = props.seated_origin(0)[0]

        # Connector retention (RULE) on both units, plus the control demo's and the test's pulls.
        forces = torch.zeros(n, 2, 1, 3, device=dev)
        s_all = []
        for k, unit in enumerate(st.units):
            pos, quat, phi = self._unit_state(st, unit, origins)
            s = seat_x - pos[:, 0]
            s_all.append(s)
            st.engaged[:, k] = torch.where(s > CONN_TRAVEL, False, st.engaged[:, k])
            st.engaged[:, k] = torch.where(s < 0.001, True, st.engaged[:, k])
            f = torch.where(st.engaged[:, k] & (s > 0),
                            CONN_PRELOAD_N + (CONN_N - CONN_PRELOAD_N) * (s / CONN_TRAVEL).clamp(0.0, 1.0), 0.0)
            forces[:, k, 0, 0] = f
        pos_n, quat_n, phi_n = self._unit_state(st, named, origins)
        pos_d, quat_d, phi_d = self._unit_state(st, decoy, origins)
        s_n = s_all[st.named]
        st.max_s = torch.maximum(st.max_s, s_n)
        st.max_phi = torch.maximum(st.max_phi, phi_n)
        newly_parted = ~st.engaged[:, st.named] & ~st.parted_by_handle & (phi_n > math.radians(60.0))
        st.parted_by_handle |= newly_parted

        # Control demo (robot idle): pull with the handle down, lift the handle, pull again.
        handle_torque = torch.zeros(n, 1, 3, device=dev)
        ctl = st.force_ctl & (st.ctl_phase < 5)
        start = ctl & (st.ctl_phase == 0)
        st.ctl_phase = torch.where(start, 1, st.ctl_phase)
        st.ctl_steps = torch.where(ctl, st.ctl_steps + 1, st.ctl_steps)
        for i, p in enumerate((1, 3, 4)):
            on = st.ctl_phase == p
            st.ctl_max_s[:, i] = torch.where(on, torch.maximum(st.ctl_max_s[:, i], s_n), st.ctl_max_s[:, i])
        forces[:, st.named, 0, 0] -= torch.where(st.ctl_phase == 1, CTL_PULL_N, 0.0)
        # Phase 4: a lighter pull, only until the unit is 10 cm out (then nothing holds it: it slides off the case
        # floor and drops, as the manual warns).
        forces[:, st.named, 0, 0] -= torch.where((st.ctl_phase == 4) & (s_n < CTL_OUT_M), CTL_PULL_OUT_N, 0.0)
        handle_torque[:, 0, 1] = torch.where((st.ctl_phase == 3) | (st.ctl_phase == 4), CTL_LIFT_NM, 0.0)
        adv = ctl & (st.ctl_steps >= st.ctl_len)
        st.ctl_phase = torch.where(adv, st.ctl_phase + 1, st.ctl_phase)
        st.ctl_steps = torch.where(adv, 0, st.ctl_steps)

        # Pads and grasp points.
        pads = st.tips.points(origins)  # (N, 4, 3)

        def local(points, pos, quat):
            m = points.shape[1]
            return math_utils.quat_apply_inverse(quat.unsqueeze(1).expand(n, m, 4).reshape(-1, 4),
                                                 (points - pos[:, None]).reshape(-1, 3)).reshape(n, m, 3)

        def in_box(p, lo, hi, margin):
            lo_t = torch.tensor(lo, device=dev) - margin
            hi_t = torch.tensor(hi, device=dev) + margin
            return ((p > lo_t) & (p < hi_t)).all(dim=-1)

        hy = props.BAR_SPAN / 2
        front_lo = (UNIT_FRONT_X, -hy, props.BEZEL_Z0 - 0.015)
        front_hi = (0.0, hy, props.BEZEL_Z0 + props.BEZEL[2])
        decoy_touched = in_box(local(pads, pos_d, quat_d), front_lo, front_hi, TOUCH_MARGIN_M).any(dim=1)
        y_n = SLOT_Y[st.named]
        case_lo = (PANEL_X + 0.005, y_n - props.CASE_W / 2 + props.WALL, BODY_Z)
        case_hi = (PANEL_X + props.CASE_DEPTH, y_n + props.CASE_W / 2 - props.WALL, BODY_Z + props.CASE_H - 0.006)
        case_reached = in_box(pads, case_lo, case_hi, 0.0).any(dim=1)

        # Named unit: out of the case, on the mat, tilt, clear of the fingers.
        up = math_utils.quat_apply(quat_n, torch.tensor([[0.0, 0.0, 1.0]], device=dev).expand(n, 3))
        tilt = torch.acos(up[:, 2].clamp(-1.0, 1.0))
        d, w, h = props.BODY
        corners = torch.tensor([[x, y, 0.0] for x in (props.BEZEL[0] * -1, d) for y in (-w / 2, w / 2)], device=dev)
        cw = pos_n[:, None] + math_utils.quat_apply(quat_n.unsqueeze(1).expand(n, 4, 4).reshape(-1, 4),
                                                     corners.expand(n, 4, 3).reshape(-1, 3)).reshape(n, 4, 3)
        back_x = (pos_n + math_utils.quat_apply(quat_n, torch.tensor([[d, 0.0, h / 2]], device=dev).expand(n, 3)))[:, 0]
        out = back_x < PANEL_X - OUT_MARGIN_M
        bottom = cw[..., 2].min(dim=1).values
        on_mat = (out & (tilt < REST_TILT_RAD) & (pos_n[:, 2] > props.MAT_TOP - REST_Z_TOL[0])
                  & (pos_n[:, 2] < props.MAT_TOP + REST_Z_TOL[1]))
        lp = local(pads, pos_n, quat_n)
        near_unit = in_box(lp, (UNIT_FRONT_X - 0.02, -hy, props.BEZEL_Z0 - 0.02),
                           (d, hy, props.BEZEL_Z0 + props.BEZEL[2]), CLEAR_M).any(dim=1)
        dropped = (tilt > DROP_TILT_RAD) | (bottom < DROP_Z_M)

        # Decoy: unlatched or moved (before the function test).
        decoy_move = (pos_d - st.decoy_ref).norm(dim=-1)
        before_test = st.phase == 0
        decoy_unlatched = before_test & ((phi_d > math.radians(DECOY_LIFT_DEG)) | (decoy_move > DECOY_MOVE_M))

        # Function test.
        ready = before_test & on_mat & ~near_unit & (st.scorer.milestone >= 75) & ~st.force_ctl
        st.clear_steps = torch.where(ready, st.clear_steps + 1, torch.zeros_like(st.clear_steps))
        start = ready & (st.clear_steps >= st.wait_steps)
        st.phase = torch.where(start, 1, st.phase)
        st.phase_steps = torch.where(st.phase == 1, st.phase_steps + 1, st.phase_steps)
        testing = st.phase == 1
        forces[:, 1 - st.named, 0, 0] -= torch.where(testing, TEST_PULL_N, 0.0)
        st.decoy_max = torch.where(testing, torch.maximum(st.decoy_max, decoy_move), st.decoy_max)
        end = testing & (st.phase_steps >= st.phase_len)
        st.decoy_held = torch.where(end, (st.decoy_max < DECOY_HELD_M) & (phi_d < math.radians(DECOY_LIFT_DEG)),
                                    st.decoy_held)
        st.test_passed = torch.where(end, st.decoy_held & on_mat, st.test_passed)
        st.phase = torch.where(end, 2, st.phase)

        # One set call per unit (a set clears the env's whole buffer): body force, handle torque.
        for k, unit in enumerate(st.units):
            torque = handle_torque if k == st.named else torch.zeros_like(handle_torque)
            f = torch.cat([forces[:, k], torch.zeros_like(torque)], dim=1)
            t = torch.cat([torch.zeros_like(torque), torque], dim=1)
            unit.permanent_wrench_composer.set_forces_and_torques_index(
                forces=f, torques=t, body_ids=[st.body_id, st.handle_id], is_global=True)

        st.debug = {"s_named": s_n, "phi_named_deg": phi_n * 180 / math.pi, "tilt_deg": tilt * 180 / math.pi,
                    "z_named": pos_n[:, 2], "back_x": back_x, "s_decoy": s_all[1 - st.named],
                    "phi_decoy_deg": phi_d * 180 / math.pi}
        st.scorer.update(RelayStep(phi_n > math.radians(LIFTED_DEG), st.parted_by_handle.clone(), out & ~dropped,
                                 st.test_passed.clone(), decoy_unlatched, decoy_touched & before_test,
                                 case_reached, dropped))


REVIEWER_EYE = (0.33, -0.42, 0.30)
REVIEWER_TARGET = (0.25, 0.08, 0.10)


@configclass
class ReviewerCameraSceneCfg:
    reviewer_cam: CameraCfg = reviewer_camera(REVIEWER_EYE, REVIEWER_TARGET)


@configclass
class RelayEventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: RelayTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))
