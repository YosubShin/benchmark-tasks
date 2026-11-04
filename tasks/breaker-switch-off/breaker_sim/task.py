"""Switch the named iC60N off: the board articulation, the task (reset, per-step scoring) and the after-release check.

Mechanism (usd/board_props.py): each handle is a revolute link with hard stops at +-30 deg. Its snap
detent is contact geometry: a sprung plunger (prismatic link, spring = joint drive) presses on a cam sphere
behind the pivot, so the handle is pushed toward whichever stop it is nearer and flips once past centre. The
task never sets the handle's target. The O/I markings are printed on the handle's rocker drum, so what shows
in the face opening follows the handle (on the real part it follows the contacts; here contacts = handle).
`window_reads_off` is true when the green O-OFF band's centre is within WINDOW_DEG of facing the robot.

After-release check (the success check): once the named handle sits at its OFF stop and all four finger pads have
been more than RELEASE_CLEAR_M from its pivot for TEST_WAIT_S, the task pushes the handle toward ON with TEST_NM
for TEST_PHASE_S. Passed if it stayed at OFF (below -HELD_DEG), the window reads O-OFF and every other breaker is
still ON. Under the station harness it runs off camera (``test_phase`` in the task file). `force_function_test`
starts it now (the control run).
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

from breaker_sim.kit import Fingertips, reset_robot_home, reviewer_camera
from breaker_sim.scorer import MILESTONES, SwitchOffScorer, SwitchOffStep
from breaker_sim.spec import FACE_X, RAIL_Z, SwitchOffSeed, make_seed
from breaker_sim.usd import board_props as props

_ASSETS = Path(__file__).resolve().parent / "assets"

HANDLES = [f"handle_{k}" for k in range(6)]
Q_ON = props.Q_THROW
PLUNGER_TARGET = -props.PLUNGER_PRELOAD
"""Spring rest position (joint coordinate): the preload is the compression at a stop (joint position 0)."""
PLUNGER_DAMPING = 4.0
HANDLE_DAMPING = 2e-5

MOVED_RAD = math.radians(5.0)
"""Level 25: the named handle turned this far toward OFF."""
AT_OFF_RAD = math.radians(26.0)
RELEASE_CLEAR_M = 0.04
TEST_WAIT_S = 1.0
TEST_PHASE_S = 1.5
TEST_NM = 0.75 * props.handle_torque(props.Q_THROW)
"""Push toward ON: three quarters of the detent's holding torque at the stop."""
HELD_DEG = 25.0
WINDOW_DEG = 25.0


def window_reads_off(q: torch.Tensor) -> torch.Tensor:
    return (math.radians(props.FLAG_DEG) + q).abs() < math.radians(WINDOW_DEG)


def board_pose() -> tuple:
    return (FACE_X, 0.0, RAIL_Z, 0.0, 0.0, 0.0, 1.0)


class Board(Object):
    def __init__(self):
        p = board_pose()
        super().__init__(name="board", usd_path=str(_ASSETS / "board.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=p[:3], rotation_xyzw=p[3:]))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {
            "handles": ImplicitActuatorCfg(joint_names_expr=["handle_.*_joint"], stiffness=0.0, damping=HANDLE_DAMPING),
            "plungers": ImplicitActuatorCfg(joint_names_expr=["plunger_.*_joint"], stiffness=props.PLUNGER_K,
                                            damping=PLUNGER_DAMPING),
        }
        cfg.init_state.joint_pos = {"handle_.*_joint": Q_ON, "plunger_.*_joint": 0.0}
        return cfg


class _TaskState:
    def __init__(self, env: ManagerBasedEnv, task: "SwitchOffTask"):
        n, dev = env.num_envs, env.device
        self.board = env.scene["board"]
        self.robot = env.scene["robot"]
        self.tips = Fingertips(self.robot)
        self.handle_joints = self.board.find_joints([f"{h}_joint" for h in HANDLES], preserve_order=True)[0]
        self.plunger_joints = self.board.find_joints([f"plunger_{k}_joint" for k in range(6)], preserve_order=True)[0]
        self.handle_bodies = self.board.find_bodies(HANDLES, preserve_order=True)[0]
        self.scorer = SwitchOffScorer(n, dev)
        self.next_layout = make_seed(task.default_seed)
        self.layouts: dict[int, SwitchOffSeed] = {}
        self.last_reports: dict[int, dict] = {}
        self.env_paths = sim_utils.find_matching_prim_paths("/World/envs/env_.*")
        self.named = torch.zeros(n, dtype=torch.long, device=dev)
        self.pivots = torch.tensor([[FACE_X + p[0], p[1], RAIL_Z + p[2]] for p in map(props.pivot_local, range(6))],
                                   device=dev)
        zb = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
        zl = lambda: torch.zeros(n, dtype=torch.long, device=dev)  # noqa: E731
        self.force_test, self.held, self.passed, self.window_off_at_end = zb(), zb(), zb(), zb()
        self.end_ok_at_test = zb()
        self.clear_steps, self.phase, self.phase_steps = zl(), zl(), zl()
        self.test_max = torch.full((n,), -10.0, device=dev)
        self.min_q = torch.zeros(n, 6, device=dev)
        self.wait_steps = max(1, round(TEST_WAIT_S / env.step_dt))
        self.phase_len = max(1, round(TEST_PHASE_S / env.step_dt))


class SwitchOffTask(TaskBase):
    def __init__(self, default_seed: int = 0, episode_length_s: float = 35.0, reviewer_camera: bool = False):
        super().__init__(episode_length_s=episode_length_s, task_description=make_seed(default_seed).instruction)
        self.default_seed = default_seed
        self.reviewer_camera = reviewer_camera

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
        layout = st.layouts.get(env_id, st.next_layout)
        q = st.board.data.joint_pos.torch[env_id, st.handle_joints]
        r.update(
            named_way=layout.way, named_circuit=layout.circuit,
            handle_deg=[round(math.degrees(float(v)), 1) for v in q.tolist()],
            min_handle_deg=[round(math.degrees(float(v)), 1) for v in st.min_q[env_id].tolist()],
            windows=["O-OFF" if v < -math.radians(props.FLAG_DEG - WINDOW_DEG) else
                     ("I-ON" if v > math.radians(props.FLAG_DEG - WINDOW_DEG) else "-") for v in q.tolist()],
            function_test_phase=int(st.phase[env_id]),
            test_max_handle_deg=round(math.degrees(float(st.test_max[env_id])), 1),
            test_held=bool(st.held[env_id]), function_test_passed=bool(st.passed[env_id]),
            end_state_ok_at_test=bool(st.end_ok_at_test[env_id]),
        )
        return r

    def get_scene_cfg(self):
        return ReviewerCameraSceneCfg() if self.reviewer_camera else None

    def get_events_cfg(self):
        return SwitchOffEventsCfg(self)

    def get_termination_cfg(self) -> TaskTerminationCfg:
        sequence = [(self._milestone_predicate(m), 1.0 / len(MILESTONES)) for m in MILESTONES]
        return TaskTerminationCfg(timeout_s=self.episode_length_s,
                                  success=[CompletionCriteria(name="switch_off_milestones", predicate_sequence=sequence)])

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
        for buf in (st.force_test, st.held, st.passed, st.window_off_at_end, st.end_ok_at_test):
            buf[env_ids] = False
        for buf in (st.clear_steps, st.phase, st.phase_steps):
            buf[env_ids] = 0
        st.test_max[env_ids] = -10.0
        st.min_q[env_ids] = Q_ON
        st.named[env_ids] = layout.named_slot
        st.board.permanent_wrench_composer.reset(env_ids=env_ids)
        on = torch.full((n, 6), Q_ON, device=dev)
        st.board.write_joint_state_to_sim(on, torch.zeros_like(on), joint_ids=st.handle_joints, env_ids=env_ids)
        st.board.set_joint_position_target(on, joint_ids=st.handle_joints, env_ids=env_ids)
        zero = torch.zeros(n, 6, device=dev)
        st.board.write_joint_state_to_sim(zero, zero, joint_ids=st.plunger_joints, env_ids=env_ids)
        st.board.set_joint_position_target(zero + PLUNGER_TARGET, joint_ids=st.plunger_joints, env_ids=env_ids)
        for env_id in env_ids.tolist():
            st.layouts[env_id] = layout

    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        n, dev = env.num_envs, env.device
        origins = env.scene.env_origins
        q = st.board.data.joint_pos.torch[:, st.handle_joints]  # (N, 6)
        st.min_q = torch.minimum(st.min_q, q)

        named = torch.nn.functional.one_hot(st.named, 6).bool()
        pads = st.tips.points(origins)  # (N, 4, 3)
        d_piv = (pads[:, :, None] - st.pivots[None, None]).norm(dim=-1)  # (N, 4, 6)

        # Function test.
        idx = st.named.unsqueeze(1)
        q_named = q.gather(1, idx).squeeze(1)
        clear = (d_piv.gather(2, idx.unsqueeze(1).expand(n, 4, 1)).squeeze(2) > RELEASE_CLEAR_M).all(dim=1)
        at_off = q_named < -AT_OFF_RAD
        others_on = ((q > math.radians(HELD_DEG)) | named).all(dim=1)
        end_ok_live = (q_named < -math.radians(HELD_DEG)) & window_reads_off(q_named) & others_on
        ready = (st.phase == 0) & ((at_off & clear) | st.force_test)
        st.clear_steps = torch.where(ready, st.clear_steps + 1, torch.zeros_like(st.clear_steps))
        start = ready & ((st.clear_steps >= st.wait_steps) | st.force_test)
        if start.any():
            ids = start.nonzero().flatten()
            torque = torch.zeros(len(ids), 1, 3, device=dev)
            torque[..., 1] = TEST_NM
            bodies = [st.handle_bodies[int(st.named[i])] for i in ids.tolist()]
            for k, i in enumerate(ids.tolist()):
                st.board.permanent_wrench_composer.add_forces_and_torques_index(
                    forces=torch.zeros_like(torque[k:k + 1]), torques=torque[k:k + 1], body_ids=[bodies[k]],
                    env_ids=ids[k:k + 1], is_global=True)
            st.phase[ids] = 1
            st.phase_steps[ids] = 0
            st.end_ok_at_test[ids] = end_ok_live[ids]
        st.phase_steps = torch.where(st.phase > 0, st.phase_steps + 1, st.phase_steps)
        st.test_max = torch.where(st.phase == 1, torch.maximum(st.test_max, q_named), st.test_max)
        end = (st.phase == 1) & (st.phase_steps >= st.phase_len)
        if end.any():
            ids = end.nonzero().flatten()
            st.board.permanent_wrench_composer.reset(env_ids=ids)
            window_off = window_reads_off(q_named)
            st.held[ids] = st.test_max[ids] < -math.radians(HELD_DEG)
            st.window_off_at_end[ids] = window_off[ids]
            st.passed[ids] = st.held[ids] & window_off[ids] & others_on[ids] & (q_named[ids] < 0)
            st.phase[ids] = 2

        st.scorer.update(SwitchOffStep(
            named_moved=q_named < Q_ON - MOVED_RAD,
            named_past_centre=q_named < 0,
            end_state_ok=torch.where(st.phase > 0, st.end_ok_at_test, end_ok_live),
            test_passed=st.passed.clone(),
            other_off=((q < 0) & ~named).any(dim=1),
        ))


REVIEWER_EYE = (0.17, 0.13, 0.25)
REVIEWER_TARGET = (0.32, -0.012, 0.148)


@configclass
class ReviewerCameraSceneCfg:
    reviewer_cam: CameraCfg = reviewer_camera(REVIEWER_EYE, REVIEWER_TARGET)


@configclass
class SwitchOffEventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: SwitchOffTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))
