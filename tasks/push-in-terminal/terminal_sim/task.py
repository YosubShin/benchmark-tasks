"""Push a ferruled wire into the named push-in terminal; pass a tug. Arena props and TaskBase.

Mechanism (terminal_sim/usd/terminal_props.py): the terminal strip is a fixed-base articulation. Each terminal's
clamping leg is a link on a revolute joint across its entry channel, pressed closed against a stop by a preloaded
drive (the spring). The wire is a second fixed-base articulation: anchored in the duct, a 40 cm chain of capsule
links (bending springs toward its routed shape), and the rigid ferruled end the robot handles. Insertion, clamping
and retention are geometry: the ferrule pushes the leg open and the leg's tip rides on it, pressing it against the
busbar; pulling back drags the tip toward the busbar (a wedge), so the wire comes out only with a much larger force
than it went in with.

Measured each step, in each terminal's wire frame W (u along the entry channel, v across it, y along the rail):
the ferrule tip's (u, y, v), the collar's depth (u of the wire end's root in the named terminal's frame) and the named
terminal's leg angle.

Function test (the success check): once the wire is seated (collar within SEATED_U of the mouth, leg riding on the
ferrule) and both grasp points have been clear of the wire for TEST_WAIT_S, the task pulls the wire end straight back
out of the entry, ramping to TUG_N over TUG_RAMP_S and holding it to TEST_PHASE_S. Passed if the wire moved out less
than TUG_HELD_M. ``force_function_test(env)`` starts the same tug at once (control runs).
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
from isaaclab.sensors import ContactSensorCfg
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

from terminal_sim.kit import GRASP_OFFSET_M, reset_robot_home
from terminal_sim.scorer import MILESTONES, PushInScorer, PushInStep
from terminal_sim.spec import N_TERMINALS, Layout, make_seed
from terminal_sim.usd import terminal_props as props

_ASSETS = Path(__file__).resolve().parent / "assets"

LIFT_M = 0.01
ENTRY_IN_M = 0.001
"""The ferrule's tip counts as in an entry once it is this far past the mouth, within the channel."""
ENTRY_LATERAL_M = props.PITCH / 2
SEATED_U = -0.0008
"""Seated: the collar within 0.8 mm of the mouth (it stops on it)..."""
LEAF_ENGAGED_DEG = 3.0
"""...and the named terminal's leg pushed open at least this much (its tip riding on the ferrule)."""
RELEASE_CLEAR_M = 0.06
DROP_Z_M = 0.03

TEST_WAIT_S = 1.0
TUG_RAMP_S = 0.5
TUG_HOLD_S = 1.5
TEST_PHASE_S = TUG_RAMP_S + TUG_HOLD_S
TUG_N = 50.0
"""Straight pull out of the entry: the conductor pull-out test force commonly cited for 2.5 mm2 (IEC 60999-1)."""
TUG_HELD_M = 0.0015

STRIP, WIRE, STAND = "terminal_strip", "wire", "wire_stand"


def u_hat(dev) -> torch.Tensor:
    return torch.tensor(props.u_hat(), device=dev)


class TerminalStrip(Object):
    def __init__(self):
        super().__init__(name=STRIP, usd_path=str(_ASSETS / "strip.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {"legs": ImplicitActuatorCfg(joint_names_expr=["leaf_.*_joint"], stiffness=props.SPRING_K,
                                                     damping=props.SPRING_DAMPING)}
        cfg.init_state.joint_pos = {"leaf_.*_joint": 0.0}
        return cfg


class Wire(Object):
    """Anchor in the duct, flexible cable, rigid ferruled end (``wire_end``)."""

    def __init__(self):
        super().__init__(name=WIRE, usd_path=str(_ASSETS / "wire.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {"cable": ImplicitActuatorCfg(joint_names_expr=["cable_j.*"], stiffness=props.CABLE_K,
                                                      damping=props.CABLE_DAMPING)}
        cfg.init_state.joint_pos = {"cable_j.*": 0.0}
        return cfg


class WireStand(Object):
    def __init__(self):
        super().__init__(name=STAND, usd_path=str(_ASSETS / "holder.usda"), object_type=ObjectType.BASE,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))


class _TaskState:
    def __init__(self, env: ManagerBasedEnv, task: "PushInTask"):
        n, dev = env.num_envs, env.device
        self.strip = env.scene[STRIP]
        self.wire = env.scene[WIRE]
        self.robot = env.scene["robot"]
        self.end_id = self.wire.find_bodies("wire_end")[0][0]
        self.leaf_joints = [self.strip.find_joints(f"leaf_{k}_joint")[0][0] for k in range(1, N_TERMINALS + 1)]
        self.flange_ids, _ = self.robot.find_bodies(["left_gripper", "right_gripper"], preserve_order=True)
        self.scorer = PushInScorer(n, dev)
        self.next_layout = make_seed(task.default_seed)
        self.layouts: dict[int, Layout] = {}
        self.last_reports: dict[int, dict] = {}
        self.target = torch.full((n,), self.next_layout.target, dtype=torch.long, device=dev)
        self.mouths = torch.tensor([props.mouth(k) for k in range(1, N_TERMINALS + 1)], device=dev)
        zb = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
        zl = lambda: torch.zeros(n, dtype=torch.long, device=dev)  # noqa: E731
        zf = lambda: torch.zeros(n, device=dev)  # noqa: E731
        self.seated_once, self.released, self.force_test, self.test_held, self.test_passed = zb(), zb(), zb(), zb(), zb()
        self.clear_steps, self.phase, self.phase_steps, self.wrong_n = zl(), zl(), zl(), zl()
        self.max_leaf_deg, self.max_depth, self.test_max_out, self.test_u0 = zf(), zf(), zf(), zf()
        self.max_push_n = zf()
        self.sensor = env.scene["wire_contacts"]
        self.seated_time = torch.full((n,), -1.0, device=dev)
        self.start_pos = torch.zeros(n, 3, device=dev)
        self.wait_steps = max(1, round(TEST_WAIT_S / env.step_dt))
        self.phase_len = max(1, round(TEST_PHASE_S / env.step_dt))

    def end_pose(self, origins):
        pos = self.wire.data.body_pos_w.torch[:, self.end_id] - origins
        return pos, self.wire.data.body_quat_w.torch[:, self.end_id]


class PushInTask(TaskBase):
    def __init__(self, default_seed: int = 0, episode_length_s: float = 60.0):
        super().__init__(episode_length_s=episode_length_s, task_description=make_seed(default_seed).instruction)
        self.default_seed = default_seed

    def state(self, env: ManagerBasedEnv) -> _TaskState:
        if getattr(env, "_task_state", None) is None:
            env._task_state = _TaskState(env, self)
        return env._task_state

    def force_function_test(self, env: ManagerBasedEnv) -> None:
        """Control runs: start the tug now, seated or not."""
        self.state(env).force_test[:] = True

    def episode_report(self, env: ManagerBasedEnv, env_id: int = 0, ended: bool = False) -> dict:
        st = self.state(env)
        if ended and env_id in st.last_reports:
            return st.last_reports.pop(env_id)
        return self._report(st, env_id)

    def _report(self, st: _TaskState, env_id: int) -> dict:
        r = st.scorer.report(env_id)
        r.update(
            target_terminal=int(st.target[env_id]), wrong_terminal_entered=int(st.wrong_n[env_id]),
            seated=bool(st.seated_once[env_id]), seated_time_s=round(float(st.seated_time[env_id]), 2),
            released=bool(st.released[env_id]), tug_phase=int(st.phase[env_id]), tug_n=TUG_N,
            tug_max_out_mm=round(float(st.test_max_out[env_id]) * 1000, 2), tug_held=bool(st.test_held[env_id]),
            tug_passed=bool(st.test_passed[env_id]), max_leg_open_deg=round(float(st.max_leaf_deg[env_id]), 2),
            max_tip_depth_mm=round(float(st.max_depth[env_id]) * 1000, 2),
            max_insert_resistance_n=round(float(st.max_push_n[env_id]), 1),
        )
        return r

    def get_scene_cfg(self):
        return SensorsCfg()

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
        for buf in (st.seated_once, st.released, st.force_test, st.test_held, st.test_passed):
            buf[env_ids] = False
        for buf in (st.clear_steps, st.phase, st.phase_steps, st.wrong_n):
            buf[env_ids] = 0
        for buf in (st.max_leaf_deg, st.max_depth, st.test_max_out, st.test_u0, st.max_push_n):
            buf[env_ids] = 0.0
        st.seated_time[env_ids] = -1.0
        st.target[env_ids] = layout.target
        st.wire.permanent_wrench_composer.reset(env_ids=env_ids)
        zero = torch.zeros(n, N_TERMINALS, device=dev)
        st.strip.write_joint_state_to_sim(zero, zero, joint_ids=st.leaf_joints, env_ids=env_ids)
        st.strip.set_joint_position_target(zero - props.LEAF_SIGN * props.SPRING_PRELOAD_RAD, joint_ids=st.leaf_joints,
                                           env_ids=env_ids)
        nj = st.wire.num_joints
        zj = torch.zeros(n, nj, device=dev)
        st.wire.write_joint_state_to_sim(zj, zj, env_ids=env_ids)
        st.wire.set_joint_position_target(zj, env_ids=env_ids)
        st.start_pos[env_ids] = torch.tensor(props.wire_start_root()[:3], device=dev)
        for env_id in env_ids.tolist():
            st.layouts[env_id] = layout

    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        origins = env.scene.env_origins
        n, dev = env.num_envs, env.device
        t = st.scorer.step.float() * env.step_dt
        uh, vh = u_hat(dev), torch.tensor(props.v_hat(), device=dev)

        pos, quat = st.end_pose(origins)
        tip = pos + math_utils.quat_apply(quat, torch.tensor([[props.SLEEVE_LEN, 0.0, 0.0]], device=dev).expand(n, 3))
        d = tip[:, None, :] - st.mouths[None]  # (N, T, 3)
        tu, tv, ty = (d * uh).sum(-1), (d * vh).sum(-1), d[..., 1]
        in_channel = (tu > ENTRY_IN_M) & (tu < 0.016) & (ty.abs() < ENTRY_LATERAL_M) & (tv.abs() < 0.004)
        tgt = st.target - 1
        ar = torch.arange(n, device=dev)
        in_target = in_channel[ar, tgt]
        others = in_channel.clone()
        others[ar, tgt] = False
        wrong = others.any(dim=1)
        st.wrong_n = torch.where((st.wrong_n == 0) & wrong, others.float().argmax(dim=1) + 1, st.wrong_n)
        st.max_depth = torch.where(in_target, torch.maximum(st.max_depth, tu[ar, tgt]), st.max_depth)

        root_u = ((pos - st.mouths[tgt]) * uh).sum(-1)
        fm = st.sensor.data.force_matrix_w
        fm = fm.torch if hasattr(fm, "torch") else fm
        push = -(fm.reshape(n, -1, 3).sum(dim=1) * uh).sum(-1)
        st.max_push_n = torch.where((st.phase == 0) & ~st.released & ~st.seated_once,
                                    torch.maximum(st.max_push_n, push), st.max_push_n)
        leaf = props.LEAF_SIGN * torch.rad2deg(st.strip.data.joint_pos.torch[:, st.leaf_joints])[ar, tgt]
        st.max_leaf_deg = torch.maximum(st.max_leaf_deg, leaf)
        seated = in_target & (root_u > SEATED_U) & (leaf > LEAF_ENGAGED_DEG)
        newly = seated & ~st.seated_once
        st.seated_time = torch.where(newly, t, st.seated_time)
        st.seated_once |= seated

        flange_pos = st.robot.data.body_pos_w.torch[:, st.flange_ids]
        flange_quat = st.robot.data.body_quat_w.torch[:, st.flange_ids]
        down = torch.tensor([0.0, 0.0, -GRASP_OFFSET_M], device=dev).expand_as(flange_pos)
        grasp = flange_pos + math_utils.quat_apply(flange_quat, down) - origins[:, None]
        mid = pos + math_utils.quat_apply(quat, torch.tensor([[-0.03, 0.0, 0.0]], device=dev).expand(n, 3))
        clear = ((grasp - mid[:, None]).norm(dim=-1) > RELEASE_CLEAR_M).all(dim=1)
        st.released = st.released | (seated & clear)

        # Tug test: pull straight back out of the entry (-u).
        ready = (st.phase == 0) & ((seated & clear) | st.force_test)
        st.clear_steps = torch.where(ready, st.clear_steps + 1, torch.zeros_like(st.clear_steps))
        start = ready & ((st.clear_steps >= st.wait_steps) | st.force_test)
        if start.any():
            ids = start.nonzero().flatten()
            st.phase[ids] = 1
            st.phase_steps[ids] = 0
            st.test_u0[ids] = root_u[ids]
        in_test = st.phase == 1
        if in_test.any():  # the pull ramps up over TUG_RAMP_S, then holds (a hand's tug, not a blow)
            ids = in_test.nonzero().flatten()
            scale = ((st.phase_steps[ids].float() + 1) * env.step_dt / TUG_RAMP_S).clamp(max=1.0)
            force = (-TUG_N * scale[:, None] * uh[None]).unsqueeze(1)
            st.wire.permanent_wrench_composer.reset(env_ids=ids)
            st.wire.permanent_wrench_composer.add_forces_and_torques_index(
                forces=force, torques=torch.zeros_like(force), body_ids=[st.end_id], env_ids=ids, is_global=True)
        st.phase_steps = torch.where(in_test, st.phase_steps + 1, st.phase_steps)
        st.test_max_out = torch.where(in_test, torch.maximum(st.test_max_out, st.test_u0 - root_u), st.test_max_out)
        end = in_test & (st.phase_steps >= st.phase_len)
        if end.any():
            ids = end.nonzero().flatten()
            st.wire.permanent_wrench_composer.reset(env_ids=ids)
            st.test_held[ids] = st.test_max_out[ids] < TUG_HELD_M
            st.test_passed[ids] = st.test_held[ids] & st.seated_once[ids] & seated[ids]
            st.phase[ids] = 2

        lifted = pos[:, 2] - st.start_pos[:, 2] > LIFT_M  # raised off its stand
        dropped = pos[:, 2] < DROP_Z_M
        st.scorer.update(PushInStep(lifted, in_target, seated & (st.phase == 0), st.test_passed.clone(),
                                    wrong & (st.phase == 0), dropped))


_STRIP_PATH = "{ENV_REGEX_NS}/" + STRIP


@configclass
class SensorsCfg:
    wire_contacts: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/" + WIRE + "/wire_end", update_period=0.0, history_length=1, debug_vis=False,
        filter_prim_paths_expr=[f"{_STRIP_PATH}/base"] + [f"{_STRIP_PATH}/leaf_{k}" for k in range(1, N_TERMINALS + 1)],
    )


@configclass
class EventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: PushInTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))
