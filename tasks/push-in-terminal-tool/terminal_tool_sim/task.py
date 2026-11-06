"""Land a bare fine-stranded wire in a push-in terminal with the operating tool; pass a tug. Arena props and TaskBase.

Mechanism (terminal_tool_sim/usd/terminal_props.py): the terminal strip is a fixed-base articulation. Each terminal's
clamping leg is a link on a revolute joint across its entry channel, pressed closed against a stop by a preloaded
drive (the spring); a tab on the leg reaches the operating slot under the entry. The wire is a second fixed-base
articulation: anchored in the duct, a 40 cm chain of capsule links, the rigid insulated end the robot handles and its
10 mm stripped end (four bare-strand links on bending joints, on a sprung slide that lets the bundle bunch up under
about 1 N: loose strands do not carry a push). The operating tool is a free rigid body.

Pushed alone, the strands meet the closed leg and bunch up; the leg does not move. The tool's blade pushed into the
operating slot rides up the tab's ramp, turning the leg open, and rests on the tab's flat, where the spring presses
it against the slot's ceiling square to its travel (it stays there by itself, as WAGO's "hands-free wiring"). With
the clamp open the strands slide in to the insulation stop; pulling the tool out lets the leg close on them and wedge
them against the busbar.

Measured each step, in each terminal's wire frame W (u along the entry channel, v across it, y along the rail): the
strand tip's (u, y, v), the insulation's depth (u of the wire end's root in the named terminal's frame), the named
terminal's leg angle, and where the tool's blade tip is (inside which terminal's operating slot, if any).

Function test (the success check): once the wire is seated (insulation home, strands in past the clamping point, the
tool out of every slot and the leg resting on the strands) and both grasp points have been clear of the wire for
TEST_WAIT_S, the task pulls the wire end straight back out of the entry, ramping to TUG_N over TUG_RAMP_S and holding
it to TEST_PHASE_S. Passed if the wire moved out less than TUG_HELD_M. ``force_function_test(env)`` starts the same
tug at once (control runs).
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

from terminal_tool_sim.kit import GRASP_OFFSET_M, reset_robot_home, write_pose
from terminal_tool_sim.scorer import MILESTONES, PushInScorer, PushInStep
from terminal_tool_sim.spec import N_TERMINALS, Layout, make_seed
from terminal_tool_sim.usd import terminal_props as props

_ASSETS = Path(__file__).resolve().parent / "assets"

ENTRY_IN_M = 0.001
"""The strands' tip counts as in an entry once it is this far past the mouth, within the channel."""
ENTRY_LATERAL_M = props.PITCH / 2
OPEN_DEG = 20.0
"""The clamp counts as held open by the tool when the named terminal's leg is open at least this much (the inserted
blade holds it at about 25 deg; the strands alone hold it at 9-18 deg)."""
PAST_CLAMP_M = 0.0075
"""Strands in past the clamping point (the leg's edge meets the channel 7 mm in)."""
SEATED_U = props.HOME_U - 0.001
"""Seated: the insulation within 1 mm of its stop in the mouth..."""
SEATED_TIP_M = 0.0085
"""...the strands' tip at least this deep (straight, not bunched)..."""
LEAF_ENGAGED_DEG = 3.0
"""...and the named terminal's leg resting on the strands: open at least this much (the tool being out of every slot)."""
SLOT_DEPTH_M = (0.001, props.BLADE_LEN + 0.003)
"""The blade's tip is in a terminal's operating slot when it is this far into the face (x), ..."""
SLOT_LATERAL_M = props.SLOT_W / 2 + 0.0008
SLOT_Z_M = (props.slot_ceiling_z() - props.SLOT_H - 0.0015, props.slot_ceiling_z() + 0.0015)
"""... and this close to its centre along the rail (y) and up or down (z, from the entry's mouth)."""
RELEASE_CLEAR_M = 0.06

TEST_WAIT_S = 1.0
TUG_RAMP_S = 0.5
TUG_HOLD_S = 1.5
TEST_PHASE_S = TUG_RAMP_S + TUG_HOLD_S
TUG_N = 50.0
"""Straight pull out of the entry: the conductor pull-out test force commonly cited for 2.5 mm2 (IEC 60999-1)."""
DROP_Z_M = 0.03
TUG_HELD_M = 0.0015

STRIP, WIRE, STAND, TOOL = "terminal_strip", "wire", "wire_stand", "op_tool"


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
        cfg.actuators = {
            "cable": ImplicitActuatorCfg(joint_names_expr=["cable_j.*"], stiffness=props.CABLE_K,
                                         damping=props.CABLE_DAMPING),
            "strand_bend": ImplicitActuatorCfg(joint_names_expr=["strand_j.*"], stiffness=props.STRAND_BEND_K,
                                               damping=props.STRAND_BEND_DAMPING),
            "strand_slide": ImplicitActuatorCfg(joint_names_expr=["strand_slide"], stiffness=props.STRAND_COLLAPSE_K,
                                                damping=props.STRAND_COLLAPSE_DAMPING),
        }
        cfg.init_state.joint_pos = {".*": 0.0}
        return cfg


class OperatingTool(Object):
    """WAGO 210-720 operating tool, lying on its stand (a free rigid body)."""

    def __init__(self):
        p = props.tool_start()
        super().__init__(name=TOOL, usd_path=str(_ASSETS / "tool.usda"), object_type=ObjectType.RIGID,
                         initial_pose=Pose(position_xyz=p[:3], rotation_xyzw=p[3:]))


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
        self.tip_id = self.wire.find_bodies(f"strand_{props.STRAND_N - 1:02d}")[0][0]
        self.slide_id = self.wire.find_joints("strand_slide")[0][0]
        self.tool = env.scene[TOOL]
        self.leaf_joints = [self.strip.find_joints(f"leaf_{k}_joint")[0][0] for k in range(1, N_TERMINALS + 1)]
        self.flange_ids, _ = self.robot.find_bodies(["left_gripper", "right_gripper"], preserve_order=True)
        self.scorer = PushInScorer(n, dev)
        self.next_layout = make_seed(task.default_seed)
        self.layouts: dict[int, Layout] = {}
        self.last_reports: dict[int, dict] = {}
        self.target = torch.full((n,), self.next_layout.target, dtype=torch.long, device=dev)
        self.mouths = torch.tensor([props.mouth(k) for k in range(1, N_TERMINALS + 1)], device=dev)
        self.tool_start = props.tool_start()
        zb = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
        zl = lambda: torch.zeros(n, dtype=torch.long, device=dev)  # noqa: E731
        zf = lambda: torch.zeros(n, device=dev)  # noqa: E731
        self.seated_once, self.released, self.force_test, self.test_held, self.test_passed = zb(), zb(), zb(), zb(), zb()
        self.tool_opened, self.strands_in, self.tool_in_slot = zb(), zb(), zb()
        self.tool_slot = zl()
        self.max_compress, self.max_tool_leaf_deg = zf(), zf()
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
    def __init__(self, default_seed: int = 0, episode_length_s: float = 120.0):
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
            tool_opened_clamp=bool(st.tool_opened[env_id]), strands_past_clamp=bool(st.strands_in[env_id]),
            tool_in_slot=bool(st.tool_in_slot[env_id]), tool_slot_terminal=int(st.tool_slot[env_id]),
            max_leg_open_by_tool_deg=round(float(st.max_tool_leaf_deg[env_id]), 2),
            max_strand_bunching_mm=round(float(st.max_compress[env_id]) * 1000, 2),
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
        for buf in (st.seated_once, st.released, st.force_test, st.test_held, st.test_passed, st.tool_opened,
                    st.strands_in, st.tool_in_slot):
            buf[env_ids] = False
        for buf in (st.clear_steps, st.phase, st.phase_steps, st.wrong_n, st.tool_slot):
            buf[env_ids] = 0
        for buf in (st.max_leaf_deg, st.max_depth, st.test_max_out, st.test_u0, st.max_push_n, st.max_compress,
                    st.max_tool_leaf_deg):
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
        tj = zj.clone()
        tj[:, st.slide_id] = props.STRAND_COLLAPSE_PRELOAD
        st.wire.set_joint_position_target(tj, env_ids=env_ids)
        write_pose(st.tool, st.tool_start, env_ids, env.scene.env_origins)
        st.start_pos[env_ids] = torch.tensor(props.wire_start_root()[:3], device=dev)
        for env_id in env_ids.tolist():
            st.layouts[env_id] = layout

    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        origins = env.scene.env_origins
        n, dev = env.num_envs, env.device
        t = st.scorer.step.float() * env.step_dt
        uh, vh = u_hat(dev), torch.tensor(props.v_hat(), device=dev)
        ar = torch.arange(n, device=dev)
        tgt = st.target - 1

        # The strands' tip, in every terminal's entry frame.
        pos, quat = st.end_pose(origins)
        tpos = st.wire.data.body_pos_w.torch[:, st.tip_id] - origins
        tquat = st.wire.data.body_quat_w.torch[:, st.tip_id]
        tip = tpos + math_utils.quat_apply(tquat, torch.tensor([[props.STRAND_LINK, 0.0, 0.0]], device=dev).expand(n, 3))
        d = tip[:, None, :] - st.mouths[None]  # (N, T, 3)
        tu, tv, ty = (d * uh).sum(-1), (d * vh).sum(-1), d[..., 1]
        in_channel = (tu > ENTRY_IN_M) & (tu < 0.016) & (ty.abs() < ENTRY_LATERAL_M) & (tv.abs() < 0.004)
        in_target = in_channel[ar, tgt]
        others = in_channel.clone()
        others[ar, tgt] = False
        wrong = others.any(dim=1)
        st.wrong_n = torch.where((st.wrong_n == 0) & wrong, others.float().argmax(dim=1) + 1, st.wrong_n)
        depth = tu[ar, tgt]
        st.max_depth = torch.where(in_target, torch.maximum(st.max_depth, depth), st.max_depth)
        compress = -st.wire.data.joint_pos.torch[:, st.slide_id]
        st.max_compress = torch.maximum(st.max_compress, compress)

        # The tool's blade tip (the tool's origin): in which terminal's operating slot, if any.
        tool_tip = st.tool.data.root_pos_w.torch - origins
        dm = tool_tip[:, None, :] - st.mouths[None]
        in_slot = ((dm[..., 0] > SLOT_DEPTH_M[0]) & (dm[..., 0] < SLOT_DEPTH_M[1]) & (dm[..., 1].abs() < SLOT_LATERAL_M)
                   & (dm[..., 2] > SLOT_Z_M[0]) & (dm[..., 2] < SLOT_Z_M[1]))
        st.tool_in_slot = in_slot.any(dim=1)
        st.tool_slot = torch.where(st.tool_in_slot, in_slot.float().argmax(dim=1) + 1, torch.zeros_like(st.tool_slot))
        tool_in_target = in_slot[ar, tgt]

        root_u = ((pos - st.mouths[tgt]) * uh).sum(-1)
        fm = st.sensor.data.force_matrix_w
        fm = fm.torch if hasattr(fm, "torch") else fm
        push = -(fm.reshape(n, -1, 3).sum(dim=1) * uh).sum(-1)
        st.max_push_n = torch.where((st.phase == 0) & ~st.released & ~st.seated_once,
                                    torch.maximum(st.max_push_n, push), st.max_push_n)
        leaf = props.LEAF_SIGN * torch.rad2deg(st.strip.data.joint_pos.torch[:, st.leaf_joints])[ar, tgt]
        st.max_leaf_deg = torch.maximum(st.max_leaf_deg, leaf)
        held_open = tool_in_target & (leaf > OPEN_DEG)
        st.max_tool_leaf_deg = torch.where(tool_in_target, torch.maximum(st.max_tool_leaf_deg, leaf),
                                           st.max_tool_leaf_deg)
        st.tool_opened |= held_open
        st.strands_in |= st.tool_opened & tool_in_target & in_target & (depth > PAST_CLAMP_M)
        seated = (st.strands_in & in_target & (root_u > SEATED_U) & (depth > SEATED_TIP_M) & ~st.tool_in_slot
                  & (leaf > LEAF_ENGAGED_DEG))
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
            st.test_passed[ids] = st.test_held[ids] & st.seated_once[ids] & ~st.tool_in_slot[ids]
            st.phase[ids] = 2

        dropped = pos[:, 2] < DROP_Z_M
        st.scorer.update(PushInStep(st.tool_opened.clone(), st.strands_in.clone(), seated & (st.phase == 0),
                                    st.test_passed.clone(), wrong & (st.phase == 0), dropped,
                                    st.tool_in_slot.clone()))


_STRIP_PATH = "{ENV_REGEX_NS}/" + STRIP


@configclass
class SensorsCfg:
    wire_contacts: ContactSensorCfg = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/" + WIRE + "/strand_.*", update_period=0.0, history_length=1, debug_vis=False,
        filter_prim_paths_expr=[f"{_STRIP_PATH}/base"] + [f"{_STRIP_PATH}/leaf_{k}" for k in range(1, N_TERMINALS + 1)],
    )


@configclass
class EventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: PushInTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))
