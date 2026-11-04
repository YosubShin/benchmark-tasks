"""CT test-terminal bridge: short the named CT circuit at its test-disconnect terminals, then open its links.
Arena props and TaskBase.

Mechanism (see ctbridge_sim/usd/ct_props.py):
- The terminal row is one fixed-base articulation: the housings (a well in the middle of each terminal holding two
  socket tubes with 4.8 mm square holes, one each side of the disconnect point) and six disconnect slides at the
  wells' bottom on prismatic joints along the terminal's length (0 = closed, +5 mm = open, toward the relay side). The
  task holds each slide with a detent (a spring drive toward the nearer end, DETENT_PEAK_N at mid-travel): a rule
  standing in for the real slide's M3 clamp screw. Only a thin blade reaches a slide (the slot is 7.2 mm wide and
  22 mm deep): the robot uses the screwdriver, as crews do.
- The KSS 8 shorting bridge is a free rigid body. Its two 4 mm pins (8.2 mm apart) enter two neighbouring sockets
  only where they are (geometry); across a partition plate the pitch is 10.4 mm, so it cannot bridge two circuits.
- Electrical state (rule, from geometry): a pin makes contact when it is at least CONTACT_DEPTH_M inside a socket;
  the named CT is shorted while both pins are in contact in its two CT-side sockets. A link is open (its contact
  broken) once its slide has moved more than BREAK_M.

Function test (the success check): once the bridge is home in the named CT-side sockets, both named links are open,
and both grasp points have been clear of the bridge for TEST_WAIT_S, the scorer tugs the bridge sideways along the
rail (TUG_N, +y) and then toward the CT side (TUG_N, -x) for TEST_PHASE_S each. Passed if the CT stayed shorted
throughout and both links are still open. `force_function_test(env)` starts the same test at once (control).
Under the station harness the test runs off camera (the task file's ``test_phase``).
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

from ctbridge_sim.kit import GRASP_OFFSET_M, Fingertips, reset_robot_home, reviewer_camera, write_pose
from ctbridge_sim.scorer import MILESTONES, BridgeScorer, BridgeStep
from ctbridge_sim.spec import EPISODE_SECONDS, HOLDER_XY, STRIP_X, STRIP_Y, Layout, make_seed
from ctbridge_sim.usd import ct_props as props

_ASSETS = Path(__file__).resolve().parent / "assets"
MM = 0.001

TRAVEL_M = props.SLIDE_TRAVEL * MM
BREAK_M = 0.5 * TRAVEL_M
"""The slide's contact with the CT-side half breaks at mid-travel (inferred)."""
DETENT_PEAK_N = 0.6
"""Force to push a loosened slide over its mid-point (inferred; the real slide is clamped by an M3 screw). 1.0 N in
the earlier build; under the station arm's soft wrist gains the blade stalled at 2.3 mm against 1.0 N."""
DETENT_K = DETENT_PEAK_N / BREAK_M
SLIDE_DAMPING = 5.0

CONTACT_DEPTH_M = (props.TUBE_TOP_DROP + 6.0) * MM
"""A pin touches the socket's spring contact once 6 mm into the tube (depth measured from the housing top; inferred)."""
HOME_DEPTH_M = (props.PIN_LEN - 1.5) * MM
"""Milestone 50: the grip within 1.5 mm of resting on the housings' top."""
LIFT_M = 0.01
DROP_Z_M = props.PLATE_TOP + 0.02
RELEASE_CLEAR_M = 0.05
DECOY_MARGIN_M = 0.001
DECOY_TOUCH_Z_M = props.TOP_Z + 0.010
"""A finger pad (6 mm above the finger's end) lower than this over a decoy terminal counts as touching it."""

TEST_WAIT_S = 2.0
TEST_PHASE_S = 1.5
TUG_N = float(os.environ.get("CTB_TUG_N", "5.0"))
"""Sideways tug on the bridge (a spring scale hooked under its grip; a test lead or a sleeve catching it). A bridge
resting on the sockets without its pins in slides off under it. The earlier 120 Hz build threw the 8 g bridge out of
its sockets at 3 N (a solver blow-out against 1.1 mm tube walls); at the station's 240 Hz with 1.6 mm walls the
bridge holds."""


def bridge_start_pose() -> tuple:
    return (HOLDER_XY[0], HOLDER_XY[1], props.TOP_Z + 0.0003, 0.0, 0.0, 0.0, 1.0)


def screwdriver_start_pose() -> tuple:
    return (props.SD_STAND_XY[0], props.SD_STAND_XY[1], props.sd_start_tip_z() + 0.0003, 0.0, 0.0, 0.0, 1.0)


class Strip(Object):
    def __init__(self):
        super().__init__(name="strip", usd_path=str(_ASSETS / "strip.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=(STRIP_X, STRIP_Y, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {"slides": ImplicitActuatorCfg(joint_names_expr=["slide_.*"], stiffness=DETENT_K,
                                                       damping=SLIDE_DAMPING)}
        cfg.init_state.joint_pos = {"slide_.*": 0.0}
        return cfg


class Bridge(Object):
    def __init__(self):
        p = bridge_start_pose()
        super().__init__(name="bridge", usd_path=str(_ASSETS / "bridge.usda"), object_type=ObjectType.RIGID,
                         initial_pose=Pose(position_xyz=p[:3], rotation_xyzw=p[3:]))


class Screwdriver(Object):
    def __init__(self):
        p = screwdriver_start_pose()
        super().__init__(name="screwdriver", usd_path=str(_ASSETS / "screwdriver.usda"), object_type=ObjectType.RIGID,
                         initial_pose=Pose(position_xyz=p[:3], rotation_xyzw=p[3:]))


class Holder(Object):
    def __init__(self):
        super().__init__(name="holder", usd_path=str(_ASSETS / "holder.usda"), object_type=ObjectType.BASE,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))


class _TaskState:
    def __init__(self, env: ManagerBasedEnv, task: "BridgeTask"):
        n, dev = env.num_envs, env.device
        self.strip = env.scene["strip"]
        self.bridge = env.scene["bridge"]
        self.screwdriver = env.scene["screwdriver"]
        self.robot = env.scene["robot"]
        self.tips = Fingertips(self.robot)
        self.slide_joints = {name: self.strip.find_joints(f"slide_{name}")[0][0] for name in props.TERMINALS}
        self.flange_ids, _ = self.robot.find_bodies(["left_gripper", "right_gripper"], preserve_order=True)
        self.scorer = BridgeScorer(n, dev)
        self.next_layout = make_seed(task.default_seed)
        self.layouts: dict[int, Layout] = {}
        self.last_reports: dict[int, dict] = {}
        ys = props.terminal_ys()
        # Socket table: (terminal index, side) -> mouth centre in the strip frame.
        self.sockets = torch.tensor([[s * props.SOCKET_X * MM, props.slot_y(name) * MM] for name in props.TERMINALS
                                     for s in (-1, 1)], device=dev)  # (12, 2): index 2*t + (0 ct, 1 relay)
        self.term_y = torch.tensor([ys[name] * MM for name in props.TERMINALS], device=dev)
        self.named_mask = torch.zeros(n, len(props.TERMINALS), dtype=torch.bool, device=dev)
        zb = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
        zl = lambda: torch.zeros(n, dtype=torch.long, device=dev)  # noqa: E731
        self.released, self.force_test, self.test_ok, self.test_passed = zb(), zb(), zb(), zb()
        self.ever_unshorted_open = zb()
        self.clear_steps, self.phase, self.phase_steps = zl(), zl(), zl()
        self.wait_steps = max(1, round(TEST_WAIT_S / env.step_dt))
        self.phase_len = max(1, round(TEST_PHASE_S / env.step_dt))
        self.max_slide_open = torch.zeros(n, len(props.TERMINALS), device=dev)
        self.last_sockets = torch.full((n, 2), -1, dtype=torch.long, device=dev)
        self.first_touch = None
        self.test_tilt_deg = torch.zeros(n, device=dev)
        self.test_rise_m = torch.zeros(n, device=dev)
        self.test_start_z = torch.zeros(n, device=dev)


class BridgeTask(TaskBase):
    def __init__(self, default_seed: int = 0, episode_length_s: float = EPISODE_SECONDS, reviewer_camera: bool = False):
        super().__init__(episode_length_s=episode_length_s, task_description=make_seed(default_seed).instruction)
        self.default_seed = default_seed
        self.reviewer_camera = reviewer_camera

    def state(self, env: ManagerBasedEnv) -> _TaskState:
        if getattr(env, "_ctbridge_state", None) is None:
            env._ctbridge_state = _TaskState(env, self)
        return env._ctbridge_state

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
        socks = [int(v) for v in st.last_sockets[env_id]]
        names = [f"{props.TERMINALS[s // 2]}:{'ct' if s % 2 == 0 else 'relay'}" if s >= 0 else "-" for s in socks]
        r.update(
            bridge_pins_in=names,
            slide_open_max_mm={name: round(float(st.max_slide_open[env_id, i]) * 1000, 1)
                               for i, name in enumerate(props.TERMINALS)},
            function_test_phase=int(st.phase[env_id]), function_test_passed=bool(st.test_passed[env_id]),
            first_decoy_touch=st.first_touch, function_test_tug_n=TUG_N, function_test_max_tilt_deg=round(float(st.test_tilt_deg[env_id]), 1),
            function_test_max_rise_mm=round(float(st.test_rise_m[env_id]) * 1000, 2),
        )
        return r

    def get_scene_cfg(self):
        return ReviewerCameraSceneCfg() if self.reviewer_camera else None

    def get_events_cfg(self):
        return BridgeEventsCfg(self)

    def get_termination_cfg(self) -> TaskTerminationCfg:
        sequence = [(self._milestone_predicate(m), 0.25) for m in MILESTONES]
        return TaskTerminationCfg(timeout_s=self.episode_length_s,
                                  success=[CompletionCriteria(name="ctbridge_milestones", predicate_sequence=sequence)])

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def _milestone_predicate(self, milestone: int):
        def predicate(env: ManagerBasedEnv) -> torch.Tensor:
            return self.state(env).scorer.milestone >= milestone

        predicate.__name__ = f"milestone_{milestone}"
        return predicate

    # Events ------------------------------------------------------------------------------------------------------
    def reset(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        reset_robot_home(env, env_ids)
        for env_id in env_ids.tolist():
            if int(st.scorer.step[env_id]) > 0:
                st.last_reports[env_id] = self._report(st, env_id)
        layout = st.next_layout
        dev, n = env.device, len(env_ids)
        origins = env.scene.env_origins
        st.scorer.reset(env_ids)
        for buf in (st.released, st.force_test, st.test_ok, st.test_passed, st.ever_unshorted_open):
            buf[env_ids] = False
        for buf in (st.clear_steps, st.phase, st.phase_steps):
            buf[env_ids] = 0
        st.max_slide_open[env_ids] = 0.0
        st.first_touch = None
        st.test_tilt_deg[env_ids] = 0.0
        st.test_rise_m[env_ids] = 0.0
        st.last_sockets[env_ids] = -1
        st.named_mask[env_ids] = False
        for i, name in enumerate(props.TERMINALS):
            if name[0] == layout.named:
                st.named_mask[env_ids, i] = True
        st.bridge.permanent_wrench_composer.reset(env_ids=env_ids)
        joints = list(st.slide_joints.values())
        zero = torch.zeros(n, len(joints), device=dev)
        st.strip.write_joint_state_to_sim(zero, zero, joint_ids=joints, env_ids=env_ids)
        st.strip.set_joint_position_target(zero, joint_ids=joints, env_ids=env_ids)
        write_pose(st.bridge, bridge_start_pose(), env_ids, origins[env_ids])
        write_pose(st.screwdriver, screwdriver_start_pose(), env_ids, origins[env_ids])
        for env_id in env_ids.tolist():
            st.layouts[env_id] = layout

    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        origins = env.scene.env_origins
        n, dev = env.num_envs, env.device
        strip_origin = origins + torch.tensor([STRIP_X, STRIP_Y, 0.0], device=dev)

        # Slides: detent toward the nearer end; open = contact broken.
        joints = list(st.slide_joints.values())
        q = st.strip.data.joint_pos.torch[:, joints]  # (N, 6), >= 0
        target = torch.where(q > BREAK_M, TRAVEL_M, 0.0)
        st.strip.set_joint_position_target(target, joint_ids=joints)
        opened = q
        st.max_slide_open = torch.maximum(st.max_slide_open, opened)
        link_open = opened > BREAK_M

        # Bridge pins: which socket each is in, and how deep.
        b_pos = st.bridge.data.root_pos_w.torch - strip_origin
        b_quat = st.bridge.data.root_quat_w.torch
        pin_tips = []
        for yp in (props.PITCH / 2 * MM, -props.PITCH / 2 * MM):
            local = torch.tensor([[0.0, yp, -props.PIN_LEN * MM]], device=dev).expand(n, 3)
            pin_tips.append(b_pos + math_utils.quat_apply(b_quat, local))
        tips = torch.stack(pin_tips, dim=1)  # (N, 2, 3) strip frame
        d = (tips[:, :, None, :2] - st.sockets[None, None]).abs()  # (N, 2, 12, 2)
        inside = (d < props.SOCKET / 2 * MM).all(dim=-1)  # (N, 2, 12)
        depth = props.TOP_Z - tips[..., 2]  # (N, 2)
        in_contact = inside & (depth[..., None] > CONTACT_DEPTH_M)
        home = inside & (depth[..., None] > HOME_DEPTH_M)
        sock = torch.where(in_contact.any(dim=-1), in_contact.float().argmax(dim=-1), torch.full_like(tips[..., 0],
                                                                                                         -1).long())
        st.last_sockets = sock
        named_t = st.named_mask.float().argmax(dim=1)  # first named terminal (S1); S2 is the next one
        ct_a, ct_b = 2 * named_t, 2 * (named_t + 1)
        pins_on_named = ((sock[:, 0] == ct_a) & (sock[:, 1] == ct_b)) | ((sock[:, 0] == ct_b) & (sock[:, 1] == ct_a))
        ct_shorted = pins_on_named
        both_in = (sock >= 0).all(dim=1)
        bridge_home = both_in & home.any(dim=-1).all(dim=1)
        wrong_short = both_in & ~pins_on_named & (st.phase == 0)

        named_open_any = (link_open & st.named_mask).any(dim=1)
        named_open_all = (link_open | ~st.named_mask).all(dim=1)
        decoy_open = (link_open & ~st.named_mask).any(dim=1) & (st.phase == 0)
        opened_unshorted = named_open_any & ~ct_shorted & (st.phase == 0)
        st.ever_unshorted_open |= opened_unshorted
        opened_shorted = named_open_all & ct_shorted & ~st.ever_unshorted_open

        # Robot: grasp points (release check) and finger pads (decoy touches).
        flange_pos = st.robot.data.body_pos_w.torch[:, st.flange_ids]
        flange_quat = st.robot.data.body_quat_w.torch[:, st.flange_ids]
        down = torch.tensor([0.0, 0.0, -GRASP_OFFSET_M], device=dev).expand_as(flange_pos)
        grasp = flange_pos + math_utils.quat_apply(flange_quat, down) - strip_origin[:, None]
        clear = ((grasp - b_pos[:, None]).norm(dim=-1) > RELEASE_CLEAR_M).all(dim=1)
        pads = st.tips.points(strip_origin)  # (N, 4, 3) strip frame
        sd_tip = (st.screwdriver.data.root_pos_w.torch - strip_origin).unsqueeze(1)
        pts = torch.cat([pads, tips, sd_tip], dim=1)
        half_len = props.LENGTH / 2 * MM + DECOY_MARGIN_M
        half_w = props.PITCH / 2 * MM + DECOY_MARGIN_M
        over = (pts[..., None, 0].abs() < half_len) & ((pts[..., None, 1] - st.term_y).abs() < half_w)  # (N, 6, T)
        low = pts[..., 2:3] < DECOY_TOUCH_Z_M
        touch = (over & low & ~st.named_mask[:, None]).any(dim=2) & (st.phase == 0)[:, None]  # (N, 7)
        decoy_touched = touch.any(dim=1)
        if bool(decoy_touched[0]) and st.first_touch is None:
            p = pts[0, touch[0].nonzero().flatten()[0]]
            who = ("pad", "pad", "pad", "pad", "pin", "pin", "blade")[int(touch[0].nonzero().flatten()[0])]
            st.first_touch = f"{who} at {[round(float(v) * 1000, 1) for v in p]} mm (strip frame)"

        b_start = torch.tensor(bridge_start_pose()[:3], device=dev) - torch.tensor([STRIP_X, STRIP_Y, 0.0], device=dev)
        lifted = (b_pos - b_start).norm(dim=-1) > LIFT_M
        dropped = (b_pos[:, 2] < DROP_Z_M - 0.0) & ~both_in

        # Function test.
        ready = (bridge_home & ct_shorted & named_open_all & clear & (st.phase == 0)
                 & ~st.ever_unshorted_open) | (st.force_test & (st.phase == 0))
        st.clear_steps = torch.where(ready, st.clear_steps + 1, torch.zeros_like(st.clear_steps))
        start = ready & ((st.clear_steps >= st.wait_steps) | st.force_test)
        if start.any():
            ids = start.nonzero().flatten()
            st.test_ok[ids] = True
            st.test_start_z[ids] = b_pos[ids, 2]
            self._tug(st, ids, (0.0, TUG_N, 0.0))
            st.phase[ids] = 1
            st.phase_steps[ids] = 0
        in_test = st.phase > 0
        st.phase_steps = torch.where(in_test, st.phase_steps + 1, st.phase_steps)
        testing = (st.phase == 1) | (st.phase == 2)
        tilt = torch.rad2deg(2 * torch.acos(b_quat[:, 3].abs().clamp(max=1.0)))
        st.test_tilt_deg = torch.where(testing, torch.maximum(st.test_tilt_deg, tilt), st.test_tilt_deg)
        st.test_rise_m = torch.where(testing, torch.maximum(st.test_rise_m, b_pos[:, 2] - st.test_start_z),
                                     st.test_rise_m)
        st.test_ok &= ~(testing & ~(ct_shorted & named_open_all))
        end1 = (st.phase == 1) & (st.phase_steps >= st.phase_len)
        if end1.any():
            ids = end1.nonzero().flatten()
            st.bridge.permanent_wrench_composer.reset(env_ids=ids)
            self._tug(st, ids, (-TUG_N, 0.0, 0.0))
            st.phase[ids] = 2
            st.phase_steps[ids] = 0
        end2 = (st.phase == 2) & (st.phase_steps >= st.phase_len)
        if end2.any():
            ids = end2.nonzero().flatten()
            st.bridge.permanent_wrench_composer.reset(env_ids=ids)
            st.test_passed[ids] = st.test_ok[ids] & ct_shorted[ids] & named_open_all[ids]
            st.phase[ids] = 3

        st.scorer.update(BridgeStep(lifted, ct_shorted & bridge_home, opened_shorted & (st.phase == 0),
                                st.test_passed.clone(), opened_unshorted, decoy_open, wrong_short, decoy_touched,
                                dropped))

    @staticmethod
    def _tug(st: _TaskState, ids: torch.Tensor, force) -> None:
        f = torch.tensor(force, device=ids.device).view(1, 1, 3).expand(len(ids), 1, 3).contiguous()
        st.bridge.permanent_wrench_composer.add_forces_and_torques_index(
            forces=f, torques=torch.zeros_like(f), body_ids=[0], env_ids=ids, is_global=True)


REVIEWER_EYE = (0.42, -0.10, 0.21)
REVIEWER_TARGET = (0.265, 0.0, 0.06)


@configclass
class ReviewerCameraSceneCfg:
    reviewer_cam: CameraCfg = reviewer_camera(REVIEWER_EYE, REVIEWER_TARGET)


@configclass
class BridgeEventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: BridgeTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))
