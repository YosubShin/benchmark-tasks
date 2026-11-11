"""Lock out an equipment cord with a plug lockout: Arena props and TaskBase.

Mechanism (pluglock_sim/usd/pluglock_props.py):
- Outlet: fixed-base articulation; behind each blade slot a contact pad on a prismatic joint is pressed onto the blade
  by this task's preloaded joint drive (about 8 N to pull a plug out).
- Cords: fixed-base articulations (clamp, 24 capsule links with bending springs toward the starting route, the
  plug). Both plugs start in the duplex outlet.
- Lockout (Brady 65674): fixed-base articulation, the lower half-shell held by its fixture, the lid on a revolute
  hinge (0 closed, 180 deg open flat, where it starts) with light damping and nothing else holding it.
- Padlock (Master Lock 410): body and shackle on a prismatic joint with a weak spring; once pushed within
  LATCH_TRAVEL_M of shut it latches (a joint-limit rule; the real lock's latch is internal).

Measured each step: how far each plug's front is out of its outlet (s); the target plug's corners in the lockout's
frame; the hinge angle; whether the shackle's short leg passes through both tabs' holes (each tab in its own shell's
frame).

Function test (the success check): once the shackle has latched through both tabs with the lid shut on the plug and
both grasp points have been clear of the padlock and the lockout for TEST_WAIT_S, the lid is pried open with PRY_NM
about the hinge for TEST_PHASE_S, then the cord is pulled straight out of the cord hole with PULL_N for TEST_PHASE_S
(the lower half is held by its fixture, as a reviewer's hand would hold it). Passed if the lid opened less than
PRY_HELD_DEG and the plug is still wholly inside. ``force_function_test(env)`` starts the same test at once (the
control run: lid closed, no padlock).
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

from pluglock_sim.kit import GRASP_OFFSET_M, q_axis, qmul, reset_robot_home
from pluglock_sim.scorer import MILESTONES, PlugLockoutScorer, Step
from pluglock_sim.spec import PEDESTAL_XY, Layout, make_seed
from pluglock_sim.usd import padlock_props as lock_props
from pluglock_sim.usd import pluglock_props as props

_ASSETS = Path(__file__).resolve().parent / "assets"

PV = lock_props.PADLOCKS["ml410"]
LOCK_TILT_DEG = 15.0
LOCK_LYING_QUAT = tuple(float(v) for v in q_axis((1, 0, 0), math.pi / 2 + math.radians(LOCK_TILT_DEG)))
"""The padlock lying on its 38 x 44 mm face, shackle end tilted down 15 degrees: width (local x, short leg at +x)
along +x, shackle (local z) along -y and 15 degrees down, toward the lockout's tabs. The arm's fingers close on its
two broad faces (so no finger comes between the padlock and the lockout), pitched down 15 degrees as the arm reaches
it best."""
SPRING_N_PER_M = 150.0
LATCH_TRAVEL_M = 0.002

UNPLUGGED_M = props.PIN_L + 0.001
"""Unplugged: the plug's front more than this out of its outlet (the ground pin's tip clear of the face)."""
DECOY_MOVED_M = 0.003
IN_TOL_M = 0.001
CLOSED_DEG = 3.0
LID_OPEN_RAD = math.pi
DROP_Z_M = 0.04
RELEASE_CLEAR_M = 0.06
BOX_CLEAR_M = 0.07

TEST_WAIT_S = 1.0
TEST_PHASE_S = 1.5
PRY_NM = 0.5
"""Prying the lid: 0.5 N*m about the hinge (about 8 N on a spring scale hooked through the lid's tab, 6.5 cm from
the hinge)."""
PRY_HELD_DEG = 10.0
PULL_N = 40.0
"""Pulling the cord straight out of the cord hole (a firm hand tug, 9 lbf)."""
PULL_LINK = props.CORD_N - 4
"""The pull acts on this cord link (7.5 to 10 cm from the plug, outside the shell)."""
PRY_RAMP_S = 0.3
HINGE_DAMPING = 0.1
"""Hinge damping, N*m*s/rad (inferred: the moulded hinge pins are stiff, so a lid let go near upright sinks slowly,
at a few degrees per second, instead of flopping; the 0.5 N*m pry still swings an unlocked lid wide open)."""

OUTLET, CORD_T, CORD_D, LOCKOUT, PADLOCK, STAND = "outlet", "cord_target", "cord_decoy", "lockout", "padlock", "stand"


def _pose(p) -> Pose:
    return Pose(position_xyz=tuple(p[:3]), rotation_xyzw=tuple(p[3:]))


def lock_start_pose() -> tuple:
    """Body centre on the pedestal's sloped top (its broad face 0.5 mm above the slab's top at the body's centre)."""
    x, y = PEDESTAL_XY
    return (x, y, props.PEDESTAL_TOP + PV.body[1] / 2 / math.cos(math.radians(LOCK_TILT_DEG)) + 0.0005) + LOCK_LYING_QUAT


class Outlet(Object):
    def __init__(self):
        super().__init__(name=OUTLET, usd_path=str(_ASSETS / "outlet.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {"pads": ImplicitActuatorCfg(joint_names_expr=["pad_.*_joint"], stiffness=props.CONTACT_K,
                                                     damping=props.CONTACT_DAMPING, armature=0.01)}
        cfg.init_state.joint_pos = {"pad_.*_joint": -props.PAD_INTERFERENCE}
        return cfg


class Cord(Object):
    """Clamp, flexible cord, plug (body ``plug``)."""

    def __init__(self, which: str):
        name = CORD_T if which == "target" else CORD_D
        super().__init__(name=name, usd_path=str(_ASSETS / f"cord_{which}.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {
            "cord": ImplicitActuatorCfg(joint_names_expr=["cord_j([01][0-9]|2[0-2])"], stiffness=props.CORD_K,
                                        damping=props.CORD_DAMPING, armature=1e-5),
            "relief": ImplicitActuatorCfg(joint_names_expr=["cord_j2[34]"], stiffness=props.RELIEF_K,
                                          damping=props.CORD_DAMPING, armature=1e-5),
        }
        cfg.init_state.joint_pos = {"cord_j.*": 0.0}
        return cfg


class Lockout(Object):
    def __init__(self):
        super().__init__(name=LOCKOUT, usd_path=str(_ASSETS / "lockout.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {"hinge": ImplicitActuatorCfg(joint_names_expr=["hinge"], stiffness=0.0, damping=HINGE_DAMPING,
                                                      armature=1e-4)}
        cfg.init_state.joint_pos = {"hinge": LID_OPEN_RAD}
        return cfg


class Padlock(Object):
    """The blue Master Lock 410 (sprung shackle; the task latches it)."""

    def __init__(self):
        p = lock_start_pose()
        super().__init__(name=PADLOCK, usd_path=str(_ASSETS / "padlock_ml410.usda"),
                         object_type=ObjectType.ARTICULATION, initial_pose=_pose(p))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {"shackle": ImplicitActuatorCfg(joint_names_expr=["shackle_joint"], stiffness=SPRING_N_PER_M,
                                                        damping=3.0)}
        cfg.init_state.joint_pos = {"shackle_joint": lock_props.OPEN_TRAVEL}
        return cfg


class Stand(Object):
    def __init__(self):
        super().__init__(name=STAND, usd_path=str(_ASSETS / "stand.usda"), object_type=ObjectType.BASE,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))


def _write_pose(asset, pose, env_ids, origins):
    dev, n = origins.device, len(env_ids)
    p = torch.tensor(pose, device=dev, dtype=torch.float32).unsqueeze(0).repeat(n, 1)
    p[:, :3] += origins[env_ids]
    asset.write_root_pose_to_sim(p, env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)


class _TaskState:
    def __init__(self, env: ManagerBasedEnv, task: "PlugLockoutTask"):
        n, dev = env.num_envs, env.device
        self.outlet = env.scene[OUTLET]
        self.cord_t = env.scene[CORD_T]
        self.cord_d = env.scene[CORD_D]
        self.box = env.scene[LOCKOUT]
        self.lock = env.scene[PADLOCK]
        self.robot = env.scene["robot"]
        self.plug_t = self.cord_t.find_bodies("plug")[0][0]
        self.plug_d = self.cord_d.find_bodies("plug")[0][0]
        self.pull_link = self.cord_t.find_bodies(f"cord_{PULL_LINK:02d}")[0][0]
        self.pad_joints = self.outlet.find_joints("pad_.*_joint")[0]
        self.hinge = self.box.find_joints("hinge")[0][0]
        self.base_id = self.box.find_bodies("base")[0][0]
        self.lid_id = self.box.find_bodies("lid")[0][0]
        self.shackle_joint = self.lock.find_joints("shackle_joint")[0][0]
        self.lock_body = self.lock.find_bodies("body")[0][0]
        self.shackle_id = self.lock.find_bodies("shackle")[0][0]
        self.flange_ids, _ = self.robot.find_bodies(["left_gripper", "right_gripper"], preserve_order=True)
        self.scorer = PlugLockoutScorer(n, dev)
        self.next_layout = make_seed(task.default_seed)
        self.layouts: dict[int, Layout] = {}
        self.last_reports: dict[int, dict] = {}
        self.plug_pts = torch.tensor(props.PLUG_POINTS, device=dev)
        zb = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
        zl = lambda: torch.zeros(n, dtype=torch.long, device=dev)  # noqa: E731
        zf = lambda: torch.zeros(n, device=dev)  # noqa: E731
        self.latched, self.latched_through, self.released, self.force_test = zb(), zb(), zb(), zb()
        self.pry_held, self.pull_held, self.test_passed, self.closed_in_once = zb(), zb(), zb(), zb()
        self.clear_steps, self.phase, self.phase_steps = zl(), zl(), zl()
        self.pry_max, self.decoy_max, self.s_t, self.s_d, self.hinge_deg = zf(), zf(), zf(), zf(), zf()
        self.wait_steps = max(1, round(TEST_WAIT_S / env.step_dt))
        self.phase_len = max(1, round(TEST_PHASE_S / env.step_dt))
        self.ramp_len = max(1, round(PRY_RAMP_S / env.step_dt))


def plug_s(pos: torch.Tensor, which: str) -> torch.Tensor:
    """How far the plug's front is out of its outlet (along the outlet's axis)."""
    return torch.tensor(props.outlet_root(which)[0], device=pos.device) - pos[:, 0]


def through_tab(tip, top, tab_y, half_t):
    """The short leg (tip -> top, in a shell's frame) crosses that shell's tab (mid-plane y = tab_y) inside its hole."""
    denom = tip[:, 1] - top[:, 1]
    denom = torch.where(denom.abs() > 1e-6, denom, torch.full_like(denom, 1e-6))
    a = ((tip[:, 1] - tab_y) / denom).clamp(0.0, 1.0)
    cross = tip + a.unsqueeze(1) * (top - tip)
    h = props.TAB_HOLE / 2
    in_hole = ((cross[:, 0] - props.TAB_HOLE_X).abs() < h) & (cross[:, 2].abs() < h)
    spans = (((tip[:, 1] < tab_y - half_t) & (top[:, 1] > tab_y + half_t))
             | ((tip[:, 1] > tab_y + half_t) & (top[:, 1] < tab_y - half_t)))
    return in_hole & spans


class PlugLockoutTask(TaskBase):
    def __init__(self, default_seed: int = 0, episode_length_s: float = 180.0):
        super().__init__(episode_length_s=episode_length_s, task_description=make_seed(default_seed).instruction)
        self.default_seed = default_seed

    def state(self, env: ManagerBasedEnv) -> _TaskState:
        if getattr(env, "_task_state", None) is None:
            env._task_state = _TaskState(env, self)
        return env._task_state

    def force_function_test(self, env: ManagerBasedEnv) -> None:
        """Control runs: start the check now, padlock or not."""
        self.state(env).force_test[:] = True

    def episode_report(self, env: ManagerBasedEnv, env_id: int = 0, ended: bool = False) -> dict:
        st = self.state(env)
        if ended and env_id in st.last_reports:
            return st.last_reports.pop(env_id)
        return self._report(st, env_id)

    def _report(self, st: _TaskState, env_id: int) -> dict:
        r = st.scorer.report(env_id)
        r.update(
            target_out_mm=round(float(st.s_t[env_id]) * 1000, 1), decoy_out_max_mm=round(float(st.decoy_max[env_id]) * 1000, 2),
            hinge_deg=round(float(st.hinge_deg[env_id]), 1), closed_on_plug=bool(st.closed_in_once[env_id]),
            shackle_latched=bool(st.latched[env_id]), latched_through_tabs=bool(st.latched_through[env_id]),
            released=bool(st.released[env_id]), check_phase=int(st.phase[env_id]),
            pry_max_deg=round(math.degrees(float(st.pry_max[env_id])), 1), pry_held=bool(st.pry_held[env_id]),
            pull_held=bool(st.pull_held[env_id]), check_passed=bool(st.test_passed[env_id]),
        )
        if hasattr(st, "in_excess"):
            r.update(plug_in_lockout=bool(st.inside_now[env_id]),
                     plug_out_of_shell_mm=[round(float(v) * 1000, 1) for v in st.in_excess[env_id]])
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
        origins = env.scene.env_origins
        st.scorer.reset(env_ids)
        for buf in (st.latched, st.latched_through, st.released, st.force_test, st.pry_held, st.pull_held,
                    st.test_passed, st.closed_in_once):
            buf[env_ids] = False
        for buf in (st.clear_steps, st.phase, st.phase_steps):
            buf[env_ids] = 0
        for buf in (st.pry_max, st.decoy_max):
            buf[env_ids] = 0.0
        for asset in (st.box, st.cord_t, st.lock):
            asset.permanent_wrench_composer.reset(env_ids=env_ids)
        # Contact pads pressed on the blades of both plugs.
        npj = len(st.pad_joints)
        zp = torch.full((n, npj), -props.PAD_INTERFERENCE, device=dev)
        st.outlet.write_joint_state_to_sim(zp, torch.zeros_like(zp), joint_ids=st.pad_joints, env_ids=env_ids)
        st.outlet.set_joint_position_target(torch.full_like(zp, props.CONTACT_PRELOAD_M), joint_ids=st.pad_joints,
                                            env_ids=env_ids)
        for cord in (st.cord_t, st.cord_d):
            zj = torch.zeros(n, cord.num_joints, device=dev)
            cord.write_joint_state_to_sim(zj, zj, env_ids=env_ids)
            cord.set_joint_position_target(zj, env_ids=env_ids)
        # Lockout open flat.
        h = torch.full((n, 1), LID_OPEN_RAD, device=dev)
        st.box.write_joint_state_to_sim(h, torch.zeros_like(h), joint_ids=[st.hinge], env_ids=env_ids)
        st.box.set_joint_position_target(h, joint_ids=[st.hinge], env_ids=env_ids)
        # Padlock open on its pedestal.
        joint = [st.shackle_joint]
        limits = torch.tensor([[[0.0, lock_props.OPEN_TRAVEL]]], device=dev).expand(n, 1, 2).contiguous()
        travel = torch.full((n, 1), lock_props.OPEN_TRAVEL, device=dev)
        st.lock.write_joint_position_limit_to_sim(limits, joint_ids=joint, env_ids=env_ids, warn_limit_violation=False)
        st.lock.write_joint_state_to_sim(travel, torch.zeros_like(travel), joint_ids=joint, env_ids=env_ids)
        st.lock.set_joint_position_target(travel + 0.005, joint_ids=joint, env_ids=env_ids)
        _write_pose(st.lock, lock_start_pose(), env_ids, origins)
        for env_id in env_ids.tolist():
            st.layouts[env_id] = layout

    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:  # noqa: C901
        st = self.state(env)
        origins = env.scene.env_origins
        n, dev = env.num_envs, env.device
        live = st.phase == 0

        # Plugs.
        pt = st.cord_t.data.body_pos_w.torch[:, st.plug_t] - origins
        qt = st.cord_t.data.body_quat_w.torch[:, st.plug_t]
        pd = st.cord_d.data.body_pos_w.torch[:, st.plug_d] - origins
        st.s_t = plug_s(pt, "target")
        st.s_d = plug_s(pd, "decoy")
        st.decoy_max = torch.where(live, torch.maximum(st.decoy_max, st.s_d), st.decoy_max)
        unplugged = st.s_t > UNPLUGGED_M

        # The plug in the lockout (lower half fixed: its frame is the box frame).
        bquat = st.box.data.body_quat_w.torch[:, st.base_id]
        # The lower half's link sits at the asset origin; the box frame is at BOX_CENTER in it.
        bpos = (st.box.data.body_pos_w.torch[:, st.base_id] - origins
                + math_utils.quat_apply(bquat, torch.tensor([props.BOX_CENTER], device=dev).expand(n, 3)))
        npt = st.plug_pts.shape[0]
        corners = pt[:, None] + math_utils.quat_apply(qt[:, None].expand(n, npt, 4).reshape(-1, 4),
                                                      st.plug_pts[None].expand(n, npt, 3).reshape(-1, 3)).reshape(n, npt, 3)
        local = math_utils.quat_apply_inverse(bquat[:, None].expand(n, npt, 4).reshape(-1, 4),
                                              (corners - bpos[:, None]).reshape(-1, 3)).reshape(n, npt, 3)
        inside = torch.ones(n, dtype=torch.bool, device=dev)
        excess = []
        for k, (lo, hi) in enumerate(props.INTERIOR):
            inside &= ((local[..., k] > lo - IN_TOL_M) & (local[..., k] < hi + IN_TOL_M)).all(dim=1)
            excess.append(torch.maximum(lo - local[..., k], local[..., k] - hi).amax(dim=1))
        st.in_excess = torch.stack(excess, dim=1)
        st.inside_now = inside
        hinge = st.box.data.joint_pos.torch[:, st.hinge]
        st.hinge_deg = torch.rad2deg(hinge)
        closed = hinge < math.radians(CLOSED_DEG)
        closed_in = closed & inside
        st.closed_in_once |= closed_in & live

        # Padlock: the short leg through both tabs (each in its own shell's frame).
        travel = st.lock.data.joint_pos.torch[:, st.shackle_joint]
        lock_pos = st.lock.data.body_pos_w.torch[:, st.lock_body] - origins
        sh_pos = st.lock.data.body_pos_w.torch[:, st.shackle_id] - origins
        sh_quat = st.lock.data.body_quat_w.torch[:, st.shackle_id]
        lpos = st.box.data.body_pos_w.torch[:, st.lid_id] - origins
        lquat = st.box.data.body_quat_w.torch[:, st.lid_id]

        def on_shackle(p):
            return sh_pos + math_utils.quat_apply(sh_quat, torch.tensor([p], device=dev).expand(n, 3))

        tip_w = on_shackle((PV.leg_x, 0.0, lock_props.SHORT_LEG_BOTTOM))
        top_w = on_shackle((PV.leg_x, 0.0, PV.bend_z - PV.rod / 2))
        half_t = props.TAB_T / 2
        tab_y = props.TAB_GAP + props.TAB_T / 2
        through_base = through_tab(math_utils.quat_apply_inverse(bquat, tip_w - bpos),
                                   math_utils.quat_apply_inverse(bquat, top_w - bpos), -tab_y, half_t)
        through_lid = through_tab(math_utils.quat_apply_inverse(lquat, tip_w - lpos),
                                  math_utils.quat_apply_inverse(lquat, top_w - lpos), tab_y, half_t)
        through = through_base & through_lid

        newly = ~st.latched & (travel < LATCH_TRAVEL_M)
        if newly.any():
            ids = newly.nonzero().flatten()
            limits = torch.tensor([[[0.0, 0.0005]]], device=dev).expand(len(ids), 1, 2).contiguous()
            st.lock.write_joint_position_limit_to_sim(limits, joint_ids=[st.shackle_joint], env_ids=ids,
                                                      warn_limit_violation=False)
            st.lock.set_joint_position_target(torch.zeros(len(ids), 1, device=dev), joint_ids=[st.shackle_joint],
                                              env_ids=ids)
            st.latched |= newly
            st.latched_through |= newly & through

        # Released: both grasp points clear of the padlock and of the lockout.
        flange_pos = st.robot.data.body_pos_w.torch[:, st.flange_ids]
        flange_quat = st.robot.data.body_quat_w.torch[:, st.flange_ids]
        down = torch.tensor([0.0, 0.0, -GRASP_OFFSET_M], device=dev).expand_as(flange_pos)
        grasp = flange_pos + math_utils.quat_apply(flange_quat, down) - origins[:, None]
        clear = (((grasp - lock_pos[:, None]).norm(dim=-1) > RELEASE_CLEAR_M).all(dim=1)
                 & ((grasp - bpos[:, None]).norm(dim=-1) > BOX_CLEAR_M).all(dim=1))
        st.released |= st.latched_through & closed_in & clear

        # Function test: pry the lid, then pull the cord.
        ready = live & ((st.latched_through & through & closed_in & clear) | st.force_test)
        st.clear_steps = torch.where(ready, st.clear_steps + 1, torch.zeros_like(st.clear_steps))
        start = ready & ((st.clear_steps >= st.wait_steps) | st.force_test)
        if start.any():
            ids = start.nonzero().flatten()
            st.phase[ids] = 1
            st.phase_steps[ids] = 0
            st.pry_max[ids] = hinge[ids]
        p1 = st.phase == 1
        if p1.any():
            ids = p1.nonzero().flatten()
            scale = ((st.phase_steps[ids].float() + 1) / st.ramp_len).clamp(max=1.0)
            axis = math_utils.quat_apply(bquat[ids], torch.tensor([[0.0, 1.0, 0.0]], device=dev).expand(len(ids), 3))
            torque = (PRY_NM * scale).unsqueeze(1) * axis
            st.box.permanent_wrench_composer.reset(env_ids=ids)
            st.box.permanent_wrench_composer.add_forces_and_torques_index(
                forces=torch.zeros_like(torque).unsqueeze(1), torques=torque.unsqueeze(1), body_ids=[st.lid_id],
                env_ids=ids, is_global=True)
            st.pry_max[ids] = torch.maximum(st.pry_max[ids], hinge[ids])
        p2 = st.phase == 2
        if p2.any():
            ids = p2.nonzero().flatten()
            scale = ((st.phase_steps[ids].float() + 1) / st.ramp_len).clamp(max=1.0)
            out = math_utils.quat_apply(bquat[ids], torch.tensor([[0.0, -1.0, 0.0]], device=dev).expand(len(ids), 3))
            force = (PULL_N * scale).unsqueeze(1) * out
            st.cord_t.permanent_wrench_composer.reset(env_ids=ids)
            st.cord_t.permanent_wrench_composer.add_forces_and_torques_index(
                forces=force.unsqueeze(1), torques=torch.zeros_like(force).unsqueeze(1), body_ids=[st.pull_link],
                env_ids=ids, is_global=True)
        in_test = st.phase > 0
        st.phase_steps = torch.where(in_test, st.phase_steps + 1, st.phase_steps)
        end1 = p1 & (st.phase_steps >= st.phase_len)
        if end1.any():
            ids = end1.nonzero().flatten()
            st.box.permanent_wrench_composer.reset(env_ids=ids)
            st.pry_held[ids] = st.pry_max[ids] < math.radians(PRY_HELD_DEG)
            st.phase[ids] = 2
            st.phase_steps[ids] = 0
        end2 = p2 & (st.phase_steps >= st.phase_len)
        if end2.any():
            ids = end2.nonzero().flatten()
            st.cord_t.permanent_wrench_composer.reset(env_ids=ids)
            st.pull_held[ids] = inside[ids]
            st.test_passed[ids] = (st.pry_held[ids] & st.pull_held[ids] & st.latched_through[ids] & through[ids]
                                   & (hinge[ids] < math.radians(PRY_HELD_DEG)))
            st.phase[ids] = 3
            for i in ids.tolist():
                print(f"[plug-lockout] after-release check: lid pried open to {math.degrees(float(st.pry_max[i])):.1f} "
                      f"deg (held: {bool(st.pry_held[i])}), plug inside after the cord pull: {bool(st.pull_held[i])}, "
                      f"padlock through both tabs: {bool(st.latched_through[i] & through[i])}, passed: "
                      f"{bool(st.test_passed[i])}", flush=True)

        dropped = pt[:, 2] < DROP_Z_M
        st.scorer.update(Step(unplugged & live, inside & live, closed_in & live, st.test_passed.clone(),
                              (st.s_d > DECOY_MOVED_M) & live, dropped))


@configclass
class EventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: PlugLockoutTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))
