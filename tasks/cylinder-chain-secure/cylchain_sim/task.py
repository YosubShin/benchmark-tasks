"""Cylinder chain task: chain an MD oxygen cylinder into a Bristol Maid CYB/DE2 wall bracket (props: cylchain_sim/usd).

Mechanism (all geometry and contact; no rule moves or holds any prop):
- The cylinder is a free rigid body (3.3 kg, 111 mm, centre of gravity 0.20 m up) standing in the bracket's left U
  cradle. About 4 N sideways at its shoulder tips it.
- The chain is a fixed-base articulation hanging from the hole in the centre tongue (the bay's right): long links
  whose four-wire rings collide, so a link dropped over the J-hook's upturned tip is threaded on the hook and held
  by the hook's geometry alone. Which link goes on the hook sets how snug the chain is. The bracket has no spring
  gate: a link lifted back over the tip comes off.

Measured each step (see scorer.py for the milestones): where the chain crosses the cylinder's front (height, gap),
which link (if any) is threaded on the J-hook, the chain's slack (the hooked link's distance along the chain from
the fixed end minus the taut path round the cylinder's front to the hook), the cylinder's tilt and its gap to the
cradle's back, and the finger pads' distance to the valve.

After-release check (the success check), once a link is on the hook and every finger pad has stayed at least
RELEASE_CLEAR from the cylinder and the chain for TEST_WAIT_S without a break (the count restarts on any approach):
pull the cylinder's shoulder (SHOULDER_PUSH_Z) straight toward the robot with TEST_PULL_N for TEST_S; passed if its
tilt stays below TEST_MAX_TILT_DEG and the 75 conditions held when the check started. ``force_test(env)`` starts
the same check at once (control runs).
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

from cylchain_sim import spec as S
from cylchain_sim.kit import Fingertips, reset_robot_home
from cylchain_sim.scorer import MILESTONES, Scorer, Step

_ASSETS = Path(__file__).resolve().parent / "assets"

CHAIN_DAMPING = 2e-4


class Bracket(Object):
    def __init__(self):
        super().__init__(name="bracket", usd_path=str(_ASSETS / "bracket.usda"), object_type=ObjectType.BASE,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))


class Cylinder(Object):
    def __init__(self):
        super().__init__(name="cylinder", usd_path=str(_ASSETS / "cylinder.usda"), object_type=ObjectType.RIGID,
                         initial_pose=Pose(position_xyz=(*S.CYL_XY, 0.0005), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))


class Chain(Object):
    def __init__(self):
        super().__init__(name="chain", usd_path=str(_ASSETS / "chain.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=S.ANCHOR, rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {
            "links": ImplicitActuatorCfg(joint_names_expr=["j.*"], stiffness=0.0, damping=CHAIN_DAMPING),
        }
        cfg.init_state.joint_pos = {".*": 0.0}
        return cfg


def all_props() -> list:
    return [Bracket(), Cylinder(), Chain()]


def _hook_samples(n_per=4):
    """Points along the J-hook's bend and upturned tip (desk frame): a link threaded on the hook encloses one."""
    pts = S.hook_points()[1:]
    out = []
    for a, b in zip(pts[:-1], pts[1:]):
        for t in range(n_per):
            u = t / n_per
            out.append(tuple(a[i] + u * (b[i] - a[i]) for i in range(3)))
    out.append(pts[-1])
    return out


class _TaskState:
    def __init__(self, env: ManagerBasedEnv, task: "CylinderChainTask"):
        n, dev = env.num_envs, env.device
        self.robot = env.scene["robot"]
        self.cyl = env.scene["cylinder"]
        self.chain = env.scene["chain"]
        self.link_ids = [self.chain.find_bodies(f"link_{k:02d}")[0][0] for k in range(S.N_LINKS)]
        self.anchor_id = self.chain.find_bodies("anchor")[0][0]
        self.joint_ids = list(range(self.chain.num_joints))
        self.tips = Fingertips(self.robot)
        self.scorer = Scorer(n, dev)
        self.last_reports: dict[int, dict] = {}
        zb = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
        zf = lambda: torch.zeros(n, device=dev)  # noqa: E731
        self.force, self.test_passed, self.secured_at_start, self.secured_now = zb(), zb(), zb(), zb()
        self.clear_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.hook_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.hook_hold = max(1, round(0.5 / env.step_dt))
        self.phase = torch.zeros(n, dtype=torch.long, device=dev)
        self.phase_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.test_tilt_max = zf()
        self.wait_steps = max(1, round(S.TEST_WAIT_S / env.step_dt))
        self.test_len = max(1, round(S.TEST_S / env.step_dt))
        self.m = {k: zf() for k in ("tilt", "cross_z", "cross_gap", "slack", "gap_top", "gap_bot", "gate", "pad_cyl",
                                    "pad_chain", "valve_d")}
        self.hook_pts = torch.tensor(_hook_samples(), device=dev)
        bend = S.hook_points()[2]
        self.hook_bend = torch.tensor([bend[0], bend[1], bend[2] + S.HOOK_R], device=dev)[None]


def _tilt(quat):
    z = math_utils.quat_apply(quat, torch.tensor([0.0, 0.0, 1.0], device=quat.device).expand(quat.shape[0], 3))
    return torch.rad2deg(torch.acos(z[:, 2].clamp(-1.0, 1.0))), z


def _taut_length(anchor, eye, centre_xy):  # eye: the hook (on the left)
    """Batched spec.taut_path_length (torch)."""
    r = S.CYL_R + 0.004

    def tangent(p, sign):
        d = (p[:, :2] - centre_xy)
        dist = d.norm(dim=-1).clamp(min=r + 1e-6)
        t = (dist ** 2 - r ** 2).sqrt()
        ang = torch.atan2(d[:, 1], d[:, 0]) + sign * torch.acos(r / dist)
        return t, ang

    ta, aa = tangent(anchor, -1.0)  # fixed end on the cylinder's right: tangent point further round to the front
    tb, ab = tangent(eye, 1.0)
    two_pi = 2 * math.pi
    arc = torch.remainder(aa, two_pi) - torch.remainder(ab, two_pi)
    arc = torch.where(arc < 0, arc + two_pi, arc)
    horiz = ta + tb + r * arc
    return torch.sqrt(horiz ** 2 + (anchor[:, 2] - eye[:, 2]) ** 2)


class CylinderChainTask(TaskBase):
    def __init__(self, episode_length_s: float = S.EPISODE_SECONDS):
        super().__init__(episode_length_s=episode_length_s, task_description=S.instruction())

    def state(self, env: ManagerBasedEnv) -> _TaskState:
        if getattr(env, "_task_state", None) is None:
            env._task_state = _TaskState(env, self)
        return env._task_state

    def force_test(self, env: ManagerBasedEnv) -> None:
        self.state(env).force[:] = True

    def episode_report(self, env: ManagerBasedEnv, env_id: int = 0, ended: bool = False) -> dict:
        st = self.state(env)
        if ended and env_id in st.last_reports:
            return st.last_reports.pop(env_id)
        return self._report(st, env_id)

    def _report(self, st: _TaskState, env_id: int) -> dict:
        r = st.scorer.report(env_id)
        f = lambda k, s=1.0, nd=1: round(float(st.m[k][env_id]) * s, nd)  # noqa: E731
        r.update(tilt_deg=f("tilt"), chain_front_height_mm=f("cross_z", 1000), chain_front_gap_mm=f("cross_gap", 1000),
                 slack_mm=f("slack", 1000), seat_gap_top_mm=f("gap_top", 1000), seat_gap_bottom_mm=f("gap_bot", 1000),
                 hooked_link=int(st.m["gate"][env_id]), pad_to_cylinder_mm=f("pad_cyl", 1000), pad_to_chain_mm=f("pad_chain", 1000),
                 pad_to_valve_mm=f("valve_d", 1000), secured_now=bool(st.secured_now[env_id]),
                 test_phase=int(st.phase[env_id]), test_tilt_max_deg=round(float(st.test_tilt_max[env_id]), 2),
                 secured_at_test_start=bool(st.secured_at_start[env_id]), test_passed=bool(st.test_passed[env_id]))
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

    # Events ------------------------------------------------------------------------------------------------
    def reset(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        reset_robot_home(env, env_ids)
        for env_id in env_ids.tolist():
            if int(st.scorer.step[env_id]) > 0:
                st.last_reports[env_id] = self._report(st, env_id)
        dev, n = env.device, len(env_ids)
        origins = env.scene.env_origins
        st.scorer.reset(env_ids)
        for buf in (st.force, st.test_passed, st.secured_at_start, st.secured_now):
            buf[env_ids] = False
        for buf in (st.clear_steps, st.phase, st.phase_steps, st.hook_steps):
            buf[env_ids] = 0
        st.test_tilt_max[env_ids] = 0.0
        st.cyl.permanent_wrench_composer.reset(env_ids=env_ids)
        pose = torch.tensor([[*S.CYL_XY, 0.0005, 0.0, 0.0, 0.0, 1.0]], device=dev).repeat(n, 1)
        pose[:, :3] += origins[env_ids]
        st.cyl.write_root_pose_to_sim(pose, env_ids=env_ids)
        st.cyl.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)
        q = torch.zeros(n, st.chain.num_joints, device=dev)
        st.chain.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
        st.chain.set_joint_position_target(q, env_ids=env_ids)

    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        origins = env.scene.env_origins
        dev = env.device
        n = env.num_envs
        idle = st.phase == 0

        cpos = st.cyl.data.root_pos_w.torch - origins
        cquat = st.cyl.data.root_quat_w.torch
        tilt, axis = _tilt(cquat)

        def axis_at(z):
            """Cylinder axis point at height z (desk frame)."""
            t = (z - cpos[:, 2]) / axis[:, 2].clamp(min=0.2)
            return cpos + axis * t[:, None]

        # Chain polyline: the fixed end, then each link's start (its joint) and the last link's far end.
        body_pos = st.chain.data.body_pos_w.torch - origins[:, None]
        lq = st.chain.data.body_quat_w.torch[:, st.link_ids]  # (N, L, 4)
        lp = body_pos[:, st.link_ids]
        nl = len(st.link_ids)
        tail = lp[:, -1] + math_utils.quat_apply(lq[:, -1], torch.tensor([S.LINK_PITCH, 0.0, 0.0], device=dev)
                                                 .expand(n, 3))
        pts = torch.cat([body_pos[:, st.anchor_id:st.anchor_id + 1], lp[:, 1:], tail[:, None]], dim=1)
        # Crossing of the vertical plane through the cylinder's axis, perpendicular to the wall (y = axis y), in front.
        a, b = pts[:, :-1], pts[:, 1:]
        ay = axis_at(torch.full((n,), 0.30, device=dev))[:, 1:2]
        da, db = a[..., 1] - ay, b[..., 1] - ay
        cross = (da * db) <= 0
        t = (-da / torch.where((db - da).abs() > 1e-9, db - da, torch.full_like(da, 1e-9))).clamp(0, 1)
        hit = a + t[..., None] * (b - a)
        ax_hit = torch.stack([axis_at(hit[:, k, 2]) for k in range(hit.shape[1])], dim=1)
        front = cross & (hit[..., 0] < ax_hit[..., 0])
        gap = (ax_hit[..., 0] - hit[..., 0]) - S.CYL_R
        big = torch.full_like(gap, 9.0)
        gap_f = torch.where(front, gap, big)
        k_best = gap_f.argmin(dim=1)
        rows = torch.arange(n, device=dev)
        cross_gap = gap_f[rows, k_best]
        cross_z = torch.where(cross_gap < 8.0, hit[rows, k_best, 2], torch.zeros_like(cross_gap))
        around = (cross_gap < 0.03) & (cross_z > 0.05) & (cross_z < S.CYL_BODY_H)
        in_band = (cross_z >= S.BAND[0]) & (cross_z <= S.BAND[1])

        # J-hook: a link is in the hook when a point of it (its two ends or its middle) lies in the hook's bend: between
        # the stem and the upturned tip, within a link's half-width of the hook's plane, below the tip's top. Held
        # there, the chain cannot leave sideways (stem and tip) or downward (the bend); only lifting frees it.
        ends = lp + math_utils.quat_apply(lq.reshape(-1, 4), torch.tensor([S.LINK_PITCH, 0.0, 0.0], device=dev)
                                          .expand(n * nl, 3)).reshape(n, nl, 3)
        cand = torch.stack([lp, ends, (lp + ends) / 2], dim=2)  # (N, L, 3, 3)
        hb = S.hook_points()
        x_lo, x_hi = hb[-1][0] + S.HOOK_R, hb[0][0] - S.HOOK_R
        z_lo, z_hi = hb[2][2] - S.HOOK_R - S.LINK_IN_W / 2 - S.LINK_WIRE, hb[-1][2] + S.HOOK_R
        in_bend = (cand[..., 0] > x_lo) & (cand[..., 0] < x_hi) & ((cand[..., 1] - S.HOOK_Y).abs() < 0.012) & \
            (cand[..., 2] > z_lo) & (cand[..., 2] < z_hi)
        on_hook = in_bend.any(dim=2)  # (N, L)
        hooked_now = on_hook.any(dim=1)
        st.hook_steps = torch.where(hooked_now, st.hook_steps + 1, torch.zeros_like(st.hook_steps))
        hooked = hooked_now & (st.hook_steps >= st.hook_hold)
        idx = torch.arange(nl, device=dev).expand(n, nl)
        k_hook = torch.where(on_hook, idx, torch.full_like(idx, 10 ** 6)).min(dim=1).values
        k_hook = torch.where(hooked, k_hook, torch.zeros_like(k_hook))

        # Slack and seat.
        # Slack: chain along the links from the fixed end to the middle of the hooked link, minus the taut path from
        # the fixed end round the cylinder's front to the hook's bend.
        anchor = pts[:, 0]
        cxy = axis_at(torch.full((n,), 0.30, device=dev))[:, :2]
        along = (k_hook.to(torch.float32) + 0.5) * S.LINK_PITCH
        slack = torch.where(hooked, along - S.TAUT_LEN, torch.full_like(along, 9.0))
        back_x = S.WALL_X - S.BAY_BACK - S.BUMPER_T
        gap_top = back_x - (axis_at(torch.full((n,), S.PLATE_TOP - 0.002, device=dev))[:, 0] + S.CYL_R)
        gap_bot = gap_top
        seated = (gap_top < S.SEATED_GAP) & (tilt < S.UPRIGHT_DEG)
        secured = hooked & in_band & (slack <= S.SLACK_MAX) & seated

        # Finger pads: distances to the valve, the cylinder and the chain.
        pads = st.tips.points(origins)  # (N, 4, 3)
        rel = math_utils.quat_apply_inverse(cquat[:, None].expand(n, 4, 4).reshape(-1, 4),
                                            (pads - cpos[:, None]).reshape(-1, 3)).reshape(n, 4, 3)
        lo = torch.tensor([S.VALVE[0][0], S.VALVE[1][0], S.VALVE[2][0]], device=dev)
        hi = torch.tensor([S.TOGGLE[0][1], S.VALVE[1][1], S.TOGGLE[2][1]], device=dev)
        hi[0] = S.VALVE[0][1] + 0.012
        d_out = torch.maximum(lo - rel, rel - hi).clamp(min=0.0).norm(dim=-1)
        valve_d = d_out.min(dim=1).values
        # The chain swung into the valve counts as knocking it too (5 mm).
        crel = math_utils.quat_apply_inverse(cquat[:, None].expand(n, pts.shape[1], 4).reshape(-1, 4),
                                             (pts - cpos[:, None]).reshape(-1, 3)).reshape(n, pts.shape[1], 3)
        chain_valve = torch.maximum(lo - crel, crel - hi).clamp(min=0.0).norm(dim=-1).min(dim=1).values
        valve = (valve_d < S.VALVE_MARGIN) | (chain_valve < 0.008)
        radial = rel[..., :2].norm(dim=-1) - S.CYL_R
        above = (rel[..., 2] - S.CYL_TOP).clamp(min=0.0)
        pad_cyl = torch.sqrt(radial.clamp(min=0.0) ** 2 + above ** 2).min(dim=1).values
        chain_pts = torch.cat([pts, (pts[:, :-1] + pts[:, 1:]) / 2], dim=1)
        pad_chain = (pads[:, :, None] - chain_pts[:, None]).norm(dim=-1).flatten(1).min(dim=1).values

        # After-release check.
        clear = (pad_cyl > S.RELEASE_CLEAR) & (pad_chain > S.RELEASE_CLEAR)
        ready = idle & ((hooked & clear) | st.force)
        st.clear_steps = torch.where(ready, st.clear_steps + 1, torch.zeros_like(st.clear_steps))
        start = ready & ((st.clear_steps >= st.wait_steps) | st.force)
        if start.any():
            st.secured_at_start = torch.where(start, secured, st.secured_at_start)
            st.phase = torch.where(start, torch.ones_like(st.phase), st.phase)
            st.phase_steps = torch.where(start, torch.zeros_like(st.phase_steps), st.phase_steps)
        testing = st.phase == 1
        if testing.any():
            ids = testing.nonzero().flatten()
            shoulder = cpos + math_utils.quat_apply(cquat, torch.tensor([0.0, 0.0, S.SHOULDER_PUSH_Z], device=dev)
                                                    .expand(n, 3))
            forces = torch.zeros(len(ids), 1, 3, device=dev)
            forces[..., 0] = -S.TEST_PULL_N
            st.cyl.permanent_wrench_composer.set_forces_and_torques_index(
                forces=forces, torques=torch.zeros_like(forces),
                positions=(shoulder[ids] + origins[ids])[:, None], body_ids=[0], env_ids=ids, is_global=True)
            st.test_tilt_max = torch.where(testing, torch.maximum(st.test_tilt_max, tilt), st.test_tilt_max)
            st.phase_steps = torch.where(testing, st.phase_steps + 1, st.phase_steps)
            end = testing & (st.phase_steps >= st.test_len)
            if end.any():
                eids = end.nonzero().flatten()
                st.cyl.permanent_wrench_composer.reset(env_ids=eids)
                st.test_passed[eids] = (st.test_tilt_max[eids] < S.TEST_MAX_TILT_DEG) & st.secured_at_start[eids]
                st.phase[eids] = 2

        for k, val in (("tilt", tilt), ("cross_z", cross_z), ("cross_gap", cross_gap), ("slack", slack),
                       ("gap_top", gap_top), ("gap_bot", gap_bot), ("gate", k_hook.to(torch.float32)), ("pad_cyl", pad_cyl),
                       ("pad_chain", pad_chain), ("valve_d", valve_d)):
            st.m[k] = val
        st.secured_now = secured
        st.scorer.update(Step(around & idle, hooked & idle, secured & idle, st.test_passed.clone(),
                              (tilt > S.TIP_HAZARD_DEG) & idle, valve & idle))


@configclass
class EventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: CylinderChainTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))
