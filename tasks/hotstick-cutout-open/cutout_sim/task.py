"""Hook-stick cutout: open a fuse cutout from a distance with a hook stick and leave the tube hanging in its hinge.

Mechanism (see usd/props.py for the geometry and sources):
- The cutout is one fixed-base articulation: base (post, crossarm, insulator, top contact, hinge, and the stick rack),
  carrier (prismatic z, the trunnion's seat in the hinge, 0..25 mm) and tube (revolute y at the hinge axis).
- Upper latch (rule): while latched, the tube's joint drive is a detent: a stiff spring toward closed whose torque
  is capped at T_HOLD, so the tube stays shut until the pull at the ring exceeds RELEASE_N (6 N along the ring's
  path), then moves; past LATCH_ZONE_DEG the drive is switched off and the tube falls open by its own weight (its top
  leans 20 deg toward the robot). Pushed back below RELATCH_DEG it latches again. While latched the carrier is held
  seated (the upper contact's spring pressing the cap down); unlatched, only the tube's weight seats the trunnion,
  so lifting the open tube by the ring lifts it out of the hinge.
- The hook stick is a free rigid body lying in its rack (two U-cradles and an end stop on the table).

Scoring (scorer.py): milestones hooked / opened / swung open 140 deg / hanging with the prong out (2 s, ends the
episode); hazards (keep-out 15 cm round the top contact, trunnion lifted out of the hinge, a stick nobody holds
touching the cutout) make the reported score 0.

``CUTOUT_CONTROL=latch`` (task file) runs the latch check with no robot: a torque about the hinge equal to a force at
the ring along its path, ramped 0 -> 10 N over 5 s, and prints the force at which the tube let go.
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

from cutout_sim import spec as S
from cutout_sim.kit import GRASP_OFFSET_M, Fingertips, reset_robot_home
from cutout_sim.scorer import MILESTONES, Scorer, Step
from cutout_sim.usd import props as P

_ASSETS = Path(__file__).resolve().parent / "assets"
MM = 0.001
DEG = math.pi / 180.0

R_RING = math.hypot(*P.tilt(*P.RING_C))
"""Hinge axis to ring centre (m)."""
TAU_GRAVITY = 0.0384
"""The closed tube's own opening torque about the hinge (0.15 kg, centre of mass 26 mm in front of the hinge axis)."""
T_HOLD = S.RELEASE_N * R_RING + TAU_GRAVITY
LATCH_K = T_HOLD / (1.0 * DEG)
"""Detent spring: reaches its cap within 1 deg of closed."""
SWING_DAMPING = 0.004
LIFT_K_LATCHED = 5000.0
LIFT_K_FREE = 250.0
LIFT_PRELOAD_M = 0.02
"""The lower contact's backup springs (rule): the carrier is driven toward 20 mm below its seat with 250 N/m, so the
seated trunnion is pressed down with 5 N and lifting it to the hinge hooks' 8 mm depth takes 7 N (plus the tube's
weight). While latched the drive is stiff (the upper contact's spring on the cap holds the tube seated)."""
LIFT_DAMPING = 2.0
LIFT_EFFORT = 40.0
TEST_RAMP_S = 5.0
DEBUG_EVERY = int(os.environ.get("CUTOUT_DEBUG", "0"))
TEST_MAX_N = 10.0


def stick_start_pose() -> tuple:
    return (P.STICK_BUTT[0], P.STICK_BUTT[1], P.STICK_AXIS_Z, 0.0, 0.0, 0.0, 1.0)


TUBE_WEIGHT_N = (P.TUBE_MASS + P.CARRIER_MASS) * 9.81


def hooked_stick_pose() -> tuple:
    """The stick with its prong through the closed tube's ring (pitched 32 deg up), for the hazard checks."""
    a = math.radians(P.PRONG_SWEEP_DEG)
    e = (P.PRONG_ROOT[0] * MM - 0.022 * math.sin(a), 0.022 * math.cos(a))
    b = math.radians(32.0)
    ring = [h + r for h, r in zip(P.HINGE, P.ring_center_hinge())]
    x = ring[0] - e[0] * math.cos(b)
    z = ring[2] - e[0] * math.sin(b)
    return (x, ring[1] - e[1], z, 0.0, -math.sin(b / 2), 0.0, math.cos(b / 2))


def _act(expr, stiffness, damping, effort=None):
    kw = {} if effort is None else {"joint_effort_limit": effort}
    return ImplicitActuatorCfg(joint_names_expr=[expr], stiffness=stiffness, damping=damping, **kw)


class Cutout(Object):
    def __init__(self):
        super().__init__(name="cutout", usd_path=str(_ASSETS / "cutout.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=P.HINGE, rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {"lift": _act("lift_joint", LIFT_K_LATCHED, LIFT_DAMPING, LIFT_EFFORT),
                         "swing": _act("swing_joint", LATCH_K, SWING_DAMPING, T_HOLD)}
        cfg.init_state.joint_pos = {"lift_joint": 0.0, "swing_joint": 0.0}
        return cfg


class Stick(Object):
    def __init__(self):
        p = stick_start_pose()
        super().__init__(name="stick", usd_path=str(_ASSETS / "stick.usda"), object_type=ObjectType.RIGID,
                         initial_pose=Pose(position_xyz=p[:3], rotation_xyzw=p[3:]))


def all_props() -> list:
    return [Cutout(), Stick()]


OBJECT_NAMES = ("cutout", "stick")


def _tprime(p):
    """Hinge-frame points (..., 3) -> frame T (x', y, z'), metres."""
    c, s = math.cos(P.TILT_DEG * DEG), math.sin(P.TILT_DEG * DEG)
    x, y, z = p.unbind(-1)
    return torch.stack([x * c + z * s, y, -x * s + z * c], -1)


def _in_box(p, lo, hi, margin):
    lo = torch.as_tensor(lo, device=p.device, dtype=p.dtype)
    hi = torch.as_tensor(hi, device=p.device, dtype=p.dtype)
    return ((p > lo - margin) & (p < hi + margin)).all(dim=-1)


def _seg_dist(p, a, b):
    """Distance from points p (..., 3) to segments a-b (broadcast)."""
    ab = b - a
    t = (((p - a) * ab).sum(-1) / (ab * ab).sum(-1).clamp_min(1e-12)).clamp(0.0, 1.0)
    return (p - (a + t.unsqueeze(-1) * ab)).norm(dim=-1)


class _TaskState:
    def __init__(self, env: ManagerBasedEnv):
        n, dev = env.num_envs, env.device
        self.robot = env.scene["robot"]
        self.cutout = env.scene["cutout"]
        self.stick = env.scene["stick"]
        self.j_lift = self.cutout.find_joints("lift_joint")[0][0]
        self.j_swing = self.cutout.find_joints("swing_joint")[0][0]
        self.tube_body = self.cutout.find_bodies("tube")[0][0]
        self.tips = Fingertips(self.robot)
        self.flange_ids, _ = self.robot.find_bodies(["left_gripper", "right_gripper"], preserve_order=True)
        self.finger_joints, _ = self.robot.find_joints(["left_joint7", "right_joint7"], preserve_order=True)
        self.arm_body_ids = list(range(self.robot.num_bodies))
        self.hold_steps = max(1, round(S.HOLD_S / env.step_dt))
        self.scorer = Scorer(n, self.hold_steps, dev)
        self.last_reports: dict[int, dict] = {}
        self.latched = torch.ones(n, dtype=torch.bool, device=dev)
        self.max_open = torch.zeros(n, device=dev)
        self.max_lift = torch.zeros(n, device=dev)
        self.min_keep_out = torch.full((n,), 9.0, device=dev)
        self.force_test = torch.zeros(n, dtype=torch.bool, device=dev)
        self.test_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.release_force = torch.full((n,), -1.0, device=dev)
        t = lambda v: torch.tensor(v, device=dev, dtype=torch.float32)  # noqa: E731
        self.hinge = t(P.HINGE)
        self.top_contact = self.hinge + t(P.top_contact_hinge())
        self.ring_t = t([P.RING_C[0] * MM, 0.0, P.RING_C[1] * MM])  # frame T of the tube
        self.prong = t(P.prong_points_stick())  # (5, 3) stick frame
        axis = [(x * MM, 0.0, 0.0) for x in range(0, 376, 25)]
        self.stick_axis = t(axis)
        self.stick_r = torch.full((len(axis),), P.STICK_HALF * MM, device=dev)
        self.grip_seg = (t([0.0, 0.0, 0.0]), t([P.HEAD[0] * MM, 0.0, 0.0]))
        # Cutout regions for "a loose stick on the cutout": tube (frame T of the tube), base (frame T of the hinge),
        # crossarm/post (hinge frame, axis-aligned). Metres.
        self.tube_boxes = [((-0.011, -0.012, -0.006), (0.011, 0.012, 0.150)), ((-0.039, -0.002, 0.111), (-0.009, 0.002, 0.137))]
        self.base_boxes_t = [((-0.016, -0.022, -0.014), (0.064, 0.022, 0.172))]
        self.base_boxes_h = [tuple(zip(*P.CROSSARM)), tuple(zip(*P.POST))]


class CutoutTask(TaskBase):
    def __init__(self, episode_length_s: float = S.EPISODE_SECONDS):
        super().__init__(episode_length_s=episode_length_s, task_description=S.instruction())

    def state(self, env: ManagerBasedEnv) -> _TaskState:
        if getattr(env, "_task_state", None) is None:
            env._task_state = _TaskState(env)
        return env._task_state

    def force_latch_test(self, env: ManagerBasedEnv) -> None:
        self.state(env).force_test[:] = True

    # Hazard checks (CUTOUT_CONTROL=hazards; the policy cutout-demo -P mode=hazards moves the left gripper into the
    # keep-out during steps 160-200 and 240-280). Each check: clear the scorer, set the milestone (0 or 75), cause
    # the hazard physically, print the reported score.
    HAZARD_PLAN = ((10, 0, "off_hinge"), (60, 0, "stick_on_cutout"), (150, 0, "keep_out"),
                   (225, 75, "keep_out"), (330, 75, "none"), (380, 75, "off_hinge"),
                   (430, 75, "stick_on_cutout"))
    HAZARD_REPORT_AFTER = 45

    def hazard_check_step(self, env, k: int) -> None:
        st = self.state(env)
        dev = env.device
        for start, pre, name in self.HAZARD_PLAN:
            if k == start:
                st.scorer.reset()
                st.scorer.step[:] = k
                st.scorer.milestone[:] = pre
                st.cutout.permanent_wrench_composer.reset()
                ids = torch.arange(env.num_envs, device=dev)
                q = torch.zeros(env.num_envs, 2, device=dev)
                st.cutout.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=[st.j_lift, st.j_swing], env_ids=ids)
                self._set_latched(st, torch.ones(env.num_envs, dtype=torch.bool, device=dev))
                p = torch.tensor(stick_start_pose(), device=dev).unsqueeze(0).repeat(env.num_envs, 1)
                p[:, :3] += env.scene.env_origins
                if name == "stick_on_cutout":
                    p = torch.tensor(hooked_stick_pose(), device=dev).unsqueeze(0).repeat(env.num_envs, 1)
                    p[:, :3] += env.scene.env_origins
                st.stick.write_root_pose_to_sim(p, env_ids=ids)
                st.stick.write_root_velocity_to_sim(torch.zeros(env.num_envs, 6, device=dev), env_ids=ids)
                if name == "off_hinge":  # unlatch, then lift the tube by its ring with 12 N
                    self._set_latched(st, torch.zeros(env.num_envs, dtype=torch.bool, device=dev))
                    f = torch.zeros(env.num_envs, 1, 3, device=dev)
                    f[..., 2] = 12.0 + TUBE_WEIGHT_N
                    st.cutout.permanent_wrench_composer.add_forces_and_torques_index(
                        forces=f, torques=torch.zeros_like(f), body_ids=[st.tube_body], env_ids=ids, is_global=True)
                print(f"HAZARD CHECK start {name}: milestone set to {pre}", flush=True)
            if k == start + self.HAZARD_REPORT_AFTER:
                st.cutout.permanent_wrench_composer.reset()
                r = self._report(st, 0)
                print(f"HAZARD CHECK {name} with milestone {pre}: reported score {r['milestone'] / 100:.2f}, "
                      f"events {r['semantic_events']}, lift {r['max_lift_mm']} mm, robot to top contact "
                      f"{r['closest_robot_to_top_contact_cm']} cm", flush=True)
                st.max_lift[:] = 0.0
                st.min_keep_out[:] = 9.0

    def episode_report(self, env: ManagerBasedEnv, env_id: int = 0, ended: bool = False) -> dict:
        st = self.state(env)
        if ended and env_id in st.last_reports:
            return st.last_reports.pop(env_id)
        return self._report(st, env_id)

    def _report(self, st: _TaskState, env_id: int) -> dict:
        r = st.scorer.report(env_id)
        q = st.cutout.data.joint_pos.torch[env_id]
        r.update(
            tube_open_deg=round(-float(q[st.j_swing]) / DEG, 1),
            trunnion_lift_mm=round(float(q[st.j_lift]) * 1000, 1),
            latched=bool(st.latched[env_id]),
            max_open_deg=round(float(st.max_open[env_id]) / DEG, 1),
            max_lift_mm=round(float(st.max_lift[env_id]) * 1000, 1),
            closest_robot_to_top_contact_cm=round(float(st.min_keep_out[env_id]) * 100, 1),
            hang_steps=int(st.scorer.hang_steps[env_id]),
            latch_test_release_n=round(float(st.release_force[env_id]), 2),
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
    def _set_latched(self, st: _TaskState, latched: torch.Tensor, env_ids: torch.Tensor | None = None) -> None:
        ids = torch.arange(st.latched.numel(), device=latched.device) if env_ids is None else env_ids
        k = torch.stack([torch.where(latched, LIFT_K_LATCHED, LIFT_K_FREE), torch.where(latched, LATCH_K, 0.0)], -1).float()
        st.cutout.write_joint_stiffness_to_sim_index(stiffness=k, joint_ids=[st.j_lift, st.j_swing], env_ids=ids)
        st.latched[ids] = latched

    def reset(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        reset_robot_home(env, env_ids)
        for env_id in env_ids.tolist():
            if int(st.scorer.step[env_id]) > 0:
                st.last_reports[env_id] = self._report(st, env_id)
        dev, n = env.device, len(env_ids)
        origins = env.scene.env_origins
        st.scorer.reset(env_ids)
        st.max_open[env_ids] = 0.0
        st.max_lift[env_ids] = 0.0
        st.min_keep_out[env_ids] = 9.0
        st.force_test[env_ids] = False
        st.test_steps[env_ids] = 0
        st.release_force[env_ids] = -1.0
        st.cutout.permanent_wrench_composer.reset(env_ids=env_ids)
        st.stick.permanent_wrench_composer.reset(env_ids=env_ids)
        q = torch.zeros(n, 2, device=dev)
        st.cutout.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=[st.j_lift, st.j_swing], env_ids=env_ids)
        target = q.clone()
        target[:, 0] = -LIFT_PRELOAD_M
        st.cutout.set_joint_position_target(target, joint_ids=[st.j_lift, st.j_swing], env_ids=env_ids)
        self._set_latched(st, torch.ones(n, dtype=torch.bool, device=dev), env_ids)
        p = torch.tensor(stick_start_pose(), device=dev, dtype=torch.float32).unsqueeze(0).repeat(n, 1)
        p[:, :3] += origins[env_ids]
        st.stick.write_root_pose_to_sim(p, env_ids=env_ids)
        st.stick.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)

    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        origins = env.scene.env_origins
        dev = env.device
        n = env.num_envs
        jp = st.cutout.data.joint_pos.torch
        jv = st.cutout.data.joint_vel.torch
        theta = -jp[:, st.j_swing]
        theta_dot = -jv[:, st.j_swing]
        lift = jp[:, st.j_lift]
        st.max_open = torch.maximum(st.max_open, theta)
        st.max_lift = torch.maximum(st.max_lift, lift)

        # Latch rule.
        release = st.latched & (theta > S.LATCH_ZONE_DEG * DEG)
        relatch = ~st.latched & (theta < S.RELATCH_DEG * DEG)
        if (release | relatch).any():
            self._set_latched(st, torch.where(release, False, torch.where(relatch, True, st.latched)))

        # Latch check with no robot (control mode): torque about the hinge = ramped force at the ring; removed once
        # the tube lets go, so it then falls open by its own weight.
        testing = st.force_test & (st.release_force < 0)
        if st.force_test.any():
            ids = st.force_test.nonzero().flatten()
            st.test_steps[ids] += 1
            f = (st.test_steps[ids].float() * env.step_dt / TEST_RAMP_S).clamp(max=1.0) * TEST_MAX_N
            torque = torch.zeros(len(ids), 1, 3, device=dev)
            torque[:, 0, 1] = torch.where(testing[ids], -f * R_RING, torch.zeros_like(f))
            st.cutout.permanent_wrench_composer.reset(env_ids=ids)
            st.cutout.permanent_wrench_composer.add_forces_and_torques_index(
                forces=torch.zeros_like(torque), torques=torque, body_ids=[st.tube_body], env_ids=ids, is_global=True)
            first = testing[ids] & ~st.latched[ids]
            if first.any():
                st.release_force[ids[first]] = f[first]
                print(f"LATCH TEST: let go at {float(f[first][0]):.2f} N at the ring "
                      f"(tube {float(theta[ids[first]][0]) / DEG:.1f} deg)", flush=True)

        # Robot points: every arm body origin, the midpoints between consecutive ones, and the finger pads.
        bp = st.robot.data.body_pos_w.torch - origins[:, None]
        mids = (bp[:, 1:] + bp[:, :-1]) / 2
        pads = st.tips.points(origins)
        flange_pos = st.robot.data.body_pos_w.torch[:, st.flange_ids]
        flange_quat = st.robot.data.body_quat_w.torch[:, st.flange_ids]
        down = torch.tensor([0.0, 0.0, -GRASP_OFFSET_M], device=dev).expand_as(flange_pos)
        tcp = flange_pos + math_utils.quat_apply(flange_quat, down) - origins[:, None]  # (N, 2, 3)
        pts = torch.cat([bp, mids, pads, tcp], dim=1)
        d_top = (pts - st.top_contact).norm(dim=-1).min(dim=1).values
        st.min_keep_out = torch.minimum(st.min_keep_out, d_top)
        keep_out = d_top < S.KEEP_OUT_M

        # Stick geometry.
        s_pos = st.stick.data.root_pos_w.torch - origins
        s_quat = st.stick.data.root_quat_w.torch
        def to_world(local):  # (K, 3) -> (N, K, 3)
            k = local.shape[0]
            return s_pos[:, None] + math_utils.quat_apply(s_quat[:, None].expand(n, k, 4), local.expand(n, k, 3))
        prong_w = to_world(st.prong)
        axis_w = to_world(st.stick_axis)
        tube_pos = st.cutout.data.body_pos_w.torch[:, st.tube_body] - origins
        tube_quat = st.cutout.data.body_quat_w.torch[:, st.tube_body]
        def in_tube(p):  # world (N, K, 3) -> tube frame T
            k = p.shape[1]
            local = math_utils.quat_apply_inverse(tube_quat[:, None].expand(n, k, 4), p - tube_pos[:, None])
            return _tprime(local)
        prong_t = in_tube(prong_w)
        a, b = prong_t[:, 0], prong_t[:, -1]
        cross = (a[:, 1] * b[:, 1]) < 0
        t = a[:, 1] / (a[:, 1] - b[:, 1]).where((a[:, 1] - b[:, 1]).abs() > 1e-9, torch.ones_like(a[:, 1]))
        pc = a + t.unsqueeze(-1) * (b - a)
        d_ring = (pc[:, [0, 2]] - st.ring_t[[0, 2]]).norm(dim=-1)
        hooked = cross & (d_ring < P.RING_R_IN * MM)
        unhooked = ((prong_t - st.ring_t).norm(dim=-1) > S.UNHOOKED_MM * MM).all(dim=1)

        # Held: a gripper's TCP within 2.5 cm of the stick's axis (butt to head) with its fingers closed past 60 %.
        ga, gb = to_world(torch.stack(st.grip_seg))[:, 0], to_world(torch.stack(st.grip_seg))[:, 1]
        d_tcp = _seg_dist(tcp, ga[:, None], gb[:, None])  # (N, 2)
        fingers = st.robot.data.joint_pos.torch[:, st.finger_joints] / 0.04695
        held = ((d_tcp < 0.025) & (fingers < 0.6)).any(dim=1)
        hands_clear = (d_tcp > S.HANDS_CLEAR_M).all(dim=1)

        # In the rack.
        ax = math_utils.quat_apply(s_quat, torch.tensor([[1.0, 0.0, 0.0]], device=dev).expand(n, 3))
        mid = s_pos + 0.30 * ax
        in_rack = ((ax[:, 2].abs() < math.sin(5 * DEG)) & (ax[:, 1].abs() < math.sin(8 * DEG))
                   & (s_pos[:, 0] > S.RACK_BUTT_X[0]) & (s_pos[:, 0] < S.RACK_BUTT_X[1])
                   & ((s_pos[:, 1] - P.STICK_BUTT[1]).abs() < S.RACK_TOL_Y_M)
                   & ((mid[:, 1] - P.STICK_BUTT[1]).abs() < S.RACK_TOL_Y_M)
                   & ((s_pos[:, 2] - P.STICK_AXIS_Z).abs() < S.RACK_TOL_Z_M)
                   & ((mid[:, 2] - P.STICK_AXIS_Z).abs() < S.RACK_TOL_Z_M))

        # A stick nobody holds touching the cutout.
        near = torch.zeros(n, dtype=torch.bool, device=dev)
        pr = P.PRONG_W / 2 * MM
        groups = ((axis_w, st.stick_r[None]), (prong_w, torch.full((1, prong_w.shape[1]), pr, device=dev)))
        for pw, rad in groups:
            m = rad.unsqueeze(-1) + S.NEAR_CUTOUT_M
            pt = in_tube(pw)
            for lo, hi in st.tube_boxes:
                near |= _in_box(pt, lo, hi, m).any(dim=1)
            ph = pw - st.hinge
            for lo, hi in st.base_boxes_t:
                near |= _in_box(_tprime(ph), lo, hi, m).any(dim=1)
            for lo, hi in st.base_boxes_h:
                near |= _in_box(ph, lo, hi, m).any(dim=1)
        stick_on_cutout = near & ~held
        stick_loose = ~held & ~in_rack & (s_pos - torch.tensor(stick_start_pose()[:3], device=dev)).norm(dim=-1).gt(0.01)

        hanging = ((theta > S.HANGING_DEG * DEG) & (theta_dot.abs() < S.STILL_DEG_S * DEG)
                   & (lift < S.SEATED_MM * MM) & unhooked)
        off_hinge = lift > S.OFF_HINGE_MM * MM
        opened = theta > S.OPENED_DEG * DEG
        test = st.force_test
        if DEBUG_EVERY and int(st.scorer.step[0]) % DEBUG_EVERY == 0:
            print(f"DBG k={int(st.scorer.step[0])} open {float(theta[0]) / DEG:.1f} deg ({float(theta_dot[0]) / DEG:.0f}/s) "
                  f"lift {float(lift[0]) * 1000:.1f} mm latched {bool(st.latched[0])} top-contact {float(d_top[0]) * 100:.1f} cm "
                  f"hooked {bool(hooked[0])} ring-d {float(d_ring[0]) * 1000:.1f} unhooked {bool(unhooked[0])} held {bool(held[0])} "
                  f"rack {bool(in_rack[0])} clear {bool(hands_clear[0])} near {bool(near[0])} "
                  f"stick {[round(v, 3) for v in s_pos[0].tolist()]} axis {[round(v, 3) for v in ax[0].tolist()]} "
                  f"prong-dir {[round(v, 2) for v in math_utils.quat_apply(s_quat[:1], torch.tensor([[0.0, 1.0, 0.0]], device=dev))[0].tolist()]} "
                  f"milestone {int(st.scorer.milestone[0])}", flush=True)
        swung = theta > S.HANGING_DEG * DEG
        st.scorer.update(Step(hooked, opened, swung, hanging, keep_out & ~test, off_hinge & ~test,
                              stick_on_cutout & ~test, stick_loose & ~test))


@configclass
class EventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: CutoutTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))
