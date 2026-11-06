"""Breaker lockout: the Master Lock 493B clamp-on lockout locks the named QO breaker through geometry.

What is geometry and what is a rule:
- geometry: the channel slides over the handle and the knob-driven jaw squeezes the handle against the fixed jaw
  (contact + friction hold the lockout on); the clamped 74 mm channel lies on the deadfront, so the handle cannot pivot
  toward ON; the closed lever covers the knob, and a shackle through the two end holes stops the lever opening.
- rules: the handle's over-centre detent (drive target flips past the middle +- DETENT_FLIP_DEG); the thumbscrew's
  screw coupling (jaw drive target p = -PITCH x turns / 2 pi; the clamp force is the drive's push against the handle)
  and thread friction (Coulomb drive on the knob, T = RUN_NM + NUT_K x d x F, self-locking); the lever's snap shut
  under 6 degrees and its cam (shut, it drives the jaw CAM_M further: the 493B tightens its grip when the lever is
  closed); the padlock's shackle spring and latch.

Milestones: 25 lockout seated on the named breaker's handle; 50 knob tightened there (jaw force >= CLAMP_N); 75 lever
shut (cammed: jaw force >= CAMMED_N) and the padlock's short leg through both holes; 100 the shackle latched through them. Success = 100 and the
after-release check passed.

After-release check (once both grasp points have been RELEASE_CLEAR_M from the lockout and the padlock for
TEST_WAIT_S after the latch; under the station harness it runs off camera, after the recording):
(A) pull the lever's middle outward (away from the panel) with LEVER_PRY_N for 1 s; (B) only if the lever came open
past KNOB_REACH_DEG (the knob is then reachable): turn the knob loosening with KNOB_TEST_NM for 1 s; (C) pull the
lockout straight off the panel with PULL_N for 1.5 s; (D) push the named handle toward ON with TEST_TORQUE_NM for 1 s.
Passes if the lever stayed shut (under LEVER_SHUT_DEG), the lockout moved less than TEST_MAX_PULL_M, and the handle
stayed on the OFF side of the middle. ``force_function_test`` starts the same check at once (control runs).
"""

from __future__ import annotations

import math
from dataclasses import MISSING
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
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

from clamp_lockout import spec as S
from clamp_lockout.kit import finger_points, grasp_points, in_box, reset_robot_home, reviewer_camera
from clamp_lockout.scorer import MILESTONES, RubricScorer
from clamp_lockout.usd import padlock_props
from clamp_lockout.usd import panel_props as P

_ASSETS = Path(__file__).resolve().parent / "assets"
IDENTITY = (0.0, 0.0, 0.0, 1.0)

HANDLE_K = 0.5
"""N m/rad: the handle's detent spring (QO handle force not published; TM). 0.24 N m tips a free handle past the
flip point."""
HANDLE_D = 0.005
DETENT_FLIP_DEG = 6.0
JAW_K = 1.5e5
"""N/m: clamp compliance (jaw, screw, handle): 0.1 mm of screw advance past contact adds 15 N (I)."""
JAW_D = 60.0
JAW_MAX_N = 300.0
KNOB_K = 5.0
KNOB_D = 0.0005
RUN_NM = 0.001
"""Running torque of the thumbscrew with no load (I/TM)."""
NUT_K = 0.2
SCREW_D = 0.008
"""T = NUT_K x F x d, M8 thumbscrew (I)."""
LEVER_K = 0.08
LEVER_D = 0.002
LEVER_SNAP_DEG = 6.0
LEVER_SLIP_DEG = 2.0
"""Pin friction: the lever holds its angle against its own weight (0.05 N m/rad x 2 deg = 0.002 N m)."""
PADLOCK_SPRING_N_PER_M = 150.0
LATCH_TRAVEL_M = 0.002
CLAMP_N = 2.0
"""Jaw force at which the knob counts as tightened ("turn the knob until the grips touch the toggle": 0.004 N m), held for CLAMP_HOLD_S."""
CLAMP_HOLD_S = 1.0
KNOB_CLEAR_M = 0.03
"""The knob's grip counts only with every fingertip this far from the knob (pressing the knob levers the lockout
against the handle and loads the jaw by itself)."""
CAM_M = 0.0001
"""Closing the lever cams the jaw a further 0.12 mm toward the handle (493B instructions, step 4: closing the handle
tightens the grip; the data sheet lists a plastic cam). The amount is inferred: +15 N with JAW_K."""
CAMMED_N = 10.0
"""Jaw force once the lever is shut over a tightened knob (milestone 75 needs it)."""
SEAT_X_TOL = 0.003
SEAT_TILT_TOL = math.radians(10.0)
DECOY_SEAT_S = 0.5
RELEASE_CLEAR_M = 0.06
TEST_WAIT_S = 1.0
LEVER_PRY_N = 20.0
KNOB_REACH_DEG = 20.0
KNOB_TEST_NM = 0.1
PULL_N = 20.0
TEST_TORQUE_NM = 0.3
"""20 N on the handle, 15 mm from its pivot, toward ON (a free handle flips to ON under it)."""
LEVER_SHUT_DEG = 5.0
TEST_PHASES_S = (1.0, 1.0, 1.5, 1.0)
TEST_MAX_PULL_M = 0.003
TEST_CATCH_M = 0.005
DECOY_MOVED_RAD = 0.002 / P.HANDLE_L
"""Another breaker's handle tip moved 2 mm (6 degrees)."""
DROP_Z_M = 0.05
TOUCH_MARGIN_M = 0.002
TIP_IN = 0.003
"""The handle point tested for 'in the jaws': 3 mm in from its tip."""


def _q(t, dev):
    return torch.tensor(t, dtype=torch.float32, device=dev)


def _pose(p) -> Pose:
    p = tuple(p) + (IDENTITY if len(p) == 3 else ())
    return Pose(position_xyz=p[:3], rotation_xyzw=p[3:])


class StaticProp(Object):
    def __init__(self, name: str, file: str):
        super().__init__(name=name, usd_path=str(_ASSETS / file), object_type=ObjectType.BASE,
                         initial_pose=_pose((0.0, 0.0, 0.0)))


class ArticulatedProp(Object):
    def __init__(self, name: str, file: str, pose, actuators: dict, joint_pos: dict):
        self._actuators, self._joint_pos = actuators, joint_pos
        super().__init__(name=name, usd_path=str(_ASSETS / file), object_type=ObjectType.ARTICULATION,
                         initial_pose=_pose(pose))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {k: ImplicitActuatorCfg(**v) for k, v in self._actuators.items()}
        cfg.init_state.joint_pos = dict(self._joint_pos)
        return cfg


PADLOCK = padlock_props.PADLOCKS["ml410"]
NAMED = S.make_seed(0).named_slot


def handle_init() -> dict:
    return {f"handle_{k}_joint": (P.off_angle(k) if k == NAMED else P.on_angle(k)) for k in range(len(S.CIRCUITS))}


def task_assets(layout=None) -> list[Object]:
    return [
        StaticProp("stands", "stands.usda"),
        ArticulatedProp("panel", "panel.usda", P.panel_origin() + IDENTITY,
                        {"handles": dict(joint_names_expr=["handle_.*_joint"], stiffness=HANDLE_K, damping=HANDLE_D)},
                        handle_init()),
        ArticulatedProp("lockout", "lockout.usda", P.lockout_start_world(),
                        {"jaw": dict(joint_names_expr=["jaw_joint"], stiffness=JAW_K, damping=JAW_D, armature=0.01),
                         "knob": dict(joint_names_expr=["knob_joint"], stiffness=KNOB_K, damping=KNOB_D,
                                      armature=1e-4),
                         "lever": dict(joint_names_expr=["lever_joint"], stiffness=LEVER_K, damping=LEVER_D,
                                       armature=1e-4)},
                        {"jaw_joint": 0.0, "knob_joint": 0.0, "lever_joint": math.radians(P.LEVER_OPEN_DEG)}),
        ArticulatedProp("padlock", "padlock_ml410.usda", P.padlock_start_world(),
                        {"shackle": dict(joint_names_expr=["shackle_joint"], stiffness=PADLOCK_SPRING_N_PER_M,
                                         damping=3.0)},
                        {"shackle_joint": padlock_props.OPEN_TRAVEL}),
    ]


class _State:
    def __init__(self, env: ManagerBasedEnv, task: "BreakerLockoutTask"):
        n, dev = env.num_envs, env.device
        self.n, self.dev = n, dev
        nb = len(S.CIRCUITS)
        self.nb = nb
        self.robot = env.scene["robot"]
        self.flange_ids, _ = self.robot.find_bodies(["left_gripper", "right_gripper"], preserve_order=True)
        self.finger_ids, _ = self.robot.find_joints(["left_joint7", "left_joint8", "right_joint7", "right_joint8"],
                                                    preserve_order=True)
        self.panel = env.scene["panel"]
        self.handle_joints = [self.panel.find_joints(f"handle_{k}_joint")[0][0] for k in range(nb)]
        self.handle_bodies = [self.panel.find_bodies(f"handle_{k}")[0][0] for k in range(nb)]
        self.lo = env.scene["lockout"]
        self.lo_body = self.lo.find_bodies("body")[0][0]
        self.lo_lever = self.lo.find_bodies("lever")[0][0]
        self.lo_knob = self.lo.find_bodies("knob")[0][0]
        self.jaw_joint = self.lo.find_joints("jaw_joint")[0][0]
        self.knob_joint = self.lo.find_joints("knob_joint")[0][0]
        self.lever_joint = self.lo.find_joints("lever_joint")[0][0]
        self.layout = S.make_seed(task.default_seed)
        self.next_layout = self.layout
        self.pv = PADLOCK
        self.lock = env.scene["padlock"]
        self.shackle_joint = self.lock.find_joints("shackle_joint")[0][0]
        self.lock_body = self.lock.find_bodies("body")[0][0]
        self.lock_shackle = self.lock.find_bodies("shackle")[0][0]
        self.scorer = RubricScorer("breaker_lockout", n, dev)
        self.last_reports: dict[int, dict] = {}
        self.step_dt = env.step_dt
        bools = ("lever_shut", "latched", "latched_through", "test_done", "test_passed", "test_caught", "force_test",
                 "knob_reached")
        for name in bools:
            setattr(self, name, torch.zeros(n, dtype=torch.bool, device=dev))
        self.handle_on = torch.zeros(n, nb, dtype=torch.bool, device=dev)
        for name in ("decoy_seat_steps", "clear_steps", "test_step", "tight_steps"):
            setattr(self, name, torch.zeros(n, dtype=torch.long, device=dev))
        for name in ("turn_prev", "turn_acc", "turn_hold", "force", "peak_force", "test_x0"):
            setattr(self, name, torch.zeros(n, device=dev))
        self.lever_hold = torch.full((n,), math.radians(P.LEVER_OPEN_DEG), device=dev)
        self.tight_turns = torch.zeros(n, device=dev)
        self.test_results = torch.zeros(n, 3, dtype=torch.bool, device=dev)
        self.test_values = torch.zeros(n, 4, device=dev)
        self.handle_ref = torch.zeros(n, nb, device=dev)
        self.on_sign = _q([P.on_sign(k) for k in range(nb)], dev)
        self.pivots = _q([P.pivot_world(k) for k in range(nb)], dev)
        self.lockout_start = _q(P.lockout_start_world()[:3], dev)
        self.phase_steps = [max(1, round(s / self.step_dt)) for s in TEST_PHASES_S]
        self.wait_steps = max(1, round(TEST_WAIT_S / self.step_dt))
        self.decoy_seat_limit = max(1, round(DECOY_SEAT_S / self.step_dt))
        self.tight_limit = max(1, round(CLAMP_HOLD_S / self.step_dt))
        self.dbg = {}


class BreakerLockoutTask(TaskBase):
    REVIEWER_EYE = (0.12, 0.10, 0.32)
    REVIEWER_TARGET = (0.30, 0.0, 0.19)

    def __init__(self, default_seed: int = 0, episode_length_s: float = S.EPISODE_SECONDS, reviewer_camera: bool = False):
        super().__init__(episode_length_s=episode_length_s, task_description=S.make_seed(default_seed).instruction)
        self.default_seed = default_seed
        self.reviewer_camera = reviewer_camera

    # Arena hooks -------------------------------------------------------------------------------
    def state(self, env) -> _State:
        if getattr(env, "_task_state", None) is None:
            env._task_state = _State(env, self)
        return env._task_state

    def set_next_seed(self, env, seed: int) -> None:
        self.state(env).next_layout = S.make_seed(seed)

    def force_function_test(self, env) -> None:
        """Control runs: start the after-release check now, whatever state the lockout and padlock are in."""
        self.state(env).force_test[:] = True

    def episode_report(self, env, env_id: int = 0, ended: bool = False) -> dict:
        st = self.state(env)
        if ended and env_id in st.last_reports:
            return st.last_reports.pop(env_id)
        return self._report(st, env_id)

    def _report(self, st: _State, env_id: int) -> dict:
        r = st.scorer.report(env_id)
        r["success"] = bool(r["milestone"] >= 100 and st.test_passed[env_id])
        r.update({k: v for k, v in st.layout.__dict__.items() if k != "seed"})
        names = ("lever_held_shut", "lockout_held_on", "breaker_held_off")
        r["function_test"] = {
            "run": bool(st.test_done[env_id]),
            "passed": bool(st.test_passed[env_id]),
            **{nm: bool(st.test_results[env_id, i]) for i, nm in enumerate(names)},
            "lever_deg_after_pry": round(math.degrees(float(st.test_values[env_id, 0])), 1),
            "knob_reached_and_turned": bool(st.knob_reached[env_id]),
            "lockout_pull_mm": round(float(st.test_values[env_id, 1]) * 1000, 1),
            "handle_deg_under_push": round(math.degrees(float(st.test_values[env_id, 2])), 1),
        }
        r["clamp_force_n"] = round(float(st.force[env_id]), 1)
        r["peak_clamp_force_n"] = round(float(st.peak_force[env_id]), 1)
        r["knob_turns_tightened"] = round(-float(st.turn_acc[env_id]) / (2 * math.pi), 3)
        r["lever_shut"] = bool(st.lever_shut[env_id])
        r["shackle_latched"] = bool(st.latched[env_id])
        r["shackle_latched_through"] = bool(st.latched_through[env_id])
        return r

    def get_scene_cfg(self):
        if not self.reviewer_camera:
            return None

        @configclass
        class ReviewerCameraSceneCfg:
            reviewer_cam: CameraCfg = reviewer_camera(self.REVIEWER_EYE, self.REVIEWER_TARGET)

        return ReviewerCameraSceneCfg()

    def get_events_cfg(self):
        return BreakerLockoutEventsCfg(self)

    def get_termination_cfg(self) -> TaskTerminationCfg:
        def passed(env):
            st = self.state(env)
            return (st.scorer.milestone >= 100) & st.test_passed

        passed.__name__ = "function_test_passed"
        sequence = [(self._milestone_predicate(m), 0.2) for m in MILESTONES] + [(passed, 0.2)]
        return TaskTerminationCfg(
            timeout_s=self.episode_length_s,
            success=[CompletionCriteria(name="breaker_lockout_milestones", predicate_sequence=sequence)],
        )

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def _milestone_predicate(self, milestone: int):
        def predicate(env):
            return self.state(env).scorer.milestone >= milestone

        predicate.__name__ = f"milestone_{milestone}"
        return predicate

    # Reset -------------------------------------------------------------------------------------
    @staticmethod
    def _write_pose(env, asset, pose, env_ids):
        n, dev = len(env_ids), env.device
        p = _q(pose, dev).unsqueeze(0).repeat(n, 1)
        p[:, :3] += env.scene.env_origins[env_ids]
        asset.write_root_pose_to_sim(p, env_ids=env_ids)
        asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)

    def reset(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        reset_robot_home(env, env_ids)
        for env_id in env_ids.tolist():
            if int(st.scorer.step[env_id]) > 0:
                st.last_reports[env_id] = self._report(st, env_id)
        st.layout = st.next_layout
        n, dev = len(env_ids), env.device
        st.scorer.reset(env_ids)
        for name in ("lever_shut", "latched", "latched_through", "test_done", "test_passed", "force_test",
                     "test_caught", "knob_reached"):
            getattr(st, name)[env_ids] = False
        for name in ("decoy_seat_steps", "clear_steps", "test_step", "tight_steps"):
            getattr(st, name)[env_ids] = 0
        for name in ("turn_prev", "turn_acc", "turn_hold", "force", "peak_force", "test_x0"):
            getattr(st, name)[env_ids] = 0.0
        st.lever_hold[env_ids] = math.radians(P.LEVER_OPEN_DEG)
        st.test_results[env_ids] = False
        st.test_values[env_ids] = 0.0
        # Breakers: the named one OFF, the others ON.
        self._write_pose(env, st.panel, P.panel_origin() + IDENTITY, env_ids)
        init = handle_init()
        ang = _q([[init[f"handle_{k}_joint"] for k in range(st.nb)]], dev).repeat(n, 1)
        st.handle_on[env_ids] = (ang * st.on_sign) > 0
        st.panel.write_joint_state_to_sim(ang, torch.zeros_like(ang), joint_ids=st.handle_joints, env_ids=env_ids)
        st.panel.set_joint_position_target(ang, joint_ids=st.handle_joints, env_ids=env_ids)
        st.panel.permanent_wrench_composer.reset(env_ids=env_ids)
        st.handle_ref[env_ids] = ang
        # Lockout on its stand: jaw open, lever open.
        lo = st.lo
        lo.permanent_wrench_composer.reset(env_ids=env_ids)
        js = [st.jaw_joint, st.knob_joint, st.lever_joint]
        q0 = _q([[0.0, 0.0, math.radians(P.LEVER_OPEN_DEG)]], dev).repeat(n, 1)
        lo.write_joint_state_to_sim(q0, torch.zeros_like(q0), joint_ids=js, env_ids=env_ids)
        lo.set_joint_position_target(q0, joint_ids=js, env_ids=env_ids)
        self._write_pose(env, lo, P.lockout_start_world(), env_ids)
        # Padlock open on its column, shackle popped up, unlatched.
        lock, j = st.lock, [st.shackle_joint]
        lock.permanent_wrench_composer.reset(env_ids=env_ids)
        lim = _q([[[0.0, padlock_props.OPEN_TRAVEL]]], dev).expand(n, 1, 2).contiguous()
        lock.write_joint_position_limit_to_sim(lim, joint_ids=j, env_ids=env_ids, warn_limit_violation=False)
        travel = torch.full((n, 1), padlock_props.OPEN_TRAVEL, device=dev)
        lock.write_joint_state_to_sim(travel, torch.zeros_like(travel), joint_ids=j, env_ids=env_ids)
        lock.set_joint_position_target(travel + 0.005, joint_ids=j, env_ids=env_ids)
        self._write_pose(env, lock, P.padlock_start_world(), env_ids)

    # Step --------------------------------------------------------------------------------------
    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        n, dev = st.n, st.dev
        origins = env.scene.env_origins
        named = st.layout.named_slot
        pv = st.pv
        before_test = ~st.test_done & (st.test_step == 0)

        # Handle detents (over-centre): the target flips once a handle is DETENT_FLIP_DEG past the middle.
        ang = st.panel.data.joint_pos.torch[:, st.handle_joints]
        toward_on = ang * st.on_sign
        flip = math.radians(DETENT_FLIP_DEG)
        st.handle_on = torch.where(toward_on > flip, True, torch.where(toward_on < -flip, False, st.handle_on))
        on_t = _q([P.on_angle(k) for k in range(st.nb)], dev)
        off_t = _q([P.off_angle(k) for k in range(st.nb)], dev)
        st.panel.set_joint_position_target(torch.where(st.handle_on, on_t, off_t), joint_ids=st.handle_joints)

        lo = st.lo
        jp = lo.data.joint_pos.torch
        # Lever: pin friction (it stays where it is put: a spring anchor that slips LEVER_SLIP_DEG behind it), snapped
        # shut under LEVER_SNAP_DEG.
        lev = jp[:, st.lever_joint]
        slip = math.radians(LEVER_SLIP_DEG)
        st.lever_hold = torch.minimum(torch.maximum(st.lever_hold, lev - slip), lev + slip)
        st.lever_shut = lev < math.radians(LEVER_SNAP_DEG)
        lo.set_joint_position_target(torch.where(st.lever_shut, torch.zeros_like(lev), st.lever_hold).unsqueeze(1),
                                     joint_ids=[st.lever_joint])

        # Thumbscrew: screw coupling and thread friction (N2's pattern). Tightening = clockwise from the knob end. The shut
        # lever's cam adds CAM_M of jaw travel.
        q_raw = jp[:, st.knob_joint]
        dq = torch.remainder(q_raw - st.turn_prev + math.pi, 2 * math.pi) - math.pi
        st.turn_prev = q_raw.clone()
        st.turn_acc += dq
        tight = -st.turn_acc  # clockwise seen from the knob end (front rim moving down) is negative about +y
        st.tight_turns = tight / (2 * math.pi)
        p = jp[:, st.jaw_joint]
        target = (-P.PITCH * tight / (2 * math.pi) - CAM_M * st.lever_shut.float()).clamp(
            -(P.jaw_open_y(named) - 0.001), 0.002)
        force = (JAW_K * (p - target)).clamp(min=0.0, max=JAW_MAX_N)
        st.force = force
        st.peak_force = torch.maximum(st.peak_force, force)
        limit = RUN_NM + NUT_K * SCREW_D * force
        st.turn_hold = torch.minimum(torch.maximum(st.turn_hold, st.turn_acc - limit / KNOB_K),
                                     st.turn_acc + limit / KNOB_K)
        lo.write_joint_effort_limit_to_sim_index(limits=limit.unsqueeze(1), joint_ids=[st.knob_joint])
        lo.set_joint_position_target((q_raw + st.turn_hold - st.turn_acc).unsqueeze(1), joint_ids=[st.knob_joint])
        lo.set_joint_position_target(target.unsqueeze(1), joint_ids=[st.jaw_joint])

        # Which handle is in the lockout's jaws, and is the lockout seated on the deadfront?
        lo_pos = lo.data.body_pos_w.torch[:, st.lo_body] - origins
        lo_quat = lo.data.body_quat_w.torch[:, st.lo_body]
        inv = math_utils.quat_inv(lo_quat)
        hb_pos = st.panel.data.body_pos_w.torch[:, st.handle_bodies] - origins[:, None]
        hb_quat = st.panel.data.body_quat_w.torch[:, st.handle_bodies]
        tip_local = _q([(-P.HANDLE_L + TIP_IN, 0.0, 0.0)], dev).expand(n * st.nb, 3)
        tips = hb_pos + math_utils.quat_apply(hb_quat.reshape(-1, 4), tip_local).reshape(n, st.nb, 3)
        tips_d = math_utils.quat_apply(inv[:, None].expand(-1, st.nb, 4).reshape(-1, 4),
                                       (tips - lo_pos[:, None]).reshape(-1, 3)).reshape(n, st.nb, 3)
        jaw_face = P.jaw_open_y(named) + p
        in_jaws = ((tips_d[..., 0] > P.JAW_X[0] - 0.002) & (tips_d[..., 0] < 0.0)
                   & (tips_d[..., 1] > -0.002) & (tips_d[..., 1] < jaw_face[:, None] + 0.002)
                   & (tips_d[..., 2].abs() < P.HALF_W - P.PLATE_T))
        upright = math_utils.quat_error_magnitude(lo_quat, _q(IDENTITY, dev).expand(n, 4)) < SEAT_TILT_TOL
        on_face = (lo_pos[:, 0] - S.PANEL_X).abs() < SEAT_X_TOL
        seated_k = in_jaws & (upright & on_face)[:, None]
        clamped_k = seated_k & (force >= CLAMP_N)[:, None]
        seated_named = seated_k[:, named]
        clamped_named = clamped_k[:, named]

        # Padlock: the short leg through the three holes (bottom plate, lever tab, top plate), in the lockout's frame.
        lock = st.lock
        travel = lock.data.joint_pos.torch[:, st.shackle_joint]
        lock_pos = lock.data.body_pos_w.torch[:, st.lock_body] - origins
        sh_pos = lock.data.body_pos_w.torch[:, st.lock_shackle] - origins
        sh_quat = lock.data.body_quat_w.torch[:, st.lock_shackle]

        def on_shackle(pt):
            return sh_pos + math_utils.quat_apply(sh_quat, _q([pt], dev).expand(n, 3))

        tip_w = on_shackle((pv.leg_x, 0.0, padlock_props.SHORT_LEG_BOTTOM))
        top_w = on_shackle((pv.leg_x, 0.0, pv.bend_z - pv.rod / 2))
        tip_d = math_utils.quat_apply(inv, tip_w - lo_pos)
        top_d = math_utils.quat_apply(inv, top_w - lo_pos)
        hx, hy = P.HOLE_C
        through_all = torch.ones(n, dtype=torch.bool, device=dev)
        for zc in ((P.LOOP_Z[0] + P.LOOP_Z[1]) / 2, (P.TAB_Z[0] + P.TAB_Z[1]) / 2):
            dz = top_d[:, 2] - tip_d[:, 2]
            a = ((zc - tip_d[:, 2]) / dz.where(dz.abs() > 1e-6, torch.full_like(dz, 1e-6)))
            inside = (a >= 0.0) & (a <= 1.0)
            cross = tip_d + a.clamp(0.0, 1.0).unsqueeze(1) * (top_d - tip_d)
            through_all &= inside & ((cross[:, 0] - hx).abs() < P.HOLE / 2) & ((cross[:, 1] - hy).abs() < P.HOLE / 2)
        through_geom = through_all & st.lever_shut & seated_named
        through = through_geom & clamped_named & (force >= CAMMED_N)

        newly = ~st.latched & (travel < LATCH_TRAVEL_M)
        if newly.any():
            ids = newly.nonzero().flatten()
            lim = _q([[[0.0, 0.0005]]], dev).expand(len(ids), 1, 2).contiguous()
            lock.write_joint_position_limit_to_sim(lim, joint_ids=[st.shackle_joint], env_ids=ids,
                                                   warn_limit_violation=False)
            lock.set_joint_position_target(torch.zeros(len(ids), 1, device=dev), joint_ids=[st.shackle_joint],
                                           env_ids=ids)
            st.latched |= newly
        # Shut through the holes: evaluated continuously (the press can tilt the lock for a moment as it latches).
        st.latched_through = st.latched & through_geom

        # Robot clear of the lockout and the padlock?
        grasps = grasp_points(st.robot, st.flange_ids, origins)
        tips_f = finger_points(st.robot, st.flange_ids, st.finger_ids, origins)
        clear = (((grasps - lo_pos[:, None]).norm(dim=-1) > RELEASE_CLEAR_M).all(dim=1)
                 & ((grasps - lock_pos[:, None]).norm(dim=-1) > RELEASE_CLEAR_M).all(dim=1))

        # After-release check.
        idle = ~st.test_done & (st.test_step == 0)
        ready = clamped_named & clear & idle & st.latched_through
        st.clear_steps = torch.where(ready, st.clear_steps + 1, torch.zeros_like(st.clear_steps))
        start = (st.clear_steps >= st.wait_steps) | (st.force_test & idle)
        st.test_step = torch.where(start, torch.ones_like(st.test_step), st.test_step)
        st.test_x0 = torch.where(start, lo_pos[:, 0], st.test_x0)
        running = st.test_step > 0
        p1, p2, p3, p4 = st.phase_steps
        e1, e2, e3, e4 = p1, p1 + p2, p1 + p2 + p3, p1 + p2 + p3 + p4
        ph_a = running & (st.test_step <= e1)
        ph_b = running & (st.test_step > e1) & (st.test_step <= e2) & st.knob_reached
        ph_c = running & (st.test_step > e2) & (st.test_step <= e3)
        ph_d = running & (st.test_step > e3) & (st.test_step <= e4)
        out_d = math_utils.quat_apply(lo_quat, _q([(-1.0, 0.0, 0.0)], dev).expand(n, 3))  # away from the panel
        f = torch.zeros(n, 3, 3, device=dev)  # body, lever, knob
        tq = torch.zeros(n, 3, 3, device=dev)
        # Forces ramp up over the first half second of their phase (a spring scale pulled steadily).
        ramp_a = (st.test_step.float() / (0.5 / st.step_dt)).clamp(max=1.0)
        f[:, 1] = torch.where(ph_a[:, None], LEVER_PRY_N * ramp_a[:, None] * out_d, 0.0)  # (A) the lever pulled out
        # (B) the knob turned loosening (anticlockwise seen from the knob end = positive about the lockout's +y), only
        # if the lever came open.
        y_d = math_utils.quat_apply(lo_quat, _q([(0.0, 1.0, 0.0)], dev).expand(n, 3))
        tq[:, 2] = torch.where(ph_b[:, None], KNOB_TEST_NM * y_d, 0.0)
        # (C) pulled off; once it has come TEST_CATCH_M off, the pull stops (the check has already failed).
        pulled = st.test_x0 - lo_pos[:, 0]
        catch = ph_c & (pulled > TEST_CATCH_M) & ~st.test_caught
        if catch.any():
            ids = catch.nonzero().flatten()
            st.test_values[ids, 1] = TEST_CATCH_M  # it came off: recorded as "more than TEST_CATCH_M"
            st.test_caught[ids] = True
            lo.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=dev), env_ids=ids)
        pulling = ph_c & ~st.test_caught
        ramp_c = ((st.test_step - e2).float() / (0.5 / st.step_dt)).clamp(0.0, 1.0)
        f[:, 0] = torch.where(pulling[:, None], PULL_N * ramp_c[:, None] * out_d, 0.0)
        lo.permanent_wrench_composer.set_forces_and_torques_index(
            forces=f, torques=tq, body_ids=[st.lo_body, st.lo_lever, st.lo_knob], is_global=True)
        # (D) the named handle pushed toward ON.
        htq = torch.zeros(n, 1, 3, device=dev)
        htq[:, 0, 2] = torch.where(ph_d, TEST_TORQUE_NM * st.on_sign[named], 0.0)
        st.panel.permanent_wrench_composer.set_forces_and_torques_index(
            forces=torch.zeros_like(htq), torques=htq, body_ids=[st.handle_bodies[named]], is_global=True)
        # Phase ends.
        end_a = running & (st.test_step == e1)
        st.test_values[:, 0] = torch.where(end_a, lev, st.test_values[:, 0])
        st.test_results[:, 0] = torch.where(end_a, lev < math.radians(LEVER_SHUT_DEG), st.test_results[:, 0])
        st.knob_reached |= end_a & (lev > math.radians(KNOB_REACH_DEG))
        end_c = running & (st.test_step == e3)
        pull = torch.where(st.test_caught, st.test_values[:, 1], st.test_x0 - lo_pos[:, 0])
        st.test_values[:, 1] = torch.where(end_c, pull, st.test_values[:, 1])
        st.test_results[:, 1] = torch.where(end_c, (pull.abs() < TEST_MAX_PULL_M) & ~st.test_caught,
                                            st.test_results[:, 1])
        end_d = running & (st.test_step == e4)
        tog = ang[:, named]
        st.test_values[:, 2] = torch.where(end_d, tog, st.test_values[:, 2])
        held = (tog * st.on_sign[named] < 0.0) & ~st.handle_on[:, named]
        st.test_results[:, 2] = torch.where(end_d, held, st.test_results[:, 2])
        st.test_passed |= end_d & st.test_results.all(dim=1) & (st.scorer.milestone >= 100)
        st.test_done |= end_d
        if bool(end_d[0]):
            print(f"[breaker-lockout] after-release check: {self._report(st, 0)['function_test']}", flush=True)
        st.test_step = torch.where(running & ~end_d, st.test_step + 1, torch.where(end_d, 0, st.test_step))

        # Hazard events.
        others = torch.ones(st.nb, dtype=torch.bool, device=dev)
        others[named] = False
        decoy_seated = seated_k[:, others].any(dim=1)
        st.decoy_seat_steps = torch.where(decoy_seated, st.decoy_seat_steps + 1, torch.zeros_like(st.decoy_seat_steps))
        on_decoy = clamped_k[:, others].any(dim=1) | (st.decoy_seat_steps >= st.decoy_seat_limit)
        decoy_touch = torch.zeros(n, dtype=torch.bool, device=dev)
        half = _q((P.HANDLE_L / 2 + 0.001, P.HANDLE_T / 2 + 0.004, P.HANDLE_W / 2), dev)
        for k in range(st.nb):
            if k == named:
                continue
            c = hb_pos[:, k] + math_utils.quat_apply(hb_quat[:, k], _q([(-P.HANDLE_L / 2 - 0.004, 0.0, 0.0)], dev)
                                                     .expand(n, 3))
            decoy_touch |= in_box(tips_f, c, half, TOUCH_MARGIN_M)
        decoy_moved = ((ang - st.handle_ref).abs() > DECOY_MOVED_RAD)[:, others].any(dim=1) & before_test
        named_on = st.handle_on[:, named] & before_test
        moved = (lo_pos - st.lockout_start).norm(dim=-1) > 0.005
        # 50 needs the knob itself to have brought the jaw onto the handle (before the lever's cam adds its share).
        # ... sustained for CLAMP_HOLD_S with the hands off the knob (a press on the knob alone loads the jaw).
        knob_c = lo_pos + math_utils.quat_apply(lo_quat, _q([(P.KNOB_X, (P.KNOB_Y[0] + P.KNOB_Y[1]) / 2, 0.0)], dev)
                                                .expand(n, 3))
        hands_off = ((tips_f - knob_c[:, None]).norm(dim=-1) > KNOB_CLEAR_M).all(dim=1)
        st.tight_steps = torch.where(clamped_named & ~st.lever_shut & hands_off, st.tight_steps + 1,
                                     torch.zeros_like(st.tight_steps))
        knob_tight = st.tight_steps >= st.tight_limit
        after_50 = st.scorer.milestone >= 50
        reached = {25: seated_named & moved, 50: knob_tight, 75: through & after_50,
                   100: st.latched_through & (st.scorer.milestone >= 75)}
        events = {
            "lockout_on_wrong_breaker": on_decoy & before_test,
            "wrong_breaker_moved": decoy_moved,
            "named_breaker_on": named_on,
            "wrong_breaker_touched": decoy_touch,
            "lockout_dropped": (lo_pos[:, 2] < DROP_Z_M) & before_test,
            "padlock_dropped": lock_pos[:, 2] < DROP_Z_M,
        }
        st.scorer.update(reached, events)
        st.dbg = dict(seated=seated_k, in_jaws=in_jaws, on_face=on_face, upright=upright, force=force, p=p,
                      target=target, turns=st.turn_acc / (2 * math.pi), lever=lev, through_all=through_all,
                      travel=travel, tog=ang[:, named], test_step=st.test_step, tip_d=tip_d, lo=lo_pos,
                      lockp=lock_pos, tips_d=tips_d[:, named])


@configclass
class BreakerLockoutEventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: BreakerLockoutTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))
