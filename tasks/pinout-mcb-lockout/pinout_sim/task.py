"""Pin-out MCB lockout: a Brady pin-out lockout (POS/POW class) locks the named breaker through geometry.

What is geometry and what is a rule:
- geometry: the lockout's pocket encloses the OFF toggle (the toggle cannot reach ON with it on). Two wire pins are
  levers in the nose, coupled to the push button (a PhysX mimic joint stands in for the internal cam: it is a joint
  constraint, not task code). Pressing the button swings both pins outward; the button's return spring swings them
  back. Pushed home with the button held, the feet pass through the breaker's two holes; released, the feet swing
  in behind the lips between the holes, so the lockout cannot be pulled off. The padlock's shackle through the
  stem's hole meets the body's end face once the button moves 3.6 mm, so the button cannot be pressed far enough to
  swing the pins out: the padlock is what keeps the pins in.
- rules: the toggle's over-centre detent (drive target flips past +-DETENT_FLIP_DEG); the padlock's shackle spring
  and latch.

Milestones: 25 lockout pushed fully onto the named breaker; 50 set there (button released, both pins back behind
the lips); 75 the short leg through the stem hole; 100 the shackle latched through it. Success = milestone 100 and
the after-release check passed.

After-release check (once both grasp points have been RELEASE_CLEAR_M from the lockout and the padlock for
TEST_WAIT_S after the latch; under the station harness it runs off camera, after the recording): (A) press the
button toward the board with TEST_FORCE_N for 1 s; (B) keep pressing it and pull the lockout straight off with a net
TEST_FORCE_N for 1.5 s; (C) push the lockout's body up (toward ON) with TOGGLE_TEST_N for 1 s while the named toggle
carries TEST_TORQUE_NM toward ON. Passes if the button went in less than half its travel, the lockout moved less
than TEST_MAX_PULL_M, and the toggle stayed short of the middle of its travel. ``force_function_test`` starts the
same check at once (control runs).
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

from pinout_sim import spec as S
from pinout_sim.kit import finger_points, grasp_points, in_box, reset_robot_home, reviewer_camera, set_visible
from pinout_sim.scorer import MILESTONES, RubricScorer
from pinout_sim.usd import breaker_props as P
from pinout_sim.usd import padlock_props

_ASSETS = Path(__file__).resolve().parent / "assets"
IDENTITY = (0.0, 0.0, 0.0, 1.0)

TOGGLE_K = 0.2
"""N m/rad: the detent spring. 0.09 N m holds the toggle at the middle, so TEST_TORQUE_NM flips a free toggle."""
TOGGLE_D = 0.005
DETENT_FLIP_DEG = 8.0
BUTTON_K = 150.0
"""N/m: the return spring (TM: not published). With BUTTON_PRELOAD_M it needs 1.2 N to start the button moving and
3.2 N to hold it pressed home."""
BUTTON_PRELOAD_M = 0.008
"""The spring's free position lies this far beyond the released stop (preload)."""
BUTTON_D = 30.0
"""N s/m: stands in for the button's friction; keeps the cap from hitting the shackle at speed in the 20 N test."""
PINS_D = 0.0005
PADLOCK_SPRING_N_PER_M = 150.0
LATCH_TRAVEL_M = 0.002
PINS_IN_FRAC = 0.35
"""Pins count as set when both are within this fraction of their swing from the released stop (feet >= 1.3 mm under
the lips) and the button is out past the same fraction."""
TEST_PRESS_MAX_FRAC = 0.5
SEAT_X_TOL = (-0.002, 0.0012)
SEAT_YZ_TOL = 0.002
SEAT_ANGLE_TOL = math.radians(8.0)
DECOY_SEAT_S = 0.5
RELEASE_CLEAR_M = 0.06
TEST_WAIT_S = 1.0
TEST_FORCE_N = 20.0
TEST_TORQUE_NM = 0.15
"""10 N on the toggle's 15 mm lever."""
TOGGLE_TEST_N = 10.0
"""Upward push on the lockout's body during check (3)."""
DECOY_MOVED_RAD = 0.002 / P.TOGGLE_LEN
"""Another breaker's toggle tip moved 2 mm (7.6 degrees on its 15 mm lever)."""
TEST_PHASES_S = (1.0, 1.5, 1.0)
TEST_MAX_PULL_M = 0.003
TEST_CATCH_M = 0.005
DROP_Z_M = 0.05
TOUCH_MARGIN_M = 0.002


def _q(t, dev):
    return torch.tensor(t, dtype=torch.float32, device=dev)


def breaker_row_pose():
    return (S.BOARD_X, S.ROW_Y, S.RAIL_Z) + IDENTITY


def park(index: int, dz: float = 0.0):
    return (0.30, (index - 2) * 0.12, S.PARK_Z + dz) + IDENTITY


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


def task_assets(layout=None) -> list[Object]:
    return [
        StaticProp("board", "board.usda"),
        StaticProp("stands", "stands.usda"),
        ArticulatedProp("breakers", "breaker_row.usda", breaker_row_pose(),
                        {"toggles": dict(joint_names_expr=["toggle_.*_joint"], stiffness=TOGGLE_K, damping=TOGGLE_D)},
                        {"toggle_.*_joint": math.radians(P.TOGGLE_DEG)}),
        ArticulatedProp("lockout", "lockout.usda", P.lockout_start_world(),
                        {"button": dict(joint_names_expr=["button_joint"], stiffness=BUTTON_K, damping=BUTTON_D, armature=0.01),
                         "pins": dict(joint_names_expr=["pin_.*_joint"], stiffness=0.0, damping=PINS_D, armature=0.0005)},
                        {"button_joint": 0.0, "pin_top_joint": 0.0, "pin_bottom_joint": 0.0}),
        ArticulatedProp("padlock", "padlock_ml410.usda", P.padlock_start_world(),
                        {"shackle": dict(joint_names_expr=["shackle_joint"], stiffness=PADLOCK_SPRING_N_PER_M,
                                         damping=3.0)},
                        {"shackle_joint": padlock_props.OPEN_TRAVEL}),
    ]


class _State:
    def __init__(self, env: ManagerBasedEnv, task: "PinoutLockoutTask"):
        n, dev = env.num_envs, env.device
        self.n, self.dev = n, dev
        self.robot = env.scene["robot"]
        self.flange_ids, _ = self.robot.find_bodies(["left_gripper", "right_gripper"], preserve_order=True)
        self.finger_ids, _ = self.robot.find_joints(["left_joint7", "left_joint8", "right_joint7", "right_joint8"],
                                                    preserve_order=True)
        self.row = env.scene["breakers"]
        self.toggle_joints = [self.row.find_joints(f"toggle_{k}_joint")[0][0] for k in range(S.NUM_BREAKERS)]
        self.toggle_bodies = [self.row.find_bodies(f"toggle_{k}")[0][0] for k in range(S.NUM_BREAKERS)]
        self.lo = env.scene["lockout"]
        self.lo_body = self.lo.find_bodies("body")[0][0]
        self.lo_button = self.lo.find_bodies("button")[0][0]
        self.button_joint = self.lo.find_joints("button_joint")[0][0]
        self.pin_joints = [self.lo.find_joints(f"pin_{s}_joint")[0][0] for s in ("top", "bottom")]
        self.layout = S.make_seed(task.default_seed)
        self.next_layout = self.layout
        self.pv = PADLOCK
        self.lock = env.scene["padlock"]
        self.shackle_joint = self.lock.find_joints("shackle_joint")[0][0]
        self.lock_body = self.lock.find_bodies("body")[0][0]
        self.lock_shackle = self.lock.find_bodies("shackle")[0][0]
        self.scorer = RubricScorer("breaker_lockout", n, dev)
        self.last_reports: dict[int, dict] = {}
        self.env_paths = sim_utils.find_matching_prim_paths("/World/envs/env_.*")
        self.step_dt = env.step_dt
        self.max_steps = int(env.max_episode_length)
        bools = ("is_set", "latched", "latched_through", "test_done", "test_passed", "test_caught", "force_test")
        for name in bools:
            setattr(self, name, torch.zeros(n, dtype=torch.bool, device=dev))
        for name in ("toggle_on",):
            setattr(self, name, torch.zeros(n, S.NUM_BREAKERS, dtype=torch.bool, device=dev))
        for name in ("set_slot",):
            setattr(self, name, torch.full((n,), -1, dtype=torch.long, device=dev))
        for name in ("decoy_seat_steps", "clear_steps", "test_step"):
            setattr(self, name, torch.zeros(n, dtype=torch.long, device=dev))
        self.test_x0 = torch.zeros(n, device=dev)
        self.test_results = torch.zeros(n, 3, dtype=torch.bool, device=dev)
        self.test_values = torch.zeros(n, 3, device=dev)
        self.toggle_ref = torch.zeros(n, S.NUM_BREAKERS, device=dev)
        self.seats = _q([P.lockout_seat_world(k) for k in range(S.NUM_BREAKERS)], dev)
        self.lockout_start = _q(P.lockout_start_world()[:3], dev)
        self.phase_steps = [max(1, round(s / self.step_dt)) for s in TEST_PHASES_S]
        self.wait_steps = max(1, round(TEST_WAIT_S / self.step_dt))
        self.decoy_seat_limit = max(1, round(DECOY_SEAT_S / self.step_dt))
        self.dbg = {}


class PinoutLockoutTask(TaskBase):
    REVIEWER_EYE = (0.17, 0.11, 0.30)
    REVIEWER_TARGET = (0.29, -0.055, 0.19)

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
        names = ("button_blocked", "lockout_held_on", "breaker_held_off")
        r["function_test"] = {
            "run": bool(st.test_done[env_id]),
            "passed": bool(st.test_passed[env_id]),
            **{n: bool(st.test_results[env_id, i]) for i, n in enumerate(names)},
            "button_travel_mm_under_press": round(float(st.test_values[env_id, 0]) * 1000, 1),
            "lockout_pull_mm": (round(float(st.test_values[env_id, 1]) * 1000, 1) if not bool(st.test_caught[env_id])
                                else "pulled off (over 5 mm)"),
            "toggle_deg_under_push": round(math.degrees(float(st.test_values[env_id, 2])), 1),
        }
        r["lockout_set"] = bool(st.is_set[env_id])
        r["shackle_latched"] = bool(st.latched[env_id])
        r["shackle_latched_through_stem"] = bool(st.latched_through[env_id])
        return r

    def get_scene_cfg(self):
        if not self.reviewer_camera:
            return None

        @configclass
        class ReviewerCameraSceneCfg:
            reviewer_cam: CameraCfg = reviewer_camera(self.REVIEWER_EYE, self.REVIEWER_TARGET)

        return ReviewerCameraSceneCfg()

    def get_events_cfg(self):
        return PinoutLockoutEventsCfg(self)

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
        lay = st.layout
        n, dev = len(env_ids), env.device
        st.scorer.reset(env_ids)
        for name in ("is_set", "latched", "latched_through", "test_done", "test_passed", "force_test"):
            getattr(st, name)[env_ids] = False
        st.test_caught[env_ids] = False
        for name in ("decoy_seat_steps", "clear_steps", "test_step"):
            getattr(st, name)[env_ids] = 0
        st.set_slot[env_ids] = -1
        st.test_results[env_ids] = False
        st.test_values[env_ids] = 0.0
        # Breakers: the named one OFF, the others ON.
        self._write_pose(env, st.row, breaker_row_pose(), env_ids)
        on = torch.ones(n, S.NUM_BREAKERS, dtype=torch.bool, device=dev)
        on[:, lay.named_slot] = False
        st.toggle_on[env_ids] = on
        ang = torch.where(on, math.radians(P.TOGGLE_DEG), -math.radians(P.TOGGLE_DEG))
        st.row.write_joint_state_to_sim(ang, torch.zeros_like(ang), joint_ids=st.toggle_joints, env_ids=env_ids)
        st.row.set_joint_position_target(ang, joint_ids=st.toggle_joints, env_ids=env_ids)
        st.row.permanent_wrench_composer.reset(env_ids=env_ids)
        st.toggle_ref[env_ids] = ang
        # Lockout on its holder: button released, pins in.
        lo = st.lo
        lo.permanent_wrench_composer.reset(env_ids=env_ids)
        js = [st.button_joint] + st.pin_joints
        z = torch.zeros(n, 3, device=dev)
        lo.write_joint_state_to_sim(z, z, joint_ids=js, env_ids=env_ids)
        lo.set_joint_position_target(torch.full((n, 1), -BUTTON_PRELOAD_M, device=dev), joint_ids=[st.button_joint],
                                     env_ids=env_ids)
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
        stage = sim_utils.get_current_stage()
        for env_id in env_ids.tolist():
            root = f"{st.env_paths[env_id]}/breakers/base"
            for k in range(S.NUM_BREAKERS):
                for c in S.BREAKER_COLORS:
                    set_visible(stage, f"{root}/flag_{k}_{c}", c == lay.slot_colors[k])

    # Step --------------------------------------------------------------------------------------
    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        n, dev = st.n, st.dev
        origins = env.scene.env_origins
        named = st.layout.named_slot
        pv = st.pv

        # Toggle detents (over-centre): the target flips once a lever is DETENT_FLIP_DEG past the middle.
        ang = st.row.data.joint_pos.torch[:, st.toggle_joints]
        flip = math.radians(DETENT_FLIP_DEG)
        st.toggle_on = torch.where(ang > flip, True, torch.where(ang < -flip, False, st.toggle_on))
        target = torch.where(st.toggle_on, math.radians(P.TOGGLE_DEG), -math.radians(P.TOGGLE_DEG))
        st.row.set_joint_position_target(target, joint_ids=st.toggle_joints)

        # Lockout seated on which breaker.
        lo = st.lo
        lo_pos = lo.data.body_pos_w.torch[:, st.lo_body] - origins
        lo_quat = lo.data.body_quat_w.torch[:, st.lo_body]
        q_btn = lo.data.joint_pos.torch[:, st.button_joint]
        d = lo_pos[:, None] - st.seats[None]
        upright = math_utils.quat_error_magnitude(lo_quat, _q(IDENTITY, dev).expand(n, 4)) < SEAT_ANGLE_TOL
        seated_k = ((d[..., 0] > SEAT_X_TOL[0]) & (d[..., 0] < SEAT_X_TOL[1]) & (d[..., 1].abs() < SEAT_YZ_TOL)
                    & (d[..., 2].abs() < SEAT_YZ_TOL) & upright[:, None])
        seated_any = seated_k.any(dim=1)
        seated_slot = torch.where(seated_any, seated_k.float().argmax(dim=1), torch.full_like(st.set_slot, -1))

        # Pins (geometry + mimic joint): set = both pins back near their released stop with the button out, while the
        # lockout sits pushed fully onto a breaker (the feet can only be there behind the lips).
        q_pins = lo.data.joint_pos.torch[:, st.pin_joints]
        spread = math.radians(P.PIN_SPREAD_DEG)
        pin_frac = (q_pins * _q([-1.0, 1.0], dev)).clamp(min=0.0) / spread
        btn_frac = q_btn / P.GAP
        pins_in = (pin_frac < PINS_IN_FRAC).all(dim=1) & (btn_frac < PINS_IN_FRAC)
        st.is_set = pins_in & seated_any
        st.set_slot = torch.where(st.is_set, seated_slot, torch.full_like(st.set_slot, -1))
        set_named = st.is_set & (seated_slot == named)

        # Padlock: the short leg through the stem hole (in the button's frame: the hole's axis is y).
        lock = st.lock
        travel = lock.data.joint_pos.torch[:, st.shackle_joint]
        lock_pos = lock.data.body_pos_w.torch[:, st.lock_body] - origins
        sh_pos = lock.data.body_pos_w.torch[:, st.lock_shackle] - origins
        sh_quat = lock.data.body_quat_w.torch[:, st.lock_shackle]

        def on_shackle(p):
            return sh_pos + math_utils.quat_apply(sh_quat, _q([p], dev).expand(n, 3))

        tip = on_shackle((pv.leg_x, 0.0, padlock_props.SHORT_LEG_BOTTOM))
        top = on_shackle((pv.leg_x, 0.0, pv.bend_z - pv.rod / 2))
        b_pos = lo.data.body_pos_w.torch[:, st.lo_button] - origins
        b_quat = lo.data.body_quat_w.torch[:, st.lo_button]
        inv = math_utils.quat_inv(b_quat)
        tip_b = math_utils.quat_apply(inv, tip - b_pos)
        top_b = math_utils.quat_apply(inv, top - b_pos)
        hole_c = _q(P.hole_center(0.0), dev)
        a = (top_b[:, 1] / (top_b[:, 1] - tip_b[:, 1]).where((top_b[:, 1] - tip_b[:, 1]).abs() > 1e-6,
                                                                torch.full_like(top_b[:, 1], 1e-6))).clamp(0.0, 1.0)
        cross = top_b + a.unsqueeze(1) * (tip_b - top_b)
        in_hole = ((cross[:, 0] - hole_c[0]).abs() < P.HOLE / 2) & (cross[:, 2].abs() < P.HOLE / 2)
        ty = P.STEM_T / 2
        straddle = (torch.minimum(tip_b[:, 1], top_b[:, 1]) < -ty) & (torch.maximum(tip_b[:, 1], top_b[:, 1]) > ty)
        through = in_hole & straddle & set_named

        newly = ~st.latched & (travel < LATCH_TRAVEL_M)
        if newly.any():
            ids = newly.nonzero().flatten()
            lim = _q([[[0.0, 0.0005]]], dev).expand(len(ids), 1, 2).contiguous()
            lock.write_joint_position_limit_to_sim(lim, joint_ids=[st.shackle_joint], env_ids=ids,
                                                   warn_limit_violation=False)
            lock.set_joint_position_target(torch.zeros(len(ids), 1, device=dev), joint_ids=[st.shackle_joint],
                                           env_ids=ids)
            st.latched |= newly
            st.latched_through |= newly & through

        # Robot clear of the lockout and the padlock?
        grasps = grasp_points(st.robot, st.flange_ids, origins)
        tips = finger_points(st.robot, st.flange_ids, st.finger_ids, origins)
        clear = (((grasps - lo_pos[:, None]).norm(dim=-1) > RELEASE_CLEAR_M).all(dim=1)
                 & ((grasps - lock_pos[:, None]).norm(dim=-1) > RELEASE_CLEAR_M).all(dim=1))

        # Function test.
        idle = ~st.test_done & (st.test_step == 0)
        ready = set_named & clear & idle & st.latched_through
        st.clear_steps = torch.where(ready, st.clear_steps + 1, torch.zeros_like(st.clear_steps))
        start = (st.clear_steps >= st.wait_steps) | (st.force_test & idle)
        st.test_step = torch.where(start, torch.ones_like(st.test_step), st.test_step)
        st.test_x0 = torch.where(start, lo_pos[:, 0], st.test_x0)
        running = st.test_step > 0
        p1, p2, p3 = st.phase_steps
        ph_a = running & (st.test_step <= p1)
        ph_b = running & (st.test_step > p1) & (st.test_step <= p1 + p2)
        ph_c = running & (st.test_step > p1 + p2) & (st.test_step <= p1 + p2 + p3)
        axis = math_utils.quat_apply(lo_quat, _q([(1.0, 0.0, 0.0)], dev).expand(n, 3))  # into the panel
        f = torch.zeros(n, 2, 3, device=dev)
        # (A) and (B): the button pressed toward the board.
        f[:, 1] = torch.where((ph_a | ph_b)[:, None], TEST_FORCE_N * axis, 0.0)
        # (B) the lockout pulled off while the button is pressed: a net TEST_FORCE_N away from the board (the hand's
        # pull on the body also carries the thumb's press). Forces act for a whole control step, so a loose lockout
        # would be flung: once it has come more than TEST_CATCH_M off, the pull stops and it is put back 12 mm off
        # the breaker, at rest, to fall (display only; the test has already failed).
        pulled = st.test_x0 - lo_pos[:, 0]
        catch = ph_b & (pulled > TEST_CATCH_M)
        if catch.any():
            ids = catch.nonzero().flatten()
            pose = torch.cat([lo_pos[ids], lo_quat[ids]], dim=1)
            pose[:, 0] = st.test_x0[ids] - 0.012
            pose[:, :3] += origins[ids]
            lo.write_root_pose_to_sim(pose, env_ids=ids)
            lo.write_root_velocity_to_sim(torch.zeros(len(ids), 6, device=dev), env_ids=ids)
            st.test_values[ids, 1] = pulled[ids]
            st.test_caught[ids] = True
        pulling = ph_b & ~st.test_caught
        f[:, 0] = torch.where(pulling[:, None], -2.0 * TEST_FORCE_N * axis, 0.0)
        f[:, 1] = torch.where((ph_b & st.test_caught)[:, None], 0.0, f[:, 1])
        # (C) the lockout's body pushed up (toward ON), as someone trying to switch the breaker on through it.
        f[:, 0] = f[:, 0] + torch.where(ph_c[:, None], _q([(0.0, 0.0, TOGGLE_TEST_N)], dev).expand(n, 3), 0.0)
        lo.permanent_wrench_composer.set_forces_and_torques_index(
            forces=f, torques=torch.zeros_like(f), body_ids=[st.lo_body, st.lo_button], is_global=True)
        tq = torch.zeros(n, 1, 3, device=dev)
        tq[:, 0, 1] = torch.where(ph_c, TEST_TORQUE_NM, 0.0)
        st.row.permanent_wrench_composer.set_forces_and_torques_index(
            forces=torch.zeros_like(tq), torques=tq, body_ids=[st.toggle_bodies[named]], is_global=True)
        # Phase ends: record (A) the button's travel under the press, (B) the pull-out, (C) the toggle angle.
        end_a = running & (st.test_step == p1)
        st.test_values[:, 0] = torch.where(end_a, q_btn, st.test_values[:, 0])
        st.test_results[:, 0] = torch.where(end_a, btn_frac < TEST_PRESS_MAX_FRAC, st.test_results[:, 0])
        end_b = running & (st.test_step == p1 + p2)
        pull = torch.where(st.test_caught, st.test_values[:, 1], st.test_x0 - lo_pos[:, 0])
        st.test_values[:, 1] = torch.where(end_b, pull, st.test_values[:, 1])
        st.test_results[:, 1] = torch.where(end_b, (pull.abs() < TEST_MAX_PULL_M) & ~st.test_caught,
                                            st.test_results[:, 1])
        end_c = running & (st.test_step == p1 + p2 + p3)
        tog = ang[:, named]
        st.test_values[:, 2] = torch.where(end_c, tog, st.test_values[:, 2])
        held = (tog < 0.5 * flip) & ~st.toggle_on[:, named] & ((st.test_x0 - lo_pos[:, 0]).abs() < TEST_MAX_PULL_M)
        st.test_results[:, 2] = torch.where(end_c, held, st.test_results[:, 2])
        st.test_passed |= end_c & st.test_results.all(dim=1)
        st.test_done |= end_c
        if bool(end_c[0]):
            print(f"[pinout-lockout] after-release check: {self._report(st, 0)['function_test']}", flush=True)
        st.test_step = torch.where(running & ~end_c, st.test_step + 1, torch.where(end_c, 0, st.test_step))

        # Hazard events.
        decoy_seated = seated_any & (seated_slot != named)
        st.decoy_seat_steps = torch.where(decoy_seated, st.decoy_seat_steps + 1, torch.zeros_like(st.decoy_seat_steps))
        on_decoy = (st.is_set & (st.set_slot >= 0) & (st.set_slot != named)) | (st.decoy_seat_steps >= st.decoy_seat_limit)
        decoy_touch = torch.zeros(n, dtype=torch.bool, device=dev)
        tb = _q((0.009, P.TOGGLE_W / 2, 0.008), dev)
        face_half = _q((0.002, P.BREAKER.pitch / 2 - 0.001, P.BREAKER.front_h / 2), dev)
        for k in range(S.NUM_BREAKERS):
            if k == named:
                continue
            px, py, pz = P.toggle_pivot_world(k)
            decoy_touch |= in_box(tips, _q((px - 0.009, py, pz), dev), tb, TOUCH_MARGIN_M)
            decoy_touch |= in_box(tips, _q((P.face_x_world() - 0.002, py, pz), dev), face_half, TOUCH_MARGIN_M)
        before_test = ~st.test_done & (st.test_step == 0)
        others = torch.ones(S.NUM_BREAKERS, dtype=torch.bool, device=dev)
        others[named] = False
        decoy_moved = ((ang - st.toggle_ref).abs() > DECOY_MOVED_RAD)[:, others].any(dim=1) & before_test
        named_on = st.toggle_on[:, named] & before_test
        moved = (lo_pos - st.lockout_start).norm(dim=-1) > 0.005
        reached = {25: (seated_slot == named) & moved, 50: set_named, 75: through, 100: st.latched_through & through}
        events = {
            "lockout_on_decoy_breaker": on_decoy,
            "decoy_breaker_moved": decoy_moved,
            "named_breaker_on": named_on,
            "decoy_breaker_touched": decoy_touch,
            "lockout_dropped": (lo_pos[:, 2] < DROP_Z_M) & ~st.test_done & (st.test_step == 0),
            "padlock_dropped": lock_pos[:, 2] < DROP_Z_M,
        }
        st.scorer.update(reached, events)
        st.dbg = dict(seated=seated_slot, q_btn=q_btn, q_pins=q_pins, is_set=st.is_set,
                      set_named=set_named, through=through, in_hole=in_hole, straddle=straddle, travel=travel,
                      tog=ang[:, named], test_step=st.test_step, tip_b=tip_b, lo=lo_pos, lockp=lock_pos,
                      lockq=lock.data.body_quat_w.torch[:, st.lock_body])


@configclass
class PinoutLockoutEventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: PinoutLockoutTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))
