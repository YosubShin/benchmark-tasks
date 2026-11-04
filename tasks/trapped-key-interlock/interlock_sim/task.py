"""Trapped-key interlock: two coded pairs (breaker + key + door); free the named breaker's key, carry it to the
named door's lock, turn it.

Mechanism (geometry except where noted; see usd/props.py):
- Each Kirk Type F is a fixed-base articulation with its station: plug (revolute 0..90 deg) and bolt (prismatic
  0..19.05 mm), coupled by a crank pin in a yoke slot (contact, self-collision on). A key (free rigid body) drives
  the plug through its shank in the plug's socket. Its bit sits under the face plate and can pass the face only
  through the notch, which lines up only at plug = 90 deg (bolt extended): the key is trapped until then.
- Coding: the A1 key has a 6 x 6 mm shank, the B1 key a 4 x 7.6 mm one; each plug's socket fits only its own code,
  so a wrong key stops on the plug's top and cannot turn it.
- Breaker: the extended bolt stands in the toggle handle's path to ON. Handle detent (rule): spring toward the
  nearer end.
- Door: the extended bolt passes through the keeper on the door. Withdrawn, the door swings free. Door locks start
  at plug 90 deg (bolt extended); the key turns -90 deg to unlock.
- Plug damping (rule) stands in for the cylinder's turning resistance (not published).

Function test (the success check): once the named key is in the named door's lock with its bolt withdrawn and both
grasp points have been clear of the key for TEST_WAIT_S: (1) pull the key outward with KEY_TUG_N for TEST_PHASE_S
(held if it rises less than KEY_HELD_MM); (2) push the named breaker's handle toward ON with HANDLE_TEST_NM (held if
it stays below HANDLE_HELD_DEG); (3) pull the named door open with DOOR_TEST_NM (opened if past DOOR_OPEN_DEG).
Passed if all three. ``force_function_test(env)`` starts the same test at once (the control run).
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

from interlock_sim.kit import GRASP_OFFSET_M, Fingertips, reset_robot_home
from interlock_sim.scorer import MILESTONES, Scorer, Step
from interlock_sim.spec import BREAKER_XY, CODES, DEFAULT_TARGET, DOOR_XY, Layout, make_layout
from interlock_sim.usd import props

_ASSETS = Path(__file__).resolve().parent / "assets"
MM = 0.001


def breaker_lock(code: str) -> tuple:
    """Desk-frame position of a breaker lock's keyway axis at its face (lock yaw 0)."""
    x, y = BREAKER_XY[code]
    return (x + props.BRK_LOCK[0] * MM, y + props.BRK_LOCK[1] * MM, props.BRK_LOCK[2] * MM)


def door_lock(code: str) -> tuple:
    x, y = DOOR_XY[code]
    return (x, y, props.DOOR_LOCK_Z * MM)


PLUG_DAMPING = 0.01
"""Plug joint damping (N*m*s/rad), standing in for the cylinder's turning resistance (not published)."""
HANDLE_DETENT_NM = 0.5
"""Torque to push the toggle over centre (inferred, MCCB class)."""
HANDLE_K = HANDLE_DETENT_NM / math.radians(props.HANDLE_ON_DEG)

IN_LOCK_XY = 0.003
IN_LOCK_Z = -0.008
"""A key counts as in a lock when its tip is within 3 mm of the axis and 8 mm below the face."""
OUT_Z = 0.002
EXTENDED_M = 0.018
WITHDRAWN_M = 0.004
"""Door unlocked: the bolt is back to within 4 mm of withdrawn (its tip then clears the keeper by 2 mm)."""
RELEASE_CLEAR_M = 0.06
WRONG_XY = 0.012
WRONG_Z = 0.010
"""A key at a lock: its tip within 12 mm of the keyway axis and less than 10 mm above the face."""
OTHER_TURNED_DEG = 10.0
HANDLE_ON_MARGIN_DEG = 5.0
DOOR_HANDLE_MARGIN = 0.015
BRK_HANDLE_MARGIN = 0.008
DROP_Z = 0.02
DOOR_MOVED_DEG = 3.0

TEST_WAIT_S = 1.0
TEST_PHASE_S = 1.5
KEY_TUG_N = 5.0
KEY_HELD_MM = 3.0
HANDLE_TEST_NM = 1.5
"""Three times the detent, toward ON."""
HANDLE_HELD_DEG = -5.0
DOOR_TEST_NM = 0.3
DOOR_OPEN_DEG = -20.0


def key_home_pose(code: str) -> tuple:
    """Key fully home in its breaker's lock, plug at 0 deg (bit under the face: trapped)."""
    x, y, z = breaker_lock(code)
    return (x, y, z + props.KEY_TIP_IN_L * MM, 0.0, 0.0, 0.0, 1.0)


def _act(expr, stiffness, damping, friction=0.0):
    return ImplicitActuatorCfg(joint_names_expr=[expr], stiffness=stiffness, damping=damping, friction=friction)


class BreakerStation(Object):
    def __init__(self, code: str):
        super().__init__(name=f"breaker_{code}", usd_path=str(_ASSETS / f"breaker_{code}.usda"),
                         object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=(*BREAKER_XY[code], 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {"plug": _act("plug_joint", 0.0, PLUG_DAMPING), "bolt": _act("bolt_joint", 0.0, 0.5),
                         "handle": _act("handle_joint", HANDLE_K, 0.05 * HANDLE_K)}
        cfg.init_state.joint_pos = {"plug_joint": 0.0, "bolt_joint": 0.0,
                                    "handle_joint": math.radians(props.HANDLE_OFF_DEG)}
        return cfg


class DoorStation(Object):
    def __init__(self, code: str):
        super().__init__(name=f"door_{code}", usd_path=str(_ASSETS / f"door_{code}.usda"),
                         object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=(*DOOR_XY[code], 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {"plug": _act("plug_joint", 0.0, PLUG_DAMPING), "bolt": _act("bolt_joint", 0.0, 0.5),
                         "door": _act("door_joint", 0.0, 0.05)}
        cfg.init_state.joint_pos = {"plug_joint": math.pi / 2, "bolt_joint": -props.THROW * MM, "door_joint": 0.0}
        return cfg


class Key(Object):
    def __init__(self, code: str):
        p = key_home_pose(code)
        super().__init__(name=f"key_{code}", usd_path=str(_ASSETS / f"key_{code}.usda"), object_type=ObjectType.RIGID,
                         initial_pose=Pose(position_xyz=p[:3], rotation_xyzw=p[3:]))


def all_props() -> list:
    return [cls(code) for code in CODES for cls in (BreakerStation, DoorStation, Key)]


OBJECT_NAMES = tuple(f"{kind}_{code}" for code in CODES for kind in ("breaker", "door", "key"))


class _Joints:
    def __init__(self, asset, names):
        self.asset = asset
        self.ids = {n: asset.find_joints(n)[0][0] for n in names}

    def q(self, name):
        return self.asset.data.joint_pos.torch[:, self.ids[name]]


def _write_pose(asset, pose, env_ids, origins):
    dev, n = origins.device, len(env_ids)
    p = torch.tensor(pose, device=dev, dtype=torch.float32).unsqueeze(0).repeat(n, 1)
    p[:, :3] += origins[env_ids]
    asset.write_root_pose_to_sim(p, env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)


class _TaskState:
    def __init__(self, env: ManagerBasedEnv, task: "TrappedKeyTask"):
        n, dev = env.num_envs, env.device
        self.layout: Layout = task.layout
        tgt, oth = self.layout.target, self.layout.other
        self.robot = env.scene["robot"]
        self.keys = {c: env.scene[f"key_{c}"] for c in CODES}
        self.brk = {c: _Joints(env.scene[f"breaker_{c}"], ("plug_joint", "bolt_joint", "handle_joint")) for c in CODES}
        self.door = {c: _Joints(env.scene[f"door_{c}"], ("plug_joint", "bolt_joint", "door_joint")) for c in CODES}
        self.tgt, self.oth = tgt, oth
        self.handle_body = self.brk[tgt].asset.find_bodies("handle")[0][0]
        self.door_body = self.door[tgt].asset.find_bodies("door")[0][0]
        self.tips = Fingertips(self.robot)
        self.flange_ids, _ = self.robot.find_bodies(["left_gripper", "right_gripper"], preserve_order=True)
        self.scorer = Scorer(n, dev)
        self.last_reports: dict[int, dict] = {}
        zb = lambda: torch.zeros(n, dtype=torch.bool, device=dev)  # noqa: E731
        zl = lambda: torch.zeros(n, dtype=torch.long, device=dev)  # noqa: E731
        zf = lambda: torch.zeros(n, device=dev)  # noqa: E731
        self.force_test, self.key_held, self.handle_held, self.door_opened, self.test_passed = zb(), zb(), zb(), zb(), zb()
        self.clear_steps, self.phase, self.phase_steps = zl(), zl(), zl()
        self.key_rise, self.handle_max, self.door_min, self.key_z0 = zf(), zf(), zf(), zf()
        self.wait_steps = max(1, round(TEST_WAIT_S / env.step_dt))
        self.phase_len = max(1, round(TEST_PHASE_S / env.step_dt))
        t = lambda v: torch.tensor(v, device=dev, dtype=torch.float32)  # noqa: E731
        self.brk_lock = {c: t(breaker_lock(c)) for c in CODES}
        self.door_lock = {c: t(door_lock(c)) for c in CODES}
        h = props.DOOR_HANDLE
        self.door_handle = {c: (t([DOOR_XY[c][0] + h[0][0] * MM, DOOR_XY[c][1] + h[1][0] * MM, h[2][0] * MM]),
                                t([DOOR_XY[c][0] + h[0][1] * MM, DOOR_XY[c][1] + h[1][1] * MM, h[2][1] * MM]))
                            for c in CODES}
        hx, hy, hh = props.HANDLE
        self.brk_handle = {c: (t([BREAKER_XY[c][0] - hx * MM, BREAKER_XY[c][1] - hy * MM, props.BRK[2] * MM]),
                               t([BREAKER_XY[c][0] + hx * MM, BREAKER_XY[c][1] + (hy + 11.0) * MM,
                                  (props.BRK[2] + hh) * MM]))
                           for c in CODES}


def _in_box(p, lo, hi, margin):
    return ((p > lo - margin) & (p < hi + margin)).all(dim=-1)


class TrappedKeyTask(TaskBase):
    def __init__(self, target: str = DEFAULT_TARGET, episode_length_s: float = 90.0):
        self.layout = make_layout(target)
        super().__init__(episode_length_s=episode_length_s, task_description=self.layout.instruction)

    def state(self, env: ManagerBasedEnv) -> _TaskState:
        if getattr(env, "_task_state", None) is None:
            env._task_state = _TaskState(env, self)
        return env._task_state

    def force_function_test(self, env: ManagerBasedEnv) -> None:
        self.state(env).force_test[:] = True

    def episode_report(self, env: ManagerBasedEnv, env_id: int = 0, ended: bool = False) -> dict:
        st = self.state(env)
        if ended and env_id in st.last_reports:
            return st.last_reports.pop(env_id)
        return self._report(st, env_id)

    def _report(self, st: _TaskState, env_id: int) -> dict:
        r = st.scorer.report(env_id)
        mm = lambda js: round(-float(js.q("bolt_joint")[env_id]) * 1000, 1)  # noqa: E731
        deg = lambda js, j: round(math.degrees(float(js.q(j)[env_id])), 1)  # noqa: E731
        r.update(
            target=st.tgt,
            breaker_bolt_mm={c: mm(st.brk[c]) for c in CODES},
            door_bolt_mm={c: mm(st.door[c]) for c in CODES},
            breaker_handle_deg={c: deg(st.brk[c], "handle_joint") for c in CODES},
            function_test_phase=int(st.phase[env_id]),
            test_key_rise_mm=round(float(st.key_rise[env_id]) * 1000, 1),
            test_handle_max_deg=round(math.degrees(float(st.handle_max[env_id])), 1),
            test_door_min_deg=round(math.degrees(float(st.door_min[env_id])), 1),
            test_key_held=bool(st.key_held[env_id]), test_handle_held=bool(st.handle_held[env_id]),
            test_door_opened=bool(st.door_opened[env_id]), function_test_passed=bool(st.test_passed[env_id]),
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
    def reset(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        reset_robot_home(env, env_ids)
        for env_id in env_ids.tolist():
            if int(st.scorer.step[env_id]) > 0:
                st.last_reports[env_id] = self._report(st, env_id)
        dev, n = env.device, len(env_ids)
        origins = env.scene.env_origins
        st.scorer.reset(env_ids)
        for buf in (st.force_test, st.key_held, st.handle_held, st.door_opened, st.test_passed):
            buf[env_ids] = False
        for buf in (st.clear_steps, st.phase, st.phase_steps):
            buf[env_ids] = 0
        for buf in (st.key_rise, st.handle_max, st.door_min, st.key_z0):
            buf[env_ids] = 0.0
        for c in CODES:
            for asset in (st.keys[c], st.brk[c].asset, st.door[c].asset):
                asset.permanent_wrench_composer.reset(env_ids=env_ids)

        def joints(js: _Joints, values: dict):
            ids = [js.ids[k] for k in values]
            q = torch.tensor([list(values.values())], device=dev).expand(n, len(ids)).contiguous()
            js.asset.write_joint_state_to_sim(q, torch.zeros_like(q), joint_ids=ids, env_ids=env_ids)
            js.asset.set_joint_position_target(q, joint_ids=ids, env_ids=env_ids)

        for c in CODES:
            joints(st.brk[c], {"plug_joint": 0.0, "bolt_joint": 0.0,
                               "handle_joint": math.radians(props.HANDLE_OFF_DEG)})
            joints(st.door[c], {"plug_joint": math.pi / 2, "bolt_joint": -props.THROW * MM, "door_joint": 0.0})
            _write_pose(st.keys[c], key_home_pose(c), env_ids, origins)

    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        origins = env.scene.env_origins
        dev = env.device
        tgt, oth = st.tgt, st.oth
        idle = st.phase == 0

        # Handle detents: spring toward the nearer end (rule).
        q_handle = {}
        for c in CODES:
            q = st.brk[c].q("handle_joint")
            q_handle[c] = q
            target = torch.where(q > 0.0, math.radians(props.HANDLE_ON_DEG), math.radians(props.HANDLE_OFF_DEG))
            st.brk[c].asset.set_joint_position_target(target.unsqueeze(1), joint_ids=[st.brk[c].ids["handle_joint"]])

        key_pos = {c: st.keys[c].data.root_pos_w.torch - origins for c in CODES}

        def rel(tip, lock):
            d = tip - lock
            return d[:, :2].norm(dim=-1), d[:, 2]

        # The named pair.
        tip = key_pos[tgt]
        b_xy, b_z = rel(tip, st.brk_lock[tgt])
        t_xy, t_z = rel(tip, st.door_lock[tgt])
        in_brk = (b_xy < IN_LOCK_XY) & (b_z < IN_LOCK_Z)
        in_door = (t_xy < IN_LOCK_XY) & (t_z < IN_LOCK_Z)
        brk_ext = -st.brk[tgt].q("bolt_joint")
        door_ext = {c: -st.door[c].q("bolt_joint") for c in CODES}
        breaker_locked = in_brk & (brk_ext > EXTENDED_M)
        key_out = (b_z > OUT_Z) | (b_xy > 0.02)
        door_unlocked = in_door & (door_ext[tgt] < WITHDRAWN_M)

        # Hazards.
        wrong = torch.zeros_like(idle)
        for k in CODES:
            for c in CODES:
                if c == k:
                    continue
                for lock in (st.brk_lock[c], st.door_lock[c]):
                    xy, z = rel(key_pos[k], lock)
                    wrong |= (xy < WRONG_XY) & (z < WRONG_Z)
        other_turned = ((st.brk[oth].q("plug_joint").abs() > math.radians(OTHER_TURNED_DEG))
                        | ((st.door[oth].q("plug_joint") - math.pi / 2).abs() > math.radians(OTHER_TURNED_DEG)))
        pads = st.tips.points(origins)  # (N, 4, 3)
        door_tried = torch.zeros_like(idle)
        for c in CODES:
            near = _in_box(pads, *st.door_handle[c], DOOR_HANDLE_MARGIN).any(dim=1)
            moved = st.door[c].q("door_joint") < -math.radians(DOOR_MOVED_DEG)
            door_tried |= (near | moved) & (door_ext[c] > 0.005)
        toward_on = torch.zeros_like(idle)
        touched = torch.zeros_like(idle)
        for c in CODES:
            toward_on |= q_handle[c] > math.radians(props.HANDLE_OFF_DEG + HANDLE_ON_MARGIN_DEG)
            touched |= _in_box(pads, *st.brk_handle[c], BRK_HANDLE_MARGIN).any(dim=1)
        dropped = tip[:, 2] < DROP_Z

        # Release: both grasp points clear of the named key's bow.
        flange_pos = st.robot.data.body_pos_w.torch[:, st.flange_ids]
        flange_quat = st.robot.data.body_quat_w.torch[:, st.flange_ids]
        down = torch.tensor([0.0, 0.0, -GRASP_OFFSET_M], device=dev).expand_as(flange_pos)
        grasp = flange_pos + math_utils.quat_apply(flange_quat, down) - origins[:, None]
        key_quat = st.keys[tgt].data.root_quat_w.torch
        bow = tip + math_utils.quat_apply(key_quat, torch.tensor([props.GRASP_K], device=dev).expand_as(tip) * MM)
        clear = ((grasp - bow[:, None]).norm(dim=-1) > RELEASE_CLEAR_M).all(dim=1)

        # Function test.
        key, brk, door = st.keys[tgt], st.brk[tgt], st.door[tgt]
        q_door = door.q("door_joint")
        ready = (door_unlocked & clear & idle) | (st.force_test & idle)
        st.clear_steps = torch.where(ready, st.clear_steps + 1, torch.zeros_like(st.clear_steps))
        start = ready & ((st.clear_steps >= st.wait_steps) | st.force_test)
        if start.any():
            ids = start.nonzero().flatten()
            st.key_z0[ids] = tip[ids, 2]
            force = torch.zeros(len(ids), 1, 3, device=dev)
            force[..., 2] = KEY_TUG_N
            key.permanent_wrench_composer.add_forces_and_torques_index(
                forces=force, torques=torch.zeros_like(force), body_ids=[0], env_ids=ids, is_global=True)
            st.phase[ids] = 1
            st.phase_steps[ids] = 0
        st.phase_steps = torch.where(st.phase > 0, st.phase_steps + 1, st.phase_steps)
        st.key_rise = torch.where(st.phase == 1, torch.maximum(st.key_rise, tip[:, 2] - st.key_z0), st.key_rise)
        st.handle_max = torch.where(st.phase == 2, torch.maximum(st.handle_max, q_handle[tgt]), st.handle_max)
        st.door_min = torch.where(st.phase == 3, torch.minimum(st.door_min, q_door), st.door_min)
        end1 = (st.phase == 1) & (st.phase_steps >= st.phase_len)
        if end1.any():
            ids = end1.nonzero().flatten()
            key.permanent_wrench_composer.reset(env_ids=ids)
            st.key_held[ids] = st.key_rise[ids] < KEY_HELD_MM * MM
            st.handle_max[ids] = q_handle[tgt][ids]
            torque = torch.zeros(len(ids), 1, 3, device=dev)
            torque[..., 0] = HANDLE_TEST_NM
            brk.asset.permanent_wrench_composer.add_forces_and_torques_index(
                forces=torch.zeros_like(torque), torques=torque, body_ids=[st.handle_body], env_ids=ids,
                is_global=True)
            st.phase[ids] = 2
            st.phase_steps[ids] = 0
        end2 = (st.phase == 2) & (st.phase_steps >= st.phase_len)
        if end2.any():
            ids = end2.nonzero().flatten()
            brk.asset.permanent_wrench_composer.reset(env_ids=ids)
            st.handle_held[ids] = st.handle_max[ids] < math.radians(HANDLE_HELD_DEG)
            torque = torch.zeros(len(ids), 1, 3, device=dev)
            torque[..., 2] = -DOOR_TEST_NM
            door.asset.permanent_wrench_composer.add_forces_and_torques_index(
                forces=torch.zeros_like(torque), torques=torque, body_ids=[st.door_body], env_ids=ids, is_global=True)
            st.phase[ids] = 3
            st.phase_steps[ids] = 0
        end3 = (st.phase == 3) & (st.phase_steps >= st.phase_len)
        if end3.any():
            ids = end3.nonzero().flatten()
            door.asset.permanent_wrench_composer.reset(env_ids=ids)
            st.door_opened[ids] = st.door_min[ids] < math.radians(DOOR_OPEN_DEG)
            st.test_passed[ids] = st.key_held[ids] & st.handle_held[ids] & st.door_opened[ids]
            st.phase[ids] = 4

        st.scorer.update(Step(breaker_locked, key_out, door_unlocked, st.test_passed.clone(), wrong & idle,
                              other_turned & idle, door_tried & idle, toward_on & idle, touched & idle, dropped))


@configclass
class EventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: TrappedKeyTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))
