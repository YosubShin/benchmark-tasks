"""Fire-extinguisher task: put a small fire out with a stored-pressure extinguisher, the PASS way.

Geometry (PhysX): the extinguisher is a floating articulation: body, lever on a sprung revolute joint, and the ring pin
on a prismatic joint along its axis. The pin passes through holes in the lever's cheeks, so the lever cannot be
squeezed while it is in (contact between the two links).
Seal (rule): the pin joint's drive is stiff (SEAL_K) until the pin has been pulled SEAL_BREAK_MM, i.e. 8 N; then the
drive is released for good and the pin slides freely to its stop.

Rules (this file, every control step):
- Valve: open while the lever is squeezed past VALVE_OPEN_DEG and spray time is left (DISCHARGE_S in total).
- Spray: a ray from the nozzle outlet along the nozzle's axis. Its first hit on the fire (the box over the pan's
  inside, from the fuel surface up to the flames' full height) is a *base hit* if it is no more than BASE_ZONE_M above
  the fuel surface and within RANGE_M of the outlet; hits higher up (the flames' bodies and tips) do nothing. The base
  is split along y into three bands, one per flame tongue; a base hit adds time to that tongue's knockdown. No
  knockdown while the unit is tilted more than UPRIGHT_MAX_DEG.
- Flames: each tongue sinks into the pan as its knockdown grows (fully gone at KNOCKDOWN_S), with a small flicker.
- Plume: a pale cone at the nozzle while the valve is open.
"""

from __future__ import annotations

import math
from dataclasses import MISSING
from pathlib import Path

import torch

import isaaclab.utils.math as math_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
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

from fire_sim import spec as S
from fire_sim.kit import Fingertips, reset_robot_home
from fire_sim.scorer import Scorer, Step
from fire_sim import geometry as P

_ASSETS = Path(__file__).resolve().parent / "assets"
MM = 0.001

LEVER_K = 0.36 / math.radians(P.LEVER_TRAVEL_DEG)
"""Return spring (N m/rad): 0.36 N m at rest (6 N at 60 mm) rising to 0.72 N m fully squeezed (12 N)."""
LEVER_PRELOAD_RAD = 0.36 / LEVER_K
PIN_OUT_M = 0.025
"""Pin counted out once slid 25 mm along its axis (its end is then past the lug and the far ear)."""
PIN_PARK_M = 0.032
"""Past 32 mm the pin is clear of the far ear; it is then held at its 60 mm stop."""
HIDDEN = (0.0, 0.0, -0.6)


def _act(expr, stiffness, damping):
    return ImplicitActuatorCfg(joint_names_expr=[expr], stiffness=stiffness, damping=damping)


def unit_pin_offset():
    return (P.PIN_AXIS[0] * MM, 0.0, P.PIN_AXIS[1] * MM)


class Extinguisher(Object):
    def __init__(self):
        x, y = S.UNIT_XY
        super().__init__(name="extinguisher", usd_path=str(_ASSETS / "extinguisher.usda"),
                         object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=(x, y, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {"lever": _act("lever_joint", LEVER_K, 0.004), "pin": _act("pin_joint", P.SEAL_K, 2.0)}
        cfg.init_state.joint_pos = {"lever_joint": 0.0, "pin_joint": 0.0}
        return cfg


class _Kinematic(Object):
    def __init__(self, name, usd, pos):
        super().__init__(name=name, usd_path=str(_ASSETS / usd), object_type=ObjectType.RIGID,
                         initial_pose=Pose(position_xyz=pos, rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))


def all_props() -> list:
    return [Extinguisher(), _Kinematic("fire_stand", "fire_stand.usda", (0.0, 0.0, 0.0)),
            *[_Kinematic(f"flame_{i}", "flame.usda", (S.FLAME_X, y, S.FUEL_Z)) for i, y in enumerate(S.FLAME_Y)],
            _Kinematic("spray", "spray.usda", HIDDEN)]


OBJECT_NAMES = ("extinguisher", "fire_stand", "flame_0", "flame_1", "flame_2", "spray")


def _pose(asset, pos, quat, env_ids, origins):
    n, dev = len(env_ids), origins.device
    p = torch.cat([pos, quat], dim=-1).to(dev, torch.float32)
    if p.dim() == 1:
        p = p.unsqueeze(0).repeat(n, 1)
    p = p.clone()
    p[:, :3] += origins[env_ids]
    asset.write_root_pose_to_sim(p, env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)


class _TaskState:
    def __init__(self, env: ManagerBasedEnv, task: "FireTask"):
        n, dev = env.num_envs, env.device
        self.dt = env.step_dt
        self.robot = env.scene["robot"]
        self.unit = env.scene["extinguisher"]
        self.flames = [env.scene[f"flame_{i}"] for i in range(3)]
        self.spray = env.scene["spray"]
        self.lever_id = self.unit.find_joints("lever_joint")[0][0]
        self.pin_id = self.unit.find_joints("pin_joint")[0][0]
        self.seal_ok = torch.ones(n, dtype=torch.bool, device=dev)
        self.pin_parked = torch.zeros(n, dtype=torch.bool, device=dev)
        self.tips = Fingertips(self.robot)
        self.scorer = Scorer(n, dev, stop_hold_s=S.STOP_HOLD_S)
        self.last_reports: dict[int, dict] = {}
        zf = lambda *s: torch.zeros(n, *s, device=dev)  # noqa: E731
        self.discharged, self.left, self.stopped = zf(), zf(), zf()
        self.knock = zf(3)
        self.pin_slide, self.tilt_deg, self.lever_deg, self.heading_deg = zf(), zf(), zf(), zf()
        self.first_inside = None
        self.base_hit = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.hit_h = zf()
        self.k = 0
        t = lambda v: torch.tensor(v, device=dev, dtype=torch.float32)  # noqa: E731
        self.nozzle = t([v * MM for v in P.NOZZLE_TIP])
        self.pin_off = t(unit_pin_offset())
        self.box_lo = t([S.PAN_IN_X[0], S.PAN_IN_Y[0], S.FUEL_Z])
        self.box_hi = t([S.PAN_IN_X[1], S.PAN_IN_Y[1], S.FUEL_Z + S.FLAME_H])
        self.band_edges = t(S.BAND_EDGES)
        # Points of the extinguisher checked against the keep-out line (U frame, m).
        pts = [P.NOZZLE_TIP, (0, 0, 0), (P.CYL_R, 0, 0), (P.CYL_R, 0, P.CYL_H), (0, 0, P.HEAD[2][1]),
               (P.HEAD[0][1], 0, P.HEAD[2][1]),
               (P.PIVOT[0] - P.LEVER_LEN, 0, P.PIVOT[1] + P.LEVER_LEN * math.sin(math.radians(P.LEVER_REST_DEG)))]
        self.unit_pts = t([[v * MM for v in p] for p in pts])


class FireTask(TaskBase):
    def __init__(self, episode_length_s: float = S.EPISODE_SECONDS):
        super().__init__(episode_length_s=episode_length_s, task_description=S.instruction())

    def state(self, env: ManagerBasedEnv) -> _TaskState:
        if getattr(env, "_task_state", None) is None:
            env._task_state = _TaskState(env, self)
        return env._task_state

    def episode_report(self, env: ManagerBasedEnv, env_id: int = 0, ended: bool = False) -> dict:
        st = self.state(env)
        if ended and env_id in st.last_reports:
            return st.last_reports.pop(env_id)
        return self._report(st, env_id)

    def _report(self, st: _TaskState, env_id: int) -> dict:
        r = st.scorer.report(env_id)
        r.update(
            pin_slide_mm=round(float(st.pin_slide[env_id]) * 1000, 1),
            seal_intact=bool(st.seal_ok[env_id]),
            lever_deg=round(float(st.lever_deg[env_id]), 1),
            tilt_deg=round(float(st.tilt_deg[env_id]), 1),
            nozzle_heading_deg=round(float(st.heading_deg[env_id]), 1),
            first_inside_line=st.first_inside,
            discharged_s=round(float(st.discharged[env_id]), 2),
            spray_left_s=round(float(st.left[env_id]), 2),
            knockdown_s=[round(float(v), 2) for v in st.knock[env_id]],
            stopped_s=round(float(st.stopped[env_id]), 2),
            last_hit_band=int(st.base_hit[env_id]), last_hit_height_mm=round(float(st.hit_h[env_id]) * 1000, 1),
        )
        return r

    def get_scene_cfg(self):
        return None

    def get_events_cfg(self):
        return EventsCfg(self)

    def get_termination_cfg(self) -> TaskTerminationCfg:
        def done(env):
            return self.state(env).scorer.success

        return TaskTerminationCfg(timeout_s=self.episode_length_s,
                                  success=[CompletionCriteria(name="fire_out", predicate_sequence=[(done, 1.0)])])

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    @staticmethod
    def _seal_drive(st, env_ids, intact: bool):
        k = torch.full((len(env_ids), 1), P.SEAL_K if intact else 0.0, device=st.unit.device)
        c = torch.full((len(env_ids), 1), 2.0 if intact else P.PIN_FREE_DAMPING, device=st.unit.device)
        st.unit.write_joint_stiffness_to_sim(k, joint_ids=[st.pin_id], env_ids=env_ids)
        st.unit.write_joint_damping_to_sim(c, joint_ids=[st.pin_id], env_ids=env_ids)

    # Events ------------------------------------------------------------------------------------
    def reset(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        reset_robot_home(env, env_ids)
        for env_id in env_ids.tolist():
            if int(st.scorer.step[env_id]) > 0:
                st.last_reports[env_id] = self._report(st, env_id)
        dev = env.device
        origins = env.scene.env_origins
        st.scorer.reset(env_ids)
        st.discharged[env_ids] = 0.0
        st.left[env_ids] = S.DISCHARGE_S
        st.stopped[env_ids] = 0.0
        st.knock[env_ids] = 0.0
        st.base_hit[env_ids] = -1
        st.first_inside = None
        q0 = torch.tensor([0.0, 0.0, 0.0, 1.0], device=dev)
        x, y = S.UNIT_XY
        unit_p = torch.tensor([x, y, 0.0], device=dev)
        _pose(st.unit, unit_p, q0, env_ids, origins)
        z = torch.zeros(len(env_ids), 1, device=dev)
        st.unit.write_joint_state_to_sim(z, z, joint_ids=[st.lever_id], env_ids=env_ids)
        st.unit.set_joint_position_target(z + LEVER_PRELOAD_RAD, joint_ids=[st.lever_id], env_ids=env_ids)
        st.unit.write_joint_state_to_sim(z, z, joint_ids=[st.pin_id], env_ids=env_ids)
        st.unit.set_joint_position_target(z, joint_ids=[st.pin_id], env_ids=env_ids)
        st.seal_ok[env_ids] = True
        st.pin_parked[env_ids] = False
        self._seal_drive(st, env_ids, intact=True)
        for i, f in enumerate(st.flames):
            _pose(f, torch.tensor([S.FLAME_X, S.FLAME_Y[i], S.FUEL_Z], device=dev), q0, env_ids, origins)
        _pose(st.spray, torch.tensor(HIDDEN, device=dev), q0, env_ids, origins)

    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        origins = env.scene.env_origins
        dev = env.device
        dt = st.dt
        st.k += 1
        n = env.num_envs

        # Lever spring: always pushing back toward rest (target beyond the rest stop: preload).
        st.unit.set_joint_position_target(torch.full((n, 1), LEVER_PRELOAD_RAD, device=dev), joint_ids=[st.lever_id])

        upos = st.unit.data.root_pos_w.torch - origins
        uquat = st.unit.data.root_quat_w.torch
        ex = math_utils.quat_apply(uquat, torch.tensor([1.0, 0.0, 0.0], device=dev).expand(n, 3))
        ez = math_utils.quat_apply(uquat, torch.tensor([0.0, 0.0, 1.0], device=dev).expand(n, 3))
        ey = math_utils.quat_apply(uquat, torch.tensor([0.0, 1.0, 0.0], device=dev).expand(n, 3))
        st.tilt_deg = torch.rad2deg(torch.acos(ez[:, 2].clamp(-1, 1)))
        st.heading_deg = torch.rad2deg(torch.atan2(ex[:, 1], ex[:, 0]))
        st.lever_deg = torch.rad2deg(-st.unit.data.joint_pos.torch[:, st.lever_id])

        # Pin and tamper seal: the seal is the pin joint's stiff drive; past SEAL_BREAK_MM of slide (8 N) it breaks
        # for good and the pin slides freely (light damping for friction in the holes).
        st.pin_slide = st.unit.data.joint_pos.torch[:, st.pin_id]
        breaking = st.seal_ok & (st.pin_slide > P.SEAL_BREAK_MM * MM)
        if breaking.any():
            ids = breaking.nonzero().flatten()
            st.seal_ok[ids] = False
            self._seal_drive(st, ids, intact=False)
        # A pin pulled clear of the far ear (in reality discarded) is held at its stop from then on, so it cannot
        # drift back into the holes as the unit is carried and swept.
        clear = ~st.seal_ok & ~st.pin_parked & (st.pin_slide > PIN_PARK_M)
        if clear.any():
            ids = clear.nonzero().flatten()
            st.pin_parked[ids] = True
            st.unit.write_joint_stiffness_to_sim(torch.full((len(ids), 1), P.SEAL_K, device=dev),
                                                 joint_ids=[st.pin_id], env_ids=ids)
            st.unit.set_joint_position_target(torch.full((len(ids), 1), P.PIN_TRAVEL * MM, device=dev),
                                              joint_ids=[st.pin_id], env_ids=ids)
        pin_out = st.pin_slide > PIN_OUT_M

        # Valve and spray.
        valve = (st.lever_deg >= S.VALVE_OPEN_DEG) & (st.left > 0)
        st.discharged = st.discharged + valve.float() * dt
        st.left = (st.left - valve.float() * dt).clamp(min=0.0)
        outlet = upos + math_utils.quat_apply(uquat, st.nozzle.expand(n, 3))
        # Ray-box entry (slab method).
        inv = 1.0 / torch.where(ex.abs() < 1e-9, torch.full_like(ex, 1e-9), ex)
        t1 = (st.box_lo - outlet) * inv
        t2 = (st.box_hi - outlet) * inv
        t_in = torch.minimum(t1, t2).max(dim=-1).values
        t_out = torch.maximum(t1, t2).min(dim=-1).values
        hits = (t_in <= t_out) & (t_in >= 0) & (t_in <= S.RANGE_M)
        entry = outlet + ex * t_in.unsqueeze(-1)
        h = entry[:, 2] - S.FUEL_Z
        base = hits & (h <= S.BASE_ZONE_M)
        band = torch.bucketize(entry[:, 1].contiguous(), st.band_edges[1:-1].contiguous())
        effective = valve & base & (st.tilt_deg <= S.UPRIGHT_MAX_DEG)
        st.base_hit = torch.where(valve & hits, torch.where(base, band, torch.full_like(band, 9)),
                                  torch.full_like(band, -1))
        st.hit_h = torch.where(valve & hits, h, torch.zeros_like(h))
        add = torch.zeros_like(st.knock)
        add.scatter_(1, band.clamp(0, 2).unsqueeze(1), (effective.float() * dt).unsqueeze(1))
        st.knock = (st.knock + add).clamp(max=S.KNOCKDOWN_S)
        out = st.knock >= S.KNOCKDOWN_S - 1e-6
        all_out = out.all(dim=1)
        knock_frac = st.knock.sum(dim=1) / (3 * S.KNOCKDOWN_S)
        st.stopped = torch.where(all_out & ~valve, st.stopped + dt, torch.zeros_like(st.stopped))

        # Visuals: flames sink with knockdown and flicker; the plume shows while the valve is open.
        ids = torch.arange(n, device=dev)
        k = st.k
        for i, f in enumerate(st.flames):
            frac = st.knock[:, i] / S.KNOCKDOWN_S
            flick = 0.004 * math.sin(1.7 * k + 2.1 * i) + 0.003 * math.sin(3.3 * k + 0.7 * i)
            z = S.FUEL_Z - frac * (S.FLAME_H - 0.015) + flick * (1 - frac)  # a 1.5 cm stub shows until it is out
            z = torch.where(out[:, i], torch.full_like(z, S.FUEL_Z - S.FLAME_H - 0.05), z)
            pos = torch.stack([torch.full_like(z, S.FLAME_X), torch.full_like(z, S.FLAME_Y[i]), z], -1)
            ax = 0.06 * math.sin(2.3 * k + i)
            ay = 0.05 * math.sin(1.9 * k + 1.3 * i)
            q = math_utils.quat_from_euler_xyz(torch.full_like(z, ax), torch.full_like(z, ay), torch.zeros_like(z))
            _pose(f, pos, q, ids, origins)
        spray_pos = torch.where(valve.unsqueeze(-1), outlet, torch.tensor(HIDDEN, device=dev).expand(n, 3))
        _pose(st.spray, spray_pos, uquat, ids, origins)

        # Hazards.
        pts = [st.tips.points(origins), st.robot.data.body_pos_w.torch - origins[:, None],
               upos[:, None] + math_utils.quat_apply(uquat[:, None].expand(n, len(st.unit_pts), 4),
                                                     st.unit_pts.expand(n, -1, 3))]
        pts = torch.cat(pts, dim=1)
        inside = (pts[..., 0] > S.KEEP_OUT_X) & (pts[..., 1] > S.KEEP_OUT_Y[0]) & (pts[..., 1] < S.KEEP_OUT_Y[1])
        keep_out = inside.any(dim=1)
        if keep_out.any() and st.first_inside is None:
            k_idx = int(inside[0].nonzero()[0, 0]) if bool(keep_out[0]) else -1
            if k_idx >= 0:
                n_tips, n_links = 4, st.robot.data.body_pos_w.torch.shape[1]
                if k_idx < n_tips:
                    label = f"finger pad {k_idx}"
                elif k_idx < n_tips + n_links:
                    label = f"arm body {st.robot.body_names[k_idx - n_tips]}"
                else:
                    label = f"extinguisher point {k_idx - n_tips - n_links}"
                st.first_inside = (label, [round(float(v), 3) for v in pts[0, k_idx]], st.k)
        vz = st.unit.data.root_lin_vel_w.torch[:, 2]
        dropped = (st.tilt_deg > S.DROP_TILT_DEG) | (vz < -S.DROP_SPEED)
        emptied = (st.left <= 0) & ~all_out

        st.scorer.update(Step(pin_out, st.discharged.clone(), knock_frac, all_out, st.stopped.clone(), keep_out,
                              dropped, emptied))


@configclass
class EventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: FireTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))
