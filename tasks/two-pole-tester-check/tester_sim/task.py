"""Two-pole tester check: a Fluke T150 checked on a Fluke PRV240 proving unit, used on the L1-N terminals of a strip
that is connected to nothing, then checked again; stop if it shows nothing on the proving unit.

Mechanism (usd/tester_props.py):
- The T150's main unit and its L1 probe are two free rigid bodies, one per arm. Each ends in a 2 mm GS38 tip, 4 mm
  long, behind which a wider nose stops it.
- Every contact is a recess the tip has to enter: the PRV240's two contacts and the UT 4 screw wells (4.5 mm
  openings, 4 mm deep). Collision walls enforce it: a tip only reaches the bottom of a recess by going in square.
- A tip is "seated" when its end is SEAT_DEPTH_M past the mouth plane and inside the opening (a rule on the geometry
  the walls allow). Electrical rule: the tester lights (its 12-230 V LEDs, and the PRV240's LED) while the two tips
  are seated in the PRV240's two contacts, one each, and the PRV240 has its batteries. The strip is connected to
  nothing and never lights it. Seed 1: the PRV240's batteries are out, so nothing ever lights.
"""

from __future__ import annotations

import math
from dataclasses import MISSING
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
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

from tester_sim.kit import q_axis, reset_robot_home, set_visible
from tester_sim.scorer import MILESTONES, TesterScorer, TesterStep
from tester_sim.spec import TARGET, TERMINALS, TesterSeed, make_seed
from tester_sim.usd import tester_props as props

_ASSETS = Path(__file__).resolve().parent / "assets"

LEVEL_QUAT = tuple(float(v) for v in q_axis((0, 1, 0), -math.pi / 2))
"""A probe held level: its axis (local +z, tip to butt) along world -x, its display (local +x) up."""
SEAT_DEPTH_M = 0.0025
"""Tip end this far past a recess mouth (of 4 mm) counts as touching the contact at its bottom."""
STRIP_MARGIN_M = 0.003
CLEAR_M = 0.08
PICKED_M = 0.01
DROP_Z_M = 0.05


def tool_start_pose(name: str) -> tuple:
    return props.tray_tool_pos(name) + LEVEL_QUAT


class TesterMain(Object):
    def __init__(self):
        p = tool_start_pose("main")
        super().__init__(name="tester_main", usd_path=str(_ASSETS / "tester_main.usda"), object_type=ObjectType.RIGID,
                         initial_pose=Pose(position_xyz=p[:3], rotation_xyzw=p[3:]))


class ProbeL1(Object):
    def __init__(self):
        p = tool_start_pose("l1")
        super().__init__(name="probe_l1", usd_path=str(_ASSETS / "probe_l1.usda"), object_type=ObjectType.RIGID,
                         initial_pose=Pose(position_xyz=p[:3], rotation_xyzw=p[3:]))


class Bench(Object):
    def __init__(self):
        super().__init__(name="bench", usd_path=str(_ASSETS / "bench.usda"), object_type=ObjectType.BASE,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))


CONTACT_NAMES = ("prv_plus", "prv_minus") + TERMINALS
"""Contacts a tip can be seated in: the PRV240's two, then each block's top-row well."""


def contact_table() -> tuple[list, list]:
    mouths = [props.prover_mouth(0), props.prover_mouth(1)] + [props.well_mouth(t) for t in TERMINALS]
    halves = [props.PRV_HOLE[:2]] * 2 + [props.WELL[:2]] * len(TERMINALS)
    return mouths, [(h[0] / 2, h[1] / 2) for h in halves]


class _TaskState:
    def __init__(self, env: ManagerBasedEnv, task: "TesterTask"):
        n, dev = env.num_envs, env.device
        self.step_dt = float(env.step_dt)
        self.main = env.scene["tester_main"]
        self.l1 = env.scene["probe_l1"]
        self.scorer = TesterScorer(n, self.step_dt, dev)
        self.next_layout = make_seed(task.default_seed)
        self.layouts: dict[int, TesterSeed] = {}
        self.last_reports: dict[int, dict] = {}
        mouths, halves = contact_table()
        self.mouths = torch.tensor(mouths, device=dev)
        self.halves = torch.tensor(halves, device=dev)
        self.flat = torch.zeros(n, dtype=torch.bool, device=dev)
        self.start = torch.zeros(n, 2, 3, device=dev)
        self.shown = torch.zeros(n, dtype=torch.bool, device=dev)
        self.env_paths = sim_utils.find_matching_prim_paths("/World/envs/env_.*")
        self.seated_now = torch.zeros(n, 2, len(CONTACT_NAMES), dtype=torch.bool, device=dev)
        ti = [CONTACT_NAMES.index(t) for t in TARGET]
        self.target_idx = ti
        self.wrong_idx = [CONTACT_NAMES.index(t) for t in TERMINALS if t not in TARGET]
        self.strip_y = (props.terminal_y(TERMINALS[-1]) - props.BLOCK_PITCH / 2,
                        props.terminal_y(TERMINALS[0]) + props.BLOCK_PITCH / 2)
        self.strip_z = (props.RAIL_Z - props.BLOCK_LEN / 2, props.RAIL_Z + props.BLOCK_LEN / 2)
        self.now: dict = {}


class TesterTask(TaskBase):
    def __init__(self, default_seed: int = 0, episode_length_s: float = 120.0):
        super().__init__(episode_length_s=episode_length_s, task_description=make_seed(default_seed).instruction)
        self.default_seed = default_seed

    def state(self, env: ManagerBasedEnv) -> _TaskState:
        if getattr(env, "_tester_state", None) is None:
            env._tester_state = _TaskState(env, self)
        return env._tester_state

    def set_next_seed(self, env: ManagerBasedEnv, seed: int) -> None:
        self.state(env).next_layout = make_seed(seed)

    def episode_report(self, env: ManagerBasedEnv, env_id: int = 0, ended: bool = False) -> dict:
        st = self.state(env)
        if ended and env_id in st.last_reports:
            return st.last_reports.pop(env_id)
        return st.scorer.report(env_id)

    def get_scene_cfg(self):
        return None

    def get_events_cfg(self):
        return TesterEventsCfg(self)

    def get_termination_cfg(self) -> TaskTerminationCfg:
        sequence = [(self._milestone_predicate(m), 0.25) for m in MILESTONES]
        return TaskTerminationCfg(timeout_s=self.episode_length_s,
                                  success=[CompletionCriteria(name="tester_milestones", predicate_sequence=sequence)])

    def get_mimic_env_cfg(self, arm_mode: ArmMode):
        raise NotImplementedError

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def _milestone_predicate(self, milestone: int):
        def predicate(env: ManagerBasedEnv) -> torch.Tensor:
            return self.state(env).scorer.milestone >= milestone

        predicate.__name__ = f"milestone_{milestone}"
        return predicate

    # Geometry --------------------------------------------------------------------------------------
    def tips(self, env: ManagerBasedEnv) -> torch.Tensor:
        """(N, 2, 3) env-frame tip ends: main unit (L2), L1 probe."""
        st = self.state(env)
        origins = env.scene.env_origins
        return torch.stack([st.main.data.root_pos_w.torch - origins, st.l1.data.root_pos_w.torch - origins], dim=1)

    def seated(self, env: ManagerBasedEnv, tips: torch.Tensor) -> torch.Tensor:
        """(N, 2, C) bool: tip k's end sits at the bottom of contact c's recess (recesses open toward -x)."""
        st = self.state(env)
        d = tips[:, :, None, :] - st.mouths[None, None]
        return ((d[..., 0] >= SEAT_DEPTH_M) & (d[..., 1].abs() <= st.halves[:, 0]) & (d[..., 2].abs() <= st.halves[:, 1]))

    # Events ----------------------------------------------------------------------------------------
    def reset(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        reset_robot_home(env, env_ids)
        for env_id in env_ids.tolist():
            if int(st.scorer.step[env_id]) > 0:
                st.last_reports[env_id] = st.scorer.report(env_id)
        layout = st.next_layout
        dev, n = env.device, len(env_ids)
        origins = env.scene.env_origins
        st.scorer.reset(env_ids)
        st.scorer.set_flat(env_ids, layout.prover_flat)
        st.flat[env_ids] = layout.prover_flat
        for k, (asset, name) in enumerate(((st.main, "main"), (st.l1, "l1"))):
            pose = torch.tensor(tool_start_pose(name), device=dev).unsqueeze(0).repeat(n, 1)
            pose[:, :3] += origins[env_ids]
            asset.write_root_pose_to_sim(pose, env_ids=env_ids)
            asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)
            st.start[env_ids, k] = torch.tensor(tool_start_pose(name)[:3], device=dev)
        for env_id in env_ids.tolist():
            st.layouts[env_id] = layout
        st.shown[env_ids] = True  # force a redraw to dark
        self._show(env, env_ids, torch.zeros(n, dtype=torch.bool, device=dev))

    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        tips = self.tips(env)
        seated = self.seated(env, tips)
        st.seated_now = seated
        a, b = seated[:, 0], seated[:, 1]
        prover_closed = (a[:, 0] & b[:, 1]) | (a[:, 1] & b[:, 0])
        t0, t1 = st.target_idx
        target_closed = (a[:, t0] & b[:, t1]) | (a[:, t1] & b[:, t0])
        lit = prover_closed & ~st.flat
        wrong = seated[:, :, st.wrong_idx].any(dim=(1, 2))
        x, y, z = tips[..., 0], tips[..., 1], tips[..., 2]
        m = STRIP_MARGIN_M
        near_strip = ((x >= props.BLOCK_FACE_X - m) & (y >= st.strip_y[0] - m) & (y <= st.strip_y[1] + m)
                      & (z >= st.strip_z[0] - m) & (z <= st.strip_z[1] + m)).any(dim=1)
        dist = (tips[:, :, None, :] - st.mouths[None, None]).norm(dim=-1)
        clear = (dist > CLEAR_M).all(dim=(1, 2))
        moved = (tips - st.start).norm(dim=-1) > PICKED_M
        dropped = (tips[..., 2] < DROP_Z_M).any(dim=1)
        st.scorer.update(TesterStep(moved.all(dim=1), prover_closed, lit, target_closed, clear, near_strip, wrong,
                                    dropped))
        st.now = dict(tips=tips, seated=seated, prover_closed=prover_closed, target_closed=target_closed, lit=lit,
                      near_strip=near_strip, clear=clear)
        self._show(env, torch.arange(env.num_envs, device=env.device), lit)

    def _show(self, env, env_ids, lit) -> None:
        st = self.state(env)
        changed = lit[env_ids] != st.shown[env_ids]
        if not changed.any():
            return
        stage = sim_utils.get_current_stage()
        for env_id in env_ids[changed].tolist():
            on = bool(lit[env_id])
            root = st.env_paths[env_id]
            for v in props.LIT_AT_240:
                set_visible(stage, f"{root}/tester_main/led_{v}_on", on)
                set_visible(stage, f"{root}/tester_main/led_{v}_off", not on)
            set_visible(stage, f"{root}/bench/prv_led_on", on)
            set_visible(stage, f"{root}/bench/prv_led_off", not on)
            st.shown[env_id] = on


@configclass
class TesterEventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: TesterTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))
