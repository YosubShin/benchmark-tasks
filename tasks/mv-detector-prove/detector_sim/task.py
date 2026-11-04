"""An HD Electric PRX detector on a short stick: self-tested, proved on its PT-PRX proof tester, touched to a target
bar that is connected to nothing, proved again; stop if it does not prove.

Mechanism (usd/detector_props.py):
- The detector and its stick are one free rigid body, held by arm A at the stick. Its SENSING ZONE end (the frame
  origin) is a 62 x 40 mm box face: it reaches the tester's end plate or the bar only by physically getting there
  (the tester, its cradle, the bar and its posts are static colliders).
- "Touching" is a rule on that geometry: the sensing end's face within TOUCH_M of the plate (or the bar's near
  surface) and over it, the detector pointing within 30 deg of its working direction.
- Buttons are 2 mm proud colliders; "pressed" is a rule: one of arm B's finger ends within PRESS_M of the button's
  top and over its footprint. ON (on the detector's face) starts the self-test: the range lights step up, then the
  indicating lights flash, then the unit stays on (every seed passes). TEST (on the PT-PRX) gives the end plate its
  output while held; the tester's LED shows it.
- Electrical rule: the detector indicates (its two red lights flash) while it is on, its sensing end touches the end
  plate and TEST is held, if the tester has output. The bar is connected to nothing and never makes it indicate.
  Seed 1: the tester's battery is flat, so its LED stays dark and the detector never indicates on it.
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

from detector_sim.kit import q_axis, qmul, reset_robot_home, set_visible
from detector_sim.scorer import MILESTONES, DetectorScorer, DetectorStep
from detector_sim.spec import STICK_YAW_DEG, DetectorSeed, make_seed
from detector_sim.usd import detector_props as props

_ASSETS = Path(__file__).resolve().parent / "assets"

LEVEL_QUAT = tuple(float(v) for v in q_axis((0, 1, 0), -math.pi / 2))
TOOL_QUAT = tuple(float(v) for v in qmul(q_axis((0, 0, 1), math.radians(STICK_YAW_DEG)), LEVEL_QUAT))
"""The detector level and face up, its sensing end pointing along the stick direction d (local +y = world n)."""
TOUCH_M = 0.003
PRESS_M = 0.0025
NEAR_M = 0.03
CLEAR_M = 0.08
PICKED_M = 0.01
DROP_Z_M = 0.05
SELFTEST_S = 3.0
"""The self-test sequence after ON: 7 range lights stepping up, back to the lowest, then the indicating lights."""
ON_PRESS_S = 0.15
ALIVE_BLINK_S = 4.0
FINGER_ENDS = ((0.024, -0.0465, -0.090), (-0.024, 0.0465, -0.090))
"""Each right-arm finger's end in its link frame (6 mm beyond the pad centre)."""
RIGHT_FINGERS = ("right_tip_left", "right_tip_right")


def tool_start_pose() -> tuple:
    return props.tip_start() + TOOL_QUAT


class Detector(Object):
    def __init__(self):
        p = tool_start_pose()
        super().__init__(name="detector", usd_path=str(_ASSETS / "detector.usda"), object_type=ObjectType.RIGID,
                         initial_pose=Pose(position_xyz=p[:3], rotation_xyzw=p[3:]))


class Bench(Object):
    def __init__(self):
        super().__init__(name="bench", usd_path=str(_ASSETS / "bench.usda"), object_type=ObjectType.BASE,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))


class _TaskState:
    def __init__(self, env: ManagerBasedEnv, task: "DetectorTask"):
        n, dev = env.num_envs, env.device
        self.det = env.scene["detector"]
        self.robot = env.scene["robot"]
        self.finger_ids, _ = self.robot.find_bodies(list(RIGHT_FINGERS), preserve_order=True)
        self.finger_off = torch.tensor(FINGER_ENDS, device=dev)
        self.step_dt = float(env.step_dt)
        k = lambda s: max(1, round(s / self.step_dt))  # noqa: E731
        self.selftest_steps, self.on_press_steps, self.blink_steps = k(SELFTEST_S), k(ON_PRESS_S), k(ALIVE_BLINK_S)
        self.flash_steps = k(0.2)
        self.scorer = DetectorScorer(n, self.step_dt, dev)
        self.next_layout = make_seed(task.default_seed)
        self.layouts: dict[int, DetectorSeed] = {}
        self.last_reports: dict[int, dict] = {}
        t = lambda v: torch.tensor(v, device=dev, dtype=torch.float32)  # noqa: E731
        self.d, self.n = t(props.D3()), t(props.N3v())
        self.plate = t(props.plate_point())
        self.test_top = t(props.test_button_top())
        self.contact = t(props.contact_point())
        a, b = props.conductor_span()
        self.cond_span = (a, b)
        self.cond_a = t(props.conductor_axis_point()) + a * self.n
        self.cond_b = t(props.conductor_axis_point()) + b * self.n
        self.on_local = t((props.BODY[0] / 2 + 0.002, props.ON_BUTTON[1], props.ON_BUTTON[0]))
        self.fails = torch.zeros(n, dtype=torch.bool, device=dev)
        self.on_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.since_on = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.start = torch.zeros(n, 3, device=dev)
        self.flags = {}
        self.env_paths = sim_utils.find_matching_prim_paths("/World/envs/env_.*")
        self.now = {}


class DetectorTask(TaskBase):
    def __init__(self, default_seed: int = 0, episode_length_s: float = 120.0):
        super().__init__(episode_length_s=episode_length_s, task_description=make_seed(default_seed).instruction)
        self.default_seed = default_seed

    def state(self, env: ManagerBasedEnv) -> _TaskState:
        if getattr(env, "_detector_state", None) is None:
            env._detector_state = _TaskState(env, self)
        return env._detector_state

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
        return DetectorEventsCfg(self)

    def get_termination_cfg(self) -> TaskTerminationCfg:
        sequence = [(self._milestone_predicate(m), 0.25) for m in MILESTONES]
        return TaskTerminationCfg(timeout_s=self.episode_length_s,
                                  success=[CompletionCriteria(name="detector_milestones", predicate_sequence=sequence)])

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
    def finger_ends(self, env) -> torch.Tensor:
        """(N, 2, 3) env-frame ends of arm B's two fingers."""
        st = self.state(env)
        pos = st.robot.data.body_pos_w.torch[:, st.finger_ids]
        quat = st.robot.data.body_quat_w.torch[:, st.finger_ids]
        off = st.finger_off.unsqueeze(0).expand(pos.shape[0], -1, -1)
        return pos + math_utils.quat_apply(quat, off) - env.scene.env_origins[:, None]

    def facts(self, env) -> dict:
        st = self.state(env)
        tip = st.det.data.root_pos_w.torch - env.scene.env_origins
        q = st.det.data.root_quat_w.torch
        n_env = tip.shape[0]
        fwd = -math_utils.quat_apply(q, torch.tensor([[0.0, 0.0, 1.0]], device=tip.device).expand(n_env, 3))
        aligned = (fwd * st.d).sum(-1) > math.cos(math.radians(30))
        rel = tip - st.plate
        along, lat, dz = (rel * st.d).sum(-1), (rel * st.n).sum(-1), rel[:, 2]
        on_plate = aligned & (along >= -TOUCH_M) & (along <= 0.02) & (lat.abs() <= props.PT[1] / 2) & (dz.abs() <= 0.02)
        rc = tip - st.contact
        a, b = st.cond_span
        c_along, c_lat, c_dz = (rc * st.d).sum(-1), (rc * st.n).sum(-1), rc[:, 2]
        on_cond = (aligned & (c_along >= -TOUCH_M) & (c_along <= 0.02) & (c_lat >= a) & (c_lat <= b)
                   & (c_dz.abs() <= 0.02 + props.COND_SIDE / 2))
        seg = st.cond_b - st.cond_a
        u = (((tip - st.cond_a) * seg).sum(-1) / (seg * seg).sum()).clamp(0, 1)
        d_cond = (tip - (st.cond_a + u[:, None] * seg)).norm(dim=-1) - props.COND_SIDE / 2
        near = d_cond <= NEAR_M
        ends = self.finger_ends(env)  # (N, 2, 3)
        # TEST button: world-aligned top face.
        rt = ends - st.test_top
        t_h = (rt * st.d).sum(-1).abs() <= props.TEST_DIAM / 2 + 0.004
        t_w = (rt * st.n).sum(-1).abs() <= props.TEST_DIAM / 2 + 0.004
        test_pressed = (t_h & t_w & (rt[..., 2] <= PRESS_M)).any(dim=1)
        # ON button, in the detector's frame (its face normal is local +x).
        loc = math_utils.quat_apply_inverse(q[:, None].expand(-1, 2, -1).reshape(-1, 4),
                                            (ends - tip[:, None]).reshape(-1, 3)).reshape(n_env, 2, 3) - st.on_local
        s = props.ON_BUTTON[2] / 2 + 0.004
        on_pressed = ((loc[..., 1].abs() <= s) & (loc[..., 2].abs() <= s) & (loc[..., 0] <= PRESS_M)
                      & (loc[..., 0] >= -0.01)).any(dim=1)
        d_plate = (tip - st.plate).norm(dim=-1)
        clear = (d_plate > CLEAR_M) & (d_cond > CLEAR_M)
        return dict(tip=tip, on_plate=on_plate, on_cond=on_cond, near=near, test_pressed=test_pressed,
                    on_pressed=on_pressed, clear=clear, along=along, c_along=c_along, d_cond=d_cond)

    # Events ----------------------------------------------------------------------------------------
    def reset(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        reset_robot_home(env, env_ids)
        for env_id in env_ids.tolist():
            if int(st.scorer.step[env_id]) > 0:
                st.last_reports[env_id] = st.scorer.report(env_id)
        layout = st.next_layout
        dev, n = env.device, len(env_ids)
        st.scorer.reset(env_ids)
        st.scorer.set_fails(env_ids, layout.prove_fails)
        st.fails[env_ids] = layout.prove_fails
        st.on_steps[env_ids] = 0
        st.since_on[env_ids] = -1
        pose = torch.tensor(tool_start_pose(), device=dev).unsqueeze(0).repeat(n, 1)
        pose[:, :3] += env.scene.env_origins[env_ids]
        st.det.write_root_pose_to_sim(pose, env_ids=env_ids)
        st.det.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)
        st.start[env_ids] = torch.tensor(tool_start_pose()[:3], device=dev)
        for env_id in env_ids.tolist():
            st.layouts[env_id] = layout
            st.flags.pop(env_id, None)

    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        f = self.facts(env)
        st.on_steps = torch.where(f["on_pressed"], st.on_steps + 1, torch.zeros_like(st.on_steps))
        st.since_on = torch.where((st.since_on < 0) & (st.on_steps >= st.on_press_steps), torch.zeros_like(st.since_on),
                                  torch.where(st.since_on >= 0, st.since_on + 1, st.since_on))
        powered = st.since_on >= 0
        selftest_done = st.since_on >= st.selftest_steps
        output = f["test_pressed"] & ~st.fails
        lit = selftest_done & f["on_plate"] & output
        moved = (f["tip"] - st.start).norm(dim=-1) > PICKED_M
        dropped = f["tip"][:, 2] < DROP_Z_M
        st.scorer.update(DetectorStep(moved, selftest_done, f["on_plate"] & f["test_pressed"], lit, f["on_cond"],
                                      f["near"], f["clear"], dropped))
        st.now = dict(f, lit=lit, powered=powered, selftest_done=selftest_done)
        self._show(env, lit, output)

    def _show(self, env, lit, tester_led) -> None:
        st = self.state(env)
        stage = sim_utils.get_current_stage()
        ramp = st.selftest_steps * 14 // 30  # the 7 range lights take the first ~1.4 s of the 3 s sequence
        for env_id in range(env.num_envs):
            k = int(st.since_on[env_id])
            step = int(st.scorer.step[env_id])
            flash = st.flash_steps
            if k < 0:
                ranges, ind = -1, False
            elif k < ramp:
                ranges, ind = min(6, k * 7 // ramp), False  # stepping up through the 7 range lights
            elif k < st.selftest_steps:
                ranges, ind = 0, (k // flash) % 2 == 0  # back to the lowest range; lights and beeper fire
            else:
                ranges = 0
                ind = bool(lit[env_id]) and (step // flash) % 2 == 0  # continuous flashing on the tester
                ind = ind or (not bool(lit[env_id]) and step % st.blink_steps == 0)  # the "alive" blink every 4 s
            flags = (ranges, ind, bool(tester_led[env_id]))
            if st.flags.get(env_id) == flags:
                continue
            st.flags[env_id] = flags
            root = st.env_paths[env_id]
            for r in range(len(props.RANGE_LIGHT_Z)):
                set_visible(stage, f"{root}/detector/range{r}_on", r == ranges)
                set_visible(stage, f"{root}/detector/range{r}_off", r != ranges)
            for s in (0, 2):
                set_visible(stage, f"{root}/detector/ind{s}_on", ind)
                set_visible(stage, f"{root}/detector/ind{s}_off", not ind)
            set_visible(stage, f"{root}/bench/pt_led_on", flags[2])
            set_visible(stage, f"{root}/bench/pt_led_off", not flags[2])


@configclass
class DetectorEventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: DetectorTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))
