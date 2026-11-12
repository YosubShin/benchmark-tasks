"""Sample the handhole's air through the cover's pull slot, then take the bolts out only on a clean reading.

Mechanism (see handhole_sim/usd/handhole_props.py):
- The box is static. The cover is a free rigid body lying in the box's recess. Each of its two bolts is a free rigid
  body (washer, hex head, shank) standing in its hole: the shank passes through the cover's bolt hole into the bore
  of the box's boss. The thread is a rule: while it is engaged the task pushes the bolt down onto the cover with
  THREAD_CLAMP_N (so the washer clamps the cover, and friction under the washer resists turning). The task measures
  each bolt's turn from its orientation; after ENGAGE_TURNS of counter-clockwise turning the thread lets go and the
  rule stops for good. From then on the bolt is an ordinary loose part: nothing holds, hides or moves it. The robot
  lifts it out of its hole, carries it and sets it down; a bolt counts as clear when it lies on the table with its
  middle at least BOLT_CLEAR_M outside the box's outer edge and no hand within HAND_OFF_BOLT_M of it.
- A bolt counts as loosened once it has turned a quarter turn counter-clockwise, or been lifted 3 mm off its seat,
  or left its hole (latched for the episode).
- The monitor (meter, hose links, inline filter, hose tail) is one floating articulation. The hose's open end (the
  tail's end point) counts as inside the box only below the cover's underside within the box's opening, with the
  cover still seated, so the hose can only get there through the 13 mm pull slot (geometry). The meter's LCD shows
  room air until the end has been inside for RESPONSE_S, then the seeded box air (a rule on geometric state).
- The meter's visual alarm bars flash (ALARM_BLINK_S on, ALARM_BLINK_S off) while it shows an alarm reading. The alarm
  latches (the Max XT II's "Latching Alarms" option): it keeps showing the alarm after the hose is out.
- Function test (the success check, run off camera by the station harness). GO: once both bolts lie clear on the
  table, the hose end is out of the box and both hands are clear, continuously for TEST_WAIT_S, the scorer lifts the
  cover with a hook-like spring at its centre (up to 3 x its weight, aiming 50 mm up). Lifted 30 mm = the box is
  open. NO-GO: WATCH_S after the hose is out with the bolts untouched, the same lift must fail (the cover is still
  bolted). `force_lift_test(env)` starts the test now (control runs).
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

from handhole_sim.kit import GRASP_OFFSET_M, reset_robot_home, set_visible
from handhole_sim.scorer import MILESTONES, HandholeScorer, HandholeStep
from handhole_sim.spec import ALARM_BLINK_S, RESPONSE_S, SAMPLE_S, WATCH_S, HandholeSeed, make_seed
from handhole_sim.usd import handhole_props as props

_ASSETS = Path(__file__).resolve().parent / "assets"

THREAD_CLAMP_N = 40.0
"""While the thread is engaged, the bolt is pushed down onto the cover with this force (the thread's pull, a rule).
With the washer's friction (WASHER_MU 0.15) it gives a running torque of a few hundredths of a N*m (the real torque is
not published); it holds the 0.6 kg cover down against the 17.7 N lift check with margin."""
HOSE_K = 0.003
HOSE_D = 0.002
TAIL_D = 0.004
"""The tail hangs on free hinges (light damping stands in for the hose's own stiffness and friction)."""
RELEASED_RAD = math.pi / 2
"""A bolt turned a quarter turn counts as the cover's fastening released."""
LIFTED_M = 0.003
"""A bolt lifted this far off its seat (while still in its hole) also counts as loosened."""
OUT_CLEAR_M = 0.003
"""A bolt is out of the cover once its shank's tip is this far above the cover's top."""
UPRIGHT_COS = math.cos(math.radians(30.0))
"""Turns are counted while the bolt stands within 30 deg of upright (in its hole)."""
BOLT_CLEAR_M = 0.04
"""A bolt is set down clear when its middle is at least this far outside the box's outer edge (horizontally)..."""
BOLT_ON_TABLE_Z = 0.045
"""... and its middle no higher than this above the table (it rests on the table: lying about 13 mm, standing on its
shank's end about 33 mm; not held up)..."""
HAND_OFF_BOLT_M = 0.05
"""... and neither hand's grasp point is within this distance of it (it has been let go)."""
INSIDE_MARGIN = 0.002
"""The hose end's centreline must be this far below the cover's underside (through the slot, in the box's air)."""
COVER_SEATED_M = 0.003
TEST_CLEAR_M = 0.10
TEST_WAIT_S = 1.0
LIFT_S = 3.0
LIFT_TARGET_M = 0.05
LIFT_MAX_G = 3.0
LIFT_PASS_M = 0.03
LIFT_HELD_M = 0.005
PICKED_Z = props.Z_HOSE + 0.04
DROP_Z = 0.03


def _layout_pose(name: str) -> tuple:
    for n, pos, q in props.hose_layout()[0]:
        if n == name:
            return tuple(float(v) for v in pos) + tuple(float(v) for v in q)
    raise KeyError(name)


class HandholeBox(Object):
    def __init__(self):
        super().__init__(name="handhole", usd_path=str(_ASSETS / "box.usda"), object_type=ObjectType.BASE,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))


class Bolt(Object):
    def __init__(self, k: int):
        p = props.bolt_pose(k)
        super().__init__(name=f"bolt{k}", usd_path=str(_ASSETS / "bolt.usda"), object_type=ObjectType.RIGID,
                         initial_pose=Pose(position_xyz=p[:3], rotation_xyzw=p[3:]))


class Cover(Object):
    def __init__(self):
        p = props.cover_pose()
        super().__init__(name="cover", usd_path=str(_ASSETS / "cover.usda"), object_type=ObjectType.RIGID,
                         initial_pose=Pose(position_xyz=p[:3], rotation_xyzw=p[3:]))


class Monitor(Object):
    def __init__(self):
        super().__init__(name="monitor", usd_path=str(_ASSETS / "monitor.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        cfg = super()._generate_articulation_cfg()
        cfg.actuators = {
            "hose": ImplicitActuatorCfg(joint_names_expr=["hose_j.*", "filter_j"], stiffness=HOSE_K, damping=HOSE_D),
            "tail": ImplicitActuatorCfg(joint_names_expr=["tail_j.*"], stiffness=0.0, damping=TAIL_D),
        }
        cfg.init_state.joint_pos = {".*": 0.0}
        return cfg


class Stand(Object):
    def __init__(self):
        super().__init__(name="monitor_stand", usd_path=str(_ASSETS / "stand.usda"), object_type=ObjectType.BASE,
                         initial_pose=Pose(position_xyz=(0.0, 0.0, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))


class _TaskState:
    def __init__(self, env: ManagerBasedEnv, task: "HandholeTask"):
        n, dev = env.num_envs, env.device
        self.bolts = [env.scene[f"bolt{k}"] for k in range(2)]
        self.cover = env.scene["cover"]
        self.mon = env.scene["monitor"]
        self.robot = env.scene["robot"]
        self.tail_id = self.mon.find_bodies(props.TIP_BODY)[0][0]
        self.filter_id = self.mon.find_bodies("filter")[0][0]
        self.root_name = self.mon.body_names[0]
        self.flange_ids, _ = self.robot.find_bodies(["left_gripper", "right_gripper"], preserve_order=True)
        self.mon_paths = sim_utils.find_matching_prim_paths("/World/envs/env_.*/monitor")
        self.scorer = HandholeScorer(n, dev)
        self.next_layout = make_seed(task.default_seed)
        self.layouts: dict[int, HandholeSeed] = {}
        self.last_reports: dict[int, dict] = {}
        self.go = torch.ones(n, dtype=torch.bool, device=dev)
        zb = lambda *s: torch.zeros(n, *s, dtype=torch.bool, device=dev)  # noqa: E731
        zf = lambda *s: torch.zeros(n, *s, device=dev)  # noqa: E731
        zl = lambda: torch.zeros(n, dtype=torch.long, device=dev)  # noqa: E731
        self.turn, self.prev_yaw = zf(2), zf(2)
        self.engaged, self.loosened, self.out, self.clear = zb(2), zb(2), zb(2), zb(2)
        self.bolt_rise = zf(2)
        self.picked, self.force_test, self.lift_lifted, self.lift_held = zb(), zb(), zb(), zb()
        self.inside_steps, self.out_steps, self.clear_steps, self.phase, self.phase_steps = zl(), zl(), zl(), zl(), zl()
        self.max_inside_s, self.lift_max = zf(), zf()
        self.display = torch.full((n,), -1, dtype=torch.long, device=dev)
        self.alarm_lit = torch.zeros(n, dtype=torch.bool, device=dev)
        self.latched = torch.zeros(n, dtype=torch.bool, device=dev)
        self.alarm_steps = zl()
        self.blink_steps = max(1, round(ALARM_BLINK_S / env.step_dt))
        self.response_steps = max(1, round(RESPONSE_S / env.step_dt))
        self.sample_steps = max(1, round(SAMPLE_S / env.step_dt))
        self.watch_steps = max(1, round(WATCH_S / env.step_dt))
        self.wait_steps = max(1, round(TEST_WAIT_S / env.step_dt))
        self.lift_steps = max(1, round(LIFT_S / env.step_dt))
        self.dt = env.step_dt
        self.cover_mass = props.COVER_MASS
        self.cover_rest_z = props.cover_pose()[2]


def _write_pose(asset, pose, env_ids, origins):
    dev, n = origins.device, len(env_ids)
    p = torch.tensor(pose, device=dev, dtype=torch.float32).unsqueeze(0).repeat(n, 1)
    p[:, :3] += origins[env_ids]
    asset.write_root_pose_to_sim(p, env_ids=env_ids)
    asset.write_root_velocity_to_sim(torch.zeros(n, 6, device=dev), env_ids=env_ids)


class HandholeTask(TaskBase):
    def __init__(self, default_seed: int = 0, episode_length_s: float = 240.0):
        super().__init__(episode_length_s=episode_length_s, task_description=make_seed(default_seed).instruction)
        self.default_seed = default_seed

    def state(self, env: ManagerBasedEnv) -> _TaskState:
        if getattr(env, "_handhole_state", None) is None:
            env._handhole_state = _TaskState(env, self)
        return env._handhole_state

    def set_next_seed(self, env: ManagerBasedEnv, seed: int) -> None:
        self.state(env).next_layout = make_seed(seed)

    def force_lift_test(self, env: ManagerBasedEnv) -> None:
        self.state(env).force_test[:] = True

    def episode_report(self, env: ManagerBasedEnv, env_id: int = 0, ended: bool = False) -> dict:
        st = self.state(env)
        if ended and env_id in st.last_reports:
            return st.last_reports.pop(env_id)
        return self._report(st, env_id)

    def _report(self, st: _TaskState, env_id: int) -> dict:
        r = st.scorer.report(env_id)
        layout = st.layouts.get(env_id, st.next_layout)
        r.update(
            atmosphere=layout.atmosphere,
            display=props.DISPLAY_STATES[max(0, int(st.display[env_id]))],
            hose_end_inside_s=round(float(st.inside_steps[env_id]) * st.dt, 1),
            hose_end_inside_max_s=round(float(st.max_inside_s[env_id]), 1),
            bolt_turns=[round(float(t) / (2 * math.pi), 2) for t in st.turn[env_id]],
            bolts_engaged=[bool(v) for v in st.engaged[env_id]], bolts_loosened=[bool(v) for v in st.loosened[env_id]],
            bolts_out=[bool(v) for v in st.out[env_id]], bolts_clear=[bool(v) for v in st.clear[env_id]],
            bolt_rise_mm=[round(float(v) * 1000, 1) for v in st.bolt_rise[env_id]],
            lift_test_phase=int(st.phase[env_id]), lift_max_mm=round(float(st.lift_max[env_id]) * 1000, 1),
            lift_test_lifted=bool(st.lift_lifted[env_id]), lift_test_held=bool(st.lift_held[env_id]),
        )
        return r

    def get_scene_cfg(self):
        return None

    def get_events_cfg(self):
        return HandholeEventsCfg(self)

    def get_termination_cfg(self) -> TaskTerminationCfg:
        sequence = [(self._milestone_predicate(m), 0.25) for m in MILESTONES]
        return TaskTerminationCfg(timeout_s=self.episode_length_s,
                                  success=[CompletionCriteria(name="handhole_milestones", predicate_sequence=sequence)])

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
        st.go[env_ids] = layout.go
        for buf in (st.loosened, st.out, st.clear, st.picked, st.force_test, st.lift_lifted, st.lift_held):
            buf[env_ids] = False
        st.engaged[env_ids] = True
        for buf in (st.inside_steps, st.out_steps, st.clear_steps, st.phase, st.phase_steps):
            buf[env_ids] = 0
        for buf in (st.turn, st.prev_yaw, st.bolt_rise, st.max_inside_s, st.lift_max):
            buf[env_ids] = 0.0
        st.cover.permanent_wrench_composer.reset(env_ids=env_ids)
        # Bolts: seated in their holes, thread engaged.
        for k, bolt in enumerate(st.bolts):
            bolt.permanent_wrench_composer.reset(env_ids=env_ids)
            _write_pose(bolt, props.bolt_pose(k), env_ids, origins)
        _write_pose(st.cover, props.cover_pose(), env_ids, origins)
        # Monitor: hose at rest, filter in its fork.
        _write_pose(st.mon, _layout_pose(st.root_name), env_ids, origins)
        q = torch.zeros(n, st.mon.num_joints, device=dev)
        st.mon.write_joint_state_to_sim(q, q, env_ids=env_ids)
        st.mon.set_joint_position_target(q, env_ids=env_ids)
        for env_id in env_ids.tolist():
            st.layouts[env_id] = layout
        st.display[env_ids] = -1
        st.alarm_lit[env_ids] = True
        st.latched[env_ids] = False
        st.alarm_steps[env_ids] = 0
        self._show_display(st, torch.zeros(env.num_envs, dtype=torch.long, device=dev))
        self._show_alarm(st, torch.zeros(env.num_envs, dtype=torch.bool, device=dev))

    def _show_display(self, st: _TaskState, want: torch.Tensor) -> None:
        changed = want != st.display
        if not changed.any():
            return
        stage = sim_utils.get_current_stage()
        for env_id in changed.nonzero().flatten().tolist():
            for i, name in enumerate(props.DISPLAY_STATES):
                set_visible(stage, f"{st.mon_paths[env_id]}/meter/lcd_{name}", i == int(want[env_id]))
        st.display = want.clone()

    def _show_alarm(self, st: _TaskState, lit: torch.Tensor) -> None:
        changed = lit != st.alarm_lit
        if not changed.any():
            return
        stage = sim_utils.get_current_stage()
        for env_id in changed.nonzero().flatten().tolist():
            on = bool(lit[env_id])
            for k in range(2):
                set_visible(stage, f"{st.mon_paths[env_id]}/meter/alarm_on_{k}", on)
                set_visible(stage, f"{st.mon_paths[env_id]}/meter/alarm_off_{k}", not on)
        st.alarm_lit = lit.clone()

    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        origins = env.scene.env_origins
        n, dev = env.num_envs, env.device

        # Bolts: measure each one's turn about its axis (counter-clockwise from above loosens) while it stands upright
        # in its hole; the engaged thread pushes it down onto the cover; after ENGAGE_TURNS it lets go for good.
        flange_pos = st.robot.data.body_pos_w.torch[:, st.flange_ids]
        flange_quat = st.robot.data.body_quat_w.torch[:, st.flange_ids]
        down = torch.tensor([0.0, 0.0, -GRASP_OFFSET_M], device=dev).expand_as(flange_pos)
        grasp = flange_pos + math_utils.quat_apply(flange_quat, down) - origins[:, None]  # (N, 2 hands, 3)
        home = torch.tensor([props.bolt_pose(k)[:3] for k in range(2)], device=dev)  # (2, 3)
        b_pos = torch.stack([b.data.root_pos_w.torch for b in st.bolts], dim=1) - origins[:, None]  # (N, 2, 3)
        b_quat = torch.stack([b.data.root_quat_w.torch for b in st.bolts], dim=1)
        ex = math_utils.quat_apply(b_quat, torch.tensor([1.0, 0.0, 0.0], device=dev).expand_as(b_pos))
        ez = math_utils.quat_apply(b_quat, torch.tensor([0.0, 0.0, 1.0], device=dev).expand_as(b_pos))
        mid = b_pos + ez * props.BOLT_CENTRE_Z
        yaw = torch.atan2(ex[..., 1], ex[..., 0])
        upright = ez[..., 2] > UPRIGHT_COS
        delta = torch.remainder(yaw - st.prev_yaw + math.pi, 2 * math.pi) - math.pi
        st.turn += torch.where(upright, delta, torch.zeros_like(delta))
        st.prev_yaw = yaw
        st.bolt_rise = b_pos[..., 2] - home[None, :, 2]
        off_hole = (b_pos[..., :2] - home[None, :, :2]).norm(dim=-1)
        st.engaged &= st.turn < props.ENGAGE_TURNS * 2 * math.pi
        st.out |= (b_pos[..., 2] - props.SHANK_LEN > props.Z_TOP + OUT_CLEAR_M) | (off_hole > 0.03)
        st.loosened |= (st.turn > RELEASED_RAD) | (st.bolt_rise > LIFTED_M) | st.out
        for k, bolt in enumerate(st.bolts):
            bolt.permanent_wrench_composer.reset()
            ids = st.engaged[:, k].nonzero().flatten()
            if len(ids):
                forces = torch.zeros(len(ids), 1, 3, device=dev)
                forces[:, 0, 2] = -THREAD_CLAMP_N
                bolt.permanent_wrench_composer.add_forces_and_torques_index(
                    forces=forces, torques=torch.zeros_like(forces), body_ids=[0], env_ids=ids, is_global=True)
        # Set down clear: on the table, its middle outside the box's outline by BOLT_CLEAR_M, let go.
        ox = ((mid[..., 0] - props.BX).abs() - props.BOX_OUT[0] / 2).clamp(min=0.0)
        oy = ((mid[..., 1] - props.BY).abs() - props.BOX_OUT[1] / 2).clamp(min=0.0)
        hands_off = ((grasp[:, None] - mid[:, :, None]).norm(dim=-1) > HAND_OFF_BOLT_M).all(dim=-1)  # (N, 2 bolts)
        st.clear = st.out & ((ox * ox + oy * oy).sqrt() >= BOLT_CLEAR_M) & (mid[..., 2] < BOLT_ON_TABLE_Z) & hands_off
        released = st.loosened.any(dim=1)

        # Hose end and the box's air.
        tq = st.mon.data.body_quat_w.torch[:, st.tail_id]
        tip = (st.mon.data.body_pos_w.torch[:, st.tail_id] - origins
               + math_utils.quat_apply(tq, torch.tensor([[props.TAIL_L, 0.0, 0.0]], device=dev).expand(n, 3)))
        cover_pos = st.cover.data.root_pos_w.torch - origins
        seated = (cover_pos[:, 2] - st.cover_rest_z).abs() < COVER_SEATED_M
        hx, hy, zf, zc = props.INTERIOR
        m = INSIDE_MARGIN
        inside = (((tip[:, 0] - props.BX).abs() < hx) & ((tip[:, 1] - props.BY).abs() < hy)
                  & (tip[:, 2] < zc - m) & (tip[:, 2] > zf) & seated)
        st.inside_steps = torch.where(inside, st.inside_steps + 1, torch.zeros_like(st.inside_steps))
        st.max_inside_s = torch.maximum(st.max_inside_s, st.inside_steps.float() * env.step_dt)
        sampled = st.inside_steps >= st.sample_steps
        box_air = torch.where(st.go, 1, 2)
        shown = torch.where(st.inside_steps >= st.response_steps, box_air, torch.zeros_like(box_air))
        # Latching alarm (BW Max XT II user option, manual p.52): once in alarm, the display keeps the peak reading
        # and the alarm bars keep flashing after the hose is out, until acknowledged (not modelled: no one does).
        st.latched |= shown == 2
        shown = torch.where(st.latched, torch.full_like(shown, 2), shown)
        self._show_display(st, shown)
        alarm = shown == 2
        st.alarm_steps = torch.where(alarm, st.alarm_steps + 1, torch.zeros_like(st.alarm_steps))
        self._show_alarm(st, alarm & (((st.alarm_steps - 1) // st.blink_steps) % 2 == 0))
        over_box = (((tip[:, 0] - props.BX).abs() < props.BOX_OUT[0] / 2)
                    & ((tip[:, 1] - props.BY).abs() < props.BOX_OUT[1] / 2))
        withdrawn = (tip[:, 2] > props.Z_TOP - 0.002) | ~over_box  # out of the slot (lying on the cover counts)
        st.out_steps = torch.where(withdrawn & st.scorer.sample_ok, st.out_steps + 1, torch.zeros_like(st.out_steps))
        filter_z = st.mon.data.body_pos_w.torch[:, st.filter_id, 2] - origins[:, 2]
        st.picked |= filter_z > PICKED_Z
        dropped = st.picked & ~st.scorer.sample_ok & (filter_z < DROP_Z)

        # Lift test (hook-like spring on the cover's centre).
        slot = torch.tensor([props.SLOT_XY[0], props.SLOT_XY[1], props.Z_TOP], device=dev)
        holes = home.clone()
        holes[:, 2] = props.Z_TOP
        clear_pts = torch.cat([holes.expand(n, 2, 3), slot.expand(n, 1, 3)], dim=1)  # (N, 3, 3)
        clear = ((grasp[:, None] - clear_pts[:, :, None]).norm(dim=-1) > TEST_CLEAR_M).all(dim=-1).all(dim=-1)
        go_ready = st.go & st.clear.all(dim=1) & st.scorer.sample_ok & withdrawn & clear
        nogo_ready = ~st.go & st.scorer.sample_ok & (st.out_steps >= st.watch_steps)
        ready = (st.phase == 0) & (go_ready | nogo_ready | st.force_test)
        st.clear_steps = torch.where(ready, st.clear_steps + 1, torch.zeros_like(st.clear_steps))
        start = ready & ((st.clear_steps >= st.wait_steps) | st.force_test)
        st.phase = torch.where(start, 1, st.phase)
        st.phase_steps = torch.where(st.phase == 1, st.phase_steps + 1, st.phase_steps)
        lifting = st.phase == 1
        rise_c = cover_pos[:, 2] - st.cover_rest_z
        st.lift_max = torch.where(lifting, torch.maximum(st.lift_max, rise_c), st.lift_max)
        st.cover.permanent_wrench_composer.reset()
        if lifting.any():
            ids = lifting.nonzero().flatten()
            w = st.cover_mass * 9.81
            k = LIFT_MAX_G * w / LIFT_TARGET_M
            c = 2.0 * math.sqrt(k * st.cover_mass)
            vz = st.cover.data.root_lin_vel_w.torch[ids, 2]
            f = (w + k * (LIFT_TARGET_M - rise_c[ids]) - c * vz).clamp(0.0, LIFT_MAX_G * w)
            forces = torch.zeros(len(ids), 1, 3, device=dev)
            forces[:, 0, 2] = f
            st.cover.permanent_wrench_composer.add_forces_and_torques_index(
                forces=forces, torques=torch.zeros_like(forces), body_ids=[0], env_ids=ids, is_global=True)
        end = lifting & (st.phase_steps >= st.lift_steps)
        st.lift_lifted |= end & (st.lift_max >= LIFT_PASS_M)
        st.lift_held |= end & (st.lift_max < LIFT_HELD_M)
        st.phase = torch.where(end, 2, st.phase)

        st.scorer.update(HandholeStep(inside, sampled, released, st.lift_lifted.clone(), withdrawn,
                                  st.lift_held.clone(), st.go.clone(), dropped))


@configclass
class HandholeEventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: HandholeTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))
