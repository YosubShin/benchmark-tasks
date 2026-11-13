"""Tubing cutter: Arena props and TaskBase. Cut the upright copper tube through at its mark with the screw-feed cutter.

Mechanism (tubecutter_sim/usd/cutter_props.py; numbers in tubecutter_sim/spec.py):
- The tube (with its stand) is a fixed articulation root. The cutter turns about the tube on ``orbit`` and can slide
  along it on ``slide``; its knob turns on ``feed``. These three joints are rule-driven each control step:
  * each has thread/rolling friction as a Coulomb drive: a stiff spring whose anchor slips once it carries the
    friction limit (the anchor technique of a self-locking screw);
  * ``feed``: knob turns -> wheel feed = turns x PITCH (tightening clockwise seen from the knob's end). Friction
    FEED_RUN_NM + FEED_K_NM x bite: the knob gets stiffer the further the wheel is ahead of the groove;
  * ``orbit``: friction ORBIT_RUN_NM + ORBIT_K_NM x bite (rolling the wheel and rollers round under load);
  * ``slide``: friction SLIDE_FREE_N while no groove holds the wheel (finger-snug, the cutter can be pushed along the
    tube), SLIDE_LOCK_N once the groove is TRACK_MM deep all the way round (the wheel runs in it).
- The cut is a rule: the circumference is split into BINS sectors; every sector the wheel passes over (the orbit
  sweep since the last step) is cut to the current feed (depth = max(depth, feed)). bite = feed - depth under the
  wheel. Each cutting pass also records the cutter's axial position (spiral and off-mark checks).

Episode end: success (cut through, both grippers clear of the cutter for HOLD_S) or the step limit.
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

from tubecutter_sim import spec as S
from tubecutter_sim.kit import GRASP_OFFSET_M, reset_robot_home
from tubecutter_sim.scorer import MILESTONES, CutScorer, CutStep
from tubecutter_sim.usd import cutter_props as props

_DEBUG = os.environ.get("TUBECUTTER_DEBUG") == "1"
_ASSETS = Path(__file__).resolve().parent / "assets"
SET = "cutter_set"

BINS = 72
ORBIT_K, ORBIT_D = 20.0, 0.02
ORBIT_RUN_NM, ORBIT_K_NM = 0.02, 0.10
"""Turning the cutter round: 0.02 N*m free, + 0.10 N*m per mm of bite (inferred; light, as the real one)."""
FEED_K, FEED_D = 20.0, 0.005
FEED_RUN_NM, FEED_K_NM = 0.03, 0.25
"""Knob: 0.03 N*m free running, + 0.25 N*m per mm of bite (a 1/4-turn feed, 0.45 mm: 0.14 N*m; inferred)."""
SLIDE_K, SLIDE_D = 2.0e4, 40.0
SLIDE_FREE_N, SLIDE_LOCK_N = 20.0, 300.0
"""Axial: friction of the finger-snug wheel and rollers on the tube (about 0.6 x a ~30 N clamp load, steel on copper):
the cutter (299 g) stays put under its weight and slides under a 20 N push; once tracking, the groove holds it
(inferred)."""
CUT_EPS_MM = 0.02
"""A pass counts as cutting (for the axial record) when the wheel is at least this far ahead of the groove."""

# Cutter-frame points the grippers must keep clear of at the end: hook, housing, knob centre.
_CLEAR_PTS = [(-0.028, -0.01, 0.0), (props.GRASP_U, 0.0, 0.0), (props.KNOB_CENTER_U, 0.0, 0.0)]


class CutterSet(Object):
    """Tube on its stand (fixed) + cutter (orbit, slide) + knob (feed)."""

    def __init__(self):
        super().__init__(name=SET, usd_path=str(_ASSETS / "set.usda"), object_type=ObjectType.ARTICULATION,
                         initial_pose=Pose(position_xyz=(S.TUBE_X, S.TUBE_Y, 0.0), rotation_xyzw=(0.0, 0.0, 0.0, 1.0)))

    def _generate_articulation_cfg(self) -> ArticulationCfg:
        import isaaclab.sim as sim_utils

        cfg = super()._generate_articulation_cfg()
        props_cfg = getattr(cfg.spawn, "articulation_props", None) or sim_utils.ArticulationRootPropertiesCfg()
        props_cfg.fix_root_link = True
        cfg.spawn.articulation_props = props_cfg
        cfg.actuators = {
            "orbit": ImplicitActuatorCfg(joint_names_expr=["orbit"], stiffness=ORBIT_K, damping=ORBIT_D,
                                         effort_limit_sim=ORBIT_RUN_NM),
            "feed": ImplicitActuatorCfg(joint_names_expr=["feed"], stiffness=FEED_K, damping=FEED_D,
                                        effort_limit_sim=FEED_RUN_NM),
            "slide": ImplicitActuatorCfg(joint_names_expr=["slide"], stiffness=SLIDE_K, damping=SLIDE_D,
                                         effort_limit_sim=SLIDE_FREE_N),
        }
        cfg.init_state.joint_pos = {"orbit": 0.0, "feed": 0.0, "slide": 0.0}
        return cfg


class _TaskState:
    def __init__(self, env: ManagerBasedEnv, task: "CutterTask"):
        n, dev = env.num_envs, env.device
        self.art = env.scene[SET]
        self.robot = env.scene["robot"]
        self.orbit_j = self.art.find_joints("orbit")[0][0]
        self.feed_j = self.art.find_joints("feed")[0][0]
        self.slide_j = self.art.find_joints("slide")[0][0]
        self.cutter_id = self.art.find_bodies("cutter")[0][0]
        self.flange_ids, _ = self.robot.find_bodies(["left_gripper", "right_gripper"], preserve_order=True)
        self.scorer = CutScorer(n, max(1, round(S.HOLD_S / env.step_dt)), dev)
        self.clear_pts = torch.tensor(_CLEAR_PTS, device=dev)
        self.bin_angle = torch.arange(BINS, device=dev, dtype=torch.float32) * (2 * math.pi / BINS)
        zf = lambda *s: torch.zeros(n, *s, device=dev)  # noqa: E731
        self.depth = zf(BINS)  # mm
        self.bin_z = zf(BINS)  # mm from the mark, axial position at the bin's deepest cut
        self.cut_bin = torch.zeros(n, BINS, dtype=torch.bool, device=dev)
        self.z_lo, self.z_hi = zf(), zf()
        self.any_cut = torch.zeros(n, dtype=torch.bool, device=dev)
        self.orbit_acc, self.orbit_prev, self.orbit_hold = zf(), zf(), zf()
        self.feed_acc, self.feed_prev, self.feed_hold = zf(), zf(), zf()
        self.slide_hold = zf()
        self.max_bite = zf()
        self.knob_id = None
        self.tips = None
        self.last = {}


class CutterTask(TaskBase):
    def __init__(self, default_seed: int = 0, episode_length_s: float = S.EPISODE_SECONDS):
        super().__init__(episode_length_s=episode_length_s, task_description=S.make_seed(default_seed).instruction)
        self.default_seed = default_seed

    def state(self, env: ManagerBasedEnv) -> _TaskState:
        if getattr(env, "_task_state", None) is None:
            env._task_state = _TaskState(env, self)
        return env._task_state

    def episode_report(self, env: ManagerBasedEnv, env_id: int = 0, ended: bool = False) -> dict:
        st = self.state(env)
        r = st.scorer.report(env_id)
        d = st.depth[env_id]
        r.update(
            min_depth_mm=round(float(d.min()), 3), max_depth_mm=round(float(d.max()), 3),
            feed_mm=round(float(self._feed_mm(st)[env_id]), 3),
            knob_turns=round(float(-st.feed_acc[env_id]) / (2 * math.pi), 3),
            orbit_turns=round(float(st.orbit_acc[env_id]) / (2 * math.pi), 3),
            max_bite_mm=round(float(st.max_bite[env_id]), 3),
            axial_spread_mm=round(float(st.z_hi[env_id] - st.z_lo[env_id]) if bool(st.any_cut[env_id]) else 0.0, 3),
            groove_offset_mm=round(float(self._groove_offset(st)[env_id]), 3),
            hands_clear_steps=int(st.scorer.clear_steps[env_id]),
            **{k: (bool(v[env_id]) if v.dtype == torch.bool else round(float(v[env_id]), 4)) for k, v in st.last.items()},
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
            return self.state(env).scorer.score >= milestone

        predicate.__name__ = f"milestone_{milestone}"
        return predicate

    @staticmethod
    def _feed_mm(st: _TaskState) -> torch.Tensor:
        """Wheel feed from finger-snug: knob turned clockwise seen from its end (negative about the cutter's +x)."""
        return -st.feed_acc / (2 * math.pi) * S.PITCH_MM

    @staticmethod
    def _groove_offset(st: _TaskState) -> torch.Tensor:
        cnt = st.cut_bin.sum(dim=1).clamp(min=1)
        return (st.bin_z * st.cut_bin).sum(dim=1) / cnt

    # Events ------------------------------------------------------------------------------------
    def reset(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        reset_robot_home(env, env_ids)
        st.scorer.reset(env_ids)
        for buf in (st.depth, st.bin_z, st.z_lo, st.z_hi, st.orbit_acc, st.orbit_prev, st.orbit_hold, st.feed_acc,
                    st.feed_prev, st.feed_hold, st.slide_hold, st.max_bite):
            buf[env_ids] = 0.0
        st.cut_bin[env_ids] = False
        st.any_cut[env_ids] = False
        n, dev = len(env_ids), env.device
        q = torch.zeros(n, st.art.num_joints, device=dev)
        st.art.write_joint_state_to_sim(q, torch.zeros_like(q), env_ids=env_ids)
        st.art.set_joint_position_target(q.clone(), env_ids=env_ids)

    @staticmethod
    def _unwrap(raw, prev, acc):
        raw = raw.clone()  # a view into the joint buffer would be updated in place, making every delta 0
        d = torch.remainder(raw - prev + math.pi, 2 * math.pi) - math.pi
        return raw, acc + d

    @staticmethod
    def _coulomb(acc, hold, limit, k):
        """Slip the anchor so the spring never carries more than ``limit``; returns the new anchor."""
        return torch.minimum(torch.maximum(hold, acc - limit / k), acc + limit / k)

    def step(self, env: ManagerBasedEnv, env_ids: torch.Tensor) -> None:
        st = self.state(env)
        origins = env.scene.env_origins
        n, dev = env.num_envs, env.device
        jp = st.art.data.joint_pos.torch

        orbit_before = st.orbit_acc.clone()
        st.orbit_prev, st.orbit_acc = self._unwrap(jp[:, st.orbit_j], st.orbit_prev, st.orbit_acc)
        st.feed_prev, st.feed_acc = self._unwrap(jp[:, st.feed_j], st.feed_prev, st.feed_acc)
        slide = jp[:, st.slide_j].clone()
        # Where the cutter sits on the tube is the slide's friction anchor: the joint itself deflects elastically about
        # it (stiff spring, up to SLIDE_FREE_N / SLIDE_K = 1 mm under a hard push) without the cutter slipping.
        z_mm = st.slide_hold * 1000.0
        feed = self._feed_mm(st)

        # The cut: every sector swept since the last step is cut to the current feed.
        lo = torch.minimum(orbit_before, st.orbit_acc)
        span = (st.orbit_acc - orbit_before).abs()
        wheel_now = torch.remainder(st.orbit_acc, 2 * math.pi)
        rel = torch.remainder(st.bin_angle[None] - torch.remainder(lo, 2 * math.pi)[:, None], 2 * math.pi)
        width = 2 * math.pi / BINS
        swept = (rel <= span[:, None] + 0.5 * width) | (span[:, None] >= 2 * math.pi)
        under = torch.remainder(st.bin_angle[None] - wheel_now[:, None] + 0.5 * width, 2 * math.pi) < width
        touched = swept | under
        bite_bins = torch.where(touched, feed[:, None] - st.depth, torch.full_like(st.depth, -1e9))
        bite = bite_bins.max(dim=1).values.clamp(min=0.0)
        st.max_bite = torch.maximum(st.max_bite, bite)
        cutting = swept & (feed[:, None] - st.depth > CUT_EPS_MM)
        st.bin_z = torch.where(cutting, z_mm[:, None], st.bin_z)
        st.cut_bin |= cutting
        st.depth = torch.where(swept, torch.maximum(st.depth, feed[:, None]), st.depth)
        any_now = cutting.any(dim=1)
        first = any_now & ~st.any_cut
        st.z_lo = torch.where(first, z_mm, torch.where(any_now, torch.minimum(st.z_lo, z_mm), st.z_lo))
        st.z_hi = torch.where(first, z_mm, torch.where(any_now, torch.maximum(st.z_hi, z_mm), st.z_hi))
        st.any_cut |= any_now
        min_depth = st.depth.min(dim=1).values
        tracking = min_depth >= S.TRACK_MM

        # Rule-driven friction (Coulomb anchors) on the three joints.
        orbit_lim = ORBIT_RUN_NM + ORBIT_K_NM * bite
        feed_lim = FEED_RUN_NM + FEED_K_NM * bite
        slide_lim = torch.where(tracking, torch.full_like(bite, SLIDE_LOCK_N), torch.full_like(bite, SLIDE_FREE_N))
        st.orbit_hold = self._coulomb(st.orbit_acc, st.orbit_hold, orbit_lim, ORBIT_K)
        st.feed_hold = self._coulomb(st.feed_acc, st.feed_hold, feed_lim, FEED_K)
        st.slide_hold = self._coulomb(slide, st.slide_hold, slide_lim, SLIDE_K)
        lims = torch.stack([orbit_lim, feed_lim, slide_lim], dim=1)
        ids = [st.orbit_j, st.feed_j, st.slide_j]
        st.art.write_joint_effort_limit_to_sim_index(limits=lims, joint_ids=ids)
        tgt = torch.stack([jp[:, st.orbit_j] + st.orbit_hold - st.orbit_acc,
                           jp[:, st.feed_j] + st.feed_hold - st.feed_acc, st.slide_hold], dim=1)
        st.art.set_joint_position_target(tgt, joint_ids=ids)

        # Hands clear of the cutter (grasp points vs hook, housing and knob).
        c_pos = st.art.data.body_pos_w.torch[:, st.cutter_id] - origins
        c_quat = st.art.data.body_quat_w.torch[:, st.cutter_id]
        k = st.clear_pts.shape[0]
        pts = c_pos[:, None] + math_utils.quat_apply(c_quat[:, None].expand(-1, k, 4), st.clear_pts[None].expand(n, -1, 3))
        flange_pos = st.robot.data.body_pos_w.torch[:, st.flange_ids]
        flange_quat = st.robot.data.body_quat_w.torch[:, st.flange_ids]
        down = torch.tensor([0.0, 0.0, -GRASP_OFFSET_M], device=dev).expand_as(flange_pos)
        grasp = flange_pos + math_utils.quat_apply(flange_quat, down) - origins[:, None]
        clear = ((grasp[:, :, None] - pts[:, None]).norm(dim=-1) > S.CLEAR_M).all(dim=-1).all(dim=-1)

        offset = self._groove_offset(st)
        spiral = st.any_cut & (st.z_hi - st.z_lo > S.SPIRAL_MM)
        off_mark = st.any_cut & tracking & (offset.abs() > S.OFF_MARK_MM)
        dented = bite > S.DENT_BITE_MM
        st.last = {"bite_mm": bite, "tracking": tracking, "hands_clear": clear, "axial_mm": z_mm}
        st.scorer.update(CutStep(min_depth, dented, spiral, off_mark, clear))
        if _DEBUG and int(st.scorer.step[0]) % 40 == 0:
            print(f"[cutter] step {int(st.scorer.step[0])} orbit {math.degrees(float(st.orbit_acc[0])):.1f} deg "
                  f"knob {float(-st.feed_acc[0]) / (2 * math.pi):.3f} turns feed {float(feed[0]):.3f} mm "
                  f"depth min/max {float(min_depth[0]):.3f}/{float(st.depth[0].max()):.3f} bite {float(bite[0]):.3f} "
                  f"axial {float(z_mm[0]):.2f} mm (joint {float(slide[0]) * 1000:.2f}) spread "
                  f"{float(st.z_hi[0] - st.z_lo[0]):.2f} offset {float(offset[0]):.2f} caps "
                  f"{bool(st.scorer.spiral[0])}/{bool(st.scorer.off_mark[0])} score {int(st.scorer.score[0])}", flush=True)
            if st.knob_id is None:
                from tubecutter_sim.kit import Fingertips

                st.knob_id = st.art.find_bodies("knob")[0][0]
                st.tips = Fingertips(st.robot)
            kp = st.art.data.body_pos_w.torch[0, st.knob_id] - origins[0]
            kq = st.art.data.body_quat_w.torch[0, st.knob_id]
            kc = kp + math_utils.quat_apply(kq[None], torch.tensor([[props.KNOB_L / 2, 0.0, 0.0]], device=dev))[0]
            tips = st.tips.points(origins)[0]
            cq = st.art.data.body_quat_w.torch[0, st.cutter_id]
            yaw = math.degrees(2 * math.atan2(float(cq[2]), float(cq[3])))
            print(f"[cutter]   joints {st.art.joint_names} = {[round(float(v), 4) for v in jp[0]]} cutter yaw {yaw:.1f} "
                  f"bodies {st.art.body_names}", flush=True)
            print(f"[cutter]   knob centre {[round(float(v), 4) for v in kc]} left grasp "
                  f"{[round(float(v), 4) for v in grasp[0, 0]]} left pads {[[round(float(v), 4) for v in t] for t in tips[:2]]}",
                  flush=True)


@configclass
class EventsCfg:
    reset_task: EventTermCfg = MISSING
    step_task: EventTermCfg = MISSING

    def __init__(self, task: CutterTask):
        self.reset_task = EventTermCfg(func=task.reset, mode="reset")
        self.step_task = EventTermCfg(func=task.step, mode="interval", interval_range_s=(0.0, 0.0))
