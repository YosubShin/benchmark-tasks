"""Run an Isaac Lab-Arena task (its own props and TaskBase) as a yam-isaac-lab-arena task file in Dream's station (yam_arena 0059d02 only).

A task file built on this harness is a few lines::

    from tubecutter_sim import station as hb
    STATION = hb.install(hb.StationSpec(
        props=lambda: [Strip(), Wire(), ...],      # our Arena Objects, desk frame (built after Kit starts)
        task=lambda: PushInTask(...),              # our Arena TaskBase (reset + per-step interval events)
        objects=("strip", "wire", ...),            # names declared to yam_arena as OBJECTS
        pose_modules=("tubecutter_sim.task",),
        physics_hz=480.0,                          # optional: this task's physics rate (yam_arena's is 240 Hz)
    ))
    SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
    success, score, demo = STATION.success, STATION.score, STATION.demo

Frames. Our tasks are authored in a desk frame: desk top z = 0, left arm base at (0, +0.305, 0), right at
(0, -0.305, 0). The station's arm bases are at (0, +-0.305, 0.75) and its table top at 0.7475, so the desk frame is
the station frame lowered by ``Z0`` = 0.75 (the arm base height, which keeps every prop exactly where it is relative
to the arms; fixed props then stand 2.5 mm above the station table). Our task code sees ``env.scene.env_origins``
raised by Z0, so all its ``pos - origins`` / ``pose + origins`` arithmetic stays in the desk frame.

What it patches, all at run time, nothing on disk (verified against yam_arena 0059d02):

1. ``yam_arena.scene.build_task_environment`` -> the station room and camera stand (``scene.task_room``), our props,
   and our TaskBase, its reset/step events wrapped to see the ``EnvView`` below. Installed by an import hook on
   ``yam_arena.scene`` (which yam_arena imports only after Kit starts).
2. ``yam_arena.scene.object_states`` -> also handles articulations (root pose; ArenaWorld has no AABB for them).
3. ``yam_arena.inspect_embodiment.YamArenaEmbodiment._ensure_sim`` -> after the env is built, wraps its ``reset``
   so that, with ``hold_reset_pose=True`` (tool-in-hand starts), yam_arena's 0.5 s reset settle commands the joint
   targets our task's reset wrote instead of READY with open grippers.
3b. ``YamArenaEmbodiment.step`` -> with ``test_phase`` set, the task's final scoring test (a function test that
   removes props or loads the mechanism) runs off camera: inside the control step where it starts, to its end, and
   the episode ends there, returning the last pre-test observation. Recordings (``--store-frames``) therefore end at
   the robot's release and never show the test. Both patches find the task file's ``STATION`` through
   ``embodiment.task.module``, so they are installed once per process.
4. While ``props()`` runs: the global ``Pose`` of each module in ``pose_modules`` -> a Pose raised by Z0 (prop
   classes build their pose in ``__init__``; a fixed-base articulation keeps its spawn pose, so shifting after
   construction is too late).
5. ``reset_robot_home`` in each ``pose_modules`` module -> no-op
   (yam_arena's own reset event puts the arms at READY; it runs before our task's reset event).
6. Optional, per task file: ``physics_hz`` / ``physx`` in the ``StationSpec`` wrap yam_arena's ``env_cfg_callback``:
   after it has set its 240 Hz physics, the physics step becomes ``1 / physics_hz`` and the decimation (and render
   interval) ``physics_hz / control_hz``, so the policy still acts at ``-E control_hz`` and one frame is rendered per
   action; ``physx`` sets PhysX scene attributes (``env_cfg.sim.physics``). yam_arena reads time only through the
   env's ``step_dt`` (reset settle, episode clock) and feeds gravity compensation every physics substep, so its arms
   behave the same at the finer rate. Other task files in the same process are unaffected (it is the task's own
   environment config).

``EnvView`` / ``ArmsView``: our task code expects one articulation ``robot`` with ``left_*``/``right_*`` bodies and
joints ``{left,right}_joint1..8``. yam_arena has ``left_robot`` and ``right_robot`` (bodies ``gripper``,
``tip_left``, ...; joints ``joint1..8``). ``ArmsView`` presents them as one: bodies and joints prefixed, the
``gripper`` link's orientation converted to our flange convention (theirs = ours rotated by Rz(90) Rx(180): our
grasp point 0.1347 m along -z is their TCP 0.134 m along +z), and finger joints sign-flipped (theirs: 0 closed,
-0.04695 open; ours: 0 closed, +0.04695 open). Finger links are the same frames in both URDFs.

Scoring: after each control step the harness reads ``task.episode_report(view)`` (every task of ours has it):
``score()`` = milestone / 100, ``success()`` = report["success"]; ``--scorer task_score`` reports it.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import math
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

from yam_arena.task_file import Box

PINNED_YAM_ARENA = "0059d02a1a91bc6834c1d0846022891df0c78730"
SCENE = "station"
Z0 = 0.75
"""Desk frame -> station frame: z + Z0 (the station's arm base height)."""
# Their gripper frame = our flange frame * Rz(pi/2) Rx(pi): q_ours = q_theirs * conj(Q_REL).
Q_REL = (math.sqrt(0.5), math.sqrt(0.5), 0.0, 0.0)  # xyzw of Rz(90) * Rx(180)


@dataclass(frozen=True)
class Prop(Box):
    """An object declared to yam_arena by name; the harness builds the real prop (``pos``/``size`` are unused)."""


@dataclass
class StationSpec:
    props: Callable[[], list]
    task: Callable[[], Any]
    objects: Sequence[str]
    pose_modules: Sequence[str] = ()
    hold_reset_pose: bool = False
    on_step: Callable[[Any, Any], None] | None = None
    """Optional ``(view, task)`` after the task's own step each control step (e.g. a forced control test)."""
    success_hold_s: float = 0.0
    demo: Callable[[], dict] | None = None
    """``yam_scripted`` waypoints, if the task has a top-down demo; else the replay policy is used."""
    test_phase: Callable[[Any, Any], str] | None = None
    """Off-camera scoring test: ``(view, task) -> "idle" | "running" | "done"``. Once the task starts its final test
    (e.g. removing a coworker's lock and loading the mechanism), the harness runs it to the end inside that one
    control step, holding the policy's last action, and ends the episode. The observation it returns, and so the
    last stored frame, is the one from before the test, so recordings stop at the robot's release."""
    test_max_s: float = 10.0
    physics_hz: float | None = None
    """This task's physics rate (None: yam_arena's 240 Hz). The control rate stays ``-E control_hz``."""
    physx: dict = field(default_factory=dict)
    """PhysX scene attributes to set on ``env_cfg.sim.physics`` (e.g. ``friction_correlation_distance``)."""


@dataclass
class Station:
    spec: StationSpec
    SCENE: str = SCENE
    OBJECTS: list = field(default_factory=list)
    SUCCESS_HOLD_S: float = 0.0
    report: dict | None = None
    last_observation: Any = None
    task: Any = None
    env: Any = None
    """The ``EnvView`` of the running env (for scripts that drive it directly, e.g. reference recorders)."""

    def success(self, objects) -> bool:
        return bool(self.report and self.report.get("success"))

    def score(self, objects) -> float:
        return (self.report or {}).get("milestone", 0) / 100.0

    @property
    def demo(self):
        return self.spec.demo


# -- views --------------------------------------------------------------------------------------------------------


def _torch():
    import torch

    return torch


class _Field:
    def __init__(self, t):
        self.torch = t


def _q_mul(a, b):
    torch = _torch()
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack([aw * bx + ax * bw + ay * bz - az * by, aw * by - ax * bz + ay * bw + az * bx,
                        aw * bz + ax * by - ay * bx + az * bw, aw * bw - ax * bx - ay * by - az * bz], -1)


class ArmsView:
    """``left_robot`` + ``right_robot`` as our single ``robot`` (see the module docstring)."""

    def __init__(self, scene):
        self.arms = (scene["left_robot"], scene["right_robot"])
        left, right = self.arms
        self.body_names = [f"left_{n}" for n in left.body_names] + [f"right_{n}" for n in right.body_names]
        self.joint_names = [f"left_{n}" for n in left.joint_names] + [f"right_{n}" for n in right.joint_names]
        self._nb, self._nj = left.num_bodies, left.num_joints
        torch = _torch()
        dev = left.data.joint_pos.torch.device
        self._gripper = torch.tensor([n.endswith("_gripper") for n in self.body_names], device=dev)
        self._q_rel_conj = torch.tensor([-Q_REL[0], -Q_REL[1], -Q_REL[2], Q_REL[3]], device=dev)
        self._sign = torch.tensor([-1.0 if n.split("_", 1)[1] in ("joint7", "joint8") else 1.0
                                   for n in self.joint_names], device=dev)
        self.data = self

    @property
    def num_bodies(self):
        return len(self.body_names)

    def _find(self, names, pool):
        ids = []
        for name in names:
            matches = [i for i, n in enumerate(pool) if n == name]
            assert matches, f"{name!r} not among {pool}"
            ids.append(matches[0])
        return ids, list(names)

    def find_bodies(self, names, preserve_order=True):
        return self._find([names] if isinstance(names, str) else names, self.body_names)

    def find_joints(self, names, preserve_order=True):
        return self._find([names] if isinstance(names, str) else names, self.joint_names)

    # data (``robot.data.X.torch``)
    @property
    def body_pos_w(self):
        return _Field(_torch().cat([a.data.body_pos_w.torch for a in self.arms], dim=1))

    @property
    def body_quat_w(self):
        q = _torch().cat([a.data.body_quat_w.torch for a in self.arms], dim=1)
        ours = _q_mul(q, self._q_rel_conj.expand_as(q))
        return _Field(_torch().where(self._gripper[None, :, None], ours, q))

    def _joints(self, attr):
        return _Field(_torch().cat([getattr(a.data, attr).torch for a in self.arms], dim=1) * self._sign)

    @property
    def joint_pos(self):
        return self._joints("joint_pos")

    @property
    def joint_vel(self):
        return self._joints("joint_vel")

    @property
    def joint_pos_target(self):
        return self._joints("joint_pos_target")

    # writes (tool-in-hand starts)
    def _split(self, values, joint_ids):
        torch = _torch()
        ids = list(range(2 * self._nj)) if joint_ids is None else [int(i) for i in joint_ids]
        values = values * self._sign[ids]
        for arm in (0, 1):
            cols = [k for k, i in enumerate(ids) if (i >= self._nj) == bool(arm)]
            if cols:
                yield self.arms[arm], values[:, cols], [ids[k] - arm * self._nj for k in cols]

    def write_joint_state_to_sim(self, position, velocity, joint_ids=None, env_ids=None, **kw):
        for (arm, pos, ids), (_, vel, _) in zip(self._split(position, joint_ids), self._split(velocity, joint_ids)):
            arm.write_joint_state_to_sim(pos, vel, joint_ids=ids, env_ids=env_ids)

    def set_joint_position_target(self, target, joint_ids=None, env_ids=None):
        for arm, tgt, ids in self._split(target, joint_ids):
            arm.set_joint_position_target(tgt, joint_ids=ids, env_ids=env_ids)


class SceneView:
    def __init__(self, scene):
        object.__setattr__(self, "_scene", scene)
        object.__setattr__(self, "_robot", None)

    @property
    def env_origins(self):
        o = self._scene.env_origins.clone()
        o[:, 2] += Z0
        return o

    def __getitem__(self, key):
        if key == "robot":
            if self._robot is None:
                object.__setattr__(self, "_robot", ArmsView(self._scene))
            return self._robot
        return self._scene[key]

    def __getattr__(self, name):
        return getattr(self._scene, name)


class EnvView:
    """The env as our task code expects it: desk-frame origins and one ``robot``. Attribute writes go to the env."""

    def __init__(self, env):
        env = getattr(env, "_env", env)
        object.__setattr__(self, "_env", env)
        if getattr(env, "_hb_scene_view", None) is None:
            env._hb_scene_view = SceneView(env.scene)
        object.__setattr__(self, "scene", env._hb_scene_view)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def __setattr__(self, name, value):
        setattr(self._env, name, value)


# -- patches ------------------------------------------------------------------------------------------------------


def _wrap_task(station: Station, task):
    reset, step = task.reset, task.step

    def wrapped_reset(env, env_ids):
        view = EnvView(env)
        station.env = view
        reset(view, env_ids)
        station.report = task.episode_report(view, 0)

    def wrapped_step(env, env_ids):
        view = EnvView(env)
        step(view, env_ids)
        if station.spec.on_step is not None:
            station.spec.on_step(view, task)
        station.report = task.episode_report(view, 0)

    task.reset, task.step = wrapped_reset, wrapped_step  # get_events_cfg() binds these instance attributes
    return task


def _env_cfg_callback(spec: StationSpec, scene_module, control_hz):
    """yam_arena's env_cfg callback, then this task's physics rate and PhysX settings (patch 6)."""
    configure = scene_module.env_cfg_callback(control_hz)
    if spec.physics_hz is None and not spec.physx:
        return configure

    def configure_task(env_cfg):
        from yam_arena import constants

        env_cfg = configure(env_cfg)
        if spec.physics_hz is not None:
            hz = control_hz if control_hz is not None else constants.CONTROL_HZ
            decimation = max(1, round(spec.physics_hz / hz))
            env_cfg.sim.dt = 1.0 / spec.physics_hz
            env_cfg.decimation = decimation
            env_cfg.sim.render_interval = decimation
        for name, value in spec.physx.items():
            assert hasattr(env_cfg.sim.physics, name), f"PhysX setting {name!r} not found"
            setattr(env_cfg.sim.physics, name, value)
        return env_cfg

    return configure_task


def _patch_scene(station: Station, scene_module) -> None:
    from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
    from isaaclab_arena.scene.scene import Scene
    from isaaclab_arena.utils.pose import Pose

    spec = station.spec
    modules = [importlib.import_module(m) for m in spec.pose_modules]
    for m in modules:
        if hasattr(m, "reset_robot_home"):
            m.reset_robot_home = lambda env, env_ids: None

    def raised(position_xyz, rotation_xyzw=(0.0, 0.0, 0.0, 1.0)):
        x, y, z = position_xyz
        return Pose(position_xyz=(x, y, z + Z0), rotation_xyzw=rotation_xyzw)

    def build_task_environment(embodiment, task, control_hz=None):
        for m in modules:
            m.Pose = raised
        try:
            props = list(spec.props())
        finally:
            for m in modules:
                m.Pose = Pose
        station.task = _wrap_task(station, spec.task())
        return IsaacLabArenaEnvironment(
            name=f"yam_bimanual_task_{task.name}",
            scene=Scene([*scene_module.task_room(task.scene, embodiment.rig.top_cam), *props]),
            embodiment=embodiment,
            task=station.task,
            env_cfg_callback=_env_cfg_callback(spec, scene_module, control_hz),
        )

    object_states = scene_module.object_states

    def object_states_with_articulations(env, names):
        import numpy as np

        base = env.unwrapped
        arts = base.scene.articulations
        out = object_states(env, [n for n in names if n not in arts])
        origin = base.scene.env_origins[0]

        def arr(t):
            return t.detach().cpu().numpy().astype(np.float64)

        for name in (n for n in names if n in arts):
            d = arts[name].data
            pos = arr(d.root_pos_w.torch[0] - origin)
            out[name] = scene_module.task_file.ObjectState(
                pos=pos, quat_xyzw=arr(d.root_quat_w.torch[0]), lin_vel=arr(d.root_lin_vel_w.torch[0]),
                ang_vel=arr(d.root_ang_vel_w.torch[0]), low=pos, high=pos)
        return {n: out[n] for n in names}

    scene_module.build_task_environment = build_task_environment
    scene_module.object_states = object_states_with_articulations


def _station_of(emb) -> Station | None:
    """The Station of the task file this embodiment loaded (patches are per class, stations per task file)."""
    module = getattr(getattr(emb, "task", None), "module", None)
    station = getattr(module, "STATION", None)
    return station if isinstance(station, Station) else None


def _patch_embodiment() -> None:
    from inspect_robots.types import StepResult
    from yam_arena import inspect_embodiment as ie

    if getattr(ie.YamArenaEmbodiment._ensure_sim, "_hb_station", False):
        return
    ensure_sim = ie.YamArenaEmbodiment._ensure_sim
    step = ie.YamArenaEmbodiment.step

    def patched_step(self, action):
        res = step(self, action)
        station = _station_of(self)
        if station is None or station.spec.test_phase is None or station.env is None:
            return res
        phase = station.spec.test_phase(station.env, station.task)
        if phase == "idle":
            station.last_observation = res.observation
            return res
        n, limit = 0, round(station.spec.test_max_s * self.info.control_hz)
        while phase != "done" and n < limit:
            res = step(self, action)
            n += 1
            phase = station.spec.test_phase(station.env, station.task)
        info = {**res.info, "scoring_test_steps": n}
        return StepResult(observation=station.last_observation or res.observation, terminated=True,
                          termination_reason="success" if res.terminated else "scored", info=info)

    def patched(self, goal):
        fresh = self._env is None
        ensure_sim(self, goal)
        station = _station_of(self)
        if not (fresh and station is not None and station.spec.hold_reset_pose):
            return
        env, emb = self._env, self
        reset = env.reset

        def reset_then_hold(*args, **kwargs):
            out = reset(*args, **kwargs)
            torch = emb._torch
            rows = []
            for name in ie.ARMS:
                arm_ids, finger_ids = emb._ids[name]
                tgt = emb._base.scene[name].data.joint_pos_target.torch[0]
                grip = (tgt[finger_ids].mean() / ie.FINGER_OPEN).clamp(0.0, 1.0)
                rows.append(torch.cat([tgt[arm_ids], grip.reshape(1)]))
            emb._ready = torch.cat(rows).unsqueeze(0).to(torch.float32)
            return out

        env.reset = reset_then_hold

    patched._hb_station = True
    ie.YamArenaEmbodiment._ensure_sim = patched
    ie.YamArenaEmbodiment.step = patched_step


class _AfterImport(importlib.abc.MetaPathFinder):
    def __init__(self, name, fn):
        self.name, self.fn = name, fn

    def find_spec(self, fullname, path, target=None):
        if fullname != self.name:
            return None
        sys.meta_path.remove(self)
        spec = importlib.machinery.PathFinder.find_spec(fullname, path)
        if spec is None:
            return None
        exec_module = spec.loader.exec_module

        def exec_and_patch(module):
            exec_module(module)
            self.fn(module)

        spec.loader.exec_module = exec_and_patch
        return spec


def install(spec: StationSpec) -> Station:
    """Register the patches (once Kit has started and yam_arena.scene is imported) and return the task file's names."""
    _check_pin()
    station = Station(spec=spec, OBJECTS=[Prop(n, pos=(0.0, 0.0, Z0 + 0.1)) for n in spec.objects],
                      SUCCESS_HOLD_S=spec.success_hold_s)
    _patch_embodiment()
    if "yam_arena.scene" in sys.modules:
        _patch_scene(station, sys.modules["yam_arena.scene"])
    else:
        for f in [f for f in sys.meta_path if isinstance(f, _AfterImport) and f.name == "yam_arena.scene"]:
            sys.meta_path.remove(f)
        sys.meta_path.insert(0, _AfterImport("yam_arena.scene", lambda m: _patch_scene(station, m)))
    return station


def _check_pin() -> None:
    """Refuse other yam_arena versions when the checkout is visible (the patches name its internals)."""
    import subprocess
    from pathlib import Path

    import yam_arena

    repo = Path(yam_arena.__file__).resolve().parents[1]
    try:
        head = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return
    if head and head != PINNED_YAM_ARENA:
        raise RuntimeError(f"this task needs yam-isaac-lab-arena at {PINNED_YAM_ARENA[:7]}, found {head[:7]} in {repo}")
