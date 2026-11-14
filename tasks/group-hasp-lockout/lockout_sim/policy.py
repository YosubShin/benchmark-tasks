"""Inspect Robots policy ``lockout-demo``: the scripted demo (lockout_sim.demo), closed loop, one action per
control step. ``act`` gets only the observation every policy gets from ``yam_arena`` (the three camera images and
the 14-D ``joint_pos`` state) and returns the 14-D joint action; it does not bind to the task file or read the
simulator."""

from __future__ import annotations

import os
from typing import Any

from inspect_robots import Action, ActionChunk, Box, CameraSpec, ObservationSpace
from inspect_robots.policy import PolicyBase, PolicyInfo
from inspect_robots.registry import policy


class DemoPolicy(PolicyBase):
    def __init__(self, speed: float = 0.10) -> None:
        self.speed = float(speed)
        self.info = PolicyInfo(name="lockout-demo", action_space=Box(shape=(14,)),
                               observation_space=ObservationSpace(
                                   cameras=tuple(CameraSpec(n, 176, 320) for n in ("top_cam", "left_cam", "right_cam")),
                                   state_keys=frozenset({"joint_pos"})))
        self._control_hz = 20.0
        self._demo = None
        self._k = 0

    def bind(self, embodiment_info: Any) -> None:
        if embodiment_info.control_hz:
            self._control_hz = float(embodiment_info.control_hz)

    def reset(self, scene: Any) -> None:
        self._demo = None
        self._k = 0

    def act(self, observation) -> ActionChunk:
        if self._demo is None:
            from lockout_sim.demo import Demo

            self._demo = Demo(control_hz=self._control_hz, speed=self.speed)
        dump = os.environ.get("LOCKOUT_DEMO_DUMP")  # debugging: save what the demo saw
        if dump:
            import numpy as np

            os.makedirs(dump, exist_ok=True)
            np.savez_compressed(f"{dump}/obs_{self._k:05d}.npz",
                                **{f"img_{n}": np.asarray(v) for n, v in observation.images.items()},
                                joint_pos=np.asarray(observation.state["joint_pos"]))
        self._k += 1
        return ActionChunk(actions=[Action(self._demo.next_action(observation))], control_hz=self._control_hz)


@policy("lockout-demo")
def lockout_demo(speed: float = 0.10) -> DemoPolicy:
    return DemoPolicy(speed=speed)
