"""Inspect Robots policy ``fuse-demo``: the scripted demo (fuse_sim.demo), closed loop, one action per control step.
``act`` gets only the observation every policy gets (three camera images and the 14-D joint state) and returns the
14-D joint action; the demo does not read the simulator."""

from __future__ import annotations

import os
from typing import Any

from inspect_robots import Action, ActionChunk, Box, CameraSpec, ObservationSpace
from inspect_robots.policy import PolicyBase, PolicyInfo
from inspect_robots.registry import policy


class DemoPolicy(PolicyBase):
    def __init__(self, speed: float = 0.10) -> None:
        self.speed = float(speed)
        self.info = PolicyInfo(name="fuse-demo", action_space=Box(shape=(14,)),
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
            from fuse_sim.demo import Demo

            self._demo = Demo(control_hz=self._control_hz, speed=self.speed)
        dump = os.environ.get("FUSE_DEMO_DUMP")  # debugging: save what the demo saw
        if dump:
            import numpy as np

            os.makedirs(dump, exist_ok=True)
            np.savez_compressed(f"{dump}/obs_{self._k:05d}.npz",
                                **{f"img_{n}": v for n, v in observation.images.items()},
                                joint_pos=np.asarray(observation.state["joint_pos"]))
        self._k += 1
        return ActionChunk(actions=[Action(self._demo.next_action(observation))], control_hz=self._control_hz)


@policy("fuse-demo")
def fuse_demo(speed: float = 0.10) -> DemoPolicy:
    return DemoPolicy(speed=speed)
