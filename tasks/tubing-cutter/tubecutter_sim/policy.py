"""Inspect Robots policy ``cutter-demo``: the scripted demo (tubecutter_sim.demo), closed loop, one action per control
step. Its only input is the observation Inspect Robots hands every policy: the ``top_cam``, ``left_cam`` and
``right_cam`` images and the 14-D ``joint_pos`` state. It holds no handle on the simulator or the task (it must run in
the simulator's process only because it imports the prop geometry module, which needs Isaac Sim's USD library)."""

from __future__ import annotations

from typing import Any

import numpy as np

from inspect_robots import Action, ActionChunk, Box, ObservationSpace
from inspect_robots.policy import PolicyBase, PolicyInfo
from inspect_robots.registry import policy
from inspect_robots.spaces import CameraSpec

CAMERAS = ("top_cam", "left_cam", "right_cam")


class DemoPolicy(PolicyBase):
    def __init__(self, control: bool = False) -> None:
        self.control = bool(control)
        self.info = PolicyInfo(
            name="cutter-demo", action_space=Box(shape=(14,)),
            observation_space=ObservationSpace(cameras=tuple(CameraSpec(c, 176, 320) for c in CAMERAS),
                                               state_keys=frozenset({"joint_pos"})))
        self._control_hz = 20.0
        self._demo = None

    def bind(self, embodiment_info: Any) -> None:
        if embodiment_info.control_hz:
            self._control_hz = float(embodiment_info.control_hz)

    def reset(self, scene: Any) -> None:
        self._demo = None

    def act(self, observation) -> ActionChunk:
        if self._demo is None:
            from tubecutter_sim.demo import Demo

            h, w = np.asarray(observation.images["top_cam"]).shape[:2]
            self._demo = Demo(w, h, control_hz=self._control_hz, control=self.control)
        return ActionChunk(actions=[Action(self._demo.next_action(observation))], control_hz=self._control_hz)


@policy("cutter-demo")
def cutter_demo(control: bool = False) -> DemoPolicy:
    """``-P control=true``: feed a full knob turn per revolution (must score 0: dents the tube)."""
    return DemoPolicy(control=str(control).lower() in ("1", "true", "yes"))
