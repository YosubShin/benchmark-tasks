"""Inspect Robots policy ``pipe-shield-demo``: the scripted demo (pipeshield_sim.demo), closed loop, one action per
control step. ``act`` gets only the observation every policy gets (three camera images and the 14-D joint state) and
returns the 14-D joint action; the demo does not touch the simulator. ``-P no_centre=1`` is the control run (the
shield is left where the first slide put it, about 12 cm off centre, and the pipe lowered onto it)."""

from __future__ import annotations

from typing import Any

from inspect_robots import Action, ActionChunk, Box, CameraSpec, ObservationSpace
from inspect_robots.policy import PolicyBase, PolicyInfo
from inspect_robots.registry import policy


class DemoPolicy(PolicyBase):
    def __init__(self, speed: float = 0.10, no_centre: int = 0) -> None:
        self.speed = float(speed)
        self.no_centre = bool(int(no_centre))
        self.info = PolicyInfo(name="pipe-shield-demo", action_space=Box(shape=(14,)),
                               observation_space=ObservationSpace(
                                   cameras=tuple(CameraSpec(n, 176, 320) for n in ("top_cam", "left_cam", "right_cam")),
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
            from pipeshield_sim.demo import Demo

            self._demo = Demo(control_hz=self._control_hz, speed=self.speed, no_centre=self.no_centre)
        return ActionChunk(actions=[Action(self._demo.next_action(observation))], control_hz=self._control_hz)


@policy("pipe-shield-demo")
def pipe_shield_demo(speed: float = 0.10, no_centre: int = 0) -> DemoPolicy:
    return DemoPolicy(speed=speed, no_centre=no_centre)
