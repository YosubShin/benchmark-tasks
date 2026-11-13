"""Inspect Robots policy ``cylinder-chain-demo``: the scripted demo (cylchain_sim.demo), closed loop, one action per
control step. ``act`` gets only the observation every policy gets (three camera images and the 14-D joint state) and
returns the 14-D joint action; the demo does not touch the simulator. ``-P slack=<n>`` is the control run (the end is
brought n links less far, so the chain is hooked further along and left slack)."""

from __future__ import annotations

from typing import Any

from inspect_robots import Action, ActionChunk, Box, CameraSpec, ObservationSpace
from inspect_robots.policy import PolicyBase, PolicyInfo
from inspect_robots.registry import policy


class DemoPolicy(PolicyBase):
    def __init__(self, speed: float = 0.10, slack: int = 0) -> None:
        self.speed = float(speed)
        self.slack = int(slack)
        self.info = PolicyInfo(name="cylinder-chain-demo", action_space=Box(shape=(14,)),
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
            from cylchain_sim.demo import Demo

            self._demo = Demo(control_hz=self._control_hz, speed=self.speed, slack=self.slack)
        return ActionChunk(actions=[Action(self._demo.next_action(observation))], control_hz=self._control_hz)


@policy("cylinder-chain-demo")
def cylinder_chain_demo(speed: float = 0.10, slack: int = 0) -> DemoPolicy:
    return DemoPolicy(speed=speed, slack=slack)
