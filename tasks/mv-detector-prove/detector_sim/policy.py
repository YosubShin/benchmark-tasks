"""Inspect Robots policy ``detector-demo``: the scripted demo (detector_sim.demo), closed loop, one action per control
step. It sees only the observation every policy gets from ``yam_arena`` (the three camera images and ``joint_pos``);
it does not bind to the task file or read the simulator."""

from __future__ import annotations

from inspect_robots import Action, ActionChunk, Box
from inspect_robots.policy import PolicyBase, PolicyInfo
from inspect_robots.registry import policy
from inspect_robots_yam.config import DEFAULT_CAMERAS, observation_space


class DemoPolicy(PolicyBase):
    def __init__(self, speed: float = 0.10) -> None:
        self.speed = float(speed)
        self.info = PolicyInfo(name="detector-demo", action_space=Box(shape=(14,)),
                               observation_space=observation_space(176, 320, DEFAULT_CAMERAS))
        self._control_hz = 20.0
        self._demo = None

    def bind(self, embodiment_info) -> None:
        if embodiment_info.control_hz:
            self._control_hz = float(embodiment_info.control_hz)

    def reset(self, scene) -> None:
        self._demo = None

    def act(self, observation) -> ActionChunk:
        if self._demo is None:
            from detector_sim.demo import Demo

            self._demo = Demo(control_hz=self._control_hz, speed=self.speed)
        return ActionChunk(actions=[Action(self._demo.next_action(observation))], control_hz=self._control_hz)


@policy("detector-demo")
def detector_demo(speed: float = 0.10) -> DemoPolicy:
    return DemoPolicy(speed=speed)
