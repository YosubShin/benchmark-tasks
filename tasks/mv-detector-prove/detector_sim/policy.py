"""Inspect Robots policy ``detector-demo``: the scripted demo (detector_sim.demo), closed loop, one action per control
step. It binds to the task file the ``yam_arena`` embodiment loaded (as yam_scripted does) and reads its simulator
state, so it runs only in the same process as that embodiment."""

from __future__ import annotations

from typing import Any

from inspect_robots import Action, ActionChunk, Box, ObservationSpace
from inspect_robots.policy import PolicyBase, PolicyInfo
from inspect_robots.registry import policy


class DemoPolicy(PolicyBase):
    def __init__(self, speed: float = 0.10) -> None:
        self.speed = float(speed)
        self.info = PolicyInfo(name="detector-demo", action_space=Box(shape=(14,)),
                               observation_space=ObservationSpace(state_keys=frozenset({"joint_pos"})))
        self._control_hz = 20.0
        self._station = None
        self._demo = None

    def bind(self, embodiment_info: Any) -> None:
        from yam_arena import task_file

        if embodiment_info.control_hz:
            self._control_hz = float(embodiment_info.control_hz)
        task = task_file.lookup(embodiment_info.environment_id)
        self._station = getattr(getattr(task, "module", None), "STATION", None)
        if self._station is None:
            raise ValueError("detector-demo needs --embodiment yam_arena -E task=mv_detector_prove.py")

    def reset(self, scene: Any) -> None:
        self._demo = None

    def act(self, observation) -> ActionChunk:
        if self._demo is None:
            from detector_sim.demo import Demo

            self._demo = Demo(self._station, control_hz=self._control_hz, speed=self.speed)
        return ActionChunk(actions=[Action(self._demo.next_action())], control_hz=self._control_hz)


@policy("detector-demo")
def detector_demo(speed: float = 0.10) -> DemoPolicy:
    return DemoPolicy(speed=speed)
