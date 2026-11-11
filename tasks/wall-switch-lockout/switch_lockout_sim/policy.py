"""Inspect Robots policy ``wall-switch-demo``: the scripted demo (switch_lockout_sim.demo), closed loop, one action per
control step. ``-P control=true``: turn the named switch OFF and shut its cover, skip the padlock, then force the function test. It binds to the task file the ``yam_arena`` embodiment loaded (as yam_scripted does) and reads its simulator
state, so it runs only in the same process as that embodiment."""

from __future__ import annotations

from typing import Any

from inspect_robots import Action, ActionChunk, Box, ObservationSpace
from inspect_robots.policy import PolicyBase, PolicyInfo
from inspect_robots.registry import policy


class DemoPolicy(PolicyBase):
    def __init__(self, speed: float = 0.10, control: bool = False) -> None:
        self.speed = float(speed)
        self.control = str(control).lower() in ("1", "true", "yes")
        self.info = PolicyInfo(name="wall-switch-demo", action_space=Box(shape=(14,)),
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
            raise ValueError("wall-switch-demo needs --embodiment yam_arena -E task=wall_switch.py")

    def reset(self, scene: Any) -> None:
        self._demo = None

    def act(self, observation) -> ActionChunk:
        if self._demo is None:
            from switch_lockout_sim.demo import Demo

            self._demo = Demo(self._station, control_hz=self._control_hz, speed=self.speed, control=self.control)
        return ActionChunk(actions=[Action(self._demo.next_action())], control_hz=self._control_hz)


@policy("wall-switch-demo")
def wall_switch_demo(speed: float = 0.10, control: bool = False) -> DemoPolicy:
    return DemoPolicy(speed=speed, control=control)
