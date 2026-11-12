"""Inspect Robots policy ``handhole-demo``: the scripted demo (handhole_sim.demo), closed loop, one action per control
step. ``act`` gets the observation every policy gets (three camera images and the 14-D joint state) and returns the
14-D joint action. It also binds to the task file the ``yam_arena`` embodiment loaded (as yam_scripted does) for its
one simulator read, the hose end's position during the insertion (see handhole_sim.demo), so it runs only in the
same process as that embodiment."""

from __future__ import annotations

from typing import Any

from inspect_robots import Action, ActionChunk, Box, CameraSpec, ObservationSpace
from inspect_robots.policy import PolicyBase, PolicyInfo
from inspect_robots.registry import policy


class DemoPolicy(PolicyBase):
    def __init__(self, speed: float = 0.10) -> None:
        self.speed = float(speed)
        self.info = PolicyInfo(name="handhole-demo", action_space=Box(shape=(14,)),
                               observation_space=ObservationSpace(
                                   cameras=tuple(CameraSpec(n, 176, 320) for n in ("top_cam", "left_cam", "right_cam")),
                                   state_keys=frozenset({"joint_pos"})))
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
            raise ValueError("handhole-demo needs --embodiment yam_arena -E task=handhole_air_check.py")

    def reset(self, scene: Any) -> None:
        self._demo = None

    def act(self, observation) -> ActionChunk:
        if self._demo is None:
            from handhole_sim.demo import Demo

            self._demo = Demo(self._station, control_hz=self._control_hz, speed=self.speed)
        return ActionChunk(actions=[Action(self._demo.next_action(observation))], control_hz=self._control_hz)


@policy("handhole-demo")
def handhole_demo(speed: float = 0.10) -> DemoPolicy:
    return DemoPolicy(speed=speed)
