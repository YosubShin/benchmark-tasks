"""Inspect Robots policy ``extinguisher-demo``: the scripted demo (fire_sim.demo), closed loop, one action per control
step. ``act`` gets only the observation every policy gets (three camera images and the 14-D joint state) and returns
the 14-D joint action; the demo does not touch the simulator. ``-P aim=tips`` aims at the flames' tips (control); ``-P test=reach_early|reach_late|drop_early|drop`` runs a
hazard check (see README)."""

from __future__ import annotations

from typing import Any

from inspect_robots import Action, ActionChunk, Box, CameraSpec, ObservationSpace
from inspect_robots.policy import PolicyBase, PolicyInfo
from inspect_robots.registry import policy


class DemoPolicy(PolicyBase):
    def __init__(self, aim: str = "base", test: str = "") -> None:
        assert aim in ("base", "tips"), "aim must be base or tips"
        self.aim, self.test = aim, test
        self.info = PolicyInfo(name="extinguisher-demo", action_space=Box(shape=(14,)),
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
            from fire_sim.demo import Demo

            self._demo = Demo(control_hz=self._control_hz, aim=self.aim, test=self.test)
        return ActionChunk(actions=[Action(self._demo.next_action(observation))], control_hz=self._control_hz)


@policy("extinguisher-demo")
def extinguisher_demo(aim: str = "base", test: str = "") -> DemoPolicy:
    return DemoPolicy(aim=aim, test=test)
