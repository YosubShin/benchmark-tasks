"""Inspect Robots policy ``ctbridge-demo``: the scripted demo (ctbridge_sim.demo), closed loop, one action per control
step. It binds to the task file the ``yam_arena`` embodiment loaded (as yam_scripted does) and reads its simulator
state, so it runs only in the same process as that embodiment."""

from __future__ import annotations

from typing import Any

from inspect_robots import Action, ActionChunk, Box, ObservationSpace
from inspect_robots.policy import PolicyBase, PolicyInfo
from inspect_robots.registry import policy


class DemoPolicy(PolicyBase):
    def __init__(self, control: bool = False) -> None:
        self.control = bool(control)
        self.info = PolicyInfo(name="ctbridge-demo", action_space=Box(shape=(14,)),
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
            raise ValueError("ctbridge-demo needs --embodiment yam_arena -E task=ct_test_terminal.py")

    def reset(self, scene: Any) -> None:
        self._demo = None

    def act(self, observation) -> ActionChunk:
        if self._demo is None:
            from ctbridge_sim.demo import Demo

            self._demo = Demo(self._station, control_hz=self._control_hz, control=self.control)
        return ActionChunk(actions=[Action(self._demo.next_action())], control_hz=self._control_hz)


@policy("ctbridge-demo")
def ctbridge_demo(control: bool = False) -> DemoPolicy:
    """``-P control=true``: leave the bridge in its holder, open both 2S links, force the check (must score 0)."""
    return DemoPolicy(control=str(control).lower() in ("1", "true", "yes"))
