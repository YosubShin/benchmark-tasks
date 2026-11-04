"""Ground clamp order: layout and instruction.

Scene (desk frame: table top z = 0, arm bases at (0, +-0.305, 0)): an insulating back board carrying three flat copper
phase bars (A, B, C, on the left arm's side) and a copper ground bar (GND, green/yellow band, on the right arm's side),
all 1/4 x 2 in, horizontal, their front edges toward the robot. A protective grounding set: two C-type grounding
clamps with eye screws (CHANCE PSC6003412 class, used upside down on the bench: hook jaw under the bar, eye screw on
top), joined by a yellow ground cable. A short clamp stick (12 cm, a square handle) is fixed on each eye screw. At the
start each clamp hangs, loose, on an insulating parking plate: the ground clamp right of the ground bar, the line clamp
left of bar C. A yellow TESTED tag hangs on the board above bar B, the bar that has been tested dead.

Order (OSHA 29 CFR 1910.269(n)(6)(i)): ground end first, then the line end, on a tested bar only. The robot hangs the
ground clamp on GND and turns its stick until the clamp is tight, then hangs the line clamp on bar B and tightens it.
"""

from __future__ import annotations

from dataclasses import dataclass

TASK_NAME = "ground-clamp-order"
EPISODE_SECONDS = 260.0

PHASES = ("A", "B", "C")
PHASE_Y = (0.135, 0.22, 0.305)
"""Desk-frame y of the phase bars' centres (left arm's side)."""
PHASE_LEN = 0.07
GROUND_Y = -0.16
GROUND_LEN = 0.12
PARK_Y = (-0.26, 0.39)
"""Insulating parking plates: ground clamp (right arm's side), line clamp (left arm's side)."""
PARK_LEN = 0.05
TAGGED = 1
"""Index into PHASES of the bar tagged TESTED (B)."""


def instruction() -> str:
    return (
        "Ground bar B, the bar with the yellow TESTED tag. First take the ground clamp (on the right, by its black "
        "stick handle), hang it on the ground bar GND and turn its stick clockwise until the clamp is tight. Only then "
        "take the line clamp (on the left), hang it on bar B and tighten it the same way. Let go of both sticks at the "
        "end. Do not touch bars A or C."
    )


@dataclass(frozen=True)
class Layout:
    seed: int = 0
    tagged: int = TAGGED

    @property
    def instruction(self) -> str:
        return instruction()


def make_seed(seed: int = 0) -> Layout:
    return Layout(seed)
