"""Rotary isolator lockout: the instruction and the one layout, with no simulator imports.

Two enclosed rotary isolators (ABB OT16 3-pole switch-disconnectors, each in a small grey enclosure with an ABB
OHB45J6 pistol handle on its front) hang side by side on a backboard on the table, both ON (handle pointing up).
An engraved nameplate on each enclosure gives its designation: Q1 on a yellow plate (the named one)
and Q2 on a white plate. The robot turns Q1's handle a quarter turn anticlockwise (as you face it) to OFF
(handle pointing left), pulls the red padlock slide (a stripe along the grip's lower edge) down by its tab at the
grip's tip (it only comes out at OFF, where its tail drops into a slot in the handle's fixed base), puts its padlock's
shackle through the hole in the tab and snaps it shut. Q2 must stay ON.
"""

from __future__ import annotations

from dataclasses import dataclass

TASK_NAME = "rotary-isolator-lockout"
EPISODE_SECONDS = 150.0
NUM_SEEDS = 1

TAGS = ("Q1", "Q2")
"""Isolator designations, slot 0 (on the midline, the named one) and slot 1 (24 cm right)."""
LOADS = ("CONVEYOR 3", "CONVEYOR 4")
PLATE_COLORS = ("yellow", "white")


def instruction() -> str:
    return ("Switch off isolator Q1, the one with the yellow nameplate: turn its handle a quarter turn anticlockwise, "
            "as you face it, from ON (pointing up) to OFF (pointing left). Then pull the red padlock slide down out "
            "of the grip by its tab at the grip's tip, put your padlock's shackle through the hole in the tab and "
            "snap it shut. "
            "Do not turn isolator Q2, the one with the white nameplate.")


@dataclass(frozen=True)
class IsolatorSeed:
    seed: int
    target_slot: int = 0

    @property
    def target_tag(self) -> str:
        return TAGS[self.target_slot]

    @property
    def instruction(self) -> str:
        return instruction()


def make_seed(seed: int) -> IsolatorSeed:
    assert seed == 0, "one layout (seed 0)"
    return IsolatorSeed(seed)
