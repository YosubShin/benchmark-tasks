"""Wall switch lockout: the instruction and the one layout, with no simulator imports.

Two wall toggle switches, each in its own flush box with a plain plate, are on a plywood wall board on the table,
both ON (toggle up). Each already carries the base of a Master Lock 496B wall switch lockout (installed once with the
plate's screws, as on site), its red cover hinged at the top and standing open. An engraved label beside each plate (on its left)
names it: S1 on yellow (the named one) and S2 on white. The robot pushes S1's toggle down to OFF, closes S1's lockout
cover over it (the cover's slot passes over the base's padlock ring), puts its padlock's shackle through the ring and
snaps it shut. S2 must stay ON.
"""

from __future__ import annotations

from dataclasses import dataclass

TASK_NAME = "wall-switch-lockout"
EPISODE_SECONDS = 150.0
NUM_SEEDS = 1

TAGS = ("S1", "S2")
"""Switch designations: slot 0 (the named one, near the midline) and slot 1 (16 cm left of it)."""
LABEL_COLORS = ("yellow", "white")


def instruction() -> str:
    return ("Turn wall switch S1, the one with the yellow label, OFF by pushing its toggle down. Then swing its red "
            "lockout cover down over the switch, put your padlock's shackle through the ring at the bottom of the "
            "cover and snap it shut. Leave switch S2, the one with the white label, ON.")


@dataclass(frozen=True)
class SwitchSeed:
    seed: int
    target_slot: int = 0

    @property
    def target_tag(self) -> str:
        return TAGS[self.target_slot]

    @property
    def instruction(self) -> str:
        return instruction()


def make_seed(seed: int) -> SwitchSeed:
    assert seed == 0, "one layout (seed 0)"
    return SwitchSeed(seed)
