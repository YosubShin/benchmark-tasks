"""Ball-valve clamp lockout: the instruction and the one layout, with no simulator imports.

Two identical 1/2 in plain-lever ball valves sit on a short pipe run on the table, both open (lever along the pipe).
Behind each valve an engraved valve tag gives its number, as plant valves are tagged. The robot closes the named
valve (a quarter turn clockwise seen from above, to the closed stop), fits a clamp-on ball valve lockout (Master
Lock S3080) over the closed lever and the valve, pushes its two halves together on the pipe, and padlocks them
through the aligned hole. The other valve must not be turned.
"""

from __future__ import annotations

from dataclasses import dataclass

TASK_NAME = "ball-valve-clamp-lockout"
EPISODE_SECONDS = 150.0
NUM_SEEDS = 1

TAGS = ("V-12", "V-14")
"""Valve tag numbers, slot 0 (on the midline) and slot 1 (25 cm left)."""


def instruction(tag: str, other: str) -> str:
    return (f"Close ball valve {tag}: turn its handle a quarter turn clockwise, seen from above, until it points at you. "
            f"Fit the red valve lockout over the handle and the valve, with the handle inside its channel and its "
            f"legs on both sides of the pipe, and push its two halves together. Then put your padlock's shackle "
            f"through the lined-up hole in the side of the lockout and snap it shut. Do not operate valve {other}.")


@dataclass(frozen=True)
class ValveSeed:
    seed: int
    target_slot: int = 0

    @property
    def target_tag(self) -> str:
        return TAGS[self.target_slot]

    @property
    def decoy_tag(self) -> str:
        return TAGS[1 - self.target_slot]

    @property
    def instruction(self) -> str:
        return instruction(self.target_tag, self.decoy_tag)


def make_seed(seed: int) -> ValveSeed:
    assert seed == 0, "one layout (seed 0)"
    return ValveSeed(seed)
