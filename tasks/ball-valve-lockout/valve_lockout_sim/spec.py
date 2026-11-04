"""Ball-valve lockout: the instruction and the one layout, with no simulator imports.

Two identical 1/2 in lockable ball valves sit on a short pipe run on the table, both open (lever along the pipe).
Behind each valve an engraved valve tag gives its number, as plant valves are tagged. The robot closes the named
valve (a quarter turn clockwise seen from above, to the closed stop), slides the lever's locking sleeve in so its
tongue drops into the stop plate's notch, and puts its padlock (Master Lock 410, open, on a pedestal) through the
hole in the lever just outboard of the sleeve and snaps it shut. The other valve must not be turned.
"""

from __future__ import annotations

from dataclasses import dataclass

TASK_NAME = "ball-valve-lockout"
EPISODE_SECONDS = 125.0
NUM_SEEDS = 1

TAGS = ("V-12", "V-14")
"""Valve tag numbers, slot 0 (on the midline) and slot 1 (25 cm left)."""


def instruction(tag: str, other: str) -> str:
    return (f"Close ball valve {tag}: turn its handle a quarter turn clockwise, seen from above, to its stop. Slide the "
            f"handle's locking sleeve toward the valve so it latches, then put your padlock's shackle through the hole "
            f"in the handle and snap it shut. Do not operate valve {other}.")


@dataclass(frozen=True)
class ValveSeed:
    seed: int
    target_slot: int = 0
    """Valve slot 0 (on the midline) or 1 (25 cm left) is the named one."""

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
