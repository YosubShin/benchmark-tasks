"""Handwheel valve lockout: the instruction and the one layout, with no simulator imports.

Two identical 1/2 in handwheel gate valves sit on a short pipe run on the table, each one turn open. Behind each valve
an engraved valve tag gives its number on a service colour (yellow, green), as plant valves are tagged. The robot
closes the named valve (clockwise seen from above, about one turn, regripping as needed), fits a rotating gate valve
lockout (Master Lock 480) over its handwheel, swings the lockout's movable half shut and padlocks the two halves'
ears together. The other valve must not be turned.
"""

from __future__ import annotations

from dataclasses import dataclass

TASK_NAME = "handwheel-valve-lockout"
EPISODE_SECONDS = 180.0
NUM_SEEDS = 1

TAGS = ("V-21", "V-23")
"""Valve tag numbers, slot 0 (on the midline) and slot 1 (25 cm left)."""
TAG_COLOURS = ("yellow", "green")


def instruction(tag: str, colour: str, other: str, other_colour: str) -> str:
    return (f"Close gate valve {tag}, the one with the {colour} tag: turn its handwheel clockwise, seen from above, "
            f"until it stops, about one full turn; let go and grip again as often as you need. Slide the red valve "
            f"cover onto that handwheel from the side, open side first, so the stem sits in the middle of the cover, "
            f"then swing the cover's loose half round until its ear lies on the other half's ear. Put your padlock's "
            f"shackle through the inner holes of both ears and snap it shut. Do not touch valve {other}, the one with the "
            f"{other_colour} tag.")


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
        t, d = self.target_slot, 1 - self.target_slot
        return instruction(TAGS[t], TAG_COLOURS[t], TAGS[d], TAG_COLOURS[d])


def make_seed(seed: int) -> ValveSeed:
    assert seed == 0, "one layout (seed 0)"
    return ValveSeed(seed)
