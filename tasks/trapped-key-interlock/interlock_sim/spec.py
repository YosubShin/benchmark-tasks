"""Trapped-key interlock task: layout and instruction (no simulator imports).

Two interlock pairs stand on the table, each a breaker with a Kirk Type F interlock (locked-open type) holding its
own trapped key, and a fuse-compartment door held shut by a second Kirk Type F (locked-closed type) with the same
code: pair A1 on the left, pair B1 on the right. The two codes are cut differently, so a key enters only the locks
of its own code. Both breakers are open (handles at OFF). The instruction names one pair (A1 by default): turn
that breaker's key a quarter turn (its bolt extends in front of the handle, locking the breaker open, and only then
does the key come free), pull it out, take it to the door lock with the same code, insert it and turn it a quarter
turn back (that door's bolt withdraws and the key is trapped there). The final check proves the function: tug the
key (trapped), push the named breaker's handle toward ON (blocked by its bolt), pull the door (opens).

Desk frame: table top z = 0, left arm base at (0, +0.305), right arm base at (0, -0.305); x forward.
Dimensions and sources: interlock_sim/usd/props.py.
"""

from __future__ import annotations

from dataclasses import dataclass

TASK_NAME = "trapped-key-interlock"
EPISODE_SECONDS = 90.0
CODES = ("A1", "B1")
BREAKER_XY = {"A1": (0.17, 0.40), "B1": (0.17, -0.40)}
"""Centre of each breaker's footprint (its lock's keyway is 52.1 mm further forward and 11 mm to the right)."""
DOOR_XY = {"A1": (0.29, 0.07), "B1": (0.29, -0.20)}
"""Each door lock's keyway axis (its door spans 4-14 cm to the left of it, 5 cm behind)."""
DEFAULT_TARGET = "A1"


def other(code: str) -> str:
    return CODES[1 - CODES.index(code)]


def instruction(target: str = DEFAULT_TARGET) -> str:
    return (
        f"Both breakers are open. Turn key {target} a quarter turn in breaker {target}'s interlock to lock that "
        f"breaker open and free the key. Then take key {target} to the interlock on fuse compartment door {target}, "
        f"insert it, and turn it a quarter turn to unlock the door. Use only the locks marked {target}."
    )


@dataclass(frozen=True)
class Layout:
    target: str = DEFAULT_TARGET

    @property
    def other(self) -> str:
        return other(self.target)

    @property
    def instruction(self) -> str:
        return instruction(self.target)


def make_layout(target: str = DEFAULT_TARGET) -> Layout:
    assert target in CODES, f"target must be one of {CODES}"
    return Layout(target)
