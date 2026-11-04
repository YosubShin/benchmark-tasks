"""CT test-terminal bridge task: layout constants and the instruction. No simulator imports.

The scene: a row of six Phoenix Contact URTK/S-class test-disconnect terminal blocks on a DIN rail, lying face up
on a 12 mm mounting plate on the table (a bench stand-in for a relay panel's terminal row). Three two-wire CT
circuits (1S1/1S2, 2S1/2S2, 3S1/3S2, separated by partition plates) come in from the CT side (toward the robot)
and leave to the relay on the far side. Each circuit's marker strips and the card in front of it carry a phase
colour (1 red, 2 yellow, 3 blue). Each terminal has a 4 mm test socket on each side of its disconnect slide. The
robot plugs a two-pole bridge (Phoenix KSS 8) into the CT-side test sockets of the named circuit's two terminals,
lets go, then with an insulated screwdriver pushes each of that circuit's two disconnect slides open.

Desk frame: table top z = 0, left arm base at (0, +0.305, 0), right arm base at (0, -0.305, 0).
"""

from __future__ import annotations

from dataclasses import dataclass

TASK_NAME = "ct-test-terminal-bridge"
EPISODE_SECONDS = 150.0

STRIP_X = 0.28
"""Desk-frame x of the terminal row's centre (the disconnect point); the CT side is toward the robot (-x)."""
STRIP_Y = 0.0
"""Desk-frame y of the named circuit's centre (between its two terminals)."""
PLATE_TOP = 0.012
"""Desk-frame z of the mounting plate the DIN rail is screwed to."""
HOLDER_XY = (0.20, 0.13)
"""The bridge's holder (two dummy sockets), on arm A's (left) side."""

CIRCUITS = ("1", "2", "3")
"""Three CT circuits from +y (robot's left) to -y; each has terminals <c>S1 and <c>S2."""
CIRCUIT_COLOURS = {"1": "red", "2": "yellow", "3": "blue"}
"""Phase colours of each circuit's marker strips and card (L1 red, L2 yellow, L3 blue)."""
NAMED = "2"


def instruction(circuit: str = NAMED) -> str:
    colour = CIRCUIT_COLOURS[circuit]
    others = " or ".join(CIRCUIT_COLOURS[c] for c in CIRCUITS if c != circuit)
    return (
        f"Take the black two-pin bridge from its holder and push it fully into the two test sockets nearest you on "
        f"terminals {circuit}S1 and {circuit}S2 (the pair with {colour} markers), then let go. After that, take the "
        f"screwdriver from its stand and, on each of those two terminals, put the blade down the slot and push the "
        f"slide at its bottom away from you until it stops. The bridge must be in before either slide moves. Do not "
        f"touch the terminals with {others} markers."
    )


@dataclass(frozen=True)
class Layout:
    seed: int
    named: str = NAMED

    @property
    def instruction(self) -> str:
        return instruction(self.named)


def make_seed(seed: int) -> Layout:
    return Layout(seed)
