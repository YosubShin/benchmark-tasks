"""Meter test switch: layout and instruction. No simulator imports.

The switch is a 10-pole ANSI C12.9 meter test switch laid out as TESCO drawing 909884, at the standard's 0.750 in
(19.05 mm) pole pitch, mounted upright on the back wall of a meter box as installed (poles vertical, hinges at the
bottom). The test plug is a single current test plug of the TESCO 1077 class (ANSI C12.9 Fig. 6); its lead runs to a
test-set box on the desk. Dimensions and sources are in testswitch_sim/usd/switch_props.py.

Desk frame: desk top z = 0, arm bases at (0, +-0.305, 0); +x forward, +y to the robot's left.
"""

from __future__ import annotations

from dataclasses import dataclass

EPISODE_SECONDS = 160.0

FACE_X = 0.37
"""x of the switch base's front face (it faces the robot, -x)."""
CENTER_Z = 0.145
"""z of the base's centre. Poles run up and down (terminal rows at z -/+ 31.75 mm), the row of ten poles along y,
centred on y = 0."""

POLES = (
    # (function, handle colour) for poles 1..10 from +y (arm A's side, the robot's left) to -y. TESCO 909884 handle
    # colours: 1 orange, 2 red, 3 black, 4 yellow, 5 green, 6 black, 7 brown, 8 blue, 9 black, 10 white.
    ("short", "orange"), ("jack", "red"), ("volt", "black"),
    ("short", "yellow"), ("jack", "green"), ("volt", "black"),
    ("short", "brown"), ("jack", "blue"), ("volt", "black"),
    ("volt", "white"),
)
"""C12.9 layout C C V C C V C C V V(neutral): each phase has a double-pole short-circuiting current switch
(a short-circuiting pole and a test-jack pole) and a voltage pole."""

PHASE = {0: "A", 1: "A", 2: "A", 3: "B", 4: "B", 5: "B", 6: "C", 7: "C", 8: "C", 9: "N"}
PLUG_REST = (0.20, 0.17, 0.14)
"""Centre of the test plug's handle as it lies on its rest (arm A's side), probe pointing +x."""
TEST_SET = (0.36, 0.46, 0.0)
"""Centre of the test-set box's footprint (20 cm along x, 15 cm along y, 10 cm tall)."""
TEST_SET_SIZE = (0.20, 0.15, 0.10)
TARGET_JACK = 1
"""The phase-A test-jack pole (red handle), where the plug goes."""
TARGET_SHORT = 0
"""The phase-A short-circuiting pole (orange handle)."""

INSTRUCTION = (
    "At the meter test switch: open the three switches with black handles and the one with the white handle at the "
    "right end, then the orange switch at the left end, then the red switch next to it, and push the red test plug "
    "into the red switch's jack. Do not open the red switch while the orange one is closed. Leave the other switche"
    "s closed."
)


def instruction() -> str:
    return INSTRUCTION


@dataclass(frozen=True)
class Layout:
    seed: int = 0

    @property
    def instruction(self) -> str:
        return instruction()


def make_seed(seed: int) -> Layout:
    assert seed == 0, "one fixed layout (seed 0)"
    return Layout(seed)
