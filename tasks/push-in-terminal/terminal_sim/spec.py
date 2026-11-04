"""Push a ferruled wire into the named push-in terminal and pass a tug: layout and instruction.

The scene: a vertical mounting plate in front of the robot with a TS 35 DIN rail and a strip of six WAGO TOPJOB S
2-conductor feedthrough terminal blocks (5.2 mm pitch), marked 1..6 with printed markers in the middle of their front
faces (group X1). X1:1-3 and X1:6 are grey (2002-1201, phase), X1:4 is blue (2002-1204, neutral), X1:5 green-yellow
(2002-1207, protective earth), as on a real strip. Their lower entries already hold wires; the upper entries are
empty. A blue 2.5 mm2 neutral conductor with a blue insulated DIN 46228-4 ferrule (WAGO 216-246) comes out of a slot
in the wiring duct under the strip, runs along the desk and up onto a stand, where its ferruled end lies. The robot
lifts the end, tilts it 15 deg like the terminal's angled front entry and pushes it straight in, no tool and no
release (Push-in CAGE CLAMP: a ferruled conductor is pushed in directly).

No simulator imports.
"""

from __future__ import annotations

from dataclasses import dataclass

EPISODE_SECONDS = 60.0

N_TERMINALS = 6
PITCH = 0.0052
"""WAGO 2002-1201: 5.2 mm wide (published)."""
STRIP_Y0 = -0.068
"""y of terminal 1 (the left-most from the robot's side; numbers rise toward -y, to the right)."""
TARGET = 4
"""The named terminal."""
TERMINAL_KINDS = {1: "grey", 2: "grey", 3: "grey", 4: "blue", 5: "green_yellow", 6: "grey"}
"""Housing colour per terminal: grey 2002-1201 (phase), blue 2002-1204 (N), green-yellow 2002-1207 (PE)."""
FRONT_X = 0.285
"""x of the terminals' front face (the profile's top, 32.9 mm above the rail)."""
TOP_Z = 0.175
"""z of the terminals' upper end (the end with the empty entries)."""
HOLDER_XY = (0.17, -0.13)
"""Where the wire's end (the collar's front face) lies at the start, on its stand."""
HOLDER_TOP = 0.09
"""z of the wire's end (on its axis) on the stand."""


def terminal_y(n: int) -> float:
    """y of terminal n (1-based)."""
    return STRIP_Y0 - (n - 1) * PITCH


def instruction(target: int = TARGET) -> str:
    return (
        f"Land the blue ferruled wire in terminal X1:{target}, the blue terminal, upper entry. Push the ferrule "
        "straight into the entry until the collar stops, then let go."
    )


@dataclass(frozen=True)
class Layout:
    seed: int = 0
    target: int = TARGET

    @property
    def instruction(self) -> str:
        return instruction(self.target)


def make_seed(seed: int) -> Layout:
    assert seed == 0, "one fixed layout (seed 0)"
    return Layout(seed)
