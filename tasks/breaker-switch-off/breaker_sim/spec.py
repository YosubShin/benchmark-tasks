"""Switch off the named circuit breaker in a small distribution board. No simulator imports.

Six Schneider Acti9 iC60N 1P breakers clip onto one 35 mm DIN rail behind a dead-front cover. Every breaker
starts ON. The ways are numbered 1-6 from the robot's left to its right. A printed label strip on the cover,
directly ABOVE the breakers, gives each way's number and circuit name on a label coloured by circuit type, and
a circuit directory card on the cover lists the same ways. The instruction names the circuit only (not the way
number), so the robot has to find which way feeds it from the labels. The breaker's contact-position window
must read O-OFF (green) at the end.

Desk frame (the station harness raises it by 0.75 m): desk top z = 0, arm bases at (0, +-0.305, 0), x forward.
"""

from __future__ import annotations

from dataclasses import dataclass

TASK_NAME = "breaker-switch-off"
EPISODE_SECONDS = 35.0
NUM_SEEDS = 1

FACE_X = 0.32
"""x of the breakers' front faces (the noses stick through the cover toward the robot, along -x)."""
RAIL_Z = 0.156
"""z of the DIN rail's centre line (the handle pivots sit 7.5 mm lower, in the YAM's level-finger band)."""
PITCH = 0.018
"""iC60N module width: 18 mm."""

CIRCUITS = (
    # (way, circuit name, rating, label colour (RGB 0-255), text colour) from the robot's left (+y) to its right.
    # Label colours mark the circuit type (our choice of colours; board label kits colour-code circuit types).
    (1, "LIGHTING - WORKSHOP", "C10", (247, 205, 30), (0, 0, 0)),
    (2, "HEATERS - OFFICE", "C16", (205, 35, 35), (255, 255, 255)),
    (3, "BATTERY CHARGER", "C20", (240, 120, 20), (0, 0, 0)),
    (4, "SOCKETS - KITCHEN", "C16", (25, 85, 215), (255, 255, 255)),
    (5, "HVAC - AHU-1", "C20", (40, 150, 60), (255, 255, 255)),
    (6, "SPARE", "C10", (245, 245, 240), (0, 0, 0)),
)
NAMED_SLOT = 3
"""Way 4, SOCKETS - KITCHEN (blue label)."""


def way_y(slot: int) -> float:
    """y of breaker `slot` (0 = way 1, at the robot's left)."""
    return (2.5 - slot) * PITCH


def instruction() -> str:
    return ("Switch off the breaker that feeds SOCKETS - KITCHEN (find it on the labels above the breakers or the "
            "circuit directory) so that its window reads O-OFF. Leave the other circuits on.")


@dataclass(frozen=True)
class SwitchOffSeed:
    seed: int
    named_slot: int = NAMED_SLOT

    @property
    def way(self) -> int:
        return CIRCUITS[self.named_slot][0]

    @property
    def circuit(self) -> str:
        return CIRCUITS[self.named_slot][1]

    @property
    def instruction(self) -> str:
        return instruction()


def make_seed(seed: int) -> SwitchOffSeed:
    assert 0 <= seed < 20, "seed must be in [0, 20)"
    return SwitchOffSeed(seed)
