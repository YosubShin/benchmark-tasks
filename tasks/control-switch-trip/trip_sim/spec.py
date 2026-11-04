"""Trip the named breaker from its control switch. No simulator imports.

A section of a control panel carries three breaker control switches side by side (Electroswitch Series 24, model
2438D: pistol-grip handle; TRIP 45 deg anticlockwise / NORMAL vertical / CLOSE 45 deg clockwise as seen from the
front; spring return to NORMAL; red/green mechanical target in a window of the nameplate). Above each switch: a green
(breaker open) and a red (breaker closed) indicating lamp, and above them an engraved nameplate naming the breaker.
Each nameplate has its own colour (white, yellow, blue) so the named one can be told apart where its text is too
small to read. Every breaker starts closed (red lamps lit, targets red).

The robot turns the named switch to TRIP and holds it there until the green lamp above it lights, then lets go. The
handle springs back to NORMAL by its own spring and the target stays green.

Desk frame (the station harness raises it by 0.75 m): desk top z = 0, arm bases at (0, +-0.305, 0), x forward.
"""

from __future__ import annotations

from dataclasses import dataclass

TASK_NAME = "control-switch-trip"
EPISODE_SECONDS = 40.0
NUM_SEEDS = 1

FACE_X = 0.30
"""x of the panel's front surface (the robot faces it, looking along +x)."""
AXIS_Z = 0.19
"""z of the switches' shafts. The grip hangs down, so a hand takes it 35-45 mm lower, in the YAM's level-finger band."""
SWITCH_Y = (-0.01, -0.10, -0.19)
"""y of the three shafts, the robot's left to right (90 mm centres). The named one (middle) is 20.5 cm left of arm B's
base. Arm B's ready pose puts its open hand at x 0.32, y -0.305 +- 0.05, z 0.26, so the panel's right edge must stay
left of y -0.25 (PANEL_RIGHT_Y)."""
PANEL_LEFT_Y = SWITCH_Y[0] + 0.09
PANEL_RIGHT_Y = SWITCH_Y[-1] - 0.042

BREAKERS = (
    # (device number, nameplate line 2, nameplate colour (RGB), text colour, colour name)
    ("52-1", "LINE 1 NORTH", (246, 246, 240), (0, 0, 0), "white"),
    ("52-2", "TRANSFORMER T1", (250, 200, 20), (0, 0, 0), "yellow"),
    ("52-3", "LINE 2 SOUTH", (25, 80, 200), (255, 255, 255), "blue"),
)
"""Device number (IEEE C37.2: 52 = AC circuit breaker), the engraved second line, and the lamicoid's colour."""

NAMED = 1
"""52-2 TRANSFORMER T1 (yellow nameplate), the middle switch."""


def instruction(slot: int = NAMED) -> str:
    dev, line2, _, _, colour = BREAKERS[slot]
    return (f"Trip breaker {dev} {line2} from its control switch, the one under the {colour} nameplate: turn that "
            "switch's pistol-grip handle anticlockwise (as you face the panel, toward the TRIP marking) and hold it "
            "there until the green lamp above it lights, then let go so the handle springs back by itself. Do not "
            "turn any other switch, and do not turn this one clockwise toward CLOSE.")


@dataclass(frozen=True)
class TripSeed:
    seed: int
    named_slot: int = NAMED

    @property
    def device(self) -> str:
        return BREAKERS[self.named_slot][0]

    @property
    def instruction(self) -> str:
        return instruction(self.named_slot)


def make_seed(seed: int) -> TripSeed:
    assert 0 <= seed < 20, "seed must be in [0, 20)"
    return TripSeed(seed)
