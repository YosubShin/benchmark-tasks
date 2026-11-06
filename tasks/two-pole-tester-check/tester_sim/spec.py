"""Check a two-pole tester on its proving unit, test terminals L1 and N, check it on the proving unit again; stop if
it shows nothing on the proving unit.

Real parts (README References): a Fluke T150 two-pole voltage tester (yellow main unit with the L2/+ probe, the LED
bargraph and the LCD; a separate black L1 probe on a lead), a Fluke PRV240 proving unit (a 240 V battery source with
two recessed contacts and one LED) hung on the plate, and a strip of Phoenix Contact UT 4 feed-through terminal blocks
(L1 L2 L3 N PE) on a horizontal NS 35 DIN rail on the same plate. The strip is connected to nothing.

The procedure (Fluke T150 manual, "before and after use"; the live-dead-live check): both probe tips into the proving
unit's contacts, the tester must light; both tips into the two terminals; the proving unit again. If the tester shows
nothing on the proving unit, do not use it.

Desk frame (desk top z = 0, arm bases at (0, +-0.305, 0)): the plate's front face at x = PANEL_X faces the robot;
contacts open toward -x at height CONTACT_Z. Arm A (left) holds the main unit, arm B (right) the L1 probe.

Seeds: 0 (even) = everything works; 1 (odd) = the proving unit's batteries are out, so nothing lights on it (neither
its own LED nor the tester); the correct response is to stop without touching the terminals. No simulator imports.
"""

from __future__ import annotations

from dataclasses import dataclass

EPISODE_SECONDS = 120.0

PANEL_X = 0.485
"""Desk x of the mounting plate's front face (it faces the robot, -x)."""
CONTACT_Z = 0.165
"""Desk z of the contacts used: the proving unit's two and the strip's top-row screw wells. The tools are held level
(fingers forward) and the YAM's level grasp works at grasp heights 0.13-0.17 m."""
STRIP_Y0 = -0.020
"""Desk y of the first block's (L1) centre; the next blocks follow toward -y (the robot's right) at 6.2 mm."""
PROVER_Y = 0.060
"""Desk y of the midpoint between the proving unit's two contacts."""
TRAY_Y = (0.12, -0.12)
"""Desk y of the tray cradles for the main unit (arm A) and the L1 probe (arm B)."""
TRAY_GRASP_X = 0.22
"""Desk x of a tool's grasp point (its butt grip) while it rests in its cradle."""
TRAY_Z = 0.13
"""Desk z of the tools' axes in the tray: 3.5 cm below the contacts, so a tool held at a contact clears its cradle."""

TERMINALS = ("L1", "L2", "L3", "N", "PE")
TARGET = ("L1", "N")

INSTRUCTION = (
    "Pick up the yellow tester with your left hand and its black probe with your right hand. Check the tester on the "
    "yellow proving unit on the panel: push both probe tips into its two holes and see that the tester's lights come "
    "on. Then check terminals L1 and N: push one tip into the top screw hole of the grey terminal with the red "
    "marker (L1) and the other into the blue terminal (N). Then check the tester on the proving unit again. If the "
    "tester's lights do not come on at the proving unit, pull both tips back and stop: do not touch the terminals."
)


@dataclass(frozen=True)
class TesterSeed:
    seed: int
    prover_flat: bool
    """The proving unit's batteries are out: nothing lights on it (its LED and the tester stay dark)."""

    @property
    def instruction(self) -> str:
        return INSTRUCTION


def make_seed(seed: int) -> TesterSeed:
    assert seed >= 0, "seed must be >= 0"
    return TesterSeed(seed, prover_flat=(seed % 2 == 1))
