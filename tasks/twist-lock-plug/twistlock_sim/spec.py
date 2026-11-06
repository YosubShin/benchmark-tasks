"""Plug a NEMA L5-30 twist-lock cord cap into the named receptacle and turn it to lock: layout and instruction.

The scene: a vertical steel plate (closing a grey box) with two identical Hubbell HBL2616 L5-30R flanged twist-lock
receptacles, each with an engraved phenolic nameplate above it: 'CKT 7 BATTERY CHARGER' (red, white letters) over the
one on the left, 'CKT 9 SPARE' (white, black letters) over the one on the right. A Hubbell HBL2611 L5-30P cord cap
lies on a small cradle on the desk, blades toward the plate; its black 10/3 SJOOW cord runs off the cap's back, down
to the desk and to a clamp on the desk to the robot's right. The robot picks up the cap, lines its three curved
blades up with the named receptacle's slots, pushes it in and turns it clockwise to lock it.

All positions are in the desk frame (desk top z = 0, arm bases at (0, +-0.305, 0), +x away from the robot, +y to the
robot's left). No simulator imports.
"""

from __future__ import annotations

from dataclasses import dataclass

EPISODE_SECONDS = 90.0

FACE_X = 0.285
"""x of the receptacle faces (where the cap's face stops when fully in)."""
REC_Z = 0.15
"""Height of both receptacle axes: the band where the YAM can hold its fingers level (push and turn both run about
the wrist's own axis)."""
TARGET_Y = -0.15
DECOY_Y = -0.26
PLATE_Y = (-0.31, -0.085)
PLUG_START_XY = (0.19, -0.03)
"""Centre of the cap's body on its cradle (beside the plate's left edge, between the arms)."""
PLUG_START_ROLL_DEG = 0.0
"""The cap lies with its blades in the entry orientation."""
CLAMP_XY = (0.10, -0.45)
"""Where the cord's far end is clamped to the desk."""

LEGENDS = (("CKT 7", "BATTERY CHARGER"), ("CKT 9", "SPARE"))
"""Nameplates: target, decoy."""
LABEL_COLORS = (((0.72, 0.06, 0.05), (0.97, 0.97, 0.97)), ((0.95, 0.95, 0.93), (0.05, 0.05, 0.05)))
"""(plate, letters) sRGB: target red with white letters, decoy white with black letters (engraved phenolic)."""


def instruction() -> str:
    return (
        "Plug the twist-lock cord cap into the receptacle labelled 'CKT 7 BATTERY CHARGER' (the red nameplate) on "
        "the plate. Push it straight in, turn it clockwise until it stops to lock it, and let go."
    )


@dataclass(frozen=True)
class Layout:
    seed: int = 0

    @property
    def instruction(self) -> str:
        return instruction()


def make_seed(seed: int) -> Layout:
    assert seed == 0, "one fixed layout (seed 0)"
    return Layout(seed)
