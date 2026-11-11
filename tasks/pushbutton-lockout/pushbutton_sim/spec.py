"""Lock out a machine's START push button with a push-button lockout cover: layout and instruction. No simulator
imports.

The scene: a light-grey steel push-button station lies on the table in front of the robot, its two 22 mm flush
push buttons facing up (Schneider Harmony XB4 style, chrome bezels): a green START (the target) nearer the robot and a
red STOP 7 cm behind it, each with a black legend plate to its right. Each button has a clear Master Lock S2152AST
install base captured under its bezel. A clear Master Lock S2151 removable push-button cover (two half-shells on a
vertical hinge, each with a padlock ear at the free edge) stands open on a grey block to the robot's right. A blue
Master Lock 410 padlock, shackle open, lies on a pedestal to the left.

The robot sets the open cover around the START button so its bottom lip goes under the base's rim, closes it, and
padlocks the two ears together.

All positions are in the desk frame (desk top z = 0, arm bases at (0, +-0.305, 0), +x away from the robot, +y to the
robot's left).
"""

from __future__ import annotations

from dataclasses import dataclass

EPISODE_SECONDS = 180.0

PANEL_Z = 0.095
"""Top face of the push-button station (the panel the buttons stand on)."""
BUTTON_XY = {"start": (0.19, -0.075), "stop": (0.26, -0.075)}
"""Centres of the two push buttons (and of their install bases): 70 mm apart along x."""
STATION_X = (0.12, 0.31)
STATION_Y = (-0.15, -0.005)

COVER_START_XY = (0.22, -0.225)
"""Where the open cover stands at the start (its axis, as if closed), on the block to the right."""
BLOCK_TOP = 0.05
COVER_OPEN_DEG = 30.0
"""The cover starts open this far (front half turned away from the back half about the hinge)."""

PEDESTAL_XY = (0.19, 0.07)
"""The open padlock lies here, shackle toward -y (toward the cover's ears)."""

LEGENDS = {"start": "START", "stop": "STOP"}


def instruction() -> str:
    return (
        "Lock out the green START button with the clear push-button cover: take the open cover from the block on "
        "the right, set it around the START button so its bottom edge goes under the clear base ring, close the "
        "cover, then put the blue padlock's shackle through both of the cover's ears and snap it shut. Do not "
        "press either button, and do not cover the red STOP button."
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
