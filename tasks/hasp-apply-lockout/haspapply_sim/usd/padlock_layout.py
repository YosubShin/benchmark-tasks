"""Padlock through a hasp (geometry shared with the group hasp): instruction and published seeds, with no simulator imports.

Two disconnect-switch boxes stand side by side on a panel, each with a colour tag and a hasp: a
horizontal plate sticking out toward the robot with a hole of 8-10 mm near its free end. One switch is
already locked (a closed padlock hangs from its hasp, with a red danger tag); the instruction names
the other by colour. An open lockout padlock stands on a pedestal between the two switches.
The padlock and the lock points follow real parts, in vendor variants that change between seeds
(PADLOCK_NAMES, LOCK_POINT_NAMES; dimensions in haspapply_sim/usd/padlock_props.py).
Arm A picks it up and threads the shackle's short leg down through the named hasp's slot; arm B
(the right arm) pushes the shackle shut until it snaps. The trial is held once the robot lets go and
the lock survives a 10 N downward tug.
"""

from __future__ import annotations

from dataclasses import dataclass

TASK_NAME = "padlock-hasp"
EPISODE_SECONDS = 90.0
NUM_SEEDS = 20
TAG_COLORS = ("blue", "yellow")
PANEL_FACE_X = 0.31
"""World x of the switch boxes' front faces (the hasps stick out toward the robot, along -x)."""
SWITCH_Y = (-0.12, 0.03)
"""World y of the two switch positions (slot 0 past the midline, slot 1 toward arm A)."""
HASP_Z = 0.24
"""World z of the hasp tabs' mid-plane."""
PADLOCK_NAMES = ("ml410", "a1105", "abus7440")
"""Master Lock 410 Zenex, American Lock A1105 (short shackle), ABUS 74/40 (thicker shackle)."""
LOCK_POINT_NAMES = ("handle_tab", "hasp420")
"""A switch handle's lockout tab (10.3 mm hole) and a Master Lock 420 hasp (8 mm hole)."""
PEDESTAL_XY = (0.27, -0.045)
"""The open padlock's pedestal, midway between the two hasps, so carrying it to either one never
passes the pre-applied padlock.
Arm A grips the padlock's body level from behind, which the YAM can hold only at x >= 0.2 and grasp
height about 0.13-0.17; the panel, pedestal and hasp height keep the
grasp there."""


def instruction(color: str) -> str:
    return (
        f"Lock out the switch with the {color} tag: put the padlock's shackle through its hasp and snap it shut. "
        "Do not touch the switch that is already locked."
    )


@dataclass(frozen=True)
class PadlockSeed:
    seed: int
    named_color: str
    named_slot: int
    """The named switch's position; the other switch carries the pre-applied lock."""
    padlock: str
    lock_point: str
    """Both switches carry the same kind of lock point."""

    @property
    def slot_colors(self) -> tuple[str, str]:
        other = TAG_COLORS[1 - TAG_COLORS.index(self.named_color)]
        return (self.named_color, other) if self.named_slot == 0 else (other, self.named_color)

    @property
    def instruction(self) -> str:
        return instruction(self.named_color)


def make_seed(seed: int) -> PadlockSeed:
    """Named colour and named slot each split 10/10 across the 20 seeds, in all four combinations. The
    padlock cycles through its three variants (7/7/6) and the lock points split 10/10, both crossed with
    colour and slot.

    The padlock always starts on the one pedestal (a second spot would need a second pedestal in the
    scene, and any spot near the first blocks the fingers' way to it).
    """
    assert 0 <= seed < NUM_SEEDS, f"seed must be in [0, {NUM_SEEDS})"
    slot = (seed // 2) % 2 if seed < 16 else (seed // 2 + 1) % 2
    return PadlockSeed(seed, TAG_COLORS[seed % 2], slot, PADLOCK_NAMES[seed % 3], LOCK_POINT_NAMES[(seed // 2 + seed // 4) % 2])
