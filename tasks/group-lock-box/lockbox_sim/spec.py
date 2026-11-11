"""Group lock box: put a job lock's key into a group lock box and add your padlock to the box.

The scene a worker meets in a group lockout. Each machine is locked out with one job padlock; the key of every job
lock goes into a group lock box, and every worker hangs a personal padlock on the box, so no key can come out until
the last worker has removed their lock. The box is a Master Lock S601 (stainless steel, red): a lid hinged at one
end, closed by a latch at the other; along both long sides the lid's flange lies on the body's flange and nine
numbered holes go through both; a key slot in the lid near the latch end lets keys drop in while the box is locked.
A coworker's yellow padlock is already through hole 5 of the front flange.

The robot takes the key of the job lock on the PUMP 1 panel (yellow nameplate) from the hook on that panel, drops
it through the box's key slot, then hangs its own purple Master Lock 410 through a free front hole between the
coworker's lock and the latch end (holes 6 to 9) and snaps it shut.

The success check proves the function: once the robot has let go, the coworker's lock is removed, the latch is
released and the lid's free end is lifted with 20 N: the robot's lock alone must keep it from opening more than
15 degrees, with the key inside.

All dimensions (desk frame: desk top z = 0, x forward from the arm bases, y left of their midline, metres) are here;
no simulator imports. Sources are in README.md.
"""

from __future__ import annotations

from dataclasses import dataclass

TASK_NAME = "group-lock-box"
EPISODE_SECONDS = 180.0
NUM_SEEDS = 1

# Master Lock S601 group lock box (sell sheet 7000-0619: lock box height 144 mm; dimension A 163 mm across the
# flanges, B 128 mm, C 286 mm, D 306 mm with hinge and latch; internal 246 x 124 x 138 mm; 10 mm max shackle;
# 18 numbered holes, 9 per long side). Inferred from the product photos: flange 17.5 mm wide just below a 24 mm lid
# skirt, holes 11 x 14 mm (oval on the real box) 8.75 mm out from the wall, 28 mm apart.
BOX_X = 0.32
"""Centre of the box body (x)."""
BOX_Y = 0.06
"""Centre of the box body (y)."""
RISER_H = 0.055
"""The box stands on a 5.5 cm steel riser block."""
BODY_W = 0.128
"""Body across (x), wall to wall."""
BODY_L = 0.272
"""Body length (y), end wall to end wall."""
BOX_H = 0.144
"""Riser top to the lid's top."""
WALL_T = 0.0015
FLANGE_W = 0.0175
FLANGE_T = 0.0015
FLANGE_DZ = 0.120
"""Height of the body flange's top above the box's base; the lid flange lies on it."""
Z_BASE = RISER_H
Z_BODY_FLANGE = (Z_BASE + FLANGE_DZ - FLANGE_T, Z_BASE + FLANGE_DZ)
Z_LID_FLANGE = (Z_BASE + FLANGE_DZ, Z_BASE + FLANGE_DZ + FLANGE_T)
Z_TOP = Z_BASE + BOX_H
"""World z of the lid's top surface."""
HOLE_X = 0.011
"""Hole size across the flange (x); the real holes are ovals, 10 mm shackles maximum."""
HOLE_Y = 0.014
"""Hole size along the flange (y)."""
HOLE_OUT = 0.00875
"""Hole centre, out from the wall's outer face."""
HOLE_PITCH = 0.028
N_HOLES = 9
COWORKER_HOLE = 5
"""Front holes are numbered 1-9 left to right as the robot sees them (1 at the hinge end)."""
TARGET_HOLES = (6, 7, 8, 9)
"""Free front holes between the coworker's lock and the latch end."""
HINGE_Y = BOX_Y + BODY_L / 2 + 0.004
HINGE_Z = Z_TOP - 0.010
"""The lid's hinge axis (along x) at the left (+y) end."""
LID_MASS = 0.7
KEY_SLOT = (BOX_X, BOX_Y - BODY_L / 2 + 0.025)
"""Centre (x, y) of the gasketed key slot in the lid, near the latch end, running across the lid (along x)."""
KEY_SLOT_SIZE = (0.040, 0.007)
"""Slot length (x) and width (y); inferred."""

# The key panels: two pump starter panels on a board to the right of the box, each with its job lock (a red Master
# Lock 410) and the key of that lock on a hook (a peg) on the panel's right side.
BOARD_X = 0.37
"""Front face of the board."""
BOARD_Y = (-0.45, -0.095)
BOARD_H = 0.22
PANEL_Y = (-0.13, -0.33)
"""PUMP 1 (yellow nameplate, the named panel) and PUMP 2 (white nameplate)."""
PANEL_Z = (0.03, 0.16)
PANEL_W = 0.07
PANEL_D = 0.07
PEG_X = BOARD_X - PANEL_D + 0.008
PEG_Z = 0.12
PEG_LEN = 0.070
PEG_SIDE = 0.004
KEY_OUT = 0.060
"""The key hangs this far out from the panel's side, on the peg."""
KEY_QUAT = (0.0, 0.0, 0.7071068, 0.7071068)
"""Hanging key: its plane is x-z (perpendicular to the peg), thin along y."""

PEDESTAL_XY = (0.19, 0.12)
"""The robot's open padlock stands on a pedestal (shackle up, short leg toward the box)."""
PEDESTAL_TOP = 0.135

PANEL_NAMES = ("PUMP 1", "PUMP 2")
TARGET_PANEL = 0


def hole_xy(k: int, front: bool = True) -> tuple[float, float]:
    """Centre of hole k (1-9) of the front (robot side) or back flange."""
    x = BOX_X - BODY_W / 2 - HOLE_OUT if front else BOX_X + BODY_W / 2 + HOLE_OUT
    y = BOX_Y + (5 - k) * HOLE_PITCH if front else BOX_Y - (5 - k) * HOLE_PITCH
    return x, y


def key_hang_pos(panel: int) -> tuple[float, float, float]:
    """The key's origin (centre of its bow hole) where it hangs on its peg."""
    return (PEG_X, PANEL_Y[panel] - PANEL_W / 2 - KEY_OUT, PEG_Z + PEG_SIDE / 2 - KEY_HOLE / 2 + 0.0002)


# Key of a Master Lock 410 (nickel-plated brass): 52 mm long, bow 22 mm wide, 2.3 mm thick, 7 g (inferred).
KEY_T = 0.0023
KEY_BOW_W = 0.022
KEY_BOW = (0.008, -0.012)
"""Bow top and bottom (z) in the key frame (origin at the bow hole's centre)."""
KEY_HOLE = 0.006
KEY_BLADE_W = 0.008
KEY_TIP_Z = -0.044
KEY_MASS = 0.007


def instruction() -> str:
    return (
        "Take the key hanging beside the PUMP 1 panel (yellow nameplate) and drop it into the red lock box through "
        "the key slot in its lid. Then hang your purple padlock through a free hole of the box's front flange to "
        "the right of the yellow padlock, and snap it shut. Do not move the yellow padlock or the other key."
    )


@dataclass(frozen=True)
class LockBoxSeed:
    seed: int

    @property
    def instruction(self) -> str:
        return instruction()


def make_seed(seed: int) -> LockBoxSeed:
    assert 0 <= seed < 20, "seed must be in [0, 20)"
    return LockBoxSeed(seed)
