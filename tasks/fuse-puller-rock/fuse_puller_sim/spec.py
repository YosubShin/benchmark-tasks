"""Pull the named midget fuse from its open-clip block with a tong puller: layout and instruction.

The scene: a small wall-mount control enclosure (Rittal AE 1036.500, 300 x 300 x 155 mm, steel) lies on its back on
the bench, its opening up, its bottom (gland plate) toward the robot and its door swung open to the left. Three Eaton
Bussmann BM6031B one-pole midget fuse blocks are snapped side by side on a 35 mm DIN rail on its mounting plate, fuse
axes pointing away from the robot (up the panel), each holding a 10 x 38 mm midget fuse in two open spring clips.
Each block is wired: a black 14 AWG line wire from its far (line) box lug and a red load wire from its near (load) lug,
each running into a slotted wire duct (one beyond the blocks, one between the blocks and the robot). A circuit label
on the far duct's cover above each block names it (F1, F2, F3 with the circuit name, colour-coded: F1 white, F2
orange, F3 green). A one-piece nylon tong fuse puller (Ideal 34-015 class) stands upright, jaws down, in a grey holder
on the bench right of the enclosure; a blue parts tray sits on the bench further right, nearer the robot.

The clips hold a fuse by spring force and friction at its two ferrules. Pulled straight out (both ferrules at once)
a fuse needs both clips' release force together; rocked out one end first (the puller held perpendicular to the
fuse, its jaws across the body, tipped about the far clip as a lever) it comes out one clip at a time.

Success (the function): the named fuse is out of both clips, carried in the puller's jaws, and lies in the tray,
with neither other fuse disturbed. No simulator imports.
"""

from __future__ import annotations

from dataclasses import dataclass

EPISODE_SECONDS = 150.0

PAN_TOP = 0.028
"""Desk-frame z of the DIN rail's top (the blocks' mounting face): the enclosure's 1.4 mm back on the bench, its 2 mm
mounting plate 17 mm above it (top at 20.5 mm), the 7.5 mm rail on that."""
ENC_X = (0.170, 0.470)
"""The enclosure's outside, front (bottom end, toward the robot) to back (top end)."""
ENC_Y = (-0.110, 0.190)
"""The enclosure's outside, right (robot's right) to left (hinge side)."""
WALL_TOP = 0.155
"""Its walls' rim above the table (the enclosure's 155 mm depth, lying on its back)."""
DOOR_OPEN_DEG = 100.0
"""The door, hinged along the left wall's rim, swung open this far (just past upright)."""
DUCT_TOP = 0.0715
"""Top of the wire ducts' covers (51 mm tall ducts on the mounting plate)."""
DUCT_NEAR_X = (0.190, 0.228)
DUCT_FAR_X = (0.415, 0.453)
"""The two ducts' extent along x (38 mm wide): one between the blocks and the robot, one beyond the blocks."""
BLOCK_X = 0.320
"""Desk-frame x of the blocks' centres (the fuse centres)."""
SLOT_Y = (0.0225, 0.0, -0.0225)
"""Desk-frame y of the fuses F1, F2, F3 (F1 on the robot's left). The 21.0 mm blocks stand 1.5 mm apart."""
PULLER_HOLDER_XY = (0.310, -0.175)
"""Centre of the puller's holder on the bench right of the enclosure (the puller stands upright in it, jaws down)."""
TRAY_XY = (0.200, -0.200)
"""Centre of the parts tray on the bench right of the enclosure, nearer the robot."""
TRAY_IN = (0.10, 0.10, 0.025)
"""Inside length, width and wall height of the tray."""

CIRCUITS = (("F1", "PLC 24V DC", "white"), ("F2", "CONTROL XFMR", "orange"), ("F3", "CABINET LIGHT", "green"))
NAMED = 1
"""Index of the named fuse (F2)."""
LABEL_RGB = {"white": (0.96, 0.96, 0.94), "orange": (0.97, 0.45, 0.05), "green": (0.10, 0.60, 0.22)}


def instruction(named: int = NAMED) -> str:
    tag, circuit, colour = CIRCUITS[named]
    others = " and ".join(c[0] for k, c in enumerate(CIRCUITS) if k != named)
    return (f"The control panel is de-energized and its door is open. Pick up the yellow fuse puller from its holder "
            f"beside the enclosure, reach into the enclosure and grip fuse {tag} ({circuit}, the fuse with the "
            f"{colour} label) across its body in the puller's jaws, holding the puller perpendicular to the fuse, tip "
            f"the puller's handles away from you to lever the near end of the fuse out of its clip first, then lift "
            f"out the far end, and drop the fuse in the blue parts tray beside the enclosure. Do not disturb fuses "
            f"{others} or the wiring.")


@dataclass(frozen=True)
class Layout:
    seed: int
    named: int = NAMED

    @property
    def instruction(self) -> str:
        return instruction(self.named)


def make_seed(seed: int = 0) -> Layout:
    return Layout(seed)
