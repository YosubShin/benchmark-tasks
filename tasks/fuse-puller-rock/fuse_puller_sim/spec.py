"""Pull the named midget fuse from its open-clip block with a tong puller: layout and instruction.

The scene: a control sub-panel's back pan lies flat on the bench (as on a panel-shop or repair bench). Three Eaton
Bussmann BM6031B one-pole midget fuse blocks are screwed to it side by side, fuse axes pointing away from the robot,
each holding a 10 x 38 mm midget fuse in two open spring clips. Each block has a circuit label on the pan in front
of it (F1, F2, F3 with the circuit name, on a colour-coded label: F1 white, F2 orange, F3 green). A one-piece nylon
tong fuse puller (Ideal 34-015 class) lies on a grey rest to the right, handle ends toward the robot; a blue parts
tray sits further right.

The clips hold a fuse by spring force and friction at its two ferrules. Pulled straight out (both ferrules at once)
a fuse needs both clips' release force together, more than the robot can pull through the puller; rocked out one
end first (the far clip as the pivot, the puller as a lever) it comes out one clip at a time.

Success (the function): the named fuse is out of both clips, carried in the puller's jaws, and lies in the tray,
with neither other fuse disturbed. No simulator imports.
"""

from __future__ import annotations

from dataclasses import dataclass

EPISODE_SECONDS = 150.0

PAN_TOP = 0.028
"""Desk-frame z of the back pan's top face (a 3 mm steel plate lying on a 25 mm plywood board on the bench, so the
puller's handle ends and the hand clear the bench while the puller is tilted)."""
BLOCK_X = 0.320
"""Desk-frame x of the blocks' centres (the fuse centres)."""
SLOT_Y = (0.0225, 0.0, -0.0225)
"""Desk-frame y of the fuses F1, F2, F3 (F1 on the robot's left). The 21.0 mm blocks stand 1.5 mm apart."""
PULLER_REST_Y = -0.105
"""y of the puller on its rest (its axis along x, handle ends toward the robot)."""
TRAY_XY = (0.215, -0.190)
"""Centre of the parts tray."""
TRAY_IN = (0.10, 0.10, 0.025)
"""Inside length, width and wall height of the tray."""

CIRCUITS = (("F1", "PLC 24V DC", "white"), ("F2", "CONTROL XFMR", "orange"), ("F3", "CABINET LIGHT", "green"))
NAMED = 1
"""Index of the named fuse (F2)."""
LABEL_RGB = {"white": (0.96, 0.96, 0.94), "orange": (0.97, 0.45, 0.05), "green": (0.10, 0.60, 0.22)}


def instruction(named: int = NAMED) -> str:
    tag, circuit, colour = CIRCUITS[named]
    others = " and ".join(c[0] for k, c in enumerate(CIRCUITS) if k != named)
    return (f"The panel is de-energized. Pick up the yellow fuse puller, grip fuse {tag} ({circuit}, the fuse with "
            f"the {colour} label) in its jaws, lift the puller's handles to lever the near end of the fuse out of its "
            f"clip first, then the far end, and drop the fuse in the blue parts tray. Do not disturb fuses {others}.")


@dataclass(frozen=True)
class Layout:
    seed: int
    named: int = NAMED

    @property
    def instruction(self) -> str:
        return instruction(self.named)


def make_seed(seed: int = 0) -> Layout:
    return Layout(seed)
