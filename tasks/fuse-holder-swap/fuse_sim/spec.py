"""Swap the fuse in the named flip-open DIN-rail fuse holder: layout and instruction.

The scene: a DIN rail on a back pan facing the robot (-x) carries three Eaton Bussmann CHM1DU one-pole modular
fuse holders for 10 x 38 mm midget fuses (F1, F2, F3), one blank 17.5 mm module between neighbours, each with its
circuit legend (tag, circuit, fuse rating, and a coloured rating field) on the pan above it. Each holder's fuse
carrier hinges at its lower end and swings forward and down about 72 deg when its top grip tab is pulled. F2 holds
the blown fuse, marked with a black X. Two spare fuses lie on a spares stand to the right (a Bussmann KTK-6 and a
KTK-10, the rating printed on the body), each cradle carrying a coloured rating label; a parts tray sits on the desk.

Success (the function): the named carrier latched shut and holding against an opening pull, with the spare of the
matching rating in it, the blown fuse in the tray, and neither other holder opened. No simulator imports.
"""

from __future__ import annotations

from dataclasses import dataclass

EPISODE_SECONDS = 180.0

FRONT_X = 0.270
"""Desk-frame x of the holders' carrier front faces (closed)."""
BASE_Z = 0.110
"""Desk-frame z of the holders' bottom faces (grasp heights 0.13-0.17 m, where the YAM can hold its fingers level)."""
MODULE = 0.0175
"""CHM module width (published, Bussmann TD 720147)."""
PITCH = 2 * MODULE
"""One blank module between holders: ganged at 17.5 mm, the YAM's fingers (15 mm wide at the tips, 35-40 mm 1-2 cm
back) hit the neighbours' grip tabs in every grasp of the open fuse (a 2D check of the finger outline)."""
SLOT_Y = (PITCH, 0.0, -PITCH)
"""Desk-frame y of F1, F2, F3 (F1 on the robot's left)."""
SPARE_Y = (-0.090, -0.130)
"""y of the two spares on the stand (axis along x)."""
SPARE_X = (0.232, 0.270)
"""x of a spare's near (robot-side) and far end."""
SPARE_AXIS_Z = 0.145
TRAY_XY = (0.205, -0.205)

CIRCUITS = (("F1", "PLC 24V DC", 2), ("F2", "CONTROL XFMR", 6), ("F3", "CABINET LIGHT", 10))
NAMED = 1
"""Index of the named holder (F2)."""
SPARE_RATINGS = (10, 6)
"""Ratings of the spares at SPARE_Y[0] and SPARE_Y[1] (the matching 6 A spare is the far one)."""
RATING_COLOURS = {2: ("pink", (0.95, 0.55, 0.72)), 6: ("green", (0.10, 0.62, 0.22)), 10: ("red", (0.85, 0.08, 0.08))}
"""Rating colour code on the legends and the spares' cradle labels (the IEC 60269-3 D-type fuse colours: 2 A pink,
6 A green, 10 A red), so the rating can be matched where the printed text is too small to read."""


def instruction(named: int = NAMED) -> str:
    tag, circuit, amps = CIRCUITS[named]
    others = " or ".join(c[0] for k, c in enumerate(CIRCUITS) if k != named)
    return (f"The panel is de-energized. Open fuse holder {tag} ({circuit}), take out its blown fuse (marked with a "
            f"black X) and drop it in the blue parts tray, put in the spare fuse whose rating matches the {tag} legend (legends "
            f"and spare labels are colour-coded by rating), and close the holder. Do not open {others}.")


@dataclass(frozen=True)
class Layout:
    seed: int
    named: int = NAMED
    spare_ratings: tuple = SPARE_RATINGS

    @property
    def instruction(self) -> str:
        return instruction(self.named)

    @property
    def matching_spare(self) -> int:
        return self.spare_ratings.index(CIRCUITS[self.named][2])


def make_seed(seed: int = 0) -> Layout:
    return Layout(seed)
