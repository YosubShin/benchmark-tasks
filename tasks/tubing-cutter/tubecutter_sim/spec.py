"""Tubing cutter: layout, instruction and the numbers of the cutting rule.

Scene (desk frame: table top z = 0, arm bases at (0, +-0.305, 0); x forward, y left): a 1/2 in Type L copper tube
stands upright in a stand (a plywood block with a floor flange), its axis at (TUBE_X, TUBE_Y). A black pen mark
rings it at MARK_Z. A Klein 50077 screw-feed tubing cutter sits on the tube at the mark, lying flat (its frame
horizontal), finger-snug: the wheel and rollers touch the tube, nothing is cut yet. Its knob points toward the left
arm (orbit angle KNOB_START_DEG, measured counterclockwise from +x seen from above).

The cutter's orbit about the tube and its axial slide on the tube are joints; the cut is a rule (spec below and
tubecutter_sim.task): each pass of the wheel over a point of the circumference cuts the groove there to the wheel's
current feed (knob turns x PITCH). Feeding the knob further ahead of the groove than DENT_BITE_MM at any point dents
the tube out of round.
"""

from __future__ import annotations

from dataclasses import dataclass

TASK_NAME = "tubing-cutter"
EPISODE_SECONDS = 200.0

TUBE_X, TUBE_Y = 0.28, 0.0
TUBE_OD = 0.015875
"""1/2 in nominal copper tube, Type L (ASTM B88): OD 0.625 in."""
TUBE_R = TUBE_OD / 2
WALL_MM = 1.016
"""Type L 1/2 in wall, 0.040 in."""
BASE_TOP = 0.022
"""Top of the stand (plywood block 2 cm + flange) the tube stands in."""
TUBE_TOP = 0.115
MARK_Z = 0.075
"""Height of the pen mark (the cut line) above the table."""
KNOB_START_DEG = 90.0
"""The cutter's knob points along +y (toward the left arm) at the start."""

PITCH_MM = 1.814
"""Feed per knob turn: 7/16-14 feed screw (RIDGID screw-feed cutters; Klein does not publish the 50077's; inferred)."""
GUIDE_TURN = 0.25
"""Maker's guidance: about 1/4 knob turn per revolution (Reed TC2Q); Ridge Tool says about 1/16."""
DENT_BITE_MM = 0.70
"""The tube is dented out of round if the wheel is ever fed more than this ahead of the groove under it (about 3/8 of
a knob turn in one go, 1.5x the 1/4-turn guidance; inferred, see README)."""
TRACK_MM = 0.10
"""Once the groove is this deep all the way round, the wheel runs in it and the cutter can no longer slide along the
tube (it tracks)."""
GROOVE_MM = (0.20, 0.50)
"""Partial-credit groove depths (all the way round)."""
SPIRAL_MM = 1.0
"""Cap 0.5 if the wheel's axial position over all cutting passes spreads more than this (the cut walks: a spiral)."""
OFF_MARK_MM = 1.5
"""Cap 0.5 if the groove's centre is more than this from the mark."""
CLEAR_M = 0.05
HOLD_S = 1.0


def instruction() -> str:
    """Goal and constraints only: the feed per revolution and the orbiting are what the task tests."""
    return (
        "Cut the copper tube through at the black mark with the tubing cutter that is on it. The cut must stay on "
        "the mark and the tube end must stay round, not dented. Finish with both grippers away from the cutter."
    )


@dataclass(frozen=True)
class Layout:
    seed: int = 0

    @property
    def instruction(self) -> str:
        return instruction()


def make_seed(seed: int = 0) -> Layout:
    return Layout(seed)
