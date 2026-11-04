"""Prove a stick-mounted detector on its proof tester, test the target bar, prove again; stop if it does not prove.

Real parts (README References): an HD Electric PRX proximity voltage detector (280 x 76 x 50 mm, 0.31 kg) on the
universal-spline head of a short fiberglass stick; its PT-PRX Proof Tester (125 x 66 x 33 mm, 199 g, 9 V battery: a
red TEST button on its face, a metal end plate at the end marked "Touch PRX to this end"); and a bare 1/0 aluminium
conductor stub on two polymer posts, connected to nothing.

The procedure (PRX and PT-PRX manuals): press ON (self-test: the range lights step through, then the indicating
lights and beeper fire; the unit stays on if it passed). Apply the PRX's SENSING ZONE end to the tester's metal end
plate and press and hold TEST: a working PRX flashes and beeps continuously. If it does not, do not use it. Then
approach the target perpendicular to it until the SENSING ZONE end contacts it, and prove on the tester again.

Desk frame (desk top z = 0, arm bases at (0, +-0.305, 0)): the detector on its stick lies across the front of the
desk in a cradle, face up, the sensing end toward arm B (-y), the stick toward arm A (+y). Arm A holds the stick from
above (fingers down, closing across it) at its balance point. The PT-PRX lies face up in a cradle in front of arm B,
its end plate facing the detector; arm B presses its TEST button (and, at the start, the PRX's ON button) from above.
The conductor stub runs along x (perpendicular to the detector), beside the tester.

Seeds: 0 (even) = everything works; 1 (odd) = the proof tester has a flat battery, so the PRX passes its own
self-test but does not indicate on the tester and the tester's LED stays dark (the correct response is to stop).
No simulator imports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

EPISODE_SECONDS = 120.0

STICK_YAW_DEG = -90.0
"""Direction the held detector points, measured from +x toward +y (negative: toward arm B)."""
TOOL_Z = 0.15
"""Height of the stick's axis while working (the YAM's fingers-down grasp tracks well at 0.13-0.17 m)."""
CRADLE_Z = 0.13
"""Height of the stick's axis in its start cradle (below TOOL_Z so the lifted tool clears the cradle)."""
GRASP_PROVE = (0.22, 0.12)
"""Desk (x, y) of arm A's grasp point while the detector touches the proof tester's end plate."""
TEST_SHIFT = 0.13
"""The target's contact point is this far from the end plate's centre, sideways to the stick (along +n)."""
CRADLE_BACK = 0.03
"""The detector starts this far back along the stick from its proving pose (tip short of the end plate)."""

INSTRUCTION = (
    "Check the red detector on the yellow stick on the black tester before and after you use it on the metal bar. "
    "Press the white ON button on the detector's face and wait about 3 seconds while its lights step through. "
    "Pick the detector up by its stick. Hold down the red TEST button on the black tester and touch the detector's "
    "top end to the tester's metal end until the detector's red lights flash. Then touch the detector's top end to "
    "the side of the bare metal bar on the two grey posts. Then do the tester check again. If the detector's lights "
    "do not flash on the tester, move it away from the tester and stop: do not bring it to the bar."
)


def stick_dir() -> tuple[float, float]:
    a = math.radians(STICK_YAW_DEG)
    return (math.cos(a), math.sin(a))


def side_dir() -> tuple[float, float]:
    """Unit vector across the stick, 90 deg counter-clockwise from it."""
    d = stick_dir()
    return (-d[1], d[0])


@dataclass(frozen=True)
class DetectorSeed:
    seed: int
    prove_fails: bool
    """The tester's battery is flat: TEST gives no output and its LED stays dark, so the PRX never indicates on it."""

    @property
    def instruction(self) -> str:
        return INSTRUCTION


def make_seed(seed: int) -> DetectorSeed:
    assert seed >= 0, "seed must be >= 0"
    return DetectorSeed(seed, prove_fails=(seed % 2 == 1))
