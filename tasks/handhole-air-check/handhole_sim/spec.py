"""Sample a handhole's air through its cover's pick slot; take the cover bolts out only if the monitor reads clean.

Real parts (README References): a Hubbell Quazite PG1118 handhole (11 x 18 in polymer-concrete box, cover bolted
down by two 3/8-7 hex bolts at diagonal corners, a 1/2 x 2 in pull slot with a 3/16 in centre pin) and a Honeywell
BW GasAlertMax XT II pumped 4-gas monitor with its 1/4 in Tygon sample hose and an inline filter.

The real job (Newfoundland & Labrador "Manhole Cover Removal" SOP: "Insert the aspirator tube through the cover
access hole and test for a minimum of 60 seconds"; OSHA 1910.269(e)(4): evaluate the atmosphere before an entrance
cover is removed): the pumped monitor's hose goes in through the pick slot of the closed cover and draws the box's
own air. Only if the readings are inside the alarm limits are the bolts taken out. On an alarm the cover stays shut.

The monitor shows room air until the hose's open end has been inside the box for RESPONSE_S (about 1 s per foot of
hose plus the sensor's T90), then the box's air, set by the seed. The sample counts after SAMPLE_S (the SOP's 60 s).

Seeds: 0 (even) = the box's air is clean (GO: the monitor keeps reading 20.9 % O2 and zeros, as a real monitor does
in a box of room air); 1 (odd) = the box's air holds 18 ppm H2S (NO-GO: above the 10 ppm low alarm; the monitor's red
alarm bars flash and the display shows the H2S reading; the alarm latches, as with the monitor's Latching Alarms
option, so it stays on after the hose is out). No simulator imports.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

EPISODE_SECONDS = 240.0
SAMPLE_S = float(os.environ.get("HANDHOLE_SAMPLE_S", 60.0))
"""Minimum time the hose end must stay in the box's air (NL SOP: 60 s). The env override is for iterating only."""
RESPONSE_S = 15.0
"""When the reading starts to show the box's air: 10 ft of hose at about 1 s/ft (City of North Vancouver procedure,
MSA ALTAIR 5X manual p.10) plus the Max XT II's 4 s T90 (manual p.87), rounded up."""
WATCH_S = 10.0
"""NO-GO: how long after the hose is out the cover must stay bolted before the final lift check."""
ALARM_BLINK_S = 0.25
"""The alarm bars flash: on and off for this long each."""

INSTRUCTION = (
    "Pick up the white filter on the yellow monitor's clear tube and lower the end of the tube through the slot in "
    "the tan cover, so the end hangs inside the box. Keep it there for at least 60 seconds, then lift the tube out "
    "and lay it down. Then look at the yellow monitor. If its red lights are flashing, leave the cover and both of "
    "its bolts alone. If they are not flashing, unscrew the two silver bolts at the cover's corners by turning them "
    "counter-clockwise, and lift both bolts out of the cover."
)


@dataclass(frozen=True)
class HandholeSeed:
    seed: int
    atmosphere: str  # "go" (clean) or "nogo" (H2S 18 ppm, above the 10 ppm low alarm)

    @property
    def go(self) -> bool:
        return self.atmosphere == "go"

    @property
    def instruction(self) -> str:
        return INSTRUCTION


def make_seed(seed: int) -> HandholeSeed:
    assert seed >= 0, "seed must be >= 0"
    return HandholeSeed(seed, "go" if seed % 2 == 0 else "nogo")
