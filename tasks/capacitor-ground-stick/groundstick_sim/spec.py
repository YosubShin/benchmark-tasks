"""Capacitor ground stick: layout and instruction, no simulator imports.

Scene (desk frame: table top z = 0, arm bases at (0, +-0.305, 0)): two oil-filled capacitors (Chicago Condenser
CQ72-class cans, two porcelain bushings on top) labelled C1 (blue labels) and C2 (orange labels), C1 behind C2; a
copper earth bar on two insulators, standing on edge; a 1 ft discharge stick (Salisbury 22629 class, universal head
with a bent-rod probe) lying in a cradle, whose yellow ground lead ends in a Mueller BU-21C copper clip clipped to an
insulating parking tab; a WAIT lamp (amber, lit for WAIT_SECONDS from the start, then dark).

The order (EFCOG BP261 6.4-6.5, Fig. 5 and 9; OSHA 1910.269(w)(1)): clip the lead to ground first, wait out the
discharge time, lay the probe across both terminals of the named capacitor, then leave the stick hanging on them.
"""

from __future__ import annotations

from dataclasses import dataclass

TASK_NAME = "capacitor-ground-stick"
EPISODE_SECONDS = 160.0

WAIT_SECONDS = 20.0
"""How long the WAIT lamp stays lit from the start (the real bench's timer setting; stands in for the 1-5 min wait)."""
DWELL_SECONDS = 1.5
"""Probe held across both terminals this long, after the lamp and with the clip on the bar, counts as discharged."""

CAP_X = (0.44, 0.33)
"""Desk-frame x of the two cases' centres: C1 (the decoy) behind, C2 nearer the robot."""
CAP_Y = -0.03
CAP_LABELS = ("C1", "C2")
CAP_COLORS = ("blue", "orange")
"""Colour of each can's ID labels (front face and lid)."""
EARTH_BAR_Y = -0.22
"""Desk-frame y of the earth bar's mid-plane (the bar stands on edge, faces normal to y)."""
STICK_CRADLE_XY = (0.12, 0.27)
"""Where the stick lies at the start (grip centre), axis along -y."""
CLIP_TAB_XY = (0.17, -0.17)
"""The insulating parking tab the ground clip starts clipped to (x, mid-plane y)."""
NAMED = 1
"""Index into CAP_LABELS of the capacitor to discharge (C2)."""


def instruction() -> str:
    return (
        "First clip the copper clip at the end of the stick's yellow lead onto the copper earth bar on the right "
        "(squeeze its handles to open it). Then wait until the amber WAIT lamp goes out. After that, pick up the "
        "yellow stick by its black handgrip, lay its copper hook across both terminal studs on top of capacitor C2 "
        "(the can with the orange labels), and let go so the stick stays hanging on the studs. Do not touch "
        "capacitor C1 (blue labels), and keep the fingers off the studs."
    )


@dataclass(frozen=True)
class Layout:
    seed: int = 0
    named: int = NAMED

    @property
    def named_label(self) -> str:
        return CAP_LABELS[self.named]

    @property
    def instruction(self) -> str:
        return instruction()


def make_seed(seed: int = 0) -> Layout:
    return Layout(seed)
