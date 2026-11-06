"""Withdraw the named drawout relay's plug-in unit: layout, labels and instruction (no simulator imports).

Two ABB REF615-class relays are flush-mounted side by side in a panel standing on the table. Each plug-in unit is
latched into its case by its bail handle: with the handle down, a lug on the handle's hub sits behind a fixed catch
and the unit cannot be pulled out. Lifting the handle to 90 degrees swings the lug clear and drives a tooth, as a
cam, against the case, which pushes the unit about 7 mm out and parts the rear connectors (ABB 615 installation
manual 1MRS756375, 4.2.1). The unit then slides out with no stop. The robot lifts the named unit's handle, pulls the
unit out by the handle with one arm while the other arm's fingers go under the body and take its weight as it
leaves the case, sets it down on the bench mat in front of the panel and lets go.

Final check (after release, off camera): the unit rests upright on the mat, wholly out of its case, and the other
relay's unit is still latched (the task pulls on it with 30 N and it does not come out). One fixed layout.

Frame: the task's desk frame (table top z = 0, arm bases at (0, +-0.305, 0)); relay_sim.station maps it into the
station. Dimensions and sources: relay_sim/usd/relay_props.py.
"""

from __future__ import annotations

EPISODE_SECONDS = 120.0

PANEL_X = 0.38
"""x of the panel's front face (the units come out toward the robot, along -x)."""
BODY_Z = 0.065
"""Height of the case floor's top, where the unit's body rests."""
SLOT_Y = (0.08, 0.30)
"""y of the two relays' centres: slot 0 (the named one) and slot 1 (the other relay, beyond it on the left arm's
side, so the right arm comes under the named unit from -y without passing the other relay's handle)."""
NAMED_SLOT = 0

LABELS = (("FEEDER 2", "52-F2  REF615"), ("FEEDER 1", "52-F1  REF615"))
"""Engraved designation labels (line 1, line 2) below each relay, slot 0 first."""
MARKER_NAMES = ("yellow", "blue")
"""Colour marker on each relay's label, slot 0 first (named in the instruction)."""
MARKER_RGB = ((0.98, 0.80, 0.05), (0.08, 0.30, 0.90))

UNIT_MASS_KG = 2.1
"""ABB 615 installation manual 1MRS756375 G, table 5: plug-in unit only 2.1 kg (the ANSI product guide says 1.8 kg
for the draw-out unit; complete relay 3.5-4.1 kg)."""


def instruction() -> str:
    name, marker = LABELS[NAMED_SLOT][0], MARKER_NAMES[NAMED_SLOT]
    return (
        f"Take out the plug-in unit of the relay labelled {name} (the label with the {marker} marker): lift its "
        "handle up to release the latch, pull the unit out of its case by the handle with one hand while the other "
        "hand goes under the body to carry its weight, and set it down on the black mat in front of it. Do not touch "
        "the other relay, and do not reach into the empty case."
    )
