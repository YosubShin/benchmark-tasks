"""The documented setup layout (task.json ``setup``) as numbers the scripted demo may use. Pure Python, no simulator.

These are the design positions and part sizes the props are built from (``usd/detector_props.py``), in the desk
frame (desk top z = 0, x forward from the line joining the arm bases, y to the left, arm bases at y = +-0.305). They
are what a person setting the task up measures with a tape; the demo uses them as nominal waypoints and corrects
them with what its cameras see. Nothing here is read from the running simulation.
"""

from __future__ import annotations

import math

import numpy as np

from detector_sim.kit import q_axis, qmul
from detector_sim.spec import STICK_YAW_DEG
from detector_sim.usd import detector_props as props

D = np.array(props.D3())
"""Direction the detector's top (sensing) end points: toward the right arm (-y)."""
N = np.array(props.N3v())
"""Across the detector, forward (+x)."""

TOOL_QUAT = qmul(q_axis((0, 0, 1), math.radians(STICK_YAW_DEG)), q_axis((0, 1, 0), -math.pi / 2))
"""The detector lying level and face up, its top end along D (detector frame: origin at the top end's centre, +z
along the stick toward its butt, +x the face normal)."""

TIP_START = np.array(props.tip_start())
"""The detector's top end in its cradle (setup: 23 cm right of the midline, 22 cm forward, 13 cm up)."""
PLATE = np.array(props.plate_point())
"""Centre of the tester's metal end plate (26 cm right of the midline, 22 cm forward, 15 cm up)."""
TEST_TOP = np.array(props.test_button_top())
"""Top of the tester's TEST button (31.7 cm right of the midline)."""
BAR_CONTACT = np.array(props.contact_point())
"""Middle of the bar's left face between its posts (26 cm right of the midline, 35 cm forward, 15 cm up)."""
BAR_SPAN = props.conductor_span()
"""Extent of the bar along +x from BAR_CONTACT (it runs from 29 to 47 cm forward)."""
BAR_SIDE = props.COND_SIDE
TESTER_SIZE = props.PT
TEST_FROM_PLATE = props.TEST_FROM_PLATE
LED_FROM_PLATE = props.LED_FROM_PLATE

HEAD, BODY = props.HEAD, props.BODY
"""The detector's sensing head and body boxes (depth = face normal, width, length along the stick)."""
DET_LEN = props.DET_LEN
GRASP_FROM_TIP = props.GRASP_Z
"""Where the stick is held: 38 cm behind the top end (4.5 cm behind the spline head, its balance point)."""
ON_BUTTON = props.ON_BUTTON
"""ON/OFF button on the face: distance from the top end, offset across the face, size."""
INDICATOR_FROM_TIP = props.INDICATOR_Z
INDICATOR_ACROSS = 0.015
"""The two red indicating lights sit 1.5 cm either side of the face's centre line."""
RANGE_LIGHTS_FROM_TIP = props.RANGE_LIGHT_Z
RANGE_LIGHT_ACROSS = 0.022
TAB = props.TAB
LABEL = (0.100, 0.230, BODY[1] - 0.012)
"""The black label on the face: from / to (distance from the top end) and width."""
LENS = (INDICATOR_FROM_TIP - 0.008, INDICATOR_FROM_TIP + 0.008, 0.050)
CRADLE_Z = props.CRADLE_Z
TOOL_Z = props.TOOL_Z
