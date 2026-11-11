"""On the station table, a plywood board stands 0.45 m in front of the arms with two grey Milbank U3802 pull-out
air-conditioner disconnects (15 x 13 x 8 cm) screwed to it side by side: AC-1 (yellow label) on the arms' midline and
AC-2 (white label) 20 cm to its right. Each has a cover hinged at the top that hangs shut over the pull-out (a black
block with a grip bar; pushed in, its blades sit in the line and load clips), and a tab with a hole out of its bottom
through a slot in the cover. A red Master Lock 410 padlock stands open on a small pedestal in front of the left arm.
Lift AC-1's cover and hold it up, pull its block out, turn it upside down and push it back in (OFF), lower the cover,
and padlock it through the tab. Do not touch AC-2.

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. PULLOUT_CONTROL=1 makes it a control run: the
after-release check starts once the block is OFF, the cover shut and the grippers clear, without a padlock (use it
with ``--policy pullout-demo``, which then skips the padlock): it must score below 1.
"""

from __future__ import annotations

import os

from pullout_sim import station as hb

CONTROL = os.environ.get("PULLOUT_CONTROL") == "1"


def _props():
    from pullout_sim import task as t

    return [t.Disconnect(0), t.Disconnect(1), t.Block(0), t.Block(1), t.Fixtures(), t.Padlock()]


def _task():
    from pullout_sim.task import PulloutTask

    return PulloutTask(default_seed=0, control=CONTROL)


def _test_phase(view, task):
    phase = int(task.state(view).phase[0])
    return "idle" if phase == 0 else ("done" if phase >= 2 else "running")


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task,
    objects=("disconnect_0", "disconnect_1", "block_0", "block_1", "fixtures", "padlock"),
    pose_modules=("pullout_sim.task",), test_phase=_test_phase,
    # Friction anchors per contact point (PhysX's 25 mm default merges a whole clip into one patch); 480 Hz for the
    # stiff compliant clip contacts.
    physics_hz=480.0, physx={"friction_correlation_distance": 0.0005, "friction_offset_threshold": 0.002},
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """The level at the end / 100 (0.25 AC-1's pull-out out of its live clips, 0.5 it is back in upside down (OFF),
    0.75 that with AC-1's cover shut, 1.0 the padlock also shut through the tab and the after-release check passed);
    0 if AC-2's cover was opened or its pull-out moved, or AC-1's pull-out was put back ON."""
    report = STATION.report or {}
    if any(report.get("semantic_events", {}).values()):
        return 0.0
    return report.get("milestone", 0) / 100.0
