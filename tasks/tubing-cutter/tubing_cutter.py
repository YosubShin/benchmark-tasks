"""A 1/2 in copper tube (Type L) stands upright in a stand on the station table, 28 cm in front of the arms on their
midline, a black pen mark round it 7.5 cm above the table. A Klein 50077 screw-feed tubing cutter (orange body, black
knob) sits on the tube at the mark, lying flat, finger-snug, its knob pointing toward the left arm.

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. Scripted demo: ``--policy cutter-demo``;
control: ``--policy cutter-demo -P control=true`` (feeds a full knob turn per revolution; must score 0).
"""

from __future__ import annotations

from tubecutter_sim import station as hb


def _props():
    from tubecutter_sim.task import CutterSet

    return [CutterSet()]


def _task():
    from tubecutter_sim.task import CutterTask

    return CutterTask(default_seed=0)


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task, objects=("cutter_set",), pose_modules=("tubecutter_sim.task",),
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """Milestone / 100: 0.25 a groove all round 0.2 mm deep, 0.5 0.5 mm deep, 0.75 cut through, 1.0 cut through with
    both grippers 5 cm or more from the cutter for 1 s. 0 once the tube was dented (the wheel fed more than 0.7 mm
    ahead of the groove); at most 0.5 once the cut spiralled (axial spread over 1 mm) or left the mark (over 1.5 mm)."""
    report = STATION.report or {}
    if any(report.get("semantic_events", {}).values()):
        return 0.0
    return report.get("milestone", 0) / 100.0
