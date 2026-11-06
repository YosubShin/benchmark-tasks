"""A grey steel plate stands upright on the table in front of the robot, right of its midline. Two identical black
NEMA L5-30R twist-lock receptacles (Hubbell HBL2616) are mounted through it side by side, facing the robot, each with
an engraved nameplate above it: 'CKT 7 BATTERY CHARGER' (red nameplate, white letters) over the left one and
'CKT 9 SPARE' (white nameplate, black letters) over the right one. A Hubbell HBL2611 L5-30P twist-lock cord cap lies
on a small grey cradle on the table, blades toward the plate; its black rubber cord runs off the cap's back, down to
the table and to a clamp on the table to the robot's right. Plug the cap into the CKT 7 receptacle, turn it clockwise
until it stops to lock it, and let go. The cap must never be brought to the CKT 9 receptacle, and must not be forced
in crooked.

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. The control run is ``--policy twistlock-demo
-P control=true``: the demo pushes the cap home but does not turn it, lets go, and the pull test is forced (it must
score below 1).
"""

from __future__ import annotations

from twistlock_sim import station as hb


def _props():
    from twistlock_sim import task as tt

    return [tt.Panel(), tt.CordCap(), tt.Cradle()]


def _task():
    from twistlock_sim.task import TwistLockTask

    return TwistLockTask(default_seed=0)


def _test_phase(view, task):
    phase = int(task.state(view).phase[0])
    return "idle" if phase == 0 else ("done" if phase >= 2 else "running")


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task, objects=("panel", "cord_cap", "cradle"),
    pose_modules=("twistlock_sim.task",), test_phase=_test_phase,
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """Milestone / 100 (0.25 the cap lifted 1 cm off its cradle, 0.5 lined up at the CKT 7 receptacle's mouth, 0.75
    pushed home and turned past 21 degrees, 1.0 it also holds a 60 N pull after release); 0 if the cap was ever brought
    to the CKT 9 receptacle or held tilted more than 10 degrees with its blades partly in."""
    report = STATION.report or {}
    if any(report.get("semantic_events", {}).values()):
        return 0.0
    return report.get("milestone", 0) / 100.0
