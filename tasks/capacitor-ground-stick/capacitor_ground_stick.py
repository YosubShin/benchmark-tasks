"""On the station table, two oil-filled capacitors stand side by side in depth, 33 and 44 cm in front of the arms, just
right of their midline: C2 (orange labels) in front, C1 (blue labels) behind it. Each has two terminal studs on top,
4 cm apart. A copper earth bar stands on edge on two insulators in front of the right arm. A 30 cm yellow discharge
stick with a black handgrip lies in a cradle in front of the left arm; its yellow ground lead ends in a copper spring
clip, clipped to a black insulating tab near the earth bar. An amber WAIT lamp is lit for the first 20 s.

Clip the lead's clip onto the earth bar first; once the WAIT lamp is out, lay the stick's copper hook across both studs
of C2 and let go, leaving the stick hanging there. The hook must not touch C2's studs before the clip is on the bar or
while the lamp is lit, and must not touch C1's studs.

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. The control run is ``--policy groundstick-demo
-P control=true``: the clip is left on its insulating tab and the stick is laid on C2 anyway (it must score 0).
"""

from __future__ import annotations

from groundstick_sim import station as hb


def _props():
    from groundstick_sim import task as gt

    return [gt.Bench(), gt.DischargeStick()]


def _task():
    from groundstick_sim.task import StickTask

    return StickTask(default_seed=0)


def _test_phase(view, task):
    phase = int(task.state(view).phase[0])
    return "idle" if phase == 0 else ("done" if phase >= 3 else "running")


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task, objects=("bench", "stick"),
    pose_modules=("groundstick_sim.task",), test_phase=_test_phase,
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """Milestone / 100 (0.25 the clip on the earth bar; 0.5 the hook held across both C2 studs for 1.5 s after the lamp,
    with the clip on the bar; 0.75 the stick left hanging there, hands clear; 1.0 the after-release check passed); 0 if
    the hook touched a C2 stud before the clip was on the bar or while the lamp was lit, touched a C1 stud, or a finger
    came within 15 mm of a C2 stud."""
    report = STATION.report or {}
    if any(report.get("semantic_events", {}).values()):
        return 0.0
    return report.get("milestone", 0) / 100.0
