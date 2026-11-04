"""On the station table a small distribution board stands upright 0.32-0.40 m in front of the arms, centred between
them. Behind a light-grey dead-front cover, six white Schneider Acti9 iC60N miniature circuit breakers sit side by
side on one DIN rail, their fronts poking through a cut-out in the cover; every handle is up (ON) and each window
shows I-ON. Directly above each breaker a coloured printed label gives its way number (1-6, left to right) and its
circuit: 1 LIGHTING - WORKSHOP (yellow), 2 HEATERS - OFFICE (red), 3 BATTERY CHARGER (orange), 4 SOCKETS - KITCHEN
(blue), 5 HVAC - AHU-1 (green), 6 SPARE (white). A circuit directory card on the cover, right of the breakers,
lists the same ways. Switch off the breaker that feeds SOCKETS - KITCHEN: push its handle down past the snap point
to OFF, so its window shows the green O-OFF band, and leave the other five ON.

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. BREAKER_CONTROL=1 with --policy breaker-demo is
the control run: the demo pushes the named handle only 30% of its throw and lets go (it must score below 1).
"""

from __future__ import annotations

from breaker_sim import station as hb


def _props():
    from breaker_sim.task import Board

    return [Board()]


def _task():
    from breaker_sim.task import SwitchOffTask

    return SwitchOffTask(default_seed=0, reviewer_camera=False)


def _test_phase(view, task):
    phase = int(task.state(view).phase[0])
    return "idle" if phase == 0 else ("done" if phase >= 2 else "running")


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task, objects=("board",), pose_modules=("breaker_sim.task",), test_phase=_test_phase,
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """0.25 the SOCKETS - KITCHEN handle (way 4) turned 5 degrees toward OFF; 0.5 it passed the middle of its travel;
    0.75 at the end it sits at its OFF stop, window O-OFF, the other five ON, but the after-release check failed or
    was not reached; 1.0 the after-release check passed (pushed toward ON with 0.063 N m for 1.5 s, it stays OFF and
    the other five stay ON). 0 if any other breaker's handle passed the middle of its travel."""
    report = STATION.report or {}
    if any(report.get("semantic_events", {}).values()):
        return 0.0
    return report.get("milestone", 0) / 100.0
