"""On the station table a grey steel control-panel section stands upright 30 cm in front of the arms, right of their
midline. It carries three breaker control switches in a row (Electroswitch Series 24, model 2438D): each has a black
pistol-grip handle hanging straight down (NORMAL), a black nameplate engraved BREAKER CONTROL with TRIP on the left
and CLOSE on the right and a red/green target window at its top, two indicating lamps above it (green on the left,
red on the right) and an engraved nameplate above the lamps naming the breaker: 52-1 LINE 1 NORTH (white), 52-2
TRANSFORMER T1 (yellow), 52-3 LINE 2 SOUTH (blue), left to right. Every breaker starts closed: red lamps lit,
targets red. Turning a handle 45 degrees anticlockwise (as seen facing the panel; the grip's lower end swings to the right) is
TRIP, clockwise is CLOSE; a spring returns it to vertical.
Trip breaker 52-2: turn its handle to TRIP and hold it until its green lamp lights, then let go so it springs back.

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. TRIP_CONTROL=1 with --policy trip-demo is the
control run: the demo lets go as soon as the handle reaches the TRIP contact, before the lamp changes (it must
score below 1).
"""

from __future__ import annotations

from trip_sim import station as hb


def _props():
    from trip_sim.task import Panel

    return [Panel()]


def _task():
    from trip_sim.task import TripTask

    return TripTask(default_seed=0)


def _test_phase(view, task):
    phase = int(task.state(view).phase[0])
    return "idle" if phase == 0 else ("done" if phase >= 2 else "running")


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task, objects=("panel",), pose_modules=("trip_sim.task",), test_phase=_test_phase,
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """0.25 the 52-2 handle turned 10 degrees toward TRIP; 0.5 it reached the TRIP contact (within 5 degrees of the
    TRIP stop); 0.75 breaker 52-2 tripped (green lamp lit); 1.0 the after-release check passed (robot clear of the
    grip, handle back at NORMAL by its spring, target green, breaker 52-2 open, the other two closed with handles at
    NORMAL). 0 if another switch's handle turned more than 10 degrees, or any handle more than 20 degrees toward
    CLOSE."""
    report = STATION.report or {}
    if any(report.get("semantic_events", {}).values()):
        return 0.0
    return report.get("milestone", 0) / 100.0
