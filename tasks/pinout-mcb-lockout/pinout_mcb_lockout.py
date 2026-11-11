"""On the station table an off-white back plate stands upright 0.37 m in front of the arms, a little right of their
midline, with a DIN rail across it carrying five grey miniature circuit breakers side by side. Each breaker has a
coloured label on its top; from right to left: blue, yellow, green, white, orange. The blue breaker's toggle is down
(OFF); the other four are up (ON). A red Brady pin-out breaker lockout (round body, black push button on a stem, two
wire pins at its open end) lies on a grey post to the right of the board, its open end toward the board. A purple
Master Lock 410 padlock stands open on a grey column to the left of the board. Hold the lockout by its body with one
hand, press and hold its button with the other (the pins swing outward only while the button is held), push it onto
the blue breaker's toggle so the pins enter the breaker's holes, then let go of the button so the springs set the
pins behind the holes' edges. Then put the padlock's shackle through the hole in the lockout's stem and push the
shackle shut. Leave the other breakers as they are.

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. PINOUT_LOCKOUT_CONTROL=1 is the control run: the
demo fits and sets the lockout but never brings the padlock, and the after-release check is forced once both arms
have moved away (it must score below 1: the button is pressed and the lockout pulled off).
"""

from __future__ import annotations

from pinout_sim import station as hb


def _props():
    from pinout_sim.task import task_assets

    return task_assets()


def _task():
    from pinout_sim.task import PinoutLockoutTask

    return PinoutLockoutTask(default_seed=0, reviewer_camera=False)


def _test_phase(view, task):
    st = task.state(view)
    if bool(st.test_done[0]):
        return "done"
    return "running" if int(st.test_step[0]) > 0 else "idle"


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task, objects=("board", "stands", "breakers", "lockout", "padlock"),
    pose_modules=("pinout_sim.task",), test_phase=_test_phase,
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """Milestone / 100 (0.25 lockout pushed fully onto the blue breaker, 0.5 set there with the button released and
    both pins behind the holes' edges, 0.75 padlock's short leg through the stem hole, 1.0 shackle shut through it and
    the after-release check passed: button pressed with 20 N stays out, lockout pulled with 20 N while pressing stays
    on, breaker pushed toward ON stays OFF). 0 if the lockout went onto another breaker, another breaker's toggle
    moved, or the blue breaker was switched ON."""
    report = STATION.report or {}
    if any(report.get("semantic_events", {}).values()):
        return 0.0
    milestone = report.get("milestone", 0)
    if milestone >= 100 and not report.get("success"):
        return 0.75
    return milestone / 100.0
