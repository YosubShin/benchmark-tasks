"""A light-grey push-button station lies on the table in front of the robot with two flush push buttons facing up: a
green START (nearer the robot) and a red STOP 7 cm behind it, each with a clear install base ring under its chrome
bezel and a black legend plate beside it. A clear push-button lockout cover (Master Lock S2151: two half-shells on a
vertical hinge, each with a padlock ear at the free edge) stands open on a grey block to the right. A blue padlock
with its shackle open lies on a pedestal to the left. Set the open cover around the START button so its bottom edge
goes under the base ring, close it, put the padlock's shackle through both ears and snap it shut. Do not press
either button and do not cover the STOP button.

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. The control run is ``--policy
pushbutton-lockout-demo -P control=true``: the demo fits and closes the cover but brings no padlock, lets go, and the
after-release check is forced (it must score below 1).
"""

from __future__ import annotations

from pushbutton_sim import station as hb


def _props():
    from pushbutton_sim import task as pt

    return [pt.Station(), pt.Cover(), pt.Padlock(), pt.Stand()]


def _task():
    from pushbutton_sim.task import PushButtonLockoutTask

    return PushButtonLockoutTask(default_seed=0)


def _test_phase(view, task):
    phase = int(task.state(view).phase[0])
    return "idle" if phase == 0 else ("done" if phase >= 4 else "running")


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task, objects=("station", "cover", "padlock", "stand"),
    pose_modules=("pushbutton_sim.task",), test_phase=_test_phase,
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """Milestone / 100 (0.25 the cover set around the START button, 0.5 closed there, 0.75 padlocked through both
    ears, 1.0 holding after release: pressed with 20 N, pried with 0.35 N*m, lifted with 10 N); 0 if either button was
    ever pushed in to its contact point or the cover was set over the STOP button."""
    report = STATION.report or {}
    if any(report.get("semantic_events", {}).values()):
        return 0.0
    return report.get("milestone", 0) / 100.0
