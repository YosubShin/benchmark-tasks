"""Two industrial gas cylinders stand in wells in front of the robot so that only their shoulders, neck rings and
brass handwheel valves rise above the table: on the right the argon cylinder (dark green shoulder, ARGON label), on
the left the oxygen cylinder (white shoulder, OXYGEN label). Both valves are open half a turn. A red cylinder
lockout cap (Brady LM023E: a can closed at the top, with a locking plate that swings across its open end) stands on
a grey stand between them and the robot, its plate swung open; a blue padlock with its shackle open lies on a small
pedestal to the left. Close the argon valve (handwheel clockwise, seen from above, until it stops), lower the cap
over that valve, swing its plate shut round the valve's neck, put the padlock's shackle through the plate's tab and
the can's tab and snap it shut. Do not touch the oxygen cylinder.

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. The control run is ``--policy
cylinder-lockout-demo -P control=true``: the demo closes the valve, fits the cap and shuts its plate but brings no
padlock, lets go, and the after-release check is forced (it must score below 1).
"""

from __future__ import annotations

from cylvalve_sim import station as hb


def _props():
    from cylvalve_sim import task as ct

    return [ct.Cylinders(), ct.Cap(), ct.Padlock(), ct.Stand()]


def _task():
    from cylvalve_sim.task import CylinderLockoutTask

    return CylinderLockoutTask(default_seed=0)


def _test_phase(view, task):
    phase = int(task.state(view).phase[0])
    return "idle" if phase == 0 else ("done" if phase >= 4 else "running")


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task, objects=("cylinders", "cap", "padlock", "stand"),
    pose_modules=("cylvalve_sim.task",), test_phase=_test_phase,
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """Milestone / 100 (0.25 the argon valve closed, 0.5 the cap seated over it, 0.75 its plate shut round the
    valve's neck, 1.0 padlocked through both tabs and holding after release: the plate's tab pushed toward open with
    10 N, the cap twisted with 1 N m and lifted with 20 N); 0 if the oxygen valve's handwheel turned 15 degrees or
    more, the argon valve was reopened 30 degrees after closing, or the cap was locked onto the oxygen valve."""
    report = STATION.report or {}
    if any(report.get("semantic_events", {}).values()):
        return 0.0
    return report.get("milestone", 0) / 100.0
