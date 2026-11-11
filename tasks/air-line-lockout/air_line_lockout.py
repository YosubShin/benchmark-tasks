"""A two-outlet air manifold stands on the table to the robot's right. Two blue air hoses are connected to its brass
quick couplers: the near one carries a yellow hose tag ("GRINDER"), the far one a green tag ("DRILL"). A red plug
lockout box (Brady PLO27E) stands open on a grey stand in front of the robot: its lower half is held in a fixture,
its lid rests open past upright, and each half has a flat tab with a hole at the middle of its front. A blue padlock
with its shackle open lies on its side on a small pedestal to the left. Pull the yellow-tagged hose's coupler sleeve
back and pull its plug out, lay the plug in the lockout's lower half with the hose through the hole at the end, close
the lid, put the padlock's shackle through both tabs and snap it shut. The green-tagged hose must stay connected.

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. The control run is ``--policy
air-line-lockout-demo -P control=true``: the demo closes the lid on the plug but brings no padlock, lets go, and the
after-release check is forced (it must score below 1).
"""

from __future__ import annotations

from airlock_sim import station as hb


def _props():
    from airlock_sim import task as at

    return [at.Manifold(), at.Hose("target"), at.Hose("decoy"), at.Lockout(), at.Padlock(), at.Stand()]


def _task():
    from airlock_sim.task import AirLockoutTask

    return AirLockoutTask(default_seed=0)


def _test_phase(view, task):
    phase = int(task.state(view).phase[0])
    return "idle" if phase == 0 else ("done" if phase >= 3 else "running")


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task, objects=("manifold", "hose_target", "hose_decoy", "lockout", "padlock", "stand"),
    pose_modules=("airlock_sim.task",), test_phase=_test_phase,
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """Milestone / 100 (0.25 the yellow-tagged hose's plug out of its coupler, 0.5 laid in the lockout's lower half,
    0.75 the lid closed on it, 1.0 padlocked through both tabs and holding after release: the lid pried with 1 N*m,
    the hose pulled with 40 N); 0 if the green-tagged hose's plug ever moved 3 mm or more from its seat."""
    report = STATION.report or {}
    if any(report.get("semantic_events", {}).values()):
        return 0.0
    return report.get("milestone", 0) / 100.0
