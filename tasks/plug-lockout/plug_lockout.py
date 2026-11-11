"""A grey plate stands upright on the table to the robot's right with a duplex wall outlet. Two black power cords are
plugged into it: the top one carries a yellow cord tag ("GRINDER"), the bottom one a green tag ("VENT FAN"). A red
plug lockout box (Brady 65674) lies open like a book on a grey stand in front of the robot: its lower half-shell is
held in a fixture, its lid lies open flat behind it, and each half has a flat tab with a hole on its front edge. A
blue padlock with its shackle open lies on its side on a small pedestal to the left. Unplug the yellow-tagged cord,
lay its plug in the lockout's lower half with the cord through the notch at the end, close the lid, put the
padlock's shackle through both tabs and snap it shut. The green-tagged cord must stay plugged in.

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. The control run is ``--policy plug-lockout-demo
-P control=true``: the demo closes the lid on the plug but brings no padlock, lets go, and the after-release check is
forced (it must score below 1).
"""

from __future__ import annotations

from pluglock_sim import station as hb


def _props():
    from pluglock_sim import task as pt

    return [pt.Outlet(), pt.Cord("target"), pt.Cord("decoy"), pt.Lockout(), pt.Padlock(), pt.Stand()]


def _task():
    from pluglock_sim.task import PlugLockoutTask

    return PlugLockoutTask(default_seed=0)


def _test_phase(view, task):
    phase = int(task.state(view).phase[0])
    return "idle" if phase == 0 else ("done" if phase >= 3 else "running")


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task, objects=("outlet", "cord_target", "cord_decoy", "lockout", "padlock", "stand"),
    pose_modules=("pluglock_sim.task",), test_phase=_test_phase,
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """Milestone / 100 (0.25 the yellow-tagged plug unplugged, 0.5 laid in the lockout's lower half, 0.75 the lid
    closed on it, 1.0 padlocked through both tabs and holding after release: the lid pried with 0.5 N*m, the cord
    pulled with 40 N); 0 if the green-tagged plug was ever pulled 3 mm or more out of its outlet."""
    report = STATION.report or {}
    if any(report.get("semantic_events", {}).values()):
        return 0.0
    return report.get("milestone", 0) / 100.0
