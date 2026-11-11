"""On the station table, a safety-switch enclosure (a grey box 14 cm deep, 20 cm wide, 44 cm tall) stands about
0.3 m in front of the arms, its side wall facing the robot. On that wall a disconnect handle points up at OFF; a hole
in the handle lines up with a hole in a fixed steel ear only at OFF. A white "DANGER - DO NOT OPERATE" lockout tag
with a brass grommet stands in a grey tag pocket in front of the left arm, and the robot's red padlock lies on its
side, shackle open and pointing toward the switch, on a small grey column nearer the robot. Thread the tag's
grommet onto the padlock's open shackle, then put the shackle through the handle's lock-off hole (and the ear behind
it) and push it shut.

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. ``--policy lock-tag-demo`` is the scripted demo.
LOCKTAG_CONTROL=1 makes it a control run: as the shackle latches, the tag is taken off the shackle and laid against the
padlock's body (a tag that was not threaded); it must score below 1.
"""

from __future__ import annotations

import os

from locktag_sim import station as hb

CONTROL = os.environ.get("LOCKTAG_CONTROL") == "1"


def _props():
    from locktag_sim import task as t

    return [t.Switch(), t.Tag(), t.Padlock(), t.Stands()]


def _task():
    from locktag_sim.task import LockTagTask

    return LockTagTask(default_seed=0, control=CONTROL)


def _test_phase(view, task):
    phase = int(task.state(view).phase[0])
    return "idle" if phase == 0 else ("done" if phase >= 3 else "running")


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task, objects=("switch", "tag", "padlock", "stands"),
    pose_modules=("locktag_sim.task",), test_phase=_test_phase,
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """Milestone / 100 (0.25 the tag's grommet on the padlock's shackle; 0.5 the padlock shut through the handle's
    and the ear's lock-off holes and holding the handle, without the tag; 0.75 shut through both holes with the tag
    on its shackle; 1.0 that, and the after-release check passed); 0 if the handle turned more than 10 degrees
    toward ON before the check."""
    report = STATION.report or {}
    if any(report.get("semantic_events", {}).values()):
        return 0.0
    return report.get("milestone", 0) / 100.0
