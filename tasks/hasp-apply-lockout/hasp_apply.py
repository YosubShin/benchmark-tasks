"""On the station table, a safety-switch enclosure (a grey box 12 cm deep, 25 cm wide, 44 cm tall) stands about
0.3 m in front of the arms, its side wall facing the robot. On that wall a disconnect handle points up at OFF; a hole
in the handle lines up with a hole in a fixed steel ear only at OFF. A red Master Lock 420 scissor hasp stands closed
in a holder to the left of the handle, and the robot's purple padlock lies on its side, shackle open, on a small
grey pedestal in front of the left arm. Take the hasp out of its holder, open its jaws, close them through the handle's
lock-off hole (and the ear behind it), then put the padlock's shackle through one of the hasp's tail holes and
push it shut.

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. HASP_CONTROL=1 makes it a control run: the
after-release check also starts once the hasp is on the handle and released, without a padlock (use it with
``--policy hasp-apply-demo``, which then skips the padlock): it must score below 1.
"""

from __future__ import annotations

import os

from haspapply_sim import station as hb

CONTROL = os.environ.get("HASP_CONTROL") == "1"


def _props():
    from haspapply_sim import task as t

    return [t.Switch(), t.Hasp(), t.Padlock(), t.Stands()]


def _task():
    from haspapply_sim.task import HaspApplyTask

    return HaspApplyTask(default_seed=0, control=CONTROL)


def _test_phase(view, task):
    phase = int(task.state(view).phase[0])
    return "idle" if phase == 0 else ("done" if phase >= 3 else "running")


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task, objects=("switch", "hasp", "padlock", "stands"),
    pose_modules=("haspapply_sim.task",), test_phase=_test_phase,
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """Milestone / 100 (0.25 hasp's jaws open at least 15 degrees with its ring within 6 cm of the handle's lock-off
    holes, 0.5 hasp closed through the handle's lock-off hole, 0.75 shackle
    through a tail hole pair, 1.0 shut there and the after-release check passed); 0 if the handle turned more than
    10 degrees toward ON before the check."""
    report = STATION.report or {}
    if any(report.get("semantic_events", {}).values()):
        return 0.0
    return report.get("milestone", 0) / 100.0
