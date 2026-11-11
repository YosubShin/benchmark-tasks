"""On the station table, a red Master Lock S601 group lock box (29 x 13 cm, 14 cm tall) stands on a low block 0.26 m in front of the
arms, its long side facing the robot. Its lid is hinged at the left end and latched at the right end; along the front
a flange carries nine numbered holes through the lid's and the body's flanges, and a coworker's yellow padlock is
already locked through the middle hole (5). A key slot crosses the lid near the latch. To the right, two pump starter
panels on a board (PUMP 1 with a yellow nameplate, PUMP 2 with a white one) each carry a red job padlock; the key of
each job lock hangs on a peg on its panel's right side. The robot's purple padlock stands open on a small grey pedestal in front
of the left arm. Take PUMP 1's key, drop it through the box's key slot, then hang the purple padlock through a free
front hole to the right of the yellow lock and push it shut.

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. LOCKBOX_CONTROL=1 makes it a control run: the
after-release check starts 2 s after the key is inside the box, with or without a robot padlock (use it with
``--policy lockbox-demo``, which then skips the padlock): it must score below 1.
"""

from __future__ import annotations

import os

from lockbox_sim import station as hb

CONTROL = os.environ.get("LOCKBOX_CONTROL") == "1"


def _props():
    from lockbox_sim import task as t

    return [t.LockBox(), t.Fixtures(), t.Key("key", 0), t.Key("other_key", 1), t.Padlock(), t.CoworkerLock()]


def _task():
    from lockbox_sim.task import LockBoxTask

    return LockBoxTask(default_seed=0, control=CONTROL)


def _test_phase(view, task):
    phase = int(task.state(view).phase[0])
    return "idle" if phase == 0 else ("done" if phase >= 2 else "running")


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task, objects=("lockbox", "fixtures", "key", "other_key", "padlock", "coworker_lock"),
    pose_modules=("lockbox_sim.task",), test_phase=_test_phase,
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """Milestone / 100 (0.25 PUMP 1's key off its hook, 0.5 it is inside the box, 0.75 with it inside, the purple
    padlock's short leg through a free flange hole pair, 1.0 shut there and the after-release check passed); 0 if
    PUMP 2's key left its hook or the yellow padlock moved before the purple shackle shut."""
    report = STATION.report or {}
    if any(report.get("semantic_events", {}).values()):
        return 0.0
    return report.get("milestone", 0) / 100.0
