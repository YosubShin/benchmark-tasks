"""On the station table a small grey load centre stands 0.31 m in front of the arms, its deadfront facing them, with
two columns of three black Square D QO breakers whose handles move left and right; a handle leaning toward the
centre strip is ON. The centre strip carries a coloured circuit label beside each breaker: 1, 3, 5 on the left
(white, green, orange) and 2, 4, 6 on the right (yellow, blue, white). Circuit 4 (blue label, right column, middle
row) is OFF; the others are ON. A red Master Lock 493B clamp-on breaker lockout lies on grey posts to the right, its
open side toward the robot and its lever swung open toward the robot. A purple Master Lock 410 padlock stands open on
a grey column to the left. Press the lockout onto circuit 4's handle, roll its black knob until the jaws grip the handle, swing the
lever shut over the knob, then hook the padlock's open shackle down through the holes at the knob end (the lever's tab
lies over the lockout's loop) and press it shut. Leave the other breakers as they are.

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. CLAMP_LOCKOUT_CONTROL=1 is the control run: the
demo seats the lockout and shuts its lever but never turns the knob or brings the padlock, and the after-release
check is forced once both arms have moved away (it must score below 1: the loose lockout pulls off).
"""

from __future__ import annotations

import os

from clamp_lockout import station as hb

CONTROL = os.environ.get("CLAMP_LOCKOUT_CONTROL") == "1"


def _props():
    from clamp_lockout.task import task_assets

    return task_assets()


def _task():
    from clamp_lockout.task import BreakerLockoutTask

    return BreakerLockoutTask(default_seed=0, reviewer_camera=False)


def _test_phase(view, task):
    st = task.state(view)
    if bool(st.test_done[0]):
        return "done"
    return "running" if int(st.test_step[0]) > 0 else "idle"


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task, objects=("stands", "panel", "lockout", "padlock"),
    pose_modules=("clamp_lockout.task",), test_phase=_test_phase,
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """Milestone / 100 (0.25 lockout seated on circuit 4's handle, 0.5 its knob turned until the jaws grip the handle,
    0.75 lever shut over it and the padlock's shackle through both end holes, 1.0 shackle shut through them and the
    after-release check passed: lever pulled at 20 N stays shut, lockout pulled at 20 N stays on, handle pushed toward
    ON at 0.3 N m stays OFF). 0 if the lockout went onto another breaker, another breaker's handle moved, or circuit 4
    was switched ON."""
    report = STATION.report or {}
    if any(report.get("semantic_events", {}).values()):
        return 0.0
    milestone = report.get("milestone", 0)
    if milestone >= 100 and not report.get("success"):
        return 0.75
    return milestone / 100.0
