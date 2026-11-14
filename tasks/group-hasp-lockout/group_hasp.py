"""On the station table, a safety-switch enclosure (a grey box 12 cm deep, 25 cm wide, 44 cm tall) stands 0.29 m in
front of the arms, midway between them, its side wall facing the robot. On that wall a disconnect handle is held in
the OFF (lever up) position by a red scissor-type group-lockout hasp that hangs through the handle's lock-off hole and
a fixed steel ear. A coworker's yellow padlock is already locked through one of the hasp's six tail holes (the hole on
the wall side, bottom row). The robot's own purple padlock lies on its side, shackle open, on a small grey pedestal
in front of the left arm. Put the purple padlock's shackle through a free hole of the hasp and push it shut. Do not
move or remove the yellow padlock.

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. LOCKOUT_CONTROL=1 forces the hold test at 3 s with no
robot padlock (a control run: it must score 0). LOCKOUT_HAZARD_TEST=<case> runs a hazard check (README "Hazard
checks"; empty in every scored run).
"""

from __future__ import annotations

import os

from lockout_sim import station as hb

CONTROL = os.environ.get("LOCKOUT_CONTROL") == "1"


def _props():
    from lockout_sim import task as gt
    from lockout_sim.spec import make_seed

    return [gt.Switch(), gt.Hasp(), gt.CoworkerLock(make_seed(0)), gt.GroupPadlock(), gt.Pedestal()]


def _task():
    from lockout_sim.task import GroupHaspTask

    return GroupHaspTask(default_seed=0, reviewer_camera=False)


def _test_phase(view, task):
    phase = int(task.state(view).phase[0])
    return "idle" if phase == 0 else ("done" if phase >= 3 else "running")


def _control(view, task):
    if CONTROL and int(task.state(view).scorer.step[0]) == round(3.0 / view.step_dt):
        task.force_function_test(view)


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task, objects=("switch", "hasp", "coworker_lock", "padlock", "pedestal"),
    pose_modules=("lockout_sim.task",), on_step=_control, test_phase=_test_phase,
    demo=(lambda: {}) if CONTROL else None,
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """The score computed by lockout_sim.scorer.GroupHaspScorer (report()["score"]): milestone / 100 (0.25 the purple
    padlock moved 1 cm off its pedestal, 0.5 its short shackle leg's tip within 12 mm of a free tail hole, 0.75 the
    leg through both jaws there, 1.0 shut there and the hold test passed), and 0 from the first step at which the
    coworker's padlock moves more than 1 cm from where it hangs in its hole (watched until the purple shackle shuts),
    or, before the hold test, the hasp's jaws open more than 8 degrees or the handle turns more than 10 degrees toward
    ON, whatever milestones were reached before or after."""
    return float((STATION.report or {}).get("score", 0.0))


if STATION.demo is not None:
    demo = STATION.demo
