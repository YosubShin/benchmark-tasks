"""On the station table a backboard stands 40 cm in front of the arms with two small grey enclosed rotary isolators
on it, their fronts 31 cm in front of the arms: Q1 on the midline and Q2 24 cm to the right. Each has a black pistol
handle on its front, 20.5 cm above the table, and an engraved nameplate above it: Q1 on a yellow plate,
Q2 on a white plate. Both are ON (handles pointing up). The robot's purple padlock lies open and flat on
a small grey post in front of the right arm. Turn Q1's handle a quarter turn anticlockwise to OFF (pointing left),
pull the red padlock slide out of the bottom of the handle (it only comes out at OFF), put the padlock's shackle
through the hole in the slide and snap it shut. Do not turn Q2.

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. The scripted demo is the policy
``isolator-lockout-demo`` (``-P control=true`` turns Q1 OFF and pulls its slide out but skips the padlock, then
forces the after-release test: a control run that must score below 1).
"""

from __future__ import annotations

from isolator_lockout_sim import station as hb


def _props():
    from isolator_lockout_sim import task as it

    return [it.Isolators(), it.IsolatorPadlock(), it.Pedestal()]


def _task():
    from isolator_lockout_sim.spec import EPISODE_SECONDS
    from isolator_lockout_sim.task import IsolatorLockoutTask

    return IsolatorLockoutTask(default_seed=0, episode_length_s=EPISODE_SECONDS + 5.0, reviewer_camera=False)


def _test_phase(view, task):
    phase = int(task.state(view).phase[0])
    if phase >= 3 and not getattr(task, "_reported", False):
        task._reported = True
        r = task.episode_report(view, 0)
        print(f"[rotary-isolator] after-release test: handle turned {r['test_handle_max_toward_on_deg']} deg toward "
              f"ON, slide pushed in {r['test_slide_max_in_mm']} mm, passed {r['function_test_passed']}; milestone "
              f"{r['milestone']}, events {r['semantic_events']} {r['contact_events']}, shackle {r['test_end_shackle']}, at test start {r['test_start_padlock']}",
              flush=True)
    return "idle" if phase == 0 else ("done" if phase >= 3 else "running")


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task, objects=("isolators", "padlock", "pedestal"),
    pose_modules=("isolator_lockout_sim.task",), test_phase=_test_phase,
    demo=lambda: {},
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """Milestone / 100 (0.25 Q1's handle turned 10 degrees from ON toward OFF, 0.5 at OFF, 0.75 the padlock's short leg
    through the hole of Q1's pulled-out slide, 1.0 the padlock shut there and the lockout holding in the after-release
    test); 0 if Q2's handle turned 10 degrees or more, or Q1's went back more than 30 degrees toward ON after reaching
    OFF, before the test."""
    report = STATION.report or {}
    if any(report.get("semantic_events", {}).values()):
        return 0.0
    return report.get("milestone", 0) / 100.0


demo = STATION.demo
