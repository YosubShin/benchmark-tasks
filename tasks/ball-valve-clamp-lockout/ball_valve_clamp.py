"""On the station table, a short 1/2 in steel pipe run stands 26 cm in front of the arms, running left-right, with
two identical bronze ball valves on it: V-12 on the midline and V-14 25 cm to the left. Each valve has a flat steel
lever (yellow vinyl grip) on top, 16.5 cm above the table; both are open, the levers pointing right along the pipe.
Behind each valve a white engraved tag on a strap clamped to the pipe gives its number. A red clamp-on ball valve
lockout (two halves on a slide: a channel with a pair of legs, and a second pair of legs with a housing of four lock
holes) stands on the table in front of the left arm, and the robot's purple padlock lies open on its side on a grey
cradle in front of the right arm. Close V-12 (a quarter turn clockwise seen from above, so the lever points at the
robot), fit the lockout over its lever with the legs astride the pipe, push the halves together, and padlock them
through the lined-up hole. Do not turn V-14.

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. The scripted demo is the policy
``valve-clamp-demo`` (``-P control=true`` closes the valve and fits and closes the lockout but skips the padlock,
then forces the after-release test: a control run that must score below 1).
"""

from __future__ import annotations

from clamp_valve_sim import station as hb


def _props():
    from clamp_valve_sim import task as vt

    return [vt.PipeRun(), vt.Lockout(), vt.ValvePadlock(), vt.Stands()]


def _task():
    from clamp_valve_sim.spec import EPISODE_SECONDS
    from clamp_valve_sim.task import ClampLockoutTask

    return ClampLockoutTask(default_seed=0, episode_length_s=EPISODE_SECONDS + 5.0)


def _test_phase(view, task):
    phase = int(task.state(view).phase[0])
    if phase >= 3 and not getattr(task, "_reported", False):
        task._reported = True
        r = task.episode_report(view, 0)
        print(f"[clamp-lockout] after-release test: lever opened {r['test_lever_max_open_deg']} deg, halves apart "
              f"{r['test_halves_max_apart_mm']} mm, passed {r['function_test_passed']}; milestone {r['milestone']}, "
              f"events {r['semantic_events']} {r['contact_events']}", flush=True)
    return "idle" if phase == 0 else ("done" if phase >= 3 else "running")


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task, objects=("pipe_run", "lockout", "padlock", "stands"),
    pose_modules=("clamp_valve_sim.task",), test_phase=_test_phase,
    demo=lambda: {},
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """Milestone / 100 (0.25 V-12's lever turned 10 degrees toward closed, 0.5 closed, 0.75 the lockout fitted on
    the closed valve with its halves pushed together, 1.0 the padlock shut through the lined-up holes and the
    lockout holding in the after-release test); 0 if V-14's lever turned 10 degrees or more, or V-12's went back
    more than 30 degrees from closed before the test."""
    report = STATION.report or {}
    if any(report.get("semantic_events", {}).values()):
        return 0.0
    return report.get("milestone", 0) / 100.0


demo = STATION.demo
