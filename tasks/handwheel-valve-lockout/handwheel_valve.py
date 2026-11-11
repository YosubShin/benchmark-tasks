"""On the station table, a short 1/2 in steel pipe run stands 26 cm in front of the arms, running left-right, with
two identical bronze gate valves on it: V-21 on the midline and V-23 25 cm to the left. Each has a small dark
handwheel on top, 15 cm above the table, one turn open. Behind each valve an engraved valve tag gives its number on a
service colour: V-21's tag is yellow, V-23's green. A red rotating gate valve lockout cover (two halves on a centre
pivot, opened) lies on a stand in front of the left arm, and the robot's purple padlock lies open on its side on a grey
cradle in front of the right arm. Close V-21 (clockwise seen from above, about one turn, regripping as needed), slide
the cover onto its handwheel so the stem sits in the cover's centre, swing the cover's loose half shut, and padlock
the two tabs together. Do not turn V-23.

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. The scripted demo is the policy
``wheel-lockout-demo`` (``-P control=true`` closes the valve, fits the cover and swings it shut but skips the
padlock, then forces the after-release test: a control run that must score below 1).
"""

from __future__ import annotations

from wheel_lockout_sim import station as hb


def _props():
    from wheel_lockout_sim import task as vt

    return [vt.PipeRun(), vt.Cover(), vt.ValvePadlock(), vt.Stands()]


def _task():
    from wheel_lockout_sim.spec import EPISODE_SECONDS
    from wheel_lockout_sim.task import WheelLockoutTask

    return WheelLockoutTask(default_seed=0, episode_length_s=EPISODE_SECONDS + 5.0)


def _test_phase(view, task):
    phase = int(task.state(view).phase[0])
    if phase >= 4 and not getattr(task, "_reported", False):
        task._reported = True
        r = task.episode_report(view, 0)
        print(f"[wheel-lockout] after-release test: swing opened {r['test_swing_max_open_deg']} deg, wheel "
              f"{r['test_wheel_max_open_deg']} deg off its seat, cover turned {r['test_cover_turned_deg']} deg, off axis "
              f"{r['test_cover_off_axis_mm']} mm, rose {r['test_cover_rise_mm']} mm, passed {r['function_test_passed']}; "
              f"milestone {r['milestone']}, events {r['semantic_events']} {r['contact_events']}", flush=True)
    return "idle" if phase == 0 else ("done" if phase >= 4 else "running")


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task, objects=("pipe_run", "cover", "padlock", "stands"),
    pose_modules=("wheel_lockout_sim.task",), test_phase=_test_phase,
    demo=lambda: {},
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """Milestone / 100 (0.25 V-21's handwheel turned 90 degrees toward closed, 0.5 closed, 0.75 the cover fitted on
    the closed valve and swung shut, 1.0 the padlock shut through both tabs and the cover holding in the
    after-release test); 0 if V-23's handwheel turned 10 degrees or more, or V-21's went back more than 30 degrees
    from its seat before the test."""
    report = STATION.report or {}
    if any(report.get("semantic_events", {}).values()):
        return 0.0
    return report.get("milestone", 0) / 100.0


demo = STATION.demo
