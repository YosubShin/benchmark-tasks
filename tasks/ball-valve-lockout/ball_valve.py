"""On the station table, a short 1/2 in steel pipe run stands 26 cm in front of the arms, running left-right, with
two identical lockable bronze ball valves on it: V-12 on the midline and V-14 25 cm to the left. Each valve has a
flat steel lever (yellow vinyl grip) on top, 16.5 cm above the table; both are open, the levers pointing right along
the pipe. Each lever carries a locking sleeve and has a square padlock hole 4 cm from the stem. Behind each valve a
white engraved tag on a strap clamped to the pipe gives its number. The robot's purple padlock stands upright and
open on a small grey pedestal in front of the left arm. Close V-12 (a quarter turn clockwise seen from above, so the
lever points at the robot), slide its sleeve in so it latches, put the padlock's shackle through the lever's hole and
snap it shut. Do not turn V-14.

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. The scripted demo is the policy
``valve-lockout-demo`` (``-P control=true`` closes and latches the valve but skips the padlock, then forces the
function test: a control run that must score below 1).
"""

from __future__ import annotations

from valve_lockout_sim import station as hb


def _props():
    from valve_lockout_sim import task as vt

    return [vt.PipeRun(), vt.ValvePadlock(), vt.Pedestal()]


def _task():
    from valve_lockout_sim.spec import EPISODE_SECONDS
    from valve_lockout_sim.task import ValveLockoutTask

    return ValveLockoutTask(default_seed=0, episode_length_s=EPISODE_SECONDS + 5.0, reviewer_camera=False)


def _test_phase(view, task):
    phase = int(task.state(view).phase[0])
    if phase >= 3 and not getattr(task, "_reported", False):
        task._reported = True
        r = task.episode_report(view, 0)
        print(f"[ball-valve] after-release test: lever opened {r['test_lever_max_open_deg']} deg, sleeve out "
              f"{r['test_sleeve_max_out_mm']} mm, passed {r['function_test_passed']}; milestone {r['milestone']}, "
              f"events {r['semantic_events']} {r['contact_events']}", flush=True)
    return "idle" if phase == 0 else ("done" if phase >= 3 else "running")


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task, objects=("pipe_run", "padlock", "pedestal"),
    pose_modules=("valve_lockout_sim.task",), test_phase=_test_phase,
    demo=lambda: {},
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """Milestone / 100 (0.25 V-12's lever turned 10 degrees toward closed, 0.5 closed, 0.75 the padlock's short leg
    through the lever's hole with the sleeve latched, 1.0 the padlock shut there and the lockout holding in the
    after-release test); 0 if V-14's lever turned 10 degrees or more, or V-12's went back more than 30 degrees from
    closed before the test."""
    report = STATION.report or {}
    if any(report.get("semantic_events", {}).values()):
        return 0.0
    return report.get("milestone", 0) / 100.0


demo = STATION.demo
