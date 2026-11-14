"""On the station table, an insulating back board stands 0.29 m in front of the arms, carrying four flat copper bars
(1/4 x 2 in) 6 cm above the table, front edges toward the robot: the ground bar GND (green/yellow band) in front of
the right arm, and phase bars A, B and C in front of the left arm. A yellow TESTED card lies on the ledge on top of the
board, behind bar B. A protective grounding set (two C-type grounding clamps with eye screws, joined by a yellow
ground cable) lies ready: its ground clamp hangs on a small black parking plate right of GND, its line clamp on one
left of bar C. A short stick with a black square handle and a red cap stands on each clamp's eye screw.

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. Control run: ``GROUNDCLAMP_CONTROL=1`` with
``--policy clamp-demo -P control=true`` (the demo hangs the line clamp on bar B without tightening it; the variable
makes the task start its after-release check once that clamp has hung on bar B with both hands clear for 1 s, so the
run ends early; it must score below 1).
"""

from __future__ import annotations

import os

from groundclamp_sim import station as hb


def _props():
    from groundclamp_sim import task as ct

    return [ct.Bench(), ct.GroundSet()]


def _task():
    from groundclamp_sim.task import ClampTask

    return ClampTask(default_seed=0)


def _test_phase(view, task):
    phase = int(task.state(view).phase[0])
    return "idle" if phase == 0 else ("done" if phase >= 3 else "running")


_control_clear_steps = [0]


def _on_step(view, task):
    """Control runs only (``GROUNDCLAMP_CONTROL=1``): once the line clamp has hung on the tagged bar with the ground
    clamp tight and both hands clear for 1 s, start the after-release check (the control never tightens the line
    clamp, so the task would otherwise wait for the step limit)."""
    if os.environ.get("GROUNDCLAMP_CONTROL") != "1":
        return
    st = task.state(view)
    last = st.last
    if not last:
        return
    ok = bool((last["line_on_tagged"] & last["hands_clear"] & last["ground_attached"])[0])
    _control_clear_steps[0] = _control_clear_steps[0] + 1 if ok else 0
    if _control_clear_steps[0] >= st.wait_steps:
        task.force_function_test(view)


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task, objects=("bench", "ground_set"),
    pose_modules=("groundclamp_sim.task",), test_phase=_test_phase, on_step=_on_step,
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """Milestone / 100 (0.25 the ground clamp tight on GND, 0.5 the line clamp also hung on bar B, 0.75 the line clamp
    tight too and both hands clear, 1.0 the after-release tug check passed on both clamps); 0 if a clamp touched a
    phase bar before the ground clamp was tight, or touched bar A or C."""
    report = STATION.report or {}
    if any(report.get("semantic_events", {}).values()):
        return 0.0
    return report.get("milestone", 0) / 100.0
