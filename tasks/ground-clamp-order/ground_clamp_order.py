"""On the station table, an insulating back board stands 0.29 m in front of the arms, carrying four flat copper bars
(1/4 x 2 in) 6 cm above the table, front edges toward the robot: the ground bar GND (green/yellow band) in front of
the right arm, and phase bars A, B and C in front of the left arm. A yellow TESTED tag hangs on the board above bar B.
A protective grounding set (two C-type grounding clamps with eye screws, joined by a yellow ground cable) lies ready:
the ground clamp hangs on a small black parking plate right of GND, the line clamp on one left of bar C. A short stick
with a black square handle stands on each clamp's eye screw; turning it clockwise (seen from above) tightens the
clamp. Hang the ground clamp on GND and tighten it first; then hang the line clamp on bar B and tighten it; let go.
The line clamp must not touch a phase bar before the ground clamp is tight, and no clamp may touch bar A or C.

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. The control run is ``--policy clamp-demo
-P control=true``: the demo hangs the line clamp on bar B without tightening it and the after-release check is forced
(it must score below 1).
"""

from __future__ import annotations

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


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task, objects=("bench", "ground_set"),
    pose_modules=("groundclamp_sim.task",), test_phase=_test_phase,
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
