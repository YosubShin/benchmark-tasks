"""A wheel-off rear corner on a hub stand on the bench in front of the arms: a 9-1/2 in GM-pattern duo-servo rear
drum brake, its axle running left-right, the drum's face and wheel studs toward the left arm, the backing plate's back
toward the right arm. The adjusting hole near the bottom of the backing plate is open (its rubber cover lies on the
bench); behind it, out of sight, is the star wheel of the shoes' adjusting screw. A 9 in brake adjusting spoon lies on
two wooden rests on the right. The shoes are backed off: the drum turns freely. Goal: the brake adjusted with the spoon
so that the drum turns by hand with a light drag. Constraints: do not over-tighten it, do not drop the spoon; the
spoon out of the hole and the drum let go at the end. The usual way (not given to the robot): put the spoon's tip
through the hole onto the star wheel's teeth, lever the handle so the teeth move upward a notch at a time (that
expands the shoes on this brake), and turn the drum by hand every few notches to feel for the drag; stop at a light
drag (going on past it cannot be undone with the spoon: the self-adjuster pawl holds the wheel).

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md.
"""

from __future__ import annotations

import os

from drumbrake_sim import station as hb


def _props():
    from drumbrake_sim.task import all_props

    return all_props()


def _task():
    from drumbrake_sim.task import DrumBrakeTask

    return DrumBrakeTask()


def _test_phase(view, task):
    st = task.state(view)
    phase = int(st.phase[0])
    if phase >= 2 and not getattr(task, "_printed", False):
        task._printed = True
        print("[drum-brake] REPORT", task.episode_report(view, 0), flush=True)
    return "idle" if phase == 0 else ("done" if phase >= 2 else "running")


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task, objects=("bench", "brake", "spoon"),
    pose_modules=("drumbrake_sim.task",), test_phase=_test_phase,
    physics_hz=float(os.environ.get("DBA_PHYSICS_HZ", "480")),
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """The reported score (drumbrake_sim.scorer.Scorer): milestone / 100 (0.25 the spoon's tip through the hole on
    the star wheel's teeth, 0.5 also 5 notches or more turned in the expanding direction, 0.75 the drum's turning torque
    within 0.3-1.0 N m, 1.0 the after-release check passed: spoon out of the hole, hands clear, a torque ramped on the
    drum turns it between 0.3 and 1.0 N m); 0 for the whole trial once the brake was over-adjusted (turning torque
    above 1.0 N m) or the spoon dropped, whatever was reached before or after."""
    report = STATION.report or {}
    return float(report.get("score", 0.0))
