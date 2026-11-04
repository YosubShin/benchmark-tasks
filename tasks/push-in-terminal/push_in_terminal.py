"""A vertical mounting plate stands on the station table 32.5 cm in front of the arms, right of their midline. On its
DIN rail, six WAGO TOPJOB S push-in terminal blocks (5.2 mm wide) stand side by side, marked X1:1 to X1:6 from left
to right: X1:1-3 and X1:6 grey, X1:4 blue (neutral), X1:5 green-yellow (earth). A grey wiring duct runs below them.
A blue 2.5 mm2 wire with a blue ferrule on its end comes out of the duct, runs along the table and up onto a small
dark stand 17 cm in front of the arms, where its ferruled end lies pointing at the strip. Push the ferrule straight
into the upper entry of terminal X1:4 until the collar stops, then let go. The wire must not go into any other
terminal.

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. The control run is ``--policy terminal-demo
-P control=true``: the demo stops the push once the tip is 4.5 mm in (short of home) and lets go, and the tug is
forced (it must score below 1).
"""

from __future__ import annotations

from terminal_sim import station as hb


def _props():
    from terminal_sim import task as tt

    return [tt.TerminalStrip(), tt.Wire(), tt.WireStand()]


def _task():
    from terminal_sim.task import PushInTask

    return PushInTask(default_seed=0)


def _test_phase(view, task):
    phase = int(task.state(view).phase[0])
    return "idle" if phase == 0 else ("done" if phase >= 2 else "running")


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task, objects=("terminal_strip", "wire", "wire_stand"),
    pose_modules=("terminal_sim.task",), test_phase=_test_phase,
    # The clamp's 0.1 mm contacts need 480 Hz physics (yam_arena runs 240 Hz) and friction patches sized for
    # millimetre contacts (PhysX's default merges contacts within 25 mm into one patch).
    physics_hz=480.0,
    physx={"friction_correlation_distance": 0.0005, "friction_offset_threshold": 0.002},
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """Milestone / 100 (0.25 the wire end lifted 1 cm off its stand, 0.5 the ferrule's tip 1 mm or more into X1:4's
    upper entry, 0.75 pushed home with the clamp engaged, 1.0 it also holds a 50 N pull after release); 0 if the
    ferrule's tip ever went more than 1 mm into any other entry."""
    report = STATION.report or {}
    if any(report.get("semantic_events", {}).values()):
        return 0.0
    return report.get("milestone", 0) / 100.0
