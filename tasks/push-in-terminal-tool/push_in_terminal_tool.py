"""A vertical mounting plate stands on the station table 32.5 cm in front of the arms, right of their midline. On its
DIN rail, six WAGO TOPJOB S push-in terminal blocks (5.2 mm wide) stand side by side, marked X1:1 to X1:6 from left
to right: X1:1-3 and X1:6 grey, X1:4 blue (neutral), X1:5 green-yellow (earth). Each has a round conductor entry near
its upper end and, just below it, a small rectangular operating slot. A grey wiring duct runs below them. A blue
2.5 mm2 fine-stranded wire, stripped 10 mm with no ferrule (bare copper strands), comes out of the duct, runs along
the table and up onto a small dark stand 17 cm in front of the arms. A green WAGO operating tool (a small
screwdriver, 3.5 mm blade) lies on a second stand between the arms. Bare strands cannot be pushed into the clamp:
push the tool's blade straight into X1:4's operating slot to open it, insert the stripped end into X1:4's upper
entry until the insulation stops, then pull the tool out so the clamp closes, and let go. The wire must not go into
any other terminal, and the tool must not be left in a slot.

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. The control run is ``--policy terminal-tool-demo
-P control=true``: the demo pushes the stripped end into X1:4 without the tool and lets go, and the tug is forced (it
must score below 1).
"""

from __future__ import annotations

from terminal_tool_sim import station as hb


def _props():
    from terminal_tool_sim import task as tt

    return [tt.TerminalStrip(), tt.Wire(), tt.WireStand(), tt.OperatingTool()]


def _task():
    from terminal_tool_sim.task import PushInTask

    return PushInTask(default_seed=0)


def _test_phase(view, task):
    phase = int(task.state(view).phase[0])
    return "idle" if phase == 0 else ("done" if phase >= 2 else "running")


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task, objects=("terminal_strip", "wire", "wire_stand", "op_tool"),
    pose_modules=("terminal_tool_sim.task",), test_phase=_test_phase,
    # The clamp's 0.1 mm contacts need 480 Hz physics (yam_arena runs 240 Hz) and friction patches sized for
    # millimetre contacts (PhysX's default merges contacts within 25 mm into one patch).
    physics_hz=480.0,
    physx={"friction_correlation_distance": 0.0005, "friction_offset_threshold": 0.002},
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """Milestone / 100 (0.25 the tool's blade in X1:4's operating slot holding its clamp open, 0.5 the strands then
    in past the clamping point, 0.75 seated: insulation home, tool out, clamp closed on the strands, 1.0 it also holds
    a 50 N pull after release); 0 if the strands' tip ever went more than 1 mm into any other entry, or if the tool's
    blade is in any operating slot at the end."""
    report = STATION.report or {}
    if any(report.get("semantic_events", {}).values()) or any(report.get("hazard_state", {}).values()):
        return 0.0
    return report.get("milestone", 0) / 100.0
