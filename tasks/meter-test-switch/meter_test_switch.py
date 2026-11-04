"""A grey steel plate stands upright on the station table 39 cm in front of the arms, a stand-in for a meter box's
back wall. On it is a 10-pole meter test switch (black base, 24 cm wide, poles vertical, 19 mm apart), its ten
knife-blade switches closed. Handle colours from the robot's left: orange, red, black, yellow, green, black, brown,
blue, black, white. Barriers stand either side of the three black (voltage) switches. A red test plug lies on a small
grey rest 20 cm in front of the left arm; a black lead runs from its handle along the table to a yellow test-set box
left of the left arm. Open the four voltage switches, open the orange phase-A short-circuiting switch and then the red
phase-A test-jack switch, push the test plug into the red switch's test jack, then push the orange switch back in.
Never open the red switch while the orange one is closed. Leave the other switches closed.

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. The control run is ``--policy switch-demo
-P control=true``: the demo leaves the red test-jack switch closed and tries the plug anyway (it must score below 1).
"""

from __future__ import annotations

from testswitch_sim import station as hb


def _props():
    from testswitch_sim import task as tt

    return [tt.MeterTestSwitch(), tt.TestLead(), tt.PlugPost()]


def _task():
    from testswitch_sim.task import SwitchTask

    return SwitchTask(default_seed=0)


def _test_phase(view, task):
    phase = int(task.state(view).phase[0])
    return "idle" if phase == 0 else ("done" if phase >= 2 else "running")


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task, objects=("switch", "lead", "plug_post"),
    pose_modules=("testswitch_sim.task",), test_phase=_test_phase,
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """Milestone / 100 (0.25 the four voltage switches open, 0.5 the orange and red switches also open, 0.75 the test
    plug seated in the red switch's jack, 1.0 the orange switch pushed back in with the plug seated and the
    after-release check passed); 0 if a test-jack switch was ever out of its jaw with its phase's CT neither shorted
    nor carried by a seated plug, or a phase-B/C current switch moved."""
    report = STATION.report or {}
    if any(report.get("semantic_events", {}).values()):
        return 0.0
    return report.get("milestone", 0) / 100.0
