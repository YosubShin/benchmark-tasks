"""On the station table, a light mounting plate 0.28 m in front of the arms carries a row of six grey test-disconnect
terminal blocks lying face up, their long axis pointing away from the robot. They form three two-wire CT circuits
separated by dark partition plates; from the robot's left: 1S1/1S2 (red markers, red CT1 card on the plate in front),
2S1/2S2 (yellow markers, yellow CT2 card, straight ahead between the arms), 3S1/3S2 (blue markers, blue CT3 card).
Each terminal has a slot along its middle with a test socket near each end and a small sliding link at the slot's
bottom. A black two-pin bridge stands in a small holder in front of the left arm; a red-handled screwdriver stands
blade-down in a stand in front of the right arm. Push the bridge into the two test sockets nearest the robot on
2S1 and 2S2 and let go, then with the screwdriver push the slide at the bottom of each of those two slots away from
the robot until it stops. The bridge must be in before either slide moves. Do not touch the red or blue circuits.

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. The control run is ``--policy ctbridge-demo
-P control=true``: the demo leaves the bridge in its holder, opens both slides, then forces the after-release check
(it must score 0).
"""

from __future__ import annotations

from ctbridge_sim import station as hb


def _props():
    from ctbridge_sim.task import Bridge, Holder, Screwdriver, Strip

    return [Strip(), Bridge(), Screwdriver(), Holder()]


def _task():
    from ctbridge_sim.task import BridgeTask

    return BridgeTask(default_seed=0, reviewer_camera=False)


def _test_phase(view, task):
    phase = int(task.state(view).phase[0])
    return "idle" if phase == 0 else ("done" if phase >= 3 else "running")


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task, objects=("strip", "bridge", "screwdriver", "holder"),
    pose_modules=("ctbridge_sim.task",), test_phase=_test_phase,
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """Milestone / 100 (0.25 the bridge lifted off its holder, 0.5 seated in 2S1/2S2's CT-side sockets, 0.75 both
    links of 2S1/2S2 open with the bridge in, 1.0 the after-release tug check passed); 0 if a 2S link was opened
    without the bridge in, a red or blue circuit's link was opened or its terminals touched, or the bridge was seated
    in any other pair of sockets."""
    report = STATION.report or {}
    if any(report.get("semantic_events", {}).values()):
        return 0.0
    return report.get("milestone", 0) / 100.0
