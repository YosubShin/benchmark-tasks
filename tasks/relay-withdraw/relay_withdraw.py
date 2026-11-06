"""A grey steel panel stands on the station table 38 cm in front of the arms with two protection relays flush-mounted
in it side by side, their fronts facing the robot: the right one (in front of the gap between the arms) is labelled
FEEDER 2 and has a yellow marker on its label; the left one, in front of the left arm, is labelled FEEDER 1 and has a
blue marker. Each relay's plug-in unit (177 mm square front, 2.1 kg) is latched in its case by a bail handle whose
cross-bar lies across the bottom of the front. A black rubber mat on a low stand is in front of the FEEDER 2 relay. Lift
the FEEDER 2 unit's handle to release the latch (the unit pops out about 7 mm), pull the unit out by the handle with
one hand while the other hand goes under the body to carry its weight, set it on the black mat and let go. Do not touch
the FEEDER 1 relay and do not reach into the empty case.

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. The control run is ``--policy relay-demo
-P control=true``: the robot stays at the ready pose while the task pulls the FEEDER 2 unit by its body with the
handle down (it must stay latched), then lifts the handle by a torque and pulls again (it comes out).
"""

from __future__ import annotations

from relay_sim import station as hb


def _props():
    from relay_sim import task as rt

    return [rt.Panel(), rt.Unit(0), rt.Unit(1)]


def _task():
    from relay_sim.task import RelayTask

    return RelayTask(default_seed=0)


def _test_phase(view, task):
    st = task.state(view)
    if bool(st.force_ctl[0]):  # control run: end the episode once the forced sequence is over (on camera)
        if int(st.ctl_phase[0]) >= 5:
            print("[relay] control report", task.episode_report(view, 0), flush=True)
            return "done"
        return "idle"
    phase = int(st.phase[0])
    if phase >= 2:
        print("[relay] after-release check", task.episode_report(view, 0), flush=True)
    return "idle" if phase == 0 else ("done" if phase >= 2 else "running")


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task, objects=("relay_panel", "relay_unit_0", "relay_unit_1"),
    pose_modules=("relay_sim.task",), test_phase=_test_phase,
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """Milestone / 100 (0.25 the FEEDER 2 handle lifted past 45 deg, 0.5 the unit cam-ejected 6 mm by its handle so
    the connectors part, 0.75 the unit wholly out of its case, 1.0 set upright on the mat, released, and the
    after-release check passed); 0 if the FEEDER 1 unit was unlatched or moved, the FEEDER 1 relay touched, a finger
    went into the empty case, or the FEEDER 2 unit dropped."""
    report = STATION.report or {}
    if any(report.get("semantic_events", {}).values()) or any(report.get("contact_events", {}).values()):
        return 0.0
    return report.get("milestone", 0) / 100.0
