"""A plywood panel stands at the far edge of the station table with a grey Bristol Maid CYB/DE2 steel wall bracket for
two D/E medical cylinders screwed to it, its plate 33 cm above the table. A small aluminium medical oxygen cylinder
(size MD, green shoulder, post valve with a black toggle on top) stands upright in the bracket's left cradle,
centred on the robot's midline 36 cm in front of the arms; nothing holds it yet. The bracket's zinc-plated chain hangs
from its hole in the tongue to the cylinder's right down to the table, its end link in a red sleeve; under the
plate's end on the cylinder's left hangs the bracket's J-hook, its tip turned up just in front of the plate. Goal:
the cylinder secured to the bracket with its chain so that it cannot tip over. Constraints: keep the cylinder upright
the whole time, do not touch its valve, finish with both grippers away from the cylinder and the chain.

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. ``CYL_CHAIN_CONTROL=<seconds>`` forces the
after-release check that many seconds into the run whatever the robot has done (control runs).
"""

from __future__ import annotations

import os

from cylchain_sim import station as hb

CONTROL_AT_S = float(os.environ.get("CYL_CHAIN_CONTROL", "0") or 0)


def _props():
    from cylchain_sim.task import all_props

    return all_props()


def _task():
    from cylchain_sim.task import CylinderChainTask

    return CylinderChainTask()


def _test_phase(view, task):
    st = task.state(view)
    phase = int(st.phase[0])
    if phase >= 2 and not getattr(task, "_printed", False):
        task._printed = True
        print("[cylinder-chain] REPORT", task.episode_report(view, 0), flush=True)
    return "idle" if phase == 0 else ("done" if phase >= 2 else "running")


def _control(view, task):
    if CONTROL_AT_S > 0:
        st = task.state(view)
        if int(st.scorer.step[0]) == round(CONTROL_AT_S / view.step_dt):
            task.force_test(view)


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task, objects=("bracket", "cylinder", "chain"),
    pose_modules=("cylchain_sim.task",), on_step=_control, test_phase=_test_phase,
    physics_hz=float(os.environ.get("CYL_CHAIN_PHYSICS_HZ", "480")),
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """Milestone / 100 (0.25 chain across the cylinder's front, 0.5 a link in the J-hook, 0.75 also in the upper
    third of the cylinder, snug and the cylinder seated, 1.0 the after-release tip check passed); 0 if the cylinder
    tilted past 10 degrees or its valve was touched."""
    report = STATION.report or {}
    if any(report.get("semantic_events", {}).values()):
        return 0.0
    return report.get("milestone", 0) / 100.0
