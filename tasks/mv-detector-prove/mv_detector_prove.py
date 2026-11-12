"""On the station table, a red HD Electric PRX detector mounted on a yellow fiberglass stick lies across the front of
the table in a grey cradle, face up, its top (sensing) end toward the right arm and its stick toward the left arm.
In front of the right arm a black PT-PRX proof tester lies face up in a cradle, its metal end plate facing the
detector's top end and its red TEST button on top. Beside the tester, farther from the robot, a bare metal bar
(an aluminium conductor stub connected to nothing) runs away from the robot on two grey posts, at the height of the
detector's top end. The goal: test the bar with the detector's top end, with the detector shown to work on the tester
(its red lights flashing there) before and after the test, never near the bar before that, and stop if it does not
flash on the tester. (The scoring follows the manuals' procedure: ON and the 3 s self-test, TEST held with the top end
on the end plate, the bar, the tester again; see task.json `context` and `scoring`.)

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. DETECTOR_SEED selects the layout: 0 (default)
everything works; 1 the tester's battery is flat, so the prove fails and the right response is to stop.
DETECTOR_CONTROL=1 with --policy detector-demo is the control run: on a failed prove the demo carries on to the bar
anyway (it must score 0).
"""

from __future__ import annotations

import os

from detector_sim import station as hb

SEED = int(os.environ.get("DETECTOR_SEED", "0"))


def _props():
    from detector_sim.task import Bench, Detector

    return [Bench(), Detector()]


def _task():
    from detector_sim.task import DetectorTask

    return DetectorTask(default_seed=SEED)


def _debug_log(view, task):
    """DETECTOR_DEBUG=1: print the simulator's view every second (for checking the demo's perception; the policy
    never sees it)."""
    st = task.state(view)
    k = int(st.scorer.step[0])
    if k % 20 or not st.now:
        return
    nw = st.now
    tip = nw["tip"][0].cpu().numpy()
    q = st.det.data.root_quat_w.torch[0].cpu().numpy()
    r = task.episode_report(view, 0)
    print(f"TRUTH k={k} tip={tip.round(4).tolist()} quat={q.round(4).tolist()} on={bool(nw['powered'][0])} "
          f"selftest={bool(nw['selftest_done'][0])} test={bool(nw['test_pressed'][0])} "
          f"plate={bool(nw['on_plate'][0])} along={float(nw['along'][0]) * 1000:.1f} cond={bool(nw['on_cond'][0])} "
          f"c_along={float(nw['c_along'][0]) * 1000:.1f} lit={bool(nw['lit'][0])} ms={r['milestone']} "
          f"ev={r['semantic_events']}", flush=True)


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task, objects=("bench", "detector"), pose_modules=("detector_sim.task",),
    on_step=_debug_log if os.environ.get("DETECTOR_DEBUG") == "1" else None,
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """Tester works (seed 0): 0.25 self-test done, 0.5 first prove (the detector flashed on the tester for 1 s),
    0.75 the bar touched for 1 s after that prove, 1.0 proved again after the bar. Tester flat (seed 1): 0.25
    self-test done, 0.5 prove attempted (sensing end on the end plate with TEST held for 1.5 s, no flash), 0.75 the
    sensing end withdrawn 8 cm from the tester and the bar, 1.0 still clear 5 s later. 0 if the sensing end came
    within 3 cm of the bar before a passed prove."""
    report = STATION.report or {}
    if any(report.get("semantic_events", {}).values()):
        return 0.0
    return report.get("milestone", 0) / 100.0
