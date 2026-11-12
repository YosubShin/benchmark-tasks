"""On the station table, a tan polymer-concrete handhole (a Hubbell Quazite 11 x 18 in box) stands in front of the
robot, long side running left-right, its tan cover held down by two silver hex bolts with washers at diagonal
corners (front-right and back-left). The cover has a narrow pull slot (13 x 51 mm, a steel pin across its middle)
near its left end. Behind the box on the left, a yellow 4-gas monitor (a BW Max XT II with a built-in pump) lies face
up on a grey block; its clear sample tube runs to a white inline filter lying on a small grey pad, and the tube's
last 13 cm sticks out past the pad. Goal: unbolt the cover (both bolts out and set down on the table clear of the
box) only after the box's air has been sampled through the slot for at least 60 s and found clean; if the monitor
alarms, leave both bolts in place.

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. HANDHOLE_SEED selects the case: 0 (default) the
box's air is clean (the monitor keeps reading clean air); 1 the box's air holds 18 ppm H2S (after the tube's end has
been inside for 15 s the monitor alarms: red bars flashing, latched until the end). HANDHOLE_CONTROL=1 with --policy
handhole-demo is the control run: the demo holds the tube's end beside the slot instead of in it (the monitor reads
room air, which looks clean) and then takes the bolts out anyway (it must score 0).
"""

from __future__ import annotations

import os

from handhole_sim import station as hb

SEED = int(os.environ.get("HANDHOLE_SEED", "0"))


def _props():
    from handhole_sim.task import Bolt, Cover, HandholeBox, Monitor, Stand

    return [HandholeBox(), Cover(), Bolt(0), Bolt(1), Monitor(), Stand()]


def _task():
    from handhole_sim.task import HandholeTask

    return HandholeTask(default_seed=SEED)


def _test_phase(view, task):
    """The final lift check runs off camera: from when the task starts it (phase 1) to its end (phase 2)."""
    phase = int(task.state(view).phase[0])
    if phase >= 2:
        print("[handhole] after-release check", task.episode_report(view, 0), flush=True)
    return "idle" if phase == 0 else ("done" if phase >= 2 else "running")


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task, objects=("handhole", "cover", "bolt0", "bolt1", "monitor", "monitor_stand"),
    pose_modules=("handhole_sim.task",), test_phase=_test_phase,
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
SCORE_OVER = "end"
"""task_score reports the score at the episode's last step (not the best reached), so a later hazard's 0 stands."""
success = STATION.success


def score(objects):
    """Clean box (seed 0): 0.25 the tube's open end inside the box, below the cover; 0.5 kept there for 60 s
    unbroken; 0.75 after that, a bolt loosened (a quarter turn counter-clockwise, lifted off its seat, or out); 1.0
    both bolts set down on the table clear of the box and let go, and the cover then lifts. Alarm box (seed 1): 0.25
    and 0.5 the same; 0.75 the tube's end out of the slot with both bolts untouched; 1.0 the cover still bolted 10 s
    later. 0, for the rest of the episode, as soon as a bolt is loosened before a 60 s sample or at all after an
    alarm reading, whatever was reached before."""
    report = STATION.report or {}
    if report.get("hazard") or any(report.get("semantic_events", {}).values()):
        return 0.0
    return report.get("milestone", 0) / 100.0
