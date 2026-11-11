"""On the station table a plywood wall board stands 32 cm in front of the arms with two wall toggle switches in it,
their plates 23 cm above the table: S1 3 cm right of the midline and S2 13 cm left of it. Each has an engraved label
under its plate (S1 yellow, S2 white) and a red Master Lock 496B wall switch lockout already installed on its plate,
its cover hinged at the top and standing open. Both switches are ON (toggle up). The robot's purple padlock stands
open, shackle up, on a small grey post in front of the left arm. Push S1's toggle down to OFF, swing S1's cover down
over it, put the padlock's shackle through the ring at the bottom of the cover and snap it shut. Leave S2 ON.

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. The scripted demo is the policy
``wall-switch-demo`` (``-P control=true`` turns S1 OFF and shuts its cover but skips the padlock, then forces the
after-release test: a control run that must score below 1).
"""

from __future__ import annotations

from switch_lockout_sim import station as hb


def _props():
    from switch_lockout_sim import task as wt

    return [wt.Switches(), wt.SwitchPadlock(), wt.Pedestal()]


def _task():
    from switch_lockout_sim.spec import EPISODE_SECONDS
    from switch_lockout_sim.task import WallSwitchTask

    return WallSwitchTask(default_seed=0, episode_length_s=EPISODE_SECONDS + 5.0, reviewer_camera=False)


def _test_phase(view, task):
    phase = int(task.state(view).phase[0])
    if phase >= 3 and not getattr(task, "_reported", False):
        task._reported = True
        r = task.episode_report(view, 0)
        print(f"[wall-switch] after-release test: cover opened {r['test_cover_max_open_deg']} deg, toggle reached "
              f"{r['test_toggle_max_deg']} deg (0 = middle, +30 = ON), passed {r['function_test_passed']}; milestone "
              f"{r['milestone']}, events {r['semantic_events']} {r['contact_events']}", flush=True)
    return "idle" if phase == 0 else ("done" if phase >= 3 else "running")


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task, objects=("switches", "padlock", "pedestal"),
    pose_modules=("switch_lockout_sim.task",), test_phase=_test_phase,
    demo=lambda: {},
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """Milestone / 100 (0.25 S1's toggle at OFF, 0.5 S1's cover shut over it, 0.75 the padlock's short leg through the
    cover's ring, 1.0 the padlock shut there and the lockout holding in the after-release test); 0 if S2 was turned
    OFF (its toggle pushed past the middle)."""
    report = STATION.report or {}
    if any(report.get("semantic_events", {}).values()):
        return 0.0
    return report.get("milestone", 0) / 100.0


demo = STATION.demo
