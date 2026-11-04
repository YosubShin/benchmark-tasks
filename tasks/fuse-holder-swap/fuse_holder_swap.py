"""On the station table, a light-grey steel back pan stands upright 0.33 m in front of the arms, midway between them,
with a DIN rail across it. Three grey Eaton Bussmann CHM1DU modular fuse holders (F1, F2, F3 from the robot's left,
red grip tabs on top, one blank module between neighbours) clip on the rail, each with a legend plate above it
(tag, circuit and rating, the rating on a colour-coded field: F1 2 A pink, F2 6 A green, F3 10 A red). F2 holds a blown
10 x 38 mm fuse marked with a black X. To the right of the holders two spare fuses lie on a dark-grey stand with
rating labels in front of their cradles (10 A red nearer the middle, 6 A green further right); a blue parts tray
sits on the table further right and nearer the robot. Open F2's carrier (it hinges at the bottom and swings forward
and down), lift the blown fuse out and drop it in the tray, lay the 6 A spare into the open carrier, and push the
carrier shut until it latches. Do not open F1 or F3.

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. FUSE_CONTROL=1 with ``--policy fuse-demo`` stops
the demo's closing push 20 degrees short of the latch and forces the hold test (a control run: it must score
below 1).
"""

from __future__ import annotations

from fuse_sim import station as hb

def _props():
    from fuse_sim.task import Fuse, Holders, Stands

    return [Holders(), Fuse("old"), Fuse("spare_a"), Fuse("spare_b"), Stands()]


def _task():
    from fuse_sim.task import FuseHolderTask

    return FuseHolderTask(default_seed=0, reviewer_camera=False)


def _test_phase(view, task):
    phase = int(task.state(view).phase[0])
    return "idle" if phase == 0 else ("done" if phase >= 2 else "running")


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task, objects=("holders", "fuse_old", "fuse_spare_a", "fuse_spare_b", "stands"),
    pose_modules=("fuse_sim.task",), test_phase=_test_phase,
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """Milestone / 100 (0.25 F2's carrier opened past 30 degrees, 0.5 the blown fuse out of it, 0.75 the 6 A spare
    seated in it, 1.0 shut on it with the blown fuse in the tray and holding a 4 N pull at the grip tab); 0 if F1 or F3
    was opened 5 degrees or more, or the 10 A spare was seated in F2."""
    report = STATION.report or {}
    if any(report.get("semantic_events", {}).values()):
        return 0.0
    return report.get("milestone", 0) / 100.0
