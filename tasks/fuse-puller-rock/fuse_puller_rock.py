"""On the station table, a small wall-mount control enclosure (Rittal AE 1036.500, 300 x 300 x 155 mm) lies on its back
in front of the arms, opening up, its bottom end toward the robot and its door swung open to the left. Inside, three
black Eaton Bussmann BM6031B one-pole midget fuse blocks are snapped side by side on a DIN rail on the mounting plate,
their fuses' axes pointing away from the robot, each with a 10 x 38 mm fuse held in two open spring clips and a 14 AWG
wire landed on each end's box lug (black line wires into a wire duct beyond the blocks, red load wires into a duct
between the blocks and the robot). A colour-coded circuit label on the far duct's cover above each block names it
(F1 PLC 24V DC white, F2 CONTROL XFMR orange, F3 CABINET LIGHT green, from the robot's left). A yellow nylon tong fuse
puller stands upright, jaws down, in a grey holder on the bench right of the enclosure; a blue parts tray sits on the
bench further right, nearer the robot. Pick up the puller by its notched handles, lift it over the enclosure's wall,
hold it perpendicular to the fuse (pointing down into F2's block) and close its jaws across F2's body between the
clips, tip the handles away from the robot so the far clip is the pivot and the near end of the fuse levers out of
its clip first, then lift the far end out, carry the fuse in the jaws out of the enclosure and drop it in the tray.
Do not disturb F1, F3 or the wiring. Pulled straight up, a fuse needs both clips' grip at once (about 70 N).

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. FPR_CONTROL=1 with ``--policy fuse-puller-demo``
makes the demo let go of the handles, pinch the puller's smooth arms just above its jaws and pull the fuse straight up
instead of rocking it (a control run: it must score below 1).
"""

from __future__ import annotations

from fuse_puller_sim import station as hb


def _props():
    from fuse_puller_sim.task import Blocks, Fuse, Puller, Stands

    return [Blocks(), Fuse(0), Fuse(1), Fuse(2), Puller(), Stands()]


def _task():
    from fuse_puller_sim.task import FusePullerTask

    return FusePullerTask(default_seed=0, reviewer_camera=False)


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task, objects=("blocks", "fuse_0", "fuse_1", "fuse_2", "puller", "stands"),
    pose_modules=("fuse_puller_sim.task",), success_hold_s=0.5,
    # Friction anchors per contact point, not merged across a whole jaw (PhysX's 25 mm default makes one patch of
    # the 16 mm jaw, which then cannot hold the fuse against turning); 480 Hz for the stiff clip contacts.
    physics_hz=480.0, physx={"friction_correlation_distance": 0.0005, "friction_offset_threshold": 0.002},
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """The score computed by fuse_puller_sim.scorer.PullScorer (report()["score"]): milestone / 100 (0.25 the puller
    picked up, 0.5 its jaws closed on F2's body, 0.75 F2 out of both clips while held in the jaws, 1.0 F2 lying in the
    tray); at most 0.5 from the first step at which F2 is out of both clips without the puller's jaws on it, even if
    the puller grips it afterwards; 0 from the first step at which F1 or F3 is moved from its seat, whatever milestones
    were reached before or after."""
    return float((STATION.report or {}).get("score", 0.0))
