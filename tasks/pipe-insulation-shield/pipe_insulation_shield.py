"""A 1.24 m length of 1/2 in copper water pipe in 1 in fiberglass insulation (white jacket) runs left to right in
front of the arms, resting in two B-Line B3100 clevis hangers that hang on 1/2 in rods from an overhead strut: hanger
A in the middle, with the insulation lying directly on its strap, and hanger B to the right, which already has its
galvanized B3151 insulation protection shield in. The loose shield for hanger A (a 12 in long half-round of 18 ga
sheet with a red centre line on its outside) lies open side up, balanced on a wooden post in front of the pipe, a little to the right.
Goal: the shield in hanger A under the pipe, centred on the hanger, the pipe resting on it. Constraints: lift the pipe
no more than 2 cm, do not shift it along its run, do not drop the shield, let go when done. The usual way (not given
to the robot): raise the pipe a little near the hanger with one hand, slide the shield in under it along the pipe
with the other, centre it, lower the pipe onto it.

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md.
"""

from __future__ import annotations

import os

from pipeshield_sim import station as hb


def _props():
    from pipeshield_sim.task import all_props

    return all_props()


def _task():
    from pipeshield_sim.task import PipeShieldTask

    return PipeShieldTask()


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task, objects=("rig", "pipe", "shield"), pose_modules=("pipeshield_sim.task",),
    physics_hz=float(os.environ.get("PIS_PHYSICS_HZ", "480")),
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """The reported score (pipeshield_sim.scorer.Scorer): 0.33 the shield under the pipe in hanger A, 0.67 also
    centred on the hanger within 10 mm, 1.0 the release check passed (hands clear for 1 s, the shield seated on the
    strap, the pipe resting on it); 0 for the whole trial once the pipe has been lifted more than 20 mm at the hanger,
    shifted along its run more than 15 mm, or the shield dropped, whatever was reached before or after."""
    report = STATION.report or {}
    return float(report.get("score", 0.0))
