"""On the station table stands a 2.5 lb stored-pressure extinguisher (Amerex B417 size: red cylinder 7.6 cm across,
39 cm tall, chrome valve with a short nozzle horn pointing away from the robot, the carry handle and squeeze lever
reaching back toward the robot, a ring pin with a green tamper seal through the valve's ears). Far out on the table,
beyond a yellow tape line, a small fire burns in a steel pan on a 24 cm stand: three flame tongues side by side.
Goal: the fire put out, then the spray stopped. Constraints: no part of either arm, either gripper or the extinguisher
crosses the taped line; the extinguisher is not dropped.

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. FIRE_CONTROL=tips is the control run's tag
(the demo then aims at the flames' tips instead of their base; select it with ``-P aim=tips`` on the demo policy).
"""

from __future__ import annotations

from fire_sim import station as hb


def _props():
    from fire_sim.task import all_props

    return all_props()


def _task():
    from fire_sim.task import FireTask

    return FireTask()


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task,
    objects=("extinguisher", "fire_stand", "flame_0", "flame_1", "flame_2", "spray"),
    pose_modules=("fire_sim.task",),
    physx={"friction_correlation_distance": 0.0005, "friction_offset_threshold": 0.002},
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """milestone / 100 (0.25 pin out, 0.5 discharging, up to 0.9 with knockdown at the flames' base, 0.9 fire out,
    1.0 fire out and the lever then released for 1 s); 0 after any hazard (inside the tape line, extinguisher
    dropped, emptied with the fire still burning)."""
    report = STATION.report or {}
    return float(report.get("score", 0.0))
