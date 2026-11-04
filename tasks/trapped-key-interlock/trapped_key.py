"""On the station table stand two trapped-key interlock pairs. Pair A1 (yellow labels, left): a molded-case breaker
lying face up, 40 cm left of the midline, with a brass Kirk Type F interlock bracketed beside its toggle handle and
key A1 trapped in it; and a grey fuse-compartment door (labelled FUSE COMPT. A1) standing 34 cm in front of the
arms, just left of the midline, held shut by a second Kirk Type F on the table in front of its lower right corner.
Pair B1 (blue labels, right) is the same: breaker B1 with key B1 trapped in its interlock 40 cm right of the
midline, door B1 just right of the midline. Both breakers are open (handles at OFF). Keys are coded: key A1 fits
only A1 locks, key B1 only B1 locks. Turn key A1 a quarter turn in breaker A1's interlock (its bolt then locks the
breaker open and the key comes free), take it to door A1's interlock, insert it and turn it a quarter turn back to
unlock the door. Use only the locks marked A1.

Task file for yam-isaac-lab-arena (commit 0059d02); see README.md. TRAPPED_KEY_CONTROL=1 forces the final check at
3 s with key A1 still in breaker A1 (a control run: it must score 0). TRAPPED_KEY_CONTROL=wrong_key checks the
coding without the robot: key A1 is set over door B1's keyway and key B1 over door A1's, bits lined up with the
notches, and each pushed straight down with 5 N for 2 s; it prints how far each went in (a matching key seats
15.7 mm below the face; a wrong one stops about 8 mm down, its bit still in the face plate) and the plugs' angles,
then ends (score 0: wrong-code locks were tried).
"""

from __future__ import annotations

import os

from interlock_sim import station as hb

MODE = os.environ.get("TRAPPED_KEY_CONTROL", "")
CONTROL = MODE in ("1", "wrong_key")
TARGET = "A1"


def _props():
    from interlock_sim.task import all_props

    return all_props()


def _task():
    from interlock_sim.task import TrappedKeyTask

    return TrappedKeyTask(target=TARGET)


def _test_phase(view, task):
    phase = int(task.state(view).phase[0])
    if phase >= 4 and not getattr(task, "_printed", False):
        task._printed = True
        print("REPORT", task.episode_report(view, 0), flush=True)
    return "idle" if phase == 0 else ("done" if phase >= 4 else "running")


def _wrong_key(view, task, k):
    """Wrong-key check (no robot): each key over the other code's door lock, pushed down."""
    import math

    import torch

    from interlock_sim.task import door_lock
    from interlock_sim.usd import props

    st = task.state(view)
    origins = view.scene.env_origins
    pairs = (("A1", "B1"), ("B1", "A1"))
    if k == 1:
        for key, lock in pairs:
            x, y, z = door_lock(lock)
            q = (0.0, 0.0, 0.0, 1.0)  # yaw 0: the bit points +y, at the door lock's notch (plug at 90 deg)
            pose = torch.tensor([[x, y, z + 0.002, *q]], device=view.device) 
            pose[:, :3] += origins[:1]
            st.keys[key].write_root_pose_to_sim(pose, env_ids=torch.tensor([0], device=view.device))
            st.keys[key].write_root_velocity_to_sim(torch.zeros(1, 6, device=view.device),
                                                    env_ids=torch.tensor([0], device=view.device))
            wrench = torch.zeros(1, 1, 3, device=view.device)
            wrench[..., 2] = -5.0
            st.keys[key].permanent_wrench_composer.add_forces_and_torques_index(
                forces=wrench, torques=torch.zeros_like(wrench), body_ids=[0], env_ids=torch.tensor([0], device=view.device),
                is_global=True)
    if k == round(2.0 / view.step_dt):
        for key, lock in pairs:
            tip = st.keys[key].data.root_pos_w.torch[0] - origins[0]
            depth = (door_lock(lock)[2] - float(tip[2])) * 1000
            plug = math.degrees(float(st.door[lock].q("plug_joint")[0]))
            print(f"WRONG KEY {key} in door {lock}'s lock: tip {depth:.1f} mm below the face (fully home: "
                  f"{-props.KEY_TIP_IN_L:.1f}), plug {plug:.1f} deg (locked: 90), bolt "
                  f"{-float(st.door[lock].q('bolt_joint')[0]) * 1000:.1f} mm", flush=True)
            st.keys[key].permanent_wrench_composer.reset(env_ids=torch.tensor([0], device=view.device))
        task.force_function_test(view)


def _control(view, task):
    k = int(task.state(view).scorer.step[0])
    if MODE == "wrong_key":
        _wrong_key(view, task, k)
    elif MODE == "1" and k == round(3.0 / view.step_dt):
        task.force_function_test(view)


STATION = hb.install(hb.StationSpec(
    props=_props, task=_task,
    objects=("breaker_A1", "door_A1", "key_A1", "breaker_B1", "door_B1", "key_B1"),
    pose_modules=("interlock_sim.task",), on_step=_control, test_phase=_test_phase,
    demo=(lambda: {}) if CONTROL else None,
))
SCENE, OBJECTS, SUCCESS_HOLD_S = STATION.SCENE, STATION.OBJECTS, STATION.SUCCESS_HOLD_S
success = STATION.success


def score(objects):
    """Milestone / 100 (0.25 breaker A1 locked open by key A1, 0.5 key A1 out, 0.75 key A1 turned in door A1's lock,
    1.0 the final check passed); 0 if a key was brought to a lock of the other code, interlock B1 turned, a bolted
    door tried, or a breaker handle moved toward ON."""
    report = STATION.report or {}
    if any(report.get("semantic_events", {}).values()):
        return 0.0
    return report.get("milestone", 0) / 100.0


if STATION.demo is not None:
    demo = STATION.demo
