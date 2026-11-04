# Prove a stick-mounted voltage detector before and after testing a bar

A red HD Electric PRX proximity voltage detector, mounted on a short fiberglass hot stick, lies in a cradle across
the front of the table. Beside it lie its PT-PRX Proof Tester (a battery-powered source the detector is checked on)
and a bare aluminium conductor stub on two posts, the target to test. The robot must run the line worker's
prove-test-prove routine: press ON and let the detector self-test, pick it up by the stick, prove it on the tester
(the other arm holds TEST while the detector's top end touches the tester's metal end plate, and the detector must
flash), touch its top end to the target bar, and prove it again. If it does not flash on the tester, it must move
away and stop without going near the bar.

The target is connected to nothing and is never energised: in the real setup the detector stays dark on it, and
only the tester (a 9 V battery unit made for this check) makes it flash. Seed 1 reproduces a failed prove safely by
taking the tester's battery out.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file
(`mv_detector_prove.py`) in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream`
policy. The detector's buttons and lights and the tester need per-step task logic that the one-file format cannot
declare, so the file uses `detector_sim` (this folder), which adds the props and task to the station at run time
(`detector_sim/station.py` documents exactly what it patches; it requires yam-isaac-lab-arena at commit `0059d02`,
the Brev launchable's pin).

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/mv-detector-prove --no-deps           # registers the policy detector-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream -P record=false \
  -E task=tasks/mv-detector-prove/mv_detector_prove.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "INSTRUCTION" \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/mv-detector-dream
```

with INSTRUCTION the `instruction` in task.json.

The scripted demo (closed loop: it reads the detector's pose and whether it is lit from the simulator, like a
scripted demo with privileged state, and sends the same 14-D joint action a policy does, through yam_arena's IK):

```bash
inspect-robots run --embodiment yam_arena --policy detector-demo \
  -E task=tasks/mv-detector-prove/mv_detector_prove.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "INSTRUCTION" --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/mv-detector-demo
```

`task_score` prints the trial's score (1.0 on success, otherwise the partial credit in task.json). Videos:
`inspect-robots video <log>.json` writes one MP4 per camera.

The demo presses ON with arm B's closed fingers, takes the stick with arm A (fingers down, closing across it at its
balance point), lifts it, measures how the detector sits in the hand, then for each touch brings the top end 15 mm in
front of the target, measures and corrects the sideways error (the held detector sags about 2.5 cm under the arm's
gains), and pushes in along the stick until the plate or the bar stops it. Arm B holds TEST during each prove. It
reads the detector after each touch and, on seed 1, withdraws 13 cm and holds still.

- **Seeds.** One fixed layout (`-E object_jitter=0`). The environment variable `DETECTOR_SEED` picks the case:
  - `DETECTOR_SEED=0` (default): the tester works; full score needs prove, test, prove.
  - `DETECTOR_SEED=1`: the tester's battery is flat (its LED stays dark when TEST is pressed) and the detector never
    flashes on it; full score needs the robot to attempt the prove, then back away and stay clear of the bar.
  Inspect Robots' `--seed` is not used (seeded jitter is off).
- **Episode end.** The episode ends when the score reaches 1.0, otherwise at `--max-steps`.
- **Control run.** `DETECTOR_CONTROL=1 DETECTOR_SEED=1` with `--policy detector-demo`: after the failed prove the
  demo carries on to the bar anyway. It must score 0.
- **Runtime per trial**: see "Measured" below; Isaac Sim's first start on a new machine adds several minutes.

## Measured

Shared RTX-class GPU (other Isaac Sim processes running at the same time), Isaac Sim shader cache warm,
`-E object_jitter=0`, 20 Hz, 320x176 frames:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `detector-demo`, `DETECTOR_SEED=0` (`demo.mp4`) | `task_score: 1` (self-test done at about 12 s, first prove at 33 s, bar touched with no flash at 50 s, second prove at 64.9 s; the top end stopped 2.6 mm from the plate and the bar each time) | 1,298 (64.9 s) | about 13 minutes |
| `detector-demo`, `DETECTOR_SEED=1`, correct stop (`demo_seed1.mp4`) | `task_score: 1` (prove attempted, no flash, tester LED dark; withdrawn 13 cm and held clear of the bar) | 850 (42.5 s) | about 9 minutes |
| Control: `DETECTOR_CONTROL=1 DETECTOR_SEED=1`, carries on to the bar after the failed prove | `task_score: 0` (top end within 3 cm of the bar at about 41 s before any passed prove) | 2,400 (step limit) | about 23 minutes |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`), `DETECTOR_SEED=0` (`dream1.mp4`) | `task_score: 0` (the right hand moves over the detector's face and the left hand toward the stick, but the ON button is never pressed and the detector is never picked up; no hazard) | 2,400 (step limit) | about 26 minutes |

Most of the demo's wall time is Isaac Sim's start-up and its IK, which it solves in Python every control step; the
GPU was shared with several other simulations during these runs.

## Layout

| Path | What |
|---|---|
| `mv_detector_prove.py` | the task file: description for agents, seed selection, the props and task, scoring |
| `detector_sim/station.py` | the run-time extension of yam_arena (props and per-step task events) |
| `detector_sim/task.py`, `scorer.py`, `spec.py` | touching and button rules, the detector's self-test and lights, the tester, the score levels, the layout and instruction |
| `detector_sim/assets/*.usda`, `detector_sim/usd/` | the props, and the script that generates them (`python -m detector_sim.usd.detector_props detector_sim/assets`) |
| `detector_sim/demo.py`, `detector_sim/policy.py` | the scripted demo and its Inspect Robots policy `detector-demo` |

## References

- HD Electric PRX proximity voltage detector, user manual (self-test p5, use p6, size and weight p9, PT-PRX p10):
  https://www.tester.co.uk/files/get/download/id/8842/hd-electric-prx-proximity-voltage-detector-user-manual.pdf
- HD Electric PRX-69D datasheet (photo, ranges): https://www.tester.co.uk/files/get/download/id/8815/hd-electric-prx-69d-proximity-voltage-detector-datasheet.pdf
- HD Electric PT-PRX Proof Tester manual (procedure, 340 V AC 300 Hz, 9 V battery, size and weight):
  https://www.tester.co.uk/files/get/download/id/8852/hd-electric-pt-prx-proof-tester-for-prx-voltage-detectors-instruction-manual.pdf
- OSHA 29 CFR 1910.269, electric power generation, transmission and distribution (testing before treating lines as
  de-energized, 1910.269(m)(3)): https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.269
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX at 240 Hz in Isaac Lab-Arena, both arms driven as in Dream's station (joint PD with i2rt's gains and gravity compensation; fingers 5000 N/m, 50 N). The detector, spline head and stick are one rigid body (0.72 kg; the 0.31 kg detector is published, the head and the stick's 0.5 kg/m are estimates); collision shapes are boxes (the stick is a 32 mm square, the sensing head a 62 x 40 mm block, the round bar a 9.4 mm square). No CAD or dimensioned drawing exists for the PRX or the PT-PRX: overall sizes are published, the head shape, button and light positions are taken from the manual figure and photos. Touching is a geometric rule on real collisions: the tester, its cradle, the bar and its posts are static colliders, so the top end reaches the plate or the bar only by being carried there (the contact offset leaves it 2-3 mm short, inside the 3 mm rule). Pressing a button is a rule: a finger end within 2.5 mm of the 2 mm-proud cap's top. The electronics are rules copying the manuals: ON starts a 3 s self-test (range lights step up, back to the lowest, indicating lights flash), then a short blink every 4 s; on the end plate with TEST held the detector flashes continuously; the dead bar never makes it flash. Not modelled: the PRX's proximity sensing (it would light up to about 25 cm from a live line; nothing here is live), range selection (the self-test leaves it on the lowest range, the one used for proving), the beeper. Simplified for the YAM's reach: the stick is 0.62 m and worked level at table height, whereas a real stick is 1.2 m or longer and raised to an overhead line; the tester sits in a cradle and is pressed by the other arm, whereas the manual has the worker grip it in the other hand (the hand is part of its capacitive return; not yet checked: whether a real PRX lights from a cradled PT-PRX, so the real cradle should wrap the tester's body in copper foil tape wired to the robot's frame); the bar stands in for an overhead conductor on full-size insulators. Under yam_arena's gains the held detector sags about 2.5 cm at the standoff; the demo measures and corrects it closed loop. The real failure is a faulty detector or a flat battery; seed 1 reproduces it with the tester's battery removed (its LED dark), which is safe to set up on demand.
