# Put a grounding set on a tested bus bar, ground end first

A bench bus mock stands in front of the robot: a ground bar (GND) and three flat copper phase bars (A, B, C) on an
insulating back board. A yellow TESTED tag lies behind bar B, the bar that has been tested dead. A protective
grounding set (two C-type grounding clamps with eye screws, joined by a yellow ground cable) hangs ready on two
parking plates. Each clamp's eye screw carries a short stick with a square handle; turning the stick tightens the
clamp. Two simplifications matter most: the stick is a 12 cm handle, not a 1.2-1.8 m clamp stick (the arm's
fingers-down reach ends about 22 cm above the table), and the tightening torque is far below real (about 0.6-0.9 N m
through the gripper's pinch against 28 N m specified), so "tight" means a 120 N clamp force. See task.json `fidelity`. The robot must hang the ground clamp on GND and tighten it first, then hang the line clamp on bar B and tighten
it, and let go. The sourced order rule is enforced: the line end may touch a phase bar only after the ground end is
tight (OSHA 29 CFR 1910.269(n)(6)(i)), and only on the tested bar. Scoring checks the result physically: after the
robot lets go, each clamp is tugged off its bar with 40 N and must hold.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file
(`ground_clamp_order.py`) in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream` policy.
The one-file format cannot declare its props: the grounding set is one articulation (two clamps, each with a sliding
pad and a turning eye screw, and a cable of 18 jointed links). So the file uses `groundclamp_sim` (this folder), which
adds them to the station at run time. `groundclamp_sim/station.py` documents exactly what it patches. It requires
yam-isaac-lab-arena at commit `0059d02`, the Brev launchable's pin.

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/ground-clamp-order --no-deps          # registers the policy clamp-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream -P record=false \
  -E task=tasks/ground-clamp-order/ground_clamp_order.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Ground bar B, the bar with the yellow TESTED tag. First take the ground clamp (on the right, by its black stick handle), hang it on the ground bar GND and turn its stick clockwise until the clamp is tight. Only then take the line clamp (on the left), hang it on bar B and tighten it the same way. Let go of both sticks at the end. Do not touch bars A or C." \
  --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/ground-clamp-dream
```

The scripted demo is closed loop. Like a scripted demo with privileged state, it reads the clamps' poses and the clamp
force from the simulator, and it sends the same 14-D joint action a policy does, through yam_arena's IK:

```bash
inspect-robots run --embodiment yam_arena --policy clamp-demo \
  -E task=tasks/ground-clamp-order/ground_clamp_order.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/ground-clamp-demo
```

`task_score` prints the trial's score: 1.0 on success, otherwise the partial credit given in task.json. To make
videos, run `inspect-robots video <log>.json`, which writes one MP4 per camera.

- **Episode end.** Once both clamps are tight on their bars and both grippers have stayed at least 5 cm from both
  stick handles for 1 s, the tug check runs (2.5 s per clamp, 5 s of simulated time) inside that control step, off
  camera, and the episode ends. The stored frames stop at the release. If the check never runs, the run ends at
  `--max-steps`.
- **Control run.** `--policy clamp-demo -P control=true` does everything except tightening the line clamp: it hangs
  it on bar B, lets go and backs off, and the check is forced. It must score below 1.
- **Seeds.** One fixed layout (`-E object_jitter=0`). Seeded jitter is not used.
- **Runtime per trial.** See "Measured" below. Isaac Sim's first start on a new machine adds several minutes.

## Measured

RTX-class GPU shared with several other simulators, Isaac Sim shader cache warm, `-E object_jitter=0`, 20 Hz,
320x176, 240 Hz physics:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `clamp-demo` | `task_score: 1`. Arm B picked the ground clamp off its plate by the stick, carried it to GND, lined it up (closed loop on the clamp's position) and slid it on, let go, and tightened it in 5 regrasp strokes of 110 deg (1.47 turns, 331 N). Meanwhile arm A picked the line clamp and waited in front of bar B; once the ground clamp counted as tight it hung the line clamp on B and tightened it in 5 strokes (1.45 turns, about 300 N). Both hands backed off and the tug check held both clamps. No order or wrong-bar event. | 2,163 (108 s), then the 5 s check off camera | about 45 minutes |
| Control: `clamp-demo -P control=true` (line clamp hung on B, not tightened) | `task_score: 0.5`. The ground clamp was tight (331 N) and the line clamp hung on bar B; with the line clamp loose (7 N) the 0.75 level is never reached, so the forced tug check cannot raise the score | 1,591, then the forced check | about 32 minutes |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`) | `task_score: 0`. The arms moved out over the bench and back; neither clamp was picked up, and no bar was touched | 3,600 (step limit) | about 49 minutes |

The wall times are long because the GPU was shared; the demo's own cost is two arms' Python IK every control step.


## Layout

| Path | What |
|---|---|
| `ground_clamp_order.py` | the task file: description for agents, props and task, scoring |
| `groundclamp_sim/station.py` | the run-time extension of yam_arena (props, per-step task events, off-camera check) |
| `groundclamp_sim/task.py`, `scorer.py`, `spec.py` | the mechanism (screw coupling, thread friction, attach and touch tests, tug check) and milestones |
| `groundclamp_sim/assets/*`, `groundclamp_sim/usd/` | the props, and the script that generates them (`python -m groundclamp_sim.usd.clamp_props`) |
| `groundclamp_sim/demo.py`, `groundclamp_sim/policy.py` | the scripted demo and its Inspect Robots policy `clamp-demo` |
| `assumptions.md` | what is inferred rather than measured |

## References

- CHANCE PSC6003412 grounding clamp, C type, 5H (jaw range, mass, 250 in-lb eye-screw torque):
  https://www.hubbell.com/hubbellpowersystems/en/products/ground-clamp-c-type-5h/p/2764024 and spec sheet
  https://hubbellcdn.com/specsheet/CH-PSC6003412-SPEC-EN.pdf
- CHANCE PST6003426 C-type clamp spec sheet (overall dimensions of a 2 in eye-screw clamp):
  https://hubbellcdn.com/specsheet/CH-PST6003426-SPEC-EN.pdf
- CHANCE H-rated grounding brochure (recommended clamp torques, ASTM F855 grades):
  https://hubbellcdn.com/literature/CENT-LTE-H-RATED-GROUNDING-BR-EN-00301.pdf
- Salisbury C-type grounding clamps and the 1924 stick-mounted clamp (catalogue page H-9, Acme thread):
  https://jmtest.com/pictures/2855-Salisbury-1923-datasheet.pdf
- CHANCE Grip-All clamp stick C4030291: https://www.hubbell.com/hubbellpowersystems/en/products/grip-all-4-ft-1-2-m/p/293660
  and https://hubbellcdn.com/specsheet/CH-C4030291-SPEC-EN.pdf
- CHANCE 2/0 clear-jacket grounding cable S6450: https://hubbellcdn.com/specsheet/CH-S6450-SPEC-EN.pdf
- Burndy BBB14212P copper bus bar, 1/4 x 2 x 12 in: https://hubbellcdn.com/specsheet/BRDY-BBB14212P-SPEC-EN.pdf
- OSHA 29 CFR 1910.269(n), grounding for the protection of employees (test first; ground end first, then the other
  end with a live-line tool): https://www.ecfr.gov/current/title-29/subtitle-B/chapter-XVII/part-1910/subpart-R/section-1910.269#p-1910.269(n)
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX at 240 Hz in Isaac Lab-Arena, both arms driven as in Dream's station (joint PD with i2rt's gains and gravity compensation). The grounding set is one articulation: each clamp's frame, a pad on a slide along the screw axis and the eye screw with its stick on a hinge; the cable is 18 jointed capsule links (lighter than real 2/0 copper, nearly no bending stiffness). Published: the clamp's jaw range, mass and 250 in-lb (28 N m) installation torque, the bar size, the cable diameter. No CAD or dimensioned drawing of the clamp exists publicly; the frame proportions, a 5/8 in Acme screw of 1/8 in pitch and the pad size are inferred, and collision shapes are boxes. The screw thread is a rule, not geometry: the pad's position follows the screw's angle, a clamp compliance of 2e5 N/m turns overtravel into clamp force, and thread friction is a torque limit on the screw (0.015 N m + 0.2 x force x 5/8 in), so the clamp gets harder to turn as it tightens and a pull never unscrews it. Holding on the bar is contact friction (1.0 on the serrated jaw and pad). The stick is a 12 cm handle, not a 1.2-1.8 m clamp stick: the station arm's fingers-down grasp only tracks up to about 22 cm above the table, so the task keeps the stick's function (turning the screw from above) but not the distance a real stick keeps between hand and bar. The stick is fixed to the screw (the stick-mounted clamp class); a Grip-All hook that is released afterwards is not modelled. Tightening torque is far below real: the gripper's pinch on the square handle slips at about 0.6-0.9 N m, giving 150-320 N of clamp force against several kN from 28 N m, so 'tight' is set at 120 N and the 40 N tug check tells tight from loose; it is not a fault-current withstand test (ASTM F855). The clamps are used upside down on flat bars (a bench mock), and the order rule is about touch, not electricity: nothing is energised, and the TESTED tag stands for a test of absence of voltage already done. The station's top camera looks almost straight down, so the bar letters and the tag lie flat on a ledge: at 320x176 the yellow tag (about 13 x 7 px) is clearly visible behind bar B, but its text and the bar letters are not legible. The tug check is applied as forces in the simulator after the recording ends. One layout.
