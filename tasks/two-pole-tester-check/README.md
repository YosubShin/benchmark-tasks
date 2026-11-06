# Check a two-pole tester on a proving unit before and after testing two terminals

A vertical mounting plate stands in front of the robot. On it hang a yellow Fluke PRV240 proving unit (a small battery
source made for checking voltage testers) and a strip of five Phoenix Contact UT 4 terminal blocks, L1 L2 L3 N PE, on a
DIN rail. A Fluke T150 two-pole tester lies in a tray: the yellow main unit (its probe, LEDs and display) in front of the
left arm and the separate black L1 probe in front of the right arm. The robot must do the electrician's
"check, test, check" routine with one probe in each hand: push both probe tips into the proving unit's two recessed
contacts and see the tester light, push them into the top screw holes of L1 and N, then check on the proving unit
again. If the tester does not light on the proving unit, it must pull back and stop without touching the terminals.

The terminal strip is connected to nothing, in the simulator and in the real setup: the tester shows nothing there. The
only source is the PRV240 (four AA batteries; its output switches on only while two probe tips are pressed into its
contacts). Seed 1 reproduces a failed check safely: take the PRV240's batteries out, and neither it nor the tester
lights.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file
(`two_pole_tester_check.py`) in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream`
policy. The tester's lights and the check order need per-step task logic that the one-file format cannot declare, so
the file uses `tester_sim` (this folder), which adds the props and task to the station at run time
(`tester_sim/station.py` documents exactly what it patches; it requires yam-isaac-lab-arena at commit `0059d02`, the
Brev launchable's pin).

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/two-pole-tester-check --no-deps       # registers the policy tester-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
TESTER_SEED=0 inspect-robots run --embodiment yam_arena --policy dream -P record=false \
  -E task=tasks/two-pole-tester-check/two_pole_tester_check.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Pick up the yellow tester with your left hand and its black probe with your right hand. Check the tester on the yellow proving unit on the panel: push both probe tips into its two holes and see that the tester's lights come on. Then check terminals L1 and N: push one tip into the top screw hole of the grey terminal with the red marker (L1) and the other into the blue terminal (N). Then check the tester on the proving unit again. If the tester's lights do not come on at the proving unit, pull both tips back and stop: do not touch the terminals." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/tester-check-dream
```

The scripted demo (closed loop: it reads the two tools' poses and whether the tester is lit from the simulator, like a
scripted demo with privileged state, and sends the same 14-D joint action a policy does, through yam_arena's IK):

```bash
TESTER_SEED=0 inspect-robots run --embodiment yam_arena --policy tester-demo \
  -E task=tasks/two-pole-tester-check/two_pole_tester_check.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/tester-check-demo
```

`task_score` prints the trial's score (1.0 on success, otherwise the partial credit in task.json). Videos:
`inspect-robots video <log>.json` writes one MP4 per camera.

The demo takes the main unit with arm A and the L1 probe with arm B, each hand rolled 90 degrees so the fingers
close top and bottom on the tool's grip (closing across the main unit's width let it pivot nose-down in the fingers
under yam_arena's gains). Both tools are held level and yawed toward each other (22 and 16 degrees) so the tips meet
each pair of contacts without the tools touching. For each contact, one arm at a time brings its tip 15 mm in front
of the recess, measures the tip and corrects (the held tools sag up to 2.8 cm at the first standoff), corrects again
1.5 mm in front, and pushes straight in; if the tip did not go in, it backs out and tries again. It reads the tester
with both tips in, then withdraws. On seed 1 it withdraws 12 cm after the failed check and holds still.

- **Seeds (both scored).** One fixed layout (`-E object_jitter=0`). The environment variable `TESTER_SEED` picks the
  case; Inspect Robots' `--seed` is not used.
  - `TESTER_SEED=0` (default): the proving unit works. Full score needs check, test L1-N, check.
  - `TESTER_SEED=1`: the proving unit's batteries are out, so nothing lights on it. Full score needs the robot to
    try the check, then pull both tips back at least 8 cm and stay clear for 5 s, never touching the strip (the
    correct stop).
- **Episode end.** The episode ends when the score reaches 1.0, otherwise at `--max-steps`.
- **Control run.** `TESTER_SEED=1 TESTER_CONTROL=1` with `--policy tester-demo`: after the failed check the demo goes
  on to L1 and N anyway. It must score 0.
- **Runtime per trial**: see "Measured" below; Isaac Sim's first start on a new machine adds several minutes.

## Measured

RTX-class GPU shared with several other simulators, Isaac Sim shader cache warm, `-E object_jitter=0`, 20 Hz,
320x176, 240 Hz physics:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `tester-demo`, `TESTER_SEED=0` | `task_score: 1`. Both tools lifted by 14 s; the tester lit on the proving unit by 44 s (tips 2.9-3.2 mm into the recesses); both tips in L1 and N by 72 s, dark; lit on the proving unit again at 95.6 s, which ended the episode. | 1,912 (96 s) | about 21 minutes |
| `tester-demo`, `TESTER_SEED=1` (the correct stop) | `task_score: 1`. Both tips in the proving unit by 44 s, nothing lit; withdrawn 12 cm by 48 s; still clear 5 s later (52.9 s). The strip was never approached. | 1,058 (53 s) | 13 minutes |
| Control: `TESTER_SEED=1 TESTER_CONTROL=1` (carries on after the failed check) | `task_score: 0`. The rule fired at about 56 s as arm A's tip reached L1. | 2,400 (step limit) | 27 minutes |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`), `TESTER_SEED=0` | `task_score: 0`. The instruction was accepted. Both arms rose and reached toward the plate, the right hand near the proving unit; neither tool was picked up from the tray. | 2,400 (step limit) | 26 minutes (Dream reported an observation as expired and was asked again 49 times) |

The wall times are long because the GPU was shared; the demo's own cost is two arms' Python IK every control step.
`demo.mp4` is the seed 0 demo, `demo_seed1.mp4` the seed 1 demo, `dream1.mp4` the Dream-1 run (top camera on top,
the left and right wrist cameras below).

## Layout

| Path | What |
|---|---|
| `two_pole_tester_check.py` | the task file: description for agents, seed selection, the props and task, scoring |
| `tester_sim/station.py` | the run-time extension of yam_arena (props and per-step task events) |
| `tester_sim/task.py`, `scorer.py`, `spec.py` | the seating rule, the tester's lights, the score levels, the layout and instruction |
| `tester_sim/assets/*.usda`, `tester_sim/usd/` | the props, and the script that generates them (`python -m tester_sim.usd.tester_props tester_sim/assets`) |
| `tester_sim/demo.py`, `tester_sim/policy.py` | the scripted demo and its Inspect Robots policy `tester-demo` |

## References

- Fluke T90/T110/T130/T150 instruction sheet (how to hold the probes, self-test "before and after use" on a known
  source, size 255 x 78 x 35 mm, 0.27 kg): https://docs.rs-online.com/fe16/A700000015028665.pdf
- Fluke T150 product page (photos: docked probes, LED bargraph and LCD, testing terminal blocks):
  https://www.fluke.com/en-gb/product/electrical-testing/basic-testers/fluke-t150
- Fluke PRV240 proving unit, product page and instruction sheet (240 V battery source, contacts activated when probes
  are inserted, one LED, TPAK magnetic strap): https://www.fluke.com/en-gb/product/electrical-testing/basic-testers/prv240-proving-unit,
  https://media.fluke.com/225f8a4b-c633-4ae6-b9cd-b10800c07f13_original%20file.pdf
- Fluke PRV240 data sheet (four AA batteries, 117 x 74 x 28 mm, 0.23 kg):
  https://media.fluke.com/c0a08480-17de-46ca-b827-b1060068a0d4_original%20file.pdf
- Phoenix Contact UT 4 feed-through terminal block, 3044102 (6.2 mm wide, 47.7 mm long, 47.5 mm high on NS 35/7.5,
  M3 screws): https://www.phoenixcontact.com/en-pc/products/feed-through-terminal-block-ut-4-3044102
- NFPA 70E, verifying the absence of voltage (the test instrument is checked on a known voltage source before and
  after each test): https://www.nfpa.org/codes-and-standards/nfpa-70e-standard-development/70e
- HSE GS38, electrical test equipment for use on low voltage systems (proving units, GS38 probe tips):
  https://www.hse.gov.uk/pubns/gs38.htm
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX at 240 Hz in Isaac Lab-Arena, both arms driven as in Dream's station (joint PD with i2rt's gains and gravity compensation; fingers 5000 N/m, 50 N). The T150's main unit and L1 probe are two free rigid bodies; collision shapes are boxes (a round tip of diameter d is a d x d square). Published: the T150's docked size and weight, the PRV240's size and weight, the UT 4's 6.2 mm pitch, 47.7 mm length and 47.5 mm height, the DIN rail. Inferred from product photos, to be measured on real parts (no CAD or dimensioned drawing was found): the split of the T150's 0.27 kg between its two parts, their body sections and lengths, the GS38 tip (2 mm, 4 mm bare) and the 5 mm nose behind it; the PRV240's contact recesses (5.5 mm square, 4 mm deep, 24 mm apart, 16 mm above its bottom edge); the UT 4 screw wells (4.5 x 5 mm, 4 mm deep). A tip reaches a contact only by entering its recess (collision walls with 0.75 mm clearance per side, the nose stops it at about 3 mm); 'in a contact' is a rule on that geometry (the tip's end 2.5 mm past the mouth and inside the opening). Electrical behaviour is a rule copying the manuals: the tester's 12-230 V LEDs and the PRV240's LED light while one tip is in each PRV240 contact and the PRV240 has batteries; the strip is connected to nothing and never lights it. Not modelled: the force needed to switch the PRV240's contacts on ('press down firmly'), the T150's LCD reading and vibration, the lead between the two parts (about 1 m, it would drag on the table). Simplified for the YAM's reach: the PRV240 and the strip's top wells are at one height (16.5 cm) facing the robot, and the tools are held level and yawed toward each other (22 and 16 degrees) rather than tilted freely as by hand; the PRV240 is rigidly fixed, whereas a real one on its magnet can be pushed off or swing (hence taped in the real setup); the tools start in an invented tray instead of docked together with a cap over the tips; only L1-N is tested, while a real dead test on a three-phase strip covers every phase-phase and phase-N/PE pair. At 320 x 176 the top camera is about 0.7 m from the strip and each 6.2 mm block is 1-2 pixels wide, so printed terminal names cannot be legible with real-size parts; L1 carries a red marker card (coloured markers are an option of the same marker system) and N is the standard blue block, which shows as a red and a blue pixel column at the strip's left and middle; the names and colours are clear in the wrist cameras near the plate. Seed 1's failure (nothing lights on the proving unit) is reproduced in the real setup by removing the PRV240's batteries; a broken tester lead, flat tester battery or faulty tester shows the same way, and the response (stop) is the same.
