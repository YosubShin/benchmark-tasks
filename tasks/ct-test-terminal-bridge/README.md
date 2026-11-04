# Bridge a CT circuit at its test-disconnect terminals, then open its links

A row of six test-disconnect terminal blocks (Phoenix Contact URTK/S class) carries three two-wire current-transformer
circuits, 1S1/1S2, 2S1/2S2 and 3S1/3S2, each marked in its phase colour (red, yellow, blue). The robot must push a
two-pole plug-in bridge (Phoenix KSS 8 class) into the CT-side test sockets of circuit 2's two terminals, let go, and
only then, with an insulated screwdriver, push each of those two terminals' disconnect slides open. Opening a link
before the bridge is in would open-circuit the CT secondary; that, bridging any other pair of sockets, or opening or
touching the red or blue circuits scores 0. Scoring checks the result physically: after release, the bridge must stay
in its sockets under a sideways tug and both links must stay open.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file
(`ct_test_terminal.py`) in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream` policy.
Its props are articulated (the six disconnect slides), which the one-file format cannot declare, so the file uses
`ctbridge_sim` (this folder), which adds them to the station at run time. `ctbridge_sim/station.py` documents exactly
what it patches. It requires yam-isaac-lab-arena at commit `0059d02`, the Brev launchable's pin.

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/ct-test-terminal-bridge --no-deps     # registers the policy ctbridge-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream -P record=false \
  -E task=tasks/ct-test-terminal-bridge/ct_test_terminal.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Take the black two-pin bridge from its holder and push it fully into the two test sockets nearest you on terminals 2S1 and 2S2 (the pair with yellow markers), then let go. After that, take the screwdriver from its stand and, on each of those two terminals, put the blade down the slot and push the slide at its bottom away from you until it stops. The bridge must be in before either slide moves. Do not touch the terminals with red or blue markers." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/ct-bridge-dream
```

The scripted demo is closed loop. Like a scripted demo with privileged state, it reads the bridge's and the
screwdriver's poses and the slides' positions from the simulator, and it sends the same 14-D joint action a policy
does, through yam_arena's IK. Once the bridge is in, the free arm serves as a camera: its wrist camera looks
down the length of the slot being worked (the slide's lugs and the blade going in), and the screwdriver arm's wrist
camera looks down its own shaft into the slot:

```bash
inspect-robots run --embodiment yam_arena --policy ctbridge-demo \
  -E task=tasks/ct-test-terminal-bridge/ct_test_terminal.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/ct-bridge-demo
```

`task_score` prints the trial's score: 1.0 on success, otherwise the partial credit given in task.json. To make
videos, run `inspect-robots video <log>.json`, which writes one MP4 per camera.

- **Episode end.** The after-release check runs once the bridge is home in 2S1/2S2's CT-side sockets, both 2S links
  are open and both grippers have stayed at least 5 cm from the bridge for 2 s. It lasts 3 s of simulated time
  (5 N along the rail, then 5 N toward the robot, 1.5 s each), runs inside that control step and off camera,
  and then the episode ends. The stored frames stop at the release. If the check never runs, the run ends at
  `--max-steps`.
- **Control run.** `--policy ctbridge-demo -P control=true` leaves the bridge in its holder, picks up the screwdriver
  and opens both 2S links, then forces the check once the arm has backed off. It must score 0.
- **Seeds.** One fixed layout (`-E object_jitter=0`). Seeded jitter is not used.
- **Runtime per trial.** See "Measured" below. Isaac Sim's first start on a new machine adds several minutes.

## Measured

RTX-class GPU shared with several other simulators, Isaac Sim shader cache warm, `-E object_jitter=0`, 20 Hz,
320x176, 240 Hz physics:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `ctbridge-demo` | `task_score: 1`. Arm A lifted the bridge (step 285), lowered it over 2S1/2S2's CT-side sockets; it dropped home during the second line-up correction (step 547). Arm A then held its wrist camera over the slot being worked while arm B, holding the screwdriver, pushed 2S1's slide to its 5 mm stop and then 2S2's, each on the first push (step 1,403). The 5 N tug check passed. | 1,443 (72 s), then the 3 s check off camera | about 21 minutes |
| Control: `ctbridge-demo -P control=true` (bridge left in its holder) | `task_score: 0` (re-run with the current demo, which uses the camera arm and the along-rail screwdriver grip): both 2S slides pushed with no bridge in; the open-circuit rule fires and the forced check fails. | up to 2,400 | about 20 minutes on a shared GPU |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`) | `task_score: 0`. The instruction was accepted. Both arms dropped low near their bases and moved about there; neither reached the terminal row, the bridge or the screwdriver, and no prop moved. | 2,400 (step limit) | about 29 minutes (Dream reported an observation as expired and was asked again 49 times) |

The wall times are long because the GPU was shared; the demo's own cost is two arms' Python IK every control step.

## Layout

| Path | What |
|---|---|
| `ct_test_terminal.py` | the task file: description for agents, props and task, scoring |
| `ctbridge_sim/station.py` | the run-time extension of yam_arena (props, per-step task events, off-camera check) |
| `ctbridge_sim/task.py`, `scorer.py`, `spec.py` | the mechanism (slides and their detent, socket contact, CT rule, decoys, check) and milestones |
| `ctbridge_sim/assets/*.usda`, `ctbridge_sim/usd/` | the props, and the script that generates them (`python -m ctbridge_sim.usd.ct_props`) |
| `ctbridge_sim/demo.py`, `ctbridge_sim/policy.py` | the scripted demo and its Inspect Robots policy `ctbridge-demo` |

## References

- Phoenix Contact URTK/S test-disconnect terminal block, 0311087, datasheet (dimensions, disconnect slide with M3
  clamp screw 0.6-0.8 N m, test sockets either side of the disconnect point, KSS 8 accessory, CT test circuit drawing
  "c = transformer short-circuit, d = relay testing"): https://files.kempstoncontrols.com/files/768da0c1fa7593a710c8003925cf1ac7/0311087.pdf
- The same URTK/S datasheet, 2019 edition: https://docs.rs-online.com/991c/0900766b8171a07b.pdf
- Phoenix Contact KSS 8 plug-in short-circuit connector, 0311540 (2 poles, 4 mm sprung pins, 8 mm pitch, PA grip, 8 g):
  https://proax.ca/en/product/158175/phx0311540
- OSHA 29 CFR 1910.269(w)(2), current transformer secondaries:
  https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.269
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX at 240 Hz in Isaac Lab-Arena, both arms driven as in Dream's station (joint PD with i2rt's gains and gravity compensation). Real sizes: the terminals' 8.2 mm pitch, 61 mm length and 58.6 mm height, the partition plates and the KSS 8's two 4 mm pins and 8 g mass are from Phoenix's datasheets. Inferred from product photos, to be measured on real parts (no CAD could be downloaded): the slot (7.2 mm wide, 22 mm deep, 40 mm long), the socket positions and depth, the slide's shape, its 5 mm travel and its opening direction (away from the robot, toward the relay side), and the bridge's grip and pin length. Collision shapes are boxes: the round 4 mm sockets are 4.8 mm square holes and the bridge's sprung (lamella) pins are rigid 4 mm square pins with 0.4 mm clearance, so the bridge is held by geometry and gravity, not by spring contact; the after-release check therefore pulls it sideways, not outward. The pins sit at 8.2 mm (the real KSS 8's 8 mm pitch springs to fit). The terminal row lies face up on a plate on the table rather than standing on a vertical relay panel, so the robot works from above. The slide's M3 clamp screw is not modelled: a real worker loosens it (0.6-0.8 N m), pushes the slide and retightens it; here the slide is held at either end of its travel by a spring detent (a rule, 0.6 N at mid-travel, inferred) and the robot only pushes it. The screwdriver is shortened (70 mm square handle, 50 mm shaft; real 80-100 mm and 60-100 mm) and its blade thickened to 1.2 mm (real 0.6 mm) for the simulator. There is no electrical circuit: 'shorted' and 'open-circuited' are rules on the pins' positions in the sockets and on the slides' travel. The circuit's markers and cards are coloured so the named circuit can be found in the 320 x 176 top camera, where the printed names are 2-3 pixels tall; real panels use printed markers, and coloured markers are an option of the same marker system. The terminals' clamp screws, bridge studs and wires are visual only. The after-release tug (5 N, about 60 times the bridge's weight) is applied as a force in the simulator after the recording ends; with the robot clear it starts while the screwdriver is still being lifted out of the second slot. One layout.
