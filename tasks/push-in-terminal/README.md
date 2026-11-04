# Land a ferruled wire in the named push-in terminal

A strip of six WAGO TOPJOB S push-in terminal blocks stands on a DIN rail above a wiring duct. Each block's lower
entry already holds a landed 1.5 mm2 wire in its IEC colour (brown, black, grey, blue, green-yellow, brown) with a
black insulated ferrule; each bends down and back (12 mm bend radius) into the duct's top comb, as in panel wiring,
and is fixed in place, 4 cm below the upper entries. A blue 2.5 mm2 wire
with a blue insulated ferrule comes out of the duct, runs along the table and rests on a small stand. The robot must
pick up the ferruled end and push it straight into the upper entry of terminal X1:4, the blue neutral block, until
the ferrule's collar stops on the entry. It then lets go. Scoring checks the connection physically: after release
the wire must hold a 50 N pull straight back out of the entry, and it must never have gone into another terminal.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file
(`push_in_terminal.py`) in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream` policy.
The one-file format cannot declare its props, which are articulated: each terminal's sprung clamping leg, and the
wire, an anchored chain of jointed capsules. The file therefore uses `terminal_sim` (this folder), which adds them
to the station at run time. `terminal_sim/station.py` documents exactly what it patches. It requires
yam-isaac-lab-arena at commit `0059d02`, the Brev launchable's pin.

**Physics rate.** The clamp's contacts are a tenth of a millimetre deep, so this task runs PhysX at 480 Hz instead of
yam_arena's 240 Hz. It also shrinks PhysX's friction-patch distance to 0.5 mm. The task file sets both itself
(`StationSpec(physics_hz=480, physx=...)`). The policy still acts at `-E control_hz` (20 Hz, so 24 physics steps per
action), and one camera frame is rendered per action. Nothing else in yam_arena changes.

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/push-in-terminal --no-deps            # registers the policy terminal-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=tasks/push-in-terminal/push_in_terminal.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Land the blue ferruled wire in terminal X1:4, the blue terminal, upper entry. Push the ferrule straight into the entry until the collar stops, then let go." \
  --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/push-in-dream
```

The scripted demo is closed loop. Like a scripted demo with privileged state, it reads the wire end's pose from the
simulator, and it sends the same 14-D joint action a policy does, through yam_arena's IK:

```bash
inspect-robots run --embodiment yam_arena --policy terminal-demo \
  -E task=tasks/push-in-terminal/push_in_terminal.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 1200 --store-frames --log-dir ~/yam-eval/push-in-demo
```

`task_score` prints the trial's score: 1.0 on success, otherwise the partial credit given in task.json. To make
videos, run `inspect-robots video <log>.json`, which writes one MP4 per camera.

- **Episode end.** The tug runs once the wire is seated and both grippers have stayed at least 6 cm from it for 1 s.
  It lasts 2 s of simulated time, runs inside that control step and off camera, and then the episode ends. The
  stored frames stop at the release. If the tug never runs, the run ends at `--max-steps`.
- **Control run.** `--policy terminal-demo -P control=true` stops the push once the ferrule's tip is past 4.5 mm (it ended 6.4 mm in, 3.6 mm short of home), short of the clamping
  leg, and lets go. The tug is then forced. It must score below 1, because the wire pulls out.
- **Seeds.** One fixed layout (`-E object_jitter=0`). Seeded jitter is not used.
- **Runtime per trial.** See "Measured" below. The 480 Hz physics makes this task about twice as slow per simulated
  second as a 240 Hz station task. Isaac Sim's first start on a new machine adds several minutes.

## Measured

RTX-class GPU, Isaac Sim shader cache warm, `-E object_jitter=0`, 20 Hz, 320x176, 480 Hz physics:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `terminal-demo` | `task_score: 1`. No retries (five line-up corrections). Peak resistance about 7 N, with the fingers pushing at the collar. The 50 N tug held. | 923 (46 s), then the 2 s tug off camera | about 9 minutes (15 minutes with a second simulator sharing the GPU) |
| Control: `terminal-demo -P control=true` (push stops 6.4 mm in, short of home; forced tug) | `task_score: 0.5` (the ferrule is in the entry but never home; the tug pulls it out) | 597 | about 5 minutes |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`) | `task_score: 0` (both arms move a little near their start, the right hand drifts toward the stand, and the wire is never lifted). | 1,200 (step limit) | about 8 minutes (the simulator runs slower than real time at 480 Hz; Dream sometimes reported an observation as expired and was asked again, 24 times in the run) |

The demo's wall time is mostly the 480 Hz physics and its Python IK every control step. A clamp probe without the
robot shows the wire going home with a few newtons. Once released, it holds 50 N with 0.35 mm of creep and pulls
out at about 85 N.

## Layout

| Path | What |
|---|---|
| `push_in_terminal.py` | the task file: description for agents, props and task, physics rate, scoring |
| `terminal_sim/station.py` | the run-time extension of yam_arena (props, per-step task events, off-camera tug, per-task physics rate) |
| `terminal_sim/task.py`, `scorer.py`, `spec.py` | the mechanism (clamping legs, wire, tug) and milestones |
| `terminal_sim/assets/*.usda`, `terminal_sim/usd/` | the props, and the script that generates them (`python -m terminal_sim.usd.terminal_props`) |
| `terminal_sim/demo.py`, `terminal_sim/policy.py` | the scripted demo and its Inspect Robots policy `terminal-demo` |

## References

- WAGO 2002-1201 TOPJOB S 2-conductor through terminal block, 2.5 mm2, grey (dimensions, Push-in CAGE CLAMP,
  ferruled conductors pushed in without a tool): https://www.wago.com/us/p/2002-1201
- WAGO 2002-1204 (blue) and 2002-1207 (green-yellow ground) terminal blocks: https://www.wago.com/us/p/2002-1204,
  https://www.wago.com/us/p/2002-1207
- WAGO 216-246 insulated ferrule, 2.5 mm2, blue, DIN 46228-4: https://www.wago.com/us/p/216-246
- WAGO 793-5501 WMB marking card (terminal markers): https://www.wago.com/us/p/793-5501
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX in Isaac Lab-Arena at 480 Hz (twice the station's 240 Hz; the clamp's sub-millimetre contacts need it), both arms driven as in Dream's station (joint PD with i2rt's gains and gravity compensation). Published dimensions: the terminal block's width, length and height above the rail, the 15-degree entry angle, the ferrule's sleeve and collar, the DIN rail. The clamp is geometry plus a spring: a hinged clamping leaf (a stand-in for the stainless CAGE CLAMP spring) on a preloaded torsion spring stands 0.5 mm into the ferrule's path and leans in the insertion direction. The ferrule pushes it open with a few newtons, and pulling drags its edge into the ferrule, wedging it against the busbar. In a pull probe the wire held 50 N with 0.35 mm of creep and came out at about 85 N. Leaf size, angle, spring (0.10 N·m/rad, 0.025 N·m preload) and mass were tuned to hold above 50 N and to go in with the push the station's arm and gripper can give, and are not measured. The entry channel is a square 3.1 x 2.9 mm box with a tapered mouth (0.6 mm wider across and 0.4 mm along the rail at the face, over 2 mm), sized from WAGO's photos (no CAD); round parts are square in collision (the 2.5 mm sleeve is a 2.5 mm square bar). The collar stopping at the mouth is geometry. Every terminal would clamp the wire equally, so 'wrong terminal' is a position rule. All six blocks share the 2002-1201 profile (the 2002-1207's rail foot is not modelled). The wire beyond its rigid last 8 cm is 16 jointed capsules (25 mm each) with a bending spring toward its routed shape; its stiffness (EI 1.5e-4 N m^2), weight (3 g per link, heavier than real) and friction are guesses. The duct is a solid box with a slot. The landed lower-entry wires are rigid, fixed to the strip and visual only (no collision), not flexible conductors; the loose blue cable can pass through their bends, which a real cable would rest on. The printed markers are about 2 pixels tall in the 320x176 top camera, so the target is told apart there by its blue housing (the neutral block), not by reading its marker. The 50 N pull is the value commonly cited from IEC 60999-1 for 2.5 mm2, not checked against the standard's text; it is applied as a force in the simulator after the recording ends. One layout. The insulation's friction against the finger pads is set to 2.5, to make up for the station gripper model's weak squeeze (about 9 N per finger); the real gripper's grip on a 3.6 mm wire is not measured.
