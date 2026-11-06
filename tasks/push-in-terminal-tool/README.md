# Land a bare stranded wire in a push-in terminal with the operating tool

A strip of six WAGO TOPJOB S push-in terminal blocks stands on a DIN rail above a wiring duct. A blue 2.5 mm2
fine-stranded wire, stripped 10 mm with no ferrule, comes out of the duct, runs along the table and rests on a small
stand. A WAGO operating tool (a small screwdriver) lies on a second stand. Bare fine strands cannot open a push-in
clamp, so the robot must push the tool's blade into the operating slot of terminal X1:4 (the blue block) to hold its
clamp open, insert the stripped end into X1:4's upper entry until the insulation stops, pull the tool back out so the
clamp closes on the strands, and let go. Scoring checks the connection physically: after release the wire must hold
a 50 N pull straight back out of the entry; it must never have gone into another terminal, and the tool must not be
left in a slot.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file
(`push_in_terminal_tool.py`) in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream`
policy. The one-file format cannot declare its props, which are articulated (each terminal's sprung clamping leg; the
wire, an anchored chain of jointed links ending in a jointed bare-strand end) or free (the tool). The file therefore
uses `terminal_tool_sim` (this folder), which adds them to the station at run time. `terminal_tool_sim/station.py`
documents exactly what it patches. It requires yam-isaac-lab-arena at commit `0059d02`, the Brev launchable's pin.

**Physics rate.** The clamp's contacts are a tenth of a millimetre deep, so this task runs PhysX at 480 Hz instead of
yam_arena's 240 Hz, and shrinks PhysX's friction-patch distance to 0.5 mm. The task file sets both itself
(`StationSpec(physics_hz=480, physx=...)`). The policy still acts at `-E control_hz` (20 Hz, so 24 physics steps per
action), and one camera frame is rendered per action. Nothing else in yam_arena changes.

## How the real procedure works

WAGO's datasheet for the 2002-1201 lists solid conductors (1-4 mm2) and fine-stranded conductors with ferrules
(1-2.5 mm2) for push-in termination; fine-stranded conductors without ferrules (0.25-4 mm2) are connected "via
operating tool": the tool is inserted into the operating slot beside the entry, square to the face, which opens the
clamping spring; the stripped conductor (10-12 mm) goes into the entry, which is angled under 15 degrees to the tool;
the tool is pulled out and the spring clamps the conductor. On TOPJOB S the tool stays in the slot on its own while it
holds the clamp open ("hands-free wiring"), so forgetting it there leaves the wire unclamped.

In the simulation: the stripped end is four jointed bare-strand links on a sprung axial slide; pushed against the
closed clamping leg, the strands bunch up (about 2.5 N of push at 4.5 mm of bunching) and the leg does not move. The operating slot is a
passage in each block; the tool's blade rides up a ramp on the clamping leg's back, turning the leg open by about 20-25
degrees, and rests on the ramp's top corner, where the spring presses it square to its travel. Pulling it back lets
the leg close and wedge the strands against the busbar. The simulated tool is less self-holding than WAGO's: in a
probe without the robot it stayed in when let go, but when the robot's gripper opened on it, it slid back out. The
scripted demo therefore keeps hold of it (arm A holds the tool, arm B lands the wire).

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD               # must print 0059d02
uv pip install -e tasks/push-in-terminal-tool --no-deps            # registers the policy terminal-tool-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream \
  -E task=tasks/push-in-terminal-tool/push_in_terminal_tool.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Land the blue wire's stripped end in terminal X1:4, the blue terminal, upper entry. The strands have no ferrule: push the green operating tool into X1:4's operating slot to open the clamp, insert the wire until the insulation stops, then pull the tool out and let go." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/push-in-tool-dream
```

The scripted demo is closed loop and uses both arms. Like a scripted demo with privileged state, it reads the wire
end's and the tool's poses from the simulator, and it sends the same 14-D joint action a policy does, through
yam_arena's IK:

```bash
inspect-robots run --embodiment yam_arena --policy terminal-tool-demo \
  -E task=tasks/push-in-terminal-tool/push_in_terminal_tool.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/push-in-tool-demo
```

`task_score` prints the trial's score: 1.0 on success, otherwise the partial credit given in task.json. To make
videos, run `inspect-robots video <log>.json`, which writes one MP4 per camera.

- **Episode end.** The tug runs once the wire is seated (insulation home, tool out of every slot, the clamp closed on
  the strands) and both grippers have stayed at least 6 cm from the wire for 1 s. It lasts 2 s of simulated time,
  runs inside that control step and off camera, and then the episode ends. The stored frames stop at the release. If
  the tug never runs, the run ends at `--max-steps`.
- **Control run.** `--policy terminal-tool-demo -P control=true` leaves the tool on its stand and pushes the stripped
  end into X1:4 just the same, lets go, and forces the tug. It must score below 1.
- **The demo's two arms.** Arm A (left) picks the tool by the rear end of its handle and holds it in the slot; arm B
  (right) picks the wire end first and parks it to the right, then lands it once the clamp is open. Both hands are
  rolled about the part they hold so their wrists clear each other. The stripped end sometimes catches on the
  entry's mouth; the demo then backs off 1.5 mm and pushes again.
- **Seeds.** One fixed layout (`-E object_jitter=0`). Seeded jitter is not used.
- **Runtime per trial.** See "Measured" below. Isaac Sim's first start on a new machine adds several minutes.

## Measured

RTX-class GPU shared with other jobs, Isaac Sim shader cache warm, `-E object_jitter=0`, 20 Hz, 320x176, 480 Hz
physics:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `terminal-tool-demo` | `task_score: 1`. Tool in after two re-line-ups at the slot, clamp held open at 20-22 deg; wire home after one back-off at the mouth; tool out; clamping leg resting on the strands at 10 deg; the 50 N pull held | 1,787 (89 s), then the 2 s pull off camera | about 29 minutes |
| Control: `terminal-tool-demo -P control=true` (no tool; same push; forced pull) | `task_score: 0`. The insulation reached the mouth but the clamping leg never moved (0.0 deg): the strands bunched 3.2 mm instead; the forced pull drew the wire out | 634 | about 21 minutes |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`) | `task_score: 0`. Both hands lift and hover near their starts, the right hand drifts toward the wire's stand; neither the tool nor the wire is picked up | 2,400 (step limit) | about 26 minutes (an earlier attempt stopped at step 416 when the Dream service reported no capacity) |

Without the robot (a probe with force-controlled "hands", same assets, 480 Hz): pushed alone up to 15 N the strands
bunch about 3.6 mm and the leg stays shut, and the wire then pulls out at about 11 N; with the tool in (it went in at
about 11-14 N and stayed when let go) the wire slid home with no resistance, the tool came out at about 5 N, and the
clamped wire held 50 N with 1.1 mm of creep (out at about 53 N). The probe ran with the strands' friction at 0.6 and
the bundle's axial spring at 200 N/m; the runs above use 0.8 and 500 N/m.

The wall times are long because the GPU was shared and the 480 Hz physics costs about twice a 240 Hz task.

## Layout

| Path | What |
|---|---|
| `push_in_terminal_tool.py` | the task file: description for agents, props and task, physics rate, scoring |
| `terminal_tool_sim/station.py` | the run-time extension of yam_arena (props, per-step task events, off-camera tug, per-task physics rate) |
| `terminal_tool_sim/task.py`, `scorer.py`, `spec.py` | the mechanism (clamping legs, operating slots, wire, tool, tug) and milestones |
| `terminal_tool_sim/assets/*.usda`, `terminal_tool_sim/usd/` | the props, and the script that generates them (`python -m terminal_tool_sim.usd.terminal_props`) |
| `terminal_tool_sim/demo.py`, `terminal_tool_sim/policy.py` | the scripted two-arm demo and its Inspect Robots policy `terminal-tool-demo` |

## References

- WAGO 2002-1201 TOPJOB S 2-conductor through terminal block, 2.5 mm2 (dimensions; connection data: fine-stranded
  conductors without ferrule via operating tool; "Inserting a conductor via operating tool"; strip length 10-12 mm;
  accessory 210-720): https://www.wago.com/us/p/2002-1201
- WAGO 2002-1204 (blue) and 2002-1207 (green-yellow ground) terminal blocks: https://www.wago.com/us/p/2002-1204,
  https://www.wago.com/us/p/2002-1207
- WAGO TOPJOB S variants (operating slot: "The operating tool remains in the slot until the conductor is fully
  terminated ... leaving both hands free for wiring"):
  https://www.wago.com/tr-en/tr-products/tr-electrical-interconnections/tr-discover-rail-mount-terminal-blocks/tr-topjob-s/tr-topjob-s-variants
- WAGO 210-720 operating tool, blade 3.5 x 0.5 mm, partially insulated shaft:
  https://www.wago.com/global/tools/operating-tool/p/210-720
- WAGO, "TOPJOB S For all Conductor Types" (animation; fine-stranded conductor via operating tool at 0:44-0:59):
  https://www.youtube.com/watch?v=2sqoHqcQGEg
- Lapp H07V-K single-core wire (2.5 mm2: 3.4-4.1 mm over the insulation): https://products.lappgroup.com/artno=4520001PDF
- Phoenix Contact, mechanical tests for terminal blocks (conductor pull-out test, IEC 60999):
  https://www.phoenixcontact.com/en-ch/quality-tests-for-terminal-blocks/basic-principles-quality-testing-for-terminal-blocks/mechanical-tests-terminal-blocks
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX in Isaac Lab-Arena at 480 Hz (twice the station's 240 Hz; the clamp's sub-millimetre contacts need it), both arms driven as in Dream's station (joint PD with i2rt's gains and gravity compensation). Published: the terminal block's width, length and height above the rail, the under-15-degree entry, the 10-12 mm strip length, that fine-stranded wire without a ferrule is connected with the operating tool, the tool's 3.5 x 0.5 mm blade, and that the tool stays in the slot by itself while it holds the clamp open. The clamp is geometry plus a spring: a hinged clamping leaf (a stand-in for the stainless CAGE CLAMP spring) on a preloaded torsion spring (0.10 N·m/rad, 0.025 N·m preload) stands 0.5 mm into the strands' path and leans in the insertion direction, so a pull wedges the strands against the busbar. The stripped end is four 2.5 mm bare-strand links (a 2.0 mm square bundle in collision, round copper look) on bending joints and a sprung axial slide: the bundle carries about 2.5 N of push before it has bunched 4.5 mm, less than the leaf needs to open, so pushed alone it bunches up and the leaf stays shut; this stands in for loose strands bowing out and is not measured. The operating slot (4 x 1.2 mm, straight into the face, 8.4 mm below the entry, tapered lead-in) and a ramp on the back of the leaf are inferred from WAGO's photos and animation (no CAD): the blade rides up the ramp, turning the leaf about 20-25 degrees open, and rests on the ramp's top corner. The simulated tool is less self-holding than WAGO's: in a probe without the robot it stayed in when let go, but in robot runs it slid back out when the gripper opened on it, so the scripted demo keeps hold of it; the tool-left-in rule is checked on the blade's position. Without the robot the clamped wire held 50 N with about 1 mm of creep. Leaf, ramp and slot sizes, springs, friction (strands 0.8, blade 0.2), the tool's 19 cm length and 50 g mass, and the bare end's stiffness are inferred and tuned, not measured. Round parts are square in collision (the strand bundle, the insulation, the tool's shaft and handle). Every terminal would clamp the wire equally, so 'wrong terminal' is a position rule. The wire beyond its rigid last 8 cm is 16 jointed capsules (25 mm each) with a bending spring toward its routed shape (stiffness, weight and friction are guesses). Strand damage (splaying, broken strands) is not simulated. The printed markers are about 2 pixels tall in the 320x176 top camera, so the target is told apart by its blue housing. The 50 N pull is the value commonly cited from IEC 60999-1 for 2.5 mm2, not checked against the standard's text; it is applied as a force in the simulator after the recording ends. One layout. The insulation's friction against the finger pads is 2.5 and the tool handle's 2.0, to make up for the station gripper model's weak squeeze.
