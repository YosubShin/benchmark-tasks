# Short a CT at a meter test switch and insert the test plug

A 10-pole ANSI C12.9 meter test switch (TESCO layout, 19.05 mm pole pitch, knife blades with coloured handles) is
mounted upright on a meter box's back wall in front of the robot. Its ten switches are closed. A red current test plug
lies on a rest; a black test lead runs from its handle to a test-set box on the table. The robot must open the four
voltage switches, open the orange phase-A short-circuiting switch (this shorts the current transformer), then the red
phase-A test-jack switch, and push the test plug into the red switch's test jack. The sourced safety rule is
enforced: the short-circuiting switch must be open before the test-jack switch leaves its jaw (otherwise a live CT
secondary is open-circuited; TESCO, quoting ANSI C12.9). Scoring checks the result physically: after release, a 6 N
push on the red handle must be stopped by the seated plug.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file
(`meter_test_switch.py`) in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream` policy.
The one-file format cannot declare its props, which are articulated (the ten blades, the jacks' sprung leaves, and the
test lead, a chain of jointed capsules ending in the plug), so the file uses `testswitch_sim` (this folder), which adds
them to the station at run time. `testswitch_sim/station.py` documents exactly what it patches. It requires
yam-isaac-lab-arena at commit `0059d02`, the Brev launchable's pin.

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/meter-test-switch --no-deps           # registers the policy switch-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream -P record=false \
  -E task=tasks/meter-test-switch/meter_test_switch.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "At the meter test switch: open the three switches with black handles and the one with the white handle at the right end, then the orange switch at the left end, then the red switch next to it, and push the red test plug into the red switch's jack. Do not open the red switch while the orange one is closed. Leave the other switches closed." \
  --scorer task_score --max-steps 3200 --store-frames --log-dir ~/yam-eval/test-switch-dream
```

The scripted demo is closed loop. Like a scripted demo with privileged state, it reads blade angles and the plug's
pose from the simulator, and it sends the same 14-D joint action a policy does, through yam_arena's IK:

```bash
inspect-robots run --embodiment yam_arena --policy switch-demo \
  -E task=tasks/meter-test-switch/meter_test_switch.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 3200 --store-frames --log-dir ~/yam-eval/test-switch-demo
```

`task_score` prints the trial's score: 1.0 on success, otherwise the partial credit given in task.json. To make
videos, run `inspect-robots video <log>.json`, which writes one MP4 per camera.

- **Episode end.** The after-release check runs once the plug is seated and both grippers have stayed at least 6 cm
  from the plug for 1 s. It lasts 1.5 s of simulated time, runs inside that control
  step and off camera, and then the episode ends. The stored frames stop at the release. If the check never runs, the
  run ends at `--max-steps`.
- **Control run.** `--policy switch-demo -P control=true` leaves the red test-jack switch closed and pushes the plug
  into its jack anyway. The closed blade in the jack's slot stops the probe short of seated, and the check is forced
  once the hand is clear. It must score below 1.
- **Procedure coverage.** The rule enforced is the sourced one: short-circuiting switch before test-jack switch. No
  published source we found says whether the voltage switches open before or after the current switches, so their
  order is free. TESCO's last step, pushing the short-circuiting switch back in with the plug seated, is not asked
  for. An open blade hangs 2-3.5 cm in front of the meter-box wall, and the YAM gripper's body meets the wall before
  its fingers get under the handle; five probe runs failed. The CT rule already allows that step: a seated plug
  carries the CT, so a robot that does it is not penalised.
- **Seeds.** One fixed layout (`-E object_jitter=0`). Seeded jitter is not used.
- **Runtime per trial.** See "Measured" below. Isaac Sim's first start on a new machine adds several minutes.

## Measured

RTX-class GPU shared with several other simulators, Isaac Sim shader cache warm, `-E object_jitter=0`, 20 Hz,
320x176, 240 Hz physics:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `switch-demo` | `task_score: 1`. Arm B opened poles 6, 9 and 10, then arm A opened 3, 1 (orange) and 2 (red), each by hooking the closed fingers behind the handle and pulling it out (40-56 control steps of pulling per pole). The first push of the plug stopped 26 mm out with the probe 9.5 mm low; the demo backed off, lined up again and seated it (18 mm from the face). The check held the red blade open. | 2,087 (104 s), then the 1.5 s check off camera | about 31 minutes |
| Control: `switch-demo -P control=true` (red test-jack switch left closed) | `task_score: 0.25`. The voltage switches and the orange switch open; the probe stops on the closed red blade 26 mm out from the face (about 8 mm into the jack) on all four tries, so it is never seated, and `plug_at_closed_jack` stops progress | 2,544, then the forced check | about 40 minutes |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`) | `task_score: 0`: the arms move a little near their start and no switch is opened. | 3,200 (step limit) | about 40 minutes on a shared GPU |

The wall times are long because the GPU was shared; the demo's own cost is two arms' Python IK every control step.


## Layout

| Path | What |
|---|---|
| `meter_test_switch.py` | the task file: description for agents, props and task, scoring |
| `testswitch_sim/station.py` | the run-time extension of yam_arena (props, per-step task events, off-camera check) |
| `testswitch_sim/task.py`, `scorer.py`, `spec.py` | the mechanism (jaw detents, sprung jacks, lead, CT rule, check) and milestones |
| `testswitch_sim/assets/*.usda`, `testswitch_sim/usd/` | the props, and the script that generates them (`python -m testswitch_sim.usd.switch_props`) |
| `testswitch_sim/demo.py`, `testswitch_sim/policy.py` | the scripted demo and its Inspect Robots policy `switch-demo` |

## References

- TESCO, "Test Switch Operation" meter-school deck (RMEMA 2025), reproducing ANSI C12.9 Figs. 1-4 and 6 and Tables 1
  and 3 (pole pitch, test jack and test plug dimensions), TESCO drawing 909884 (10-pole layout and handle colours) and
  the quiz on opening the shunt before the test jack (p.22):
  https://www.tescometering.com/wp-content/uploads/2025/06/RMEMA-2025-Test-Switch-Presentation.pdf
- TESCO meter test switches catalogue sheet: https://www.tescometering.com/wp-content/uploads/2025/07/Meter-Test-Switches_2026-v5.pdf
- TESCO 1077 test plug sheet: https://www.tescometering.com/wp-content/uploads/2025/07/1077-Test-Plug.pdf
- TESCO Tuesdays, "Test Switch Operation & Accessories" (the shunt / test jack / plug / shunt-back-in sequence):
  https://www.youtube.com/watch?v=MvJIKc1ssFs
- Milbank TS10-0109 10-pole test switch spec sheet (outline and side view):
  https://www.stoneway.com/globalassets/catalogs/crescentelectric.com/c/c89/c895e/c895eb5f-ec49-4ef4-a471-285c6e90da523/c895eb5f-ec49-4ef4-a471-285c6e90da52329/milb-ts10-0109-specsheet-17.pdf
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Note on Dream-1 and the first instruction wording

An earlier wording of the instruction began "short phase A's current transformer and put the test plug in: ...".
With that wording the Dream API refused the first inference request with
"DA-4000 BAD_REQUEST: unsafe instruction, refused: Shorting current transformers and electrical switch manipulation
poses high electrical hazards." and no policy action ran. The procedure is protective (shorting the CT before opening
its test jack is what prevents an open-circuited CT secondary), and the bench task is de-energized. The instruction
was reworded to name only the physical steps, with the reason kept in `context`; the submitted Dream-1 run uses the
reworded instruction and was not refused.

## Fidelity details

PhysX at 240 Hz in Isaac Lab-Arena, both arms driven as in Dream's station (joint PD with i2rt's gains and gravity compensation). Rigid bodies on joints at real size: 19.05 mm pitch, 12.7 mm handles, barriers where C12.9 puts them; dimensions come from the C12.9 tables, TESCO's drawing and the Milbank sheet (no CAD; collision shapes are boxes). The jaw that holds a closed blade is a rule: a spring with a 0.2 N m breakout (about 4 N pulling the handle out), not measured; our first guess, 0.35 N m, was more than the station arm can pull with its soft joint gains. An open blade hangs from a hinge-friction rule with a 0.1 N m bias toward open. The plug's seating is geometry: two spring leaves (about 2.9 N each, inferred) pinch the probe, the C12.9 guide rib rides in a notch in each leaf, and a closed blade in the slot stops the probe about 9 mm in. There is no electrical circuit or shorting contact: 'CT shorted' and 'CT open-circuited' are rules on blade angles and the plug's pose. The order rule enforced is the sourced one, the short-circuiting switch before the test-jack switch (ANSI C12.9 as quoted by TESCO). We found no source that says when the voltage switches are opened, so their order is free. Only phase A is switched. TESCO's procedure ends by pushing the short-circuiting switch back in with the plug seated. That step is left out: an open blade hangs 2 to 3.5 cm in front of the meter-box wall, and the YAM gripper's body meets the wall before its fingers get under the handle. The switches are named by TESCO's handle colours, and each handle is about 3 pixels wide at 320x176. The clear cover is left out, and the label strip is blank. The plug's orientation in the jack is our reading of the standard's drawing. The test lead is a chain of 24 jointed capsules (60 cm free length; its stiffness, weight and friction are guesses), and the test set is an inert box. The 6 N check force is our choice. It is applied as a torque in the simulator after the recording ends. One layout.
