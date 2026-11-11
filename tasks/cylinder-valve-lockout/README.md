# Close a named gas cylinder's valve and lock it out with a cylinder lockout

Two industrial gas cylinders stand in wells in front of the robot, so only their shoulders, neck rings and brass
handwheel valves rise above the table: argon (dark green shoulder, ARGON label) on the right and oxygen (white
shoulder, OXYGEN label) on the left. Both valves are open half a turn. The robot closes the argon valve by turning
its handwheel clockwise until it seats. It then lowers a red Brady LM023E cylinder lockout cap over that valve,
swings the cap's locking plate shut round the valve's neck, and padlocks the plate's tab to the can's tab with a
Master Lock 410. It must not touch the oxygen cylinder. Scoring checks the lockout physically. After the robot
lets go, the plate's tab is pushed toward open, the cap is twisted and then lifted. The plate must stay shut, the
cap must stay on and the handwheel must stay closed.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file
(`cylinder_valve_lockout.py`) in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream`
policy. Its props are articulated: the handwheels, the cap's plate and the padlock's shackle. The one-file format
cannot declare articulated props, so the file uses `cylvalve_sim` (this folder), which adds them to the station at
run time. `cylvalve_sim/station.py` documents exactly what it patches. It requires yam-isaac-lab-arena at commit
`0059d02`, the Brev launchable's pin.

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/cylinder-valve-lockout --no-deps     # registers the policy cylinder-lockout-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream -P record=false \
  -E task=tasks/cylinder-valve-lockout/cylinder_valve_lockout.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Close the valve of the argon cylinder, the one with the dark green shoulder: turn its handwheel clockwise, seen from above, until it stops. Then lower the red cylinder lockout cap over that valve, swing the cap's locking plate shut around the valve's neck, put the blue padlock's shackle through the two tabs and snap it shut. Do not touch the oxygen cylinder, the one with the white shoulder." \
  --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/cylinder-lockout-dream
```

The scripted demo is closed loop. Like any scripted demo with privileged state, it reads the handwheel and plate
angles and the cap's and padlock's poses from the simulator. It sends the same 14-D joint action a policy does,
through yam_arena's IK:

```bash
inspect-robots run --embodiment yam_arena --policy cylinder-lockout-demo \
  -E task=tasks/cylinder-valve-lockout/cylinder_valve_lockout.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/cylinder-lockout-demo
```

`task_score` prints the trial's score: 1.0 on success, otherwise the partial credit in task.json. To make videos,
run `inspect-robots video <log>.json`, which writes one MP4 per camera. demo.mp4 and dream1.mp4 put the top camera,
at 2x, above the two wrist cameras.

- **Episode end.** The after-release test starts once the padlock is shut through both tabs with the plate shut
  round the argon valve's neck, and both grippers have stayed clear for 1 s (6 cm from the padlock, 7.4 cm from the
  cap's axis). The test takes about 4.5 s of simulated time and runs inside that control step, off camera. The
  episode then ends, so the stored frames stop at the release. If the test never starts, the run ends at
  `--max-steps`.
- **Control run.** `--policy cylinder-lockout-demo -P control=true` closes the valve, fits the cap and shuts its
  plate, but leaves the padlock on its pedestal. It then forces the after-release test, which must score below 1.
- **Seeds.** One fixed layout (`-E object_jitter=0`). Seeded jitter is not used.

## Measured

Measured on an RTX-class GPU shared with other Isaac Sim processes, with the Isaac Sim shader cache warm,
`-E object_jitter=0`, 20 Hz and 320x176 frames:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `cylinder-lockout-demo` | `task_score: 1` | 2,795 (140 s), then the 4.5 s check off camera | about 59 minutes |
| Control: `cylinder-lockout-demo -P control=true` (valve closed, cap fitted, plate shut, no padlock) | `task_score: 0.75` | 1,960, then the check | about 39 minutes |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`) | `task_score: 0` | 3,600 (step limit) | about 55 minutes (Dream reported an observation as expired and was asked again 74 times) |

What happened in each run:

- **Demo.**
  - The argon handwheel was closed from 180 degrees open in one grip, by rolling the wrist.
  - The cap was lifted off its stand, carried over the valve with its tilt corrected, and held with its plate 4.5 mm
    above the neck ring.
  - The plate was pushed round from 150 degrees, first from behind and then from the right, and shut round the
    valve's neck. The cap was let go and settled onto the plate.
  - The padlock was brought in with the tabs in the gap under its open shackle, lowered so the short leg went down
    through both holes, and snapped shut by arm B pressing the shackle's bend.
  - After-release check passed: the plate was pushed open only 5.4 degrees, the cap rose 10 mm until the plate met
    the wrench block, the handwheel stayed at 0 degrees and the padlock still hung from the tabs.
- **Control.** The 10 N push swung the plate fully open (155 degrees), and the lift took the cap off. The handwheel
  stayed closed.
- **Dream-1.** Dream accepted the instruction. Both arms moved a little in front of their start and the left hand came
  near the cap and the oxygen valve, but nothing was grasped and the cap stayed on its stand.

The wall time is mostly GPU contention and the demo's Python IK, which it solves every control step; alone on the GPU
a trial is several times faster. The demo is closed loop because GPU PhysX is not deterministic and the shackle has
about 1.5 mm of clearance in the holes, so replaying its actions open loop is not expected to reach 1.0.

## Layout

| Path | What |
|---|---|
| `cylinder_valve_lockout.py` | the task file: description for agents, props and task, scoring |
| `cylvalve_sim/station.py` | the run-time extension of yam_arena: props, per-step task events, the off-camera check |
| `cylvalve_sim/task.py`, `scorer.py`, `spec.py` | the mechanism (handwheel and plate friction, shackle latch, after-release check), hazard rules and milestones, layout and instruction |
| `cylvalve_sim/assets/`, `cylvalve_sim/usd/` | the props, and the script that generates them from the dimensions below (`python -m cylvalve_sim.usd.cylvalve_props`; `padlock_410.py` draws the padlock) |
| `cylvalve_sim/demo.py`, `cylvalve_sim/policy.py` | the scripted demo and its Inspect Robots policy `cylinder-lockout-demo` |

## How the real device works

The Brady LM023E (Prinzing) gas cylinder lockout is a red polypropylene can, 159 mm tall and 88 mm across, closed
at the top. A flat locking plate lies across its open end on a rivet at the rim. A keyhole slot in the plate runs
from its edge to a round hole. The worker swings the plate aside, lowers the can over the cylinder valve and its
handwheel, and swings the plate back. The valve's neck (the inlet stem between the valve body and the cylinder's
neck ring) slides along the slot into the hole. With the plate shut, a tab on the plate lies against a tab on the
can, and one padlock goes through both. The hole is smaller than the valve body above it, so the can cannot be
lifted off, and the can covers the handwheel, so the valve cannot be reached or turned. Brady gives one padlock
with a shackle up to 0.34 in and stems up to 1.41 in. ABUS's animation shows the same design, as does a
hands-on video of a Beian unit (References).

## References

- Brady (Prinzing) LM023E gas cylinder lockout, distributor listing with Brady's photos (6 1/4 in tall, 3.47 in
  across, polypropylene, one padlock, 0.34 in shackle):
  https://eindustrialsolutions.com/brady-lm023e-gas-cylinder-lockout-1-padlock-0-34-inch-diameter-padlock-shackle
- ABUS, "Gas Cylinder Lockout" (animation of the same design: the can lowered over the valve, the plate swung round
  the neck, the padlock through the tabs): https://www.youtube.com/watch?v=hib5jWwJnGw
- Beian Q31 cylinder lockout, hands-on: https://www.youtube.com/watch?v=YSruBTnxg5k
- Master Lock S3910 pressurized gas valve lockout instructions (a different design that covers the handwheel
  only): https://solutions.masterlock.com/@api/deki/files/4139/S3910_instructions.pdf
- Sherwood GSV cylinder valve datasheet (4.00 in tall, 1.89 in inlet, outlet 1.42 in from the axis; operating
  torque 1.1-2.2 N m, closing torque 2.2-3.3 N m):
  https://sherwoodvalve.com/wp-content/uploads/compressed-gas-gsv-series-cylinders-datasheet_b.pdf
- EN 1089-3 cylinder shoulder colours (argon dark green, oxygen white), as summarised by a gas supplier:
  https://www.bocgases.co.uk/en/images/cylinder-colour-chart_tcm410-36556.pdf
- Master Lock 410 Zenex thermoplastic safety padlock (410BLU): https://www.masterlock.com/business-use/product/410BLU
- OSHA 29 CFR 1910.147, control of hazardous energy:
  https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.147
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX runs at the station's 240 Hz in Isaac Lab-Arena. Both arms are driven as in Dream's station: joint PD with
i2rt's gains and gravity compensation. The props are rigid bodies and joints, and every link of the locking chain is
geometry:

- **Lift.** Shut, the plate's keyhole closes round the valve's inlet stem under the 42 mm wrench block. Its round
  end is 33 mm, so the block stops the cap being lifted off; it can rise about 9 mm before the plate meets the
  block.
- **Handwheel.** The can covers the handwheel, so nothing can reach it.
- **Padlock.** A shackle down through both tabs' vertical holes stops the plate's tab moving sideways, which is the
  only way it can move as the plate swings.
- **Without the padlock.** The plate swings open freely against a light rivet friction, and the cap lifts off.

In the control run (cap fitted, plate shut, no padlock) the after-release push swings the plate open and the lift
takes the cap off. The handwheel's seat is the joint's lower limit.

Simplifications, most important first:

- **Cylinders in wells.** Only each cylinder's shoulder, neck ring and valve rise above the table. Standing on the
  table, a cylinder's valve would be far above the arms' reach. The neck ring with the neck below it is 6 cm tall,
  which leaves room for the padlock to hang beside it, as in Brady's installed photo. A rebuild can use cut-off
  cylinder tops or empty cylinders in a floor well. No gas is involved.
- **Handwheel torque.** The handwheel takes a constant 1.1 N m to turn: a position drive that follows the wheel,
  with its effort capped. That is the low end of Sherwood's published operating torque (1.1-2.2 N m). The real valve
  also needs 2.2-3.3 N m to seat; the sim has no seating peak (cause: SIM; the station wrist's soft gains, 10 N m/rad,
  turn the wheel through the pads' friction, and a firmer seat was not tried). Both valves start open half a turn;
  a real cylinder in use is often open further, which only means more turning.
- **Cap and plate.** The Brady LM023E's published size is used (88 mm across, 159 mm tall). The rest is inferred
  from photos and ABUS's animation of the same design: the 4 mm plate, the rivet at the rim opposite the tabs, the
  33 mm keyhole and the 20 mm wide tabs, each with a 9.5 mm hole. The can is a 24-sided tube of boxes, and the
  plate is 2 mm cells, so its keyhole edge is stepped. The plate opens to 155 degrees and starts at 150. The can and
  plate together weigh 0.2 kg (inferred).
- **Valve.** The valve's body, outlet, bonnet and handwheel are boxes and polygons sized from Sherwood's GSV drawing
  (4 in tall, outlet 1.42 in from the axis). The handwheel is 12-sided in collision, with six lobes in the visual.
- **Shoulder colours.** These follow EN 1089-3 (argon dark green, oxygen white). In the US, colours are not
  standardised and the label decides. Both cues are present, and the shoulder colour is what makes the named
  cylinder visible in the 320 x 176 top camera (the labels are legible in the wrist cameras).
- **Padlock.** A Master Lock 410: the shackle pops straight up 16 mm on a 150 N/m spring and latches by rule within
  2 mm of shut. There is no key.
- **Check forces.** The after-release check forces (10 N on the plate's tab with the can held, a quarter turn at 1 N m, a 20 N lift) are applied in
  the simulator after the recording ends; a reviewer applies them with a spring scale. There is one layout.
