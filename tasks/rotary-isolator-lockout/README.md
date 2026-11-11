# Switch off a named rotary isolator and padlock its handle at OFF

Two enclosed rotary isolators (ABB OT16 switch-disconnectors behind ABB OHB45J6 pistol handles) hang side by side,
both ON, each with an engraved nameplate: Q1 (yellow) and Q2 (white). The robot turns Q1's
handle a quarter turn to OFF, pulls the handle's red padlock slide out of the grip by its tab at the grip's tip (it
only comes out at OFF) and padlocks the tab with a Master Lock 410, without turning Q2. Scoring checks the lockout physically: after the robot lets go, the slide is
pushed back in and the handle is turned toward ON, and the handle must stay at OFF.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file (`rotary_isolator.py`)
in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream` policy. Its props are
articulated (the two handles, their padlock slides, the padlock's shackle), which the one-file format cannot declare,
so the file uses `isolator_lockout_sim` (this folder), which adds them to the station at run time
(`isolator_lockout_sim/station.py` documents exactly what it patches; it requires yam-isaac-lab-arena at commit
`0059d02`, the Brev launchable's pin).

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/rotary-isolator-lockout --no-deps     # registers the policy isolator-lockout-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream -P record=false \
  -E task=tasks/rotary-isolator-lockout/rotary_isolator.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Switch off isolator Q1, the one with the yellow nameplate: turn its handle a quarter turn anticlockwise, as you face it, from ON (pointing up) to OFF (pointing left). Then pull the red padlock slide out of the bottom of the handle, put your padlock's shackle through the hole in the slide and snap it shut. Do not turn isolator Q2, the one with the white nameplate." \
  --scorer task_score --max-steps 3000 --store-frames --log-dir ~/yam-eval/rotary-isolator-dream
```

The scripted demo (closed loop: it reads the handle angle, the slide position and the padlock's pose from the
simulator, like a scripted demo with privileged state, and sends the same 14-D joint action a policy does, through
yam_arena's IK):

```bash
inspect-robots run --embodiment yam_arena --policy isolator-lockout-demo \
  -E task=tasks/rotary-isolator-lockout/rotary_isolator.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 3000 --store-frames --log-dir ~/yam-eval/rotary-isolator-demo
```

`task_score` prints the trial's score (1.0 on success, otherwise the partial credit in task.json). Videos:
`inspect-robots video <log>.json` writes one MP4 per camera (demo.mp4 and dream1.mp4 here put the top camera, at 2x,
above the two wrist cameras).

- **Episode end.** Once the padlock is shut through Q1's slide and both grippers have stayed at least 6 cm clear of it
  for 2 s, the after-release test runs (about 4 s of simulated time) inside that control step, off camera, and the
  episode ends; the stored frames stop at the release. Otherwise the run ends at `--max-steps`.
- **Control run.** `--policy isolator-lockout-demo -P control=true` turns Q1 to OFF and pulls its slide out but
  leaves the padlock on its stand, then forces the after-release test: it must score below 1.
- **Seeds.** One fixed layout (`-E object_jitter=0`); seeded jitter is not used.

## Measured

RTX-class GPU shared with other simulations, Isaac Sim shader cache warm, `-E object_jitter=0`, 20 Hz, 320x176:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `isolator-lockout-demo` | `task_score: 1` (Q1 turned to OFF; slide pulled out its full 32 mm by the finger tab; padlock brought in under the grip and its short leg slid 14 mm through the tab's hole; arm B held the shackle's bend while arm A pushed the body on, latched after 9 steps; after-release check passed: slide pushed in 19 mm until the shackle met the grip, handle turned 14.9 degrees toward ON and stopped) | 2,191 (110 s), then the 4 s test off camera | about 13 minutes |
| Control: `isolator-lockout-demo -P control=true` (Q1 OFF, slide out, no padlock) | `task_score: 0.5` (after-release check fails: the slide is pushed back in its full 32 mm and the handle turns the full 90 degrees to ON) | 791, then the test | about 5 minutes |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`) | `task_score: 0` (the arms move a little in front of their start and never touch either handle or the padlock) | 3,000 (step limit) | about 16 minutes (59 "session expired" retries on the shared GPU) |

The demo's wall time is mostly its IK, which it solves in Python every control step, and the shared GPU; on an
otherwise idle machine expect less. The demo is closed loop because GPU PhysX is not deterministic and the shackle
has under 1 mm of clearance in the hole: replaying its actions open loop is not expected to reach 1.0.

## Layout

| Path | What |
|---|---|
| `rotary_isolator.py` | the task file: description for agents, the props and task, scoring |
| `isolator_lockout_sim/station.py` | the run-time extension of yam_arena (props, per-step task events, off-camera test) |
| `isolator_lockout_sim/task.py`, `scorer.py`, `spec.py` | the mechanism (turning friction, shackle latch, decoy watch, after-release test) and milestones |
| `isolator_lockout_sim/assets/`, `isolator_lockout_sim/usd/` | the props, and the scripts that generate them from the dimensions in task.json (`python -m isolator_lockout_sim.usd.isolator_props isolator_lockout_sim/assets`); `ohb45j6_mesh.npz` is the handle's outline from ABB's CAD file |
| `isolator_lockout_sim/demo.py`, `isolator_lockout_sim/policy.py` | the scripted demo and its Inspect Robots policy `isolator-lockout-demo` |

## References

- ABB switch-disconnectors OT16...630 catalogue, 1SCC301009K0201 (OH_ pistol handles padlockable with three padlocks
  in the OFF position, bail 5-8 mm; OHB45J6 for OT16...40F, 0.1 kg; dimension drawings):
  https://search.abb.com/library/Download.aspx?DocumentID=1SCC301009K0201
- ABB OHB45J6 handle, CAD (STEP) used for the handle's outline:
  https://search.abb.com/library/Download.aspx?DocumentID=ohb45j6.stp&LanguageCode=en&DocumentPartId=&Action=Launch
- ABB OT16F3 data (operating torque 1 N m): https://www.mymectronic.com/part-detail/ABB/OT16F3
- Master Lock 410 Zenex thermoplastic safety padlock (410PRP): https://www.masterlock.com/business-use/product/410PRP
- OSHA 29 CFR 1910.147, control of hazardous energy:
  https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.147
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX at 240 Hz in Isaac Lab-Arena, both arms driven as in Dream's station (joint PD with i2rt's gains and gravity
compensation). Rigid bodies on joints, with every link of the locking chain in geometry: the red slide's tail can
only come out where the handle's fixed base has its slot (at OFF; anywhere else the base stops it), the tail in the
slot stops the handle turning, and a shackle in the slide's hole meets the grip's lower edge if the slide is
pushed back in, before the tail leaves the slot. The quarter-turn end stops are joint limits.

- **Handle.** Its visual is ABB's own CAD outline of the OHB45J6 (STEP file from ABB's library); its colliders are
  boxes (an octagonal hub and the grip). ON is the grip pointing up and OFF pointing left, as on installed units
  (ABB's leaflet draws the other mounting, OFF pointing right); switching ON is clockwise in both.
- **Turning torque.** ABB lists 1 N m operating torque for the OT16F3. The sim uses a constant 0.4 N m friction-like
  torque (12.5 N at the middle of the grip). With the real 1 N m (31 N at the grip) the demo's arm stalled: the station
  arm's soft joint gains (shoulder 80 N m/rad, wrist 10 N m/rad) give only about 15 N at the hand for a few cm of
  position lead. The real switch is also snap-action (the contacts flip over a toggle spring); this is not modelled.
  The after-release check still applies 2 N m, twice the real torque.
- **Padlock slide.** As ABB's leaflet and installed handles show, the red slide lies flush along the grip's lower
  edge (at OFF) from the hub to the grip's tip, and is pulled out at the tip at OFF (up to three padlocks, bail 5-8 mm
  in the catalogue, 5-10 mm in the leaflet). Padlocks then hang under the grip's outer half. Ours:
  - the stripe (6 x 13 mm, from 20 mm out from the axis) ends at the tip in a flat red tab, hidden in the grip except for a 16 mm finger tab
    below the tip, which is enlarged so the robot's fingers can take it;
  - the whole slide pulls straight down 32 mm (the real slide's exact motion is not published); the tab then shows
    one 8.5 mm square hole, its axis along the grip, 17 mm below the grip's outer half. It takes one padlock (the real
    slide takes up to three);
  - its inside is ours: a tail behind the tab runs back to a fixed stop behind the handle (an invisible annulus sector
    45-65 mm from the axis) with a straight slot at OFF. Out at OFF, the tail sits in the slot and the handle cannot
    turn; at any other angle the stop keeps the slide in. A shackle in the tab's hole meets the grip's lower edge if
    the slide is pushed back in, before the tail leaves the slot; pushed in that far, the handle has about 12 degrees of
    play (the after-release check allows 20);
  - the slide is held where it is by 2 N of friction.
- **Padlock.** A Master Lock 410 (1/4 in shackle), the same model as in our other lockout tasks: rounded
  38 x 44 x 19 mm body with a danger label, round 6.35 mm shackle with a semicircular bend (collision: square legs and
  six straight segments around the bend), the short leg's collider a low-friction square. Its shackle pops straight out
  and is pushed straight in (no swivel), with a 150 N/m spring and a catch that locks it once pushed within 2 mm of
  shut; the key is not simulated.
- **Labels.** Machine isolators are told apart by engraved nameplates; ours are 86 x 30 mm two-ply laminate, one
  yellow and one white so the named one is visible in the 320 x 176 overhead camera (the letters are not legible
  there; they are in the wrist cameras). Plate sizes, colours and the enclosures (100 x 150 x 90 mm) are ours.
- **Check forces.** The 10 N slide push and the 2 N m turn are our choice and are applied as forces in the simulator
  after the recording ends. One layout.
