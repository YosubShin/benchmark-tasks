# Lock out a named breaker in a US panel with a clamp-on lockout and padlock

A small Square D QO load centre carries six breakers in two columns, numbered on the centre strip with colour-coded
circuit labels. The robot must lock out circuit 4 (blue label, OFF): press a Master Lock 493B clamp-on breaker
lockout onto its handle, roll the lockout's knob until the jaws grip the handle, swing the lockout's lever shut over
the knob (the lever is a cover that closes over the knob's top, bottom and end), then hook a Master Lock 410
padlock's open shackle down through the holes at the knob end (the cover's tab lies over the lockout's loop) and press
it shut.
Scoring checks the lockout physically: once the robot has let go, the lever is pulled, the lockout is pulled off the
panel and the handle is pushed toward ON, and the lockout must hold against all three.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file
(`breaker_lockout.py`) in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream` policy. Its
props are articulated (the breaker handles; the lockout's jaw, knob and lever; the padlock's shackle), which the
one-file format cannot declare, so the file uses `clamp_lockout` (this folder), which adds them to the station at run
time (`clamp_lockout/station.py` documents exactly what it patches; it requires yam-isaac-lab-arena at commit
`0059d02`, the Brev launchable's pin).

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/breaker-lockout --no-deps             # registers the policy clamp-lockout-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream -P record=false \
  -E task=tasks/breaker-lockout/breaker_lockout.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Lock out circuit 4, the breaker with the blue label: press the red clamp-on lockout onto its handle, turn the lockout's black knob until it grips the handle, close the lockout's lever, then put the padlock's shackle through the holes at the lever's end and snap it shut. Leave the other breakers as they are." \
  --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/breaker-lockout-dream
```

The scripted demo (closed loop: it reads the lockout's and padlock's poses and the clamp force from the simulator,
like a scripted demo with privileged state, and sends the same 14-D joint action a policy does, through yam_arena's
IK):

```bash
inspect-robots run --embodiment yam_arena --policy clamp-lockout-demo \
  -E task=tasks/breaker-lockout/breaker_lockout.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 3600 --store-frames --log-dir ~/yam-eval/breaker-lockout-demo
```

`task_score` prints the trial's score (1.0 on success, otherwise the partial credit in task.json, or 0 after a
hazard). Videos: `inspect-robots video <log>.json` writes one MP4 per camera.

- **Episode end.** Once the shackle is shut through both holes and both grippers have stayed at least 6 cm clear of
  the lockout and the padlock for 1 s, the after-release check runs inside that control step, off camera (4.5 s of
  simulated time: 20 N pulling the lever's middle outward for 1 s; only if the lever opened past 20 degrees, 0.1 N m
  loosening on the knob for 1 s; 20 N pulling the lockout off the panel for 1.5 s; 0.3 N m pushing circuit 4's
  handle toward ON for 1 s), and the episode ends; the stored frames stop at the release. Otherwise the run ends at
  `--max-steps`.
- **Control run.** `CLAMP_LOCKOUT_CONTROL=1` with `--policy clamp-lockout-demo`: the demo seats the lockout and shuts
  its lever but never turns the knob or brings the padlock, and the after-release check is forced once arm B has
  moved away. It must score below 1.
- **Seeds.** One fixed layout (`-E object_jitter=0`); seeded jitter is not used. PhysX on the GPU is not
  bit-deterministic, which is why the demo corrects its targets from measurements.
- **Runtime per trial**: see "Measured" below; Isaac Sim's first start on a new machine adds several minutes.

## Measured

RTX-class GPU shared with other Isaac Sim processes, Isaac Sim shader cache warm, `-E object_jitter=0`, 20 Hz,
320x176 frames:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `clamp-lockout-demo` | `task_score: 1` (lockout slid onto circuit 4's handle; the exposed knob rolled with a fingertip until the jaws grip (3.2 N hands-off), then the lever cover swept shut over it (cam: about 10 N); shackle through the loop and the cover's tab and latched; after-release check passed: lever opened 1.7 degrees under 20 N, lockout moved 2.1 mm under the 20 N pull, handle held at 13 degrees on the OFF side under 0.3 N m) | 2,802 (140 s), then the 4.5 s check off camera | about 25 minutes |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`) | `task_score: 0` (the right arm hovers over the lockout on its posts without picking it up; the left arm stays near its start; nothing reaches the panel; no hazard events) | 3,600 (step limit) | about 18 minutes |
| Control: `CLAMP_LOCKOUT_CONTROL=1`, `clamp-lockout-demo`, knob never turned, no padlock | `task_score: 0.25` (lockout seated and lever shut; the forced check pries the cover open to 158 degrees, turns the now reachable knob loose, the lockout comes off, and the handle goes to ON) | 885, then the check | about 9 minutes |

The demo's wall time is mostly its IK, which it solves in Python every control step, and the shared GPU.

## Layout

| Path | What |
|---|---|
| `breaker_lockout.py` | the task file: description for agents, the props and task, scoring |
| `clamp_lockout/station.py` | the run-time extension of yam_arena (props, per-step task events, off-camera after-release check) |
| `clamp_lockout/task.py`, `scorer.py`, `spec.py` | the mechanism (handle detents, thumbscrew coupling, lever, shackle latch, after-release check), hazards and milestones, layout and instruction |
| `clamp_lockout/assets/`, `clamp_lockout/usd/` | the props (`panel_props.py`: load centre, QO breakers, 493B lockout, stands; `padlock_410.py`: the padlock), generated with `python -m clamp_lockout.usd.panel_props clamp_lockout/assets` |
| `clamp_lockout/demo.py`, `clamp_lockout/policy.py` | the scripted demo and its Inspect Robots policy `clamp-lockout-demo` |

## References

- Master Lock 493B Grip Tight circuit breaker lockout, technical data sheet (74.0 x 19.8 x 18.9 mm, 11.4 mm slot,
  8.3 mm padlock hole, 53 g, materials):
  https://solutions.masterlock.com/@api/deki/files/356/493B_TECHNICAL_DATA_SHEET_REV_A.pdf?revision=1
- Master Lock instruction sheet 65573 for the 491B/493B (open the lever, turn the knob until the grips touch the
  toggle, close the lever, apply the padlock): https://solutions.masterlock.com/@api/deki/files/3139/65573_ART_A.pdf?revision=3
- Master Lock 493B product literature page:
  https://solutions.masterlock.com/Technical_Literature/Master_Lock/Workplace_Safety%3A_Lockout_Tagout/Electrical_Lockout/Electrical_Lockout%3A_493B
- Brady clamp-on breaker lockout 65396 (same principle: thumbscrew clamp, cover locked over the thumbscrew), catalog page:
  https://static.datasheets.com/doc/3001011-brady-65396-ds.pdf
- Square D QO120 product data sheet (19 mm pole, 76 mm length, handle 10 mm wide, ON 22 deg / OFF 21 deg):
  https://static.raptorsupplies.com/catalog/product/datasheet/square-d-qo120.pdf
- Master Lock 410 Zenex thermoplastic safety padlock: https://www.masterlock.com/business-use/product/410PRP
- OSHA 29 CFR 1910.147, control of hazardous energy: https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.147
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX at 240 Hz in Isaac Lab-Arena, both arms driven as in Dream's station (joint PD with i2rt's gains and gravity compensation). Rigid bodies on joints; collision shapes are boxes except the knob (a cylinder). Geometry does the safety work: the lockout's channel takes the handle between its jaws; clamped, its 74 mm body lies on the deadfront, so the handle cannot pivot toward ON without driving the body into the deadfront; the shut lever is a cover that encloses the knob on top, bottom and end (open, the knob is fully exposed past the side plates' end, as on the real part); and the padlock's shackle through the two 8.3 mm holes (the channel's loop and the lever's tab, flat tabs one over the other at the knob end, 3 and 4 mm thick in the sim; square in the sim, 0.97 mm play around the 6.35 mm shackle) stops the lever opening. Rules stand in for parts inside the devices: the knob's thread is a screw coupling (the jaw's drive target follows the knob's turns at 1.25 mm per turn, an inferred M8 pitch) with thread friction as a self-locking torque limit (0.001 N m + 0.2 x 8 mm x clamp force); the clamp force is that drive pressing the jaw against the handle (150 N/mm compliance, inferred); the shut lever cams the jaw 0.10 mm further (the instruction sheet says closing the lever tightens the grip; the cam travel is inferred); the lever's pin friction holds it where it is put, and it snaps shut under 6 degrees; each breaker handle has an over-centre spring detent (0.5 N m/rad; 0.24 N m tips a free handle to ON); the padlock's shackle spring (150 N/m), 16 mm pop-up and latch are rules, and it has no key. The knob is rolled from the front with one fingertip in short strokes, as the video shows a thumb doing; which way a stroke turns it depends on exactly where the tapered fingertip meets it, so the scripted demo reads the knob and changes stroke height or direction when a stroke loosens it. The YAM's fingertip transmits only about 0.01 N m to the knob, so the knob only brings the jaws onto the handle (2 N or more, held 1 s with the hand off the knob) and the lever's cam supplies the rest (about 15-20 N in all, which holds the 20 N pull through the serrated jaws). The serrated jaws are flat faces parallel to the handle's flanks at OFF with a friction coefficient of 1.5; the lockout goes on along the tilted handle. Dimensions follow the 493B data sheet (74 x 19.8 x 18.9 mm, 8.3 mm holes, 53 g); a single QO handle fits between the side plates, and the top plate's slot over the handle is cut 6 mm deep instead of 11.4 mm so the plate still rests on the handle before it is clamped (the handle shows through it; the jaws are light zinc so they read against the red channel) and the QO120 data sheet (19 mm pole, 76 mm length, 10 mm handle, ON 22 / OFF 21 degrees); the jaw positions, knob diameter (15 mm), hole position, lever shape and the handle's thickness (7 mm) and length (19 mm) are read off drawings and photos. The small moving parts are 8-10 g each (heavier than real) for solver stability. The load centre is a simplified 6-space tub with a flat deadfront (no cover screws, wiring or directory card); its 4 cm centre strip is wider than most so the circuit labels can be 18 x 17 mm; the labels are visual only. The handle force of a real QO breaker is not published; the after-release check's forces are our choice. The after-release check is applied in the simulator after the recording ends: 20 N on the lever's middle away from the panel for 1 s, 0.1 N m loosening on the knob for 1 s only if the lever opened past 20 degrees, 20 N pulling the lockout off for 1.5 s, then 0.3 N m on circuit 4's handle toward ON for 1 s. One layout; circuit 4 starts OFF, so switching it off is not part of the task.
