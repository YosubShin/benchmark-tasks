# Lock out a named miniature breaker with a spring-pin lockout, two hands, and a padlock

A row of five miniature circuit breakers sits on a DIN rail. The robot must lock out the one with the blue label with
a push-button pin-out breaker lockout. Its two wire pins swing outward only while its button is held in, so it takes two
hands: one holds the lockout by its body, lined up with the breaker's two pin holes. The other presses and holds the
button. The robot then pushes the lockout onto the OFF toggle and lets go of the button, and the return spring swings
the pins back in behind the edges of the holes. Then it puts a safety padlock's shackle through the hole in the
button's stem and snaps it shut. With the shackle in the stem, the button can no longer be pressed far enough to free
the pins.

Scoring checks the lockout physically once the robot has let go. A reviewer presses the button, pulls the lockout off
while still pressing, and pushes the breaker toward ON. The lockout must hold through all three.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file
(`pinout_mcb_lockout.py`) in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream` policy.
Its props are articulated: the breaker toggles, the lockout's button and pins, and the padlock's shackle. The one-file
format cannot declare these, so the file uses `pinout_sim` (this folder), which adds them to the station at run time.
`pinout_sim/station.py` documents exactly what it patches. It requires yam-isaac-lab-arena at commit `0059d02`, the
Brev launchable's pin.

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/pinout-mcb-lockout --no-deps          # registers the policy pinout-lockout-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream -P record=false \
  -E task=tasks/pinout-mcb-lockout/pinout_mcb_lockout.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Lock out the breaker with the blue label: hold the red lockout by its body with one hand and press and hold its black button with the other, push it onto the breaker's switch, then let go of the button so its pins catch. Put the padlock's shackle through the hole in the lockout's stem and snap it shut. Leave the other breakers as they are." \
  --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/pinout-mcb-lockout-dream
```

The scripted demo is closed loop. Like any scripted demo with privileged state, it reads the lockout's and padlock's
poses and the button's position from the simulator. It sends the same 14-D joint action a policy does, through
yam_arena's IK:

```bash
inspect-robots run --embodiment yam_arena --policy pinout-lockout-demo \
  -E task=tasks/pinout-mcb-lockout/pinout_mcb_lockout.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 2400 --store-frames --log-dir ~/yam-eval/pinout-mcb-lockout-demo
```

`task_score` prints the trial's score: 1.0 on success, otherwise the partial credit in task.json, or 0 after a hazard.
Videos: `inspect-robots video <log>.json` writes one MP4 per camera.

- **Episode end.** The after-release check starts once two things hold: the shackle is shut through the stem hole, and
  both grippers have stayed at least 6 cm clear of the lockout and padlock for 1 s. The check runs inside that control
  step, off camera, and takes 3.5 s of simulated time:
  1. 20 N on the button toward the board for 1 s.
  2. The 20 N press kept on while the lockout is pulled off with a net 20 N for 1.5 s.
  3. 10 N pushing the lockout up for 1 s, while the blue toggle carries 0.15 N m toward ON.

  The episode then ends. The stored frames stop at the release. If the check never starts, the run ends at
  `--max-steps`.
- **Control run.** Set `PINOUT_LOCKOUT_CONTROL=1` and use `--policy pinout-lockout-demo`. The demo fits and sets the
  lockout but never brings the padlock. Once both arms have moved away, it forces the after-release check. The run must
  score below 1.
- **Seeds.** One fixed layout (`-E object_jitter=0`); seeded jitter is not used. PhysX on the GPU is not
  bit-deterministic, which is why the demo corrects its targets from measurements.
- **Runtime per trial**: see "Measured" below. On a new machine, Isaac Sim's first start adds several minutes.

## Measured

Shared RTX-class GPU (other Isaac Sim processes running at the same time), Isaac Sim shader cache warm,
`-E object_jitter=0`, 20 Hz, 320x176 frames:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `pinout-lockout-demo` | `task_score: 1`. Arm B picks the lockout by its body and holds it 4 cm out from the blue breaker. Arm A presses the button home: 12.4 mm of 13 mm, pins spread 12.4 degrees. Both move in together, and the nose stops 0.28 mm from the breaker face. Arm A lets the button out with arm B still pushing: the pins swing in and the lockout is set. Arm A threads the padlock's short leg through the stem hole; arm B pushes the shackle shut. After-release check passed: the button went in 2.3 mm under 20 N (the shackle stopped it), the lockout moved 0.3 mm under the press-and-pull, and the toggle held at 0 degrees. | 1,932 (96.6 s), then the 3.5 s check off camera | about 60 minutes |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`) | Not scored. At step 2,097 the official Dream client raised `PolicyError: q_l_chunk gripper leaves 0..1: [-0.051, 0.797]` (a gripper command below 0), and the trial errored. Up to then both arms hovered near their start; neither the lockout nor the padlock was touched, and there were no hazard events (it would have scored 0). One run only. | 2,097 (errored) | about 48 minutes |
| Control: `PINOUT_LOCKOUT_CONTROL=1`, `pinout-lockout-demo`, no padlock | `task_score: 0.5`. The lockout is fitted and set the same way. In the forced check, the 20 N press takes the button its full 13 mm, the pins swing out, the pull takes the lockout straight off, and the toggle goes to ON (28 degrees). | 1,086, then the check | about 28 minutes |

Wall time is mostly the demo's IK, which it solves in Python every control step, plus the shared GPU. Expect it to be
several times faster on an unshared machine.

## Layout

| Path | What |
|---|---|
| `pinout_mcb_lockout.py` | the task file: description for agents, the props and task, scoring |
| `pinout_sim/station.py` | the run-time extension of yam_arena (props, per-step task events, off-camera after-release check) |
| `pinout_sim/task.py`, `scorer.py`, `spec.py` | the mechanism (toggle detents, button spring, shackle latch, after-release check), hazards and milestones, layout and instruction |
| `pinout_sim/assets/`, `pinout_sim/usd/` | the props (`breaker_props.py`: breaker row with pin holes, the lockout with its button and mimic-coupled pins, board, stands; `padlock_410.py`: the padlock), generated with `python -m pinout_sim.usd.breaker_props pinout_sim/assets` |
| `pinout_sim/demo.py`, `pinout_sim/policy.py` | the scripted demo and its Inspect Robots policy `pinout-lockout-demo` |

## References

- Brady, "Miniature Circuit Breaker (MCB) Lockout Devices Overview" (pin-out devices: press and hold the button to
  spread the pins, push on, release so the pins set in the breaker's holes): https://www.youtube.com/watch?v=3SyLe04HFWg
- Brady 090844 miniature circuit breaker lockout, pin-out standard (dimensions, photos):
  https://www.standardelectricsupply.com/Brady-90844-Circuit-Breaker-Lockout
- Brady 090844 specification (push button, 9/32 in maximum shackle): https://uk.rs-online.com/web/p/lockouts/8113034
- ABB S201-C16 miniature circuit breaker (2CDS251001R0164): https://automa.net/product/abb/2cds251001r0164-63a37b97ca04b4f2592d3c55
- Master Lock 410 Zenex thermoplastic safety padlock (410PRP): https://www.masterlock.com/business-use/product/410PRP
- OSHA 29 CFR 1910.147, control of hazardous energy: https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.147
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX at 240 Hz in Isaac Lab-Arena. Both arms are driven as in Dream's station: joint PD with i2rt's gains and
gravity compensation. Everything is rigid bodies on joints, with box colliders.

**What geometry does.** Geometry does the safety work:

- The lockout's nose pocket encloses the OFF toggle, so the toggle cannot swing to ON while the lockout is on.
- The two steel pins are levers pivoted in the nose. Released, their inward-pointing feet sit behind the 3 mm face wall
  between each breaker's two holes, so the lockout cannot be pulled off.
- The padlock's shackle through the stem's 7.1 mm hole meets the body's end face after a few millimetres of button
  travel (2.3 mm measured in the demo's check, 3.4 mm in a development run). That is under 30% of the button's 13 mm
  travel, so the pins stay hooked and the lockout stays on.

**What stands in for hidden parts.**

- The cam inside the body that turns the button's travel into the pins' swing (13 degrees outward at full press,
  3.1 mm at the foot) is a PhysX mimic joint. That is a joint constraint, not task code. The button's return spring is
  its drive: 1.2 N to start the press, 3.2 N held home.
- Each toggle has an over-centre spring detent that flips it once it passes 8 degrees from the middle. This is a rule.
- The padlock's shackle spring (150 N/m), its 16 mm pop-up and its latch are rules. The padlock has no key, so once
  shut it cannot be reopened in the sim.

**What is not published, so is our choice.**

- The breakers' pin holes: 4.5 by 4 mm, 8 to 13.5 mm above and below the toggle's line.
- The pins' shape and swing.
- The button's travel and spring force. The real spring force is unpublished, so the sim's is set low enough for the
  arms' soft wrists. A stiffer real spring would only need a firmer press.
- The toggle (7 x 7 x 15 mm, +-25 degrees). Its collider stops at the breaker face, so the lever's root inside the
  housing cannot touch the pin feet.
- The 20 N and 10 N check forces.

**Variant.** The real Brady pin-out device spreads its wire pins when the button is pressed (Brady's overview
animation). The bent-pin variant is not modelled.

**Masses and padlock shape.** The moving parts are heavier than real for solver stability: button 10 g, pins 5 g each.
The padlock follows the Master Lock 410's published outline.

**How the scripted demo does it.** One hand holds the body by its sides and keeps it in place against the other hand's
press. The other hand presses with closed fingertips. The left arm has to approach the cap on a different IK branch
from its ready pose's: the warm-started IK on the ready branch runs out of reach.

**Labels.** The breakers' coloured top labels are visual only. In the first top-camera frame at 320x176, the blue label
is a few pixels across and clearly blue.

**After-release check.** It is applied in the simulator after the recording ends:

1. 20 N on the button for 1 s.
2. The 20 N press kept on while a net 20 N pulls the lockout off, for 1.5 s.
3. 10 N pushing the lockout up for 1 s, while the blue toggle carries 0.15 N m toward ON (10 N on its 15 mm lever).

**Layout.** One layout. The blue breaker starts OFF, so switching it off is not part of the task.
