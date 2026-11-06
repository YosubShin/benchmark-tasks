# Put a group lockout hasp on a disconnect handle and padlock it

A disconnect switch's handle is at OFF. The robot must apply a group lockout hasp the way the first worker in a group
lockout does: take a Master Lock 420 scissor hasp from its holder, open its jaws, close them through the handle's
lock-off hole (and the fixed ear behind it), then put its own padlock through one of the hasp's tail holes and snap
it shut. One hand holds the hasp while the other opens and closes the jaws. Scoring checks the lockout physically:
once the robot has let go, the jaws are pried and the handle is pushed toward ON, and the padlocked hasp must hold
both. A hasp that is closed on the handle but not padlocked opens under the same check and comes off.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file (`hasp_apply.py`)
in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream` policy. Its props are
articulated (the hasp's jaws, the handle, the padlock's shackle), which the one-file format cannot declare, so the
file uses `haspapply_sim` (this folder), which adds them to the station at run time (`haspapply_sim/station.py`
documents exactly what it patches; it requires yam-isaac-lab-arena at commit `0059d02`, the Brev launchable's pin).

## How the real thing works

The Master Lock 420 is a scissor hasp: two steel jaws riveted together, each a part of the jaw ring above the
rivet and a red vinyl-coated tail with six lock holes below it. Spreading the tails opens the ring (the jaw tips part at the top);
pressing the tails together closes it, and only then do the two tails' holes line up. The worker spreads the tails,
hooks one jaw through the lock-off hole of the switch handle at OFF (on a heavy-duty safety switch the handle's hole
lines up with a hole in a fixed bracket only at OFF), closes the jaws, and hangs a padlock through a pair of aligned
tail holes; each later worker adds a padlock to a free hole. A padlock through the holes keeps the tails from
spreading, so the ring stays closed through the handle and the handle cannot move to ON. Without a padlock the jaws
open with a light pull and the hasp slides off. (Brady's hasp video shows both hands, each holding one tail,
spreading the jaws, hooking the jaw on a disconnect handle and squeezing it shut; Master Lock's 420 photos show the
open and closed hasp and the hasp on a Square D switch.)

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/hasp-apply-lockout --no-deps         # registers the policy hasp-apply-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream -P record=false \
  -E task=tasks/hasp-apply-lockout/hasp_apply.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Lock out this switch with the red hasp and your purple padlock: take the hasp out of its holder, open its jaws, close them through the hole in the switch handle, then put the padlock's shackle through a hole in the hasp and snap it shut." \
  --scorer task_score --max-steps 4800 --store-frames --log-dir ~/yam-eval/hasp-apply-dream
```

The scripted demo (closed loop: it reads the hasp's, handle's and padlock's poses and the jaw angle from the
simulator, like a scripted demo with privileged state, and sends the same 14-D joint action a policy does, through
yam_arena's IK):

```bash
inspect-robots run --embodiment yam_arena --policy hasp-apply-demo \
  -E task=tasks/hasp-apply-lockout/hasp_apply.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 4800 --store-frames --log-dir ~/yam-eval/hasp-apply-demo
```

`task_score` prints the trial's score (1.0 on success, otherwise the partial credit in task.json, or 0 after the
handle was turned toward ON). Videos: `inspect-robots video <log>.json` writes one MP4 per camera.

- **Episode end.** Once the shackle is shut through a hole pair of the hasp on the handle and both grippers have
  stayed at least 6 cm clear of the padlock and of the hasp's ring for 2 s, the after-release check runs inside that
  control step, off camera (3 s of simulated time: 0.3 N m prying the tails apart for 1.5 s, then 1.5 N m on the
  handle toward ON for 1.5 s), and the episode ends; the stored frames stop at the release. Otherwise the run ends
  at `--max-steps`.
- **Control run.** `HASP_CONTROL=1` with `--policy hasp-apply-demo`: the demo puts the hasp on the handle and closes
  it but brings no padlock; the after-release check then starts once both grippers have been clear for 2 s. It must
  score below 1.
- **Partial credit.** 0.25 needs purposeful progress: the hasp's jaws open at least 15 degrees while its jaw ring's
  centre is within 6 cm of the middle of the two lock-off holes (the hasp opened and brought up to the handle). A
  hasp that is only knocked out of its holder, dropped or moved scores 0. The demo and control scores were re-scored
  under this rule from their logs: both pass the 0.25 condition on the way (jaws open about 38 degrees, ring about
  4.5 cm from the holes), and their final levels (1.0 and 0.5) come from the higher milestones, so they are unchanged.
- **Seeds.** One fixed layout (`-E object_jitter=0`); seeded jitter is not used. PhysX on the GPU is not
  bit-deterministic, which is why the demo corrects its targets from measurements.
- **Runtime per trial**: see "Measured" below; Isaac Sim's first start on a new machine adds several minutes.

## Measured

RTX-class GPU shared with other Isaac Sim processes, Isaac Sim shader cache warm, `-E object_jitter=0`, 20 Hz,
320x176 frames:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `hasp-apply-demo` | `task_score: 1` (hasp taken out of its holder by jaw A's ring; jaw B pushed open to 38 degrees from inside the ring; jaw A's bar threaded through the handle's and the ear's holes in corrected stages; jaw B pushed back shut (1.3 degrees as it hangs); padlock's short leg through the bottom free-side hole pair and latched; after-release check passed) | 3,216 (161 s), then the 3 s check off camera | about 16 minutes |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`) | `task_score: 0` (the arms move around the hasp holder and the padlock and nudge them; the hasp is never opened near the handle, nothing is put on the handle, the handle does not move) | 4,800 (step limit) | about 16 minutes (7 expired Dream sessions were retried) |
| Control: `HASP_CONTROL=1`, `hasp-apply-demo`, hasp closed on the handle, no padlock | `task_score: 0.5` (the check pries the jaws open to 40 degrees, the joint's stop: the hasp would come off) | 2,236, then the check | about 10 minutes |

In a development run of the same demo the after-release check measured the padlocked hasp's jaws opening 1.7 degrees
under 0.3 N m and the handle turning 12.2 degrees under 1.5 N m (the ring's play in the 12 mm holes), then springing
back to OFF. The demo's wall time is mostly its IK, which it solves in Python every control step, and the shared GPU.

## Layout

| Path | What |
|---|---|
| `hasp_apply.py` | the task file: description for agents, the props and task, scoring |
| `haspapply_sim/station.py` | the run-time extension of yam_arena (props, per-step task events, off-camera after-release check) |
| `haspapply_sim/task.py`, `scorer.py`, `spec.py` | the mechanism (handle detent, rivet friction, shackle latch, after-release check), hazard and milestones, layout and instruction |
| `haspapply_sim/assets/`, `haspapply_sim/usd/` | the props (`hasp_props.py`: switch side wall, handle and ear, Master Lock 420 hasp, stands; `padlock_410.py`: the padlock), generated with `python -m haspapply_sim.usd.hasp_props haspapply_sim/assets` |
| `haspapply_sim/demo.py`, `haspapply_sim/policy.py` | the scripted demo and its Inspect Robots policy `hasp-apply-demo` |

## References

- Master Lock 420 lockout hasp (1 in jaw, 4-1/2 in long, 1-3/4 in wide, six locks, shackles up to 9/32 in):
  https://www.masterlock.com/business-use/product/420 and
  https://solutions.masterlock.com/Technical_Literature/Master_Lock/Workplace_Safety%3A_Lockout_Tagout/Lockout_Hasps/Lockout_Hasp%3A_420
- Master Lock 410 Zenex thermoplastic safety padlock (410PRP): https://www.masterlock.com/business-use/product/410PRP
- Square D heavy-duty safety switch, side-operated handle with padlock provision (HU361):
  https://www.se.com/us/en/product/HU361/
- Brady, "Brady Lockout Hasp | How To Install and Use" (both hands spread the tails, hook the jaw on a disconnect
  handle's hole, close it; padlocks hang from the tail holes): https://www.youtube.com/watch?v=-afpkxEjKZ4
- OSHA 29 CFR 1910.147, control of hazardous energy (group lockout, 1910.147(f)(3)):
  https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.147
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX at 240 Hz in Isaac Lab-Arena, both arms driven as in Dream's station (joint PD with i2rt's gains and gravity
compensation). What the geometry enforces: the hasp's jaw A top bar passes through the handle's 12 mm lock-off hole
and the fixed ear's hole, so the handle cannot turn while the ring is closed (the bar jams in both holes); a shackle
leg through a lined-up hole pair of both tails stops the jaws opening; with the jaws open the ring's gap lets the
lock-off plates in and out. Rules: the handle's over-centre detent is a spring drive toward the nearer end (1.0 N m to
push it over, inferred); the hasp's rivet friction is a drive whose target follows the jaw with a 0.6 degree dead
band, so the jaws stay where they are put against up to about 0.01 N m and move freely beyond (inferred; a real 420
moves stiffly by hand and stays where put; nothing in the hasp holds it shut, as on the real part); the padlock's
shackle spring (150 N/m), 16 mm pop-up and latch within 2 mm of shut are rules, and the lock has no key.
Published dimensions: the hasp's 1 in jaw, 4-1/2 in length, 1-3/4 in width and six holes; the padlock's body and
shackle. Inferred, to be measured on real parts: the hasp's 2 mm jaw plates, 8 mm square holes, hole spacing and
rivet position; all handle dimensions (12 mm square holes, 3 mm plates, ear 3 mm behind the tab, 90 degree throw).
Simplified: the enclosure is a plain box with no door; the lever points up at OFF and turns sideways to ON (the real
Square D lever points down at OFF), so it clears the hanging hasp; the handle stands on a shaft boss 4.5 cm out from
the wall so the open jaw clears it; collision shapes are boxes (the round jaw ring is an octagon of flat bars, round
rods are square); the two jaws do not collide with each other (they turn only about the rivet, so they cannot
interpenetrate); the hasp holder is a plain slotted block standing in for a lockout station; the after-release check
is applied as torques in the simulator after the recording ends. The YAM's fingertips are too wide to spread the
tails the way a person's fingers do, so the demo opens the jaws by pushing jaw B's bar outward from inside the ring
with arm B's closed fingertips, and closes them by pushing jaw B's tail back from beyond its edge; arm A holds the hasp
by jaw A's ring throughout. Forces involved are small: about 0.5 N opens the jaws against the rivet friction; the
real 420's opening force is not published.
