# Lock and tag out a disconnect handle

A disconnect switch's handle is at OFF. The robot must do a full OSHA 1910.147 lock-and-tag: thread a danger tag's
brass grommet onto its open padlock's shackle, put the shackle through the handle's lock-off hole (and the fixed ear
behind it), and snap the padlock shut. Scoring checks both parts physically: once the robot has let go, the handle is
pushed toward ON (the padlock must hold it) and the tag is pulled down with a spring scale (it must still hang from the
shackle). A padlock without its tag scores partial credit; a tag that only lies on the lock is pulled away and never
counts.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file (`lock_and_tag.py`)
in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream` policy. Its props are
articulated (the handle, the padlock's shackle, the tag's hinged plates), which the one-file format cannot declare, so
the file uses `locktag_sim` (this folder), which adds them to the station at run time (`locktag_sim/station.py`
documents exactly what it patches; it requires yam-isaac-lab-arena at commit `0059d02`, the Brev launchable's pin).

## How the real thing works

The worker turns the disconnect to OFF, puts a personal padlock through the handle's lock-off provision and attaches a
danger tag that names the worker to that lock. On a Square D heavy-duty safety switch the operating handle carries a
lock-off hole that lines up with a hole in a fixed part of the enclosure only at OFF; a shackle through both stops the
handle being turned to ON. A Master Lock 410 Zenex padlock is unlocked with its key: its shackle pops up (and on the
real lock can swivel about the long leg). A lockout tag (Brady 65502: 5-3/4 x 3 in polyester laminate with a 3/8 in
metal grommet; Master Lock 497A: the same size with a 7/16 in brass grommet, rated over 50 lb pull-out) is hung by
passing the open shackle's short leg through the grommet; once the shackle is snapped shut the grommet is trapped on
the closed loop and the tag hangs from the shackle. Brady's safety-padlock animation shows the sequence (shackle
swung out, short leg through the hole, tag slipped on, shackle snapped shut), and Brady's 65502 product photo shows a
padlock through a disconnect handle's hole with the tag on its shackle.

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/lock-and-tag --no-deps               # registers the policy lock-tag-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream -P record=false \
  -E task=tasks/lock-and-tag/lock_and_tag.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Lock and tag out this switch: thread the danger tag's grommet onto the open shackle of the red padlock, then put the shackle through the hole in the switch handle and snap the padlock shut." \
  --scorer task_score --max-steps 4800 --store-frames --log-dir ~/yam-eval/lock-and-tag-dream
```

The scripted demo (closed loop: it reads the padlock's, tag's and handle's poses from the simulator, like a scripted
demo with privileged state, and sends the same 14-D joint action a policy does, through yam_arena's IK):

```bash
inspect-robots run --embodiment yam_arena --policy lock-tag-demo \
  -E task=tasks/lock-and-tag/lock_and_tag.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 4800 --store-frames --log-dir ~/yam-eval/lock-and-tag-demo
```

`task_score` prints the trial's score (1.0 on success, otherwise the partial credit in task.json, or 0 after the
handle was turned toward ON). Videos: `inspect-robots video <log>.json` writes one MP4 per camera.

- **Episode end.** Once the shackle has latched and both grippers have stayed at least 6 cm clear of the padlock's
  body and of the tag's grommet for 2 s, the after-release checks run inside that control step, off camera (3 s of
  simulated time: 1.5 N m on the handle toward ON for 1.5 s, then a 3 N downward pull on the tag's bottom plate for
  1.5 s), and the episode ends; the stored frames stop at the release. Otherwise the run ends at `--max-steps`.
- **Control run.** `LOCKTAG_CONTROL=1` with `--policy lock-tag-demo`: the demo runs unchanged, but as the shackle
  latches the task takes the tag off the shackle and lays it against the side of the padlock's body (a tag that was
  never threaded, only put next to the lock). It must score below 1: the 3 N pull takes the tag away and the score is
  0.5, a padlock without its tag.
- **Seeds.** One fixed layout (`-E object_jitter=0`); seeded jitter is not used. PhysX on the GPU is not
  bit-deterministic, which is why the demo corrects its targets from measurements.

## Measured

RTX-class GPU, Isaac Sim shader cache warm, `-E object_jitter=0`, 20 Hz, 320x176 frames:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `lock-tag-demo` | `task_score: 1` (tag threaded on the short leg and lifted out of its pocket; lock rolled, its plates-gap put over the handle's tab and ear, drawn back so the short leg passed through both holes; rolled back, handed to the right arm, swung down to hang and pushed shut; after-release checks: handle turned 4.7 degrees under 1.5 N m, grommet 0.5 mm from the shackle after the 3 N pull) | 2,524 (126 s), then the 3 s checks off camera | about 13 minutes |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`) | `task_score: 0` (the left arm hovers over the padlock and the tag pocket and nudges them; the tag is never threaded, the padlock never leaves its column, the handle does not move) | 4,800 (step limit) | about 17 minutes (2 expired Dream sessions were retried) |
| Control: `LOCKTAG_CONTROL=1`, `lock-tag-demo` | `task_score: 0.5` (same lockout; the tag laid against the padlock's body instead of threaded is pulled away by the 3 N check: grommet 37 mm from the shackle; handle held at 4.7 degrees) | 2,524, then the checks | about 13 minutes |

The demo's wall time is mostly its IK, which it solves in Python every control step. The demo is closed loop but has
open-loop stretches (the tag riding on the shackle, the hand-over); PhysX on the GPU is not bit-deterministic, and
in development runs the tag sometimes slid off during the carry or the hand-over failed, which ends the trial below 1.

## Layout

| Path | What |
|---|---|
| `lock_and_tag.py` | the task file: description for agents, the props and task, scoring |
| `locktag_sim/station.py` | the run-time extension of yam_arena (props, per-step task events, off-camera after-release checks) |
| `locktag_sim/task.py`, `scorer.py`, `spec.py` | the mechanism (handle detent, shackle latch, threaded tests, after-release checks), hazard and milestones, layout and instruction |
| `locktag_sim/assets/`, `locktag_sim/usd/` | the props (`locktag_props.py`: switch wall, handle and ear, danger tag, stands; `tag_texture.py`: the tag's print; `padlock_410.py`: the padlock), generated with `python -m locktag_sim.usd.locktag_props locktag_sim/assets` |
| `locktag_sim/demo.py`, `locktag_sim/policy.py` | the scripted demo and its Inspect Robots policy `lock-tag-demo` |

## References

- Brady 65502 "DANGER - DO NOT OPERATE" lockout tag (5-3/4 x 3 in, B-837 polyester, 3/8 in metal grommet, 50 lb),
  distributor data sheet and product photo: https://www.standardelectricsupply.com/Brady-65502-Lockout-Tag
- Master Lock 497A "Do Not Operate" safety tag (5-3/4 x 3-1/8 in laminated polyester, 7/16 in brass grommet, over
  50 lb pull-out): https://www.masterlock.com/products/product/497A
- Master Lock 410 Zenex thermoplastic safety padlock (410RED): https://www.masterlock.com/business-use/product/410RED
- Square D heavy-duty safety switch, side-operated handle with padlock provision (HU361):
  https://www.se.com/us/en/product/HU361/
- Brady, "Lockout Tagout | Lockout Devices | Safety Padlocks | Demo" (shackle pops up and swings out, the short leg is
  hooked through a hasp hole, a tag is put on the shackle, the shackle is snapped shut):
  https://www.youtube.com/watch?v=CwSbc9_fqSk
- OSHA 29 CFR 1910.147, control of hazardous energy (lockout and tagout devices, 1910.147(c)(5)):
  https://www.osha.gov/laws-regs/regulations/standardnumber/1910/1910.147
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX at 240 Hz in Isaac Lab-Arena, both arms driven as in Dream's station (joint PD with i2rt's gains and gravity
compensation).

What the geometry enforces: the shackle threads the tag only through the grommet's hole (an octagon of brass
colliders, 9.5 mm across flats, around which the card is solid), so a tag can only be on the shackle by having the
short leg passed through the grommet; once the shackle is shut the grommet is trapped on the closed loop. The shackle
through the handle tab's and the fixed ear's 12 mm holes stops the handle turning toward ON (the holes line up only at
OFF). Rules: the handle's over-centre detent is a spring drive toward the nearer end (1.0 N m to push it over,
inferred); the padlock's shackle spring (150 N/m), 16 mm pop-up and latch within 2 mm of shut are rules, and the lock
has no key. "On the shackle" and "through the holes" are read from the simulator as the grommet's and the holes'
centres lying within 4 mm of the shackle's centre line (a 9.5 mm grommet or 12 mm hole around a 6.35 mm rod), which a
photo shows as the shackle passing through them.

Published dimensions: the padlock's body and shackle; the tag's 146 x 76 mm size and 9.5 mm (3/8 in) grommet hole.
Inferred, to be measured on real parts: the grommet's 20 mm flange and 2.5 mm thickness, its centre 13 mm below the
top edge, the tag's 8 g mass and bending stiffness; all handle dimensions (12 mm square holes, 3 mm plates 1 mm
apart, the hole 70 mm below the pivot, the plates 65 mm out from the side wall, the ear's arm 45 mm above the hole).

Simplified: the tag is three rigid 1.2 mm plates hinged across its width (0.02 N m/rad), not a continuously bending
0.4 mm laminate (a thinner collider lets the shackle tunnel through it). The padlock's shackle pops up 16 mm but does
not swivel about its long leg as a real 410's does; to go through the handle the lock is held legs-forward with its
plates in the gap between the body's top and the raised short leg, then drawn back so the short leg's tip passes
through both holes, which needs the 65 mm gap behind the plates; it is then rolled so its bend stands in the holes'
plane and swung down about the bend's centre into the hanging position, where the body is pushed up to snap the
shackle shut (the demo hands the lock from one arm to the other for that swing; a person would swing the shackle
out and hook it through instead). The enclosure is a plain box with no door; the lever
points up at OFF and turns sideways to ON (the real Square D lever points down at OFF); collision shapes are boxes
(round rods are square or cross-shaped rods); the after-release checks are applied as a torque and a force in the
simulator after the recording ends.
