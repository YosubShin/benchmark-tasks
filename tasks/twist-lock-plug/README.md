# Plug a twist-lock cord cap into the named receptacle and lock it

A steel plate stands on the table with two identical NEMA L5-30R twist-lock receptacles (Hubbell HBL2616) side by
side, told apart only by their engraved nameplates: CKT 7 BATTERY CHARGER (red) and CKT 9 SPARE (white). A Hubbell
HBL2611 L5-30P cord cap lies on a cradle, its black 10/3 SJOOW cord running off its back to a clamp on the table. The
robot must pick up the cap, line its three curved blades up with the CKT 7 receptacle's slots, push it straight in,
turn it clockwise until it stops, and let go. Scoring checks the lock physically: after release, the cap must hold a
60 N pull straight out. Bringing the cap to the CKT 9 receptacle, or holding it more than 10 degrees crooked with its
blades partly in, scores 0.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file
(`twist_lock_plug.py`) in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream` policy.
The one-file format cannot declare its props, which are articulated: the receptacles' contact springs, and the cord,
an anchored chain of jointed capsules with the cap on its end. The file therefore uses `twistlock_sim` (this folder),
which adds them to the station at run time. `twistlock_sim/station.py` documents exactly what it patches. It requires
yam-isaac-lab-arena at commit `0059d02`, the Brev launchable's pin.

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/twist-lock-plug --no-deps             # registers the policy twistlock-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream -P record=false \
  -E task=tasks/twist-lock-plug/twist_lock_plug.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Plug the twist-lock cord cap into the receptacle labelled 'CKT 7 BATTERY CHARGER' (the red nameplate) on the plate. Push it straight in, turn it clockwise until it stops to lock it, and let go." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/twist-lock-dream
```

The scripted demo is closed loop. Like a scripted demo with privileged state, it reads the cap's pose from the
simulator, and it sends the same 14-D joint action a policy does, through yam_arena's IK:

```bash
inspect-robots run --embodiment yam_arena --policy twistlock-demo \
  -E task=tasks/twist-lock-plug/twist_lock_plug.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/twist-lock-demo
```

`task_score` prints the trial's score: 1.0 on success, otherwise the partial credit given in task.json. To make
videos, run `inspect-robots video <log>.json`, which writes one MP4 per camera.

- **Episode end.** The pull test runs once the cap is home and turned past 21 degrees and both grippers have stayed
  at least 6 cm from it for 1 s. It lasts 2 s of simulated time (0.5 s ramp to 60 N, 1.5 s hold), runs inside that
  control step and off camera, and then the episode ends. The stored frames stop at the release. If the test never
  runs, the run ends at `--max-steps`.
- **Control run.** `--policy twistlock-demo -P control=true` pushes the cap home but does not turn it, lets go, and
  forces the pull test once the hand is clear. It must score below 1, because the unturned cap pulls out.
- **Seeds.** One fixed layout (`-E object_jitter=0`). Seeded jitter is not used.
- **Runtime per trial.** See "Measured" below. Isaac Sim's first start on a new machine adds several minutes.

## Measured

RTX-class GPU shared with other simulators, Isaac Sim shader cache warm, `-E object_jitter=0`, 20 Hz, 320x176:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `twistlock-demo` | `task_score: 1`. Two line-up corrections, pushed home in 14 steps of 2 mm, turned in 3-degree steps to 37.7 degrees (the slots' end), let go; the 60 N pull held. | 806 (40 s), then the 2 s pull off camera | about 15 minutes (with three other simulators sharing the GPU) |
| Control: `twistlock-demo -P control=true` (pushed home, not turned, let go; forced pull) | `task_score: 0.5` (lined up at the mouth, never turned, so 0.75 and the pull test's 1.0 are out of reach) | 708 | about 9 minutes |
| `dream` (`dreamlabs/dream-1-beta`, `-P record=false`) | `task_score: 0` (both arms move near their start, the right hand passes over the cord and the plate's corner; the cap is never lifted) | 1,800 (step limit) | about 24 minutes (Dream sometimes reported an observation as expired and was asked again, 36 times in the run) |

The demo's wall time is mostly GPU contention and its Python IK every control step; alone on the GPU a trial is
several times faster.

**Simulator stability.** Development runs used to blow up as the cap landed on the receptacle face or during the first
15 degrees of the turn (NaN, or the cap pushed through the face). The cause was the collision shapes: the cap's front
and the receptacle face were both cut into 8-degree pie slices, and wherever two slices overlapped by a thin sliver of
angle, PhysX's shortest way out of a slight overlap was sideways across the sliver, not along the axis. Pressed
together, the faces then pushed sideways and twisted instead of holding the cap out. Each full disk now collides as
one convex prism, and the earlier workaround (a limit on the cap's depenetration speed) is gone. With the robot
replaced by a scripted force on the cap (push 10 to 40 N, twist up to 1.5 N*m, sideways offsets), the old shapes
failed in 4 of 9 trials and the new ones in none of 21. After the fix, two more official `twistlock-demo` runs each
scored `task_score: 1` (810 steps, turned to 33.8 degrees, the 60 N pull held); the table above and `demo.mp4` are
from a run before the fix (the scene looks the same; only invisible collision shapes changed).

## Fidelity details

PhysX in Isaac Lab-Arena at the station's 240 Hz, both arms driven as in Dream's station (joint PD with i2rt's gains
and gravity compensation). Rigid bodies and joints. The lock is geometry: each blade has a wider foot near its tip,
and the receptacle face lets a foot through only at a notch at one end of its arc slot. Pushed home and turned, the
feet sit behind solid face, which holds the pull (in a development run the 60 N pull moved the cap out 0.2 mm); an
unturned cap comes straight out. Blade layout (not 120 degrees apart, so it fits one way), blade thickness, cap and
flange sizes are from Hubbell's dimensioned drawings; blade circle, foot size, the 35-degree turn and its clockwise
direction are measured off those drawings and how-to videos, as NEMA's configuration sheet is not public. The ground
blade's L-shaped foot is not modelled. Contact springs: behind the face, one contact per blade is pressed onto the
blade's inner face by a preloaded spring (about 4.6 N), giving up to about 5 N of insertion drag and 0.05 N*m of
turning friction; with the cap off-centre in its 0.4 mm slot clearance some contacts lose touch, so the resistance is
small and uneven. Their force and shape are guesses: real contacts are not published and grip harder (a firm wrist
turn). The cord is 24 jointed capsules (2.5 cm each) at the real cord's diameter and weight, with a bending spring
toward its starting shape; its stiffness (EI 0.0075 N*m^2) and friction are guesses. Round parts are drawn with 45
sides; each full disk (the cap's front, body and tail, the centre of the receptacle face) collides as one convex
32-sided prism, and the ring-shaped parts (slotted face, flange) as convex pieces of at most 8 degrees. The body's
ribs are visual only. The plate, box, cradle, cord route and nameplate legends are invented. The cap starts lined up
with the slots rather than at a random roll. The printed nameplate text is 2 to 3 pixels tall in the 320x176 top
camera, so there the target is told apart by its red nameplate (the legends are legible in the wrist cameras). The 60
N pull is our choice (UL 498's latch test for straight-blade plugs uses 89 N); it is applied as a force in the
simulator after the recording ends. One layout.

## Layout

| Path | What |
|---|---|
| `twist_lock_plug.py` | the task file: description for agents, props and task, scoring |
| `twistlock_sim/station.py` | the run-time extension of yam_arena (props, per-step task events, off-camera pull test) |
| `twistlock_sim/task.py`, `scorer.py`, `spec.py` | the mechanism (contact springs, cord, pull test), hazard rules and milestones |
| `twistlock_sim/assets/*.usda`, `twistlock_sim/usd/` | the props, and the script that generates them (`python -m twistlock_sim.usd.twistlock_props`) |
| `twistlock_sim/demo.py`, `twistlock_sim/policy.py` | the scripted demo and its Inspect Robots policy `twistlock-demo` |

## References

- Hubbell HBL2611 Twist-Lock plug, 30A 125V, L5-30P (dimensioned drawing, blade layout):
  https://www.hubbell.com/wiringdevice-kellems/en/products/p/179850
- Hubbell HBL2616 Twist-Lock flanged receptacle, 30A 125V, L5-30R (flange, panel cutout):
  https://www.hubbell.com/wiringdevice-kellems/en/products/p/180726
- Hubbell HBL2610 Twist-Lock receptacle, L5-30R: https://www.hubbell.com/wiringdevice-kellems/en/products/p/178515
- 10/3 SJOOW portable cord, 300 V (0.58 in outside diameter, 0.23 lb/ft):
  https://www.globalcablewire.com/products/10-3-sjoow-portable-cord-300v-ul-csa
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm
