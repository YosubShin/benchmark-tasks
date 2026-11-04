# Carry a trapped interlock key from a breaker to the matching door

Two trapped-key interlock pairs stand on the table. Each pair is a breaker whose Kirk Type F interlock holds a
trapped key, and a fuse-compartment door bolted shut by a second Type F with the same code. The pairs are coded
differently, so a key enters only the locks of its own code. The robot must turn key A1 in breaker A1's interlock
(this locks the breaker open, and only then does the key come free), carry it to door A1's interlock, insert it and
turn it to unlock the door. Taking a key to the other pair's locks, turning the B1 interlocks, trying a bolted door
or moving a breaker handle toward ON scores 0. Scoring checks the interlock physically: key A1 must be trapped in
door A1's lock, breaker A1's handle must be held at OFF by its bolt, and door A1 must open.

The task is a [yam-isaac-lab-arena](https://github.com/kairosiann/yam-isaac-lab-arena) task file (`trapped_key.py`)
in Dream's `station` scene, with the `yam_arena` embodiment and the official `dream` policy. Its props are
articulated (each lock's plug and bolt, the breaker handles, the doors), which the one-file format cannot declare,
so the file uses `interlock_sim` (this folder), which adds them to the station at run time
(`interlock_sim/station.py` documents exactly what it patches; it requires yam-isaac-lab-arena at commit `0059d02`,
the Brev launchable's pin).

## Install (on the launchable, in Arena's venv)

```bash
source ~/IsaacLab-Arena/yam.env
git -C ~/yam-isaac-lab-arena rev-parse --short HEAD          # must print 0059d02
uv pip install -e tasks/trapped-key-interlock --no-deps       # registers the policy interlock-demo
```

## Run and score a trial

Dream-1 (20 Hz, 320x176 frames, as in the walkthrough):

```bash
inspect-robots run --embodiment yam_arena --policy dream -P record=false \
  -E task=tasks/trapped-key-interlock/trapped_key.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "Both breakers are open. Turn key A1 a quarter turn in breaker A1's interlock to lock that breaker open and free the key. Then take key A1 to the interlock on fuse compartment door A1, insert it, and turn it a quarter turn to unlock the door. Use only the locks marked A1." \
  --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/trapped-key-dream
```

The scripted demo (closed loop: it reads the key's pose and the locks' joints from the simulator, like a scripted
demo with privileged state, and sends the same 14-D joint action a policy does, through yam_arena's IK):

```bash
inspect-robots run --embodiment yam_arena --policy interlock-demo \
  -E task=tasks/trapped-key-interlock/trapped_key.py -E object_jitter=0 -E control_hz=20 -E cam_width=320 -E cam_height=176 \
  --instruction "<the same instruction>" --scorer task_score --max-steps 1800 --store-frames --log-dir ~/yam-eval/trapped-key-demo
```

`task_score` prints the trial's score (1.0 on success, otherwise the partial credit in task.json, or 0 after a
hazard). Videos: `inspect-robots video <log>.json` writes one MP4 per camera.

- **Episode end.** Once key A1 is turned in door A1's lock and both grippers have stayed at least 6 cm from its bow
  for 1 s, the after-release check (4.5 s of simulated time) runs inside that control step, off camera, and the
  episode ends; the stored frames stop at the release. Otherwise the run ends at `--max-steps`.
- **Control run.** `TRAPPED_KEY_CONTROL=1` with `--policy yam_scripted -P hold_s=12` forces the check at 3 s with
  key A1 still in breaker A1's lock: it must score 0.
- **Wrong-key check.** `TRAPPED_KEY_CONTROL=wrong_key` (same command) sets key A1 over door B1's keyway and key B1
  over door A1's, pushes each down with 5 N and turns it with 0.3 N m, and prints how far each went in and how far
  each plug turned.
- **Seeds.** One fixed layout (`-E object_jitter=0`); seeded jitter is not used. PhysX on the GPU is not
  bit-deterministic, which is why the demo corrects its key position from measurements.
- **Runtime per trial** (RTX PRO 6000, warm shader cache): see "Measured" below; Isaac Sim's first start on a new
  machine adds several minutes.

## Measured

RTX-class GPU, Isaac Sim shader cache warm, `-E object_jitter=0`, 20 Hz, 320x176 frames:

| Run | Score | Steps | Wall time per trial |
|---|---|---|---|
| `interlock-demo` | `task_score: 1` (breaker A1 bolt out 18.8 mm, key A1 out, inserted 16.0 mm deep in door A1's lock and turned; door bolt back to 1.7 mm; no hazard) | 966 (48.3 s), then the 4.5 s check off camera | about 14.5 minutes |
| `dream` (`dreamlabs/dream-1-beta`) | `task_score: 0` (the left arm hovers over breaker A1 and the right arm drifts toward the doors; key A1 is never turned) | 1,800 (step limit) | about 22 minutes |
| Control: `TRAPPED_KEY_CONTROL=1`, `yam_scripted`, key A1 left in breaker A1 | `task_score: 0` (key held, rose 1.1 mm; handle went to ON, +20 degrees; door A1 stayed bolted, moved 0.6 degrees) | 51, then the check | about 2.5 minutes |
| Wrong-key check: `TRAPPED_KEY_CONTROL=wrong_key` | key A1 on door B1's lock and key B1 on door A1's lock each stop 8.0 mm below the face under 5 N (a matching key seats 15.7 mm deep); both plugs stay at 90 degrees | 40, then the check | about 2.5 minutes |

The demo's wall time is mostly its IK, which it solves in Python every control step.

## Layout

| Path | What |
|---|---|
| `trapped_key.py` | the task file: description for agents, the props and task, scoring, control modes |
| `interlock_sim/station.py` | the run-time extension of yam_arena (props, per-step task events, off-camera check) |
| `interlock_sim/task.py`, `scorer.py`, `spec.py` | the mechanism (handle detents, function test), hazards and milestones, layout and instruction |
| `interlock_sim/assets/*.usda`, `interlock_sim/usd/props.py` | the props, and the script that generates them from the dimensions in task.json |
| `interlock_sim/demo.py`, `interlock_sim/policy.py` | the scripted demo and its Inspect Robots policy `interlock-demo` |

## References

- Kirk Key Interlock Type F datasheet (dimensions of the SD single-cylinder lock, bolt and throw):
  https://www.sentricsafetygroup.com/us/wp-content/uploads/sites/2/2026/05/f-kirk01-2602.pdf
- Kirk white paper, "Key Interlock Operation" (key removable only with the bolt extended):
  https://sentricsafetygroup.com/fr/wp-content/uploads/sites/3/2019/12/KirkWP-KeyInterlockOperation-1.pdf
- Kirk white paper, "Examples of Key Interlock Systems" (breaker to compartment door transfer):
  https://sentricsafetygroup.com/fr/wp-content/uploads/sites/3/2019/12/KirkWP-ExamplesofKeyInterlockSystems.pdf
- Square D PowerPact H-frame molded case circuit breakers (frame size):
  https://www.se.com/us/en/product-range/7564-powerpact-h-frame-molded-case-circuit-breakers/
- I2RT YAM arm: https://i2rt.com/products/yam-6-dof-arm

## Fidelity details

PhysX at 240 Hz in Isaac Lab-Arena, both arms driven as in Dream's station (joint PD with i2rt's gains and gravity compensation). Rigid bodies on joints; geometry does the interlocking: the key's shank drives a plug whose crank pin moves the bolt (a scotch-yoke stand-in for Kirk's unpublished cam); the key's bit can pass the face plate only at the bolt-extended angle, so the key is trapped until the bolt is out; an extended bolt jams the breaker handle short of ON; and a door bolt through the keeper holds the door shut. The coding is warded, not pin-tumbler: each key's shank cross-section fits only the plug socket of its own code, so a wrong key stops on the plug's top and cannot turn it, whereas a real wrong-code Kirk key enters and will not turn. The keys are stand-ins, not flat 7-pin Kirk blades. The breakers lie face up and the door interlocks sit on the table with their cylinders up; real breaker interlocks are usually on a vertical panel with the key horizontal, so this is a bench mock-up chosen so the key can be turned with fingers pointing down. The breakers and doors are generic boxes and plates, not specific products; the coloured labels are our addition so the pairs can be told apart at camera resolution. Turning resistance is a small damping term (the real torque is not published, likely 0.1-0.5 N m), the handle detent and all stops are joint rules, and the after-release check is applied as forces and torques in the simulator after the recording ends.
