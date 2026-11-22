# phost-lab

Headless lab for [phost](https://github.com/cdevingo-bd/phost): describe which implementation sits on which stage (a graph, as
JSON), run it on a named scene in an isolated process, and get one JSON result with step time,
speedup, accuracy against a native control, per-stage metrics and a verdict.

```
phost-lab --plugin phost_coupledchol.phost_impl list
phost-lab validate examples/coupled.json
phost-lab run examples/coupled.json --scene trampoline12:impact --worlds 1 8 32 --out result.json
```

A graph may select any phost stage (`phost-lab list` shows them). A graph with a model-changing stage is not
gated on matching native, because the host is not the truth for it.

Accuracy is judged against the native-vs-native difference of the same run (MJWarp is not bitwise
repeatable), and a ratio is reported only when that control is above rounding noise.
Run on an idle GPU: the result records how many other GPU processes were running.

## Examples

| graph | scene | what it shows |
| --- | --- | --- |
| `examples/native.json` | any | the control: native against native reads 1.00x |
| `examples/coupled.json`, `coupled_structural.json` | `trampoline12:impact` | the coupled-block Cholesky, 2.4x at nv=330 |
| `examples/effective_direct.json` | `trampoline12:discrete` | a direct solve of the discrete step's effective metric: **0.57x**, a swap that loses (see below) |
| `examples/probe_peak.json` | `trampoline12:impact` | an observer: no measurable cost, trajectory unchanged |
| `examples/pd_arm.json` | `arm7:swing` | a controller (model changing) |
| `examples/spring_trampoline.json` | `trampoline12:impact` | a force law (model changing) |

## Scenes

`trampoline8/12/14:impact|rest` (a flexible membrane with balls on it, nv 138 to 462), `trampoline8/12/14:discrete` (the
same membrane under the discrete integrator, where the host solves an effective metric each step and the
`effective.solve` stage applies), `arm7:swing`, and `tank:boat`:
a glass tank that shuttles side to side with a boat in it, a cloth sail on the mast and three crates on the deck
(nv=622). **The water in `tank:boat` is 150 frictionless spheres, not a fluid**: MuJoCo has no fluid solver, and the boat
floats because spheres hold it up. It is a busy scene that mixes contact, a flexible body and free bodies. It is also
past the coupled-block solver's range (nv up to 512), which refuses it with an applicability error.

A scene settles before it is timed. The settled state is cached on disk (`$PHOST_LAB_CACHE`, else
`$XDG_CACHE_HOME/phost-lab`), keyed by the scene's XML and step count, so a large scene settles once.

## Where the time goes

```
phost-lab --plugin phost_coupledchol.phost_impl profile --graph examples/effective_direct.json --scene trampoline12:discrete --worlds 1 8
```

times each phase of the step (kinematics, inertia, collision, constraint build, dynamics, solver, integrate) as its own
CUDA graph, native next to the graph installed, plus the whole step as one graph. Under the discrete integrator it also
times the effective-metric solve alone. A swap can only pay for itself on a phase that matters, and this shows which:

```
trampoline12:discrete, nv=330, 1 world, RTX PRO 2000 laptop GPU          native ms   graph ms
  velocity+actuation+accel                                                  0.255      1.559
    of which effective solve                                                0.153      1.454
  solver                                                                    1.830      1.829
  full step (one graph)                                                     1.688      2.986
```

The host's own effective solve is 9% of a 1.7 ms step here; a direct solve that costs 1.45 ms cannot help. Phases are
read from MuJoCo Warp's private modules, so this is a diagnostic tied to the build it runs on. Per-phase graphs add
a small gap each, so the phase sum sits above the full-step time. Replaying `integrate` alone would drift the state to
NaN, so it is replayed with the state restored after it and the cost of the restore subtracted.

## Watching a run

`phost_lab.live.run_live` steps native, a second native copy and the graph in lockstep from the same state, and emits a
`layout` event once and a `live` event every few steps with both poses and how far the graph and the control are from
native. The control is the yardstick: MJWarp is not bitwise repeatable, so "the graph is this far from native" only
means something next to "native is this far from itself". It is for looking, not timing; the ordinary run does the
timing. [phost-dash](https://github.com/cdevingo-bd/phost-dash) draws it.

MIT licensed.
