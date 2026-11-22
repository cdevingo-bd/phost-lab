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
| `examples/probe_peak.json` | `trampoline12:impact` | an observer: no measurable cost, trajectory unchanged |
| `examples/pd_arm.json` | `arm7:swing` | a controller (model changing) |
| `examples/spring_trampoline.json` | `trampoline12:impact` | a force law (model changing) |

## Scenes

`trampoline8/12/14:impact|rest` (a flexible membrane with balls on it, nv 138 to 462), `arm7:swing`, and `tank:boat`:
a glass tank that shuttles side to side with a boat in it, a cloth sail on the mast and three crates on the deck
(nv=622). **The water in `tank:boat` is 150 frictionless spheres, not a fluid**: MuJoCo has no fluid solver, and the boat
floats because spheres hold it up. It is a busy scene that mixes contact, a flexible body and free bodies. It is also
past the coupled-block solver's range (nv up to 512), which refuses it with an applicability error.

A scene settles before it is timed. The settled state is cached on disk (`$PHOST_LAB_CACHE`, else
`$XDG_CACHE_HOME/phost-lab`), keyed by the scene's XML and step count, so a large scene settles once.

## Watching a run

`phost_lab.live.run_live` steps native, a second native copy and the graph in lockstep from the same state, and emits a
`layout` event once and a `live` event every few steps with both poses and how far the graph and the control are from
native. The control is the yardstick: MJWarp is not bitwise repeatable, so "the graph is this far from native" only
means something next to "native is this far from itself". It is for looking, not timing; the ordinary run does the
timing. [phost-dash](https://github.com/cdevingo-bd/phost-dash) draws it.

MIT licensed.
