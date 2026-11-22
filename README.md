# phost-lab

Headless lab for [phost](../phost): describe which implementation sits on which stage (a graph, as
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

MIT licensed.
