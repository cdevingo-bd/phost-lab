# phost-lab

Headless lab for [phost](../phost): describe which implementation sits on which stage (a graph, as
JSON), run it on a named scene in an isolated process, and get one JSON result with step time,
speedup, accuracy against a native control, per-stage metrics and a verdict.

```
phost-lab --plugin coupledchol.phost_impl list
phost-lab validate examples/coupled.json
phost-lab run examples/coupled.json --scene trampoline12:impact --worlds 1 8 32 --out result.json
```

Accuracy is judged against the native-vs-native difference of the same run (MJWarp is not bitwise
repeatable), and a ratio is reported only when that control is above rounding noise.
Run on an idle GPU: the result records how many other GPU processes were running.

MIT licensed.
