# phost-lab

Run a graph of [phost](https://github.com/cdevingo-bd/phost) implementations on a scene, in an isolated worker, and get one
JSON result: step time, speedup, accuracy against a native control, per-stage metrics and a verdict. A graph says which
implementation sits on which stage.

Three reports need no implementation: `profile` times each phase of the step as its own CUDA graph, `opportunities` gives
each stage's share of the step and the most a replacement could gain, and `applicability` says which stages apply to which
scenes and why not.

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
| `examples/pd_chain.json` | `chain32:pluck` | the same controller on a NumPy simulator: no GPU, no MuJoCo |
| `examples/spring_trampoline.json` | `trampoline12:impact` | a force law (model changing) |

## Scenes

`trampoline8/12/14:impact|rest` (a flexible membrane with balls on it, nv 138 to 462), `trampoline8/12/14:discrete` (the
same membrane under the discrete integrator, where the host solves an effective metric each step and the
`effective.solve` stage applies), `arm7:swing`, `chain32:pluck` and `chain256:ringing` (a NumPy simulator, see
[The simulator](#the-simulator)), and `tank:boat`:
a glass tank that shuttles side to side with a boat in it, a cloth sail on the mast and three crates on the deck
(nv=622). **The water in `tank:boat` is 150 frictionless spheres, not a fluid**: MuJoCo has no fluid solver, and the boat
floats because spheres hold it up. It is a busy scene that mixes contact, a flexible body and free bodies. It is also
past the coupled-block solver's range (nv up to 512), which refuses it with an applicability error.

A scene settles before it is timed. The settled state is cached on disk (`$PHOST_LAB_CACHE`, else
`$XDG_CACHE_HOME/phost-lab`), keyed by the scene's XML and step count, so a large scene settles once.

## Steps per call

A rollout loop pays something on every call, not just every step. `--steps-per-call 1 5 20` also times calls of that many
graph launches plus a sync, fits `fixed + n * per_step` to them, and reports both for the native step and the graph, with
the speedup at each call length:

```
phost-lab run examples/effective_direct.json --scene trampoline12:discrete --worlds 1 --steps-per-call 1 5 20
        per call (ms)            1        5       20   fixed ms  per-step ms
          native             1.549    8.716   35.196      0.000       1.769
          graph              2.840   15.201   61.006      0.000       3.059
          speedup            0.55x    0.57x    0.58x
```

This measures a Warp graph-launch loop from Python, and there the fixed cost is below noise (the fit is clamped at zero).
Anything a wrapper adds on top, such as converting state between frameworks on every call, shows up only if the wrapper is
what is being timed. deform3d's notes put that at 4.5-6 ms per call for a JAX wrapper, which is more than a whole step here,
so a speedup measured per step can read very differently from one measured per call.

## Where a change can apply

```
phost-lab --plugin phost_coupledchol.phost_impl applicability --scene trampoline12:impact --scene trampoline12:discrete --scene tank:boat
```

```
stage                  trampoline12:impact  trampoline12:discrete  tank:boat
newton.linear_solve                    yes                    yes        yes
dof.passive_force                      yes                    yes        yes
control.law                            yes                    yes        yes
probe.state                            yes                    yes        yes
effective.solve                         no                    yes         no
body.wrench                            yes                    yes        yes

why not:
  trampoline12:impact: effective.solve is refused: the host only calls this solve with the discrete integrator
  trampoline12:discrete: control.law/pd does not apply: the discrete integrator builds its own effective-force solve, which bypasses this stage
  tank:boat: newton.linear_solve/coupled does not apply: nv=622 is outside 1..512
```

Stages by scenes, with the reason for each "no". Nothing is run: the matrix is read from the host's own refusal for a stage
and each implementation's own `applicable`, the same data a session uses to refuse an install. A stage added to the host, or
an implementation added by a plugin, appears here without any change to the lab. `phost_lab.opportunity.scene_for(stage, ...)`
uses the same data to pick the first scene on which a stage applies, which is how phost-author chooses where to check a
submission.

## What a stage could be worth

```
phost-lab opportunities --scene trampoline12:discrete --worlds 1 8
```

```
1 world(s), nv=330, discrete integrator: whole step 1.760 ms (graph-launch floor 0.004 ms, subtracted from each stage)
  stage                 class            native ms   share  ceiling  note
  newton.linear_solve   exact           <=   1.102<=  63%        -  upper bound: the solver phase (solver.solve), which runs the stage once per Newton iteration
  dof.passive_force     model changing       0.035     2%        -  changes the physics or only observes: not a speed swap
  control.law           model changing       0.013     1%        -  changes the physics or only observes: not a speed swap
  probe.state           observer                                     attaches after the whole step; nothing to time alone
  effective.solve       exact                0.149     8%    1.09x
  body.wrench           model changing       0.000     0%        -  changes the physics or only observes: not a speed swap
```

For each stage: how long the host's own version takes, its share of the whole step, and, for a swap that should give the same
answer (exact or approximate), the ceiling `1 / (1 - share)`: what an infinitely fast replacement would gain. Each stage's
binding says how to run the host's version once (`NativeTiming`); the lab replays it as its own CUDA graph, subtracts the
cost of replaying a graph of one empty kernel, and compares with the whole step timed on separate data. A stage whose
host version cannot run alone (the Newton solve runs inside the solver loop) is reported as the enclosing phase and marked
`<=`, an upper bound, with no ceiling. A stage that changes the physics or only observes has a share but no ceiling, because
being faster than the host is not what it is for. A stage that does not apply to the scene says why instead of a time.

Here the effective solve is 8% of the step, so no replacement can give more than 1.09x; this is the calculation that
`profile` below confirms for one replacement.

## The structure of a stage's matrix

```
phost-lab sparsity --scene trampoline12:discrete
```

```
scene trampoline12:discrete (nv=330, discrete integrator)
  effective.solve: 330 x 330, 9678 entries (8.9% dense), 300 of 330 rows coupled
    band: 65 as the host orders it, 59 after reverse Cuthill-McKee
    factorization work, in flops: dense 1.20e+07, coupled rows only 9.00e+06 (75.1%), banded 1.15e+06 (9.6%)
    flops are not time: whether a variant wins depends on how its kernel moves data (see profile)
```

For every stage whose binding can say what its system matrix looks like, the pattern of nonzeros and what a factorization
that exploits it would cost in floating-point operations, compared with a dense one. It is read from the model, not
timed, and answers whether a structure-exploiting solve has anything to exploit before one is written. Here 300 of 330 rows
are coupled, so factoring only the coupled rows saves a quarter of the work, while a banded factorization after
reordering needs a tenth. That is a count of operations: the replacement's kernel still has to be faster than the host's own
iteration (see `profile`) for it to matter. An implementation gets the same pattern through the `sparsity` extension
(phost's `docs/ADDING_A_STAGE.md`).

## The simulator

A run needs a short list of things from a simulator, and none of them is MuJoCo: load a scene and settle it, make instances of
it with some number of worlds, step one, reset it to the start, read its state, and say what device it runs on. An `Engine`
(`phost_lab.engine`) provides them, and the runner, the reports and the drivers speak only that.

```
engine.load(scene)       -> Loaded     parse and settle on the CPU; entering it opens the device scope
loaded.instance(worlds)  -> Instance   the data for that many worlds, and a phost.Host over it (instance.host)
instance.step() / reset() / state() / frame()
loaded.device                          synchronize, capture, how busy, the cost of launching nothing
```

Two engines ship. `mjwarp` is MuJoCo Warp on a CUDA device and is what every scene used until now. `chain` is a damped
spring-mass chain stepped in NumPy (`phost.hosts.chain`): no GPU, no MuJoCo, its arrays called `x`, `v` and `a`, and an
effective system that is a band of half-width 1. It offers the stages a chain has (passive force, control law, probe and the
banded effective solve) and not the ones it has no use for (`body.wrench`, `newton.linear_solve`), and the same typed
equations, implementations and checks run on it. It exists to show the boundary is not MuJoCo's.

A scene names its engine (`Scene.engine`, default `mjwarp`) and a graph names the host its stages are bound to
(`GraphSpec.host`); a graph with stages that does not match the scene's engine is refused before anything loads. Three
things follow from an engine being a different simulator:

* **The driver.** `graph` needs a device that can capture a step. On one that cannot, a run steps eagerly and says so
  (`config.driver_requested`, `meta.driver_note`) rather than claim it replayed a graph.
* **The noise floor.** MuJoCo Warp does not repeat exactly, so native against native is the yardstick for accuracy. A chain
  does repeat exactly, so that yardstick is zero and a ratio to it says nothing; an engine that reports `repeatable` is
  measured against a rounding floor instead (`cell.accuracy_basis`, `runner.ROUNDING_FLOOR`), and a wrong swap still fails.
* **The tools.** `opportunities`, `sparsity` and `applicability` read a scene through its engine. `profile` (the phases of
  `mjw.step`) and the live view (MuJoCo geometry) are tools for the `mjwarp` engine and say so on any other.

```
phost-lab run examples/pd_chain.json --scene chain32:pluck      # on the CPU: the graph's host is "chain"
```

A plugin adds an engine with `register_engine(...)` from its `register`; an engine over a framework that steps in JAX would
give its device a `capture` that returns a jitted replay and leave the rest alone.

## How the host is stepped

```
phost-lab run examples/effective_direct.json --scene trampoline12:discrete --driver eager
```

A run times and compares steps and does not care how one is launched, but the launch changes what a step costs. A `Driver`
(handed the engine's device) is that choice: `graph` (the default) captures the step once as a CUDA graph and replays it; `eager` runs the step from Python
every time, so every kernel is launched from the host thread. Everything that advances the host (the per-step time, the
`--steps-per-call` curve, the accuracy replay) goes through the driver's `advance()`, and the result records which one ran
(`config.driver`). A new way of stepping, such as a step wrapped by another framework that adds its own per-call cost, is one
class and `register_driver(...)` from a plugin's `register`; the runner does not change. The driver is separate from
`--steps-per-call`: that varies how many steps a call makes, this varies how a step is made.

The reports above run in a worker too: `run_isolated(..., report=dict(kinds=["opportunities", "structure"], worlds=[1]))`
returns them as data (`kinds` also takes `profile`, and the graph in the call is what it installs), which is how phost-dash's
Build screen shows them.

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
