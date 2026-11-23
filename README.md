# phost-lab

Run a graph of [phost](https://github.com/cdevingo-bd/phost) implementations on a scene, in an isolated worker, and get one
JSON result: step time, speedup, accuracy against a native control, per-stage metrics and a verdict. A graph says which
implementation sits on which stage. The simulator behind a scene is an **engine**: MuJoCo Warp on a CUDA GPU, a NumPy chain on the
CPU, a JAX chain compiled by XLA, or one you register.

![A graph and a scene are checked to agree, then run through an Engine (mjwarp, chain, jaxchain or your own) in an isolated worker: native, a native control and the graph run from the same start state through a driver, timed and compared. The result is one JSON. Four reports need no implementation: opportunities, applicability, sparsity and profile.](docs/flow.svg)

Reports that need no implementation: `opportunities` gives each stage's share of the step and the most a replacement could gain,
`applicability` says which stages apply to which scenes and why not, `sparsity` says what a stage's matrix looks like, and
`profile` times each phase of the step as its own graph, where the engine offers it.

```
phost-lab run examples/pd_chain.json --scene chain32:pluck                      # a NumPy chain on the CPU: no GPU, no MuJoCo
phost-lab --plugin phost_coupledchol.phost_impl list
phost-lab validate examples/coupled.json
phost-lab run examples/coupled.json --scene trampoline12:impact --worlds 1 8 32 --out result.json    # MuJoCo Warp on a GPU
```

A graph may select any phost stage (`phost-lab list` shows them), among those the scene's engine offers. A graph with a
model-changing stage is not gated on matching native, because the host is not the truth for it.

Accuracy is judged against the native-vs-native difference of the same run, because a GPU simulator such as MuJoCo Warp is not
bitwise repeatable; an engine whose steps do repeat is measured against a rounding floor instead (see [Engines](#engines)). A ratio
is reported only when that control is above rounding noise. On a GPU, run on an idle device: the result records how many other GPU
processes were running.

## Engines

A run needs a short list of things from a simulator, and none of them is MuJoCo: load a scene and settle it, make instances of
it with some number of worlds, step one, reset it to the start, read its state, and say what device it runs on. An `Engine`
(`phost_lab.engine`) provides them, and the runner, the reports and the drivers speak only that.

```
engine.load(scene)       -> Loaded     parse and settle on the CPU; entering it opens the device scope
loaded.instance(worlds)  -> Instance   the data for that many worlds, and a phost.Host over it (instance.host)
instance.step() / reset() / state() / frame()
loaded.device                          synchronize, capture, how busy, the cost of launching nothing
```

and three optional pieces an engine gives only if it has them: `loaded.check_host(worlds)` (a host a contract check can drive, when
the engine's ordinary host cannot be: the chains' can, MuJoCo Warp's needs a driver over a state with contacts),
`engine.render(scene, frames, w, h)` (frames drawn as images) with `render_unavailable()`, and `profile` / `layout` / `pose`.
Everything outside an engine's own module goes through these: phost-author's and phost-agent's checkers get their hosts from
`phost_lab.engine.check_host`, and phost-dash draws pictures with `render_frames`, so none of them names a simulator.

Three engines ship. `mjwarp` is MuJoCo Warp on a CUDA device, the one that needs a GPU. `chain` is a damped
spring-mass chain stepped in NumPy (`phost.hosts.chain`): no GPU, no MuJoCo, its arrays called `x`, `v` and `a`, and an
effective system that is a band of half-width 1. It offers the stages a chain has (passive force, control law, probe and the
banded effective solve) and not the ones it has no use for (`body.wrench`, `newton.linear_solve`), and the same typed
equations, implementations and checks run on it. It exists to show the boundary is not MuJoCo's. `jaxchain` is the same chain
over JAX arrays (`phost.hosts.jaxchain`): arrays that cannot be written in place, a step that compiles, and a device whose
`capture` is an ahead-of-time compile. It needs `jax` (`engines()["jaxchain"].unavailable()` says so when it is missing, and
the reports skip its scenes). A numpy-only stage cannot be traced, so the `graph` driver refuses one with a message that says
to use `eager`.

A scene names its engine (`Scene.engine`, default `mjwarp`) and a graph names the host its stages are bound to
(`GraphSpec.host`); a graph with stages that does not match the scene's engine is refused before anything loads. Three
things follow from an engine being a different simulator:

* **The driver.** `graph` needs a device that can capture a step. On one that cannot, a run steps eagerly and says so
  (`config.driver_requested`, `meta.driver_note`) rather than claim it replayed a graph.
* **The noise floor.** MuJoCo Warp does not repeat exactly, so native against native is the yardstick for accuracy. A chain
  does repeat exactly, so that yardstick is zero and a ratio to it says nothing; an engine that reports `repeatable` is
  measured against a rounding floor instead (`cell.accuracy_basis`, `runner.ROUNDING_FLOOR`), and a wrong swap still fails.
* **The tools.** `opportunities`, `sparsity` and `applicability` read a scene through its engine. `profile` (the phases of
  a step) and the live view (something to draw) are optional: an engine lists what it offers in `Engine.capabilities`
  (`profile`, `live`, `render`), and `supports(scene, capability)` / `require_capability` say what a scene can do. `mjwarp` offers
  all three; `chain` and `jaxchain` offer `profile` (the phases of their step) and `live` (the particles as spheres), and not
  `render`: `mjwarp` draws frames with MuJoCo's offscreen renderer (it needs `mujoco`, not a CUDA device), and the chains have
  nothing to draw with. A tool an engine does not offer refuses with the engines that do.
* **Where it runs.** An engine may say `device_kind` (`cpu` or `cuda`). A tool that wants to run several things at once (phost-agent's
  CPU checks) does so only on `cpu`; an engine that does not say is treated as one at a time.
* **Scene files.** `SceneRegistry.add_directory` reads `<id>.xml` for `mjwarp`, `<id>.chain` for `chain` and `<id>.jaxchain` for
  `jaxchain` (each engine names its own `suffix`), with an optional `<id>.json` beside it for `settle_steps`, `description`, `tags` and
  `hidden`. A name in two engines is refused.

```
phost-lab run examples/pd_chain.json --scene chain32:pluck      # on the CPU: the graph's host is "chain"
```

A plugin adds an engine with `register_engine(...)` from its `register`. There is no MJX engine: `jaxchain` is a chain in JAX, and
a MuJoCo model stepped by MJX is a different simulator that nobody has written an engine, a host or stage bindings for.

## Examples

| graph | scene | what it shows |
| --- | --- | --- |
| `examples/native.json` | any | the control: native against native reads 1.00x |
| `examples/pd_chain.json` | `chain32:pluck` | a controller (model changing) on a NumPy simulator: no GPU, no MuJoCo |
| `examples/coupled.json`, `coupled_structural.json` | `trampoline12:impact` | the coupled-block Cholesky on MuJoCo Warp, 2.4x at nv=330 |
| `examples/effective_direct.json` | `trampoline12:discrete` | a direct solve of the discrete step's effective metric: **0.57x**, a swap that loses (see below) |
| `examples/probe_peak.json` | `trampoline12:impact` | an observer: no measurable cost, trajectory unchanged |
| `examples/pd_arm.json` | `arm7:swing` | the same kind of controller on a MuJoCo arm |
| `examples/spring_trampoline.json` | `trampoline12:impact` | a force law (model changing) |

## Scenes

A scene names its engine, and a graph is bound to that engine's host.

| engine | scenes |
| --- | --- |
| `chain` (NumPy, CPU) | `chain32:pluck`, `chain256:ringing`: 32 and 256 particles joined by springs; the effective system is a band of half-width 1 |
| `jaxchain` (JAX, XLA) | `jaxchain32:pluck`: the 32-particle chain on JAX arrays, the same start state |
| `mjwarp` (MuJoCo Warp, CUDA) | `trampoline8/12/14:impact\|rest` (a flexible membrane with balls on it, nv 138 to 462), `trampoline8/12/14:discrete` (the same membrane under the discrete integrator, where the host solves an effective metric each step and the `effective.solve` stage applies), `arm7:swing`, and `tank:boat` |

`tank:boat` is a glass tank that shuttles side to side with a boat in it, a cloth sail on the mast and three crates on the deck
(nv=622). **The water in `tank:boat` is 150 frictionless spheres, not a fluid**: MuJoCo has no fluid solver, and the boat
floats because spheres hold it up. It is a busy scene that mixes contact, a flexible body and free bodies. It is also
past the coupled-block solver's range (nv up to 512), which refuses it with an applicability error.

A scene settles before it is timed, which for a big MuJoCo scene takes about a minute. The lab keeps settled start states on disk
(`phost_lab.startcache`: `$PHOST_LAB_CACHE`, else `$XDG_CACHE_HOME/phost-lab`) for any engine that asks. The cache knows no
simulator: an engine names what its start state depends on, gives named arrays, and says whether a stored file fits (one that does
not is recomputed and rewritten; a folder that cannot be written does not stop a run). The MuJoCo engine keys it by the scene's
XML, the step count and MuJoCo's version; the chain engines settle their own scenes directly (the larger one loaded in 0.3 s) and do
not use it.

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
