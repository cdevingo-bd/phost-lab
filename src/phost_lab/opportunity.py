"""Where a change can apply, and how much it can be worth, read from what each stage's binding says about itself.

``applicability`` answers, for every stage and scene: can the stage be installed on this model, and which of its
implementations apply. Nothing is run: it reads the host's own refusal (``Host.refusal``) and each implementation's
``applicable``, the same data a session uses to refuse an install.

``opportunities`` answers, for one scene: how much of the step each stage's native version takes, and so the most any
replacement could gain. A stage's binding says how to run the host's own version once (``NativeTiming``); the lab replays it
as its own CUDA graph and compares it with the whole step. For a stage whose native version cannot run alone, the binding
names the larger phase that contains it and the figure is an upper bound.

Both are generic over stages and over engines: a stage added to a host appears in both without any change here, and a
scene on another engine (``Scene.engine``) is read through that engine.
"""

from __future__ import annotations

from typing import Any

import phost

from .engine import engine_for
from .scene import Scene, SceneRegistry


def _build(scene: Scene, worlds: int, settle: bool):
    """A host for ``scene`` on its engine, and the engine's facts about the loaded model (``nv``, ...). The scene is settled
    to its start state when ``settle``; applicability needs no state. The device scope is not held open afterwards."""
    with engine_for(scene).load(scene, settle=settle) as loaded:
        return loaded.instance(worlds).host, dict(loaded.meta)


def applicability(registry: phost.Registry, scenes: SceneRegistry, scene_ids: list[str] | None = None) -> dict:
    """For each scene and each stage the host offers: whether the stage can be installed, and which implementations apply.

    ``{"scenes": {id: {"nv", "integrator", "stages": {stage: {"applies", "reason", "impls": {impl: reason | None}}}}}}``
    where ``reason`` is None when it applies. ``applies`` is the host's own answer for the stage; an implementation can
    still refuse a model the stage accepts (the coupled-block Cholesky refuses nv above 512).
    """
    out: dict[str, Any] = dict(scenes={})
    for scene_id in scene_ids or [s["id"] for s in scenes.describe() if not s["hidden"]]:
        host, meta = _build(scenes.get(scene_id), 1, settle=False)
        facts = host.facts()
        stages = {}
        for name in host.stages():
            reason = host.refusal(name)
            impls = {}
            for impl in registry.impls_for(name):
                if impl.kind.value != "native":
                    impls[impl.id] = impl.applicable(facts) if impl.applicable else None
            stages[name] = dict(applies=reason is None, reason=reason, impls=impls)
        out["scenes"][scene_id] = dict(nv=int(meta["nv"]), integrator=facts.integrator, stages=stages)
        del host
    return out


def scene_for(stage: str, scenes: SceneRegistry, registry: phost.Registry | None = None) -> str | None:
    """The first scene, in the registry's own order, on which the host accepts ``stage`` and (if ``registry`` is given) on
    which at least one non-native implementation of it applies; None if there is none.

    The registry's order is its owner's statement of which scene to try first, so a stage that only applies to some
    models lands on the first scene that has them, without a table of scene names per stage.
    """
    for described in scenes.describe():
        if described["hidden"]:
            continue
        host, _ = _build(scenes.get(described["id"]), 1, settle=False)
        if stage not in host.stages() or host.refusal(stage):
            continue
        facts = host.facts()
        impls = [i for i in (registry.impls_for(stage) if registry is not None else []) if i.kind.value != "native"]
        if not impls or any(not i.applicable or not i.applicable(facts) for i in impls):
            return described["id"]
    return None


def opportunities(scene: Scene, worlds: tuple[int, ...] = (1,), launches: int = 10, repeats: int = 9) -> dict:
    """For each world count: the whole step, the launch floor, and for each stage its native time, share of the step
    and speedup ceiling.

    ``ceiling`` is 1 / (1 - share): what an infinitely fast replacement would give. It is reported only for stages whose
    swap class is a speed swap (exact, approximate); for a stage that changes the physics or only observes, the share says what
    the call it attaches to costs. A stage with ``enclosing`` set could not be timed alone and the share is an upper bound.
    A stage whose binding declares no native timing (every stage of a host that has none) is listed without a figure.
    """
    out: dict[str, Any] = dict(scene=scene.id, cells=[])
    with engine_for(scene).load(scene) as loaded:
        device = loaded.device
        floor = device.floor_ms(launches, repeats)
        for w in worlds:
            step_instance = loaded.instance(w)
            for _ in range(3):
                step_instance.step()
            device.synchronize()
            step_ms = device.replay_ms(step_instance.step, launches, repeats)
            # the stages are timed on separate data: replaying single functions leaves it in a state a step does not start from
            instance = loaded.instance(w)
            for _ in range(3):
                instance.step()
            device.synchronize()
            host = instance.host
            facts = host.facts()
            stages = {}
            for name, binding in host.stages().items():
                refusal = host.refusal(name)
                entry: dict[str, Any] = dict(swap_class=binding.contract.swap_class.value, applies=refusal is None, reason=refusal,
                                             native_ms=None, share=None, ceiling=None, enclosing=None)
                timing = binding.native_timing
                run = timing.run(host) if timing is not None and refusal is None else None
                if run is not None:
                    raw = device.replay_ms(run, launches, repeats)
                    ms = max(raw - floor, 0.0)
                    share = min(ms / step_ms, 1.0)
                    speed_swap = binding.contract.swap_class.value in ("exact", "approximate")
                    entry.update(native_ms=ms, share=share, enclosing=timing.enclosing or None,
                                 ceiling=(1.0 / (1.0 - share) if speed_swap and not timing.enclosing and share < 1.0 else None))
                stages[name] = entry
            out["cells"].append(dict(worlds=w, nv=int(loaded.meta["nv"]), integrator=facts.integrator, step_ms=step_ms,
                                     launch_floor_ms=floor, stages=stages))
            del step_instance, instance, host
    return out


def structure(scene: Scene) -> dict:
    """The nonzero structure of each stage's system matrix on this scene, for every stage whose binding can say.

    ``{"scene", "nv", "stages": {stage: {"applies", "reason", "structure"}}}`` where ``structure`` is
    ``Sparsity.describe()`` (size, density, coupled rows, bandwidth before and after reordering, and the floating-point
    operations a dense, a coupled-rows and a banded factorization would take), or None when the stage has no matrix here.
    Nothing is timed: this says whether a structure-exploiting solve has something to exploit, before one is written.
    """
    host, meta = _build(scene, 1, settle=False)
    stages = {}
    for name, binding in host.stages().items():
        if binding.sparsity is None:
            continue
        reason = host.refusal(name)
        pattern = binding.sparsity(host) if reason is None else None
        stages[name] = dict(applies=reason is None, reason=reason, structure=pattern.describe() if pattern is not None else None)
    return dict(scene=scene.id, nv=int(meta["nv"]), integrator=host.facts().integrator, stages=stages)


def structure_table(result: dict) -> str:
    lines = [f"scene {result['scene']} (nv={result['nv']}, {result['integrator']} integrator)"]
    if not result["stages"]:
        return "\n".join(lines + ["no stage on this host reports a matrix structure"])
    for name, cell in result["stages"].items():
        if cell["structure"] is None:
            lines.append(f"  {name}: not applicable here: {cell['reason']}")
            continue
        s = cell["structure"]
        lines += [f"  {name}: {s['n']} x {s['n']}, {s['nnz']} entries ({100 * s['density']:.1f}% dense), "
                  f"{s['coupled']} of {s['n']} rows coupled",
                  f"    band: {s['bandwidth']} as the host orders it, {s['bandwidth_rcm']} after reverse Cuthill-McKee",
                  f"    factorization work, in flops: dense {s['dense_flops']:.2e}, coupled rows only {s['coupled_flops']:.2e} "
                  f"({100 * s['coupled_flops'] / s['dense_flops']:.1f}%), banded {s['banded_flops']:.2e} "
                  f"({100 * s['banded_flops'] / s['dense_flops']:.1f}%)",
                  "    flops are not time: whether a variant wins depends on how its kernel moves data (see profile)"]
    return "\n".join(lines)


REPORTS = ("applicability", "opportunities", "structure", "profile")


def reports(kinds, scene: Scene, scenes: SceneRegistry, registry: phost.Registry, worlds=(1,), graph=None) -> dict:
    """Several of the reports for one scene, by name, in one go (they share a process, so a tool pays for the GPU once).
    ``graph`` is the experiment the ``profile`` report installs for its second column (none: the native step only)."""
    out: dict[str, Any] = {}
    for kind in kinds:
        if kind == "opportunities":
            out[kind] = opportunities(scene, tuple(worlds))
        elif kind == "structure":
            out[kind] = structure(scene)
        elif kind == "applicability":
            out[kind] = applicability(registry, scenes, [scene.id])
        elif kind == "profile":
            from .graph import NATIVE
            from .profile import profile_scene
            out[kind] = profile_scene(graph or NATIVE, scene, tuple(worlds), registry)
        else:
            raise phost.ContractError(f"no report {kind!r}; known {list(REPORTS)}")
    return out


def applicability_table(result: dict) -> str:
    """Stages by scenes, then why not for each refusal."""
    scene_ids = list(result["scenes"])
    stages = list(next(iter(result["scenes"].values()))["stages"]) if scene_ids else []
    width = max([len(s) for s in stages] + [5]) + 2
    lines = [f"{'stage':<{width}}" + "".join(f"{s:>{len(s) + 2}}" for s in scene_ids),
             f"{'':<{width}}" + "".join(f"{('nv=' + str(result['scenes'][s]['nv'])):>{len(s) + 2}}" for s in scene_ids)]
    for stage in stages:
        lines.append(f"{stage:<{width}}" + "".join(
            f"{('yes' if result['scenes'][s]['stages'][stage]['applies'] else 'no'):>{len(s) + 2}}" for s in scene_ids))
    why = []
    for s in scene_ids:
        for stage, cell in result["scenes"][s]["stages"].items():
            if cell["reason"]:
                why.append(f"  {s}: {stage} is refused: {cell['reason']}")
            for impl, reason in cell["impls"].items():
                if reason and cell["applies"]:
                    why.append(f"  {s}: {impl} does not apply: {reason}")
    return "\n".join(lines + (["", "why not:"] + why if why else []))


def opportunities_table(result: dict) -> str:
    lines = [f"scene {result['scene']}"]
    for cell in result["cells"]:
        lines += ["", f"{cell['worlds']} world(s), nv={cell['nv']}, {cell['integrator']} integrator: whole step {cell['step_ms']:.3f} ms "
                      f"(graph-launch floor {cell['launch_floor_ms']:.3f} ms, subtracted from each stage)",
                  f"  {'stage':<22}{'class':<16}{'native ms':>10}{'share':>8}{'ceiling':>9}  note"]
        for name, e in cell["stages"].items():
            klass = e["swap_class"].replace("_", " ")
            if not e["applies"]:
                lines.append(f"  {name:<22}{klass:<16}{'':>10}{'':>8}{'':>9}  not applicable here: {e['reason']}")
            elif e["native_ms"] is None:
                lines.append(f"  {name:<22}{klass:<16}{'':>10}{'':>8}{'':>9}  attaches after the whole step; nothing to time alone")
            else:
                bound = "<=" if e["enclosing"] else "  "
                ceiling = f"{e['ceiling']:.2f}x" if e["ceiling"] is not None else "-"
                note = (f"upper bound: {e['enclosing']}" if e["enclosing"]
                        else "" if e["ceiling"] is not None else "changes the physics or only observes: not a speed swap")
                lines.append(f"  {name:<22}{klass:<16}{bound}{e['native_ms']:>8.3f}{bound}{100 * e['share']:>4.0f}%{ceiling:>9}  {note}")
    return "\n".join(lines)
