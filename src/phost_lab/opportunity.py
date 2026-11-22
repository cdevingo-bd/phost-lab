"""Where a change can apply, and how much it can be worth, read from what each stage's binding says about itself.

``applicability`` answers, for every stage and scene: can the stage be installed on this model, and which of its
implementations apply. Nothing is run: it reads the host's own refusal (``Host.refusal``) and each implementation's
``applicable``, the same data a session uses to refuse an install.

``opportunities`` answers, for one scene: how much of the step each stage's native version takes, and so the most any
replacement could gain. A stage's binding says how to run the host's own version once (``NativeTiming``); the lab replays it
as its own CUDA graph and compares it with the whole step. For a stage whose native version cannot run alone, the binding
names the larger phase that contains it and the figure is an upper bound.

Both are generic over stages: a stage added to the host appears in both without any change here.
"""

from __future__ import annotations

from typing import Any

import phost

from .profile import _capture, _replay_ms
from .scene import Scene, SceneRegistry
from .settle import settled


def _build(scene: Scene, worlds: int, settle: bool):
    """A MuJoCo Warp host on ``scene`` (settled to its start state when ``settle``; applicability needs no state)."""
    import mujoco
    import mujoco_warp as mjw
    import warp as wp
    from phost.hosts.mjwarp import MJWarpHost

    xml = scene.xml()
    model_cpu = mujoco.MjModel.from_xml_string(xml)
    cpu = settled(model_cpu, xml, scene.settle_steps) if settle else mujoco.MjData(model_cpu)
    with wp.ScopedDevice("cuda:0"):
        model = mjw.put_model(model_cpu)
        data = mjw.put_data(model_cpu, cpu, nworld=worlds, nconmax=scene.nconmax(worlds), njmax=scene.constraints)
    return MJWarpHost(model, data, cpu_model=model_cpu), model_cpu


def applicability(registry: phost.Registry, scenes: SceneRegistry, scene_ids: list[str] | None = None) -> dict:
    """For each scene and each stage the host offers: whether the stage can be installed, and which implementations apply.

    ``{"scenes": {id: {"nv", "integrator", "stages": {stage: {"applies", "reason", "impls": {impl: reason | None}}}}}}``
    where ``reason`` is None when it applies. ``applies`` is the host's own answer for the stage; an implementation can
    still refuse a model the stage accepts (the coupled-block Cholesky refuses nv above 512).
    """
    out: dict[str, Any] = dict(scenes={})
    for scene_id in scene_ids or [s["id"] for s in scenes.describe() if not s["hidden"]]:
        host, model_cpu = _build(scenes.get(scene_id), 1, settle=False)
        facts = host.facts()
        stages = {}
        for name in host.stages():
            reason = host.refusal(name)
            impls = {}
            for impl in registry.impls_for(name):
                if impl.kind.value != "native":
                    impls[impl.id] = impl.applicable(facts) if impl.applicable else None
            stages[name] = dict(applies=reason is None, reason=reason, impls=impls)
        out["scenes"][scene_id] = dict(nv=int(model_cpu.nv), integrator=facts.integrator, stages=stages)
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


def _launch_floor_ms(wp, launches: int, repeats: int) -> float:
    """What replaying a graph of one trivial kernel costs: the fixed price of timing anything as its own graph."""
    from ._floor import touch

    x = wp.zeros(1, dtype=float)
    return _replay_ms(wp, _capture(wp, lambda: wp.launch(touch, dim=1, inputs=[x])), launches, repeats)


def opportunities(scene: Scene, worlds: tuple[int, ...] = (1,), launches: int = 10, repeats: int = 9) -> dict:
    """For each world count: the whole step, the graph-launch floor, and for each stage its native time, share of the step
    and speedup ceiling.

    ``ceiling`` is 1 / (1 - share): what an infinitely fast replacement would give. It is reported only for stages whose
    swap class is a speed swap (exact, approximate); for a stage that changes the physics or only observes, the share says what
    the call it attaches to costs. A stage with ``enclosing`` set could not be timed alone and the share is an upper bound.
    """
    import mujoco_warp as mjw
    import warp as wp

    out: dict[str, Any] = dict(scene=scene.id, cells=[])
    with wp.ScopedDevice("cuda:0"):
        floor = _launch_floor_ms(wp, launches, repeats)
        for w in worlds:
            step_host, model_cpu = _build(scene, w, settle=True)
            m, d = step_host.model, step_host.data
            for _ in range(3):
                mjw.step(m, d)
            wp.synchronize()
            step_ms = _replay_ms(wp, _capture(wp, lambda: mjw.step(m, d)), launches, repeats)
            # the stages are timed on separate data: replaying single functions leaves it in a state a step does not start from
            host, _ = _build(scene, w, settle=True)
            for _ in range(3):
                mjw.step(host.model, host.data)
            wp.synchronize()
            facts = host.facts()
            stages = {}
            for name, binding in host.stages().items():
                refusal = host.refusal(name)
                entry: dict[str, Any] = dict(swap_class=binding.contract.swap_class.value, applies=refusal is None, reason=refusal,
                                             native_ms=None, share=None, ceiling=None, enclosing=None)
                timing = binding.native_timing
                run = timing.run(host) if timing is not None and refusal is None else None
                if run is not None:
                    raw = _replay_ms(wp, _capture(wp, run), launches, repeats)
                    ms = max(raw - floor, 0.0)
                    share = min(ms / step_ms, 1.0)
                    speed_swap = binding.contract.swap_class.value in ("exact", "approximate")
                    entry.update(native_ms=ms, share=share, enclosing=timing.enclosing or None,
                                 ceiling=(1.0 / (1.0 - share) if speed_swap and not timing.enclosing and share < 1.0 else None))
                stages[name] = entry
            out["cells"].append(dict(worlds=w, nv=int(model_cpu.nv), integrator=facts.integrator, step_ms=step_ms,
                                     launch_floor_ms=floor, stages=stages))
            del step_host, host
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
