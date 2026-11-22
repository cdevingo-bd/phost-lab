"""``phost-lab``: list, validate, run, profile and survey graphs, scenes and stages."""

from __future__ import annotations

import argparse
import json
import sys

from phost import ContractError

from .bootstrap import load_registry, load_scenes
from .graph import GraphSpec
from .runner import RunConfig


def _load_graph(path: str) -> GraphSpec:
    with open(path) as stream:
        return GraphSpec.from_json(stream.read())


def _table(result: dict) -> str:
    rows = [f"scene {result['scene']}  nv={result['meta']['nv']}  device {result['meta']['device']}  graph '{result['graph']['name']}'",
            f"{'worlds':>7} {'step ms':>9} {'native ms':>10} {'speedup':>8} {'acc ratio':>10} {'native spread':>14}  verdict"]
    for c in result["cells"]:
        ratio = "n/a" if c["accuracy_ratio"] is None else f"{c['accuracy_ratio']:.2f}"
        verdict = "ok" if c["verdict"]["ok"] else "FAIL: " + "; ".join(c["verdict"]["problems"])
        rows.append(f"{c['worlds']:>7} {c['step_ms']:>9.3f} {c['native_ms']:>10.3f} {c['speedup']:>7.2f}x {ratio:>10} {c['native_spread']:>13.1%}  {verdict}")
        for note in c["verdict"]["notes"] + c["advisories"]:
            rows.append(f"        note: {note}")
        if "per_call" in c:
            pc = c["per_call"]
            ns = pc["steps_per_call"]
            rows.append(f"        per call (ms)    " + "".join(f"{n:>9}" for n in ns) + "   fixed ms  per-step ms")
            for label in ("native", "graph"):
                rows.append(f"          {label:<14} " + "".join(f"{ms:>9.3f}" for ms in pc[label]["ms"])
                            + f"   {pc[label]['fixed_ms']:>8.3f}  {pc[label]['per_step_ms']:>10.3f}")
            rows.append(f"          {'speedup':<14} " + "".join(f"{x:>8.2f}x" for x in pc["speedup"]))
    return "\n".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="phost-lab", allow_abbrev=False)     # --scene must never match --scene-dir
    ap.add_argument("--plugin", action="append", default=[], help="module exposing register(registry); repeatable")
    ap.add_argument("--scene-module", action="append", default=[], help="module exposing scenes(registry); repeatable")
    ap.add_argument("--scene-dir", action="append", default=[], help="directory of <id>.xml scenes; repeatable")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("list", allow_abbrev=False, help="print scenes and registered stages/implementations as JSON")
    v = sub.add_parser("validate", allow_abbrev=False, help="check a graph file against the registry")
    v.add_argument("graph")
    r = sub.add_parser("run", allow_abbrev=False, help="run a graph on a scene")
    r.add_argument("graph")
    r.add_argument("--scene", required=True)
    r.add_argument("--worlds", type=int, nargs="+", default=[1, 8])
    r.add_argument("--steps", type=int, default=20)
    r.add_argument("--steps-per-call", type=int, nargs="+", default=[], metavar="N",
                   help="also time calls of N graph launches (for example 1 5 20) and report the fixed cost of a call")
    r.add_argument("--driver", default="graph", metavar="NAME",
                   help="how the host is stepped while timed: graph (replay a CUDA graph, default), eager (run the step from "
                        "Python), or a driver a plugin registers")
    r.add_argument("--out", help="write the full JSON result here")
    r.add_argument("--timeout", type=float, default=900.0)
    r.add_argument("--no-isolate", action="store_true", help="run in this process (faster, no crash isolation)")
    ap_ = sub.add_parser("applicability", allow_abbrev=False,
                         help="which stages (and implementations) apply to which scenes, and why not")
    ap_.add_argument("--scene", action="append", default=[], help="a scene id; repeatable (default: every visible scene)")
    ap_.add_argument("--out", help="write the full JSON result here")
    op = sub.add_parser("opportunities", allow_abbrev=False,
                        help="how much of the step each stage's native version takes: the most a replacement could gain")
    op.add_argument("--scene", required=True)
    op.add_argument("--worlds", type=int, nargs="+", default=[1])
    op.add_argument("--out", help="write the full JSON result here")
    sp = sub.add_parser("sparsity", allow_abbrev=False,
                        help="the nonzero structure of each stage's system matrix: what a structure-exploiting solve could save")
    sp.add_argument("--scene", required=True)
    sp.add_argument("--out", help="write the full JSON result here")
    pr = sub.add_parser("profile", allow_abbrev=False, help="time each phase of the step, native and with a graph installed")
    pr.add_argument("--graph", help="graph file to install for the second column (omit for the native step only)")
    pr.add_argument("--scene", required=True)
    pr.add_argument("--worlds", type=int, nargs="+", default=[1])
    pr.add_argument("--out", help="write the full JSON result here")
    args = ap.parse_args(argv)

    try:
        return _dispatch(args)
    except ContractError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _dispatch(args) -> int:
    if args.command == "list":
        registry = load_registry(args.plugin)
        print(json.dumps(dict(scenes=load_scenes(args.scene_module, args.scene_dir).describe(), registry=registry.describe()), indent=2))
        return 0
    if args.command == "applicability":
        from .opportunity import applicability, applicability_table
        scenes = load_scenes(args.scene_module, args.scene_dir)
        result = applicability(load_registry(args.plugin), scenes, args.scene or None)
        if args.out:
            with open(args.out, "w") as stream:
                json.dump(result, stream, indent=1)
        print(applicability_table(result))
        return 0
    if args.command == "opportunities":
        from .opportunity import opportunities, opportunities_table
        result = opportunities(load_scenes(args.scene_module, args.scene_dir).get(args.scene), tuple(args.worlds))
        if args.out:
            with open(args.out, "w") as stream:
                json.dump(result, stream, indent=1)
        print(opportunities_table(result))
        return 0
    if args.command == "sparsity":
        from .opportunity import structure, structure_table
        result = structure(load_scenes(args.scene_module, args.scene_dir).get(args.scene))
        if args.out:
            with open(args.out, "w") as stream:
                json.dump(result, stream, indent=1)
        print(structure_table(result))
        return 0
    if args.command == "profile":
        from .profile import profile_scene, table
        graph = _load_graph(args.graph) if args.graph else GraphSpec("native")
        registry = load_registry(list(graph.plugins) + args.plugin)
        result = profile_scene(graph, load_scenes(args.scene_module, args.scene_dir).get(args.scene), tuple(args.worlds), registry)
        if args.out:
            with open(args.out, "w") as stream:
                json.dump(result, stream, indent=1)
        print(table(result))
        return 0
    graph = _load_graph(args.graph)
    registry = load_registry(list(graph.plugins) + args.plugin)
    graph.validate(registry)
    if args.command == "validate":
        print(f"ok: graph '{graph.name}' with {len(graph.stages)} stage(s)")
        return 0
    scene = load_scenes(args.scene_module, args.scene_dir).get(args.scene)
    config = RunConfig(worlds=tuple(args.worlds), steps=args.steps, calls=tuple(args.steps_per_call), driver=args.driver)
    if args.no_isolate:
        from .runner import run_graph
        result = run_graph(graph, scene, config, registry)
    else:
        from .isolate import run_isolated
        result = run_isolated(graph, args.scene, config, timeout=args.timeout, plugins=args.plugin,
                              scene_modules=args.scene_module, scene_dirs=args.scene_dir)
    if args.out:
        with open(args.out, "w") as stream:
            json.dump(result, stream, indent=1)
    if "error" in result:
        print(f"RUN FAILED: {result['error']}", file=sys.stderr)
        return 2
    print(_table(result))
    return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
