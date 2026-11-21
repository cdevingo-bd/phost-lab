"""``phost-lab``: list, validate and run graphs."""

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
    return "\n".join(rows)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="phost-lab")
    ap.add_argument("--plugin", action="append", default=[], help="module exposing register(registry); repeatable")
    ap.add_argument("--scene-module", action="append", default=[], help="module exposing scenes(registry); repeatable")
    sub = ap.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="print scenes and registered stages/implementations as JSON")
    v = sub.add_parser("validate", help="check a graph file against the registry")
    v.add_argument("graph")
    r = sub.add_parser("run", help="run a graph on a scene")
    r.add_argument("graph")
    r.add_argument("--scene", required=True)
    r.add_argument("--worlds", type=int, nargs="+", default=[1, 8])
    r.add_argument("--steps", type=int, default=20)
    r.add_argument("--out", help="write the full JSON result here")
    r.add_argument("--timeout", type=float, default=900.0)
    r.add_argument("--no-isolate", action="store_true", help="run in this process (faster, no crash isolation)")
    args = ap.parse_args(argv)

    try:
        return _dispatch(args)
    except ContractError as error:
        print(f"error: {error}", file=sys.stderr)
        return 2


def _dispatch(args) -> int:
    if args.command == "list":
        registry = load_registry(args.plugin)
        print(json.dumps(dict(scenes=load_scenes(args.scene_module).describe(), registry=registry.describe()), indent=2))
        return 0
    graph = _load_graph(args.graph)
    registry = load_registry(list(graph.plugins) + args.plugin)
    graph.validate(registry)
    if args.command == "validate":
        print(f"ok: graph '{graph.name}' with {len(graph.stages)} stage(s)")
        return 0
    scene = load_scenes(args.scene_module).get(args.scene)
    config = RunConfig(worlds=tuple(args.worlds), steps=args.steps)
    if args.no_isolate:
        from .runner import run_graph
        result = run_graph(graph, scene, config, registry)
    else:
        from .isolate import run_isolated
        result = run_isolated(graph, args.scene, config, timeout=args.timeout, plugins=args.plugin,
                              scene_modules=args.scene_module)
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
