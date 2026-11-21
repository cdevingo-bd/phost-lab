"""Worker process: read one JSON request on stdin, run it, print one result line.

Each run gets a fresh process so a faulting kernel, a leaked CUDA context or a stuck graph cannot
affect the next run or the parent.
"""

from __future__ import annotations

import json
import sys
import traceback

PREFIX = "PHOST_LAB_RESULT "


def main() -> int:
    request = json.loads(sys.stdin.read())
    try:
        from .bootstrap import load_registry, load_scenes
        from .graph import GraphSpec
        from .runner import RunConfig, run_graph

        graph = GraphSpec.from_dict(request["graph"])
        registry = load_registry(list(graph.plugins) + list(request.get("plugins", ())))
        scene = load_scenes(request.get("scene_modules", ())).get(request["scene"])
        config = RunConfig(**{**request.get("config", {}), "worlds": tuple(request["config"]["worlds"])})
        out = run_graph(graph, scene, config, registry)
        print(PREFIX + json.dumps(out), flush=True)
        return 0
    except BaseException as error:                       # report, never hide: the parent decides
        print(PREFIX + json.dumps(dict(error=f"{type(error).__name__}: {error}", traceback=traceback.format_exc()[-4000:])),
              flush=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
