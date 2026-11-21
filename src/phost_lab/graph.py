"""Graph spec: which implementation sits on which stage, as plain data.

A graph is what a node-graph editor would draw: one node per stage, each holding an implementation
choice and its parameters. It is JSON, diffable and versioned, and says nothing about how to run it.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping

from phost import ContractError, Registry, Selection

VERSION = 1


@dataclass(frozen=True)
class Node:
    impl: str
    params: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraphSpec:
    name: str
    stages: Mapping[str, Node] = field(default_factory=dict)       # stage name -> node
    host: str = "mjwarp"
    plugins: tuple[str, ...] = ()                                   # modules exposing register(registry)
    note: str = ""

    def selections(self) -> list[Selection]:
        return [Selection(stage, node.impl, dict(node.params)) for stage, node in self.stages.items()]

    def validate(self, registry: Registry) -> None:
        """Check every node against the registry: stage exists, impl targets it, params are valid."""
        for stage, node in self.stages.items():
            impl = registry.impl(node.impl)
            if impl.stage != stage:
                raise ContractError(f"node for stage {stage!r} uses impl {impl.id!r}, which is for {impl.stage!r}")
            impl.resolve_params(node.params)

    def to_dict(self) -> dict:
        return dict(version=VERSION, name=self.name, host=self.host, plugins=list(self.plugins), note=self.note,
                    stages={s: dict(impl=n.impl, params=dict(n.params)) for s, n in self.stages.items()})

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "GraphSpec":
        if d.get("version") != VERSION:
            raise ContractError(f"unsupported graph version {d.get('version')!r}; expected {VERSION}")
        try:
            stages = {s: Node(n["impl"], dict(n.get("params", {}))) for s, n in d.get("stages", {}).items()}
            return cls(name=d["name"], stages=stages, host=d.get("host", "mjwarp"),
                       plugins=tuple(d.get("plugins", ())), note=d.get("note", ""))
        except (KeyError, TypeError, AttributeError) as error:
            raise ContractError(f"malformed graph: {error!r}") from error

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2, sort_keys=True)

    @classmethod
    def from_json(cls, text: str) -> "GraphSpec":
        return cls.from_dict(json.loads(text))


NATIVE = GraphSpec(name="native")
