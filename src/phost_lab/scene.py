"""Scene registry: named, reproducible starting states for a run."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from phost import ContractError

BALLS = [(-0.12, -0.08, 0.95), (0.10, 0.10, 1.25), (0.02, -0.17, 1.55), (-0.14, 0.14, 1.85), (0.16, -0.02, 2.15)]


def trampoline_xml(n: int = 12) -> str:
    """Balls dropped onto an n x n membrane pinned at its border (Euler, Newton). nv = 3 (n-2)^2 + 30."""
    span, k = 0.77, n - 1
    spacing, rim = span / (n - 1), span / 2 + 0.03
    bodies = "\n".join(
        f'    <body name="ball_{i}" pos="{x} {y} {z}"><joint type="free" damping="0.5"/>'
        f'<geom type="sphere" size="0.05" mass="0.25"/></body>' for i, (x, y, z) in enumerate(BALLS))
    return f"""<mujoco model="trampoline_{n}">
  <option timestep="0.0005" integrator="Euler"/>
  <worldbody>
    <geom name="floor" type="plane" size="3 3 .1" pos="0 0 0"/>
    <geom name="rim_n" type="box" size="{rim+.015} .03 .02" pos="0 {rim} .45" contype="0" conaffinity="0"/>
    <geom name="rim_s" type="box" size="{rim+.015} .03 .02" pos="0 {-rim} .45" contype="0" conaffinity="0"/>
    <geom name="rim_e" type="box" size=".03 {rim+.015} .02" pos="{rim} 0 .45" contype="0" conaffinity="0"/>
    <geom name="rim_w" type="box" size=".03 {rim+.015} .02" pos="{-rim} 0 .45" contype="0" conaffinity="0"/>
    <flexcomp name="sheet" type="grid" dim="2" count="{n} {n} 1" spacing="{spacing:.6f} {spacing:.6f} 0.1" pos="0 0 .45"
              mass=".4" radius="5e-3">
      <pin gridrange="0 0 {k} 0"/><pin gridrange="0 {k} {k} {k}"/><pin gridrange="0 0 0 {k}"/><pin gridrange="{k} 0 {k} {k}"/>
      <contact solref="0.003"/>
      <elasticity elastic2d="both" poisson="0" thickness="2e-3" young="1e5" damping="0.01"/>
    </flexcomp>
{bodies}
  </worldbody>
</mujoco>
"""


def arm_xml(links: int = 7) -> str:
    """A chain of hinge links, each with a motor. Gravity swings it; a controller can hold or drive it."""
    open_tags, close_tags = [], []
    for i in range(links):
        offset = "0 0 1.5" if i == 0 else "0.2 0 0"
        open_tags.append(f'<body pos="{offset}"><joint name="j{i}" type="hinge" axis="0 1 0" damping="0.01"/>'
                         f'<geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02" mass="0.2"/>')
        close_tags.append("</body>")
    motors = "".join(f'<motor joint="j{i}" gear="20" ctrlrange="-1 1"/>' for i in range(links))
    return (f'<mujoco><option timestep="0.002" integrator="Euler" solver="Newton"/><worldbody>'
            f'{"".join(open_tags)}{"".join(close_tags)}</worldbody><actuator>{motors}</actuator></mujoco>')


@dataclass(frozen=True)
class Scene:
    id: str
    xml: Callable[[], str]
    settle_steps: int = 0                    # CPU MuJoCo steps from the model's initial state
    description: str = ""
    tags: tuple[str, ...] = ()
    hidden: bool = False                     # a held-out scene: runnable, but its model and description are never shown

    def public(self) -> dict:
        """What a tool may show about this scene. A hidden scene shows its id and nothing about what is in it."""
        if self.hidden:
            return dict(id=self.id, hidden=True, description="", settle_steps=0, tags=[])
        return dict(id=self.id, hidden=False, description=self.description, settle_steps=self.settle_steps,
                    tags=list(self.tags))


class SceneRegistry:
    def __init__(self):
        self._scenes: dict[str, Scene] = {}

    def add(self, scene: Scene) -> Scene:
        if scene.id in self._scenes:
            raise ContractError(f"scene {scene.id!r} already registered")
        self._scenes[scene.id] = scene
        return scene

    def get(self, scene_id: str) -> Scene:
        try:
            return self._scenes[scene_id]
        except KeyError:
            raise ContractError(f"no scene {scene_id!r}; known {sorted(self._scenes)}") from None

    def ids(self) -> list[str]:
        return sorted(self._scenes)

    def describe(self) -> list[dict]:
        return [s.public() for s in self._scenes.values()]

    def add_directory(self, directory: str | Path) -> list[str]:
        """Register every ``<id>.xml`` in ``directory``. An optional ``<id>.json`` beside it may set ``settle_steps``,
        ``description``, ``tags`` and ``hidden``. This is how a held-out set lives outside any repository."""
        added = []
        for path in sorted(Path(directory).glob("*.xml")):
            meta_path = path.with_suffix(".json")
            meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
            unknown = set(meta) - {"settle_steps", "description", "tags", "hidden"}
            if unknown:
                raise ContractError(f"{meta_path}: unknown keys {sorted(unknown)}")
            added.append(self.add(Scene(path.stem, lambda p=path: p.read_text(), int(meta.get("settle_steps", 0)),
                                        meta.get("description", ""), tuple(meta.get("tags", ())),
                                        bool(meta.get("hidden", False)))).id)
        return added


def default_scenes() -> SceneRegistry:
    reg = SceneRegistry()
    for n in (8, 12, 14):
        nv = 3 * (n - 2) ** 2 + 30
        reg.add(Scene(f"trampoline{n}:impact", lambda n=n: trampoline_xml(n), 2400,
                      f"membrane {n}x{n}, nv={nv}, 1.2 s in: balls hitting the sheet", ("flex", "contact")))
        reg.add(Scene(f"trampoline{n}:rest", lambda n=n: trampoline_xml(n), 10000,
                      f"membrane {n}x{n}, nv={nv}, 5 s in: balls resting on the sheet", ("flex", "contact")))
    reg.add(Scene("arm7:swing", lambda: arm_xml(7), 0,
                  "7-link hinge chain with a motor per joint, released horizontal; nv=7, nu=7", ("actuated",)))
    return reg
