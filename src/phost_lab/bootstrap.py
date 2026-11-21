"""Build the registries a run needs: stage contracts, implementations from plugins, scenes."""

from __future__ import annotations

import importlib
from typing import Iterable

import phost
from phost.stages import newton_linear

from .scene import SceneRegistry, default_scenes

_BUILTIN_DONE = "_phost_lab_builtin"


def load_registry(plugins: Iterable[str] = (), registry: phost.Registry | None = None) -> phost.Registry:
    """Register built-in stage contracts, then every plugin module's ``register(registry)``."""
    registry = registry or phost.default
    if not getattr(registry, _BUILTIN_DONE, False):
        newton_linear.register(registry)
        setattr(registry, _BUILTIN_DONE, True)
    done = getattr(registry, "_phost_lab_plugins", set())
    for name in plugins:
        if name in done:
            continue
        importlib.import_module(name).register(registry)
        done.add(name)
    registry._phost_lab_plugins = done
    return registry


def load_scenes(modules: Iterable[str] = ()) -> SceneRegistry:
    """Default scenes plus any module's ``scenes(registry)`` additions (internal scenes live elsewhere)."""
    scenes = default_scenes()
    for name in modules:
        importlib.import_module(name).scenes(scenes)
    return scenes
