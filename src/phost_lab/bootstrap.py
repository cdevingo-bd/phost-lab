"""Build the registries a run needs: stage contracts, implementations from plugins, scenes."""

from __future__ import annotations

import importlib
from typing import Iterable

import phost
from phost.impls import register_builtin
from phost.stages import register_all

from .scene import SceneRegistry, default_scenes

_BUILTIN_DONE = "_phost_lab_builtin"


def load_registry(plugins: Iterable[str] = (), registry: phost.Registry | None = None) -> phost.Registry:
    """Register phost's stage contracts and built-in implementations, then every plugin's ``register(registry)``."""
    registry = registry or phost.default
    if not getattr(registry, _BUILTIN_DONE, False):
        register_all(registry)
        register_builtin(registry)
        setattr(registry, _BUILTIN_DONE, True)
    done = getattr(registry, "_phost_lab_plugins", set())
    for name in plugins:
        if name in done:
            continue
        importlib.import_module(name).register(registry)
        done.add(name)
    registry._phost_lab_plugins = done
    return registry


def load_scenes(modules: Iterable[str] = (), directories: Iterable[str] = ()) -> SceneRegistry:
    """Default scenes, any module's ``scenes(registry)`` additions, and every ``*.xml`` scene in the given directories
    (a held-out set lives in a directory that is not part of any repository)."""
    scenes = default_scenes()
    for name in modules:
        importlib.import_module(name).scenes(scenes)
    for directory in directories:
        scenes.add_directory(directory)
    return scenes
