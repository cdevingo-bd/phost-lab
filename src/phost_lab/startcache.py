"""A folder of settled start states, for an engine whose scenes take many steps to settle (for a big scene, about a minute).

The cache knows nothing about any simulator. An engine says what its start state depends on (``parts``: its own version, the
step count, the scene text ...), computes it as named arrays, and says whether a stored file fits the model it is loading
(``accept``). The cache keeps the arrays in a small ``.npz`` file named by a hash of the parts, and never trusts one it cannot
read or that ``accept`` turns down: that is recomputed and rewritten. A cache folder that cannot be written does not stop a run.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

VERSION = "1"


def cache_dir() -> Path:
    """``$PHOST_LAB_CACHE``, else ``$XDG_CACHE_HOME/phost-lab``, else ``~/.cache/phost-lab``."""
    explicit = os.environ.get("PHOST_LAB_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base).expanduser() if base else Path.home() / ".cache") / "phost-lab"


def key(*parts) -> str:
    """A hash of everything the start state depends on. Leave a dependency out and a stale state is trusted."""
    return hashlib.sha256("\n".join([VERSION, *(str(p) for p in parts)]).encode()).hexdigest()


def cached(parts: Sequence, compute: Callable[[], Mapping[str, np.ndarray]],
           accept: Callable[[Mapping[str, np.ndarray]], bool], folder: Path | None = None) -> dict[str, np.ndarray]:
    """The arrays for ``parts``: read from the cache when a file there fits (``accept`` says so), else ``compute()`` and keep them."""
    path = (folder or cache_dir()) / f"{key(*parts)}.npz"
    arrays = _load(path, accept)
    if arrays is None:
        arrays = {name: np.asarray(value) for name, value in compute().items()}
        _store(path, arrays)
    return arrays


def _load(path: Path, accept) -> dict[str, np.ndarray] | None:
    try:
        with np.load(path, allow_pickle=False) as saved:
            arrays = {name: saved[name] for name in saved.files}
    except (OSError, ValueError, KeyError):
        return None
    try:
        return arrays if accept(arrays) else None
    except (KeyError, ValueError):
        return None


def _store(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npz")
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, **arrays)
        os.replace(tmp, path)
    except OSError:                                          # a read-only home must not stop a run
        pass
