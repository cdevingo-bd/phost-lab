"""Settling a scene to its starting state takes CPU MuJoCo steps, and for a big scene that is a minute. The result depends
only on the model, the step count and the MuJoCo version, so it is computed once and kept.

The cache is a folder of small ``.npz`` files named by a hash of those three things. A file that does not fit the model it
is asked for (wrong shapes, unreadable) is ignored and rewritten, never trusted.
"""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path

import numpy as np

VERSION = "1"


def cache_dir() -> Path:
    """``$PHOST_LAB_CACHE``, else ``$XDG_CACHE_HOME/phost-lab``, else ``~/.cache/phost-lab``."""
    explicit = os.environ.get("PHOST_LAB_CACHE")
    if explicit:
        return Path(explicit).expanduser()
    base = os.environ.get("XDG_CACHE_HOME")
    return (Path(base).expanduser() if base else Path.home() / ".cache") / "phost-lab"


def key(xml: str, steps: int) -> str:
    import mujoco
    return hashlib.sha256(f"{VERSION}\n{mujoco.__version__}\n{steps}\n{xml}".encode()).hexdigest()


def settled(model, xml: str, steps: int, folder: Path | None = None):
    """MjData for ``model`` after ``steps`` CPU steps and a forward pass; from the cache when it has this scene."""
    import mujoco
    data = mujoco.MjData(model)
    if steps > 0:
        path = (folder or cache_dir()) / f"{key(xml, steps)}.npz"
        if not _load(path, model, data):
            for _ in range(steps):
                mujoco.mj_step(model, data)
            _store(path, data)
    mujoco.mj_forward(model, data)
    return data


def _load(path: Path, model, data) -> bool:
    try:
        with np.load(path, allow_pickle=False) as saved:
            qpos, qvel, act, time = saved["qpos"], saved["qvel"], saved["act"], float(saved["time"])
    except (OSError, ValueError, KeyError):
        return False
    if qpos.shape != (model.nq,) or qvel.shape != (model.nv,) or act.shape != (model.na,) \
            or not (np.isfinite(qpos).all() and np.isfinite(qvel).all() and np.isfinite(act).all()):
        return False
    data.qpos[:], data.qvel[:], data.act[:], data.time = qpos, qvel, act, time
    return True


def _store(path: Path, data) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=path.parent, suffix=".npz")
        with os.fdopen(fd, "wb") as handle:
            np.savez(handle, qpos=data.qpos, qvel=data.qvel, act=data.act, time=data.time)
        os.replace(tmp, path)
    except OSError:                                          # a read-only home must not stop a run
        pass
