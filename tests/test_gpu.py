"""End to end on a real GPU with MJWarp. Skipped when mujoco_warp or CUDA is not available."""

import pytest

pytest.importorskip("mujoco")
pytest.importorskip("mujoco_warp")
wp = pytest.importorskip("warp")

from phost_lab import NATIVE, default_scenes
from phost_lab.runner import RunConfig, run_graph

try:
    wp.init()
    HAVE_CUDA = wp.is_cuda_available()
except Exception:                                   # pragma: no cover
    HAVE_CUDA = False
pytestmark = pytest.mark.skipif(not HAVE_CUDA, reason="no CUDA device")

FAST = RunConfig(worlds=(1,), steps=5, launches=2, repeats=3)


def test_native_graph_matches_itself(registry):
    out = run_graph(NATIVE, default_scenes().get("trampoline8:impact"), FAST, registry)
    cell, = out["cells"]
    assert out["pass"] and cell["stage_metrics"] == {}
    assert 0.5 < cell["speedup"] < 2.0               # native against native: timing noise only
