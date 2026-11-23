"""The start-state cache knows no simulator: it keeps named arrays under a hash of what they depend on."""

import numpy as np
import pytest

from phost_lab import startcache


def arrays(n=3, value=1.0):
    return dict(x=np.full(n, value), t=np.asarray(0.5))


def fits(saved):
    return set(saved) >= {"x", "t"} and saved["x"].shape == (3,) and bool(np.isfinite(saved["x"]).all())


def test_the_first_call_computes_and_stores_the_second_reads(tmp_path):
    calls = []
    first = startcache.cached(("engine", "1.0", 10, "scene text"), lambda: (calls.append(1), arrays())[1], fits, tmp_path)
    second = startcache.cached(("engine", "1.0", 10, "scene text"), lambda: (calls.append(1), arrays(value=9.0))[1], fits, tmp_path)
    assert calls == [1] and np.array_equal(first["x"], second["x"]) and float(second["t"]) == 0.5
    assert len(list(tmp_path.glob("*.npz"))) == 1


def test_every_part_the_state_depends_on_changes_the_key():
    base = ("engine", "1.0", 10, "scene text")
    keys = {startcache.key(*base)} | {startcache.key(*(base[:i] + (changed,) + base[i + 1:])) for i, changed in
                                      enumerate(("other", "2.0", 11, "scene text!"))}
    assert len(keys) == 5
    assert startcache.key(*base) == startcache.key(*base)


@pytest.mark.parametrize("damage", ["garbage", "wrong_shape", "nan", "missing"])
def test_a_file_that_does_not_fit_is_ignored_and_rewritten(tmp_path, damage):
    parts = ("engine", "1.0", 5, "s")
    startcache.cached(parts, arrays, fits, tmp_path)
    path = next(tmp_path.glob("*.npz"))
    if damage == "garbage":
        path.write_bytes(b"not an npz")
    elif damage == "wrong_shape":
        np.savez(path, x=np.zeros(7), t=0.0)
    elif damage == "nan":
        np.savez(path, x=np.full(3, np.nan), t=0.0)
    else:
        np.savez(path, x=np.zeros(3))
    again = startcache.cached(parts, arrays, fits, tmp_path)
    assert np.array_equal(again["x"], np.ones(3))
    with np.load(path) as saved:
        assert np.array_equal(saved["x"], np.ones(3))                           # rewritten with the right state


def test_an_unwritable_cache_does_not_stop_a_run(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("x")
    got = startcache.cached(("e", 1), arrays, fits, blocker / "sub")             # cannot mkdir under a file
    assert np.array_equal(got["x"], np.ones(3))


def test_the_folder_follows_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("PHOST_LAB_CACHE", str(tmp_path / "a"))
    assert startcache.cache_dir() == tmp_path / "a"
    monkeypatch.delenv("PHOST_LAB_CACHE")
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "x"))
    assert startcache.cache_dir() == tmp_path / "x" / "phost-lab"
    monkeypatch.delenv("XDG_CACHE_HOME")
    assert startcache.cache_dir().name == "phost-lab"
