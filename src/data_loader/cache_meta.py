"""Per-gene feature caches that know what produced them (G19).

Extraction used to skip any cache file that existed, so features built from
old windows, old tokens or another model revision were reused silently. Every
cache file now carries a ``meta`` record (model, revision, chunking, input
sha256, ...). A file is reused only when its meta equals the caller's exactly;
any difference raises ``StaleCache`` so a stale directory is never mixed in.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np


class StaleCache(RuntimeError):
    """A cache file was produced by different inputs, code or model."""


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("ascii")).hexdigest()


def _dump(meta: dict) -> str:
    return json.dumps(meta, sort_keys=True)


def write_npz(path: str | Path, arrays: dict[str, np.ndarray], meta: dict) -> None:
    """Write atomically: a killed run leaves no truncated file to trip the resume."""
    if "meta" in arrays:
        raise ValueError("'meta' is reserved")
    path = Path(path)
    tmp = path.with_name(path.name + ".partial")
    with tmp.open("wb") as fh:
        np.savez(fh, meta=np.array(_dump(meta)), **arrays)
    os.replace(tmp, path)


def read_meta(path: str | Path) -> dict | None:
    with np.load(Path(path), allow_pickle=False) as data:
        return json.loads(str(data["meta"])) if "meta" in data.files else None


def read_npz(path: str | Path, meta: dict, ignore: tuple[str, ...] = ()
             ) -> dict[str, np.ndarray] | None:
    """The arrays in ``path`` if its meta equals ``meta``; None if it doesn't exist.

    Keys in ``ignore`` are not compared (a builder that cannot know them checks
    them another way). Raises ``StaleCache`` for a file with no meta or a
    different one.
    """
    path = Path(path)
    if not path.exists():
        return None
    with np.load(path, allow_pickle=False) as data:
        if "meta" not in data.files:
            raise StaleCache(f"{path} has no meta record (pre-G19 cache); use a fresh cache dir")
        found = json.loads(str(data["meta"]))
        diff = sorted(k for k in set(found) | set(meta)
                      if k not in ignore and found.get(k) != meta.get(k))
        if diff:
            raise StaleCache(f"{path} was built with different {diff}: {found} != {meta}")
        return {k: data[k] for k in data.files if k != "meta"}
