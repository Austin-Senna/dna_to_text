"""build_counts.chunk_counts refuses cache entries that were not extracted from the pinned CDS."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import build_counts as bc  # noqa: E402


def _write(cache: Path, gene: str, sha: str, n_chunks: int = 2) -> None:
    meta = {k: "x" for k in bc.RUN_KEYS} | {"input_sha256": sha}
    np.savez(cache / f"{gene}.npz", mean=np.zeros((n_chunks, 4), dtype=np.float32), meta=np.array(json.dumps(meta)))


@pytest.fixture
def cache(monkeypatch):
    """Point the dnabert2 chunk cache (a property derived from DATA) at a scratch dir inside
    the repo, since the stamp records the cache path relative to the repo root."""
    import shutil
    scratch = bc.DATA / "_test_chunk_cache"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir()
    monkeypatch.setattr(type(bc.ENCODER_SPECS["dnabert2"]), "chunk_dir", property(lambda self: scratch))
    yield scratch
    shutil.rmtree(scratch, ignore_errors=True)


def test_matching_hashes_are_counted(cache):
    _write(cache, "ENSG1", "a" * 64, 3)
    _write(cache, "ENSG2", "b" * 64, 1)
    counts, stamp = bc.chunk_counts("dnabert2", pinned={"ENSG1": "a" * 64, "ENSG2": "b" * 64})
    assert counts == {"ENSG1": 3, "ENSG2": 1}
    assert stamp["n_files"] == 2


def test_a_stale_hash_stops_the_count(cache):
    _write(cache, "ENSG1", "a" * 64)
    _write(cache, "ENSG2", "0" * 64)          # extracted from an earlier CDS
    with pytest.raises(RuntimeError, match="does not match the pinned CDS"):
        bc.chunk_counts("dnabert2", pinned={"ENSG1": "a" * 64, "ENSG2": "b" * 64})


def test_a_gene_missing_from_the_manifest_stops_the_count(cache):
    _write(cache, "ENSG1", "a" * 64)
    _write(cache, "ENSG9", "c" * 64)          # left over from a gene no longer in the table
    with pytest.raises(RuntimeError, match="not in the CDS manifest"):
        bc.chunk_counts("dnabert2", pinned={"ENSG1": "a" * 64})
