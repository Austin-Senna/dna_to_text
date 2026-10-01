"""The camera-ready records (``data/v2``): one loader for every builder.

``scripts/recompute_all.py`` writes one JSON array per split file. Every
table, figure and statistic reads them through ``load``, which refuses:

- records from more than one commit or protocol, in one file or across the
  files a builder combines (``load_many``; G7, G12);
- a key recorded twice;
- a record whose purge differs from the policy for its split (G2): other
  rules, or masks derived from another pair table or window manifest, so a
  cell run without its purge can't reach a table.

Selection helpers pick among explicitly named candidates on validation scores
only (``selection.select_pool``, G1 and G14), and every named candidate must
have a record: a partial file raises instead of picking from fewer.
"""
from __future__ import annotations

import json
from pathlib import Path

from data_loader.model_registry import ENCODER_SPECS, encoder_pools
from linear_trainer.selection import MissingRecord, select_pool
from splits.leaks import rules_for

REPO_ROOT = Path(__file__).resolve().parents[2]
V2 = REPO_ROOT / "data" / "v2"
ENCODERS = tuple(ENCODER_SPECS)
AA_KMERS = ("aa1", "aa2", "aa3")
NT_KMERS = ("kmer", "kmer6")


class MixedRecords(RuntimeError):
    """A records file mixes commits, protocols or purge policies, or repeats a key."""


def _path(split: str, null: bool, root: Path) -> Path:
    stem = Path(split).stem
    return root / (f"null_{stem}.json" if null else f"metrics_{stem}.json")


def _sha(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _purge_inputs() -> dict[str, str]:
    from splits.leaks import PAIRS, WINDOWS
    return {"pairs_sha256": _sha(PAIRS), "windows_sha256": _sha(WINDOWS)}


def load(split: str, *, null: bool = False, root: Path = V2) -> dict[str, dict]:
    """Records of one split file, keyed by cell key, after the G7/G2 checks."""
    path = _path(split, null, root)
    if not path.exists():
        raise MissingRecord(f"no records file {path.name}; run scripts/recompute_all.sh")
    recs = json.loads(path.read_text())
    out: dict[str, dict] = {}
    commits = {(r["stamp"]["git_sha"], r["protocol_hash"]) for r in recs}
    if len(commits) != 1:
        raise MixedRecords(f"{path.name} mixes {len(commits)} commit/protocol stamps")
    inputs = _purge_inputs()
    for r in recs:
        if r["key"] in out:
            raise MixedRecords(f"{path.name}: {r['key']} recorded twice")
        if r["split"] != Path(split).name:
            raise MixedRecords(f"{path.name}: {r['key']} names split {r['split']}")
        want = list(rules_for(r["split"], r["arm"]))
        if r["purge"]["rules"] != want:
            raise MixedRecords(f"{r['key']}: purge rules {r['purge']['rules']}, policy {want}")
        for k, v in inputs.items():
            if k in r["purge"] and r["purge"][k] != v:
                raise MixedRecords(f"{r['key']}: purge built from another {k.split('_')[0]} file")
        out[r["key"]] = r
    return out


def stamp_of(*files: dict[str, dict]) -> dict:
    """The one (commit, protocol) every given records file shares (G7)."""
    stamps = {(r["stamp"]["git_sha"], r["protocol_hash"]) for recs in files for r in recs.values()}
    if len(stamps) != 1:
        raise MixedRecords(f"the records combined here come from {len(stamps)} commit/protocol stamps")
    (sha, proto), = stamps
    return {"git_sha": sha, "protocol_hash": proto}


def cells(recs: dict[str, dict], arm: str, task: str) -> dict[str, dict]:
    """Source -> record for one arm and task (shuffled runs excluded)."""
    return {r["feature_source"]: r for r in recs.values()
            if r["arm"] == arm and r["task"] == task and not r["shuffled_labels"]}


def pick(by_source: dict[str, dict], candidates: list[str]) -> str:
    """``select_pool`` over candidates that must all have a record."""
    missing = [c for c in candidates if c not in by_source]
    if missing:
        raise MissingRecord(f"candidates without a record: {missing}")
    return select_pool(by_source, candidates)


def best_pool(by_source: dict[str, dict], encoder: str, arm: str) -> str:
    """The encoder's validation-selected pool on this arm."""
    prefix = "tss_" if arm == "tss" else ""
    pools = encoder_pools(encoder, "TSS" if arm == "tss" else "CDS")
    return pick(by_source, [f"{prefix}{encoder}_{p}" for p in pools])


def best_encoder(by_source: dict[str, dict], arm: str) -> str:
    """The validation-selected cell across the encoders' own picks."""
    return pick(by_source, [best_pool(by_source, e, arm) for e in ENCODERS])


def encoder_of(source: str) -> str:
    name = source.removeprefix("tss_")
    return next(e for e in ENCODERS if name.startswith(e + "_"))


def best_aa(by_source: dict[str, dict]) -> str:
    """The validation-selected amino-acid k-mer (k selected on validation, G6)."""
    return pick(by_source, list(AA_KMERS))


def best_nt_kmer(by_source: dict[str, dict]) -> str:
    """The validation-selected nucleotide k-mer, 4 or 6 (G6: k is selected for
    nucleotide composition as it is for amino acids)."""
    return pick(by_source, list(NT_KMERS))
