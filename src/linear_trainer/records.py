"""The camera-ready records (``data/v2``): one loader for every builder.

``scripts/recompute_all.py`` writes one JSON array per split file. Every
table, figure and statistic reads them through ``load``, which refuses:

- records from more than one commit or protocol, in one file or across the
  files a builder combines (``stamp_of``; G7, G12);
- a record fitted on another version of its split file than the one on disk;
- a key recorded twice;
- records fitted under another protocol than ``protocol.V2`` (a trial at
  another thread count, ``recompute_all.py --threads``);
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
from linear_trainer.protocol import V2 as V2_PROTOCOL
from linear_trainer.selection import MissingRecord, select_pool
from splits.leaks import rules_for

REPO_ROOT = Path(__file__).resolve().parents[2]
V2 = REPO_ROOT / "data" / "v2"
ENCODERS = tuple(ENCODER_SPECS)
AA_KMERS = ("aa1", "aa2", "aa3")
NT_KMERS = ("kmer", "kmer6")
# Composition plus CDS length (Rule 3 control): k is selected on validation within it.
AA_KMERS_LEN = tuple(f"{k}_len" for k in AA_KMERS)
NT_KMERS_LEN = tuple(f"{k}_len" for k in NT_KMERS)


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


def load(split: str, *, null: bool = False, root: Path | None = None) -> dict[str, dict]:
    """Records of one split file, keyed by cell key, after the G7/G2 checks.

    ``root`` defaults to ``V2``, looked up at call time.
    """
    path = _path(split, null, V2 if root is None else root)
    if not path.exists():
        raise MissingRecord(f"no records file {path.name}; run scripts/recompute_all.sh")
    recs = json.loads(path.read_text())
    out: dict[str, dict] = {}
    commits = {(r["stamp"]["git_sha"], r["protocol_hash"]) for r in recs}
    if len(commits) != 1:
        raise MixedRecords(f"{path.name} mixes {len(commits)} commit/protocol stamps")
    (_, proto), = commits
    if proto != V2_PROTOCOL.hash:   # a trial at another thread count (recompute_all --threads)
        raise MixedRecords(f"{path.name}: fitted under protocol {proto[:8]}, not "
                           f"{V2_PROTOCOL.name} ({V2_PROTOCOL.hash[:8]})")
    inputs = _purge_inputs()
    split_sha = _sha(REPO_ROOT / "data" / Path(split).name)
    for r in recs:
        if r["key"] in out:
            raise MixedRecords(f"{path.name}: {r['key']} recorded twice")
        if r["split"] != Path(split).name:
            raise MixedRecords(f"{path.name}: {r['key']} names split {r['split']}")
        want = list(rules_for(r["split"], r["arm"]))
        if r["purge"]["rules"] != want:
            raise MixedRecords(f"{r['key']}: purge rules {r['purge']['rules']}, policy {want}")
        purge_split = r["purge"].get("split_sha256", None if want else split_sha)
        if split_sha != r["splits_sha256"] or split_sha != purge_split:
            raise MixedRecords(f"{r['key']}: fitted on another version of {r['split']}")
        needed = ({"pairs_sha256"} if any(x.startswith("protein@") for x in want) else set()) | \
                 ({"windows_sha256"} if "window" in want else set())
        for k in needed:
            if r["purge"].get(k) != inputs[k]:
                raise MixedRecords(f"{r['key']}: purge built from another (or no) {k.split('_')[0]} file")
        out[r["key"]] = r
    return out


COMPLETE = "run_complete.json"


class IncompleteRun(RuntimeError):
    """No whole-manifest run (completeness and G1) vouches for these records."""


def check_complete(stamp: dict, root: Path | None = None) -> dict:
    """The marker an unsharded ``recompute_all.py --group all`` writes after every
    cell has one record and G1 passed, for the same commit and protocol (G17, G1)."""
    path = (V2 if root is None else root) / COMPLETE
    if not path.exists():
        raise IncompleteRun(f"no {COMPLETE}: finish with an unsharded recompute_all.py --group all")
    marker = json.loads(path.read_text())
    if (marker["git_sha"], marker["protocol_hash"]) != (stamp["git_sha"], stamp["protocol_hash"]):
        raise IncompleteRun(f"{COMPLETE} vouches for another run ({marker['git_sha'][:7]})")
    return marker


def input_digests(root: Path | None = None) -> dict[str, str]:
    """sha256 of every records file, so an output built from them can tell when one
    was rewritten at the same commit (a rerun after re-extraction keeps the stamp)."""
    root = V2 if root is None else root
    return {p.name: _sha(p) for p in sorted(root.glob("metrics_*.json")) + sorted(root.glob("null_*.json"))}


def check_inputs(built: dict, root: Path | None = None) -> None:
    """Refuse an output (statistics.json, ridge_robust.json) whose input records changed."""
    now = input_digests(root)
    stale = sorted(k for k, v in built["inputs"].items() if now.get(k) != v)
    if stale:
        raise MixedRecords(f"built from older versions of {stale}; rebuild it")


REPRODUCED = "reproduction.json"
MIN_REPRO_BOOT = 1000


class NotReproduced(RuntimeError):
    """The independent reimplementation has not matched these records (Rule 3)."""


def check_reproduced(root: Path | None = None) -> dict:
    """The verdict ``scripts/reproduce_headline.py`` writes after refitting the
    headline cells without the pipeline's code: it must pass, and for exactly the
    records files on disk now (a rerun at the same commit rewrites them)."""
    root = V2 if root is None else root
    path = root / REPRODUCED
    if not path.exists():
        raise NotReproduced(f"no {REPRODUCED}: run scripts/reproduce_headline.py")
    verdict = json.loads(path.read_text())
    if verdict.get("ok") is not True:
        raise NotReproduced(f"{REPRODUCED} records failures: {verdict.get('failures')}")
    if verdict.get("inputs") != input_digests(root):
        raise NotReproduced(f"{REPRODUCED} checked other versions of the records files; rerun it")
    # A partial check (--cells, a short bootstrap) can pass; it does not vouch.
    expected = set(verdict.get("cells_expected", [])) | set(range(1, 11))
    missing = sorted(expected - set(verdict.get("cells_run", [])))
    if missing or verdict.get("n_boot", 0) < MIN_REPRO_BOOT:
        raise NotReproduced(f"{REPRODUCED} is partial: cells not refitted {missing}, "
                            f"{verdict.get('n_boot')} bootstrap resamples (need {MIN_REPRO_BOOT})")
    return verdict


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


def best_aa_len(by_source: dict[str, dict]) -> str:
    """The validation-selected amino-acid k-mer with CDS length."""
    return pick(by_source, list(AA_KMERS_LEN))


def best_nt_kmer_len(by_source: dict[str, dict]) -> str:
    """The validation-selected nucleotide k-mer with CDS length."""
    return pick(by_source, list(NT_KMERS_LEN))


def best_nt_kmer(by_source: dict[str, dict]) -> str:
    """The validation-selected nucleotide k-mer, 4 or 6 (G6: k is selected for
    nucleotide composition as it is for amino acids)."""
    return pick(by_source, list(NT_KMERS))
