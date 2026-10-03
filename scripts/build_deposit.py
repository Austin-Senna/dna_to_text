"""Pack the Zenodo deposit (Phase 7, G30).

The deposit holds what the public repo cannot: the stored test predictions, the
per-gene feature caches, the CDS sequences and TSS windows, and the stamped
feature parquets that git does not track. Every file is chosen by a rule that
can fail, never by globbing ``data/``, where May-era caches sit next to the v2
ones:

- predictions and stamped parquets come from the canonical records (the run
  ``run_complete.json`` and ``reproduction.json`` vouch for), each checked
  against the hash its record stamps, and the predictions directory must hold
  exactly the stamped files;
- a cache or sequence directory must hold exactly one file per gene of
  ``data/gene_table.parquet``, nothing else;
- files derived from NT-v2 outputs go to their own tarballs under CC BY-NC-SA
  4.0 (the model's licence); everything else is CC BY 4.0.

The cache contents are checked by ``scripts/check_deposit.sh``, which unpacks
the tarballs into a clean copy of the repo and rebuilds every stamped parquet
with the Stage 4 builders (their G19 meta checks run unchanged). Upload only
after it passes.

Archives are byte-identical across runs (sorted members, zeroed owners and
times, gzip without a timestamp), so SHA256SUMS can be re-derived.

Run:
  uv run scripts/build_deposit.py --out outputs/deposit [--jobs 4]
  scripts/check_deposit.sh outputs/deposit
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
import posixpath
import subprocess
import sys
import tarfile
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from data_loader.model_registry import ENCODER_SPECS  # noqa: E402
from linear_trainer import records  # noqa: E402
from linear_trainer.cell import arrays_sha256  # noqa: E402

DATA = REPO_ROOT / "data"
PREDICTIONS = REPO_ROOT / "outputs" / "predictions" / "v2"
# The G1 replay writes predictions no canonical record stamps.
UNSTAMPED_PREDICTION_DIRS = ("g1_check",)
GENE_TABLE = "data/gene_table.parquet"
HGNC = "data/hgnc/hgnc_complete_set.tsv"

CC_BY = "CC-BY-4.0"
NC_LICENCE = "CC-BY-NC-SA-4.0"
NC_ENCODERS = ("nt_v2",)
# Sources computed from the DNA under an encoder's chunks, not from its outputs.
COMPOSITION_SUFFIXES = ("chunk4mergc", "chunk6mer")
TARGETS_SOURCE = "genept_targets"

PRIVATE = ("docs/reviews/", "STATUS.md", "MINA.md", "logs/")
MAX_FILES = 100   # Zenodo's per-record limit


class DepositError(RuntimeError):
    """A file the deposit would carry is missing, stale, extra or private."""


@dataclass(frozen=True)
class Stamp:
    sha256: str
    kind: str     # "arrays": arrays_sha256 of an npz; "file": sha256 of the bytes
    source: str   # the feature source whose licence the file inherits


@dataclass(frozen=True)
class Part:
    name: str
    licence: str
    files: tuple[str, ...]
    description: str


def licence_of_source(source: str) -> str:
    """CC BY-NC-SA for anything computed from an NT-v2 output; CC BY otherwise."""
    name = source.removeprefix("tss_")
    if name.startswith(NC_ENCODERS) and not name.endswith(COMPOSITION_SUFFIXES):
        return NC_LICENCE
    return CC_BY


def check_public(rel: str) -> str:
    """``rel`` normalised; refuses anything outside the repo or in a private path."""
    norm = posixpath.normpath(rel)
    if norm.startswith(("/", "..")) or any(norm == p or norm.startswith(p) for p in PRIVATE):
        raise DepositError(f"{rel} is private or outside the repo; it never goes in the deposit")
    return norm


def load_records(root: Path | None = None) -> list[dict]:
    """Every canonical record, main and null, after the records module's checks,
    from a run that finished and was independently reproduced."""
    root = records.V2 if root is None else root
    files = []
    for path in sorted(root.glob("metrics_*.json")):
        files.append(records.load(path.stem.removeprefix("metrics_") + ".json", root=root))
    for path in sorted(root.glob("null_*.json")):
        files.append(records.load(path.stem.removeprefix("null_") + ".json", null=True, root=root))
    records.check_complete(records.stamp_of(*files), root)
    records.check_reproduced(root)
    return [r for recs in files for r in recs.values()]


def stamped(recs: list[dict]) -> dict[str, Stamp]:
    """Repo-relative path -> the hash its records stamp. One path, one hash."""
    out: dict[str, Stamp] = {}

    def add(path: str, stamp: Stamp) -> None:
        path = check_public(path)
        if path in out and out[path].sha256 != stamp.sha256:
            raise DepositError(f"{path} is stamped with two hashes")
        out.setdefault(path, stamp)

    for r in recs:
        add(r["pred_file"], Stamp(r["pred_sha256"], "arrays", r["feature_source"]))
        if "targets_file" in r:
            add(r["targets_file"], Stamp(r["targets_sha256"], "arrays", TARGETS_SOURCE))
        for path, sha in _input_pairs(r.get("features", {})):
            add(path, Stamp(sha, "file", source_of_file(path, r["feature_source"])))
    return out


def _input_pairs(features: dict) -> list[tuple[str, str]]:
    """Every (path, sha256) a record's feature stamp names: ``path``/``sha256`` for a
    parquet, ``<k>``/``<k>_sha256`` for the inputs of an on-the-fly featurizer."""
    pairs = [(features["path"], features["sha256"])] if "path" in features else []
    pairs += [(features[k], features[f"{k}_sha256"]) for k in sorted(features)
              if f"{k}_sha256" in features]
    return pairs


def source_of_file(path: str, fallback: str) -> str:
    """A dataset parquet's own source (``data/dataset_<source>.parquet``), so a file
    read only for its labels keeps its encoder's licence."""
    name = Path(path).name
    if name.startswith("dataset_") and name.endswith(".parquet"):
        return name.removeprefix("dataset_").removesuffix(".parquet")
    return fallback


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    return h.hexdigest()


def verify_stamped(root: Path, stamps: dict[str, Stamp]) -> None:
    for rel, stamp in sorted(stamps.items()):
        path = root / rel
        if not path.exists():
            raise DepositError(f"{rel} is stamped by a record but missing")
        if stamp.kind == "arrays":
            with np.load(path, allow_pickle=False) as data:
                found = arrays_sha256({k: data[k] for k in data.files})
        else:
            found = _sha256(path)
        if found != stamp.sha256:
            raise DepositError(f"{rel} does not match its record ({found[:12]} != {stamp.sha256[:12]})")


def exact_files(directory: Path, ids, suffix: str, ignore: tuple[str, ...] = ()) -> list[Path]:
    """One ``<id><suffix>`` per id in ``directory`` and nothing else (``ignore`` aside)."""
    want = {f"{i}{suffix}" for i in ids}
    have = {p.name for p in Path(directory).iterdir() if p.name not in ignore}
    if have - want:
        extra = sorted(have - want)
        raise DepositError(f"{directory}: {len(extra)} unexpected files {extra[:5]}")
    if want - have:
        gone = sorted(want - have)
        raise DepositError(f"{directory}: {len(gone)} missing files {gone[:5]}")
    return [Path(directory) / name for name in sorted(want)]


def check_predictions_dir(root: Path, stamps: dict[str, Stamp]) -> None:
    """The predictions directory holds exactly the stamped files (no orphans from
    an older run would ride along)."""
    stored = {p.relative_to(root).as_posix() for p in (root / "outputs/predictions/v2").rglob("*.npz")
              if p.relative_to(root / "outputs/predictions/v2").parts[0] not in UNSTAMPED_PREDICTION_DIRS}
    wanted = {k for k, s in stamps.items() if s.kind == "arrays"}
    if stored - wanted:
        extra = sorted(stored - wanted)
        raise DepositError(f"{len(extra)} stored predictions no record stamps: {extra[:5]}")
    if wanted - stored:
        raise DepositError(f"stamped predictions missing: {sorted(wanted - stored)[:5]}")


def _by_licence(stamps: dict[str, Stamp], kind: str) -> dict[str, tuple[str, ...]]:
    out: dict[str, list[str]] = {CC_BY: [], NC_LICENCE: []}
    for rel, s in stamps.items():
        if s.kind == kind:
            out[licence_of_source(s.source)].append(rel)
    return {k: tuple(sorted(v)) for k, v in out.items()}


def prediction_parts(stamps: dict[str, Stamp]) -> list[Part]:
    files = _by_licence(stamps, "arrays")
    return [
        Part("mina_predictions_v2.tar.gz", CC_BY, files[CC_BY],
             "Stored test predictions (and GenePT targets) of every canonical probe cell"),
        Part("mina_predictions_v2_nt_v2.tar.gz", NC_LICENCE, files[NC_LICENCE],
             "Stored test predictions of the probe cells on NT-v2 features"),
    ]


def _tracked(root: Path, paths) -> set[str]:
    out = subprocess.run(["git", "ls-files", "--", *paths], cwd=root, check=True,
                         capture_output=True, text=True).stdout
    return set(out.split())


def _rel(paths: list[Path], root: Path) -> tuple[str, ...]:
    return tuple(p.relative_to(root).as_posix() for p in paths)


def plan_parts(root: Path, stamps: dict[str, Stamp]) -> list[Part]:
    genes = pd.read_parquet(root / GENE_TABLE, columns=["ensembl_id"])["ensembl_id"].tolist()
    if len(set(genes)) != len(genes):
        raise DepositError(f"{GENE_TABLE} repeats genes")
    # Stamped parquets git tracks ship in the source tarball; the rest go here.
    features = _by_licence(stamps, "file")
    tracked = _tracked(root, [*features[CC_BY], *features[NC_LICENCE]])
    untracked = {lic: tuple(f for f in fs if f not in tracked) for lic, fs in features.items()}

    def cache(directory: str, suffix: str = ".npz", ignore: tuple[str, ...] = ()) -> tuple[str, ...]:
        return _rel(exact_files(root / directory, genes, suffix, ignore), root)

    parts = [
        Part("mina_inputs.tar.gz", CC_BY,
             (*cache("data/sequences", ".fa", ignore=("_lookup",)), GENE_TABLE, HGNC, *untracked[CC_BY]),
             "CDS sequences (Ensembl 115), gene table with GenePT targets, HGNC snapshot, "
             "and the stamped E5 parquets git does not track"),
        Part("mina_tss_windows_e115.tar.gz", CC_BY, cache("data/tss_windows_e115", ".fa"),
             "Strand-aware 196,608 bp canonical-TSS windows (Ensembl 115)"),
        *prediction_parts(stamps),
        Part("mina_enformer_v2.tar.gz", CC_BY, cache("data/enformer_features_v2"),
             "Enformer per-gene features (trunk whole window and centre, tracks)"),
        Part("mina_esm2_v2.tar.gz", CC_BY,
             (*cache("data/esm2_150m_embeddings_v2"), *cache("data/esm2_650m_embeddings_v2")),
             "ESM-2 150M and 650M per-gene protein embeddings (fp32)"),
    ]
    nc: list[str] = list(untracked[NC_LICENCE])
    for name in ENCODER_SPECS:
        spec = ENCODER_SPECS[name]
        cds = cache(spec.chunk_dir.relative_to(REPO_ROOT).as_posix())
        tss = cache(spec.tss_chunk_dir.relative_to(REPO_ROOT).as_posix())
        if licence_of_source(name) == NC_LICENCE:
            nc += [*cds, *tss]
            continue
        parts += [Part(f"mina_cds_chunks_{name}.tar.gz", CC_BY, cds,
                       f"{spec.display_name}: per-chunk reductions over each CDS"),
                  Part(f"mina_tss_chunks_{name}.tar.gz", CC_BY, tss,
                       f"{spec.display_name}: per-chunk reductions over each TSS window")]
    parts.append(Part("mina_nt_v2.tar.gz", NC_LICENCE, tuple(nc),
                      "NT-v2: per-chunk reductions over each CDS and TSS window, and the "
                      "stamped NT-v2 parquets git does not track"))
    return parts


def pack_part(root: Path, part: Part, out_dir: Path) -> dict:
    """One deterministic tar.gz; returns its entry for the manifest."""
    out_dir.mkdir(parents=True, exist_ok=True)
    final = out_dir / part.name
    tmp = final.with_name(final.name + ".partial")
    files = []
    with open(tmp, "wb") as raw, \
            gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as gz, \
            tarfile.open(fileobj=gz, mode="w", format=tarfile.GNU_FORMAT) as tar:
        for rel in sorted(check_public(f) for f in part.files):
            data = (root / rel).read_bytes()
            info = tarfile.TarInfo(rel)
            info.size, info.mode, info.mtime = len(data), 0o644, 0
            tar.addfile(info, io.BytesIO(data))
            files.append({"path": rel, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)})
    tmp.replace(final)
    return {"name": part.name, "licence": part.licence, "description": part.description,
            "sha256": _sha256(final), "bytes": final.stat().st_size, "n_files": len(files),
            "files": files}


def write_sha256sums(out_dir: Path, entries: list[dict]) -> None:
    lines = [f"{e['sha256']}  {e['name']}\n" for e in sorted(entries, key=lambda e: e["name"])]
    (out_dir / "SHA256SUMS").write_text("".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, help="directory for the tarballs (not tracked)")
    ap.add_argument("--verify", type=Path, metavar="ROOT",
                    help="only check every stamped file under ROOT (check_deposit.sh's rebuilt copy)")
    ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--dry-run", action="store_true", help="check and list the parts; pack nothing")
    args = ap.parse_args()

    recs = load_records()
    stamps = stamped(recs)
    if args.verify:
        verify_stamped(args.verify.resolve(), stamps)
        print(f"{len(stamps)} stamped files under {args.verify} match their records")
        return
    if args.out is None and not args.dry_run:
        ap.error("--out is required unless --verify or --dry-run")
    dirty = subprocess.run(["git", "status", "--porcelain", "--", "src", "scripts"], cwd=REPO_ROOT,
                           check=True, capture_output=True, text=True).stdout
    if dirty and not args.dry_run:
        raise DepositError(f"uncommitted changes under src or scripts; the manifest names HEAD:\n{dirty}")
    verify_stamped(REPO_ROOT, stamps)
    check_predictions_dir(REPO_ROOT, stamps)
    parts = [p for p in plan_parts(REPO_ROOT, stamps) if p.files]
    if len(parts) + 2 > MAX_FILES:   # + deposit_manifest.json and SHA256SUMS
        raise DepositError(f"{len(parts)} tarballs exceed Zenodo's {MAX_FILES} files")
    print(f"{len(recs)} records, {len(stamps)} stamped files verified; {len(parts)} tarballs")
    for p in parts:
        print(f"  {p.name:40s} {p.licence:16s} {len(p.files):6d} files")
    if args.dry_run:
        return

    out = args.out.resolve()
    with ProcessPoolExecutor(max_workers=args.jobs) as pool:
        entries = list(pool.map(pack_part, [REPO_ROOT] * len(parts), parts, [out] * len(parts)))
    stamp = records.stamp_of({r["key"]: r for r in recs})
    manifest = {"records_git_sha": stamp["git_sha"], "protocol_hash": stamp["protocol_hash"],
                "source_git_sha": subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, check=True,
                                                 capture_output=True, text=True).stdout.strip(),
                "parts": entries}
    (out / "deposit_manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    write_sha256sums(out, [*entries, {"name": "deposit_manifest.json",
                                      "sha256": _sha256(out / "deposit_manifest.json")}])
    total = sum(e["bytes"] for e in entries)
    print(f"wrote {len(entries)} tarballs, {total / 1e9:.2f} GB, to {out}")


if __name__ == "__main__":
    main()
