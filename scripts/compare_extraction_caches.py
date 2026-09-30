"""Phase 3 pilot gate: pick the pilot genes and compare two feature caches.

``pick`` chooses genes for which the old and new code see byte-identical input:
plus-strand, unpadded windows whose span equals the May window, and whose
protein does not change under full-length translation; an equal share per
family. It writes the gene list plus template parquets that the old scripts
(commit 095dcf5, no gene flag) and the new ones take as their gene table,
sha256sum files for every input the old and the new clone will read (checked on
the box before anything runs), and
``gena_cuda_genes.tsv``: the pilot genes whose May GENA-LM TSS cache was built
on CUDA (910 of its 3,244 files were, in family order, before the run fell
back to CPU). Only those gate GENA-LM TSS against the local GPU; the rest
measure CPU vs GPU.

``compare A B`` reads two cache dirs in any format (old npz, npz with a meta
record, ESM-2 ``.npy`` or ``.npz["emb"]``) and checks every gene and array:
the same arrays on both sides (a dropped array fails unless ``--keys`` names
the ones to compare), shapes, NaNs, then either bit-identity (``--exact``) or
relative L2 error and the minimum per-row cosine against thresholds. B is the
reference. Exits 1 on any failure unless ``--report-only`` (a measurement, not
a gate).

``census DIR`` checks a finished cache against the manifest's genes: exactly
those genes, every file stamped with a meta record from one run (device, card,
torch build, and for ESM-2 the checkpoint and precision), every array finite,
and no torn ``.partial`` file (G17, G19).

Run:
  uv run scripts/compare_extraction_caches.py pick --out-dir outputs/aws_pilot
  uv run scripts/compare_extraction_caches.py compare NEW OLD --exact --label legB_tss_dnabert2
  uv run scripts/compare_extraction_caches.py census data/tss_chunk_reductions_v2_nt_v2
"""
from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA = REPO_ROOT / "data"
# The GENA-LM TSS run fell back to CPU from here on (logs/e5_cpu_run.log).
GENA_CPU_FROM = datetime.datetime(2026, 8, 13, 23, 41)
GENA_CUDA_FILES = 910  # logs/e5_cpu_run.log: "encoding pending sequences: 2334 on cpu"
TEMPLATE = DATA / "dataset_nt_v2_meanD.parquet"  # the old TSS script's default template


# --- pick ---------------------------------------------------------------------

def pilot_candidates(rows, old_spans: dict, exceptions: set) -> list[str]:
    """Genes whose old and new inputs are identical; rows are (gene, strand,
    pad_up, pad_down, start, end) from the window manifest."""
    out = []
    for gene, strand, pad_up, pad_down, start, end in rows:
        if str(strand) not in ("1", "+1") or int(pad_up) or int(pad_down):
            continue
        if old_spans.get(gene) != (int(start), int(end)) or gene in exceptions:
            continue
        out.append(gene)
    return sorted(out)


def pick_pilot(candidates, families: dict, n: int, seed: int) -> list[str]:
    """``n`` genes, an equal share per family, deterministic in the gene set."""
    rng = np.random.default_rng(seed)
    by_family: dict[str, list[str]] = {}
    for g in sorted(candidates):
        by_family.setdefault(families[g], []).append(g)
    quota = n // len(by_family)
    picked, rest = [], []
    for fam in sorted(by_family):
        genes = [by_family[fam][i] for i in rng.permutation(len(by_family[fam]))]
        picked += genes[:quota]
        rest += genes[quota:]
    rest = sorted(rest)
    picked += [rest[i] for i in rng.permutation(len(rest))[: n - len(picked)]]
    return sorted(picked)


def _old_spans(window_dir: Path) -> dict:
    spans = {}
    for fa in window_dir.glob("*.fa"):
        with fa.open() as fh:
            parts = fh.readline().strip().lstrip(">").split(":")  # chromosome:GRCh38:chr:start:end:1
        spans[fa.stem] = (int(parts[3]), int(parts[4]))
    return spans


def old_window_sha(path: Path) -> str:
    """sha256 of a May window as the old and new readers parse it (strip, skip
    the header, uppercase), comparable with the manifest's."""
    from data_loader.enformer_windows import sha256_seq

    return sha256_seq("".join(ln.strip() for ln in path.read_text().splitlines()
                              if ln and not ln.startswith(">")).upper())


def _write_sha256sums(path: Path, rel_paths: list[str]) -> None:
    """``sha256sum -c`` input, relative to the repo root."""
    lines = [f"{hashlib.sha256((REPO_ROOT / r).read_bytes()).hexdigest()}  {r}" for r in rel_paths]
    path.write_text("\n".join(lines) + "\n")


def _cuda_built(cache_dir: Path) -> set:
    cut = GENA_CPU_FROM.timestamp()
    return {p.stem for p in cache_dir.glob("*.npz") if p.stat().st_mtime < cut}


def cmd_pick(args) -> None:
    manifest = pd.read_csv(DATA / "tss_windows.tsv", sep="\t", dtype={"strand": str})
    rows = manifest[["ensembl_id", "strand", "pad_up", "pad_down", "start", "end"]].itertuples(
        index=False, name=None)
    exceptions = set(pd.read_csv(DATA / "translation_exceptions.tsv", sep="\t")["ensembl_id"])
    cands = pilot_candidates(rows, _old_spans(DATA / "enformer_windows"), exceptions)
    template = pd.read_parquet(TEMPLATE)
    families = dict(zip(template["ensembl_id"], template["family"]))
    pick = pick_pilot(cands, families, n=args.n, seed=args.seed)
    sha = dict(zip(manifest["ensembl_id"], manifest["sha256"]))
    differ = [g for g in pick if old_window_sha(DATA / "enformer_windows" / f"{g}.fa") != sha[g]]
    if differ:
        raise RuntimeError(f"May windows differ from the manifest for {differ[:5]}: not a Leg B input")
    cuda_all = _cuda_built(DATA / "tss_chunk_reductions_gena_lm")
    if len(cuda_all) != GENA_CUDA_FILES:  # file times lost in a copy would silently change the set
        raise RuntimeError(f"{len(cuda_all)} GENA-LM TSS files predate the CPU fallback, "
                           f"expected {GENA_CUDA_FILES}")
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    chosen = template[template["ensembl_id"].isin(pick)].sort_values("ensembl_id")
    chosen[["ensembl_id", "symbol", "family"]].to_csv(out / "pilot_genes.tsv", sep="\t", index=False)
    chosen.to_parquet(out / "pilot_template.parquet")
    chosen.head(args.n_repeat).to_parquet(out / "repeat_template.parquet")
    cuda = sorted(set(pick) & cuda_all)
    _write_sha256sums(out / "pilot_old_inputs.sha256",
                      [f"data/sequences/{g}.fa" for g in pick] + [f"data/enformer_windows/{g}.fa" for g in pick])
    _write_sha256sums(out / "pilot_new_inputs.sha256",
                      [f"data/sequences/{g}.fa" for g in pick] + [f"data/tss_windows_e115/{g}.fa" for g in pick])
    pd.DataFrame({"ensembl_id": cuda}).to_csv(out / "gena_cuda_genes.tsv", sep="\t", index=False)
    print(f"candidates {len(cands)}; picked {len(pick)}, May windows byte-identical; "
          f"{chosen['family'].value_counts().sort_index().to_dict()}; "
          f"GENA-LM TSS built on CUDA {len(cuda)}; "
          f"repeat subset {min(args.n_repeat, len(pick))} -> {out}")


# --- compare ------------------------------------------------------------------

@dataclass
class Result:
    rows: list = field(default_factory=list)
    failures: list = field(default_factory=list)
    keys_only_in_a: set = field(default_factory=set)
    keys_only_in_b: set = field(default_factory=set)


def load_arrays(cache_dir: Path, gene: str) -> dict[str, np.ndarray] | None:
    npz, npy = cache_dir / f"{gene}.npz", cache_dir / f"{gene}.npy"
    if npz.exists():
        with np.load(npz, allow_pickle=False) as data:
            return {k: data[k] for k in data.files if k != "meta"}
    if npy.exists():
        return {"emb": np.load(npy, allow_pickle=False)}
    return None


def _genes_in(cache_dir: Path) -> set:
    return {p.stem for p in cache_dir.iterdir() if p.suffix in (".npz", ".npy")}


def compare_arrays(a: np.ndarray, b: np.ndarray) -> dict:
    a64, b64 = a.astype(np.float64), b.astype(np.float64)
    diff = np.linalg.norm(a64 - b64)
    ref = np.linalg.norm(b64)
    rel = 0.0 if diff == 0 else (diff / ref if ref else np.inf)
    ra, rb = np.atleast_2d(a64), np.atleast_2d(b64)
    na, nb = np.linalg.norm(ra, axis=1), np.linalg.norm(rb, axis=1)
    both_zero = (na == 0) & (nb == 0)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos = np.where(both_zero, 1.0, (ra * rb).sum(axis=1) / (na * nb))
    cos = np.nan_to_num(cos, nan=0.0)
    return {"max_abs": float(np.abs(a64 - b64).max()) if a.size else 0.0, "rel_l2": float(rel),
            "min_cos": float(cos.min()) if cos.size else 1.0,
            "identical": a.dtype == b.dtype and a.tobytes() == b.tobytes()}  # bit for bit (-0.0 != 0.0)


def compare_dirs(dir_a, dir_b, *, genes=None, keys=None, exact: bool = False,
                 max_rel_l2: float | None = None, min_cos: float | None = None) -> Result:
    if not exact and (max_rel_l2 is None or min_cos is None):
        raise ValueError("give --exact or both thresholds")
    dir_a, dir_b = Path(dir_a), Path(dir_b)
    res = Result()
    genes = sorted(genes) if genes is not None else sorted(_genes_in(dir_a) | _genes_in(dir_b))
    if not genes:
        res.failures.append("no genes to compare")
    for g in genes:
        a, b = load_arrays(dir_a, g), load_arrays(dir_b, g)
        if a is None or b is None:
            res.failures.append(f"{g}: missing in {'A' if a is None else 'B'}")
            continue
        res.keys_only_in_a |= set(a) - set(b)
        res.keys_only_in_b |= set(b) - set(a)
        if keys is None and set(a) != set(b):
            res.failures.append(f"{g}: keys differ, A only {sorted(set(a) - set(b))}, "
                                f"B only {sorted(set(b) - set(a))}")
        for k in (keys or sorted(set(a) & set(b))):
            if k not in a or k not in b:
                res.failures.append(f"{g}: key {k} missing in {'A' if k not in a else 'B'}")
                continue
            if a[k].shape != b[k].shape:
                res.failures.append(f"{g}/{k}: shape {a[k].shape} != {b[k].shape}")
                continue
            if not (np.isfinite(a[k]).all() and np.isfinite(b[k]).all()):
                res.failures.append(f"{g}/{k}: NaN or inf")
                continue
            m = compare_arrays(a[k], b[k])
            res.rows.append({"gene": g, "key": k, "shape": "x".join(map(str, a[k].shape)), **m})
            if exact and not m["identical"]:
                res.failures.append(f"{g}/{k}: not bit-identical (rel_l2 {m['rel_l2']:.3g})")
            elif not exact and (m["rel_l2"] > max_rel_l2 or m["min_cos"] < min_cos):
                res.failures.append(f"{g}/{k}: rel_l2 {m['rel_l2']:.3g}, min_cos {m['min_cos']:.6f}")
    return res


# What makes two files the same run; per-gene keys (input sha, protein sha) differ.
RUN_KEYS = ("encoder", "model", "revision", "boundary_tokens", "max_content_tokens", "stride",
            "center_bins", "checkpoint_sha256", "fp16", "max_residues", "translation",
            "device", "device_name", "torch", "cuda")


def census(cache_dir, genes) -> Result:
    from data_loader.cache_meta import read_meta

    cache_dir = Path(cache_dir)
    res = Result()
    files = sorted(cache_dir.glob("*.npz")) + sorted(cache_dir.glob("*.npy"))
    torn = sorted(cache_dir.glob("*.partial"))
    if torn:
        res.failures.append(f"{len(torn)} torn .partial files, e.g. {torn[0].name}")
    have, want = {f.stem for f in files}, set(genes)
    if have != want or len(files) != len(want):
        res.failures.append(f"{len(files)} files for {len(want)} genes: missing {sorted(want - have)[:5]}, "
                            f"unexpected {sorted(have - want)[:5]}")
    runs: dict = {}
    for f in files:
        meta = read_meta(f) if f.suffix == ".npz" else None
        if meta is None:
            res.failures.append(f"{f.name}: no meta record")
            continue
        if meta.get("device_name") is None or meta.get("torch") is None:
            res.failures.append(f"{f.name}: unstamped (no device_name/torch)")
        with np.load(f, allow_pickle=False) as data:
            bad = [k for k in data.files if k != "meta" and not np.isfinite(data[k]).all()]
        if bad:
            res.failures.append(f"{f.name}: non-finite values in {bad}")
        runs.setdefault(tuple(meta.get(k) for k in RUN_KEYS), []).append(f.stem)
    if len(runs) > 1:
        res.failures.append(f"{len(runs)} runs mixed: " + "; ".join(
            f"{len(g)} genes {dict(zip(RUN_KEYS, key))}" for key, g in runs.items()))
    res.rows = [dict(zip(RUN_KEYS, key), n=len(g)) for key, g in runs.items()]
    return res


def cmd_census(args) -> int:
    res = census(args.cache_dir, _read_genes(args.genes))
    for row in res.rows:
        print(f"[{Path(args.cache_dir).name}] {row['n']} files: "
              + ", ".join(f"{k}={v}" for k, v in row.items() if k != "n" and v is not None))
    print(f"  {'FAIL' if res.failures else 'PASS'}: {len(res.failures)} failures"
          + "".join(f"\n    {f}" for f in res.failures[:10]))
    return 1 if res.failures else 0


def _read_genes(path: str | None):
    if path is None:
        return None
    p = Path(path)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, sep="\t")
    return list(df["ensembl_id"])


def cmd_compare(args) -> int:
    res = compare_dirs(args.dir_a, args.dir_b, genes=_read_genes(args.genes),
                       keys=args.keys.split(",") if args.keys else None, exact=args.exact,
                       max_rel_l2=args.max_rel_l2, min_cos=args.min_cos)
    rows = pd.DataFrame(res.rows)
    label = args.label or f"{Path(args.dir_a).name} vs {Path(args.dir_b).name}"
    mode = "exact" if args.exact else f"rel_l2<={args.max_rel_l2:g}, cos>={args.min_cos:g}"
    if len(rows):
        print(f"[{label}] {rows['gene'].nunique()} genes, keys {sorted(rows['key'].unique())}, {mode}: "
              f"identical {int(rows['identical'].sum())}/{len(rows)}; rel_l2 median "
              f"{rows['rel_l2'].median():.3g} max {rows['rel_l2'].max():.3g}; "
              f"min_cos {rows['min_cos'].min():.6f}; max_abs {rows['max_abs'].max():.3g}")
    if res.keys_only_in_a or res.keys_only_in_b:
        print(f"  keys only in A: {sorted(res.keys_only_in_a)}; only in B: {sorted(res.keys_only_in_b)}")
    if args.csv:
        Path(args.csv).parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(args.csv, index=False, quoting=csv.QUOTE_MINIMAL)
    verdict = "MEASURED" if args.report_only else ("FAIL" if res.failures else "PASS")
    print(f"  {verdict}: {len(res.failures)} {'beyond the reference thresholds' if args.report_only else 'failures'}"
          + ("".join(f"\n    {f}" for f in res.failures[:10])))
    return 1 if res.failures and not args.report_only else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("pick", help="choose the pilot genes and write their templates")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--n", type=int, default=50)
    p.add_argument("--n-repeat", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    c = sub.add_parser("compare", help="compare cache dir A against reference B")
    c.add_argument("dir_a")
    c.add_argument("dir_b")
    c.add_argument("--genes", help="tsv or parquet with an ensembl_id column (default: every file)")
    c.add_argument("--keys", help="comma-separated arrays to compare (default: the shared ones)")
    c.add_argument("--exact", action="store_true", help="require bit-identical arrays")
    c.add_argument("--max-rel-l2", type=float)
    c.add_argument("--min-cos", type=float)
    c.add_argument("--csv", help="write per-gene, per-array metrics here")
    c.add_argument("--label")
    c.add_argument("--report-only", action="store_true", help="measure; never exit 1")
    s = sub.add_parser("census", help="check a finished cache: count, one run, no torn files")
    s.add_argument("cache_dir")
    s.add_argument("--genes", default=str(DATA / "tss_windows.tsv"),
                   help="tsv or parquet with an ensembl_id column (default: the window manifest)")
    args = ap.parse_args()
    if args.cmd == "census":
        return cmd_census(args)
    if args.cmd == "pick":
        cmd_pick(args)
        return 0
    return cmd_compare(args)


if __name__ == "__main__":
    raise SystemExit(main())
