"""The counts the paper states about its genes (D5), each with its denominator.

* **Single-chunk share per encoder:** the genes whose CDS fits one encoder
  window, so that Ends + Mean (first chunk, last chunk, mean) is Mean repeated
  three times. Read from the v2 chunk caches (``data/chunk_reductions_v2_<enc>/``;
  ``n_chunks`` is the row count of a gene's per-chunk arrays). Denominators: all
  genes, the test genes of the CDS primary split, and those test genes left after
  the evaluation purge.
* **Noisy TF labels and templated GenePT summaries** (``data_loader.label_audit``),
  over all genes and over the same test sets.

Writes ``data/v2/counts.json``. Reads no probe records.

Run: uv run scripts/build_counts.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from data_loader import label_audit
from data_loader.model_registry import ENCODER_SPECS
from linear_trainer import records as R
from linear_trainer import stats
from splits.leaks import PAIRS, purge_for

DATA = Path(__file__).resolve().parents[1] / "data"
CDS = "splits.json"
OUT = R.V2 / "counts.json"
# The cache fields that pin what was run; the per-gene input hash is folded into the digest.
RUN_KEYS = ("encoder", "model", "revision", "max_content_tokens", "stride", "boundary_tokens",
            "device_name", "torch", "cuda")


def chunk_counts(encoder: str) -> tuple[dict[str, int], dict]:
    """Gene -> number of chunks, from the encoder's v2 cache, and the cache's stamp."""
    cache = ENCODER_SPECS[encoder].chunk_dir
    files = sorted(cache.glob("*.npz"))
    if not files:
        raise FileNotFoundError(f"no v2 chunk cache at {cache}")
    counts, runs, digest = {}, set(), hashlib.sha256()
    for f in files:
        with np.load(f, allow_pickle=False) as z:
            n = int(z["mean"].shape[0])
            meta = json.loads(str(z["meta"]))
        counts[f.stem] = n
        runs.add(tuple(str(meta.get(k)) for k in RUN_KEYS))
        digest.update(f"{f.stem}\t{n}\t{meta.get('input_sha256')}\n".encode())
    if len(runs) != 1:
        raise RuntimeError(f"{cache} mixes {len(runs)} extraction runs")
    return counts, {"cache": cache.relative_to(DATA.parent).as_posix(), "n_files": len(files),
                    **dict(zip(RUN_KEYS, runs.pop())), "counts_sha256": digest.hexdigest()}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _share(genes: set[str], single: set[str]) -> dict:
    n = len(genes)
    k = len(genes & single)
    return {"single_chunk": k, "of": n, "share": k / n}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    split_path = DATA / CDS
    test = set(json.loads(split_path.read_text())["test"])
    scored = test - purge_for(split_path, "cds").test
    gene_table, hgnc, inputs = label_audit.load_inputs()
    every = set(gene_table["ensembl_id"])
    denominators = {"all genes": every, f"{CDS} test": test, f"{CDS} test after the purge": scored}

    chunks, caches = {}, {}
    for e in ENCODER_SPECS:
        counts, caches[e] = chunk_counts(e)
        if set(counts) != every:
            raise RuntimeError(f"{e}: the cache covers {len(counts)} genes, the gene table {len(every)}")
        single = {g for g, n in counts.items() if n == 1}
        chunks[e] = {name: _share(genes, single) for name, genes in denominators.items()}
        chunks[e]["max_chunks"] = max(counts.values())

    noisy = label_audit.noisy_tf_genes(gene_table, hgnc)
    groups = label_audit.shared_summary_groups(gene_table)
    templated = label_audit.templated_genes(gene_table)
    family = dict(zip(gene_table["ensembl_id"], gene_table["family"]))
    largest = next(iter(groups.values()))

    def within(genes: frozenset[str]) -> dict:
        return {name: len(genes & d) for name, d in denominators.items()}

    result = {
        "inputs": {**inputs, CDS: _sha(split_path), "protein_pairs": _sha(PAIRS)},
        "caches": caches,
        "single_chunk": chunks,
        "noisy_tf_labels": {"n": within(noisy),
                            "of_tf_labelled": int(sum(f == "tf" for f in family.values()))},
        "templated_summaries": {
            "n": within(templated), "n_groups": len(groups),
            "largest_group": {"n": len(largest), "families": sorted({family[g] for g in largest})},
            "empty_summary": {"n": within(frozenset(groups.get("", [])))},
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    stats.write_json(args.out, result)
    print(f"wrote {args.out}")
    for e, row in chunks.items():
        print(f"  {e}: " + ", ".join(f"{k} {v['share']:.1%} of {v['of']}"
                                     for k, v in row.items() if isinstance(v, dict)))
    print(f"  noisy TF labels {result['noisy_tf_labels']['n']}; templated {result['templated_summaries']['n']}, "
          f"{len(groups)} groups, largest {len(largest)}")


if __name__ == "__main__":
    main()
