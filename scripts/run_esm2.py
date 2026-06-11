"""Embed gene proteins with ESM-2 (#9 protein-LM comparator).

For each gene: read cached CDS -> translate to amino acids -> ESM-2 -> per-protein
embedding (mean over residues of the final-layer representation). Proteins longer
than the model's positional limit are handled by chunk-and-mean (residue-weighted).
Embeddings are cached per gene so dataset assembly + probing can reuse them.

Run (sequential on an 8 GB card):
  uv run scripts/run_esm2.py --size 150m
  uv run scripts/run_esm2.py --size 650m
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import esm  # noqa: E402
from data_loader.sequence_fetcher import fetch_cds  # noqa: E402
from protein import translate_cds  # noqa: E402
from splits.loader import resolve_dataset_path  # noqa: E402

DATA = REPO_ROOT / "data"

# size tag -> (fair-esm loader name, embedding dim)
CHECKPOINTS = {
    "150m": ("esm2_t30_150M_UR50D", 640),
    "650m": ("esm2_t33_650M_UR50D", 1280),
}


def _pick_device(choice: str) -> str:
    if choice != "auto":
        return choice
    return "cuda" if torch.cuda.is_available() else "cpu"


@torch.inference_mode()
def embed_protein(seq: str, model, batch_converter, repr_layer: int, device: str,
                  max_residues: int) -> np.ndarray:
    """Residue-weighted mean of final-layer representations over (chunked) residues."""
    chunks = [seq[i:i + max_residues] for i in range(0, len(seq), max_residues)]
    total = None
    n_res = 0
    for chunk in chunks:
        _, _, toks = batch_converter([("p", chunk)])
        toks = toks.to(device)
        out = model(toks, repr_layers=[repr_layer], return_contacts=False)
        rep = out["representations"][repr_layer][0]  # [len(chunk)+2, d]
        # drop BOS (0) and EOS (len(chunk)+1); sum over residues
        res = rep[1:len(chunk) + 1].float().sum(0)
        total = res if total is None else total + res
        n_res += len(chunk)
    return (total / max(n_res, 1)).cpu().numpy().astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", choices=list(CHECKPOINTS), required=True)
    ap.add_argument("--template", type=Path, default=None,
                    help="dataset parquet for the gene list (default: auto-resolve)")
    ap.add_argument("--seq-cache", type=Path, default=DATA / "sequences")
    ap.add_argument("--out-cache", type=Path, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--fp16", dest="fp16", action="store_true", default=None,
                    help="half precision (default: on for cuda)")
    ap.add_argument("--no-fp16", dest="fp16", action="store_false")
    ap.add_argument("--max-residues", type=int, default=1022,
                    help="per-chunk residue cap (ESM-2 positional limit is 1024 incl. BOS/EOS)")
    args = ap.parse_args()

    loader_name, dim = CHECKPOINTS[args.size]
    device = _pick_device(args.device)
    use_fp16 = (device == "cuda") if args.fp16 is None else args.fp16
    out_cache = args.out_cache or (DATA / f"esm2_{args.size}_embeddings")
    out_cache.mkdir(parents=True, exist_ok=True)

    template = args.template or resolve_dataset_path()
    meta = pd.read_parquet(template, columns=["ensembl_id"])
    gene_ids = list(meta["ensembl_id"])
    print(f"=== ESM-2 {loader_name} (dim {dim}) device={device} fp16={use_fp16} ===")
    print(f"  genes: {len(gene_ids)}  seq cache: {args.seq_cache}  out: {out_cache}")

    model, alphabet = getattr(esm.pretrained, loader_name)()
    batch_converter = alphabet.get_batch_converter()
    repr_layer = model.num_layers
    model = model.eval().to(device)
    if use_fp16:
        model = model.half()

    done = skipped = 0
    skips: list[str] = []
    for eid in tqdm(gene_ids, desc=f"esm2 {args.size}"):
        out_path = out_cache / f"{eid}.npy"
        if out_path.exists():
            done += 1
            continue
        cds = fetch_cds(eid, args.seq_cache)
        aa = translate_cds(cds, to_stop=True) if cds else ""
        if not aa:
            skipped += 1
            skips.append(eid)
            continue
        emb = embed_protein(aa, model, batch_converter, repr_layer, device, args.max_residues)
        np.save(out_path, emb)
        done += 1

    print(f"  done: {done} cached, skipped {skipped} (no CDS/empty translation)")
    if skips:
        print(f"  skipped ids: {skips[:10]}{' ...' if len(skips) > 10 else ''}")


if __name__ == "__main__":
    main()
