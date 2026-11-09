"""Embed gene proteins with ESM-2 (#9 protein-LM comparator).

For each gene: read cached CDS -> translate to amino acids -> ESM-2 -> per-protein
embedding (mean over residues of the final-layer representation). Proteins longer
than the model's positional limit are handled by chunk-and-mean (residue-weighted).

Proteins are translated at full length (internal stops become X, G5), the same
sequences the AA k-mer baselines count. Embeddings are cached per gene as npz
with a meta record (model, checkpoint sha256, precision, device, card and torch
build, chunk cap, protein sha256); a file built differently raises instead of
being reused (G19). The checkpoint must be the pinned file (G20). Precision is
fp32 like every other feature source; ``--fp16`` exists only to replay the May
embeddings, which were fp16. Run every gene in one run: the builder refuses a mix.

Run:
  uv run scripts/run_esm2.py --size 150m
  uv run scripts/run_esm2.py --size 650m
"""
from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

import esm  # noqa: E402
from data_loader.cache_meta import read_npz, runtime_stamp, sha256_text, write_npz  # noqa: E402
from data_loader.load_checks import LoadError  # noqa: E402
from data_loader.model_registry import ESM2_CHECKPOINT_SHA256  # noqa: E402
from data_loader.sequence_fetcher import fetch_cds  # noqa: E402
from protein import translate_cds  # noqa: E402
from splits.loader import resolve_dataset_path  # noqa: E402

DATA = REPO_ROOT / "data"

# size tag -> (fair-esm loader name, embedding dim)
CHECKPOINTS = {
    "150m": ("esm2_t30_150M_UR50D", 640),
    "650m": ("esm2_t33_650M_UR50D", 1280),
}


def esm2_meta(loader_name: str, protein: str, *, fp16: bool, device: str,
              max_residues: int, checkpoint_sha256: str, runtime: dict) -> dict:
    """What an embedding depends on; build_esm2_datasets checks the same record."""
    return {"model": loader_name, "checkpoint_sha256": checkpoint_sha256, "fp16": bool(fp16),
            "device": device.split(":")[0], **runtime,
            "max_residues": int(max_residues), "translation": "through",
            "protein_sha256": sha256_text(protein)}


def checkpoint_path(loader_name: str) -> Path:
    """Where fair-esm keeps the weights it downloads for ``loader_name``."""
    return Path(torch.hub.get_dir()) / "checkpoints" / f"{loader_name}.pt"


def verify_checkpoint(path: Path, expected: str) -> str:
    """The checkpoint's sha256, if it is the pinned file; raises otherwise."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 24), b""):
            h.update(block)
    if h.hexdigest() != expected:
        raise LoadError(f"{path} has sha256 {h.hexdigest()}, not the pinned {expected}")
    return expected


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


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--size", choices=list(CHECKPOINTS), required=True)
    ap.add_argument("--template", type=Path, default=None,
                    help="dataset parquet for the gene list (default: auto-resolve)")
    ap.add_argument("--seq-cache", type=Path, default=DATA / "sequences")
    ap.add_argument("--out-cache", type=Path, default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--fp16", action="store_true",
                    help="half precision, only to replay the May fp16 embeddings (default: fp32)")
    ap.add_argument("--max-residues", type=int, default=1022,
                    help="per-chunk residue cap (ESM-2 positional limit is 1024 incl. BOS/EOS)")
    return ap


def main() -> None:
    args = build_parser().parse_args()

    loader_name, dim = CHECKPOINTS[args.size]
    device = _pick_device(args.device)
    use_fp16 = args.fp16
    if use_fp16 and device != "cuda":
        raise ValueError("--fp16 needs a cuda device")
    out_cache = args.out_cache or (DATA / f"esm2_{args.size}_embeddings_v2")
    out_cache.mkdir(parents=True, exist_ok=True)

    template = args.template or resolve_dataset_path()
    meta = pd.read_parquet(template, columns=["ensembl_id"])
    gene_ids = list(meta["ensembl_id"])
    print(f"=== ESM-2 {loader_name} (dim {dim}) device={device} fp16={use_fp16} ===")
    print(f"  genes: {len(gene_ids)}  seq cache: {args.seq_cache}  out: {out_cache}")

    model, alphabet = getattr(esm.pretrained, loader_name)()  # downloads if absent
    ckpt_sha = verify_checkpoint(checkpoint_path(loader_name), ESM2_CHECKPOINT_SHA256[loader_name])
    runtime = runtime_stamp(device)
    batch_converter = alphabet.get_batch_converter()
    repr_layer = model.num_layers
    model = model.eval().to(device)
    if use_fp16:
        model = model.half()

    done = skipped = 0
    skips: list[str] = []
    for eid in tqdm(gene_ids, desc=f"esm2 {args.size}"):
        out_path = out_cache / f"{eid}.npz"
        cds = fetch_cds(eid, args.seq_cache)
        aa = translate_cds(cds, mode="through") if cds else ""
        if not aa:
            skipped += 1
            skips.append(eid)
            continue
        rec = esm2_meta(loader_name, aa, fp16=use_fp16, device=device,
                        max_residues=args.max_residues, checkpoint_sha256=ckpt_sha,
                        runtime=runtime)
        if read_npz(out_path, rec) is not None:
            done += 1
            continue
        emb = embed_protein(aa, model, batch_converter, repr_layer, device, args.max_residues)
        write_npz(out_path, {"emb": emb}, rec)
        done += 1

    print(f"  done: {done} cached, skipped {skipped} (no CDS/empty translation)")
    if skips:
        print(f"  skipped ids: {skips[:10]}{' ...' if len(skips) > 10 else ''}")


if __name__ == "__main__":
    main()
