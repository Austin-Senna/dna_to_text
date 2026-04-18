# Findings — DNABERT-2 linear probe

Results from running the phase-3 probe on `data/dataset.parquet` (3244 human genes across 5 functional families, stratified 70/15/15 split, seed 42).

Commits: `572b034` (splits), `88112c6` (probe), `6b601ce` (anti-baseline), `5aaf631` (4-mer baseline).

## Setup

- **Input (X):** DNABERT-2 (`zhihan1996/DNABERT-2-117M`) embeddings of the canonical CDS, mean-pooled across chunks → 768-d per gene.
- **Target (Y):** GenePT text embeddings of NCBI gene summaries (OpenAI `text-embedding-*` family) → 1536-d per gene.
- **Probe:** `sklearn.linear_model.Ridge`, multi-output. Single linear layer `y_hat = X W + b`, no nonlinearity — deliberately probing *linear* alignment.
- **Alpha sweep:** `[1e-2, 1e-1, 1, 10, 100, 1000]`, picked on mean cosine over the val split. Best alpha refit on train+val, evaluated once on test.
- **Evaluation:** mean cosine, median cosine, macro R² on the 487-gene test split.

## Results

| Run | Mean cosine | Median cosine | R² macro | Best config |
|---|---|---|---|---|
| anti-baseline (shuffled Y) | 0.9128 | 0.9131 | **−0.003** | α=1000 |
| kmer baseline (4-mer, 256-d) | 0.9306 | 0.9223 | 0.1743 | α=0.01 |
| **linear probe (DNABERT-2, 768-d)** | **0.9313** | 0.9221 | **0.1812** | α=10 |
| mlp probe (1 hidden, 256 units) | 0.9300 | 0.9220 | 0.1616 | hidden=(256,), α=0.01 |

Deltas that matter:

| Pair | Δ mean cosine | Δ R² macro |
|---|---|---|
| probe − anti-baseline | +0.019 | +0.184 |
| **probe − kmer baseline** | **+0.001** | **+0.007** |
| **mlp − linear probe** | **−0.001** | **−0.020** |

## Interpretation

### The pipeline is honest
The anti-baseline (fit on scrambled `(X, Y)` pairs, evaluated on real test `Y`) produces R² ≈ 0 — exactly the sanity outcome `framework.md` prescribes. The pipeline is not leaking. Any signal in the probe is real.

Note that the anti-baseline's cosine is still ~0.91. That is the **anisotropy floor** of the GenePT embedding space: when W collapses toward zero under heavy regularisation, predictions collapse toward the space-mean, which itself has cosine ~0.91 with most real GenePT vectors. Cosine is a misleading headline metric in this space; R² is the honest signal.

### The probe barely beats a 4-mer histogram
The 4-mer frequency baseline (256-d L1-normalised counts over the CDS) lands at **mean cosine 0.9306 / R² 0.174** — effectively tied with the DNABERT-2 probe (0.9313 / 0.181). The lift from using a 117M-parameter pretrained DNA transformer over a bag-of-4-mers is:

- Cosine: **+0.001** — inside noise.
- R²: **+0.007** — negligible.

### What this means
Per `framework.md` §Success / failure criteria, this is the **informative negative** outcome:

> *probe matches or barely beats 4-mer baseline → DNABERT-2's extra capacity is not functional, it is compositional. Still publishable as a limits-of-genomic-LLMs finding.*

In plain English: whatever information DNABERT-2's representation carries about human gene function *as measured by linear alignment into GenePT text space* is already present in the raw nucleotide composition statistics of the CDS. The transformer has not learned semantic features beyond what a 256-d 4-mer histogram already encodes for this task.

## Caveats

This is a real result under the setup used, but three honest follow-ups could change the picture:

1. **Pooling choice.** We mean-pool DNABERT-2 chunk embeddings. CLS or max-pool could extract different signal. `next_steps.md` flagged this as the specific knob to revisit if cosine plateaus near baseline — which is what happened.
2. **Target quality.** GenePT summaries vary wildly in informativeness. Many are short or boilerplate. If the target itself is too noisy, a linear probe cannot separate genes even when X carries the signal.
3. ~~**Linear-only.** Restricting to a linear map is deliberate (it's what "linear alignment" means), but a small MLP probe would reveal whether there is non-linear signal that the linear probe cannot reach. That would stop being a clean alignment claim but is informative on its own.~~ **Checked.** A 1-hidden-layer MLP (256 units, alpha=0.01) lands at test cosine 0.930 / R² 0.162 — effectively tied with the linear probe. No detectable non-linear signal the linear probe is missing. This strengthens the informative-negative framing: the ceiling is not the probe's capacity, it's the representation itself.

Model choice is a separate axis to vary: Nucleotide Transformer v2, HyenaDNA, Caduceus, and GENA-LM are all plausible drop-in replacements that would test whether a different DNA encoder changes the conclusion.

## Open items

- Retrieval@k on the current probe — turns 0.93 cosine into a human-readable "top-k accuracy" number.
- Family-classification accuracy on `y_hat` vs real `y` — measures class-level structure preservation.
- Retry with CLS / max-pool DNABERT-2 features.
- Re-run with one other DNA encoder (recommend NT-v2 100M multi-species) for a second data point.
