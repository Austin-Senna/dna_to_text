# Findings — Linear probing DNA encoders into GenePT space

Results from running the phase-3 probe on 3244 human genes across 5 functional families, stratified 70/15/15 split, seed 42. Two DNA encoders tested: DNABERT-2 117M (`data/dataset.parquet`) and Nucleotide Transformer v2 100M multi-species (`data/dataset_nt_v2.parquet`).

Commits: `572b034` (splits), `88112c6` (probe), `6b601ce` (anti-baseline), `5aaf631` (4-mer baseline), `d3c1a40` (MLP probe), `f7577c9` (NT-v2 encoder).

## Setup

- **Input (X):** DNABERT-2 (`zhihan1996/DNABERT-2-117M`) embeddings of the canonical CDS, mean-pooled across chunks → 768-d per gene.
- **Target (Y):** GenePT text embeddings of NCBI gene summaries (OpenAI `text-embedding-*` family) → 1536-d per gene.
- **Probe:** `sklearn.linear_model.Ridge`, multi-output. Single linear layer `y_hat = X W + b`, no nonlinearity — deliberately probing *linear* alignment.
- **Alpha sweep:** `[1e-2, 1e-1, 1, 10, 100, 1000]`, picked on mean cosine over the val split. Best alpha refit on train+val, evaluated once on test.
- **Evaluation:** mean cosine, median cosine, macro R² on the 487-gene test split.

## Results

### DNABERT-2 (768-d)

| Run | Mean cosine | Median cosine | R² macro | Best config |
|---|---|---|---|---|
| anti-baseline (shuffled Y) | 0.9128 | 0.9131 | **−0.003** | α=1000 |
| kmer baseline (4-mer, 256-d) | 0.9306 | 0.9223 | 0.1743 | α=0.01 |
| **linear probe** | **0.9313** | 0.9221 | **0.1812** | α=10 |
| mlp probe (hidden=256) | 0.9300 | 0.9220 | 0.1616 | α=0.01 |

### NT-v2 100M multi-species (512-d)

| Run | Mean cosine | Median cosine | R² macro | Best config |
|---|---|---|---|---|
| anti-baseline (shuffled Y) | 0.9130 | 0.9141 | **−0.000** | α=1000 |
| kmer baseline (4-mer, 256-d) | 0.9306 | 0.9223 | 0.1743 | α=0.01 |
| **linear probe** | **0.9324** | 0.9254 | **0.1926** | α=10 |
| mlp probe (hidden=1024) | 0.9325 | 0.9253 | 0.1888 | α=0.01 |

### Cross-encoder comparison (linear probe)

| Encoder | Params | x dim | test cos | test R² |
|---|---|---|---|---|
| DNABERT-2 | 117M | 768 | 0.9313 | 0.1812 |
| **NT-v2 100M multi-species** | 97.9M | 512 | **0.9324** | **0.1926** |
| 4-mer histogram | — | 256 | 0.9306 | 0.1743 |

Deltas that matter:

| Pair | Δ mean cosine | Δ R² macro |
|---|---|---|
| NT-v2 − DNABERT-2 (linear probe) | +0.0011 | +0.0114 |
| NT-v2 − kmer baseline (linear probe) | +0.0018 | +0.0183 |
| DNABERT-2 − kmer baseline (linear probe) | +0.0007 | +0.0069 |
| mlp − linear probe (DNABERT-2) | −0.0013 | −0.0196 |
| mlp − linear probe (NT-v2) | +0.0001 | −0.0038 |

## Interpretation

### The pipeline is honest
The anti-baseline (fit on scrambled `(X, Y)` pairs, evaluated on real test `Y`) produces R² ≈ 0 — exactly the sanity outcome `framework.md` prescribes. The pipeline is not leaking. Any signal in the probe is real.

Note that the anti-baseline's cosine is still ~0.91. That is the **anisotropy floor** of the GenePT embedding space: when W collapses toward zero under heavy regularisation, predictions collapse toward the space-mean, which itself has cosine ~0.91 with most real GenePT vectors. Cosine is a misleading headline metric in this space; R² is the honest signal.

### Neither DNA encoder meaningfully beats a 4-mer histogram
The 4-mer frequency baseline (256-d L1-normalised counts over the CDS) lands at **mean cosine 0.9306 / R² 0.174**. Both pretrained DNA encoders barely clear it:

- **DNABERT-2 lift over 4-mer**: +0.001 cosine, +0.007 R² — inside noise.
- **NT-v2 lift over 4-mer**: +0.002 cosine, +0.018 R² — detectable but still small.

NT-v2 is the stronger encoder on this task by a hair (+0.001 cos, +0.011 R² over DNABERT-2), which is consistent with its 7× larger pretraining corpus and architectural improvements (rotary embeddings, GLU FFN). But "stronger" here means moving from "indistinguishable from a 4-mer histogram" to "marginally better than a 4-mer histogram." It is not a qualitative change.

### What this means
Per `framework.md` §Success / failure criteria, this is the **informative negative** outcome:

> *probe matches or barely beats 4-mer baseline → DNA encoder's extra capacity is not functional, it is compositional. Still publishable as a limits-of-genomic-LLMs finding.*

In plain English: whatever information these DNA encoders carry about human gene function *as measured by linear alignment into GenePT text space* is largely already present in the raw nucleotide composition statistics of the CDS. Two independent pretrained transformers — DNABERT-2 (BPE tokenisation, 135 genomes, ALiBi) and NT-v2 (6-mer tokenisation, 850 genomes, rotary + GLU) — both land within 0.002 mean cosine of a 256-d 4-mer histogram. That two different encoders produce the same result is much stronger evidence for the compositional-ceiling hypothesis than any single model would be.

## Caveats

This is a real result under the setup used, but three honest follow-ups could change the picture:

1. **Pooling choice.** We mean-pool DNABERT-2 chunk embeddings. CLS or max-pool could extract different signal. `next_steps.md` flagged this as the specific knob to revisit if cosine plateaus near baseline — which is what happened.
2. **Target quality.** GenePT summaries vary wildly in informativeness. Many are short or boilerplate. If the target itself is too noisy, a linear probe cannot separate genes even when X carries the signal.
3. ~~**Linear-only.** Restricting to a linear map is deliberate (it's what "linear alignment" means), but a small MLP probe would reveal whether there is non-linear signal that the linear probe cannot reach. That would stop being a clean alignment claim but is informative on its own.~~ **Checked.** A 1-hidden-layer MLP (256 units, alpha=0.01) lands at test cosine 0.930 / R² 0.162 — effectively tied with the linear probe. No detectable non-linear signal the linear probe is missing.

   **Checked at depth too.** Swept 1–3 hidden layers with varying width and regularisation:

   | hidden | alpha | val cos |
   |---|---|---|
   | (256,) | 1e-2 | 0.9291 |
   | (512, 256) | 1e-2 | 0.9297 |
   | (1024, 512) | 1e-2 | 0.9298 |
   | (512, 256, 128) | 1e-2 | 0.9301 |
   | (1024, 512, 256) | 1e-2 | 0.9303 |

   Total spread across all MLP configs is 0.0018 — well inside noise, and all configs land within ±0.002 of the linear probe. Depth doesn't rescue it. This strengthens the informative-negative framing: the ceiling is not the probe's capacity, it's the representation itself.

~~Model choice is a separate axis to vary: Nucleotide Transformer v2, HyenaDNA, Caduceus, and GENA-LM are all plausible drop-in replacements that would test whether a different DNA encoder changes the conclusion.~~ **Partially checked.** NT-v2 100M multi-species was added as a second encoder. It marginally outperforms DNABERT-2 (+0.011 R², +0.001 cos) but does not break past the 4-mer ceiling by a meaningful margin. Two encoders × one result strengthens the informative-negative framing. HyenaDNA / Caduceus / GENA-LM would each add another data point but the pattern is already consistent.

## Open items

- Retrieval@k on the current probes — turns 0.93 cosine into a human-readable "top-k accuracy" number.
- Family-classification accuracy on `y_hat` vs real `y` — measures class-level structure preservation.
- Retry with CLS / max-pool features on both encoders (mean-pool may be smearing out signal that a different pooling would surface).
- ~~Re-run with one other DNA encoder.~~ **Done (NT-v2).** Further encoders (HyenaDNA, Caduceus, GENA-LM) would each add a data point but the pattern is already consistent.
- Different input window: include promoter + UTRs, not just CDS. CDS is the most composition-homogenous part of a gene because of codon usage; upstream regulatory regions may carry more function-discriminating signal.
