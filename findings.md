# Findings — Linear probing DNA encoders into GenePT space

We test whether a simple linear map can transport DNA-language-model embeddings into the GenePT text-embedding space well enough to recover gene function. Two encoders are compared — DNABERT-2 (117M) and Nucleotide Transformer v2 100M multi-species — against a 4-mer composition baseline and a shuffled-target anti-baseline, each on the same 3244-gene corpus and the same frozen 70/15/15 split (seed 42, stratified by family).

Relevant commits: `572b034` splits, `88112c6` probe, `6b601ce` anti-baseline, `5aaf631` kmer baseline, `d3c1a40` MLP probe, `f7577c9` NT-v2 encoder, `2895ad4` NT-v2 experiments.

> Methods detail: see `framework.md`. Forward-looking work: see `next_steps.md`.

## Setup

- **Corpus.** 3244 human protein-coding genes spanning 5 functional families (TF, GPCR, kinase, ion channel, immune receptor). Split 70/15/15 stratified by family.
- **Targets (Y).** GenePT text embeddings of NCBI gene summaries (OpenAI `text-embedding-*` family), 1536-d per gene. Held constant across every run.
- **DNA encoders (X).**
  - DNABERT-2 (`zhihan1996/DNABERT-2-117M`): BPE tokeniser, ALiBi, 768-d. Trained on 135 genomes.
  - NT-v2 100M multi-species (`InstaDeepAI/nucleotide-transformer-v2-100m-multi-species`): 6-mer non-overlapping tokeniser, rotary embeddings, GLU FFN, 512-d. Trained on 850 genomes.
  - Both run with mean-pooling over chunks of the canonical CDS.
- **Probes.**
  - **Linear probe** (primary): `sklearn.linear_model.Ridge`, multi-output. `y_hat = XW + b`. Alpha swept over `[1e-2 … 1e3]` on val, refit on train+val, evaluated once on test.
  - **MLP probe** (diagnostic for non-linear signal): 1–3 hidden layers, ReLU, sklearn `MLPRegressor`, same sweep/refit/eval protocol.
- **Controls.**
  - **4-mer histogram baseline.** 256-d L1-normalised 4-mer frequencies of the CDS fed through the same Ridge recipe. Encoder-independent — same numbers regardless of X.
  - **Shuffled-Y anti-baseline.** Same Ridge recipe but Y permuted in train+val, then evaluated against real test Y. Sanity gate from `framework.md`: if this scores non-trivially, the pipeline is leaking.
- **Metrics.** Mean cosine, median cosine, macro R² on the 487-gene test split.

## Results

### Per-encoder

**DNABERT-2** (768-d)

| Run | Mean cosine | Median cosine | R² macro | Best config |
|---|---|---|---|---|
| anti-baseline (shuffled Y) | 0.9128 | 0.9131 | **−0.003** | α=1000 |
| 4-mer baseline | 0.9306 | 0.9223 | 0.1743 | α=0.01 |
| **linear probe** | **0.9313** | 0.9221 | **0.1812** | α=10 |
| MLP probe | 0.9300 | 0.9220 | 0.1616 | hidden=(256,), α=0.01 |

**NT-v2 100M multi-species** (512-d)

| Run | Mean cosine | Median cosine | R² macro | Best config |
|---|---|---|---|---|
| anti-baseline (shuffled Y) | 0.9130 | 0.9141 | **−0.000** | α=1000 |
| 4-mer baseline | 0.9306 | 0.9223 | 0.1743 | α=0.01 |
| **linear probe** | **0.9324** | 0.9254 | **0.1926** | α=10 |
| MLP probe | 0.9325 | 0.9253 | 0.1888 | hidden=(1024,), α=0.01 |

### Side-by-side

| Probe | Encoder | x dim | test cos | test R² |
|---|---|---|---|---|
| 4-mer histogram | — | 256 | 0.9306 | 0.1743 |
| Linear | DNABERT-2 | 768 | 0.9313 | 0.1812 |
| Linear | **NT-v2** | 512 | **0.9324** | **0.1926** |

Deltas that matter:

| Pair | Δ mean cosine | Δ R² macro |
|---|---|---|
| DNABERT-2 − 4-mer (linear probe) | +0.0007 | +0.0069 |
| NT-v2 − 4-mer (linear probe) | +0.0018 | +0.0183 |
| NT-v2 − DNABERT-2 (linear probe) | +0.0011 | +0.0114 |
| Anti-baseline − 4-mer (R² only) | −0.18 | — |

### MLP depth sweep (DNABERT-2)

Swept 1–3 hidden layers to rule out the "non-linear signal a linear probe can't reach" story:

| hidden | alpha | val cos |
|---|---|---|
| (256,) | 1e-2 | 0.9291 |
| (512, 256) | 1e-2 | 0.9297 |
| (1024, 512) | 1e-2 | 0.9298 |
| (512, 256, 128) | 1e-2 | 0.9301 |
| (1024, 512, 256) | 1e-2 | 0.9303 |

Total spread across all MLP configs is 0.002. All configs land within ±0.002 of the linear probe. Depth does not help. The ceiling is the representation, not the probe's capacity.

## Interpretation

### The pipeline is honest
Both anti-baselines produce macro R² ≈ 0 on real test Y. The anti-baseline's cosine still sits around 0.91, which is the **anisotropy floor** of GenePT space: under heavy regularisation W collapses toward zero, predictions collapse toward mean-Y, and mean-Y has cosine ~0.91 with essentially every real GenePT vector. Cosine is a misleading headline in this space. R² is the honest signal.

### Both DNA encoders barely clear the 4-mer baseline
The 4-mer frequency baseline (256-d L1-normalised counts) scores cosine 0.9306 / R² 0.174. Both pretrained DNA encoders land within a hair of this number:

- **DNABERT-2**: +0.001 cos, +0.007 R² over 4-mer — inside noise.
- **NT-v2**: +0.002 cos, +0.018 R² over 4-mer — detectable but small.

NT-v2 is the stronger encoder by a hair (+0.001 cos, +0.011 R² over DNABERT-2), consistent with its 7× larger pretraining corpus and architectural improvements (rotary + GLU). But "stronger" means moving from *indistinguishable from a 4-mer histogram* to *marginally better than a 4-mer histogram*. It is not a qualitative change.

### No non-linear signal the linear probe is missing
A 1–3 hidden layer MLP ties the linear probe within 0.002 cosine on both encoders. Going wider or deeper does not move the number. That rules out the "there is non-linear signal the linear probe can't reach" hypothesis and localises the ceiling to the representation itself.

### What this means
Per `framework.md` §Success / failure criteria, this is the **informative negative** outcome:

> *probe matches or barely beats 4-mer baseline → DNA encoder's extra capacity is not functional, it is compositional. Still publishable as a limits-of-genomic-LLMs finding.*

Whatever information these DNA encoders carry about human gene function — *as measured by linear alignment into GenePT text space, trained on CDS only* — is largely already present in the raw nucleotide composition of the CDS. Two independent pretrained transformers, built on different tokenisations and different pretraining corpora, both land within 0.002 mean cosine of a 256-d 4-mer histogram. Two encoders converging on the same ceiling is stronger evidence for the compositional-ceiling read than any single model would be.

## Caveats

Three axes that could change the conclusion:

1. **Pooling.** We mean-pool chunk embeddings on both encoders. CLS, max-pool, or attention-weighted pooling could surface signal that mean-pool smears out. `next_steps.md` flagged this as the specific knob to revisit if cosine plateaued at baseline — which it did.
2. **Target quality.** GenePT summaries vary wildly in informativeness; many are short or boilerplate. If the target itself is noisy, a linear probe cannot separate genes even when X carries real signal. Filtering short or generic summaries before training would isolate this.
3. **Input window.** We embed the canonical CDS only. CDS is the most composition-homogenous part of a gene because of codon usage. Promoter + UTRs may carry more function-discriminating signal; a fair test would embed the full transcript or gene body rather than CDS alone.

## Open items

- Retrieval@k on the current probes — turns 0.93 cosine into a human-readable "for X% of test genes, the predicted vector's top-k nearest real summaries contain the correct one."
- Family-classification accuracy: logistic regression on `y_hat` → `family`, compared to the same classifier on real `y`. Measures how much class-level structure survives the projection.
- Retry with CLS / max-pool features on both encoders.
- Retry with promoter + UTR + CDS input on both encoders.
- Optional: add HyenaDNA, Caduceus, or GENA-LM as a third encoder. Each additional encoder converging on the same ceiling further strengthens the read, but the pattern is already consistent across two very different models.
