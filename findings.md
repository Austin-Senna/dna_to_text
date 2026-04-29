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

## Phase 4 — Classification reframing

Per the 2026-04-29 classification-pivot spec, we re-ran the question as a *classification* task on the same cached embeddings. Three tasks:

- **5-way:** predict family ∈ {tf, gpcr, kinase, ion, immune} on the full 3244-gene corpus, original 70/15/15 split.
- **tf-vs-gpcr (binary):** 591 of each, downsampled tf, frozen split in `data/binary_tf_vs_gpcr.json`.
- **tf-vs-kinase (binary):** 558 of each, frozen split in `data/binary_tf_vs_kinase.json`.

Probe: logistic regression (L2, multinomial for 5-way), C swept on val over `[1e-2 … 1e3]`, refit on train+val, evaluated once on test. Headline metric: macro-F1.

Relevant commits: `e99f5f3` binary subsets, `679af53` length baseline, `b0b2df5` logistic probe + matrix script, `14be331` matrix run.

### Results

| Task | Encoder / baseline | C | Test macro-F1 | Test bal. acc | Test acc |
|---|---|---:|---:|---:|---:|
| **5-way** | **NT-v2** | 1.0 | **0.8031** | 0.7748 | 0.8727 |
| 5-way | 4-mer | 1000 | 0.6722 | 0.6319 | 0.8172 |
| 5-way | DNABERT-2 | 10 | 0.6490 | 0.6170 | 0.7721 |
| 5-way | shuffled-label | 100 | 0.2078 | 0.2119 | 0.4353 |
| 5-way | length-only | 0.1 | 0.1382 | 0.1954 | 0.5236 |
| **tf-vs-gpcr** | **NT-v2** | 0.1 | **0.9775** | 0.9775 | 0.9775 |
| tf-vs-gpcr | 4-mer | 1000 | 0.9607 | 0.9607 | 0.9607 |
| tf-vs-gpcr | DNABERT-2 | 1.0 | 0.9381 | 0.9382 | 0.9382 |
| tf-vs-gpcr | length-only | 1.0 | 0.7341 | 0.7360 | 0.7360 |
| tf-vs-gpcr | shuffled-label | 0.1 | 0.4659 | 0.4663 | 0.4663 |
| **tf-vs-kinase** | **NT-v2** | 100 | **0.8444** | 0.8452 | 0.8452 |
| tf-vs-kinase | 4-mer | 1000 | 0.8392 | 0.8393 | 0.8393 |
| tf-vs-kinase | DNABERT-2 | 1.0 | 0.8330 | 0.8333 | 0.8333 |
| tf-vs-kinase | length-only | 0.01 | 0.6427 | 0.6429 | 0.6429 |
| tf-vs-kinase | shuffled-label | 0.1 | 0.5474 | 0.5476 | 0.5476 |

Δ macro-F1 vs the 4-mer baseline:

| Task | NT-v2 − 4-mer | DNABERT-2 − 4-mer |
|---|---:|---:|
| 5-way | **+0.1308** | −0.0232 |
| tf-vs-gpcr | +0.0169 | −0.0226 |
| tf-vs-kinase | +0.0052 | −0.0063 |

### Read

The classification reframing surfaced signal that regression hid — but only for the stronger encoder. **NT-v2 beats the 4-mer baseline on the 5-way task by +0.131 macro-F1**, well above the 0.02 decision threshold. NT-v2 also beats kmer on tf-vs-gpcr (+0.017) and ties on tf-vs-kinase (+0.005). DNABERT-2 ties or loses to kmer on every task, consistent with the Phase 3 read.

Two things changed at once between Phase 3 and Phase 4: the target (1536-d GenePT vector → categorical family) and the metric (R² / cosine → macro-F1). Either could in principle have surfaced the NT-v2 signal. Practically the target change is the bigger lever — GenePT summaries vary wildly in informativeness, and a noisy 1536-d target dilutes every gene's contribution to the regression loss. A categorical label has no such noise floor.

The encoder-level divergence is the more interesting finding. NT-v2 (850 genome pretraining, rotary embeddings, GLU FFN, 6-mer non-overlapping tokeniser, 100M params) carries family-discriminative information that survives mean-pooling and a linear probe; DNABERT-2 (135 genome pretraining, ALiBi, BPE, 117M params) does not. The two encoders are nominally similar in size and both pretrained on multi-species genomes; the 7× larger pretraining corpus and the architectural changes are the candidates for the gap.

### Confusion (5-way)

NT-v2 confusion matrix (`data/confusion_5way_nt_v2.json`, rows = true, cols = pred, classes alphabetical):

```
         gpcr immune  ion kinase   tf
gpcr       83      1    4      1    0
immune      1     16    1      3    2
ion         2      0   16      4    8
kinase      0      1    0     65   18
tf          3      0    1     12  245
```

DNABERT-2 (`data/confusion_5way_dnabert2.json`):

```
         gpcr immune  ion kinase   tf
gpcr       76      0    3      3    7
immune      0     10    0      7    6
ion         4      1    9      9    7
kinase      0      1    2     52   29
tf          2      2    2     26  229
```

DNABERT-2 collapses much of the kinase population into tf (29/84 misclassified as tf vs NT-v2's 18/84) and underperforms across every minority class. NT-v2 holds clean diagonals on tf (94%), gpcr (93%), kinase (77%), immune (70%); ion is the weakest at 53%, and most ion misclassifications go to tf or kinase rather than gpcr — biologically reasonable since ion channels are a structurally heterogeneous group.

### Anti-baseline

Shuffled-label runs landed at macro-F1 0.208 / 0.466 / 0.547 for {family5, tf-vs-gpcr, tf-vs-kinase} — analytical chance ≈ 0.20 / 0.50 / 0.50. All within ±0.05 of chance. Pipeline is honest.

### Length baseline

CDS log-length alone hits 0.138 / 0.734 / 0.643. The 0.73 on tf-vs-gpcr is a real asymmetry (membrane vs nuclear protein length distributions differ), but it's well below kmer (0.961) and well below the encoders. The encoder is not a length proxy.

### Decision gate outcome

Per the spec: **Branch 1 (write up)** — at least one encoder beat 4-mer by ≥ 0.02 macro-F1 on at least one task. Phase 4b pooling re-extraction is **skipped**. The negative-result framing from Phase 3 is now bounded: it applies to DNABERT-2 + regression + GenePT-as-target; it does not generalise to NT-v2 + classification + family-as-target.

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
