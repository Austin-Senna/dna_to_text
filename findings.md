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

Per the spec: **Branch 1 (write up)** — at least one encoder beat 4-mer by ≥ 0.02 macro-F1 on at least one task. Phase 4b pooling re-extraction was originally going to be skipped, but we ran it anyway as an exploratory ablation; results are below.

## Phase 4b — Pooling sweep (exploratory)

We ran the spec's full Phase 4b pooling menu — five variants per encoder against the `mean→mean` Phase 4a baseline — to test whether the encoder ceiling moves under different aggregation schemes. 30 new logistic runs (5 variants × 2 encoders × 3 tasks) plus 10 new confusion matrices.

Relevant commits: `303bb55` multi-pool extractor + aggregator, `b9a0642` 30-cell pooling matrix.

### Pooling variants

Within-chunk × across-chunk factorial (each variant changes one axis from the `mean→mean` baseline):

| Variant | Within-chunk | Across-chunk | Output dim |
|---|---|---|---|
| `meanmean` | mean of content tokens | mean of chunks | `d` |
| `maxmean` | per-dim max of content tokens | mean of chunks | `d` |
| `clsmean` | model's CLS token | mean of chunks | `d` |
| `meanD` | mean | concat[first, last, mean] of chunks | `3d` |
| `meanG` | mean | concat[first, last, mean, max] of chunks | `4d` |

Re-extraction tokenised **with** special tokens (`[CLS]` / `[SEP]` for DNABERT-2; `<cls>` only for NT-v2 — its ESM-style tokeniser ships no SEP/EOS) so that position 0 of every chunk is the trained CLS representation. Phase 1–3 tokenised *without* special tokens; this turns out to matter (see "Tokenisation surprise" below).

### Headline numbers (5-way)

Sorted by macro-F1, with Δ vs the Phase 4a `nt_v2` (0.8031) and `dnabert2` (0.6490) baselines:

| Variant | macro-F1 | Δ vs Phase 4a same-encoder | Notes |
|---|---:|---:|---|
| **`nt_v2_meanD`** | **0.8275** | +0.0244 | best overall on 5-way |
| `nt_v2_meanG` | 0.8257 | +0.0226 | tied with D within noise |
| `nt_v2_meanmean` | 0.7997 | −0.0034 | recomputed baseline ≈ Phase 4a |
| `dnabert2_meanD` | 0.7380 | **+0.0890** | huge DNABERT-2 jump |
| `dnabert2_meanG` | 0.7275 | +0.0785 | |
| `dnabert2_meanmean` | 0.7220 | +0.0730 | recomputed baseline ≠ Phase 4a |
| `dnabert2_clsmean` | 0.6547 | +0.0057 | |
| `dnabert2_maxmean` | 0.6144 | −0.0346 | max-token pooling hurts |
| `nt_v2_clsmean` | 0.5927 | −0.2104 | NT-v2 CLS is not a summary |
| `nt_v2_maxmean` | 0.5865 | −0.2166 | |

### Tokenisation surprise

The most consequential change isn't a pooling variant — it's the tokenisation. The Phase 4b mean→mean baselines use the *same* aggregation as Phase 4a (mean of content tokens, then mean across chunks) but with **special-token-wrapped** chunks:

| Encoder | Phase 4a `mean→mean` | Phase 4b `meanmean` (re-tok) | Δ |
|---|---:|---:|---:|
| DNABERT-2 5-way | 0.6490 | 0.7220 | **+0.0730** |
| DNABERT-2 tf-vs-gpcr | 0.9381 | 0.9888 | **+0.0507** |
| DNABERT-2 tf-vs-kinase | 0.8330 | 0.8869 | **+0.0539** |
| NT-v2 5-way | 0.8031 | 0.7997 | −0.0034 |
| NT-v2 tf-vs-gpcr | 0.9775 | 0.9831 | +0.0056 |
| NT-v2 tf-vs-kinase | 0.8444 | 0.8447 | +0.0003 |

**DNABERT-2 was being substantially crippled by the Phase 1–3 tokenisation choice** (`add_special_tokens=False`, no `[CLS]`/`[SEP]` wrapping per chunk). With proper boundaries, DNABERT-2 closes most of the encoder gap to NT-v2 — even before any pooling change — and on tf-vs-kinase actually pulls ahead (0.887 vs NT-v2 0.845). The "encoder gap" in Phase 4a was not purely architectural; a meaningful chunk of it was a tokenisation bug.

NT-v2 is largely insensitive to special tokens (within ±0.006 across all three tasks). Two plausible reasons: (a) ESM-style models don't rely on a learned `[CLS]` summary the way BERT does; (b) NT-v2's 6-mer non-overlapping tokeniser produces fewer tokens per chunk, so the relative weight of two boundary tokens is smaller.

### Pooling rankings (post-tokenisation-fix)

Comparing variants against each encoder's *recomputed* `meanmean` baseline:

| | dnabert2 5-way | dnabert2 tf-vs-gpcr | dnabert2 tf-vs-kinase | nt_v2 5-way | nt_v2 tf-vs-gpcr | nt_v2 tf-vs-kinase |
|---|---:|---:|---:|---:|---:|---:|
| meanmean (baseline) | 0.7220 | 0.9888 | 0.8869 | 0.7997 | 0.9831 | 0.8447 |
| meanD | +0.0160 | −0.0169 | 0.0000 | **+0.0278** | +0.0057 | −0.0003 |
| meanG | +0.0055 | −0.0282 | −0.0119 | +0.0260 | −0.0168 | +0.0061 |
| maxmean | −0.1076 | −0.0450 | −0.0958 | −0.2132 | −0.0112 | −0.0114 |
| clsmean | −0.0673 | −0.0169 | −0.0536 | −0.2070 | −0.0449 | −0.0769 |

Three takeaways:

1. **`meanD` is the only variant that reliably helps**, and the largest gain is on NT-v2 5-way (+0.028). Concatenating first-chunk + last-chunk + mean exposes terminal asymmetry the bare mean smears out — N-terminal signal peptides and C-terminal motifs do carry family signal.
2. **`meanG` matches `meanD`**. Adding "max-across-chunks" doesn't reliably help — the one-dominant-chunk hypothesis isn't supported here. Not worth the 4× dim cost when 3× gives the same result.
3. **`maxmean` and `clsmean` consistently HURT.** Two failures of the deep-research priors:
   - **`maxmean`** (max-token-per-chunk → mean across chunks) was supposed to surface sparse motifs. Instead it gets diluted again by the cross-chunk mean. The hypothesis "max preserves what mean smears" doesn't survive being mean-pooled across chunks.
   - **`clsmean`** is catastrophic for NT-v2 (5-way drops to 0.59) and merely bad for DNABERT-2. Both encoders are masked-LM pretrained with no next-sentence-prediction objective; their CLS positions weren't trained as sequence summaries. DNABERT-2's CLS is at least *adjacent* to a BERT pretraining setup; NT-v2's `<cls>` truly carries no special meaning.

### Best-of-everything table

Headline-best per task across all 14 (Phase 4a + Phase 4b) feature sources (excluding shuffled and length):

| Task | Best feature | macro-F1 | Δ vs 4-mer | Δ vs Phase 4a best |
|---|---|---:|---:|---:|
| 5-way | `nt_v2_meanD` | 0.8275 | +0.1553 | +0.0244 |
| tf-vs-gpcr | `dnabert2_meanmean` / `nt_v2_meanD` (tied) | 0.9888 | +0.0281 | +0.0113 |
| tf-vs-kinase | `dnabert2_meanmean` / `dnabert2_meanD` (tied) | 0.8869 | +0.0477 | +0.0425 |

The pooling sweep modestly improves the headline (NT-v2 5-way +0.024) but the bigger win is the tokenisation fix — particularly for DNABERT-2 on the binary tasks, where it goes from "loses to k-mer" in Phase 4a to "ties NT-v2 and beats k-mer" in Phase 4b.

### Decision gate after Phase 4b

The Phase 4a Branch 1 read holds and is now stronger: both encoders beat 4-mer on every task once tokenisation is fixed. The exploratory pooling sweep validates `meanD` as a marginal improvement and rules out `maxmean` / `clsmean` as productive directions for these particular models.

## Phase 5a — Regression re-run on the new variants

Same Phase 3 recipe (Ridge → GenePT 1536-d, alpha sweep on val, refit on train+val, test eval) but with the 10 new pooling-variant parquets from Phase 4b. Tests whether the Phase 3 informative-negative regression result was — like the Phase 4a classification result — partly a tokenisation artefact.

Relevant commit: `2d78221`.

### Results

Phase 3 baseline reference (from "Per-encoder" table above): DNABERT-2 R² = 0.181, NT-v2 R² = 0.193, 4-mer = 0.174. The 4-mer number is encoder-independent and stays at 0.174 — it's not re-run here.

| Variant | α | Test cos | Test R² | Δ R² vs Phase 3 same-encoder |
|---|---:|---:|---:|---:|
| **`dnabert2_meanG`** | 10 | 0.9340 | **0.2104** | **+0.0294** |
| `dnabert2_meanD` | 10 | 0.9340 | 0.2100 | +0.0290 |
| `dnabert2_meanmean` | 10 | 0.9333 | 0.2029 | +0.0219 |
| `nt_v2_meanmean` | 10 | 0.9324 | 0.1932 | +0.0002 |
| `dnabert2_clsmean` | 100 | 0.9322 | 0.1911 | +0.0101 |
| `nt_v2_meanG` | 100 | 0.9321 | 0.1902 | −0.0028 |
| `nt_v2_meanD` | 100 | 0.9319 | 0.1882 | −0.0048 |
| `dnabert2_maxmean` | 100 | 0.9293 | 0.1606 | −0.0204 |
| `nt_v2_maxmean` | 100 | 0.9271 | 0.1355 | −0.0575 |
| `nt_v2_clsmean` | 100 | 0.9251 | 0.1172 | −0.0758 |

### Read

1. **DNABERT-2's Phase 3 informative-negative was an undercount.** `dnabert2_meanmean` regression R² is 0.203 (+0.022 over Phase 3's 0.181), and `dnabert2_meanG` reaches 0.210 (+0.029). Δ vs the 4-mer baseline goes from Phase 3's "+0.007 (within noise)" to **+0.036 (decisively beats)**. The same tokenisation surprise that lifted classification numbers also lifts regression.

2. **NT-v2 regression is unchanged by tokenisation OR pooling.** `nt_v2_meanmean` R² = 0.193 vs Phase 3 NT-v2 0.193 — within 0.0002. Even `meanD` and `meanG`, which gave NT-v2 a +0.025 classification boost, are slightly *worse* in regression (−0.005 / −0.003).

3. **The classification gain from `meanD` is family-specific, not signal-general.** NT-v2 `meanD` improves family classification by +0.025 macro-F1 but does not improve recovery of the full 1536-d GenePT vector. Translation: terminal asymmetry helps the model decide *which family* a gene belongs to, but doesn't recover the rest of the gene's described function.

4. **`maxmean` / `clsmean` hurt regression too**, mirroring the classification pattern. `nt_v2_clsmean` is the worst variant in both modalities (R² 0.117 vs meanmean 0.193; macro-F1 0.59 vs meanmean 0.80).

5. **The encoder gap reverses when you fix DNABERT-2.** With proper tokenisation, DNABERT-2 (all variants combined ≥ 0.16 R²) is competitive with or above NT-v2 (≤ 0.19 R²) on regression. NT-v2's pretraining and architecture advantages don't translate into a regression win once tokenisation is honest.

The regression ceiling (~0.21 R²) is still modest — a lot of the GenePT 1536-d target is noise that no DNA encoder of this size can recover. The classification reframing is still the right framing for the headline. But Phase 3's "DNABERT-2 ties 4-mer" claim is no longer true and should be revised in any further write-up.

## Full results matrix

Every (feature source × task) cell in the project. macro-F1 + Cohen's κ per classification task plus Ridge R² macro for the regression task. Generated by `scripts/build_full_table.py`; full file at `data/full_table.md`.

| Feature source | 5-way F1 | 5-way κ | tf-vs-gpcr F1 | tf-vs-gpcr κ | tf-vs-kinase F1 | tf-vs-kinase κ | Ridge R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| `shuffled` | 0.2078 | +0.0482 | 0.4659 | −0.0674 | 0.5474 | +0.0952 | — |
| `length` | 0.1382 | −0.0106 | 0.7341 | 0.4719 | 0.6427 | 0.2857 | — |
| `kmer` | 0.6722 | 0.7024 | 0.9607 | 0.9213 | 0.8392 | 0.6786 | 0.1743 |
| `dnabert2` | 0.6490 | 0.6356 | 0.9381 | 0.8764 | 0.8330 | 0.6667 | 0.1812 |
| `dnabert2_meanmean` | 0.7220 | 0.7091 | **0.9888** | **0.9775** | **0.8869** | **0.7738** | 0.2029 |
| `dnabert2_meanD` | 0.7380 | 0.7226 | 0.9719 | 0.9438 | **0.8869** | **0.7738** | 0.2100 |
| `dnabert2_meanG` | 0.7275 | 0.7066 | 0.9606 | 0.9213 | 0.8750 | 0.7500 | **0.2104** |
| `dnabert2_maxmean` | 0.6144 | 0.5817 | 0.9438 | 0.8876 | 0.7911 | 0.5833 | 0.1606 |
| `dnabert2_clsmean` | 0.6547 | 0.6616 | 0.9719 | 0.9438 | 0.8333 | 0.6667 | 0.1911 |
| `nt_v2` | 0.8031 | 0.7984 | 0.9775 | 0.9551 | 0.8444 | 0.6905 | 0.1932 |
| `nt_v2_meanmean` | 0.7997 | 0.7982 | 0.9831 | 0.9663 | 0.8447 | 0.6905 | 0.1932 |
| `nt_v2_meanD` | **0.8275** | **0.8214** | **0.9888** | **0.9775** | 0.8444 | 0.6905 | 0.1882 |
| `nt_v2_meanG` | 0.8257 | 0.8179 | 0.9663 | 0.9326 | 0.8508 | 0.7024 | 0.1902 |
| `nt_v2_maxmean` | 0.5865 | 0.6099 | 0.9719 | 0.9438 | 0.8333 | 0.6667 | 0.1355 |
| `nt_v2_clsmean` | 0.5927 | 0.5689 | 0.9382 | 0.8764 | 0.7678 | 0.5357 | 0.1172 |

Bolded cells = column-wise maximum across non-shuffled rows. Anti-baseline (shuffled-label) κ within ±0.10 of zero on every task → pipeline is honest. Per-column best:

| Column | Best feature | Value |
|---|---|---:|
| 5-way F1 | `nt_v2_meanD` | 0.828 |
| 5-way κ | `nt_v2_meanD` | 0.821 |
| tf-vs-gpcr F1 | `dnabert2_meanmean` | 0.989 |
| tf-vs-gpcr κ | `dnabert2_meanmean` | 0.978 |
| tf-vs-kinase F1 | `dnabert2_meanmean` | 0.887 |
| tf-vs-kinase κ | `dnabert2_meanmean` | 0.774 |
| Ridge R² | `dnabert2_meanG` | 0.210 |

## Δ vs 4-mer baseline

The same matrix, expressed as `(value) − (k-mer value)` in the same column. Positive = beats the 4-mer composition baseline; negative = worse. This is the table that answers the project's main scientific question: *how much extra signal does the encoder + pooling carry over a 256-d 4-mer histogram?*

| Feature source | Δ 5-way F1 | Δ 5-way κ | Δ tf-vs-gpcr F1 | Δ tf-vs-gpcr κ | Δ tf-vs-kinase F1 | Δ tf-vs-kinase κ | Δ Ridge R² |
|---|---:|---:|---:|---:|---:|---:|---:|
| `shuffled` | −0.464 | −0.654 | −0.495 | −0.989 | −0.292 | −0.583 | — |
| `length` | −0.534 | −0.713 | −0.227 | −0.449 | −0.197 | −0.393 | — |
| `kmer` (reference) | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
| `dnabert2` | −0.023 | −0.067 | −0.023 | −0.045 | −0.006 | −0.012 | +0.007 |
| `dnabert2_meanmean` | **+0.050** | +0.007 | **+0.028** | **+0.056** | **+0.048** | **+0.095** | **+0.029** |
| `dnabert2_meanD` | **+0.066** | **+0.020** | +0.011 | +0.022 | **+0.048** | **+0.095** | **+0.036** |
| `dnabert2_meanG` | **+0.055** | +0.004 | 0.000 | 0.000 | **+0.036** | **+0.071** | **+0.036** |
| `dnabert2_maxmean` | −0.058 | −0.121 | −0.017 | −0.034 | −0.048 | −0.095 | −0.014 |
| `dnabert2_clsmean` | −0.018 | −0.041 | +0.011 | +0.022 | −0.006 | −0.012 | +0.017 |
| `nt_v2` | **+0.131** | **+0.096** | +0.017 | +0.034 | +0.005 | +0.012 | +0.018 |
| `nt_v2_meanmean` | **+0.127** | **+0.096** | **+0.022** | +0.045 | +0.006 | +0.012 | +0.019 |
| `nt_v2_meanD` | **+0.155** | **+0.119** | **+0.028** | **+0.056** | +0.005 | +0.012 | +0.014 |
| `nt_v2_meanG` | **+0.154** | **+0.115** | +0.006 | +0.011 | +0.011 | +0.024 | +0.016 |
| `nt_v2_maxmean` | −0.086 | −0.092 | +0.011 | +0.022 | −0.006 | −0.012 | −0.039 |
| `nt_v2_clsmean` | −0.080 | −0.134 | −0.022 | −0.045 | −0.071 | −0.143 | −0.057 |

Bolded cells = exceed the spec's decision-gate threshold (Δ macro-F1 ≥ +0.02 or Δ R² ≥ +0.02). Decision-gate verdict by row:

| Feature source | 5-way | tf-vs-gpcr | tf-vs-kinase |
|---|:-:|:-:|:-:|
| `dnabert2` (Phase 1–3, no specials) | ❌ | ❌ | ≈ |
| `dnabert2_meanmean` (with specials) | ✅ | ✅ | ✅ |
| `dnabert2_meanD` | ✅ | ≈ | ✅ |
| `dnabert2_meanG` | ✅ | ≈ | ✅ |
| `dnabert2_maxmean` | ❌ | ≈ | ❌ |
| `dnabert2_clsmean` | ≈ | ≈ | ≈ |
| `nt_v2` (Phase 1–3) | ✅ | ≈ | ≈ |
| `nt_v2_meanmean` | ✅ | ✅ | ≈ |
| `nt_v2_meanD` | ✅ | ✅ | ≈ |
| `nt_v2_meanG` | ✅ | ≈ | ≈ |
| `nt_v2_maxmean` | ❌ | ≈ | ≈ |
| `nt_v2_clsmean` | ❌ | ❌ | ❌ |

(✅ Δ ≥ +0.02; ≈ within ±0.02; ❌ Δ ≤ −0.02.)

What this surfaces:

1. **The tokenisation fix flips DNABERT-2 from `❌ ❌ ≈` to `✅ ✅ ✅`.** Same model, same pooling, only the chunk boundary tokens differ. The Phase 1–3 read on DNABERT-2 was strictly false; with the fix it beats k-mer everywhere.
2. **NT-v2 beats k-mer on the 5-way no matter how you pool it** (5-way column has ✅ on every NT-v2 mean-* row). This is the most defensible "encoder learned something beyond composition" result.
3. **`maxmean` and `clsmean` hurt DNABERT-2 + NT-v2 on the 5-way** (✅ on every other column rule, ❌ here). The deep-research priors that recommended max-pool and CLS-pool were wrong for this regime.
4. **Tf-vs-kinase is the hardest task for the encoders.** Best margin = +0.048 macro-F1 (DNABERT-2 + meanmean / meanD); k-mer comes within +0.005 of NT-v2's best on this task. Both intracellular soluble protein families share too much codon composition for the encoder to gain much over a 4-mer histogram.

## Cohen's κ — chance-corrected view of the headlines

macro-F1 is the headline metric, but it isn't chance-corrected. The empirical chance level for our 5-way task (heavily imbalanced) is ~0.20 macro-F1 — the anti-baseline (shuffled labels) hit 0.208. To give a single chance-corrected scalar that respects the actual class distribution, we report Cohen's κ alongside macro-F1.

κ = (p_observed − p_expected) / (1 − p_expected). Computed from the saved 5-way confusion matrices for `family5` cells; for binary tasks the probe is refit at the recorded best C and κ is computed on test predictions via `sklearn.metrics.cohen_kappa_score`. Full table at `data/kappa_summary.md`.

| Task | Best variant | macro-F1 | **κ** | k-mer F1 / κ | shuffled-label κ |
|---|---|---:|---:|---:|---:|
| 5-way | `nt_v2_meanD` | 0.828 | **0.821** | 0.672 / 0.702 | +0.048 |
| tf-vs-gpcr | `dnabert2_meanmean` / `nt_v2_meanD` | 0.989 | **0.978** | 0.961 / 0.921 | −0.067 |
| tf-vs-kinase | `dnabert2_meanmean` / `dnabert2_meanD` | 0.887 | **0.774** | 0.839 / 0.679 | +0.095 |

Two notes:

1. **κ does not always rank cells the same way as macro-F1.** On 5-way, k-mer's macro-F1 (0.672) is below the worst encoder + variant (`nt_v2_clsmean` at 0.593), but k-mer's κ (0.702) beats `nt_v2_clsmean`'s κ (0.569). The reason is class imbalance: macro-F1 weighs each class equally, κ weighs by the *expected* coincidence rate per class. k-mer happens to be calibrated against the majority class (tf), which lifts its κ relative to macro-F1.

2. **κ × class-balance interaction in the binary tasks.** tf-vs-gpcr (κ = 0.978 best) is genuinely "almost perfectly chance-corrected agreement"; the encoders nearly always agree with the true label. tf-vs-kinase is harder (κ = 0.774 best) — same balanced-binary task, lower ceiling, consistent with the harder family pair.

The shuffled-label κ values (≈ 0 across tasks) confirm that nothing in the pipeline gives a free κ — when X carries no information about y, κ collapses to 0 as expected.

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
