# Cross-Modal Alignment of DNA Encoders into Gene Function — Write-up

**Audience:** the team's slide deck reviewers; in-class presentation.
**Source of truth for numbers:** `findings.md`. This file restages the same numbers as a presentation narrative.

## Thesis (one line)

A pre-trained DNA language model can be linearly aligned to gene-function semantics — but whether you *see* that depends entirely on (a) how you tokenise the input and (b) what target you train the probe to predict.

## Headline numbers (lead with these)

| | Best result | macro-F1 | Cohen's κ | 4-mer baseline | Margin (F1) |
|---|---|---:|---:|---:|---:|
| **Family classification (5-way)** | NT-v2 + `meanD` | **0.828** | **0.821** | 0.672 (κ=0.702) | **+0.156** |
| **Binary classification (tf-vs-gpcr)** | DNABERT-2 / NT-v2 (best variant) | **0.989** | **0.978** | 0.961 (κ=0.921) | +0.028 |
| **Binary classification (tf-vs-kinase)** | DNABERT-2 + `meanmean`/`meanD` | **0.887** | **0.774** | 0.839 (κ=0.679) | +0.048 |
| **Regression (predict GenePT 1536-d)** | DNABERT-2 + `meanG` | R² = **0.210** | — | R² = 0.174 | +0.036 R² |

Cohen's κ is chance-corrected (0 = chance, 1 = perfect). R² is chance-corrected by construction (R² = 0 ≡ predict-the-mean). Anti-baseline (shuffled labels) lands at κ ≈ 0.05 / −0.07 / +0.10 across the three classification tasks — pipeline is honest. Full table: `data/kappa_summary.md`.

These numbers supersede the original Phase 3 "informative negative" framing. With the right target and the right tokenisation, both encoders carry meaningful gene-function signal that a 4-mer composition baseline cannot reach.

## Setup (one slide)

- 3244 human protein-coding genes, 5 functional families (TF 1743, GPCR 591, kinase 558, ion channel 198, immune receptor 154).
- Two pretrained DNA encoders: **DNABERT-2** (117M, BPE, 768-d) and **NT-v2 100M multi-species** (100M, 6-mer non-overlapping, 512-d). Both run on the canonical CDS only.
- Frozen 70/15/15 split stratified by family, seed 42.
- One linear probe per `(encoder, target)` pair. `Ridge` for regression, `LogisticRegression` for classification. C/α swept on val, refit on train+val, evaluated once on test.

## The story arc (4 slides)

### Slide 1 — Phase 3: a clean informative negative (and what it missed)

Original hypothesis: a linear map from DNA encoder space into the GenePT 1536-d text-embedding space recovers gene function. Result: both encoders barely cleared the 4-mer baseline (Δ R² ≤ +0.018). Two converging encoders + an MLP probe sweep + a working anti-baseline together pointed at the *representation* as the ceiling, not the probe.

Read at the time: "DNA encoders' extra capacity over a 4-mer histogram is compositional, not functional."

That read had two unexamined assumptions: GenePT is the right target, and the existing tokenisation is the right tokenisation. Phase 4 broke both.

### Slide 2 — Phase 4a: pivot to classification

Two changes at once:
1. **Target.** GenePT 1536-d → categorical family label (5-way + two binary tasks).
2. **Metric.** R² macro → macro-F1.

Same cached embeddings. Same probe discipline. Three tasks, four feature sources + shuffled-label control = 15 cells.

Result: NT-v2 5-way macro-F1 = **0.803** vs k-mer 0.672 (**+0.131**). DNABERT-2 still tied or lost to k-mer on every task. The pivot helped only the stronger encoder.

| Task | NT-v2 − 4-mer | DNABERT-2 − 4-mer |
|---|---:|---:|
| 5-way | **+0.131** | −0.023 |
| tf-vs-gpcr | +0.017 | −0.023 |
| tf-vs-kinase | +0.005 | −0.006 |

What this told us at the time: the regression target was masking signal; classification surfaces it. But the encoder gap (NT-v2 ≫ DNABERT-2) looked architectural — bigger pretraining corpus, rotary, GLU.

### Slide 3 — Phase 4b: the tokenisation surprise

Phase 4b was supposed to be a pooling ablation (mean→D, mean→G, max→mean, CLS→mean against the existing mean→mean baseline). It surfaced something we weren't looking for.

The Phase 1–3 pipeline tokenised CDS with `add_special_tokens=False` — no `[CLS]` / `[SEP]` wrapping per chunk. The Phase 4b re-extraction (which had to wrap chunks with special tokens to define a "CLS" pool) accidentally fixed this. Result:

| Encoder | Phase 4a `mean→mean` (no specials) | Phase 4b `meanmean` (with specials) | Δ |
|---|---:|---:|---:|
| **DNABERT-2 5-way** | 0.649 | 0.722 | **+0.073** |
| DNABERT-2 tf-vs-gpcr | 0.938 | 0.989 | **+0.051** |
| DNABERT-2 tf-vs-kinase | 0.833 | 0.887 | **+0.054** |
| NT-v2 5-way | 0.803 | 0.800 | −0.003 |
| NT-v2 tf-vs-gpcr | 0.978 | 0.983 | +0.006 |
| NT-v2 tf-vs-kinase | 0.844 | 0.845 | +0.000 |

**The encoder gap from Phase 4a was largely a tokenisation artefact.** With proper tokenisation, DNABERT-2 either ties NT-v2 (5-way, tf-vs-gpcr) or beats it (tf-vs-kinase). NT-v2 is insensitive to special tokens — likely because its ESM-style architecture doesn't pretrain `<cls>` as a sequence summary, so adding/removing boundary tokens barely shifts its representation.

This is the single most important finding of the project: a 50-line tokenisation choice silently capped one encoder's apparent ability.

### Slide 4 — Phase 4b: pooling sweep results

Five variants per encoder against the recomputed `meanmean` baseline. Three takeaways:

1. **`meanD` (concat[first, last, mean] across chunks) is the only variant that reliably helps.** Best on the headline 5-way for NT-v2: 0.800 → **0.828** (+0.028). Concatenating positional anchors exposes terminal asymmetry — N-terminal signal peptides and C-terminal motifs do carry family-discriminative signal that bare cross-chunk mean smears out.

2. **`meanG` (D + max-across-chunks) ≈ `meanD`.** The "one dominant chunk" hypothesis isn't supported. Not worth the 4× dim cost.

3. **`maxmean` and `clsmean` consistently HURT.** Two failures of the prior literature:
   - `maxmean` was supposed to surface sparse motifs (per-dim max within chunk). It does — but the cross-chunk mean then averages them right back into noise.
   - `clsmean` is catastrophic for NT-v2 (5-way drops to 0.59) and merely bad for DNABERT-2. Both encoders are masked-LM pretrained without next-sentence prediction; their CLS positions weren't trained as sequence summaries, so pooling on them throws away signal.

### Slide 5 — Phase 5a: the regression view, re-run

Same 10 new pooling-variant parquets, but back to the original Phase 3 regression task (Ridge → GenePT 1536-d).

| Variant | Test R² | Δ R² vs Phase 3 same-encoder |
|---|---:|---:|
| **`dnabert2_meanG`** | **0.2104** | **+0.029** |
| `dnabert2_meanD` | 0.2100 | +0.029 |
| `dnabert2_meanmean` | 0.2029 | +0.022 |
| `nt_v2_meanmean` | 0.1932 | +0.000 |
| `nt_v2_meanG` | 0.1902 | −0.003 |
| `nt_v2_meanD` | 0.1882 | −0.005 |
| `nt_v2_clsmean` | 0.1172 | −0.076 |

Three reads:

- **DNABERT-2's Phase 3 informative-negative was an undercount.** Δ vs 4-mer goes from +0.007 (Phase 3) to **+0.036** (now). With proper tokenisation DNABERT-2 decisively beats the 4-mer baseline in regression too — not just classification.
- **NT-v2 regression is unchanged.** `meanmean` R² = 0.193 = Phase 3's 0.193. Confirms NT-v2's regression ceiling is real, not a tokenisation artefact.
- **`meanD`'s classification gain for NT-v2 does NOT translate to regression.** NT-v2 + `meanD` is +0.025 macro-F1 in classification but −0.005 R² in regression. Translation: terminal asymmetry helps the model decide *which family* a gene belongs to, but doesn't recover *what the gene does* in the full GenePT description.

## Three things to say in the talk

If we get only three sentences, these are them:

1. **NT-v2 carries family-discriminative signal beyond raw nucleotide composition** — 5-way macro-F1 = 0.828 (Cohen's κ = 0.821) vs 4-mer baseline 0.672 (κ = 0.702) and shuffled-label chance κ ≈ 0.05. Pretrained DNA encoders are doing real work, when you ask the right question.

2. **DNABERT-2 was being silently degraded by missing special tokens.** A boundary-token-tokenisation fix lifted its classification by +0.05–0.07 macro-F1 across all three tasks and lifted its regression R² by +0.022 (from "ties 4-mer" to "beats 4-mer by 5×"). The encoder-architecture gap we initially attributed to NT-v2's bigger pretraining corpus was largely this.

3. **Pooling matters less than people think, except for one variant.** Across 30 pooling-variant cells, only `meanD` (concat first + last + mean across chunks) reliably helped — and only modestly. Two priors from the literature (`max-pool tokens` and `CLS pool`) actually *hurt* both encoders in this setting. The right takeaway: **tokenisation > pooling > architecture** as levers in this regime.

## Caveats (slide if asked)

1. **One corpus, five families, one organism.** All numbers are on a 3244-gene human dataset selected by HGNC family regex. The classification task structure favours family-level signal over fine-grained function.
2. **CDS only.** Promoter and UTR sequences may carry more function-discriminating signal than the codon-bias-dominated CDS; not tested here.
3. **No third encoder.** Two-encoder convergence weakens slightly because we now know one was a tokenisation bug. A HyenaDNA / Caduceus / GENA-LM run would re-establish whether the "encoders carry function signal" claim is architecture-general.
4. **Regression ceiling is still modest.** Best R² = 0.21 — most of GenePT's 1536-d target is noise no DNA encoder of this size will recover. Classification is the right framing; regression is the diagnostic.

## What stays in vs what gets cut

For a 5–7 minute slot:

| Slide | Keep / cut | Why |
|---|---|---|
| Thesis + headlines | KEEP | Anchor the talk |
| Setup (corpus, encoders, probes) | KEEP | One slide, fast |
| Phase 3 informative negative | KEEP, brief | Sets up the pivot |
| Phase 4a classification | KEEP | The first pivot win |
| Phase 4b tokenisation surprise | KEEP — this is the talk | Single biggest finding |
| Phase 4b pooling sweep | KEEP | Validates `meanD`, rules out two bad priors |
| Phase 5a regression re-run | KEEP | Closes the loop with Phase 3 |
| Three headline takeaways | KEEP | The "if you remember nothing else" slide |
| Caveats | optional | Have ready for Q&A |
| Confusion matrices | optional | Per-class accuracy nuance |

For a 3-minute pitch: drop Phase 3 setup and Phase 4b pooling sweep; lead with Phase 4a result + Phase 4b tokenisation surprise + Phase 5a regression re-run.

## Provenance

All numbers in this write-up trace to `data/metrics.json` (per-cell entries) and `data/confusion_5way_*.json` (per-encoder, per-variant 5-way confusion matrices). Each metrics entry has a `run_id` of form `{model}_{encoder}_{task_or_dataset}_{utc_timestamp}` so any number in the tables above can be located in one grep.
