# Cross-Modal Alignment of DNA Encoders into Gene Function

**8-minute in-class talk · 9 content slides + 1 references slide.** Audience has not seen prior decks; this is a complete intro.

Source of numbers: `findings.md`, `data/full_table.md`, `data/kappa_summary.md`. ~50 seconds per content slide; references slide is just visible while wrapping up.

---

## Slide 1 — Title + thesis

**Cross-Modal Alignment of DNA Encoders into Gene Function**
Austin · Andrew · Hayden

> A *frozen* pretrained DNA encoder plus a *linear* probe is enough to read out **gene family identity** well beyond nucleotide composition (Δκ +0.12 over a 4-mer baseline), but only **partially aligns** to free-text descriptions of gene function (ΔR² +0.04). Functional category is linearly readable from the encoder's geometry; full functional description is not — the text target carries substantial noise that an encoder of this size cannot fully recover.

We probe the same encoders into two output spaces:
1. **DNA → gene family** — 5-way classification into functional families.
2. **DNA → text** — linear projection into a 1536-d gene-summary text-embedding space.

> **[Slide diagram — central visual on this slide]**
> ```
>                              ┌──→  5-way gene family
> DNA sequence  ──→ frozen ──→ embedding ──→ linear probe ──┤
>                  encoder                                  └──→  1536-d text embedding
> ```
> Same frozen encoder, same linear probe machinery, two output spaces. Every result slide is one of the two arrows.

---

## Slide 2 — Why this matters (motivation + related work)

**The biological question.** A gene's nucleotide sequence is the source of its function — but the mapping is opaque. Pretrained DNA language models (DNABERT-2, NT-v2) are trained on raw sequence via masked-language modelling, with no functional labels. **Do they already encode gene function in their geometry?**

**Two ways we test this.**
- **DNA → gene family.** Predict one of 5 functional categories (TF / GPCR / kinase / ion channel / immune receptor). Discrete, interpretable failure mode, clean ground truth.
- **DNA → text.** Linearly project DNA embeddings into a frozen gene-summary text-embedding space (GenePT, 1536-d). If a linear map suffices, full functional semantics are present in the encoder.

**Why linear, why frozen.** A linear probe is the cleanest readout of an encoder's geometry — no fine-tuning, no compute-confounded comparison, no overfitting risk on small data. If the linear probe wins, the encoder already had the structure.

**Related work goes the other way (expensive).**
- **Omni-DNA** (Fan *et al.* 2025) — autoregressive DNA transformer pretrained on **300B nucleotides**, English vocabulary inserted into the tokenizer, full supervised fine-tuning on DNA → function. *Vocabulary-level integration.*
- **SeqCLIP** (Li *et al.* 2026) — keeps a DNA encoder + an LLM, joins them with a structural adapter, fine-tunes on massive DNA-text paired data. *Semantic embedding-level alignment.*

Both prove DNA-text alignment is achievable. Both require **A100-cluster-scale compute and weeks of training**.

**Our research question.** Can we read out the encoder's functional knowledge **cheaply** — frozen encoder + linear probe, no adapter, no paired-corpus training — for either output space:
- distinguishing **gene families** (5-way classification)?
- recovering **free-text descriptions** of gene function (regression into the GenePT 1536-d space)?

If both work, the latent geometry already contains function in a linearly-readable form (we don't directly compare to Omni-DNA / SeqCLIP, but we'd establish that linear-probe-level signal exists). If only the discrete one works, the encoder separates *kinds* of genes but doesn't recover their *full* functional descriptions — a partial answer with a clean upper bound.

**Why two encoders.** We run the whole pipeline on **DNABERT-2** *and* **NT-v2**. Both are masked-LM pretrained on raw DNA but differ on tokenizer, corpus, and context (next slide). Comparing them tests whether any answer generalises across encoder choice — not just architecture-of-the-week.

---

## Slide 3 — Setup: data, encoders, families

**Data sources (assembled into one corpus).**
- **HGNC** — canonical gene symbols, family-group annotations (used to define the 5 functional families).
- **Ensembl** — canonical CDS nucleotide sequences for each gene.
- **GenePT** (Chen & Zou 2024) — pre-computed OpenAI text embeddings of NCBI gene summaries, 1536-d. The text-side target for DNA→text regression.

After intersecting and de-duplicating: **3244 human protein-coding genes** with a canonical CDS, a family label, and a GenePT embedding.

**Functional families (5 classes, imbalanced):**

| Family | Count | Selection rule (HGNC `gene_group`) |
|---|---:|---|
| TF (transcription factor) | 1743 | zinc finger / homeobox / bHLH / bZIP / forkhead / nuclear receptor / SOX / ETS / "transcription factor" |
| GPCR (G-protein-coupled receptor) | 591 | "G protein-coupled receptor" + adrenoceptor / 5-HT / dopamine / opioid / chemokine / olfactory |
| Kinase | 558 | `\bkinase` (excluding inhibitors / regulators / pseudokinases) |
| Ion channel | 198 | `\bchannel` (excluding regulators / auxiliary subunits) |
| Immune receptor | 154 | TLR / IL receptor / Fc receptor / NLR / TCR / BCR / C-type lectin |

**Pretrained DNA encoders compared:**

| Axis | **DNABERT-2** (117M) | **NT-v2 multi-species** (100M) |
|---|---|---|
| Tokenizer | BPE (~4k vocab, variable-length) | Fixed 6-mer non-overlapping |
| Hidden dim | 768 | 512 |
| Context window | 512 tokens (~2–3 kb) → CDS chunked | ~6000 tokens (~36 kb) → most CDSs fit whole |
| Pretraining corpus | 135 genomes | 850 species |
| CLS token | Used with `[CLS]`/`[SEP]` boundary tokens during pretraining (BERT-style) | ESM-style tokenisation — boundary tokens not used the same way during pretraining |

Same masked-LM training objective; everything else differs. Both frozen — no fine-tuning anywhere in this work.

---

## Slide 4 — Method

**Split.** Stratified 70 / 15 / 15 on family, seed 42, frozen in `data/splits.json` → 2270 / 487 / 487.

**Embeddings (X).** A canonical CDS is too long for either encoder in a single pass, so we split it into overlapping chunks, encode each, and combine. We test 4 ways to produce one dense vector per CDS:

| Variant | What it is | Hypothesis it tests |
|---|---|---|
| `regular_mean` | Averaged token embedding over the whole CDS. No `[CLS]`/`[SEP]` boundary tokens. | Distributed-signal-everywhere — the default baseline. |
| `special_mean` | Same average, but each chunk is wrapped with `[CLS]`/`[SEP]` before encoding. | Isolates the effect of boundary-token wrapping (only difference vs `regular_mean`). |
| `meanD` | Concat of first-chunk mean + last-chunk mean + mean across all chunks. | Terminal asymmetry — N-/C-terminal motifs (signal peptides, DNA-binding domains) that mean-across-chunks smears out. |
| `clsmean` | `[CLS]` position of each chunk, then average across chunks. | Whether `[CLS]` is itself a useful sequence summary in a frozen MLM-pretrained encoder. |

`regular_mean`, `special_mean`, and `clsmean` come from the DNA foundation-model benchmark¹; `meanD` is our adaptation from the protein-pooling literature². Main-text numbers use the best variant per encoder × task cell; the variant comparison itself is a Slide 7 result.

> *¹ Tang et al. 2025; DNABERT-2 / GUE+ setup. ² Light Attention (Stärk et al.); BoM-Pooling. Full citations on Slide 10.*

**Targets (Y) and probes.**
- *DNA → text:* GenePT 1536-d. **Ridge regression**, per-output. Metric: macro R².
- *DNA → family:* one-of-5 categorical label. **Logistic regression** (multinomial, L2). Metric: macro-F1 + Cohen's κ (chance-corrected).

**Probe protocol.** Sweep regularisation (α for Ridge, C for logistic) on val; refit on train+val at the chosen value; evaluate **once** on the held-out test set.

**Baselines.**
- **4-mer composition** — 256-d L1-normalised k-mer histogram, same probe recipe. Encoder-independent floor.
- **Length-only** — log(CDS length), 1 feature. Tests trivial size cues.
- **Shuffled-label anti-baseline** — labels permuted in train+val only, real test labels. Tests pipeline honesty.

---

## Slide 5 — Result A: family identity (the strong result)

**5-way classification, macro-F1 + Cohen's κ on test:**

| Feature source | macro-F1 | Cohen's κ | Δκ vs 4-mer |
|---|---:|---:|---:|
| Shuffled-label (chance) | 0.208 | +0.048 | — |
| Length-only | 0.138 | −0.011 | — |
| **4-mer composition** | 0.672 | 0.702 | 0.000 |
| DNABERT-2 (best variant) | 0.738 | 0.723 | +0.020 |
| **NT-v2 (best variant)** | **0.828** | **0.821** | **+0.119** |

**Metric definitions** (slide-side inset):

> **Macro-F1.** Per-class F1 averaged across classes (no class-size weighting):
> $$\mathrm{F1}_c = \frac{2 \, P_c \, R_c}{P_c + R_c}, \qquad \text{macro-F1} = \frac{1}{C} \sum_{c=1}^{C} \mathrm{F1}_c$$
> Treats every family equally → small classes (immune, ion) count as much as TF.
>
> **Cohen's κ.** Chance-corrected agreement:
> $$\kappa = \frac{p_o - p_e}{1 - p_e}$$
> $p_o$ = observed accuracy. $p_e$ = expected accuracy under random labelling, from the row/column marginals: $p_e = \sum_c \frac{n_{c,\cdot} \cdot n_{\cdot,c}}{N^2}$. κ = 0 ≡ chance, κ = 1 ≡ perfect, κ < 0 ≡ worse than chance. Robust to class imbalance (unlike raw accuracy).

**Reading.**
- Both encoders beat 4-mer; **NT-v2 decisively** (Δκ +0.12), DNABERT-2 marginally (Δκ +0.02).
- Anti-baseline κ ≈ +0.05 — pipeline is honest. The headroom is real.
- NT-v2 leads on this task. *Why* — broader pretraining corpus, longer context, different tokenizer — would need an ablation we didn't run; we report the gap empirically and don't attribute it to a specific axis.

**Headline figure — test confusion matrix** (`viz/figures/confusion_5way_nt_v2_meanD.png`, generated by `viz/confusion_meanD.py`). Row-normalised 5×5 heatmap, NT-v2 + meanD on the test set. Diagonal: TF 0.95 · GPCR 0.94 · Kinase 0.80 · Ion channel 0.60 · Immune receptor 0.70. Off-diagonal mass concentrates where biology predicts — Kinase ↔ TF (the largest non-diagonal cell, n=17) and Ion ↔ TF (n=6) — consistent with TF being the largest, most diverse class. Every cell on the slide ties to one cell in this figure.

> **Visual-design note.** Replace the headline-numbers *table* with a **horizontal bar chart of Cohen's κ per feature source**. Five bars (shuffled · length · 4-mer · DNABERT-2 best · NT-v2 best). Add a vertical reference line at κ = 0 (chance) and at the 4-mer value. This makes the Δκ +0.12 gap visible at a glance instead of buried in a table. Keep the confusion matrix as-is.

---

## Slide 6 — Result B: DNA → text regression (the honest result)

**Ridge → GenePT 1536-d, test set:**

| Feature source | R² macro | Mean cosine | Δ R² vs 4-mer |
|---|---:|---:|---:|
| Shuffled-Y (chance) | −0.002 | 0.913 | — |
| **4-mer composition** | 0.174 | 0.931 | 0.000 |
| **DNABERT-2 (best variant — `meanG`)** | **0.210** | **0.934** | **+0.036** |
| NT-v2 (best variant — `meanmean`) | 0.193 | 0.932 | +0.019 |

**Reading.**
- Both encoders beat 4-mer, but the margin is small. R² 0.21 is modest — much of the 1536-d GenePT target appears to be noise this scale of encoder cannot recover. The target itself is derived from English summaries via a generic text-embedding model, so part of the 1536-d geometry encodes *language* structure rather than biology.
- **Cosine similarity has a very high floor on this target.** Even the shuffled-Y anti-baseline scores 0.913, because GenePT gene-summary embeddings are tightly clustered in 1536-d. R² is the chance-corrected metric that actually shows the gap; cosine is reported for completeness.
- The cross-modal alignment claim is **partially supported, not decisively** — there's some signal beyond k-mer composition, but not much.

> **Visual-design note.** Replace the table with a **two-panel bar chart**: (a) R² per feature source (shuffled · 4-mer · NT-v2 best · DNABERT-2 best) with a horizontal reference line at R² = 0 and at the 4-mer value; (b) mean cosine per source on the same x-axis. The two panels together visualise the slide's pedagogical point — R² shows the gap (0 → 0.21), cosine doesn't (0.91 → 0.93). Stack the panels vertically with a shared x-axis.

---

## Slide 7 — How pooling shapes what we read out

Each of the 4 pooling variants (Slide 4) gives a different reading of the same frozen encoder. Δ values are vs `regular_mean` within each encoder.

**DNABERT-2:**

| Variant | macro-F1 | ΔF1 | κ | Δκ | R² | ΔR² |
|---|---:|---:|---:|---:|---:|---:|
| `regular_mean` | 0.649 | — | 0.636 | — | 0.181 | — |
| `special_mean` | 0.722 | **+0.073** | 0.709 | **+0.073** | 0.203 | +0.022 |
| `meanD` | **0.738** | **+0.089** | **0.723** | **+0.087** | **0.210** | **+0.029** |
| `clsmean` | 0.655 | +0.006 | 0.662 | +0.026 | 0.191 | +0.010 |

**NT-v2:**

| Variant | macro-F1 | ΔF1 | κ | Δκ | R² | ΔR² |
|---|---:|---:|---:|---:|---:|---:|
| `regular_mean` | 0.803 | — | 0.798 | — | 0.193 | — |
| `special_mean` | 0.800 | −0.003 | 0.798 | 0.000 | 0.193 | +0.000 |
| `meanD` | **0.828** | **+0.025** | **0.821** | **+0.023** | 0.188 | −0.005 |
| `clsmean` | 0.593 | **−0.210** | 0.569 | **−0.229** | 0.117 | **−0.076** |

**Reading the tables.**
- **Boundary tokens help DNABERT-2, not NT-v2.** `regular_mean` → `special_mean`: ΔF1 +0.07 for DNABERT-2 vs −0.003 for NT-v2. DNABERT-2 includes `[CLS]`/`[SEP]` as part of its standard BERT-style tokenisation; NT-v2 is ESM-style and uses boundary tokens differently. We observe the asymmetry empirically — the precise mechanism (e.g. whether DNABERT-2's `[CLS]` learns sequence-summary behaviour during MLM pretraining) is plausible but not directly tested here.
- **Terminal positions help both** (modestly). `special_mean` → `meanD`: +0.016 / +0.025 macro-F1.
- **`[CLS]`-only pooling fails — catastrophically for NT-v2** (−0.21 F1, −0.08 R²). Consistent with NT-v2 not having a useful sequence-summary readout at the `[CLS]` position.

**Two-panel figure** (`viz/figures/umap_dnabert2_tokenisation_compare.png`): DNABERT-2 UMAP under `regular_mean` vs `special_mean`.

**Takeaway.** Pooling choice significantly changes how each encoder is read out, and **the two encoders respond very differently to the same pooling choice**. The differences are likely a consequence of how each was pretrained, but we don't isolate which aspect.

> **Visual-design note.** Replace both tables with a **grouped bar chart** of Δ macro-F1 vs `regular_mean` (the y=0 reference). x-axis: 4 variants (`regular_mean`=0, `special_mean`, `meanD`, `clsmean`). Two bars per variant (DNABERT-2 in one colour, NT-v2 in another). The visual story is then immediate: DNABERT-2's `special_mean` bar stands tall (+0.07), NT-v2's is flat; NT-v2's `clsmean` bar plunges (−0.21), DNABERT-2's is small. Add a second small inset bar chart for ΔR² showing the same pattern. Reserve the full numerical tables for the backup / appendix.

---

## Slide 8 — Cross-modal demo (DNA → family + retrieved description)

**Design.** Two probes on the **same DNA input**, both trained on train+val of the same frozen 70/15/15 split:

| Probe | Pipeline | Output |
|---|---|---|
| **Family probe** (Slide 5) | NT-v2 + `meanD` → logistic (C=1.0) | predicted gene family + class probabilities |
| **Text probe** (Slide 6) | DNABERT-2 + `meanG` → Ridge (α=10.0) | predicted GenePT 1536-d vector → top-3 nearest train+val gene summaries by cosine in text space |

**Sample.** 4 test genes picked by NCBI summary length — only a literature-coverage proxy for which genes to demo. Summaries are *never* a model input.

**Results.** Family predictions 4/4 correct; the *retrieved summaries* are the more interesting result.

| Gene | Cohort | Summary len | Family probe (prob) | Retrieved-summary families |
|---|---|---:|---|---|
| **JAK2** | well-characterised | 1636 | kinase (0.98) ✅ | 3 × kinase |
| **TRAF6** | well-characterised | 1450 | tf (0.76) ✅ | 3 × kinase ⚠️ |
| **ZNF839** | poorly characterised | 66 | tf (0.94) ✅ | 2 × kinase + 1 × tf |
| **ZNHIT2** | poorly characterised | 66 | tf (0.90) ✅ | 3 × tf |

**Two stories worth telling:**

**ZNHIT2 (the success).** Its entire NCBI summary: *"Predicted to enable metal ion binding activity"* (66 chars, useless). From DNA alone, the text probe retrieves three rich zinc-finger / TF descriptions:
- `ZFX` — *"krueppel C2H2-type zinc-finger protein family…"*
- `ZFHX3` — *"transcription factor with multiple homeodomains and zinc finger motifs…"*
- `NKX1-1` — *"transcription factor… NKX family of homeodomain-containing proteins…"*

The model retrieves a **better functional description than NCBI has on file**.

**TRAF6 (the productive disagreement).** Family probe says `tf` (correct), but the text probe retrieves *three kinases* (TNK1, MAP3K12, MAP3K8). Biologically: TRAF6 is a TNF-receptor-associated factor that **operates inside MAP-kinase / JNK signalling pathways** — a major adapter protein in kinase cascades.

The two probes are doing different things:
- **Family probe** — discrete classification. *"What kind of protein is this?"* → tf.
- **Text probe** — semantic retrieval in continuous text-embedding space. *"What known proteins is this functionally most similar to?"* → kinase-pathway proteins.

Retrieving a kinase summary for a major kinase-pathway adapter is **not an error** — it's a successful mapping of a protein to its functional pathway. Two probes, one DNA, two valid views.

> **Anticipated Q&A.** *"Isn't the text probe technically wrong about TRAF6 being a kinase?"*
> The text probe isn't doing discrete classification — it's doing semantic retrieval. Retrieving a kinase summary for a major kinase-pathway adapter protein isn't an error; it's a successful mapping of a protein to its **functional pathway**. The GenePT target encodes function as written in NCBI summaries, and TRAF6's function lives inside the kinase cascade.

> **Reproduce:** `uv run python demo/cross_modal.py` — full per-gene retrieved summaries in `demo/output.md`.

---

## Slide 9 — Conclusions

**Summary — what we found.**

1. **Family identity reads out for free.** Frozen encoder + linear probe → Δκ +0.12 over a 4-mer baseline (NT-v2 leads; we don't isolate which axis of the recipe is responsible).
2. **Cross-modal alignment to text is partial.** Best R² 0.21 vs 4-mer 0.17 — some signal beyond nucleotide composition, but bounded by encoder size and the noisiness of the GenePT text-embedding target (which itself encodes language structure, not just biology).
3. **Pooling is a result, not a knob.** The same pooling change can lift one encoder by +0.07 F1 while moving the other not at all, or collapse one by −0.21 while leaving the other intact. Pooling significantly changes how each encoder is read out, and the encoder-specific responses are likely a consequence of how each was pretrained — but we don't isolate which aspect.

**Validity — how we know these claims hold.**
- **Anti-baseline:** shuffled-label κ ≈ +0.05 across classification, R² ≈ 0 for regression — pipeline is honest.
- **Compositional baseline:** every encoder claim is Δ vs 4-mer, not absolute.
- **Probe protocol:** regularisation swept on val, refit on train+val, evaluated *once* on test — no test contamination.

**Limitations.**
- One organism (human), one corpus (HGNC ∩ GenePT, ~3.2k genes), 5 families — class-imbalanced.
- CDS only — no UTRs, no promoter regions where regulatory signal is dense.
- Two encoders. Findings about pretraining recipes generalise weakly without a third encoder.
- GenePT target is itself derived from English summaries via OpenAI embeddings — its 1536-d geometry is noisy and partially captures language structure rather than biology.

**Future work.**
- **Embedding representation as the primary lever.** Our pooling sweep already shows that *how* you turn an encoder into a single vector is responsible for ±0.21 F1. The natural extension: this becomes the substrate for **interpretable gene-LLMs** — choosing the embedding representation that best preserves the linearly-readable structure (so a downstream LLM can use it without re-learning what the encoder already knows). Our 4-variant ablation is a starting menu for "best-representation-for-interpretability."
- **Controlled recipe study:** vary one axis at a time (tokenizer / corpus / context) holding the other two fixed — isolates which knob matters for the encoder gap we observed.
- **Third encoder** (HyenaDNA, Caduceus, GENA-LM) to test architecture-generality of the "frozen geometry encodes function" claim.
- **Non-CDS sequences** (UTR, promoter) — likely more function-discriminating signal.
- **Larger / longer-context encoders** to test whether the cross-modal regression ceiling moves.

> **Visual-design note.** Keep this slide text-heavy on purpose — it's the conclusions slide and the audience is reading along while you wrap up. Use a 4-column layout: Summary · Validity · Limitations · Future work. Single-line bullets.

---

## Slide 10 — References

Four sections (lay out as four columns or 2×2):

**MODELS / FOUNDATION MODELS**

- **Zhou et al.** 2024. *DNABERT-2: Efficient Foundation Model and Benchmark for Multi-Species Genome.* `zhihan1996/DNABERT-2-117M`. *Source encoder.*
- **Dalla-Torre et al.** 2024 · InstaDeepAI. *The Nucleotide Transformer: building and evaluating robust foundation models for human genomics.* `InstaDeepAI/nucleotide-transformer-v2-100m-multi-species`. *Second source encoder.*
- **Chen & Zou** 2024. *GenePT: a simple but effective foundation model for genes and cells built from ChatGPT.* *1536-d text targets from NCBI gene summaries.*
- **Fan et al.** 2025. *Omni-DNA.* *Related work — vocabulary-level integration; cited for compute contrast.*
- **Li et al.** 2026. *Alignment or Integration?* (SeqCLIP). *Related work — embedding-level alignment; cited for compute contrast.*

**DATA SOURCES**

- **HGNC** — HUGO Gene Nomenclature Committee. `genenames.org`. *Gene symbol → group → Ensembl ID join; defines the 5 functional families.*
- **Ensembl REST API.** `rest.ensembl.org`. *Canonical CDS sequences per gene.*
- **NCBI gene summaries.** *Source text for the GenePT 1536-d targets.*

**POOLING LITERATURE** (motivation for Slide 4)

- **Tang et al.** 2025. *Benchmarking DNA Foundation Models for Genomic and Genetic Tasks.* *5 models × 52 binary-classification datasets; mean-token pooling > summary > max for frozen DNA embeddings — supports `regular_mean` / `special_mean` / `clsmean`.* *(Slide 4, citation ¹.)*
- **DNABERT-2 / GUE+ long-sequence setup** (Zhou et al. 2024). *Hierarchical averaging used as a workaround, not a validated principle — motivates the comparison itself.* *(Citation ².)*
- **Stärk et al.** *Light Attention Predicts Protein Location from the Language of Life.* *Protein-pooling locality-aware aggregation — inspires `meanD`.* *(Citation ³.)*
- **Sun et al.** *BoM-Pooling: Locality-aware pooling enhances PLM performance.* *Locality-aware aggregation outperforms mean / CLS / max for sparse-signal tasks — supports the motivation for terminal-asymmetry pooling.* *(Citation ⁴.)*

**LIBRARIES**

- **Pedregosa et al.** 2011 · JMLR. *scikit-learn: Machine Learning in Python.* *Ridge, LogisticRegression, train/val/test splits, k-fold sweeps.*
- **Wolf et al.** 2020 · EMNLP. *Transformers: State-of-the-Art Natural Language Processing.* *Loader for both DNA encoders.*
- **McInnes et al.** 2018. *UMAP: Uniform Manifold Approximation and Projection.* *Used for the embedding visualisations in backup.*

> **Visual-design note.** Render exactly like the prior project's references slide (3-column layout, bold author + year, italic role-in-project). Add Omni-DNA and SeqCLIP under MODELS to match the related-work citations on Slide 2; add the Pooling-literature column to back the Slide 4 motivation.

---

## Backup slides (have ready, don't show unless asked)

- **Binary classification (tf-vs-gpcr, tf-vs-kinase).** Encoders beat k-mer by Δ F1 +0.03 / +0.05 — small because k-mer is already at 0.96 / 0.84. Confirms encoders also help on easier sub-problems, but adds little to the headline.
- **Pooling sweep.** Four variants in the main results (`regular_mean`, `special_mean`, `meanD`, `clsmean`). We also ran two more that we cut: `maxmean` (max-pool tokens within chunk, then average across chunks) and `meanG` (`meanD` plus an extra max-across-chunks term). `meanG` ties `meanD` within ±0.002 on every task at 4× the dimension — not worth the cost. `maxmean` and `clsmean` consistently hurt.
- **Unsupervised UMAP** (`viz/figures/umap_nt_v2_meanD.png`). Only TF and GPCR form clean clusters; the three smaller families overlap in 2D. The linear probe still recovers them (κ 0.82) — UMAP optimises local density, not the linear directions the probe uses.
- **Caveats.** One corpus, one organism, CDS-only (no UTR/promoter), two encoders. A third encoder (HyenaDNA / Caduceus / GENA-LM) would re-establish the architecture-general claim.
- **Full results table.** `data/full_table.md` — every (feature source × task × metric) cell.

---

## Provenance

All numbers trace to `data/metrics.json` (per-cell entries with `run_id`) and `data/confusion_5way_*.json`. Cohen's κ values come from `data/kappa_summary.md`. Full matrix in `data/full_table.md`.
