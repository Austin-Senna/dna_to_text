# Paper revision follow-ups (deferred reader-feedback items)

Items 18–30 from the section-by-section reader review (PI, methods skeptic, cross-disciplinary postdoc) that were out of scope for the textual edit pass. Some need new computation; some need code verification; some were judged already-addressed.

Grouped by what unblocks them.

---

## A. Need new computation (high priority — would materially strengthen the paper)

### 18. Bootstrap confidence intervals on every headline number
**Source:** stats reader (raised at §3.1, §3.2, §3.4, Discussion); PI synthesis; PI noted at §3.4
**Concern:** All Tables 2, 3, and the context-ablation numbers are point estimates. With test set n=487 and minority classes n=23 (immune) / n=30 (ion), the macro-F1 differences driving the encoder ranking — and the Δκ +0.119 headline — could be within sampling noise.
**What's needed:** Stratified bootstrap (e.g., 1,000 resamples of the test split, refitting the probe each time or just resampling test predictions at the saved best-C/α). Report 95% CIs in the table cells.
**Feasibility:** High. The probes are cheap (logistic + ridge on cached embeddings). `scripts/train_logistic_probe.py` and `scripts/train_probe.py` would need a `--bootstrap N` flag that resamples the test set after refit. ~30 minutes of dev + CPU runtime per encoder cell.
**Output:** `data/family5_table.md` and `data/regression_table.md` get a `95% CI` column; abstract / Results §3.1 / §3.2 add `[CI lo, hi]` in parentheses to the headline numbers.

### 19. Per-class F1 (or full confusion matrix) in main-text Table 2
**Source:** stats reader (§3.1); PI synthesis
**Concern:** Macro-F1 0.828 averages over 5 classes — readers can't tell whether the score is driven by the TF majority (n=261 test) or whether minority classes (immune n=23, ion n=30) are also separated.
**What's needed:** Add per-class F1 columns to Table 2, or include the 5×5 confusion matrix as Figure 2.5 / inline. Per-class F1 numbers exist in `data/confusion_5way_*.json` already.
**Feasibility:** High — pure presentation work; data already cached. Decision: which to add (per-class column, normalised confusion matrix, or both)?

### 20. Per-dimension or per-PC R² distribution for GenePT regression
**Source:** PI (§3.2)
**Concern:** Macro-R² averages over 1,536 GenePT dimensions. A handful of well-predicted dimensions (e.g., text-summary length proxies) could dominate the average without telling us anything about gene function.
**What's needed:** Distribution of per-dimension R² across the 1,536 dims; ideally also per-PC R² after PCA on Y. If 90% of the explainable variance lives in 50 PCs, that changes the interpretation.
**Feasibility:** Medium. Requires modifying `scripts/train_probe.py` to dump per-dim R² (small change). PCA analysis is one notebook cell. Plot would be a histogram or rank-ordered curve (good supplementary figure).

---

## B. Need code verification (cheap to resolve, would tighten methods) — RESOLVED 2026-05-08

### 22. Was α retuned under the shuffled-Y anti-baseline? — DONE
**Source:** PI (§3.2)
**Resolution:** Yes. `scripts/train_anti_baseline.py:57-58` re-runs the full α grid `[1e-2 ... 1e3]` on shuffled-Y validation, picks the new best α, refits on train+val, evaluates once on test. Methods §Baselines and controls now documents this explicitly: "an analogous shuffled-Y anti-baseline permutes the 1,536-d GenePT vectors in train+val only and re-runs the full α sweep on the shuffled validation split (rather than reusing the α selected from the real run), so any non-trivial test R² would imply leakage in either the data split or the hyperparameter selection."

### 23. Specify Enformer trunk "global" reduction precisely — DONE
**Source:** stats reader (§Methods)
**Resolution:** All three Enformer summaries are means; only the bin range differs. `src/data_loader/enformer_encoder.py:84-86`: `trunk_global` = mean across all 896 bins; `trunk_center` = mean across the central 16 bins (default; configurable via `--center-bins`); `tracks_center` = mean across the same central 16 bins of the 5,313 human + 1,643 mouse track outputs. Methods §TSS context and Enformer comparator now lists all three by name with the exact reduction.

### 24. Family priority order for first-family-wins dedup — DONE (with surprise)
**Source:** PI (§Methods)
**Resolution:** Order is `kinase → transcription factor → ion channel → GPCR → immune receptor` per `src/data_loader/dataset_loader.py:21-66` (the `FAMILIES` list) + `:176-183` (the iteration loop with `seen_ensembl` exclusion). This is **not** the order in Table 1 (which is size-ranked, TF first). Net effect: a HIPK kinase that's also a TF is counted as kinase. Methods §Dataset now states the assignment order explicitly and notes that the Family column in Table 1 is size-ranked, not assignment-ranked.

---

## C. Need author / external verification (not code-resolvable here)

### 21. Verify the Li 2026 SeqCLIP citation date
**Source:** PI (§Introduction)
**Concern:** A 2026 reference cited in early 2026 is plausible but worth checking — could be a typo for 2025, or could be a preprint with a different bibliographic year.
**What's needed:** Cross-check against the actual paper / preprint server. Update `bibliography.bib` if needed.
**Feasibility:** 5-minute external lookup. Author's call.

---

## D. Already addressed by current text (decided no action needed)

These were raised by readers but already covered by existing wording in the current draft. Logging here so we don't re-litigate.

| # | Item | Source | Why no action |
|---|---|---|---|
| 25 | Add explicit "what does R²=0.21 on a 1,536-d target mean?" anchoring sentence in §3.2 | XDisc | The sentence "R² 0.210, only +0.036 above the CDS 4-mer baseline" plus the shuffled-Y R²≈0 row in Table 3 already anchor the magnitude implicitly. Adding more would dilute. |
| 26 | Define "anisotropy floor" formally | XDisc | The current sentence "GenePT space has a high anisotropy floor: the shuffled-Y anti-baseline still attains cosine 0.913" defines the term operationally via example — sufficient for a Bioinformatics audience. |
| 27 | Tighten Omni-DNA / SeqCLIP "throat-clearing" in Intro | XDisc | The contrast they represent (alternative path = heavy multimodal training) is drawn explicitly: "These systems show that sequence-text modeling is possible, but they require substantial additional multimodal training. ... Our question is narrower." |
| 28 | Define `k` in 4-mer in the abstract | XDisc | Standard shorthand in genomics; defined as "256-d L1-normalised 4-mer histogram" in Methods. |
| 29 | Define "GenePT" in the abstract | XDisc | Defined when first used in Introduction ("GenePT represents each gene by embedding a natural-language summary of that gene..."). Abstracts can use shorthand. |
| 30 | Biological framing of "5-way family signal" in the abstract | XDisc | The five families are named in Introduction and Methods. Adding biological framing in the abstract would lengthen it without adding information for the target reader. |

---

## Suggested order if working through this list

1. **A·22 / A·23 / A·24** (code-read documentation fixes) — 15 minutes total, knock out one Methods paragraph tightening.
2. **A·19** (per-class F1 in Table 2) — pure presentation; data already exists.
3. **A·18** (bootstrap CIs) — biggest credibility lift; ~1 day of work to add and re-render.
4. **A·20** (per-dim R² distribution) — supplementary figure; moderate effort.
5. **C·21** (citation check) — author task, can be done at any time.

Items D·25–30 stay on the "decided not to act" shelf.
