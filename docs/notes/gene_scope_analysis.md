# Gene-scope analysis (G-1 / G-2 / G-3)

Date: 2026-05-15
Branch: `revision/tss-and-gene-scope`
Source: re-derivation of the curation pipeline at `src/data_loader/dataset_loader.py:21–195` against the local GenePT v2 pickle (extracted from `data.zip`) and the HGNC complete TSV at `data/hgnc/hgnc_complete_set.tsv`.

Answers the §Phase 2 questions in `docs/conference_resubmission_plan.md`: would expanding the gene set materially help, and at what cost?

---

## Inputs

| Source | Count |
|---|---|
| GenePT ada_text pickle keys (gene symbols) | **93,800** |
| HGNC complete set (all rows) | 44,210 |
| HGNC protein-coding with non-null `ensembl_id` | **19,253** |
| HGNC PC ∩ GenePT (matched by symbol) | **18,836** |

## Per-family breakdown

After applying the 5 HGNC-regex filters (case-insensitive against `gene_group`, with paper's exclude patterns):

| Family | HGNC regex match (PC only) | ∩ GenePT | After first-family-wins dedup |
|---|---:|---:|---:|
| Protein kinases | 564 | 558 | 558 |
| Transcription factors | 1,757 | 1,745 | 1,743 |
| Ion channels | 201 | 200 | 198 |
| GPCRs | 617 | 601 | 594 |
| Immune receptors | 164 | 160 | 154 |
| **Union (unique ensembls)** | — | **3,247** | **3,247** |

Cross-family overlaps (genes whose `gene_group` matches multiple family regexes, pre-dedup):

| Pair | Count |
|---|---:|
| kinase ∩ tf | 2 |
| kinase ∩ ion | 2 |
| kinase ∩ gpcr | 7 |
| kinase ∩ immune | 4 |
| gpcr ∩ immune | 2 |
| **Total cross-family ambiguities** | **17** |

These 17 are assigned to the higher-priority family (kinase → tf → ion → gpcr → immune). They are **not lost** — they are assigned-once, just possibly to the wrong family from a domain-biology standpoint.

## G-1 — Within-family ceiling

**Ceiling = 3,247 unique ensembl IDs** (union of HGNC-PC ∩ regex ∩ GenePT across all 5 families).

Current gene_table has **3,244 rows**. So the within-5-families ceiling is +3 genes (**0.1 % lift**) — below decision-rule threshold.

## G-2 — Paralog drops under first-family-wins

**0 drops in the strict sense.** The dedup logic is "first family wins"; every ensembl that matches ≥1 family is assigned to exactly one. The 17 cross-family-ambiguous genes (above) are *assigned*, not *dropped*.

**There is no recoverable gene count from relaxing the dedup.** Removing the `seen_ensembl` exclusion would only let one gene appear in multiple training rows under different labels — which is label noise, not new genes, and would hurt classification rather than help.

## G-3 — CDS-fetch failures

| Quantity | Count |
|---|---:|
| `.fa` files under `data/sequences/` | 3,244 |
| Genes in `gene_table.parquet` without a `.fa` file | **0** |
| `.fa` files not in `gene_table.parquet` | 0 |

The 3-gene gap between the regex-ceiling (3,247) and the current table (3,244) is consistent with three Ensembl REST fetches having failed at original-build time. Recovery would require re-running `scripts/prepare_data.py` against a stable REST snapshot or fetching from a secondary source (NCBI RefSeq) — but at +3 genes the lift is negligible.

## Scope-(b) ceiling — all protein-coding genes with GenePT

**18,836 unique ensembl IDs.** This is **5.81 × the current corpus.**

But: scope-(b) would require a new family ontology (or treating the bulk as an "other" / hierarchical multi-label problem), and would directly invalidate the paper's central framing of *"five broad protein-family categories that all carry strong, well-characterised domain motifs"* (`dna_to_text_paper/paper/methods.tex:5`). It's a different paper, not a revision.

## Decision (applying §Step 2b rule)

| Rule | Outcome |
|---|---|
| < 15 % within-family lift | **Skip gene expansion.** |
| 15–50 % | (would have required paralog-aware split) |
| > 50 % | (would have been the headline upgrade) |

**Decision: SKIP.** The recoverable within-family lift is 0.1 %. Scope-(b) is out of revision scope.

### What to communicate in the paper instead

- Leave the §Limitations paragraph about "five broad protein-family categories" as-is — the constraint is real and the corpus is already at the within-scope ceiling.
- The paralog-leakage paragraph (`discussion.tex:19`) still stands as a limitation; we did not add MMseqs2/CD-HIT clustered splitting (out of scope for this revision cycle). Note in `docs/archive/notes/paper_revision_followups.md` that paralog-aware splitting remains an open item, separate from gene-count.
- Reviewer feedback "include more genes if feasible" is answered: within the stated family scope, the corpus is already saturated (3,244 / 3,247 ≈ 99.9 %). Broader expansion is a separate study.

## Implication for the rest of the plan

- **G-5 (paralog-aware split):** keep as deferred. The case for it is independent of N — it's about whether the *existing* 3,244 genes have homology-leaking train/test splits. That's a separate workstream that the paper already flags.
- **T-1 .. T-7 (TSS expansion):** unchanged. This was always the higher-value item.

## Reproducer

```bash
uv run python -c "
import pickle, re
from pathlib import Path
import pandas as pd
# … see git history of this file for full script …
"
```

Or rerun the inline block from the chat session of 2026-05-15 (see `docs/conference_resubmission_plan.md` live tracker for the commit hash that landed this report).
