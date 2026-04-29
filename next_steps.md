# Next Steps

Living project log. Phases 1–3 are done; phase 4 carries the deck's Week 4–5 deliverables; phase 5 collects the open scientific questions surfaced by `findings.md`.

## Phase 1 — Data pipeline   ✅ done   (deck Weeks 1–2)

- 3244 genes, 5 families: tf 1743, gpcr 591, kinase 558, ion 198, immune 154
- HGNC TSV cached (`data/hgnc/hgnc_complete_set.tsv`); Ensembl CDS cached (3244 FASTAs in `data/sequences/`)
- Stratified 70/15/15 split frozen in `data/splits.json` (2270 / 487 / 487)
- Code: `src/data_loader/`, `src/splits/`, `scripts/prepare_data.py`, `scripts/make_splits.py`

## Phase 2 — Encoder embeddings   ✅ done   (deck Week 2)

- DNABERT-2:  3244 × 768  →  `data/embeddings/`, `data/dataset.parquet`
- NT-v2 100M: 3244 × 512  →  `data/embeddings_nt_v2/`, `data/dataset_nt_v2.parquet`

NT-v2 was added beyond the original deck plan as a second encoder so we have converging-evidence on any ceiling we hit.

Code: `src/data_loader/{encoder_runner,nt_v2_encoder}.py`, `scripts/run_encoder.py`, `scripts/run_nt_v2_encoder.py`.

## Phase 3 — Probe + baseline + sanity   ✅ done   (deck Week 3)

| Run | Encoder | Test cos | Test R² | Best config |
|---|---|---:|---:|---|
| Linear probe | DNABERT-2 | 0.9313 | 0.181 | α=10 |
| Linear probe | NT-v2 | 0.9324 | 0.193 | α=10 |
| 4-mer baseline | — | 0.9306 | 0.174 | α=0.01 |
| Anti-baseline | DNABERT-2 | 0.9128 | −0.003 | α=1000 |
| Anti-baseline | NT-v2 | 0.9130 | −0.000 | α=1000 |
| MLP probe | DNABERT-2 | 0.9300 | 0.162 | hidden=(256,), α=0.01 |
| MLP probe | NT-v2 | 0.9325 | 0.189 | hidden=(1024,), α=0.01 |

Outcome: **informative negative** per `framework.md` § Success / failure criteria — both encoders tie the 4-mer baseline within ±0.002 cosine. Anti-baseline R² ≈ 0 confirms the pipeline is honest. MLP depth doesn't move the number, so the ceiling is the representation, not the probe's capacity. See `findings.md` for the full read.

Code: `src/linear_trainer/`, `src/kmer_baseline/`, `scripts/train_{probe,baseline,anti_baseline,mlp_probe}.py`.

## Phase 4 — Classification reframing   (deck Weeks 4–5)

Original Phase 4 (retrieval@k, IG attribution, family-classification-on-`y_hat`, zero-shot demo, viz, write-up) was superseded by the 2026-04-29 classification-pivot spec — see `docs/superpowers/specs/2026-04-29-classification-pivot-design.md`.

### Phase 4a — Classification probes   ✅ done

15-cell run matrix (3 tasks × 5 feature sources). Headline: **NT-v2 5-way macro-F1 = 0.803 vs 4-mer baseline 0.672 (+0.131)**. Decision gate landed in Branch 1 — encoder beats 4-mer; Phase 4b pooling re-extraction is **skipped**. Full results in `findings.md` § "Phase 4 — Classification reframing".

Code: `src/binary_tasks/`, `src/length_baseline/`, `src/linear_trainer/logistic_probe.py`, `scripts/{make_binary_subsets,train_logistic_probe}.py`.

Artefacts: `data/binary_tf_vs_gpcr.json`, `data/binary_tf_vs_kinase.json`, `data/confusion_5way_{dnabert2,nt_v2}.json`, 15 entries in `data/metrics.json` with `model == "logistic_probe"`.

### Phase 4b — Pooling re-extraction   skipped

Decision gate from `findings.md` landed in Branch 1 (encoder beats 4-mer). The spec's pooling menu (mean→D, mean→G, max→mean, CLS→mean against the mean→mean baseline) remains documented in the spec for reference but is not triggered.

### Phase 4c — Write-up   ⏳ open

- [x] Results table (15 cells) in `findings.md`
- [x] 5-way confusion matrix per encoder (saved as JSON, embedded in `findings.md`)
- [ ] Slide-deck write-up: intro (from `project.md`) → methods (point at `framework.md`) → Phase 3 informative-negative + Phase 4 classification result → discussion of the encoder gap (NT-v2 vs DNABERT-2)

## Phase 5 — Optional ceiling-breaker experiments   🔬 open, lower priority

Strictly after Phase 4c. These address the three caveats in `findings.md`.

- [ ] **Pooling sweep** — the spec's Phase 4b menu (mean→D, mean→G, max→mean, CLS→mean) could still be informative as an ablation even though the headline doesn't require it. Would establish whether DNABERT-2's gap to NT-v2 is recoverable by better pooling.
- [ ] **Window sweep** — full transcript (promoter + UTR + CDS) instead of CDS-only. CDS is the most composition-homogenous part of a gene because of codon usage; promoters and UTRs may carry more function-discriminating signal.
- [ ] **Optional third encoder** — HyenaDNA, Caduceus, or GENA-LM. Lowest priority now that NT-v2 has demonstrated the encoder is doing real work; a third encoder would test whether the NT-v2 result generalises or is architecture-specific.

## Resolved / archived

- **Mean-of-chunks pooling vs CLS / max-pool?** → Unresolved empirically; deferred to Phase 5. Cosine plateaued at the 4-mer baseline on both encoders, which was the trigger condition.
- **Five families enough resolution?** → Yes for the current corpus and probe. Sub-family classification is a Phase 4-or-later ask after we see the cluster figures.
- **Version `dataset.parquet` if pooling changes?** → Handled implicitly by adding `dataset_nt_v2.parquet` as a sibling artefact rather than mutating in place. Future encoder/pooling variants follow the same convention.

## Dependency graph

```
[1 data] → [2 encoders] → [3 probe + baseline + sanity] → [4 deliverables] → [5 ceiling-breakers]
                                                                  └→ write-up
```
