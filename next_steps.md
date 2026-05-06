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

### Phase 4b — Pooling sweep   ✅ done (exploratory)

Originally going to be skipped (Phase 4a already cleared the decision gate), but ran the full menu anyway as an ablation. Results in `findings.md` § "Phase 4b — Pooling sweep (exploratory)".

Headline outcomes:
- **Tokenisation fix is the biggest win.** Re-tokenising with special tokens (`[CLS]`/`[SEP]` for DNABERT-2, `<cls>` for NT-v2) lifts DNABERT-2 substantially even at the mean→mean baseline. Phase 1–3 was tokenising without specials and crippling DNABERT-2.
- **`meanD` is the only pooling variant that reliably helps.** Best 5-way: `nt_v2_meanD` 0.828 (+0.024 vs Phase 4a NT-v2).
- **`maxmean` and `clsmean` consistently hurt.** Both failures of deep-research priors.

Code: `src/data_loader/multi_pool.py`, `src/data_loader/pooling_aggregator.py`, `scripts/run_multi_pool_extract.py`, `scripts/build_pooling_datasets.py`.

Artefacts: `data/chunk_reductions_{dnabert2,nt_v2}/` (per-chunk reductions cache), 10 new `data/dataset_{encoder}_{variant}.parquet` files, 30 new entries in `data/metrics.json`, 10 new `data/confusion_5way_{encoder}_{variant}.json`.

### Phase 4c — Write-up   ⏳ open

- [x] Results table (15 cells) in `findings.md`
- [x] 5-way confusion matrix per encoder (saved as JSON, embedded in `findings.md`)
- [ ] Slide-deck write-up: intro (from `project.md`) → methods (point at `framework.md`) → Phase 3 informative-negative + Phase 4 classification result → discussion of the encoder gap (NT-v2 vs DNABERT-2)

## Phase 5a — Regression re-run on new variants   ✅ done

Ridge probe (Ridge → GenePT 1536-d) re-run on all 10 new pooling-variant parquets. Results in `findings.md` § "Phase 5a — Regression re-run on the new variants".

Headline:
- **`dnabert2_meanG` R² = 0.210**, +0.036 vs 4-mer baseline (vs Phase 3's +0.007). DNABERT-2's Phase 3 informative-negative was an undercount.
- NT-v2 regression unchanged by tokenisation or pooling — confirms the regression ceiling for NT-v2 is real.
- Classification gain from `meanD` for NT-v2 is family-specific, not signal-general (doesn't recover the full GenePT vector).

Code: `scripts/train_probe.py` (no changes — already accepts `--dataset`).
Artefacts: 10 new `data/probe_{encoder}_{variant}.npz`, 10 new entries in `data/metrics.json` with `model == "linear_probe"`.

## Phase 5 — Encoder expansion   🔬 in progress

Revised paper path: keep the headline task to 5-way family classification plus Ridge-to-GenePT regression. Move `tf-vs-gpcr` and `tf-vs-kinase` to legacy/appendix, and do not run new length baselines.

- [x] **Model registry** — central encoder specs for DNABERT-2, NT-v2, GENA-LM base, and HyenaDNA.
- [x] **GENA-LM + HyenaDNA plumbing** — multi-pool extraction and parquet materialisation use the registry.
- [x] **Enformer comparator plumbing** — TSS-centered windows, internal trunk features, supervised output-track features, and matched TSS-window 4-mer dataset.
- [x] **Focused table builder** — `scripts/build_family5_table.py` keeps the main table to family5 + Ridge R².
- [x] **Run GENA-LM extraction** — cached 3,244 CDS embeddings and materialised all supported pooling datasets.
- [x] **Run GENA-LM probes** — logistic family5 and Ridge-to-GenePT cells for `meanmean`, `meanD`, `meanG`, `maxmean`, and `clsmean`.
- [x] **Regression table builder** — `scripts/build_regression_table.py` writes `data/regression_table.md`.
- [x] **Run HyenaDNA extraction/probes** — cached 3,244 CDS embeddings and ran family5 + Ridge-to-GenePT cells for `meanmean`, `meanD`, `meanG`, `maxmean`, and `clsmean`.
- [x] **Run Enformer extraction/probes** — TSS windows, Enformer trunk/track datasets, matched TSS 4-mer, family5, and Ridge cells are cached.
- [ ] **Run TSS self-supervised ablation** — execute NT-v2 on TSS windows, materialise TSS pooling datasets, and run family5 + Ridge probes.

HyenaDNA checkpoint:

- Best 5-way cell: `hyena_dna_meanG`, macro-F1 0.7149, kappa 0.6944, accuracy 0.8090.
- Best Ridge cell: `hyena_dna`/`meanmean`, R² 0.1822, slightly above the CDS 4-mer baseline (0.1743) but below DNABERT-2's best pooled cell (0.2104).
- `clsmean` collapses on classification (macro-F1 0.1396, kappa 0) and regression (R² -0.0015), consistent with HyenaDNA not having a trained CLS-style summary token.

Open caveat after encoder expansion:

- [ ] **Window sweep for self-supervised encoders** — full transcript or promoter/UTR/CDS instead of CDS-only.

## Resolved / archived

- **Mean-of-chunks pooling vs CLS / max-pool?** → Addressed empirically in Phase 4b/5. `meanD`/`meanG` are the strongest variants; `maxmean` is consistently weak; `clsmean` is model-dependent and collapses for NT-v2/HyenaDNA.
- **Five families enough resolution?** → Yes for the current corpus and probe. Sub-family classification is a Phase 4-or-later ask after we see the cluster figures.
- **Version `dataset.parquet` if pooling changes?** → Handled implicitly by adding `dataset_nt_v2.parquet` as a sibling artefact rather than mutating in place. Future encoder/pooling variants follow the same convention.

## Dependency graph

```
[1 data] → [2 encoders] → [3 probe + baseline + sanity] → [4 deliverables] → [5 ceiling-breakers]
                                                                  └→ write-up
```
