# Visualisations for the deck

UMAP figures over the 3244-gene corpus. Both scripts are deterministic
(`random_state=42`) so re-runs produce the same layout.

## `figures/umap_nt_v2_meanD.png`

Single-panel UMAP of the headline pipeline (NT-v2 + meanD), coloured by
family. Anchors the "encoder carries family-discriminative signal" claim.
Visible: clean TF cluster, clean GPCR cluster, more scattered kinase/ion/
immune (consistent with the per-class accuracies in
`data/confusion_5way_nt_v2_meanD.json`).

Generate: `uv run python analysis/viz/umap_meanD.py`

## `figures/umap_dnabert2_tokenisation_compare.png`

Two-panel UMAP comparing DNABERT-2 mean→mean BEFORE the tokenisation fix
(`add_special_tokens=False`, the Phase 1–3 pipeline) and AFTER (`=True`,
Phase 4b re-extraction). Identical model, identical pooling, only the
chunk boundary tokens differ. AFTER shows visibly more family separation,
consistent with DNABERT-2's +0.05–0.07 macro-F1 lift on classification
and +0.022 R² lift on regression.

Generate: `uv run python analysis/viz/umap_tokenisation_compare.py`
