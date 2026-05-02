# Cross-modal demo

Two linear probes on the **same DNA input**, both trained on train+val of the
frozen 70/15/15 family-stratified split (seed 42):

1. **Family probe** — NT-v2 + `meanD` → logistic (C=1.0) → predicted family.
2. **Text probe** — DNABERT-2 + `meanG` → Ridge (α=10.0) → predicted GenePT
   1536-d vector → top-3 nearest train+val gene summaries by cosine
   similarity in the GenePT text-embedding space.

Sample: 4 test-set genes — 2 with the longest NCBI summaries (well-
characterised, calibration) and 2 with the shortest non-trivial summaries
(poorly characterised, novelty test). Summaries are *never* a model input;
they are only used to choose which test genes to demo and to display the
retrieved neighbours alongside their actual NCBI text.

Output: `demo/output.md` (markdown, ready to drop into the deck).

Generate: `uv run python demo/cross_modal.py`

Latest run: 4/4 family predictions correct (JAK2/kinase 0.98, TRAF6/tf 0.76,
ZNF839/tf 0.94, ZNHIT2/tf 0.90). The retrieved summaries are the more
interesting result — for ZNHIT2 (66-char NCBI entry), the text probe
returns three rich zinc-finger / transcription-factor descriptions
(ZFX, ZFHX3, NKX1-1). For TRAF6, the text probe places it next to kinases
(TNK1, MAP3K12, MAP3K8), reflecting TRAF6's role inside MAP-kinase /
JNK signalling pathways — a different read than the family probe gives.

`zero_shot.py` is the earlier single-probe demo (DNA → family + DNA-space
neighbours), kept for reference.
