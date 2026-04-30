# Zero-shot demo — NT-v2 + meanD

Pipeline: tokenise CDS with NT-v2 (special tokens wrapped per chunk) → extract per-chunk mean-token vectors → reduce across chunks via `concat[first_chunk, last_chunk, mean_chunks]` (the headline `meanD` variant) → 5-way logistic probe (C=1.0) trained on train+val of the frozen 70/15/15 split.

Sample: 4 test-set genes — the two with the longest NCBI summaries (*well-characterised*, calibration) and the two with the shortest non-trivial summaries (*poorly characterised*, novelty).

## JAK2  (ENSG00000096968)

- **Cohort:** well-characterised
- **True family:** `kinase`
- **Predicted family:** `kinase` ✅
- **Class probabilities:** kinase=0.980 · ion=0.017 · tf=0.003 · gpcr=0.001 · immune=0.000
- **Summary (1636 chars):** Gene Symbol JAK2 This gene encodes a non-receptor tyrosine kinase that plays a central role in cytokine and growth factor signalling. The primary isoform of this protein has an N-terminal FERM domain …

**Top-5 nearest neighbours in train+val (NT-v2 + meanD embedding cosine):**

| # | Symbol | Family | Cos | Summary (first 80 chars) |
|---|---|---|---:|---|
| 1 | `PIK3CA` | `kinase` | 0.9906 | Gene Symbol PIK3CA Phosphatidylinositol 3-kinase is composed of an 85 kDa regula… |
| 2 | `TRPM7` | `kinase` | 0.9874 | Gene Symbol TRPM7 This gene belongs to the melastatin subfamily of transient rec… |
| 3 | `ADGRL2` | `gpcr` | 0.9868 | Gene Symbol ADGRL2 This gene encodes a member of the latrophilin subfamily of G-… |
| 4 | `PIK3C2A` | `kinase` | 0.9826 | Gene Symbol PIK3C2A The protein encoded by this gene belongs to the phosphoinosi… |
| 5 | `PIK3CB` | `kinase` | 0.9795 | Gene Symbol PIK3CB This gene encodes an isoform of the catalytic subunit of phos… |

Top-5 neighbour-vote agreement with predicted family: **4/5**

---

## TRAF6  (ENSG00000175104)

- **Cohort:** well-characterised
- **True family:** `tf`
- **Predicted family:** `tf` ✅
- **Class probabilities:** tf=0.755 · kinase=0.241 · immune=0.003 · ion=0.000 · gpcr=0.000
- **Summary (1450 chars):** Gene Symbol TRAF6 The protein encoded by this gene is a member of the TNF receptor associated factor (TRAF) protein family. TRAF proteins are associated with, and mediate signal transduction from, mem…

**Top-5 nearest neighbours in train+val (NT-v2 + meanD embedding cosine):**

| # | Symbol | Family | Cos | Summary (first 80 chars) |
|---|---|---|---:|---|
| 1 | `ZBTB44` | `tf` | 0.9882 | Gene Symbol ZBTB44 Predicted to enable DNA binding activity and metal ion bindin… |
| 2 | `TADA2A` | `tf` | 0.9877 | Gene Symbol TADA2A Many DNA-binding transcriptional activator proteins enhance t… |
| 3 | `BMPR1B` | `kinase` | 0.9872 | Gene Symbol BMPR1B This gene encodes a member of the bone morphogenetic protein … |
| 4 | `PLAG1` | `tf` | 0.9868 | Gene Symbol PLAG1 Pleomorphic adenoma gene 1 encodes a zinc finger protein with … |
| 5 | `ZCCHC4` | `tf` | 0.9865 | Gene Symbol ZCCHC4 Enables S-adenosyl-L-methionine binding activity; rRNA (adeni… |

Top-5 neighbour-vote agreement with predicted family: **4/5**

---

## ZNF839  (ENSG00000022976)

- **Cohort:** poorly characterised
- **True family:** `tf`
- **Predicted family:** `tf` ✅
- **Class probabilities:** tf=0.943 · kinase=0.050 · immune=0.006 · gpcr=0.000 · ion=0.000
- **Summary (66 chars):** Gene Symbol ZNF839 Predicted to enable metal ion binding activity.

**Top-5 nearest neighbours in train+val (NT-v2 + meanD embedding cosine):**

| # | Symbol | Family | Cos | Summary (first 80 chars) |
|---|---|---|---:|---|
| 1 | `ANKZF1` | `tf` | 0.9801 | Gene Symbol ANKZF1 Predicted to enable metal ion binding activity. Involved in c… |
| 2 | `DGKI` | `kinase` | 0.9788 | Gene Symbol DGKI This gene is a member of the type IV diacylglycerol kinase subf… |
| 3 | `ZMYND15` | `tf` | 0.9778 | Gene Symbol ZMYND15 This gene encodes a MYND-containing zinc-binding protein wit… |
| 4 | `PGR` | `tf` | 0.9771 | Gene Symbol PGR This gene encodes a member of the steroid receptor superfamily. … |
| 5 | `RBSN` | `tf` | 0.9756 | Gene Symbol RBSN This gene encodes a protein that belongs to the FYVE zinc finge… |

Top-5 neighbour-vote agreement with predicted family: **4/5**

---

## ZNHIT2  (ENSG00000174276)

- **Cohort:** poorly characterised
- **True family:** `tf`
- **Predicted family:** `tf` ✅
- **Class probabilities:** tf=0.899 · kinase=0.100 · immune=0.001 · ion=0.000 · gpcr=0.000
- **Summary (66 chars):** Gene Symbol ZNHIT2 Predicted to enable metal ion binding activity.

**Top-5 nearest neighbours in train+val (NT-v2 + meanD embedding cosine):**

| # | Symbol | Family | Cos | Summary (first 80 chars) |
|---|---|---|---:|---|
| 1 | `IRX6` | `tf` | 0.9775 | Gene Symbol IRX6 Predicted to enable DNA-binding transcription activator activit… |
| 2 | `KLF2` | `tf` | 0.9770 | Gene Symbol KLF2 This gene encodes a protein that belongs to the Kruppel family … |
| 3 | `SOX18` | `tf` | 0.9759 | Gene Symbol SOX18 This gene encodes a member of the SOX (SRY-related HMG-box) fa… |
| 4 | `IRX3` | `tf` | 0.9757 | Gene Symbol IRX3 |
| 5 | `ZSCAN18` | `tf` | 0.9754 | Gene Symbol ZSCAN18 Predicted to enable DNA-binding transcription factor activit… |

Top-5 neighbour-vote agreement with predicted family: **5/5**

---

