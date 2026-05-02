# Cross-modal zero-shot demo

Two probes on the **same DNA input**, both trained on train+val of the frozen 70/15/15 family-stratified split (seed 42):

1. **NT-v2 + meanD → logistic probe** (C=1.0) → predicted family.
2. **DNABERT-2 + meanG → Ridge probe** (α=10.0) → predicted GenePT 1536-d vector → top-3 nearest train+val gene summaries by cosine similarity in the GenePT text-embedding space.

Sample: 4 test genes — 2 with the longest NCBI summaries (well-characterised, calibration) and 2 with the shortest non-trivial summaries (poorly characterised, novelty test). Summaries are *never* a model input — only used to pick which test genes to demo and to display retrieved neighbours.

## JAK2  (ENSG00000096968)

- **Cohort:** well-characterised
- **True family:** `kinase`
- **Predicted family:** `kinase` ✅
- **Family probabilities:** kinase=0.980 · ion=0.017 · tf=0.003 · gpcr=0.001 · immune=0.000
- **Actual NCBI summary (1636 chars):** Gene Symbol JAK2 This gene encodes a non-receptor tyrosine kinase that plays a central role in cytokine and growth factor signalling. The primary isoform of this protein has an N-terminal FERM domain that is required for…

**Top-3 retrieved gene summaries** (nearest train+val genes by cosine similarity in predicted GenePT 1536-d text-embedding space):

1. `MAP3K12` (family `kinase`, cos=0.9453)
   > Gene Symbol MAP3K12 This gene encodes a member of the serine/threonine protein kinase family. This kinase contains a leucine-zipper domain and is predominately expressed in neuronal cells. The phosphorylation state of th…

2. `TNK1` (family `kinase`, cos=0.9451)
   > Gene Symbol TNK1 The protein encoded by this gene belongs to the tyrosine protein kinase family. Tyrosine protein kinases are important regulators of intracellular signal transduction pathways, mediating cellular prolife…

3. `PRKD3` (family `kinase`, cos=0.9450)
   > Gene Symbol PRKD3 This gene belongs to the multigene protein kinase D family of serine/threonine kinases, which bind diacylglycerol and phorbol esters. Members of this family are characterized by an N-terminal regulatory…

Retrieved-summary family agreement with predicted family: **3/3**

---

## TRAF6  (ENSG00000175104)

- **Cohort:** well-characterised
- **True family:** `tf`
- **Predicted family:** `tf` ✅
- **Family probabilities:** tf=0.755 · kinase=0.241 · immune=0.003 · ion=0.000 · gpcr=0.000
- **Actual NCBI summary (1450 chars):** Gene Symbol TRAF6 The protein encoded by this gene is a member of the TNF receptor associated factor (TRAF) protein family. TRAF proteins are associated with, and mediate signal transduction from, members of the TNF rece…

**Top-3 retrieved gene summaries** (nearest train+val genes by cosine similarity in predicted GenePT 1536-d text-embedding space):

1. `TNK1` (family `kinase`, cos=0.9423)
   > Gene Symbol TNK1 The protein encoded by this gene belongs to the tyrosine protein kinase family. Tyrosine protein kinases are important regulators of intracellular signal transduction pathways, mediating cellular prolife…

2. `MAP3K12` (family `kinase`, cos=0.9402)
   > Gene Symbol MAP3K12 This gene encodes a member of the serine/threonine protein kinase family. This kinase contains a leucine-zipper domain and is predominately expressed in neuronal cells. The phosphorylation state of th…

3. `MAP3K8` (family `kinase`, cos=0.9397)
   > Gene Symbol MAP3K8 This gene is an oncogene that encodes a member of the serine/threonine protein kinase family. The encoded protein localizes to the cytoplasm and can activate both the MAP kinase and JNK kinase pathways…

Retrieved-summary family agreement with predicted family: **0/3**

---

## ZNF839  (ENSG00000022976)

- **Cohort:** poorly characterised
- **True family:** `tf`
- **Predicted family:** `tf` ✅
- **Family probabilities:** tf=0.943 · kinase=0.050 · immune=0.006 · gpcr=0.000 · ion=0.000
- **Actual NCBI summary (66 chars):** Gene Symbol ZNF839 Predicted to enable metal ion binding activity.

**Top-3 retrieved gene summaries** (nearest train+val genes by cosine similarity in predicted GenePT 1536-d text-embedding space):

1. `MAP3K12` (family `kinase`, cos=0.9362)
   > Gene Symbol MAP3K12 This gene encodes a member of the serine/threonine protein kinase family. This kinase contains a leucine-zipper domain and is predominately expressed in neuronal cells. The phosphorylation state of th…

2. `TNK1` (family `kinase`, cos=0.9344)
   > Gene Symbol TNK1 The protein encoded by this gene belongs to the tyrosine protein kinase family. Tyrosine protein kinases are important regulators of intracellular signal transduction pathways, mediating cellular prolife…

3. `ZNF148` (family `tf`, cos=0.9343)
   > Gene Symbol ZNF148 The protein encoded by this gene is a member of the Kruppel family of zinc finger DNA binding proteins. The encoded protein activates transcription of the T-cell receptor and intestinal alkaline phosph…

Retrieved-summary family agreement with predicted family: **1/3**

---

## ZNHIT2  (ENSG00000174276)

- **Cohort:** poorly characterised
- **True family:** `tf`
- **Predicted family:** `tf` ✅
- **Family probabilities:** tf=0.899 · kinase=0.100 · immune=0.001 · ion=0.000 · gpcr=0.000
- **Actual NCBI summary (66 chars):** Gene Symbol ZNHIT2 Predicted to enable metal ion binding activity.

**Top-3 retrieved gene summaries** (nearest train+val genes by cosine similarity in predicted GenePT 1536-d text-embedding space):

1. `ZFX` (family `tf`, cos=0.9401)
   > Gene Symbol ZFX This gene on the X chromosome is structurally similar to a related gene on the Y chromosome. It encodes a member of the krueppel C2H2-type zinc-finger protein family. The full-length protein contains an a…

2. `ZFHX3` (family `tf`, cos=0.9394)
   > Gene Symbol ZFHX3 This gene encodes a transcription factor with multiple homeodomains and zinc finger motifs, and regulates myogenic and neuronal differentiation. The encoded protein suppresses expression of the alpha-fe…

3. `NKX1-1` (family `tf`, cos=0.9379)
   > Gene Symbol NKX1-1 This gene encodes a transcription factor that belongs to NKX family of homeodomain-containing proteins which are critical regulators of organ development. In mice, the orthologous gene is expressed pre…

Retrieved-summary family agreement with predicted family: **3/3**

---

