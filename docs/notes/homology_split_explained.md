# How the homology-aware split works — a primer

_Companion to `homology_phase2_results.md`, `homology70_supplementary.md`, and the
clustering code in `src/cluster/mmseqs_cluster.py` + `src/splits/make_splits.py`._

This note explains, from first principles, what the MMseqs2 homology split is, why we
split on sequence homology instead of the classification labels, what the 40% / 70%
thresholds mean, and why scores go **up** at 70%. It also clears up a common
misconception: **clustering here does not delete any genes.**

---

## 1. What is a "residue" and an "aligned residue"?

A **residue** is one unit of a protein chain — a single amino acid. A protein is just a
string of residues, e.g. `MKLVAG` is 6 residues long.

To compare two proteins we **align** them: slide them against each other to line up
matching positions, inserting gaps (`-`) where one sequence has an insertion or deletion.
An *aligned residue* is a position in that lined-up comparison.

```
Seq A:  M K L V A G
Seq B:  M K - V A S
        ✓ ✓   ✓ ✓ ✗     4 matches / 5 aligned positions = 80% identity
```

**% identity** = (matching aligned positions) / (aligned length). We align the
**protein translations** of each gene's CDS, not the raw DNA, because protein sequence is
what reflects shared function. (Pipeline: CDS → `translate_cds` → FASTA → `mmseqs
easy-cluster`; see `src/cluster/mmseqs_cluster.py`.)

---

## 2. Why split on homology, not on the classification label?

The classification task is 5-way protein-**family** prediction (`family5`). It's tempting
to think "just hold out whole families." Two reasons that's wrong:

**(a) Splitting *by* label breaks the task.** If whole families go to train and *different*
families go to test, the probe is asked to predict classes it never saw a single example
of — impossible by construction. Family classification needs all 5 labels present in
**both** train and test. (The regression arm — predicting a continuous GenePT text
embedding — has no discrete label to split on at all.) So both our splits **stratify by
family**: every family appears in every split at ~70/15/15 proportions
(`src/splits/make_splits.py:80`).

**(b) The leakage is finer than the label.** Two genes in the *same* family can be
near-identical **paralogs** (a recent gene duplication). A family-stratified *random* split
still throws individual genes into train/test independently, so one copy of a duplicate
lands in train and its near-twin in test — the model effectively memorizes the test gene.
Family-level reasoning can't catch this, because both twins carry the *same* family label.

Homology clustering works **below the label**: within "kinases" it finds the actual
sequence-similar sub-groups and forces each sub-group onto one side of the split. You keep
every family in every split (task still works) **and** stop near-duplicates from straddling
train/test (leakage removed).

```
family label:     [ kinase ][ kinase ][ kinase ][ kinase ]   ...all "kinase"
                    gene A    gene A'   gene B    gene C
homology cluster:  └ cluster 7 ┘      [clstr 9] [clstr 9]
                   A,A' are paralogs   B,C unrelated kinases
                   → same split        → free to differ
```

The label cannot make the A/A' vs B/C distinction; sequence identity can.

(The five families in this dataset are `gpcr`, `immune`, `ion`, `kinase`, `tf` — there is
no "enzyme" class; `kinase` is used above only as a concrete example.)

---

## 3. What the 40% / 70% threshold is, and how it was chosen

It's the `--min-seq-id` parameter to MMseqs2 (`mmseqs_cluster.py:56`): link two proteins
into the same cluster if they are **≥ X% identical over ≥ 80% of their length**
(`-c 0.8`, bidirectional coverage).

It is **not learned or fitted** from this dataset — it's a chosen convention. ~40% identity
is a standard, conservative redundancy-reduction cutoff in protein bioinformatics (the same
family of thresholds CD-HIT / PISCES pipelines use). Below ~30–35% you enter the homology
"twilight zone" where shared sequence no longer reliably implies shared function, so 40% is
a defensible "treat anything from clearly-homologous down to moderately-similar as the same
thing." The reviewer (MINA #1) asked for a homology-aware primary split plus a stricter
supplementary at 70–90%; we adopted 40% as primary and 70% as the supplementary check
rather than tuning either.

---

## 4. The terminology trap: "stricter threshold" ≠ "stricter test"

There are two opposite senses of "strict," and they pull in opposite directions:

- A **stricter identity threshold** (70%) demands *more* similarity to group two genes →
  *smaller, more numerous* clusters → *fewer* genes locked together.
- A **more conservative split** (40%, the lower number) groups *more* aggressively →
  *bigger, fewer* clusters → *more* genes locked together → *less* leakage.

So the **lower** identity number gives the **harder, more honest** test. That feels
backwards until you trace one pair of genes through it (next section).

---

## 5. Why do scores go *up* at 70%? (Grouping less ⇒ more leakage)

The key correction: **grouping doesn't raise scores — grouping *less* raises scores**, and
70% groups less than 40%.

Trace one pair. Suppose gene X and gene Y are **55% identical** (moderately-related
paralogs):

- **At 40%:** rule is "group if ≥40% identical." 55% ≥ 40% → X, Y grouped → forced to the
  same side → **no leakage** from this pair.
- **At 70%:** rule is "group if ≥70% identical." 55% < 70% → X, Y **not** grouped → free to
  land in opposite splits → X memorized in train, Y tested → **leakage returns** → score
  goes up.

A higher threshold is a *coarser net* that catches only near-twins and lets the moderate
paralogs slip through to leak. The whole ladder:

| Split        | Group if…       | Locked together         | Leakage left | Score        |
|--------------|-----------------|-------------------------|--------------|--------------|
| Random (old) | (no grouping)   | nothing                 | maximum      | highest      |
| Homology 70% | ≥70% identical  | near-twins only         | moderate     | high         |
| Homology 40% | ≥40% identical  | near-twins + moderate   | least        | lowest (honest) |

More grouping → less leakage → **lower** score. 40% is the lowest score *because* it groups
the most. That is exactly why 40% is the conservative primary and 70% is a looser sanity
check: the conclusions (ESM-2 wins, AA-composition ties the DNA-LMs, TSS near chance) hold
at **both** thresholds, so they don't depend on where the dial is set. The TSS arm climbing
0.326 (40%) → 0.505 (70%) is the bonus — the leftover "signal" really is leakage seeping
back in. (See `homology70_supplementary.md`.)

---

## 6. Do we deduplicate? No. (And the exact counts)

**No genes are deleted.** There is a bioinformatics technique where clustering means "keep
one representative per cluster and throw the rest away" (CD-HIT-style redundancy reduction).
**That is not what happens here.** Clustering is used *only to decide which side of the
split a gene goes to* — the cluster id is a "these must stay together" tag, not a delete
list (`build_cluster_splits`, `make_splits.py:64`). Every one of the 3,244 genes is still
trained or tested on, at every threshold.

So the **gene count is constant (3,244)**; only the *partitioning* changes. What the
threshold changes is the number of clusters (how the same genes are grouped), which controls
how much leakage survives.

### Actual numbers (from `data/splits*.json`, seed 42, 70/15/15 family-stratified)

| Split                       | file                       | min-seq-id | # clusters | train | val | test |
|-----------------------------|----------------------------|-----------:|-----------:|------:|----:|-----:|
| **Homology 40% (PRIMARY)**  | `splits.json`              | 0.40       | **1,751**  | 2271  | 486 | 487  |
| Homology 70% (supplementary)| `splits_homology70.json`   | 0.70       | **2,869**  | 2271  | 487 | 486  |
| Random (sensitivity)        | `splits_random.json`       | —          | —          | 2270  | 487 | 487  |

- **Total genes: 3,244** in every split (no deduplication). 0 genes have missing CDS
  (`n_translated = 3244`), so every gene is clustered on its real protein, not a singleton
  fallback.
- **40%** → 1,751 clusters: ~1.85 genes/cluster on average. Many clusters are singletons (a
  gene with no close paralog in the set); the rest are small families of duplicates. By the
  arithmetic, 3244 − 1751 = **1,493 genes share a cluster with ≥1 other gene** — i.e. ~1,500
  genes had a paralog the old random split was free to leak across train/test.
- **70%** → 2,869 clusters: stricter identity ⇒ the moderate-paralog clusters fragment into
  more, smaller clusters (2,869 > 1,751), covering the **same** 3,244 genes. Now only
  3244 − 2869 = **375 genes** share a cluster with another — so far fewer genes are locked
  together ⇒ more leakage survives ⇒ higher scores.
- Test size is ~15% (≈486–487) in every split; it barely changes because whole clusters move
  as a block but the target stays 70/15/15. **What differs between splits is *which* genes
  land where, not how many exist.**
- Family balance is preserved across splits by `build_cluster_splits`, which assigns whole
  clusters greedily (largest first) to whichever split least overshoots its per-family quota
  (`make_splits.py:96-112`), then asserts no cluster spans two splits.

### Family balance (40% primary split, `splits.json`)

The dataset is dominated by transcription factors. Proportions are held nearly constant
across train/val/test (this is the stratification working):

| family   | overall | train        | val        | test       |
|----------|--------:|-------------:|-----------:|-----------:|
| tf       | 1743    | 1220 (53.7%) | 262 (53.9%)| 261 (53.6%)|
| gpcr     | 591     | 414 (18.2%)  | 88 (18.1%) | 89 (18.3%) |
| kinase   | 558     | 391 (17.2%)  | 83 (17.1%) | 84 (17.2%) |
| ion      | 198     | 138 (6.1%)   | 30 (6.2%)  | 30 (6.2%)  |
| immune   | 154     | 108 (4.8%)   | 23 (4.7%)  | 23 (4.7%)  |
| **total**| **3244**| **2271**     | **486**    | **487**    |

(The 70% split has the same per-family proportions to within <0.2 pp; the chance floor for
5-way macro-F1 under this imbalance is ≈0.224, the number the TSS arm sits at.)

---

## TL;DR

- **Residue** = one amino acid; **aligned residues** = positions lined up between two
  proteins; **% identity** = fraction of those positions that match.
- We split on **homology, not labels**, because (a) the task needs every family in every
  split and (b) the cheating happens between near-duplicate genes that *share* a label —
  which label-splitting can't separate.
- **40%** is a standard conservative cutoff for "near-duplicate"; it's a chosen convention,
  validated by re-running at **70%**.
- **Lower threshold = more grouping = less leakage = lower, honest score.** Scores rise at
  70% because we group *less*, not because grouping helps.
- **No deduplication:** all **3,244** genes are kept in every split. 40% → 1,751 clusters,
  70% → 2,869 clusters; train/val/test ≈ 2271/486/487.
