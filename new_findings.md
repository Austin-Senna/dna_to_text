# New Findings: Encoder Expansion, Enformer, and TSS Context

This note summarizes the new results after expanding beyond the original
DNABERT-2 vs NT-v2 comparison. The current paper direction is a 5-way
gene-family classification benchmark on the 3,244-gene corpus, with
Ridge-to-GenePT regression as a secondary cross-modal probe.

## Current Experimental Frame

- Corpus: 3,244 human protein-coding genes.
- Families: transcription factors, GPCRs, kinases, ion channels, immune receptors.
- Primary task: 5-way family classification.
- Secondary task: Ridge regression from DNA features into 1536-d GenePT summary embeddings.
- Main input for DNA encoders so far: canonical CDS.
- TSS/Enformer input: 196,608 bp windows centered at each gene's transcription start site.
- Frozen split: same 70/15/15 stratified split across models.

## Important Definitions

CDS means coding sequence: the nucleotide sequence that is translated into
protein. This is where protein-domain signal lives, such as kinase domains,
GPCR transmembrane patterns, ion-channel motifs, and DNA-binding domains.

TSS means transcription start site: the genomic coordinate where transcription
begins. A TSS-centered window includes broad genomic context around the start
of transcription: promoter sequence, nearby regulatory sequence, UTRs, introns,
and nearby non-coding context. It is much less directly tied to protein family
than CDS.

This distinction matters because family labels are mostly protein/function
labels, while TSS windows are mostly regulatory-context inputs.

## Headline CDS Results

### 5-Way Classification

| Feature source | Macro-F1 | Kappa | Accuracy | Notes |
|---|---:|---:|---:|---|
| NT-v2 meanD | 0.8275 | 0.8214 | 0.8871 | Best current family5 cell |
| NT-v2 meanG | 0.8257 | 0.8179 | 0.8850 | Essentially tied with meanD |
| DNABERT-2 meanD | 0.7380 | 0.7226 | 0.8234 | Strong, but below NT-v2 |
| HyenaDNA meanG | 0.7149 | 0.6944 | 0.8090 | Competitive long-context self-supervised model |
| CDS 4-mer | 0.6722 | 0.7024 | 0.8172 | Strong composition baseline |
| GENA-LM clsmean | 0.4982 | 0.4776 | 0.6858 | Best GENA-LM cached cell |
| Shuffled labels | 0.2078 | 0.0482 | 0.4353 | Sanity anti-baseline |

The strongest story remains that NT-v2 has the clearest family-discriminative
signal beyond k-mer composition. DNABERT-2 and HyenaDNA also beat or approach
the CDS 4-mer baseline, while GENA-LM is substantially weaker on this corpus.

### Ridge-To-GenePT Regression

| Feature source | R2 macro | Delta vs CDS 4-mer | Mean cosine | Notes |
|---|---:|---:|---:|---|
| DNABERT-2 meanG | 0.2104 | +0.0361 | 0.9340 | Best regression cell |
| DNABERT-2 meanD | 0.2100 | +0.0357 | 0.9340 | Essentially tied |
| NT-v2 meanmean | 0.1932 | +0.0189 | 0.9324 | Best NT-v2 regression cell |
| HyenaDNA meanmean | 0.1822 | +0.0079 | 0.9313 | Slightly above 4-mer |
| CDS 4-mer | 0.1743 | +0.0000 | 0.9306 | Strong baseline |
| GENA-LM meanmean | 0.1173 | -0.0569 | 0.9251 | Below 4-mer |

Regression remains a weaker story than classification. The GenePT target is
high-dimensional and noisy, and cosine has a high anisotropy floor. R2 is the
more honest metric. The best regression result is DNABERT-2 meanG, but the
overall deltas are modest compared with the family5 classification gains.

## New Model Findings

### NT-v2

NT-v2 is the current classification winner. Its best family5 cells are meanD
and meanG, both around 0.826 to 0.828 macro-F1. This is a large improvement
over the CDS 4-mer baseline and a clear improvement over DNABERT-2 and HyenaDNA.

Discussion angle:

NT-v2 appears to encode family-level information in CDS beyond simple nucleotide
composition. This supports the claim that self-supervised DNA encoders can learn
functionally relevant sequence structure, but this result depends strongly on
architecture, tokenization, pretraining data, and pooling.

### DNABERT-2

DNABERT-2 is strong but not the top classifier. Its best family5 cell is meanD
at 0.7380 macro-F1. In regression, however, DNABERT-2 has the best observed
Ridge-to-GenePT result, with meanG R2 = 0.2104.

Discussion angle:

DNABERT-2 may align somewhat better with GenePT summary space than NT-v2, while
NT-v2 is better for direct categorical family recovery. That split is useful:
the choice of target changes what "best representation" means.

### GENA-LM

GENA-LM base underperforms relative to the CDS 4-mer baseline on both primary
and secondary probes. Its best family5 cell is around 0.498 macro-F1, and its
best regression R2 is around 0.117.

Discussion angle:

GENA-LM is an important negative comparison. It shows that simply adding another
DNA language model does not guarantee improved biological signal. Model family,
tokenization, pretraining objective, and pooling compatibility all matter.

Possible explanations to discuss carefully:

- The checkpoint may be less suited to canonical CDS family discrimination.
- The pooling strategy may not match the geometry of the learned representation.
- GENA-LM may encode signal that is not linearly accessible in this probe.
- The result makes k-mer baselines essential, because some pretrained encoders
  can underperform simple composition on this task.

### HyenaDNA

HyenaDNA is competitive on classification, with meanG reaching 0.7149 macro-F1.
It is below NT-v2 and DNABERT-2, but it clearly beats the shuffled-label
anti-baseline and sits above or near the CDS 4-mer baseline depending on metric.
For regression, HyenaDNA meanmean reaches R2 = 0.1822, slightly above CDS 4-mer.

The CLS-style pooling result is poor for HyenaDNA, especially in regression
where clsmean falls below zero R2. This is expected for a causal long-context
model without a trained BERT-style CLS summary representation.

Discussion angle:

HyenaDNA supports the pooling-geometry argument. Long-context models can carry
useful gene-family signal, but summary-token assumptions do not transfer across
architectures. Mean-based and first/last/global pooling are safer comparisons.

## Enformer Findings

Enformer was run as a supervised sequence-to-function comparator, not as a DNA
LLM row. It uses 196,608 bp TSS-centered genomic windows rather than CDS.

### Enformer 5-Way Results

| Feature source | Macro-F1 | Kappa | Accuracy | R2 macro |
|---|---:|---:|---:|---:|
| TSS-window 4-mer | 0.2452 | 0.2050 | 0.5873 | 0.0413 |
| Enformer trunk global | 0.5450 | 0.4392 | 0.6448 | 0.1389 |
| Enformer trunk center | 0.5127 | 0.4541 | 0.6530 | 0.1425 |
| Enformer tracks center | 0.4862 | 0.4264 | 0.6386 | 0.0135 |

The best Enformer classification feature is trunk_global, with macro-F1 0.5450.
The best Enformer regression feature is trunk_center, with R2 = 0.1425.

### Interpretation

Enformer substantially improves over the matched TSS-window 4-mer baseline,
but it remains far below the best CDS encoders. This is not necessarily a model
failure. It suggests that the family5 task is mostly a CDS/protein-domain task,
not a regulatory-context task.

The TSS-window 4-mer baseline is very weak compared with the CDS 4-mer baseline:

- CDS 4-mer family5 macro-F1: 0.6722.
- TSS-window 4-mer family5 macro-F1: 0.2452.

That gap is crucial. It means the TSS input itself carries much less direct
family signal than CDS. Enformer recovers some useful signal from this weaker
input, but it cannot fully recover protein-family labels from regulatory
context alone.

The supervised tracks are weaker than the trunk embeddings. This may be because
the output tracks are compressed toward regulatory predictions, while the trunk
retains broader sequence information.

## TSS Context Findings

### TSS Window Coverage

The Enformer/TSS pipeline successfully cached 3,244 TSS windows and 3,244
feature files. A quick gene-body coverage check showed:

- 81.8% of genes fit fully inside the 196,608 bp TSS-centered window.
- GPCRs fit especially often: 93.9%.
- Immune receptors fit often: 90.3%.
- TFs fit often: 84.7%.
- Ion channels fit less often: 66.2%.
- Kinases fit less often: 62.9%.

This matters because long genes may have coding or regulatory signal outside
the fixed Enformer window. The under-coverage is worst for ion channels and
kinases, which are also among the harder families for Enformer.

### Pilot Runs With Our DNA Encoders On TSS Windows

We tested whether our self-supervised DNA encoders can technically run on the
same TSS windows.

HyenaDNA TSS pilot:

- 10 genes.
- Each 196,608 bp window produced 26 chunks.
- Cached reduction shapes: mean/max/cls = (26, 256) per gene.
- Runtime: 117.9 seconds for 10 genes on MPS.
- Estimated full 3,244-gene runtime: roughly 10 to 11 hours.

NT-v2 TSS pilot:

- 5 genes.
- Each 196,608 bp window produced 36 chunks.
- Cached reduction shapes: mean/max/cls = (36, 512) per gene.
- Runtime: 54.1 seconds for 5 genes on MPS.
- Estimated full 3,244-gene runtime: roughly 9 to 10 hours.

### Why TSS Runs Are Expensive

A 196,608 bp window is much longer than most canonical CDS inputs. Current
tokenization estimates for one TSS window:

- HyenaDNA: about 196,608 tokens, around 26 chunks with the current 8,192-token setup.
- NT-v2: about 32,768 tokens, around 36 chunks with the current 998-token setup.
- DNABERT-2: about 49,154 tokens, many more short chunks.
- GENA-LM: about 49,153 tokens, many more short chunks.

This makes a full TSS encoder run feasible, but overnight-scale on laptop
hardware. NT-v2 or HyenaDNA should be the first full TSS DNA-encoder run.

## Main Conclusions So Far

### 1. The headline result should stay CDS family5 classification

The cleanest result is still family5 classification from CDS features. NT-v2
meanD reaches 0.8275 macro-F1, clearly above CDS 4-mer and the shuffled-label
anti-baseline. This is the most convincing evidence that some DNA encoders
contain gene-family signal beyond simple composition.

### 2. The paper should not claim generic "gene function from sequence"

The Enformer/TSS results show that "sequence" is not a single thing. CDS and
TSS windows ask different biological questions. CDS is protein-domain-rich;
TSS context is regulatory. The benchmark recovers much more signal from CDS.

Better claim:

Frozen DNA encoder representations recover gene-family signal from coding
sequence, and the strength of this recovery depends strongly on architecture,
pretraining, pooling, and genomic context.

### 3. Enformer is useful as a comparator, not a leaderboard row

Enformer should be reported separately as a supervised sequence-to-function
comparator. It helps answer a reviewer question: what happens if the model was
trained with biological supervision and sees regulatory context?

The answer is nuanced:

- Enformer beats the matched TSS 4-mer baseline.
- Enformer does not beat CDS encoders on a CDS/protein-family task.
- Therefore, input context matters as much as model sophistication.

### 4. We need one full self-supervised TSS encoder run

To make the TSS/Enformer argument airtight, the next useful experiment is one
full TSS run with a self-supervised DNA encoder:

- Preferred first choice: NT-v2 on TSS windows.
- Backup/alternative: HyenaDNA on TSS windows.

This would create the clean comparison:

| Input | Model | Question |
|---|---|---|
| CDS | NT-v2 | Can coding-sequence embeddings classify family? |
| TSS | NT-v2 | Does the same encoder recover family from regulatory context? |
| TSS | Enformer | Does supervised regulatory pretraining help on the same context? |
| CDS | 4-mer | How much signal is simple CDS composition? |
| TSS | 4-mer | How much signal is simple TSS composition? |

Possible outcomes:

- If TSS NT-v2 is low like Enformer, the limitation is mostly input context.
- If TSS NT-v2 beats Enformer, self-supervised representations may preserve
  broader sequence information than supervised regulatory outputs.
- If Enformer beats TSS NT-v2, supervised regulatory pretraining helps, but not
  enough to match CDS-based family recovery.

### 5. Pooling geometry is a real result

The model rankings change by pooling choice. NT-v2 works best with meanD/meanG
for classification, DNABERT-2 meanG/meanD is strongest for regression, and
HyenaDNA CLS-style pooling fails. This supports a paper section about how
architecture and tokenization dictate pooling geometry.

### 6. GENA-LM is an important negative model

GENA-LM underperforms the simple CDS 4-mer baseline. This is scientifically
useful because it prevents the paper from sounding like all DNA LMs work
equally well. The result argues for benchmarked representation analysis, not
model-name optimism.

## Suggested Results/Discussion Language

### Conservative Abstract Sentence

Across a 3,244-gene human benchmark, frozen DNA encoders recover 5-way
gene-family signal from coding sequence, with NT-v2 reaching 0.8275 macro-F1
and outperforming a strong CDS 4-mer baseline. However, this signal is highly
dependent on genomic context and model architecture: GENA-LM underperforms
composition, HyenaDNA is competitive but pooling-sensitive, and Enformer
features from TSS-centered regulatory windows recover only partial family
signal despite supervised sequence-to-function pretraining.

### Discussion Point: CDS vs TSS

The contrast between CDS and TSS inputs suggests that this benchmark primarily
measures coding-sequence and protein-domain information rather than generic
gene function from arbitrary genomic context. TSS-centered windows contain
regulatory information and can support partial recovery, but the strongest
family signal remains concentrated in the coding sequence.

### Discussion Point: Enformer

Enformer is best interpreted as a supervised regulatory-context comparator.
Its trunk embeddings improve substantially over TSS-window 4-mer composition,
but its final output tracks are weaker for GenePT regression and do not match
CDS encoder performance. This suggests that regulatory supervision does not
automatically produce representations aligned with protein-family labels.

### Discussion Point: Baselines

The CDS 4-mer baseline is strong enough that every model must be judged against
it. NT-v2 clears this bar convincingly for classification. HyenaDNA clears it
modestly. DNABERT-2 clears it in some settings. GENA-LM does not. This baseline
is therefore not a throwaway control; it defines the real difficulty of the
benchmark.

### Discussion Point: Regression

Ridge-to-GenePT regression should remain secondary. It provides a cross-modal
probe, but its signal is muted by the geometry and noise of text-summary
embeddings. Classification produces a clearer and more interpretable test of
family-level recovery.

## Next Recommended Experiments

1. Run one full TSS self-supervised DNA encoder experiment, preferably NT-v2.
2. Materialize TSS NT-v2 pooling datasets and evaluate family5 plus Ridge.
3. Add a small CDS-vs-TSS ablation table:
   - CDS 4-mer.
   - CDS NT-v2.
   - TSS 4-mer.
   - TSS NT-v2.
   - TSS Enformer trunk.
   - TSS Enformer tracks.
4. Keep binary tasks and length-only rows in appendix/legacy, not the main story.
