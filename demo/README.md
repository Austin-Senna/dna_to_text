# Zero-shot demo

Picks 4 test-set genes (2 well-characterised by summary length, 2 poorly
characterised), predicts each one's family with the headline NT-v2 + meanD
logistic probe, and surfaces the top-5 nearest neighbour genes by NT-v2 +
meanD embedding cosine.

Output is a markdown table at `demo/output.md` ready to drop into the deck.

Generate: `uv run python demo/zero_shot.py`

Latest run: 4/4 predictions correct, including both poorly-characterised
zinc-finger genes (`ZNF839`, `ZNHIT2`) whose only NCBI annotation is
"Predicted to enable metal ion binding activity" — the probe correctly
predicts `tf` and the top-5 neighbours are TFs with zinc-finger / DNA-
binding annotations.
