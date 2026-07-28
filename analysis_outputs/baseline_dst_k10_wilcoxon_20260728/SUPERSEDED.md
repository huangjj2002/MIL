# SUPERSEDED — do not use for reporting

This directory used an incorrect Mammo-CLIP-based DST path and compared nominal training folds with Wilcoxon signed-rank tests. Those results and conclusions are withdrawn.

The corrected analysis uses the GLAM-embedding DST-Prototype k=10 source:

`G:\611\glam\proto_embedding_rerun\DST_k_10`

Use the replacement outputs in:

`G:\Final_MIL\code\analysis_outputs\correct_dst_k10_subject_level_20260728`

The replacement primary inference is paired on the same independent test patients using correlated-ROC DeLong tests with Holm correction, supported by paired bootstrap confidence intervals and score-swap permutation checks. Training-fold identity is therefore not required.
