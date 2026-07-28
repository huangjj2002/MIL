# DST-Prototype reviewer experiments

## Final method configuration

| Setting | Paper value |
| --- | ---: |
| Prototypes per class, K | 10 |
| Attraction weight, lambda_att | 0.1 |
| Separation weight, lambda_sep | 0.1 |
| Diversity weight, lambda_div | 0.01 |
| Separation hinge margin, gamma_sep | 1.0 |
| Diversity hinge margin, gamma_div | 1.0 |
| Prototype initialization | `fold_best_scores` |
| Prototype normalization | enabled |
| Class-balanced prototype loss | enabled |
| Cross-validation | five patient-level folds |
| Default experiment seed | 10 |
| Epochs per fold | 250 |

`gamma_sep` and `gamma_div` are regularization margins. They are different from
`edl_proto_gamma_init`, which initializes the trainable DST distance sharpness.

`fold_best_scores` restricts candidates to the current fold's training patients.
Within each ground-truth class it prioritizes correctly predicted samples and then
sorts them by original-MIL true-class probability. Validation and held-out test
patients are rejected as prototype sources.

## Run the 9 core reviewer variants

```powershell
python run_embedding_dst_ablation.py `
  --preset reviewer9 `
  --gpu-ids 0 1 2 `
  --embedding-cache-dir <bag-origin-cache> `
  --data-dir <dataset-directory> `
  --csv-file <dataset-csv> `
  --run-root <new-reviewer-run-directory>
```

Use `--dry-run` to validate the cache and write the complete command manifest
without starting training. Use `--resume` with the same run root to skip variants
already marked completed.

`--gpu-ids 0 1 2` runs up to three variants concurrently: each process receives
only its assigned physical GPU through `CUDA_VISIBLE_DEVICES`, and its training
script uses local device `0`. Omit `--gpu-ids` to use the single-GPU shorthand
`--gpu-id 2`.

The core preset contains the full model, K={5,10,20}, removal of each loss,
removal of all regularization, and one lower-margin check for each of gamma_sep
and gamma_div. It therefore needs 45 five-fold trainings. Use `--preset
reviewer17` only if a later response requires the full 85-training sensitivity
scan. All commands explicitly pass every paper parameter; they do not rely on
trainer defaults.

## Analyze completed runs

```powershell
python analyze_embedding_dst_ablation.py `
  --run-root <completed-reviewer-run-directory> `
  --bootstrap-samples 20000 `
  --permutation-samples 20000
```

Optionally add `--mil-validation-csv` and `--mil-test-csv` to include the original
MIL in the component-ablation table. Both files must use the same patient, label,
score, and fold columns as the DST-Prototype prediction exports.

The primary analysis is patient-level paired DeLong for AUROC, supported by
patient-stratified paired bootstrap confidence intervals and patient-level
score-swap permutation tests. Holm adjustment is applied within each prespecified
parameter family. Five-fold exact Wilcoxon results are supplementary only; with
five non-zero paired differences, the smallest attainable two-sided exact p value
is 0.0625.

Do not combine these results with the old `DST_k_10` run whose manifest records
KMeans initialization and zero attraction, separation, and diversity weights.
