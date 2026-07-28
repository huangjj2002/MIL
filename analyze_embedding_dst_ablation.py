"""Analyze embedding-based Prototype-DST ablations at patient level.

Thresholds are selected on validation predictions only and then applied to the
corresponding held-out test predictions.  The script never optimizes a threshold
on the test set.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_curve,
)

from run_embedding_dst_ablation import PRESETS, VARIANT_FAMILIES


METRICS = ("auroc", "auprc", "balanced_accuracy", "sensitivity", "specificity", "f1")


@dataclass
class VariantData:
    name: str
    result_dir: Path
    validation: pd.DataFrame
    test: pd.DataFrame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Patient-level analysis for reviewer-requested Prototype-DST ablations."
    )
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--variants", default="all", help="Comma-separated names, or 'all'.")
    parser.add_argument("--reference", default="full_k10")
    parser.add_argument("--label-col", default="cancer")
    parser.add_argument("--patient-col", default="patient_id")
    parser.add_argument("--score-col", default="prediction_score")
    parser.add_argument("--fold-col", default="fold")
    parser.add_argument("--preset", default=None, choices=sorted(PRESETS))
    parser.add_argument("--bootstrap-samples", default=20000, type=int)
    parser.add_argument("--permutation-samples", default=20000, type=int)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--output-dir", default=None, type=Path)
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--mil-validation-csv", default=None, type=Path)
    parser.add_argument("--mil-test-csv", default=None, type=Path)
    return parser.parse_args()


def select_variant_names(spec: str, preset: str = "reviewer9") -> list[str]:
    ordered = [variant.name for variant in PRESETS[preset]]
    if spec.strip().lower() == "all":
        return ordered
    names = [name.strip() for name in spec.split(",") if name.strip()]
    unknown = sorted(set(names).difference(ordered))
    if unknown:
        raise ValueError(f"Unknown variants: {unknown}. Available: {ordered}")
    if len(names) != len(set(names)):
        raise ValueError("--variants contains duplicate names.")
    return names


def find_result_dir(variant_root: Path) -> Path:
    dev_files = list(variant_root.rglob("*_dst_proto_dev_predictions.csv"))
    if not dev_files:
        raise FileNotFoundError(f"No dev prediction file found below {variant_root}")
    candidates = []
    for dev_path in dev_files:
        test_files = list(dev_path.parent.glob("*_dst_proto_test_all_folds.csv"))
        if len(test_files) == 1:
            candidates.append((dev_path.stat().st_mtime, dev_path.parent))
    if not candidates:
        raise FileNotFoundError(
            f"Found dev predictions below {variant_root}, but no matching test_all_folds file."
        )
    return max(candidates, key=lambda item: item[0])[1]


def read_one_matching(result_dir: Path, pattern: str) -> pd.DataFrame:
    paths = list(result_dir.glob(pattern))
    if len(paths) != 1:
        raise ValueError(f"Expected exactly one {pattern} in {result_dir}, found {len(paths)}.")
    return pd.read_csv(paths[0], dtype={"patient_id": str, "image_id": str})


def load_variant(run_root: Path, name: str, args: argparse.Namespace) -> VariantData:
    result_dir = find_result_dir(run_root / name)
    validation = read_one_matching(result_dir, "*_dst_proto_dev_predictions.csv")
    test = read_one_matching(result_dir, "*_dst_proto_test_all_folds.csv")
    required = {args.patient_col, args.label_col, args.score_col, args.fold_col}
    for split_name, frame in (("validation", validation), ("test", test)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"{name} {split_name} predictions are missing columns: {missing}")
        if frame.empty:
            raise ValueError(f"{name} {split_name} predictions are empty.")
        if frame[list(required)].isna().any().any():
            raise ValueError(f"{name} {split_name} predictions contain null required values.")
    return VariantData(name=name, result_dir=result_dir, validation=validation, test=test)


def load_external_mil(args: argparse.Namespace) -> VariantData | None:
    supplied = (args.mil_validation_csv is not None, args.mil_test_csv is not None)
    if any(supplied) and not all(supplied):
        raise ValueError("--mil-validation-csv and --mil-test-csv must be supplied together.")
    if not any(supplied):
        return None
    validation_path = args.mil_validation_csv.expanduser().resolve()
    test_path = args.mil_test_csv.expanduser().resolve()
    for path in (validation_path, test_path):
        if not path.is_file():
            raise FileNotFoundError(f"MIL prediction CSV not found: {path}")
    validation = pd.read_csv(validation_path, dtype={args.patient_col: str, "image_id": str})
    test = pd.read_csv(test_path, dtype={args.patient_col: str, "image_id": str})
    required = {args.patient_col, args.label_col, args.score_col, args.fold_col}
    for split_name, frame in (("validation", validation), ("test", test)):
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"MIL {split_name} predictions are missing columns: {missing}")
    return VariantData(
        name="mil_baseline",
        result_dir=test_path.parent,
        validation=validation,
        test=test,
    )


def patient_aggregate(frame: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    working = frame[[args.patient_col, args.fold_col, args.label_col, args.score_col]].copy()
    working[args.patient_col] = working[args.patient_col].astype(str)
    working[args.fold_col] = pd.to_numeric(working[args.fold_col], errors="raise").astype(int)
    working[args.label_col] = pd.to_numeric(working[args.label_col], errors="raise").astype(int)
    working[args.score_col] = pd.to_numeric(working[args.score_col], errors="raise").astype(float)
    if not working[args.label_col].isin([0, 1]).all():
        raise ValueError("Only binary labels 0/1 are supported.")
    if not np.isfinite(working[args.score_col]).all():
        raise ValueError("Prediction scores contain NaN or infinite values.")
    if not working[args.score_col].between(0.0, 1.0).all():
        raise ValueError("Prediction scores must lie in [0, 1].")
    return (
        working.groupby([args.patient_col, args.fold_col], as_index=False)
        .agg({args.label_col: "max", args.score_col: "mean"})
        .sort_values([args.fold_col, args.patient_col])
        .reset_index(drop=True)
    )


def youden_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if np.unique(labels).size != 2:
        raise ValueError("Youden threshold requires both classes in validation data.")
    false_positive_rate, true_positive_rate, thresholds = roc_curve(labels, scores)
    finite = np.isfinite(thresholds)
    if not finite.any():
        raise ValueError("No finite validation threshold was produced.")
    objective = true_positive_rate[finite] - false_positive_rate[finite]
    candidate_thresholds = thresholds[finite]
    best = np.flatnonzero(np.isclose(objective, objective.max(), rtol=0.0, atol=1e-12))
    return float(candidate_thresholds[best[-1]])


def binary_auc_auprc(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    positives = int(labels.sum())
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        raise ValueError("AUROC/AUPRC require both classes.")
    order = np.argsort(-scores, kind="mergesort")
    ordered_labels = labels[order]
    ordered_scores = scores[order]
    group_ends = np.r_[ordered_scores[1:] != ordered_scores[:-1], True]
    true_positives = np.cumsum(ordered_labels)[group_ends].astype(float)
    false_positives = np.cumsum(1 - ordered_labels)[group_ends].astype(float)
    true_positive_rate = np.r_[0.0, true_positives / positives]
    false_positive_rate = np.r_[0.0, false_positives / negatives]
    auroc = float(np.trapz(true_positive_rate, false_positive_rate))
    precision = true_positives / (true_positives + false_positives)
    recall = true_positives / positives
    auprc = float(np.sum(np.diff(np.r_[0.0, recall]) * precision))
    return auroc, auprc


def calculate_metrics(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    if np.unique(labels).size != 2:
        raise ValueError("Metrics require both classes.")
    predictions = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if tp + fn else np.nan
    specificity = tn / (tn + fp) if tn + fp else np.nan
    auroc, auprc = binary_auc_auprc(labels, scores)
    return {
        "auroc": auroc,
        "auprc": auprc,
        "balanced_accuracy": float(balanced_accuracy_score(labels, predictions)),
        "sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1": float(f1_score(labels, predictions, zero_division=0)),
    }


def validate_fold_sets(validation: pd.DataFrame, test: pd.DataFrame, args: argparse.Namespace) -> list[int]:
    val_folds = sorted(validation[args.fold_col].unique().tolist())
    test_folds = sorted(test[args.fold_col].unique().tolist())
    if val_folds != test_folds:
        raise ValueError(f"Validation/test fold mismatch: {val_folds} vs {test_folds}")
    if len(val_folds) != 5:
        raise ValueError(f"Standard analysis expects five folds, found {val_folds}.")
    validation_fold_counts = validation.groupby(args.patient_col)[args.fold_col].nunique()
    if not (validation_fold_counts == 1).all():
        raise ValueError("A validation patient appears in more than one out-of-fold split.")
    expected_test_folds = len(test_folds)
    test_fold_counts = test.groupby(args.patient_col)[args.fold_col].nunique()
    if not (test_fold_counts == expected_test_folds).all():
        raise ValueError("Every held-out test patient must have a prediction from every fold.")
    test_label_counts = test.groupby(args.patient_col)[args.label_col].nunique()
    if not (test_label_counts == 1).all():
        raise ValueError("Held-out test patient labels disagree across folds.")
    if set(validation[args.patient_col]).intersection(test[args.patient_col]):
        raise ValueError("Validation and held-out test patient IDs overlap.")
    return val_folds


def assert_same_validation_assignments(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    args: argparse.Namespace,
    reference_name: str,
    current_name: str,
) -> None:
    columns = [args.patient_col, args.fold_col, args.label_col]
    left = reference[columns].sort_values(args.patient_col).reset_index(drop=True)
    right = current[columns].sort_values(args.patient_col).reset_index(drop=True)
    if not left.equals(right):
        raise ValueError(
            f"Validation patient labels/fold assignments for {current_name} do not match "
            f"the reference {reference_name}."
        )


def evaluate_variant(
    data: VariantData,
    args: argparse.Namespace,
) -> tuple[list[dict[str, object]], list[dict[str, object]], dict[str, float], pd.DataFrame]:
    validation = patient_aggregate(data.validation, args)
    test = patient_aggregate(data.test, args)
    folds = validate_fold_sets(validation, test, args)
    fold_rows: list[dict[str, object]] = []
    threshold_rows: list[dict[str, object]] = []
    for fold in folds:
        val_fold = validation[validation[args.fold_col] == fold]
        test_fold = test[test[args.fold_col] == fold]
        threshold = youden_threshold(val_fold[args.label_col].to_numpy(), val_fold[args.score_col].to_numpy())
        metrics = calculate_metrics(
            test_fold[args.label_col].to_numpy(),
            test_fold[args.score_col].to_numpy(),
            threshold,
        )
        fold_rows.append(
            {
                "variant": data.name,
                "fold": fold,
                "threshold_source": "validation_fold",
                "threshold": threshold,
                "n_validation_patients": len(val_fold),
                "n_test_patients": len(test_fold),
                **metrics,
            }
        )
        threshold_rows.append(
            {
                "variant": data.name,
                "scope": f"fold_{fold}",
                "threshold": threshold,
                "source": "validation_only",
                "n_patients": len(val_fold),
            }
        )

    oof_validation = (
        validation.groupby(args.patient_col, as_index=False)
        .agg({args.label_col: "max", args.score_col: "mean"})
        .sort_values(args.patient_col)
    )
    ensemble_test = (
        test.groupby(args.patient_col, as_index=False)
        .agg({args.label_col: "max", args.score_col: "mean"})
        .sort_values(args.patient_col)
        .reset_index(drop=True)
    )
    ensemble_threshold = youden_threshold(
        oof_validation[args.label_col].to_numpy(),
        oof_validation[args.score_col].to_numpy(),
    )
    ensemble_metrics = calculate_metrics(
        ensemble_test[args.label_col].to_numpy(),
        ensemble_test[args.score_col].to_numpy(),
        ensemble_threshold,
    )
    ensemble_metrics.update(
        {
            "variant": data.name,
            "threshold": ensemble_threshold,
            "threshold_source": "pooled_oof_validation",
            "n_validation_patients": len(oof_validation),
            "n_test_patients": len(ensemble_test),
        }
    )
    threshold_rows.append(
        {
            "variant": data.name,
            "scope": "ensemble",
            "threshold": ensemble_threshold,
            "source": "pooled_oof_validation_only",
            "n_patients": len(oof_validation),
        }
    )
    ensemble_test = ensemble_test.rename(
        columns={args.label_col: "label", args.score_col: "score"}
    )
    return fold_rows, threshold_rows, ensemble_metrics, ensemble_test


def summarize_folds(fold_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for variant, variant_frame in fold_metrics.groupby("variant", sort=False):
        for metric in METRICS:
            values = variant_frame[metric].astype(float).to_numpy()
            mean = float(np.mean(values))
            std = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
            half_width = float(stats.t.ppf(0.975, len(values) - 1) * std / math.sqrt(len(values))) if len(values) > 1 else 0.0
            rows.append(
                {
                    "variant": variant,
                    "metric": metric,
                    "mean": mean,
                    "std": std,
                    "ci95_lower": max(0.0, mean - half_width),
                    "ci95_upper": min(1.0, mean + half_width),
                    "n_folds": len(values),
                }
            )
    return pd.DataFrame(rows)


def holm_adjust(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    total = len(p_values)
    for rank, original_index in enumerate(order):
        candidate = min(1.0, (total - rank) * float(p_values[original_index]))
        running = max(running, candidate)
        adjusted[original_index] = running
    return adjusted.tolist()


def analysis_families(variant_names: list[str]) -> dict[str, list[str]]:
    selected = set(variant_names)
    families = {
        family: [name for name in names if name in selected]
        for family, names in VARIANT_FAMILIES.items()
    }
    if "mil_baseline" in selected:
        families.setdefault("component", []).insert(0, "mil_baseline")
    return {family: names for family, names in families.items() if names}


def wilcoxon_comparisons(
    fold_metrics: pd.DataFrame,
    reference: str,
    variant_names: list[str],
) -> pd.DataFrame:
    rows = []
    reference_frame = fold_metrics[fold_metrics["variant"] == reference].set_index("fold")
    for family, family_variants in analysis_families(variant_names).items():
        for metric in METRICS:
            metric_row_indexes = []
            raw_p_values = []
            for variant in family_variants:
                if variant == reference:
                    continue
                current = fold_metrics[fold_metrics["variant"] == variant].set_index("fold")
                if set(current.index) != set(reference_frame.index):
                    raise ValueError(f"Fold IDs for {variant} do not match {reference}.")
                current = current.loc[reference_frame.index]
                differences = current[metric].to_numpy(float) - reference_frame[metric].to_numpy(float)
                nonzero = differences[~np.isclose(differences, 0.0, atol=1e-15)]
                if len(nonzero) == 0:
                    statistic, p_value = 0.0, 1.0
                else:
                    result = stats.wilcoxon(nonzero, alternative="two-sided", method="exact")
                    statistic, p_value = float(result.statistic), float(result.pvalue)
                rows.append(
                    {
                        "family": family,
                        "reference": reference,
                        "variant": variant,
                        "metric": metric,
                        "mean_delta_variant_minus_reference": float(np.mean(differences)),
                        "wilcoxon_statistic": statistic,
                        "p_value_raw": p_value,
                        "n_paired_folds": len(differences),
                        "n_nonzero_differences": len(nonzero),
                        "minimum_two_sided_exact_p_if_all_five_nonzero": 0.0625,
                        "role": "supplementary_exploratory",
                    }
                )
                metric_row_indexes.append(len(rows) - 1)
                raw_p_values.append(p_value)
            adjusted = holm_adjust(raw_p_values)
            for row_index, adjusted_p in zip(metric_row_indexes, adjusted):
                rows[row_index]["p_value_holm_within_family"] = adjusted_p
    return pd.DataFrame(rows)


def align_ensembles(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    patient_col: str,
    reference_name: str,
    current_name: str,
) -> pd.DataFrame:
    reference_ids = set(reference[patient_col])
    current_ids = set(current[patient_col])
    if reference_ids != current_ids:
        raise ValueError(
            f"Test patient mismatch for {current_name} vs {reference_name}: "
            f"missing={len(reference_ids - current_ids)}, extra={len(current_ids - reference_ids)}"
        )
    merged = reference.merge(
        current,
        on=patient_col,
        how="inner",
        validate="one_to_one",
        suffixes=("_reference", "_variant"),
    ).sort_values(patient_col)
    if not np.array_equal(merged["label_reference"].to_numpy(), merged["label_variant"].to_numpy()):
        raise ValueError(f"Test labels disagree for {current_name} vs {reference_name}.")
    return merged


def compute_midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    midranks = np.zeros(len(values), dtype=float)
    start = 0
    while start < len(values):
        stop = start + 1
        while stop < len(values) and sorted_values[stop] == sorted_values[start]:
            stop += 1
        midranks[start:stop] = 0.5 * (start + stop - 1)
        start = stop
    result = np.empty(len(values), dtype=float)
    result[order] = midranks + 1.0
    return result


def fast_delong(predictions_sorted_by_label: np.ndarray, positive_count: int):
    classifiers, total = predictions_sorted_by_label.shape
    positive_count = int(positive_count)
    negative_count = total - positive_count
    if positive_count < 2 or negative_count < 2:
        raise ValueError("Paired DeLong requires at least two positive and two negative patients.")
    positive = predictions_sorted_by_label[:, :positive_count]
    negative = predictions_sorted_by_label[:, positive_count:]
    tx = np.empty((classifiers, positive_count), dtype=float)
    ty = np.empty((classifiers, negative_count), dtype=float)
    tz = np.empty((classifiers, total), dtype=float)
    for row in range(classifiers):
        tx[row] = compute_midrank(positive[row])
        ty[row] = compute_midrank(negative[row])
        tz[row] = compute_midrank(predictions_sorted_by_label[row])
    aucs = (
        tz[:, :positive_count].sum(axis=1) / (positive_count * negative_count)
        - (positive_count + 1.0) / (2.0 * negative_count)
    )
    v01 = (tz[:, :positive_count] - tx) / negative_count
    v10 = 1.0 - (tz[:, positive_count:] - ty) / positive_count
    covariance = (
        np.atleast_2d(np.cov(v01, bias=False)) / positive_count
        + np.atleast_2d(np.cov(v10, bias=False)) / negative_count
    )
    return aucs, covariance


def paired_delong(labels: np.ndarray, variant_scores: np.ndarray, reference_scores: np.ndarray):
    labels = np.asarray(labels, dtype=int)
    order = np.argsort(-labels, kind="stable")
    predictions = np.vstack([variant_scores, reference_scores])[:, order]
    aucs, covariance = fast_delong(predictions, int(labels.sum()))
    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast)
    delta = float(aucs[0] - aucs[1])
    if variance <= np.finfo(float).eps:
        z_value = 0.0 if abs(delta) <= 1e-15 else math.copysign(math.inf, delta)
        p_value = 1.0 if abs(delta) <= 1e-15 else 0.0
    else:
        z_value = delta / math.sqrt(variance)
        p_value = float(2.0 * stats.norm.sf(abs(z_value)))
    return {
        "variant_auc_delong": float(aucs[0]),
        "reference_auc_delong": float(aucs[1]),
        "delta_auroc_variant_minus_reference": delta,
        "delong_variance_delta": variance,
        "delong_z": z_value,
        "p_delong_raw": p_value,
    }


def bootstrap_comparison(
    merged: pd.DataFrame,
    reference_threshold: float,
    variant_threshold: float,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    labels = merged["label_reference"].to_numpy(int)
    reference_scores = merged["score_reference"].to_numpy(float)
    variant_scores = merged["score_variant"].to_numpy(float)
    class_indexes = [np.flatnonzero(labels == label) for label in (0, 1)]
    if any(len(indexes) == 0 for indexes in class_indexes):
        raise ValueError("Paired bootstrap requires both test classes.")
    deltas = {
        metric: np.empty(n_bootstrap, dtype=float)
        for metric in ("auroc", "auprc", "balanced_accuracy")
    }
    for bootstrap_index in range(n_bootstrap):
        sampled = np.concatenate(
            [rng.choice(indexes, size=len(indexes), replace=True) for indexes in class_indexes]
        )
        sampled_labels = labels[sampled]
        reference_metrics = calculate_metrics(sampled_labels, reference_scores[sampled], reference_threshold)
        variant_metrics = calculate_metrics(sampled_labels, variant_scores[sampled], variant_threshold)
        for metric in deltas:
            deltas[metric][bootstrap_index] = variant_metrics[metric] - reference_metrics[metric]
    return deltas


def score_swap_permutation(
    labels: np.ndarray,
    reference_scores: np.ndarray,
    variant_scores: np.ndarray,
    n_permutations: int,
    rng: np.random.Generator,
) -> dict[str, float]:
    variant_auc, variant_auprc = binary_auc_auprc(labels, variant_scores)
    reference_auc, reference_auprc = binary_auc_auprc(labels, reference_scores)
    observed_auc = variant_auc - reference_auc
    observed_auprc = variant_auprc - reference_auprc
    extreme_auc = 0
    extreme_auprc = 0
    for _ in range(n_permutations):
        swap = rng.integers(0, 2, size=len(labels), dtype=np.int8).astype(bool)
        perm_variant = np.where(swap, reference_scores, variant_scores)
        perm_reference = np.where(swap, variant_scores, reference_scores)
        perm_variant_auc, perm_variant_auprc = binary_auc_auprc(labels, perm_variant)
        perm_reference_auc, perm_reference_auprc = binary_auc_auprc(labels, perm_reference)
        delta_auc = perm_variant_auc - perm_reference_auc
        delta_auprc = perm_variant_auprc - perm_reference_auprc
        extreme_auc += int(abs(delta_auc) >= abs(observed_auc) - 1e-15)
        extreme_auprc += int(abs(delta_auprc) >= abs(observed_auprc) - 1e-15)
    return {
        "p_permutation_auroc_raw": (extreme_auc + 1) / (n_permutations + 1),
        "p_permutation_auprc_raw": (extreme_auprc + 1) / (n_permutations + 1),
    }


def paired_patient_tests(
    ensembles: dict[str, pd.DataFrame],
    ensemble_metrics: pd.DataFrame,
    reference: str,
    variant_names: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    if args.bootstrap_samples < 100 or args.permutation_samples < 100:
        raise ValueError("Bootstrap and permutation sample counts must each be at least 100.")
    thresholds = ensemble_metrics.set_index("variant")["threshold"].to_dict()
    metrics_by_variant = ensemble_metrics.set_index("variant")
    base_rows = {}
    unique_variants = [name for name in dict.fromkeys(variant_names) if name != reference]
    for variant_index, variant in enumerate(unique_variants):
        merged = align_ensembles(
            ensembles[reference], ensembles[variant], args.patient_col, reference, variant
        )
        labels = merged["label_reference"].to_numpy(int)
        reference_scores = merged["score_reference"].to_numpy(float)
        variant_scores = merged["score_variant"].to_numpy(float)
        bootstrap_rng = np.random.default_rng(args.seed + 1000 * (variant_index + 1))
        deltas = bootstrap_comparison(
            merged,
            float(thresholds[reference]),
            float(thresholds[variant]),
            args.bootstrap_samples,
            bootstrap_rng,
        )
        permutation = score_swap_permutation(
            labels,
            reference_scores,
            variant_scores,
            args.permutation_samples,
            np.random.default_rng(args.seed + 100000 + 1000 * (variant_index + 1)),
        )
        row = {
            "reference": reference,
            "variant": variant,
            "n_test_patients": len(merged),
            **paired_delong(labels, variant_scores, reference_scores),
            "delta_auprc_variant_minus_reference": float(
                metrics_by_variant.loc[variant, "auprc"]
                - metrics_by_variant.loc[reference, "auprc"]
            ),
            "delta_balanced_accuracy_variant_minus_reference": float(
                metrics_by_variant.loc[variant, "balanced_accuracy"]
                - metrics_by_variant.loc[reference, "balanced_accuracy"]
            ),
            "delta_auroc_ci95_low": float(np.quantile(deltas["auroc"], 0.025)),
            "delta_auroc_ci95_high": float(np.quantile(deltas["auroc"], 0.975)),
            "delta_auprc_ci95_low": float(np.quantile(deltas["auprc"], 0.025)),
            "delta_auprc_ci95_high": float(np.quantile(deltas["auprc"], 0.975)),
            "delta_balanced_accuracy_ci95_low": float(
                np.quantile(deltas["balanced_accuracy"], 0.025)
            ),
            "delta_balanced_accuracy_ci95_high": float(
                np.quantile(deltas["balanced_accuracy"], 0.975)
            ),
            "bootstrap_samples": args.bootstrap_samples,
            "permutation_samples": args.permutation_samples,
            **permutation,
        }
        base_rows[variant] = row

    rows = []
    for family, names in analysis_families(variant_names).items():
        family_indexes = []
        for name in names:
            if name not in base_rows:
                continue
            rows.append({"family": family, **base_rows[name]})
            family_indexes.append(len(rows) - 1)
        for p_column, adjusted_column in (
            ("p_delong_raw", "p_delong_holm_within_family"),
            ("p_permutation_auroc_raw", "p_permutation_auroc_holm_within_family"),
            ("p_permutation_auprc_raw", "p_permutation_auprc_holm_within_family"),
        ):
            adjusted = holm_adjust([rows[index][p_column] for index in family_indexes])
            for index, value in zip(family_indexes, adjusted):
                rows[index][adjusted_column] = value
    return pd.DataFrame(rows)


def patient_tests_to_bootstrap_long(patient_tests: pd.DataFrame) -> pd.DataFrame:
    unique = patient_tests.drop_duplicates("variant", keep="first")
    rows = []
    mappings = (
        ("auroc", "delta_auroc_variant_minus_reference", "delta_auroc_ci95_low", "delta_auroc_ci95_high"),
        ("auprc", "delta_auprc_variant_minus_reference", "delta_auprc_ci95_low", "delta_auprc_ci95_high"),
        (
            "balanced_accuracy",
            "delta_balanced_accuracy_variant_minus_reference",
            "delta_balanced_accuracy_ci95_low",
            "delta_balanced_accuracy_ci95_high",
        ),
    )
    for _, row in unique.iterrows():
        for metric, point_col, low_col, high_col in mappings:
            rows.append({
                "reference": row["reference"],
                "variant": row["variant"],
                "metric": metric,
                "point_delta_variant_minus_reference": row[point_col],
                "ci95_lower": row[low_col],
                "ci95_upper": row[high_col],
                "bootstrap_samples": row["bootstrap_samples"],
                "n_test_patients": row["n_test_patients"],
            })
    return pd.DataFrame(rows)


def bootstrap_comparisons(
    ensembles: dict[str, pd.DataFrame],
    ensemble_metrics: pd.DataFrame,
    reference: str,
    variant_names: list[str],
    args: argparse.Namespace,
) -> pd.DataFrame:
    if args.bootstrap_samples < 100:
        raise ValueError("--bootstrap-samples must be at least 100.")
    rng = np.random.default_rng(args.seed)
    thresholds = ensemble_metrics.set_index("variant")["threshold"].to_dict()
    rows = []
    for variant in variant_names:
        if variant == reference:
            continue
        merged = align_ensembles(
            ensembles[reference], ensembles[variant], args.patient_col, reference, variant
        )
        deltas = bootstrap_comparison(
            merged,
            float(thresholds[reference]),
            float(thresholds[variant]),
            args.bootstrap_samples,
            rng,
        )
        for metric, values in deltas.items():
            below_or_equal = int(np.sum(values <= 0.0))
            above_or_equal = int(np.sum(values >= 0.0))
            p_value = min(
                1.0,
                2.0 * min(below_or_equal + 1, above_or_equal + 1) / (len(values) + 1),
            )
            rows.append(
                {
                    "reference": reference,
                    "variant": variant,
                    "metric": metric,
                    "point_delta_variant_minus_reference": float(
                        ensemble_metrics.set_index("variant").loc[variant, metric]
                        - ensemble_metrics.set_index("variant").loc[reference, metric]
                    ),
                    "bootstrap_mean_delta": float(np.mean(values)),
                    "ci95_lower": float(np.quantile(values, 0.025)),
                    "ci95_upper": float(np.quantile(values, 0.975)),
                    "p_value_two_sided": p_value,
                    "bootstrap_samples": len(values),
                    "n_test_patients": len(merged),
                }
            )
    result = pd.DataFrame(rows)
    for metric in result["metric"].unique():
        indexes = result.index[result["metric"] == metric].tolist()
        adjusted = holm_adjust(result.loc[indexes, "p_value_two_sided"].tolist())
        result.loc[indexes, "p_value_holm"] = adjusted
    return result


def make_plots(
    summary: pd.DataFrame,
    bootstrap: pd.DataFrame,
    variant_names: list[str],
    output_dir: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = {"auroc": "#3569A8", "balanced_accuracy": "#D39B2A"}
    plots = []

    selected = summary[summary["metric"].isin(colors)].copy()
    pivot_mean = selected.pivot(index="variant", columns="metric", values="mean").loc[variant_names]
    pivot_std = selected.pivot(index="variant", columns="metric", values="std").loc[variant_names]
    x = np.arange(len(variant_names))
    width = 0.36
    fig, axis = plt.subplots(figsize=(12, 6.5))
    for offset, metric, hatch in ((-width / 2, "auroc", ""), (width / 2, "balanced_accuracy", "//")):
        axis.bar(
            x + offset,
            pivot_mean[metric],
            width,
            yerr=pivot_std[metric],
            capsize=3,
            label=metric.replace("_", " ").title(),
            color=colors[metric],
            edgecolor="#27313D",
            linewidth=0.6,
            hatch=hatch,
        )
    axis.set_title("Prototype-DST reviewer ablations")
    axis.set_ylabel("Five-fold test metric (mean ± SD)")
    axis.set_ylim(0.0, 1.0)
    axis.set_xticks(x, variant_names, rotation=35, ha="right")
    axis.grid(axis="y", color="#D8DEE6", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    axis.legend(frameon=False, ncol=2, loc="upper center")
    fig.tight_layout()
    path = output_dir / "ablation_auroc_bacc.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    plots.append(path)

    delta = bootstrap[bootstrap["metric"] == "auroc"].set_index("variant").loc[
        [name for name in variant_names if name != "full_k10"]
    ]
    y = np.arange(len(delta))
    values = delta["point_delta_variant_minus_reference"].to_numpy(float)
    lower = delta["ci95_lower"].to_numpy(float)
    upper = delta["ci95_upper"].to_numpy(float)
    fig, axis = plt.subplots(figsize=(9, 5.5))
    axis.hlines(y, lower, upper, color="#27313D", linewidth=1.3)
    axis.plot(lower, y, "|", color="#27313D", markersize=8)
    axis.plot(upper, y, "|", color="#27313D", markersize=8)
    axis.scatter(values, y, color="#3569A8", edgecolor="#27313D", linewidth=0.5, zorder=3)
    axis.axvline(0.0, color="#27313D", linewidth=1.0, linestyle="--")
    axis.set_yticks(y, delta.index)
    axis.invert_yaxis()
    axis.set_title("AUROC difference versus full_k10")
    axis.set_xlabel("Patient-level ensemble AUROC difference (95% paired bootstrap CI)")
    axis.grid(axis="x", color="#D8DEE6", linewidth=0.7)
    axis.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    path = output_dir / "ablation_auroc_delta_vs_full.png"
    fig.savefig(path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    plots.append(path)
    return plots


def write_markdown(
    output_path: Path,
    ensemble: pd.DataFrame,
    summary: pd.DataFrame,
    wilcoxon: pd.DataFrame,
    bootstrap: pd.DataFrame,
    patient_tests: pd.DataFrame,
    reference: str,
) -> None:
    columns = [
        "variant",
        "auroc",
        "auprc",
        "balanced_accuracy",
        "sensitivity",
        "specificity",
        "f1",
        "threshold",
    ]
    table = ensemble[columns].copy()
    for column in columns[1:]:
        table[column] = table[column].map(lambda value: f"{value:.4f}")
    significant_wilcoxon = wilcoxon[
        wilcoxon["p_value_holm_within_family"] < 0.05
    ]
    significant_delong = patient_tests[
        patient_tests["p_delong_holm_within_family"] < 0.05
    ]
    header = "| " + " | ".join(table.columns) + " |"
    separator = "| " + " | ".join(["---"] * len(table.columns)) + " |"
    body = [
        "| " + " | ".join(str(value) for value in row) + " |"
        for row in table.itertuples(index=False, name=None)
    ]
    lines = [
        "# Embedding Prototype-DST Ablation Results",
        "",
        "## Evaluation contract",
        "",
        "- Image scores are averaged per patient; patient labels use the maximum image label.",
        "- Each fold threshold is selected by Youden's J on that fold's validation patients only.",
        "- Ensemble threshold is selected from pooled out-of-fold validation patients only.",
        "- The held-out test set is never used to choose a threshold.",
        "- Fold uncertainty is reported as mean, sample SD, and t-based 95% CI.",
        f"- Primary paired tests compare each variant with `{reference}` on the same test patients.",
        "- AUROC uses paired DeLong; AUROC/AUPRC differences use stratified paired bootstrap CIs.",
        "- Patient-level score-swap permutation is a robustness analysis.",
        "- Holm correction is applied within each prespecified parameter/component family.",
        "",
        "## Patient-level ensemble metrics",
        "",
        header,
        separator,
        *body,
        "",
        "## Statistical checks",
        "",
        f"Holm-significant paired DeLong comparisons: {len(significant_delong)}.",
        f"Holm-significant exact fold-Wilcoxon comparisons: {len(significant_wilcoxon)}.",
        "The fold Wilcoxon analysis is supplementary and exploratory. With five non-zero paired "
        "differences, its minimum attainable two-sided exact p value is 0.0625.",
        "",
        "Detailed fold metrics, confidence intervals, thresholds, and paired-test outputs are saved as CSV files in this directory.",
        "",
        "## Interpretation guardrail",
        "",
        "A non-significant result is evidence of insufficient detectable difference under this five-fold design; it is not proof that two variants are equivalent.",
    ]
    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    if not run_root.is_dir():
        raise FileNotFoundError(f"Run root not found: {run_root}")
    run_manifest_path = run_root / "run_manifest.json"
    run_manifest = (
        json.loads(run_manifest_path.read_text(encoding="utf-8"))
        if run_manifest_path.is_file()
        else {}
    )
    preset = args.preset or run_manifest.get("preset", "reviewer9")
    variant_names = select_variant_names(args.variants, preset)
    if len(variant_names) < 2:
        raise ValueError("Analysis requires the reference and at least one comparison variant.")
    if args.reference not in variant_names:
        raise ValueError(f"Reference {args.reference!r} must be included in --variants.")
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir is not None
        else run_root / "analysis"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    all_fold_rows = []
    all_threshold_rows = []
    all_ensemble_rows = []
    ensembles: dict[str, pd.DataFrame] = {}
    validation_assignments: dict[str, pd.DataFrame] = {}
    sources = {}
    variant_data = [load_variant(run_root, name, args) for name in variant_names]
    external_mil = load_external_mil(args)
    if external_mil is not None:
        variant_data.append(external_mil)
        variant_names.append(external_mil.name)
    for data in variant_data:
        name = data.name
        fold_rows, threshold_rows, ensemble_row, ensemble_predictions = evaluate_variant(data, args)
        all_fold_rows.extend(fold_rows)
        all_threshold_rows.extend(threshold_rows)
        all_ensemble_rows.append(ensemble_row)
        ensembles[name] = ensemble_predictions
        validation_assignments[name] = patient_aggregate(data.validation, args)
        sources[name] = str(data.result_dir)
        print(f"[{name}] loaded and evaluated from {data.result_dir}")

    for name in variant_names:
        if name != args.reference:
            assert_same_validation_assignments(
                validation_assignments[args.reference],
                validation_assignments[name],
                args,
                args.reference,
                name,
            )

    fold_metrics = pd.DataFrame(all_fold_rows)
    thresholds = pd.DataFrame(all_threshold_rows)
    ensemble_metrics = pd.DataFrame(all_ensemble_rows)
    fold_summary = summarize_folds(fold_metrics)
    wilcoxon = wilcoxon_comparisons(fold_metrics, args.reference, variant_names)
    patient_tests = paired_patient_tests(
        ensembles,
        ensemble_metrics,
        args.reference,
        variant_names,
        args,
    )
    bootstrap = patient_tests_to_bootstrap_long(patient_tests)

    variant_configs = pd.DataFrame(
        [{"variant": variant.name, **variant.__dict__} for variant in PRESETS[preset]]
    )
    parameter_sensitivity = variant_configs.merge(
        ensemble_metrics,
        on="variant",
        how="inner",
        validate="one_to_one",
    )
    component_order = [
        "mil_baseline",
        "no_regularization",
        args.reference,
        "no_attraction",
        "no_separation",
        "no_diversity",
    ]
    component_ablation = ensemble_metrics[
        ensemble_metrics["variant"].isin(component_order)
    ].copy()
    component_ablation["display_order"] = component_ablation["variant"].map(
        {name: index for index, name in enumerate(component_order)}
    )
    component_ablation = component_ablation.sort_values("display_order").drop(
        columns="display_order"
    )

    fold_metrics.to_csv(output_dir / "fold_metrics.csv", index=False)
    fold_summary.to_csv(output_dir / "fold_metrics_summary.csv", index=False)
    ensemble_metrics.to_csv(output_dir / "ensemble_metrics.csv", index=False)
    thresholds.to_csv(output_dir / "validation_thresholds.csv", index=False)
    wilcoxon.to_csv(output_dir / "paired_wilcoxon_vs_full.csv", index=False)
    bootstrap.to_csv(output_dir / "paired_bootstrap_vs_full.csv", index=False)
    patient_tests.to_csv(output_dir / "patient_primary_tests.csv", index=False)
    parameter_sensitivity.to_csv(
        output_dir / "parameter_sensitivity_metrics.csv", index=False
    )
    component_ablation.to_csv(output_dir / "component_ablation_metrics.csv", index=False)
    for name, frame in ensembles.items():
        frame.assign(variant=name).to_csv(
            output_dir / f"{name}_patient_test_ensemble.csv", index=False
        )

    plots = [] if args.no_plots else make_plots(fold_summary, bootstrap, variant_names, output_dir)
    write_markdown(
        output_dir / "ablation_results.md",
        ensemble_metrics,
        fold_summary,
        wilcoxon,
        bootstrap,
        patient_tests,
        args.reference,
    )
    manifest = {
        "schema_version": 2,
        "run_root": str(run_root),
        "output_dir": str(output_dir),
        "preset": preset,
        "variants": variant_names,
        "reference": args.reference,
        "sources": sources,
        "patient_aggregation": {"label": "max", "prediction_score": "mean"},
        "threshold_selection": "Youden J on validation only",
        "bootstrap_samples": args.bootstrap_samples,
        "permutation_samples": args.permutation_samples,
        "seed": args.seed,
        "primary_unit": "patient",
        "primary_test": "two-sided paired DeLong for AUROC",
        "multiple_testing": "Holm correction within each prespecified family",
        "wilcoxon_role": (
            "supplementary exploratory exact two-sided fold comparison; "
            "minimum p=0.0625 for five non-zero paired differences"
        ),
        "analysis_families": analysis_families(variant_names),
        "plots": [str(path) for path in plots],
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"Analysis complete: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
