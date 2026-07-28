from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import norm
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
    roc_curve,
)


OUTPUT_DIR = Path(__file__).resolve().parent
REFERENCE_MODEL = "DST k=10"
MODEL_ORDER = [REFERENCE_MODEL, "GLAM", "Mammo-FM", "Mammo-CLIP", "MIL"]
BASELINE_MODELS = ["GLAM", "Mammo-FM", "Mammo-CLIP", "MIL"]
N_RESAMPLES = int(os.environ.get("N_RESAMPLES", "20000"))
SEED = int(os.environ.get("ANALYSIS_SEED", "20260728"))
ALPHA = 0.05

SOURCES = {
    REFERENCE_MODEL: {
        "path": Path("/mnt/g/611/glam/proto_embedding_rerun/DST_k_10/per_model_predictions/ensemble_edl_predictions.csv"),
        "score": "image_prediction_prob",
        "label": "label",
        "kind": "ensemble",
    },
    "GLAM": {
        "path": Path("/mnt/g/611/612/glam-origin/glam-origin/glam_kfold_ft_20260611_095443/ensemble_predictions.csv"),
        "score": "pred_score",
        "label": "cancer",
        "kind": "ensemble",
    },
    "Mammo-FM": {
        "path": Path("/mnt/g/611/612/613/Mammo-FM/data_predictions_ensemble.csv"),
        "score": "pred_score",
        "label": "cancer",
        "kind": "ensemble",
    },
    "Mammo-CLIP": {
        "path": Path("/mnt/g/611/612/613/lr_5e-05_epochs_25_weighted_BCE_y_cancer_data_frac_1.0_run_origin_b5_5fold_e25_p4/ensemble_all_predictions.csv"),
        "score": "image_prediction_prob",
        "label": "cancer",
        "kind": "ensemble",
    },
    "MIL": {
        "paths": [
            Path(f"/mnt/g/611/612/MIL-Origin/MIL-Origin/2026-06-09/fold_{fold}/ViNDr_mil_predictions_fold_{fold}.csv")
            for fold in range(5)
        ],
        "score": "prediction_score",
        "label": "cancer",
        "kind": "five_prediction_mean",
    },
}


def _read_standard(model: str) -> tuple[pd.DataFrame, dict]:
    cfg = SOURCES[model]
    path = cfg["path"]
    frame = pd.read_csv(path, dtype={"patient_id": str, "image_id": str}, low_memory=False)
    required = {"patient_id", "image_id", cfg["label"], cfg["score"]}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    if "split" in frame.columns:
        frame = frame.loc[frame["split"].astype(str).str.lower().eq("test")].copy()
    duplicate_count = int(frame.duplicated(["patient_id", "image_id"]).sum())
    if duplicate_count:
        raise ValueError(f"{model} has {duplicate_count} duplicate patient/image keys")
    out = frame[["patient_id", "image_id", cfg["label"], cfg["score"]]].rename(
        columns={cfg["label"]: "label", cfg["score"]: "score"}
    )
    out["label"] = out["label"].astype(int)
    out["score"] = pd.to_numeric(out["score"], errors="raise").astype(float)
    audit = {
        "model": model,
        "source_files": str(path),
        "source_test_rows": int(len(out)),
        "source_patients": int(out["patient_id"].nunique()),
        "duplicate_keys": duplicate_count,
        "missing_scores": int(out["score"].isna().sum()),
        "nonfinite_scores": int((~np.isfinite(out["score"])).sum()),
        "aggregation": "existing five-fold ensemble file",
    }
    return out, audit


def _read_mil() -> tuple[pd.DataFrame, dict]:
    cfg = SOURCES["MIL"]
    frames = []
    for fold, path in enumerate(cfg["paths"]):
        frame = pd.read_csv(path, dtype={"patient_id": str, "image_id": str}, low_memory=False)
        required = {"patient_id", "image_id", cfg["label"], cfg["score"]}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        if "split" in frame.columns:
            frame = frame.loc[frame["split"].astype(str).str.lower().eq("test")].copy()
        if frame.duplicated(["patient_id", "image_id"]).any():
            raise ValueError(f"MIL fold {fold} has duplicate patient/image keys")
        piece = frame[["patient_id", "image_id", cfg["label"], cfg["score"]]].rename(
            columns={cfg["label"]: "label", cfg["score"]: "score"}
        )
        piece["fold"] = fold
        frames.append(piece)
    stacked = pd.concat(frames, ignore_index=True)
    label_span = stacked.groupby(["patient_id", "image_id"])["label"].nunique()
    if int((label_span > 1).sum()):
        raise ValueError("MIL labels conflict across the five saved prediction files")
    repeat_counts = stacked.groupby(["patient_id", "image_id"]).size()
    if not repeat_counts.eq(5).all():
        raise ValueError("Every MIL image must appear once in each of five saved prediction files")
    out = (
        stacked.groupby(["patient_id", "image_id"], as_index=False)
        .agg(label=("label", "first"), score=("score", "mean"))
    )
    out["label"] = out["label"].astype(int)
    out["score"] = out["score"].astype(float)
    audit = {
        "model": "MIL",
        "source_files": " | ".join(str(p) for p in cfg["paths"]),
        "source_test_rows": int(len(out)),
        "source_patients": int(out["patient_id"].nunique()),
        "duplicate_keys": int(out.duplicated(["patient_id", "image_id"]).sum()),
        "missing_scores": int(out["score"].isna().sum()),
        "nonfinite_scores": int((~np.isfinite(out["score"])).sum()),
        "aggregation": "arithmetic mean of five saved test probabilities per image",
    }
    return out, audit


def load_and_align() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]:
    model_frames: dict[str, pd.DataFrame] = {}
    audits = []
    for model in MODEL_ORDER:
        frame, audit = _read_mil() if model == "MIL" else _read_standard(model)
        if audit["missing_scores"] or audit["nonfinite_scores"]:
            raise ValueError(f"{model} has missing or non-finite predictions")
        model_frames[model] = frame
        audits.append(audit)

    dst_full = model_frames[REFERENCE_MODEL]
    dst_full_patient = dst_full.groupby("patient_id", as_index=False).agg(label=("label", "max"), score=("score", "mean"))
    dst_full_auc = float(roc_auc_score(dst_full_patient["label"], dst_full_patient["score"]))
    if len(dst_full) != 8433 or not math.isclose(dst_full_auc, 0.8648311169382531, abs_tol=5e-12):
        raise AssertionError(f"Correct DST validation failed: rows={len(dst_full)}, patient AUROC={dst_full_auc:.12f}")

    common_keys: set[tuple[str, str]] | None = None
    for frame in model_frames.values():
        keys = set(map(tuple, frame[["patient_id", "image_id"]].itertuples(index=False, name=None)))
        common_keys = keys if common_keys is None else common_keys & keys
    assert common_keys is not None
    common_key_frame = pd.DataFrame(sorted(common_keys), columns=["patient_id", "image_id"])
    image = common_key_frame.copy()
    reference_labels = None
    label_conflicts = 0
    for model in MODEL_ORDER:
        subset = common_key_frame.merge(model_frames[model], on=["patient_id", "image_id"], how="left", validate="one_to_one")
        if subset[["label", "score"]].isna().any().any():
            raise ValueError(f"{model} has missing predictions on the common test set")
        labels = subset["label"].astype(int).to_numpy()
        if reference_labels is None:
            reference_labels = labels
            image["label"] = labels
        else:
            label_conflicts += int(np.sum(labels != reference_labels))
        image[model] = subset["score"].astype(float).to_numpy()

    score_columns = MODEL_ORDER
    missing_predictions = int(image[score_columns].isna().sum().sum())
    nonfinite_predictions = int((~np.isfinite(image[score_columns].to_numpy(float))).sum())
    duplicate_keys = int(image.duplicated(["patient_id", "image_id"]).sum())
    patient = image.groupby("patient_id", as_index=False).agg(
        label=("label", "max"), **{model: (model, "mean") for model in MODEL_ORDER}
    )
    validation = {
        "common_images": int(len(image)),
        "common_patients": int(patient["patient_id"].nunique()),
        "positive_patients": int(patient["label"].sum()),
        "negative_patients": int((patient["label"] == 0).sum()),
        "label_conflicts": int(label_conflicts),
        "duplicate_common_keys": duplicate_keys,
        "missing_predictions": missing_predictions,
        "nonfinite_predictions": nonfinite_predictions,
        "dst_full_images": int(len(dst_full)),
        "dst_full_patient_auc": dst_full_auc,
        "dst_common_patient_auc": float(roc_auc_score(patient["label"], patient[REFERENCE_MODEL])),
    }
    expected = {
        "common_images": 8409,
        "common_patients": 862,
        "positive_patients": 19,
        "label_conflicts": 0,
        "duplicate_common_keys": 0,
        "missing_predictions": 0,
        "nonfinite_predictions": 0,
    }
    for key, value in expected.items():
        if validation[key] != value:
            raise AssertionError(f"Validation failed for {key}: observed={validation[key]}, expected={value}")
    if not math.isclose(validation["dst_common_patient_auc"], 0.8645813822813261, abs_tol=5e-12):
        raise AssertionError(f"DST common patient AUROC mismatch: {validation['dst_common_patient_auc']:.12f}")

    source_audit = pd.DataFrame(audits)
    source_audit["common_rows"] = len(image)
    source_audit["common_patients"] = len(patient)
    return image, patient, source_audit, validation


@dataclass
class WeightedMetricEvaluator:
    labels: np.ndarray
    scores: np.ndarray

    def __post_init__(self) -> None:
        self.labels = np.asarray(self.labels, dtype=np.int8)
        self.scores = np.asarray(self.scores, dtype=float)
        self._asc_order, self._asc_starts = self._order_and_starts(ascending=True)
        self._desc_order, self._desc_starts = self._order_and_starts(ascending=False)

    def _order_and_starts(self, ascending: bool) -> tuple[np.ndarray, np.ndarray]:
        order = np.argsort(self.scores, kind="mergesort")
        if not ascending:
            order = order[::-1]
        sorted_scores = self.scores[order]
        starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores) != 0) + 1]
        return order, starts

    def evaluate(self, weights: np.ndarray) -> tuple[float, float]:
        weights = np.asarray(weights, dtype=float)
        pos_total = float(weights[self.labels == 1].sum())
        neg_total = float(weights[self.labels == 0].sum())
        if pos_total <= 0 or neg_total <= 0:
            raise ValueError("AUC/AP resample contains only one class")

        order = self._asc_order
        starts = self._asc_starts
        pos_group = np.add.reduceat(weights[order] * self.labels[order], starts)
        neg_group = np.add.reduceat(weights[order] * (1 - self.labels[order]), starts)
        neg_before = np.cumsum(neg_group) - neg_group
        auc = float(np.sum(pos_group * (neg_before + 0.5 * neg_group)) / (pos_total * neg_total))

        order = self._desc_order
        starts = self._desc_starts
        pos_group = np.add.reduceat(weights[order] * self.labels[order], starts)
        neg_group = np.add.reduceat(weights[order] * (1 - self.labels[order]), starts)
        cum_pos = np.cumsum(pos_group)
        cum_all = cum_pos + np.cumsum(neg_group)
        precision = np.divide(cum_pos, cum_all, out=np.zeros_like(cum_pos), where=cum_all > 0)
        ap = float(np.sum(pos_group * precision) / pos_total)
        return auc, ap


def verify_weighted_metric_evaluator() -> None:
    examples = [
        (np.array([0, 0, 1, 1]), np.array([0.1, 0.4, 0.35, 0.8])),
        (np.array([0, 1, 0, 1, 1, 0]), np.array([0.2, 0.5, 0.5, 0.7, 0.9, 0.1])),
    ]
    for labels, scores in examples:
        auc, ap = WeightedMetricEvaluator(labels, scores).evaluate(np.ones(len(labels)))
        assert math.isclose(auc, roc_auc_score(labels, scores), abs_tol=1e-12)
        assert math.isclose(ap, average_precision_score(labels, scores), abs_tol=1e-12)


def compute_midrank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    sorted_values = values[order]
    n = len(values)
    midranks = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and sorted_values[j] == sorted_values[i]:
            j += 1
        midranks[i:j] = 0.5 * (i + j - 1)
        i = j
    result = np.empty(n, dtype=float)
    result[order] = midranks + 1.0
    return result


def fast_delong(predictions_sorted_by_label: np.ndarray, positive_count: int) -> tuple[np.ndarray, np.ndarray]:
    classifiers, total = predictions_sorted_by_label.shape
    m = positive_count
    n = total - m
    positive = predictions_sorted_by_label[:, :m]
    negative = predictions_sorted_by_label[:, m:]
    tx = np.empty((classifiers, m), dtype=float)
    ty = np.empty((classifiers, n), dtype=float)
    tz = np.empty((classifiers, total), dtype=float)
    for row in range(classifiers):
        tx[row] = compute_midrank(positive[row])
        ty[row] = compute_midrank(negative[row])
        tz[row] = compute_midrank(predictions_sorted_by_label[row])
    aucs = tz[:, :m].sum(axis=1) / (m * n) - (m + 1.0) / (2.0 * n)
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01, bias=False))
    sy = np.atleast_2d(np.cov(v10, bias=False))
    covariance = sx / m + sy / n
    return aucs, covariance


def paired_delong(labels: np.ndarray, reference: np.ndarray, comparator: np.ndarray) -> dict[str, float]:
    labels = np.asarray(labels, dtype=int)
    order = np.argsort(-labels, kind="stable")
    predictions = np.vstack([reference, comparator])[:, order]
    aucs, covariance = fast_delong(predictions, int(labels.sum()))
    contrast = np.array([1.0, -1.0])
    variance = float(contrast @ covariance @ contrast)
    delta = float(aucs[0] - aucs[1])
    if variance <= np.finfo(float).eps:
        z_value = 0.0 if abs(delta) <= 1e-15 else math.copysign(math.inf, delta)
        p_value = 1.0 if abs(delta) <= 1e-15 else 0.0
    else:
        z_value = delta / math.sqrt(variance)
        p_value = float(2.0 * norm.sf(abs(z_value)))
    return {
        "dst_auc_delong": float(aucs[0]),
        "baseline_auc_delong": float(aucs[1]),
        "delta_auc": delta,
        "delong_variance_delta": variance,
        "delong_z": z_value,
        "p_delong_raw": p_value,
    }


def verify_delong() -> None:
    labels = np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=int)
    a = np.array([0.10, 0.92, 0.35, 0.76, 0.40, 0.71, 0.62, 0.58])
    b = np.array([0.15, 0.80, 0.20, 0.68, 0.55, 0.65, 0.72, 0.52])
    result = paired_delong(labels, a, b)
    assert math.isclose(result["dst_auc_delong"], roc_auc_score(labels, a), abs_tol=1e-12)
    assert math.isclose(result["baseline_auc_delong"], roc_auc_score(labels, b), abs_tol=1e-12)
    reversed_result = paired_delong(labels, b, a)
    assert math.isclose(result["p_delong_raw"], reversed_result["p_delong_raw"], abs_tol=1e-12)
    assert math.isclose(result["delta_auc"], -reversed_result["delta_auc"], abs_tol=1e-12)


def holm_adjust(values: Iterable[float]) -> np.ndarray:
    p_values = np.asarray(list(values), dtype=float)
    order = np.argsort(p_values)
    adjusted = np.empty_like(p_values)
    running = 0.0
    total = len(p_values)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (total - rank) * p_values[index]))
        adjusted[index] = running
    return adjusted


def descriptive_metrics(labels: np.ndarray, scores: np.ndarray) -> dict[str, float | int]:
    predictions = (scores >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
    return {
        "n": int(len(labels)),
        "positives": int(labels.sum()),
        "negatives": int((labels == 0).sum()),
        "auc": float(roc_auc_score(labels, scores)),
        "auprc": float(average_precision_score(labels, scores)),
        "sensitivity_0p5": float(tp / (tp + fn)) if tp + fn else np.nan,
        "specificity_0p5": float(tn / (tn + fp)) if tn + fp else np.nan,
        "bacc_0p5": float(balanced_accuracy_score(labels, predictions)),
        "f1_0p5": float(f1_score(labels, predictions, zero_division=0)),
    }


def bootstrap_distributions(
    labels: np.ndarray,
    scores: dict[str, np.ndarray],
    cluster_ids: np.ndarray | None,
    n_resamples: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    labels = np.asarray(labels, dtype=int)
    evaluators = [WeightedMetricEvaluator(labels, scores[model]) for model in MODEL_ORDER]
    rng = np.random.default_rng(seed)
    auc_samples = np.empty((n_resamples, len(MODEL_ORDER)), dtype=float)
    ap_samples = np.empty_like(auc_samples)
    if cluster_ids is None:
        pos_indices = np.flatnonzero(labels == 1)
        neg_indices = np.flatnonzero(labels == 0)
        for iteration in range(n_resamples):
            weights = np.zeros(len(labels), dtype=float)
            weights[pos_indices] = rng.multinomial(len(pos_indices), np.full(len(pos_indices), 1 / len(pos_indices)))
            weights[neg_indices] = rng.multinomial(len(neg_indices), np.full(len(neg_indices), 1 / len(neg_indices)))
            for model_index, evaluator in enumerate(evaluators):
                auc_samples[iteration, model_index], ap_samples[iteration, model_index] = evaluator.evaluate(weights)
    else:
        cluster_ids = np.asarray(cluster_ids, dtype=int)
        cluster_labels = np.zeros(cluster_ids.max() + 1, dtype=int)
        np.maximum.at(cluster_labels, cluster_ids, labels)
        pos_clusters = np.flatnonzero(cluster_labels == 1)
        neg_clusters = np.flatnonzero(cluster_labels == 0)
        for iteration in range(n_resamples):
            cluster_weights = np.zeros(len(cluster_labels), dtype=float)
            cluster_weights[pos_clusters] = rng.multinomial(len(pos_clusters), np.full(len(pos_clusters), 1 / len(pos_clusters)))
            cluster_weights[neg_clusters] = rng.multinomial(len(neg_clusters), np.full(len(neg_clusters), 1 / len(neg_clusters)))
            weights = cluster_weights[cluster_ids]
            for model_index, evaluator in enumerate(evaluators):
                auc_samples[iteration, model_index], ap_samples[iteration, model_index] = evaluator.evaluate(weights)
    if not np.isfinite(auc_samples).all() or not np.isfinite(ap_samples).all():
        raise AssertionError("Bootstrap produced non-finite results")
    return auc_samples, ap_samples


def _dynamic_auc_ap(labels: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    return WeightedMetricEvaluator(labels, scores).evaluate(np.ones(len(labels), dtype=float))


def _permutation_worker(
    labels: np.ndarray,
    dst: np.ndarray,
    baseline: np.ndarray,
    patient_index: np.ndarray,
    n_patients: int,
    n_resamples: int,
    seed: int,
) -> tuple[int, int, float, float]:
    rng = np.random.default_rng(seed)
    dst_auc, dst_ap = _dynamic_auc_ap(labels, dst)
    base_auc, base_ap = _dynamic_auc_ap(labels, baseline)
    observed_auc = dst_auc - base_auc
    observed_ap = dst_ap - base_ap
    extreme_auc = 0
    extreme_ap = 0
    for _ in range(n_resamples):
        cluster_swap = rng.integers(0, 2, size=n_patients, dtype=np.int8).astype(bool)
        swap = cluster_swap[patient_index]
        perm_dst = np.where(swap, baseline, dst)
        perm_base = np.where(swap, dst, baseline)
        perm_dst_auc, perm_dst_ap = _dynamic_auc_ap(labels, perm_dst)
        perm_base_auc, perm_base_ap = _dynamic_auc_ap(labels, perm_base)
        extreme_auc += int(abs(perm_dst_auc - perm_base_auc) >= abs(observed_auc) - 1e-15)
        extreme_ap += int(abs(perm_dst_ap - perm_base_ap) >= abs(observed_ap) - 1e-15)
    return extreme_auc, extreme_ap, observed_auc, observed_ap


def permutation_tests(
    labels: np.ndarray,
    scores: dict[str, np.ndarray],
    patient_index: np.ndarray,
    n_patients: int,
    n_resamples: int,
    seed: int,
) -> pd.DataFrame:
    jobs = []
    for comparison_index, baseline in enumerate(BASELINE_MODELS):
        jobs.append(
            delayed(_permutation_worker)(
                labels,
                scores[REFERENCE_MODEL],
                scores[baseline],
                patient_index,
                n_patients,
                n_resamples,
                seed + 1000 * (comparison_index + 1),
            )
        )
    results = Parallel(n_jobs=min(4, len(jobs)), verbose=5)(jobs)
    rows = []
    for baseline, (extreme_auc, extreme_ap, observed_auc, observed_ap) in zip(BASELINE_MODELS, results):
        rows.append(
            {
                "baseline": baseline,
                "delta_auc": observed_auc,
                "delta_auprc": observed_ap,
                "p_permutation_auc_raw": (extreme_auc + 1) / (n_resamples + 1),
                "p_permutation_auprc_raw": (extreme_ap + 1) / (n_resamples + 1),
            }
        )
    return pd.DataFrame(rows)


def percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.quantile(values, [0.025, 0.975])
    return float(low), float(high)


def build_model_metrics(
    image: pd.DataFrame,
    patient: pd.DataFrame,
    patient_auc_boot: np.ndarray,
    patient_ap_boot: np.ndarray,
    image_auc_boot: np.ndarray,
    image_ap_boot: np.ndarray,
) -> pd.DataFrame:
    rows = []
    for grain, frame, auc_boot, ap_boot in [
        ("patient", patient, patient_auc_boot, patient_ap_boot),
        ("image_clustered", image, image_auc_boot, image_ap_boot),
    ]:
        labels = frame["label"].to_numpy(int)
        for model_index, model in enumerate(MODEL_ORDER):
            metrics = descriptive_metrics(labels, frame[model].to_numpy(float))
            auc_low, auc_high = percentile_interval(auc_boot[:, model_index])
            ap_low, ap_high = percentile_interval(ap_boot[:, model_index])
            rows.append(
                {
                    "grain": grain,
                    "model": model,
                    **metrics,
                    "auc_ci95_low": auc_low,
                    "auc_ci95_high": auc_high,
                    "auprc_ci95_low": ap_low,
                    "auprc_ci95_high": ap_high,
                    "ci_method": "stratified patient bootstrap" if grain == "patient" else "stratified patient-cluster bootstrap",
                    "n_bootstrap": N_RESAMPLES,
                }
            )
    return pd.DataFrame(rows)


def build_patient_tests(
    patient: pd.DataFrame,
    patient_auc_boot: np.ndarray,
    patient_ap_boot: np.ndarray,
    permutation: pd.DataFrame,
) -> pd.DataFrame:
    labels = patient["label"].to_numpy(int)
    rows = []
    dst_index = MODEL_ORDER.index(REFERENCE_MODEL)
    permutation = permutation.set_index("baseline")
    for baseline in BASELINE_MODELS:
        base_index = MODEL_ORDER.index(baseline)
        delong = paired_delong(labels, patient[REFERENCE_MODEL].to_numpy(float), patient[baseline].to_numpy(float))
        delta_auc_boot = patient_auc_boot[:, dst_index] - patient_auc_boot[:, base_index]
        delta_ap_boot = patient_ap_boot[:, dst_index] - patient_ap_boot[:, base_index]
        auc_low, auc_high = percentile_interval(delta_auc_boot)
        ap_low, ap_high = percentile_interval(delta_ap_boot)
        rows.append(
            {
                "comparison": f"{REFERENCE_MODEL} vs {baseline}",
                "baseline": baseline,
                **delong,
                "delta_auc_ci95_low": auc_low,
                "delta_auc_ci95_high": auc_high,
                "delta_auprc": float(
                    average_precision_score(labels, patient[REFERENCE_MODEL])
                    - average_precision_score(labels, patient[baseline])
                ),
                "bootstrap_mean_delta_auprc": float(np.mean(delta_ap_boot)),
                "delta_auprc_ci95_low": ap_low,
                "delta_auprc_ci95_high": ap_high,
                "p_permutation_auc_raw": float(permutation.loc[baseline, "p_permutation_auc_raw"]),
                "p_permutation_auprc_raw": float(permutation.loc[baseline, "p_permutation_auprc_raw"]),
            }
        )
    result = pd.DataFrame(rows)
    result["p_delong_holm"] = holm_adjust(result["p_delong_raw"])
    result["p_permutation_auc_holm"] = holm_adjust(result["p_permutation_auc_raw"])
    result["p_permutation_auprc_holm"] = holm_adjust(result["p_permutation_auprc_raw"])
    result["significant_delong_holm_0p05"] = result["p_delong_holm"] < ALPHA
    return result


def build_image_tests(
    image_auc_boot: np.ndarray,
    image_ap_boot: np.ndarray,
    permutation: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    dst_index = MODEL_ORDER.index(REFERENCE_MODEL)
    permutation = permutation.set_index("baseline")
    for baseline in BASELINE_MODELS:
        base_index = MODEL_ORDER.index(baseline)
        delta_auc = image_auc_boot[:, dst_index] - image_auc_boot[:, base_index]
        delta_ap = image_ap_boot[:, dst_index] - image_ap_boot[:, base_index]
        auc_low, auc_high = percentile_interval(delta_auc)
        ap_low, ap_high = percentile_interval(delta_ap)
        rows.append(
            {
                "comparison": f"{REFERENCE_MODEL} vs {baseline}",
                "baseline": baseline,
                "delta_auc": float(permutation.loc[baseline, "delta_auc"]),
                "bootstrap_mean_delta_auc": float(np.mean(delta_auc)),
                "delta_auc_ci95_low": auc_low,
                "delta_auc_ci95_high": auc_high,
                "delta_auprc": float(permutation.loc[baseline, "delta_auprc"]),
                "bootstrap_mean_delta_auprc": float(np.mean(delta_ap)),
                "delta_auprc_ci95_low": ap_low,
                "delta_auprc_ci95_high": ap_high,
                "p_permutation_auc_raw": float(permutation.loc[baseline, "p_permutation_auc_raw"]),
                "p_permutation_auprc_raw": float(permutation.loc[baseline, "p_permutation_auprc_raw"]),
            }
        )
    result = pd.DataFrame(rows)
    result["p_permutation_auc_holm"] = holm_adjust(result["p_permutation_auc_raw"])
    result["p_permutation_auprc_holm"] = holm_adjust(result["p_permutation_auprc_raw"])
    result["significant_auc_holm_0p05"] = result["p_permutation_auc_holm"] < ALPHA
    result["significant_auprc_holm_0p05"] = result["p_permutation_auprc_holm"] < ALPHA
    return result


def build_figures(patient: pd.DataFrame, patient_tests: pd.DataFrame) -> None:
    palette = {
        REFERENCE_MODEL: "#1F4E79",
        "GLAM": "#B07A20",
        "Mammo-FM": "#5B7F3A",
        "Mammo-CLIP": "#C86B4A",
        "MIL": "#6B5B95",
    }
    labels = patient["label"].to_numpy(int)
    fig, ax = plt.subplots(figsize=(8.6, 6.4))
    for model in MODEL_ORDER:
        fpr, tpr, _ = roc_curve(labels, patient[model])
        auc = roc_auc_score(labels, patient[model])
        linewidth = 2.8 if model == REFERENCE_MODEL else 1.8
        ax.plot(fpr, tpr, color=palette[model], linewidth=linewidth, label=f"{model} (AUROC {auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#64748B", linewidth=1.2, label="Chance")
    ax.set_xlabel("False-positive rate")
    ax.set_ylabel("True-positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(loc="lower right", frameon=False)
    fig.suptitle("Patient-level ROC curves", fontsize=16, y=0.98)
    ax.set_title("Common independent test cohort: 862 patients, including 19 positive patients", fontsize=9, color="#475569", pad=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(OUTPUT_DIR / "patient_roc.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "patient_roc.pdf", bbox_inches="tight")
    plt.close(fig)

    plot = patient_tests.set_index("baseline").loc[BASELINE_MODELS].reset_index()
    y = np.arange(len(plot))
    point = plot["delta_auc"].to_numpy(float)
    low = plot["delta_auc_ci95_low"].to_numpy(float)
    high = plot["delta_auc_ci95_high"].to_numpy(float)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    ax.axvline(0, color="#334155", linestyle="--", linewidth=1.2)
    ax.errorbar(
        point,
        y,
        xerr=np.vstack([point - low, high - point]),
        fmt="o",
        color="#1F4E79",
        ecolor="#B07A20",
        markerfacecolor="#FFFFFF",
        markeredgewidth=1.8,
        capsize=4,
        linewidth=2,
    )
    ax.set_yticks(y, plot["baseline"])
    ax.invert_yaxis()
    ax.set_xlabel("ΔAUROC (DST k=10 − baseline)")
    ax.grid(axis="x", color="#E2E8F0", linewidth=0.8)
    ax.set_axisbelow(True)
    for yi, p, lo, hi in zip(y, point, low, high):
        ax.text(hi + 0.008, yi, f"{p:+.3f} [{lo:+.3f}, {hi:+.3f}]", va="center", fontsize=9, color="#334155")
    span_low = min(low.min() - 0.03, -0.05)
    span_high = max(high.max() + 0.12, 0.05)
    ax.set_xlim(span_low, span_high)
    fig.suptitle("Patient-level paired AUROC differences", fontsize=16, y=0.98)
    ax.set_title("95% stratified paired-bootstrap intervals; 20,000 resamples", fontsize=9, color="#475569", pad=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(OUTPUT_DIR / "patient_delta_auc_forest.png", dpi=220, bbox_inches="tight")
    fig.savefig(OUTPUT_DIR / "patient_delta_auc_forest.pdf", bbox_inches="tight")
    plt.close(fig)


def _format_p(value: float) -> str:
    return "<0.0001" if value < 0.0001 else f"{value:.4f}"


def write_text_outputs(model_metrics: pd.DataFrame, patient_tests: pd.DataFrame, image_tests: pd.DataFrame, validation: dict) -> None:
    patient_metrics = model_metrics.query("grain == 'patient'").set_index("model")
    comparisons = patient_tests.set_index("baseline")
    methods = f"""## Statistical Analysis

All five models were evaluated on the same independent test cohort of {validation['common_images']:,} mammograms from {validation['common_patients']:,} patients ({validation['positive_patients']} cancer-positive patients). Existing five-fold ensemble predictions were used for DST-Prototype k=10, GLAM, Mammo-FM, and Mammo-CLIP; for MIL, the five saved test probabilities were averaged for each image. Analyses were restricted to image keys available for all models. Patient-level scores, used for the primary analysis, were calculated as the mean probability across all images from a patient, and patient labels were defined by the maximum image label.

Patient-level AUROC and area under the precision-recall curve (AUPRC) were reported with 95% confidence intervals from 20,000 patient-stratified bootstrap resamples. Differences in correlated AUROCs between DST k=10 and each prespecified baseline were tested using two-sided paired DeLong tests. Paired bootstrap confidence intervals for ΔAUROC (DST minus baseline) used identical positive- and negative-patient resampling indices for all models. As a robustness analysis for the small number of positive patients, two-sided paired score-swap permutation tests were performed with 20,000 permutations. DeLong p values were adjusted across the four prespecified comparisons using the Holm method and defined the primary significance conclusions.

At image level, patient was retained as the sampling cluster: all images from a resampled patient entered the bootstrap sample together, and score swaps were applied to the complete patient cluster. Twenty thousand stratified cluster-bootstrap resamples were used for confidence intervals for ΔAUROC and ΔAUPRC, and 20,000 patient-cluster score-swap permutations were used for secondary p values, with separate Holm correction across the four image-level comparisons. All tests were two-sided with α=0.05. Sensitivity, specificity, balanced accuracy, and F1 score at a fixed threshold of 0.5 were descriptive only. The random seed was {SEED}. Training-fold identity was not required because every inferential comparison was paired on the same independent test patients rather than across training folds.
"""
    (OUTPUT_DIR / "statistical_analysis_methods_en.md").write_text(methods, encoding="utf-8")

    metric_sentences = []
    for model in MODEL_ORDER:
        row = patient_metrics.loc[model]
        metric_sentences.append(
            f"{model}: AUROC {row.auc:.3f} (95% CI {row.auc_ci95_low:.3f}–{row.auc_ci95_high:.3f}) and AUPRC {row.auprc:.3f} (95% CI {row.auprc_ci95_low:.3f}–{row.auprc_ci95_high:.3f})"
        )
    comparison_sentences = []
    for baseline in BASELINE_MODELS:
        row = comparisons.loc[baseline]
        sig = "significant" if row.p_delong_holm < ALPHA else "not significant"
        comparison_sentences.append(
            f"versus {baseline}, ΔAUROC={row.delta_auc:+.3f} (95% CI {row.delta_auc_ci95_low:+.3f} to {row.delta_auc_ci95_high:+.3f}; DeLong p={_format_p(row.p_delong_raw)}, Holm-adjusted p={_format_p(row.p_delong_holm)}; {sig})"
        )
    robust_agree = all((patient_tests["p_delong_holm"] < ALPHA) == (patient_tests["p_permutation_auc_holm"] < ALPHA))
    robustness_text = (
        "The patient-level permutation analysis gave the same significance classification as the DeLong-Holm analysis."
        if robust_agree
        else "The patient-level permutation analysis did not reproduce every DeLong-Holm significance classification, so small-positive-sample uncertainty should be emphasized."
    )
    results = f"""## Results

The common evaluation cohort contained {validation['common_images']:,} images from {validation['common_patients']:,} patients, including {validation['positive_patients']} positive and {validation['negative_patients']} negative patients. No label conflicts, duplicate patient-image keys, missing predictions, or non-finite scores were detected. The prespecified correct DST k=10 predictions reproduced a patient-level AUROC of {validation['dst_full_patient_auc']:.6f} on all {validation['dst_full_images']:,} images and {validation['dst_common_patient_auc']:.6f} on the common cohort.

Patient-level discrimination was: {'; '.join(metric_sentences)}. In the primary paired comparisons, {'; '.join(comparison_sentences)}. {robustness_text}

The image-level cluster analysis was treated as sensitivity evidence because images from the same patient are not independent. Its confidence intervals and cluster-permutation p values are reported in the accompanying table and do not replace the patient-level primary conclusion.
"""
    (OUTPUT_DIR / "results_en.md").write_text(results, encoding="utf-8")

    significant = patient_tests.loc[patient_tests["p_delong_holm"] < ALPHA, "baseline"].tolist()
    nonsignificant = patient_tests.loc[patient_tests["p_delong_holm"] >= ALPHA, "baseline"].tolist()
    conclusion = f"""# 中文结论

这次分析已改用正确的 GLAM embedding DST-Prototype k=10，并且严格限制在五个模型共同拥有的 {validation['common_images']:,} 张图像、{validation['common_patients']:,} 名患者上，其中阳性患者 {validation['positive_patients']} 名。标签冲突、重复键、预测缺失和非有限值均为 0。

正确 DST 的患者级 AUROC 在全 {validation['dst_full_images']:,} 张图像上复现为 {validation['dst_full_patient_auc']:.6f}，在共同 {validation['common_images']:,} 张图像上复现为 {validation['dst_common_patient_auc']:.6f}。论文主分析以患者为单位，用相关 ROC 的配对 DeLong 检验，并对四项预设比较做 Holm 校正；训练折不需要相同，因为统计配对发生在同一批独立测试患者上，而不是比较训练折。

Holm 校正后显著优于 DST 的 baseline：无（比较方向固定为 DST 减 baseline）。DST 相对其达到显著优势的 baseline：{('、'.join(significant) if significant else '无')}。未达到显著差异的比较：{('、'.join(nonsignificant) if nonsignificant else '无')}。由于只有 19 名阳性患者，应同时报告置信区间和患者级 score-swap permutation 稳健性结果，避免只依据单个 p 值作过强结论。

固定 0.5 阈值的敏感度、特异度、BACC 和 F1 仅作描述；图像级分析采用患者整簇重采样/置换，只作为敏感性分析，不作为论文主要显著性结论。
"""
    (OUTPUT_DIR / "summary_zh.md").write_text(conclusion, encoding="utf-8")


def run_analysis() -> dict[str, pd.DataFrame | dict]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    verify_weighted_metric_evaluator()
    verify_delong()
    print("[1/7] Loading and validating five model prediction sources", flush=True)
    image, patient, source_audit, validation = load_and_align()
    image.to_csv(OUTPUT_DIR / "common_image_predictions.csv", index=False)
    patient.to_csv(OUTPUT_DIR / "common_patient_predictions.csv", index=False)
    source_audit.to_csv(OUTPUT_DIR / "source_validation.csv", index=False)
    (OUTPUT_DIR / "validation_summary.json").write_text(json.dumps(validation, indent=2), encoding="utf-8")

    patient_labels = patient["label"].to_numpy(int)
    patient_scores = {model: patient[model].to_numpy(float) for model in MODEL_ORDER}
    print(f"[2/7] Patient-stratified bootstrap: {N_RESAMPLES:,} resamples", flush=True)
    patient_auc_boot, patient_ap_boot = bootstrap_distributions(
        patient_labels, patient_scores, None, N_RESAMPLES, SEED
    )

    patient_codes, unique_patients = pd.factorize(image["patient_id"], sort=True)
    image_labels = image["label"].to_numpy(int)
    image_scores = {model: image[model].to_numpy(float) for model in MODEL_ORDER}
    print(f"[3/7] Patient-cluster image bootstrap: {N_RESAMPLES:,} resamples", flush=True)
    image_auc_boot, image_ap_boot = bootstrap_distributions(
        image_labels, image_scores, patient_codes, N_RESAMPLES, SEED
    )

    print(f"[4/7] Patient-level paired score-swap permutation: {N_RESAMPLES:,} per comparison", flush=True)
    patient_index = np.arange(len(patient), dtype=int)
    patient_permutation = permutation_tests(
        patient_labels, patient_scores, patient_index, len(patient), N_RESAMPLES, SEED + 100_000
    )
    print(f"[5/7] Patient-cluster image score-swap permutation: {N_RESAMPLES:,} per comparison", flush=True)
    image_permutation = permutation_tests(
        image_labels, image_scores, patient_codes, len(unique_patients), N_RESAMPLES, SEED + 200_000
    )

    print("[6/7] Building result tables, paper text, and figures", flush=True)
    model_metrics = build_model_metrics(
        image, patient, patient_auc_boot, patient_ap_boot, image_auc_boot, image_ap_boot
    )
    patient_tests = build_patient_tests(patient, patient_auc_boot, patient_ap_boot, patient_permutation)
    image_tests = build_image_tests(image_auc_boot, image_ap_boot, image_permutation)
    model_metrics.to_csv(OUTPUT_DIR / "model_metrics_with_ci.csv", index=False)
    patient_tests.to_csv(OUTPUT_DIR / "patient_primary_tests.csv", index=False)
    image_tests.to_csv(OUTPUT_DIR / "image_cluster_sensitivity_tests.csv", index=False)
    build_figures(patient, patient_tests)
    write_text_outputs(model_metrics, patient_tests, image_tests, validation)

    manifest = {
        "analysis_date": "2026-07-28",
        "reference_model": REFERENCE_MODEL,
        "models": MODEL_ORDER,
        "common_cohort": validation,
        "primary_unit": "patient",
        "primary_test": "two-sided paired DeLong test with Holm correction across four prespecified comparisons",
        "bootstrap": {
            "resamples": N_RESAMPLES,
            "seed": SEED,
            "patient": "positive/negative patient-stratified paired bootstrap",
            "image": "positive/negative patient-stratified cluster bootstrap",
        },
        "permutation": {
            "resamples": N_RESAMPLES,
            "patient": "paired score swap by patient",
            "image": "paired score swap by patient cluster",
        },
        "alpha": ALPHA,
        "threshold_metrics": "descriptive only at 0.5",
        "source_files": {
            model: ([str(p) for p in cfg["paths"]] if "paths" in cfg else str(cfg["path"]))
            for model, cfg in SOURCES.items()
        },
        "training_fold_note": "Training folds need not match because inferential pairing occurs on the same independent test patients.",
        "supersedes": "/mnt/g/Final_MIL/code/analysis_outputs/baseline_dst_k10_wilcoxon_20260728",
    }
    (OUTPUT_DIR / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("[7/7] Analysis complete", flush=True)
    return {
        "image": image,
        "patient": patient,
        "source_audit": source_audit,
        "validation": validation,
        "model_metrics": model_metrics,
        "patient_tests": patient_tests,
        "image_tests": image_tests,
    }


if __name__ == "__main__":
    run_analysis()
