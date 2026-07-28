from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    roc_auc_score,
)


OUTPUT_DIR = Path(__file__).resolve().parent
REFERENCE_MODEL = "DST k=10"
MODEL_ORDER = ["Mammo-FM", "Mammo-CLIP", "MIL", "GLAM", REFERENCE_MODEL]
BASELINE_MODELS = [model for model in MODEL_ORDER if model != REFERENCE_MODEL]

SOURCES = {
    "Mammo-FM": {
        "pattern": "/mnt/g/611/612/613/Mammo-FM/data_predictions_fold{fold}.csv",
        "score": "pred_score",
        "split": "split",
        "validation_pattern": "/mnt/g/611/612/613/Mammo-FM/data_fold{fold}_splits.csv",
    },
    "Mammo-CLIP": {
        "pattern": "/mnt/g/611/612/613/lr_5e-05_epochs_25_weighted_BCE_y_cancer_data_frac_1.0_run_origin_b5_5fold_e25_p4/fold{fold}_all_predictions.csv",
        "score": "image_prediction_prob",
        "split": "split",
    },
    "MIL": {
        "pattern": "/mnt/g/611/612/MIL-Origin/MIL-Origin/2026-06-09/fold_{fold}/ViNDr_mil_predictions_fold_{fold}.csv",
        "score": "prediction_score",
        "split": "split",
    },
    "GLAM": {
        "pattern": "/mnt/g/611/612/glam-origin/glam-origin/glam_kfold_ft_20260611_095443/per_model_predictions/fold{fold}_predictions.csv",
        "score": "pred_score",
        "split": "split",
    },
    REFERENCE_MODEL: {
        "pattern": "/mnt/g/611/612/Mammo-CLIP-DST-EDL/Mammo-CLIP-DST-EDL/embedding_proto_compare_loss_5fold_e250_rerun/DST_k10_run_20260611_143428/fold_{fold}/dst_all_predictions.csv",
        "score": "image_prediction_prob",
        "split": "split",
        "validation_split": "dst_split",
    },
}


def read_prediction(model: str, fold: int) -> pd.DataFrame:
    config = SOURCES[model]
    path = Path(config["pattern"].format(fold=fold))
    frame = pd.read_csv(path, dtype={"patient_id": str, "image_id": str}, low_memory=False)
    required = {"patient_id", "image_id", "cancer", config["score"], config["split"]}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing {sorted(missing)}")
    return frame


def sample_keys(frame: pd.DataFrame, split_col: str = "split", split: str = "test") -> set[tuple[str, str]]:
    mask = frame[split_col].astype(str).str.lower().eq(split)
    return set(map(tuple, frame.loc[mask, ["patient_id", "image_id"]].itertuples(index=False, name=None)))


def validation_patients(model: str, fold: int, frame: pd.DataFrame) -> set[str]:
    config = SOURCES[model]
    if "validation_pattern" in config:
        split_frame = pd.read_csv(
            Path(config["validation_pattern"].format(fold=fold)),
            dtype={"patient_id": str},
            usecols=["patient_id", "split"],
        )
        mask = split_frame["split"].astype(str).str.lower().eq("val")
        return set(split_frame.loc[mask, "patient_id"])
    split_col = config.get("validation_split", config["split"])
    mask = frame[split_col].astype(str).str.lower().eq("val")
    return set(frame.loc[mask, "patient_id"])


def specificity(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    negative = y_true == 0
    return float(np.mean(y_pred[negative] == 0))


def calculate_metrics(frame: pd.DataFrame, score_col: str, grain: str) -> dict[str, float | int]:
    if grain == "patient":
        scored = (
            frame.groupby("patient_id", as_index=False)
            .agg(cancer=("cancer", "max"), score=(score_col, "mean"))
        )
    elif grain == "image":
        scored = frame[["patient_id", "image_id", "cancer", score_col]].rename(columns={score_col: "score"})
    else:
        raise ValueError(grain)

    labels = scored["cancer"].astype(int).to_numpy()
    scores = scored["score"].astype(float).to_numpy()
    predictions = (scores >= 0.5).astype(int)
    return {
        "n": int(len(scored)),
        "positives": int(labels.sum()),
        "negatives": int((labels == 0).sum()),
        "auc": float(roc_auc_score(labels, scores)),
        "average_precision": float(average_precision_score(labels, scores)),
        "bacc_0p5": float(balanced_accuracy_score(labels, predictions)),
        "sensitivity_0p5": float(recall_score(labels, predictions, pos_label=1)),
        "specificity_0p5": specificity(labels, predictions),
        "f1_0p5": float(f1_score(labels, predictions, zero_division=0)),
    }


def holm_adjust(p_values: pd.Series) -> pd.Series:
    values = p_values.astype(float).to_numpy()
    order = np.argsort(values)
    adjusted = np.empty_like(values)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (total - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return pd.Series(adjusted, index=p_values.index)


def rank_biserial(differences: np.ndarray) -> float:
    nonzero = differences[differences != 0]
    if len(nonzero) == 0:
        return 0.0
    ranks = pd.Series(np.abs(nonzero)).rank(method="average").to_numpy()
    positive = ranks[nonzero > 0].sum()
    negative = ranks[nonzero < 0].sum()
    total = positive + negative
    return float((positive - negative) / total) if total else 0.0


def compare_against_dst(fold_metrics: pd.DataFrame, metrics: list[str], correction_family: str) -> pd.DataFrame:
    rows = []
    for grain in ["image", "patient"]:
        for baseline in BASELINE_MODELS:
            for metric in metrics:
                reference = (
                    fold_metrics.query("model == @REFERENCE_MODEL and grain == @grain")
                    .sort_values("fold")[metric]
                    .to_numpy(float)
                )
                comparator = (
                    fold_metrics.query("model == @baseline and grain == @grain")
                    .sort_values("fold")[metric]
                    .to_numpy(float)
                )
                differences = reference - comparator
                if np.allclose(differences, 0):
                    statistic, p_value = 0.0, 1.0
                else:
                    result = wilcoxon(
                        reference,
                        comparator,
                        alternative="two-sided",
                        zero_method="wilcox",
                        method="exact",
                    )
                    statistic, p_value = float(result.statistic), float(result.pvalue)
                rows.append(
                    {
                        "correction_family": correction_family,
                        "grain": grain,
                        "metric": metric,
                        "comparison": f"{REFERENCE_MODEL} vs {baseline}",
                        "baseline": baseline,
                        "n_pairs": int(len(differences)),
                        "dst_mean": float(reference.mean()),
                        "baseline_mean": float(comparator.mean()),
                        "mean_delta_dst_minus_baseline": float(differences.mean()),
                        "median_delta_dst_minus_baseline": float(np.median(differences)),
                        "wins_dst": int((differences > 0).sum()),
                        "losses_dst": int((differences < 0).sum()),
                        "ties": int((differences == 0).sum()),
                        "wilcoxon_w": statistic,
                        "p_raw": p_value,
                        "rank_biserial_dst_minus_baseline": rank_biserial(differences),
                    }
                )
    result = pd.DataFrame(rows)
    result["p_holm"] = holm_adjust(result["p_raw"])
    result["significant_raw_0p05"] = result["p_raw"] < 0.05
    result["significant_holm_0p05"] = result["p_holm"] < 0.05
    return result.sort_values(["grain", "metric", "p_raw", "baseline"]).reset_index(drop=True)


def build_charts(fold_metrics: pd.DataFrame, ensemble_metrics: pd.DataFrame) -> None:
    palette = {
        "Mammo-FM": "#3B6FB6",
        "Mammo-CLIP": "#C58A22",
        "MIL": "#6B7D3A",
        "GLAM": "#C7654D",
        REFERENCE_MODEL: "#334155",
    }

    patient_ensemble = ensemble_metrics.query("grain == 'patient'").set_index("model").loc[MODEL_ORDER]
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    bars = ax.barh(
        patient_ensemble.index,
        patient_ensemble["auc"],
        color=[palette[model] for model in patient_ensemble.index],
        edgecolor="#1f2937",
        linewidth=0.7,
    )
    ax.set_xlim(0, 1)
    ax.set_xlabel("Patient-level AUROC")
    fig.suptitle("Five-model ensemble AUROC", fontsize=16, y=0.98)
    ax.set_title(
        "Common test cohort: 862 patients; patient score = mean image score",
        fontsize=9,
        color="#475569",
        pad=12,
    )
    ax.grid(axis="x", color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)
    for bar, value in zip(bars, patient_ensemble["auc"]):
        ax.text(value + 0.012, bar.get_y() + bar.get_height() / 2, f"{value:.3f}", va="center", fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUTPUT_DIR / "ensemble_patient_auc.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    patient_folds = fold_metrics.query("grain == 'patient'")
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    positions = np.arange(len(MODEL_ORDER))
    for index, model in enumerate(MODEL_ORDER):
        values = patient_folds.query("model == @model").sort_values("fold")["auc"].to_numpy()
        jitter = np.linspace(-0.12, 0.12, len(values))
        ax.scatter(
            np.full(len(values), positions[index]) + jitter,
            values,
            s=58,
            color=palette[model],
            edgecolor="#1f2937",
            linewidth=0.6,
            zorder=3,
        )
        ax.plot([positions[index] - 0.22, positions[index] + 0.22], [values.mean(), values.mean()], color="#111827", linewidth=2)
    ax.set_xticks(positions, MODEL_ORDER)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Patient-level AUROC")
    fig.suptitle("Fold-model AUROC distributions", fontsize=16, y=0.98)
    ax.set_title(
        "Five dots per model; horizontal segment = mean; fold partitions differ across systems",
        fontsize=9,
        color="#475569",
        pad=12,
    )
    ax.grid(axis="y", color="#e2e8f0", linewidth=0.8)
    ax.set_axisbelow(True)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    fig.savefig(OUTPUT_DIR / "fold_patient_auc.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def run_analysis() -> dict[str, pd.DataFrame]:
    frames = {(model, fold): read_prediction(model, fold) for model in MODEL_ORDER for fold in range(5)}
    common_test_keys = None
    for model in MODEL_ORDER:
        current = sample_keys(frames[(model, 0)], split_col=SOURCES[model]["split"], split="test")
        common_test_keys = current if common_test_keys is None else common_test_keys & current
    assert common_test_keys is not None

    source_rows = []
    fold_rows = []
    test_frames = {}
    for model in MODEL_ORDER:
        config = SOURCES[model]
        for fold in range(5):
            frame = frames[(model, fold)].copy()
            mask = frame[config["split"]].astype(str).str.lower().eq("test")
            test = frame.loc[mask].copy()
            keys = pd.MultiIndex.from_frame(test[["patient_id", "image_id"]])
            common_index = pd.MultiIndex.from_tuples(sorted(common_test_keys), names=["patient_id", "image_id"])
            test = test.loc[keys.isin(common_index)].copy()
            test_frames[(model, fold)] = test
            source_rows.append(
                {
                    "model": model,
                    "fold": fold,
                    "source_file": config["pattern"].format(fold=fold),
                    "source_test_rows": int(mask.sum()),
                    "common_test_rows": int(len(test)),
                    "common_test_patients": int(test["patient_id"].nunique()),
                    "positives": int(test["cancer"].astype(int).sum()),
                    "duplicate_patient_image_keys": int(test.duplicated(["patient_id", "image_id"]).sum()),
                    "score_nulls": int(test[config["score"]].isna().sum()),
                }
            )
            for grain in ["image", "patient"]:
                metrics = calculate_metrics(test, config["score"], grain)
                fold_rows.append({"model": model, "fold": fold, "grain": grain, **metrics})

    fold_metrics = pd.DataFrame(fold_rows)
    source_validation = pd.DataFrame(source_rows)

    label_checks = []
    reference_labels = (
        test_frames[(MODEL_ORDER[0], 0)][["patient_id", "image_id", "cancer"]]
        .set_index(["patient_id", "image_id"])["cancer"]
        .sort_index()
    )
    for model in MODEL_ORDER:
        labels = (
            test_frames[(model, 0)][["patient_id", "image_id", "cancer"]]
            .set_index(["patient_id", "image_id"])["cancer"]
            .sort_index()
        )
        label_checks.append(
            {
                "model": model,
                "common_keys": int(len(labels)),
                "label_mismatches_vs_mammo_fm": int((labels != reference_labels).sum()),
            }
        )
    label_validation = pd.DataFrame(label_checks)

    ensemble_rows = []
    for model in MODEL_ORDER:
        score_col = SOURCES[model]["score"]
        merged = None
        for fold in range(5):
            fold_scores = test_frames[(model, fold)][["patient_id", "image_id", "cancer", score_col]].rename(
                columns={score_col: f"score_fold_{fold}"}
            )
            merged = fold_scores if merged is None else merged.merge(
                fold_scores,
                on=["patient_id", "image_id", "cancer"],
                how="inner",
                validate="one_to_one",
            )
        score_columns = [f"score_fold_{fold}" for fold in range(5)]
        merged["ensemble_score"] = merged[score_columns].mean(axis=1)
        for grain in ["image", "patient"]:
            metrics = calculate_metrics(merged, "ensemble_score", grain)
            ensemble_rows.append({"model": model, "grain": grain, **metrics})
    ensemble_metrics = pd.DataFrame(ensemble_rows)

    alignment_rows = []
    validation_sets = {}
    for model in MODEL_ORDER:
        for fold in range(5):
            validation_sets[(model, fold)] = validation_patients(model, fold, frames[(model, fold)])
    for baseline in BASELINE_MODELS:
        for baseline_fold in range(5):
            for dst_fold in range(5):
                left = validation_sets[(baseline, baseline_fold)]
                right = validation_sets[(REFERENCE_MODEL, dst_fold)]
                union = left | right
                alignment_rows.append(
                    {
                        "baseline": baseline,
                        "baseline_fold": baseline_fold,
                        "dst_fold": dst_fold,
                        "baseline_val_patients": len(left),
                        "dst_val_patients": len(right),
                        "intersection": len(left & right),
                        "jaccard": len(left & right) / len(union) if union else np.nan,
                    }
                )
    fold_alignment = pd.DataFrame(alignment_rows)

    wilcoxon_auc = compare_against_dst(fold_metrics, ["auc"], "AUROC across both grains (8 tests)")
    wilcoxon_secondary = compare_against_dst(
        fold_metrics,
        ["bacc_0p5", "sensitivity_0p5", "specificity_0p5"],
        "Fixed-threshold metrics across both grains (24 tests)",
    )

    outputs = {
        "source_validation": source_validation,
        "label_validation": label_validation,
        "fold_metrics": fold_metrics,
        "ensemble_metrics": ensemble_metrics,
        "fold_alignment": fold_alignment,
        "wilcoxon_auc": wilcoxon_auc,
        "wilcoxon_secondary": wilcoxon_secondary,
    }
    for name, frame in outputs.items():
        frame.to_csv(OUTPUT_DIR / f"{name}.csv", index=False)
    build_charts(fold_metrics, ensemble_metrics)

    audit = {
        "analysis_date": "2026-07-28",
        "common_test_images": int(source_validation["common_test_rows"].min()),
        "common_test_patients": int(source_validation["common_test_patients"].min()),
        "models": MODEL_ORDER,
        "reference": REFERENCE_MODEL,
        "primary_test": "two-sided exact Wilcoxon signed-rank on nominal fold-index AUROC pairs",
        "multiple_testing": {
            "auc": "Holm correction over 8 comparisons",
            "secondary": "Holm correction over 24 fixed-threshold comparisons",
        },
        "major_caveat": "Validation partitions differ across systems, so fold-index pairing is nominal rather than a truly matched resampling design.",
        "source_files": {model: SOURCES[model]["pattern"] for model in MODEL_ORDER},
    }
    (OUTPUT_DIR / "analysis_manifest.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return outputs


if __name__ == "__main__":
    run_analysis()
