from __future__ import annotations

import json
import math
from pathlib import Path

import nbformat
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from analysis import MODEL_ORDER, paired_delong, verify_delong, verify_weighted_metric_evaluator


OUTPUT_DIR = Path(__file__).resolve().parent


def main() -> None:
    validation = json.loads((OUTPUT_DIR / "validation_summary.json").read_text(encoding="utf-8"))
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
        assert validation[key] == value, (key, validation[key], value)
    assert math.isclose(validation["dst_full_patient_auc"], 0.8648311169382531, abs_tol=5e-12)
    assert math.isclose(validation["dst_common_patient_auc"], 0.8645813822813261, abs_tol=5e-12)

    image = pd.read_csv(OUTPUT_DIR / "common_image_predictions.csv", dtype={"patient_id": str, "image_id": str})
    patient = pd.read_csv(OUTPUT_DIR / "common_patient_predictions.csv", dtype={"patient_id": str})
    metrics = pd.read_csv(OUTPUT_DIR / "model_metrics_with_ci.csv")
    primary = pd.read_csv(OUTPUT_DIR / "patient_primary_tests.csv")
    sensitivity = pd.read_csv(OUTPUT_DIR / "image_cluster_sensitivity_tests.csv")
    assert len(image) == 8409 and len(patient) == 862
    assert image.duplicated(["patient_id", "image_id"]).sum() == 0
    assert patient["label"].sum() == 19
    assert np.isfinite(image[MODEL_ORDER].to_numpy(float)).all()
    assert np.isfinite(patient[MODEL_ORDER].to_numpy(float)).all()
    assert np.isfinite(metrics.select_dtypes(include=["number"]).to_numpy()).all()
    assert np.isfinite(primary.select_dtypes(include=["number"]).to_numpy()).all()
    assert np.isfinite(sensitivity.select_dtypes(include=["number"]).to_numpy()).all()

    labels = patient["label"].to_numpy(int)
    metric_index = metrics.query("grain == 'patient'").set_index("model")
    for model in MODEL_ORDER:
        auc = roc_auc_score(labels, patient[model])
        ap = average_precision_score(labels, patient[model])
        assert math.isclose(auc, metric_index.loc[model, "auc"], abs_tol=1e-12)
        assert math.isclose(ap, metric_index.loc[model, "auprc"], abs_tol=1e-12)
    for _, row in primary.iterrows():
        direct = paired_delong(labels, patient["DST k=10"].to_numpy(), patient[row["baseline"]].to_numpy())
        assert math.isclose(direct["p_delong_raw"], row["p_delong_raw"], abs_tol=1e-12)
        assert math.isclose(direct["delta_auc"], row["delta_auc"], abs_tol=1e-12)

    verify_weighted_metric_evaluator()
    verify_delong()
    notebook = nbformat.read(OUTPUT_DIR / "correct_dst_k10_statistical_analysis.executed.ipynb", as_version=4)
    code_cells = [cell for cell in notebook.cells if cell.cell_type == "code"]
    assert code_cells and all(cell.execution_count is not None for cell in code_cells)
    errors = [output for cell in code_cells for output in cell.get("outputs", []) if output.output_type == "error"]
    assert not errors, errors
    for name in ["patient_roc.png", "patient_delta_auc_forest.png", "patient_roc.pdf", "patient_delta_auc_forest.pdf"]:
        path = OUTPUT_DIR / name
        assert path.exists() and path.stat().st_size > 1000, name
    print("VALIDATION_OK")


if __name__ == "__main__":
    main()
