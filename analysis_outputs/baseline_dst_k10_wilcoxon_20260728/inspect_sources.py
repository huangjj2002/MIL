from pathlib import Path

import pandas as pd


SOURCES = {
    "Mammo-FM": {
        "pattern": "/mnt/g/611/612/613/Mammo-FM/data_predictions_fold{fold}.csv",
        "score": "pred_score",
        "split": "split",
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
    "DST k=10": {
        "pattern": "/mnt/g/611/612/Mammo-CLIP-DST-EDL/Mammo-CLIP-DST-EDL/embedding_proto_compare_loss_5fold_e250_rerun/DST_k10_run_20260611_143428/fold_{fold}/dst_all_predictions.csv",
        "score": "image_prediction_prob",
        "split": "split",
        "model_split": "dst_split",
    },
}


def key_set(frame: pd.DataFrame, split: str) -> set[tuple[str, str]]:
    subset = frame.loc[frame["split"].astype(str).str.lower() == split, ["patient_id", "image_id"]]
    return set(map(tuple, subset.astype(str).itertuples(index=False, name=None)))


frames: dict[tuple[str, int], pd.DataFrame] = {}
for model, config in SOURCES.items():
    for fold in range(5):
        path = Path(config["pattern"].format(fold=fold))
        frame = pd.read_csv(path, low_memory=False)
        frames[(model, fold)] = frame
        split_counts = frame["split"].astype(str).str.lower().value_counts().to_dict()
        model_split_counts = (
            frame[config["model_split"]].astype(str).str.lower().value_counts().to_dict()
            if "model_split" in config
            else None
        )
        duplicate_keys = frame.duplicated(["patient_id", "image_id", "split"]).sum()
        print(
            model,
            fold,
            "rows=", len(frame),
            "splits=", split_counts,
            "model_splits=", model_split_counts,
            "patients=", frame["patient_id"].astype(str).nunique(),
            "duplicate_keys=", int(duplicate_keys),
            "score_nulls=", int(frame[config["score"]].isna().sum()),
        )

print("\nTEST KEY OVERLAP")
reference = key_set(frames[("Mammo-FM", 0)], "test")
for model in SOURCES:
    for fold in range(5):
        current = key_set(frames[(model, fold)], "test")
        print(model, fold, len(current), "missing", len(reference - current), "extra", len(current - reference))

print("\nVALIDATION KEY OVERLAP BY FOLD")
for fold in range(5):
    val_sets = {}
    for model, config in SOURCES.items():
        frame = frames[(model, fold)]
        if "model_split" in config:
            mask = frame[config["model_split"]].astype(str).str.lower().eq("val")
            subset = frame.loc[mask, ["patient_id", "image_id"]]
            val_sets[model] = set(map(tuple, subset.astype(str).itertuples(index=False, name=None)))
        elif frame["split"].astype(str).str.lower().eq("val").any():
            val_sets[model] = key_set(frame, "val")
        else:
            val_sets[model] = set()
    names = list(val_sets)
    for model in names:
        print(fold, model, "n=", len(val_sets[model]), "vs Mammo-FM symmetric_diff=", len(val_sets[model] ^ val_sets["Mammo-FM"]))

print("\nVALIDATION PATIENT OVERLAP MATRICES")
validation_patients: dict[str, dict[int, set[str]]] = {model: {} for model in SOURCES}
for fold in range(5):
    fm_splits = pd.read_csv(
        f"/mnt/g/611/612/613/Mammo-FM/data_fold{fold}_splits.csv",
        usecols=["patient_id", "split"],
    )
    validation_patients["Mammo-FM"][fold] = set(
        fm_splits.loc[fm_splits["split"].astype(str).str.lower().eq("val"), "patient_id"].astype(str)
    )
    for model, config in SOURCES.items():
        if model == "Mammo-FM":
            continue
        frame = frames[(model, fold)]
        split_col = config.get("model_split", "split")
        validation_patients[model][fold] = set(
            frame.loc[frame[split_col].astype(str).str.lower().eq("val"), "patient_id"].astype(str)
        )

for model in SOURCES:
    print("MODEL", model)
    header = "       " + " ".join(f"f{fold:>4}" for fold in range(5))
    print(header)
    for ref_fold in range(5):
        row = []
        reference_patients = validation_patients["Mammo-FM"][ref_fold]
        for fold in range(5):
            current_patients = validation_patients[model][fold]
            union = reference_patients | current_patients
            row.append(len(reference_patients & current_patients) / len(union) if union else 0.0)
        print(f"fm{ref_fold}: " + " ".join(f"{value:5.3f}" for value in row))

print("\nTEST LABEL CONSISTENCY")
reference_frame = frames[("Mammo-FM", 0)]
reference_labels = (
    reference_frame.loc[reference_frame["split"].astype(str).str.lower().eq("test"), ["patient_id", "image_id", "cancer"]]
    .assign(patient_id=lambda x: x["patient_id"].astype(str), image_id=lambda x: x["image_id"].astype(str))
    .set_index(["patient_id", "image_id"])["cancer"]
)
for model in SOURCES:
    frame = frames[(model, 0)]
    labels = (
        frame.loc[frame["split"].astype(str).str.lower().eq("test"), ["patient_id", "image_id", "cancer"]]
        .assign(patient_id=lambda x: x["patient_id"].astype(str), image_id=lambda x: x["image_id"].astype(str))
        .set_index(["patient_id", "image_id"])["cancer"]
    )
    common = reference_labels.index.intersection(labels.index)
    print(model, "common=", len(common), "mismatches=", int((reference_labels.loc[common] != labels.loc[common]).sum()))
