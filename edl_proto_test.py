"""
Prototype + EDL inference script.

Loads Prototype+EDL fold checkpoints, predicts validation and held-out test
splits, and writes CSVs aligned with the existing MIL/EDL outputs.
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch

from Datasets.dataset_utils import MIL_dataloader
from edl_proto_train import (
    _add_proto_args,
    build_edl_proto_model,
    build_prediction_df,
    edl_proto_predict,
)
from edl_test import config as base_edl_test_config
from utils.data_split_utils import (
    adaptive_stratified_train_val_split,
    generator_cross_val_folds,
    split_df_by_cohorts,
)
from utils.generic_utils import clear_memory, seed_all
from utils.metrics import auroc, evaluate_metrics


def config():
    """Parse base EDL test arguments plus Prototype+EDL-only arguments."""
    proto_parser = argparse.ArgumentParser(add_help=False)
    _add_proto_args(proto_parser)
    proto_args, remaining = proto_parser.parse_known_args()

    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0]] + remaining
        args = base_edl_test_config()
    finally:
        sys.argv = original_argv

    for key, value in vars(proto_args).items():
        setattr(args, key, value)
    return args


def _mode_first(series):
    mode = series.mode(dropna=True)
    if len(mode) > 0:
        return mode.iloc[0]
    return series.iloc[0] if len(series) else np.nan


def _reorder_prediction_columns(df, args):
    label_col = args.label.lower()
    base_cols = [
        "patient_id",
        "image_id",
        "split",
        "cohort_num",
        label_col,
        "prediction_score",
        "predicted_class",
        "evidence_0",
        "evidence_1",
        "alpha_0",
        "alpha_1",
        "uncertainty",
        "fold",
    ]
    proto_cols = [col for col in df.columns if col.startswith("proto_")]
    keep_cols = [col for col in base_cols if col in df.columns] + sorted(proto_cols)
    return df[keep_cols]


def _build_ensemble(test_all_df, args):
    label_col = args.label.lower()
    agg_spec = {
        "prediction_score": "mean",
        "predicted_class": lambda x: (x.mean() >= 0.5).astype(int),
        "evidence_0": "mean",
        "evidence_1": "mean",
        "alpha_0": "mean",
        "alpha_1": "mean",
        "uncertainty": "mean",
        label_col: "first",
        "cohort_num": "first",
        "split": "first",
    }

    for col in test_all_df.columns:
        if col.startswith("proto_"):
            agg_spec[col] = _mode_first if col.endswith("_idx") else "mean"

    ensemble = test_all_df.groupby(["patient_id", "image_id"]).agg(agg_spec).reset_index()
    ensemble["fold"] = "ensemble"
    return _reorder_prediction_columns(ensemble, args)


def run_edl_proto_test(args, device, checkpoint_dir=None, output_dir=None):
    """
    Prototype+EDL test function that can be called from training or standalone.
    """
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
    elif hasattr(args, "checkpoint_dir"):
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        raise ValueError("checkpoint_dir must be provided via argument or args.checkpoint_dir")

    if output_dir is not None:
        output_dir = Path(output_dir)
    elif hasattr(args, "output_dir") and args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = checkpoint_dir / "edl_proto_test_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    seed_all(args.seed)
    args.data_dir = Path(args.data_dir)
    args.n_class = 1
    df = pd.read_csv(args.data_dir / args.csv_file).fillna(0)

    print(f"df shape: {df.shape}")
    print(df.columns)

    _, dev_df, test_df = split_df_by_cohorts(
        df,
        train_cohorts=args.train_cohorts,
        test_cohorts=args.test_cohorts,
    )

    if args.data_frac < 1.0:
        dev_df = dev_df.sample(frac=args.data_frac, random_state=1, ignore_index=True)

    single_internal_val = args.n_folds == 0
    total_folds = 1 if single_internal_val else args.n_folds
    label_col = args.label.lower()

    assignment_path = checkpoint_dir / f"{args.dataset}_edl_proto_val_fold_assignments.csv"
    fold_val_dfs = {}
    if assignment_path.exists():
        assignment_df = pd.read_csv(assignment_path)
        for fold in range(total_folds):
            fold_val_dfs[fold] = assignment_df[assignment_df["fold"] == fold].reset_index(drop=True)
    else:
        if single_internal_val:
            _, val_df_fold = adaptive_stratified_train_val_split(
                dev_df,
                val_frac=args.kfold0_val_frac,
                max_val_frac=args.kfold0_val_max_frac,
                args=args,
                context="EDL_PROTO test n_folds=0 internal train/val split",
            )
            fold_val_dfs[0] = val_df_fold.reset_index(drop=True)
        else:
            for fold_idx, (_, val_df_fold) in enumerate(
                generator_cross_val_folds(dev_df, args.n_folds, args.label, random_state=args.seed)
            ):
                fold_val_dfs[fold_idx] = val_df_fold.reset_index(drop=True)

    print("\n===== Predicting on development data with Prototype+EDL =====")
    all_dev_results = []
    dev_results_df = None
    test_ensemble = None

    for fold in range(total_folds):
        print(f"\n--- Fold {fold} ---")
        ckpt_path = checkpoint_dir / f"fold_{fold}" / "best_model.pth"
        if not ckpt_path.exists():
            print(f"Warning: checkpoint not found at {ckpt_path}, skipping fold {fold}")
            continue

        model, _ = build_edl_proto_model(args, ckpt_path)
        model.to(device)

        val_df = fold_val_dfs.get(fold, pd.DataFrame()).reset_index(drop=True)
        if len(val_df) == 0:
            print(f"Warning: no validation rows found for fold {fold}, skipping")
            del model
            clear_memory()
            continue
        if args.label not in val_df.columns and label_col in val_df.columns:
            val_df[args.label] = val_df[label_col]

        val_loader = MIL_dataloader(val_df, "test", args)
        val_results = edl_proto_predict(
            val_loader,
            model,
            args,
            device,
            desc=f"EDL_PROTO val fold {fold}",
        )
        val_result_df = build_prediction_df(val_df, val_results, "val", fold, args)
        all_dev_results.append(val_result_df)

        targs = np.array(val_results["label"])
        probs = np.array(val_results["score"])
        preds = np.array(val_results["pred"])
        try:
            fold_auc = auroc(targs, probs)
            fold_f1, fold_bacc = evaluate_metrics(targs, preds)
            print(f"  Fold {fold} Val - AUC: {fold_auc:.4f}, F1: {fold_f1:.4f}, BAcc: {fold_bacc:.4f}")
        except Exception as exc:
            print(f"  Fold {fold} metrics error: {exc}")

        del model
        clear_memory()

    if all_dev_results:
        dev_results_df = pd.concat(all_dev_results, ignore_index=True)
        dev_results_df.to_csv(output_dir / f"{args.dataset}_edl_proto_dev_predictions.csv", index=False)
        print(f"\nDev predictions saved: {len(dev_results_df)} samples")

    if len(test_df) > 0:
        print("\n===== Predicting on test data with Prototype+EDL =====")
        test_all_fold_results = []

        for fold in range(total_folds):
            print(f"\n--- Test with Fold {fold} model ---")
            ckpt_path = checkpoint_dir / f"fold_{fold}" / "best_model.pth"
            if not ckpt_path.exists():
                print(f"Warning: checkpoint not found at {ckpt_path}, skipping")
                continue

            model, _ = build_edl_proto_model(args, ckpt_path)
            model.to(device)

            test_loader = MIL_dataloader(test_df, "test", args)
            test_results = edl_proto_predict(
                test_loader,
                model,
                args,
                device,
                desc=f"EDL_PROTO test fold {fold}",
            )
            test_result_df = build_prediction_df(test_df, test_results, "test", fold, args)
            test_all_fold_results.append(test_result_df)

            targs = np.array(test_results["label"])
            probs = np.array(test_results["score"])
            preds = np.array(test_results["pred"])
            try:
                fold_auc = auroc(targs, probs)
                fold_f1, fold_bacc = evaluate_metrics(targs, preds)
                print(f"  Fold {fold} Test - AUC: {fold_auc:.4f}, F1: {fold_f1:.4f}, BAcc: {fold_bacc:.4f}")
            except Exception as exc:
                print(f"  Fold {fold} metrics error: {exc}")

            del model
            clear_memory()

        if test_all_fold_results:
            test_all_df = pd.concat(test_all_fold_results, ignore_index=True)
            test_all_df.to_csv(output_dir / f"{args.dataset}_edl_proto_test_all_folds.csv", index=False)

            test_ensemble = _build_ensemble(test_all_df, args)
            test_ensemble.to_csv(output_dir / f"{args.dataset}_edl_proto_test_ensemble.csv", index=False)

            targs_ens = test_ensemble[label_col].values
            probs_ens = test_ensemble["prediction_score"].values.astype(float)
            preds_ens = test_ensemble["predicted_class"].values.astype(int)
            try:
                ens_auc = auroc(targs_ens, probs_ens)
                ens_f1, ens_bacc = evaluate_metrics(targs_ens.astype(int), preds_ens)
                print(f"\nEnsemble Test - AUC: {ens_auc:.4f}, F1: {ens_f1:.4f}, BAcc: {ens_bacc:.4f}")
            except Exception as exc:
                print(f"Ensemble metrics error: {exc}")

    if dev_results_df is not None and test_ensemble is not None:
        combined_df = pd.concat([dev_results_df, test_ensemble], ignore_index=True)
    elif dev_results_df is not None:
        combined_df = dev_results_df
    else:
        combined_df = None

    if combined_df is not None:
        combined_df.to_csv(output_dir / f"{args.dataset}_edl_proto_all_predictions.csv", index=False)
        print(
            f"\nCombined predictions saved: {len(combined_df)} samples -> "
            f"{output_dir / f'{args.dataset}_edl_proto_all_predictions.csv'}"
        )

    print("\n===== Prototype+EDL Test Complete =====")
    return output_dir


def main():
    args = config()
    args.edl_proto_normalize = args.edl_proto_normalize == "y"

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print(f"[INFO] Using GPU {args.gpu_id}")

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args.apex = True if args.apex == "y" else False
    run_edl_proto_test(args, device)


if __name__ == "__main__":
    main()
