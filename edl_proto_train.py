"""
Prototype + EDL training script.

This is a separate third path for comparing MIL, EDL, and Prototype+EDL. The
existing EDL scripts are intentionally left unchanged.
"""

import argparse
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import yaml
from tqdm import tqdm

from Datasets.dataset_utils import MIL_dataloader
from MIL import build_model
from MIL.edl_losses import EDLCombinedLoss
from MIL.edl_proto_models import MIL_EDL_Prototype_Wrapper
from edl_train import (
    LinearWarmupCosineAnnealingLR,
    _get_checkpoint_state_dict,
    _print_load_summary,
    config as base_edl_config,
    edl_train_loop,
    freeze_mil_backbone_train_edl_only,
    get_edl_class_weights,
    resolve_mil_checkpoint,
)
from utils.data_split_utils import (
    adaptive_stratified_train_val_split,
    generator_cross_val_folds,
    split_df_by_cohorts,
)
from utils.generic_utils import clear_memory, seed_all
from utils.metrics import auroc, evaluate_metrics


def _add_proto_args(parser):
    parser.add_argument("--edl_proto_k", default=4, type=int,
                        help="Number of prototypes per class.")
    parser.add_argument("--edl_proto_topk", default=3, type=int,
                        help="Number of top prototypes exported per class.")
    parser.add_argument("--edl_proto_gamma_init", default=1.0, type=float,
                        help="Initial distance sharpness for prototype similarity.")
    parser.add_argument("--edl_proto_normalize", default="y", choices=["y", "n"],
                        help="Normalize embeddings and prototypes before distance computation.")
    parser.add_argument("--edl_proto_init", default="kmeans", choices=["kmeans", "random"],
                        help="Prototype initialization method.")


def config():
    """Parse base EDL arguments plus Prototype+EDL-only arguments."""
    proto_parser = argparse.ArgumentParser(add_help=False)
    _add_proto_args(proto_parser)
    proto_args, remaining = proto_parser.parse_known_args()

    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0]] + remaining
        args = base_edl_config()
    finally:
        sys.argv = original_argv

    for key, value in vars(proto_args).items():
        setattr(args, key, value)
    return args


def _looks_like_wrapped_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        return False
    return any(key.startswith("mil_model.") for key in state_dict.keys())


def _looks_like_edl_proto_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        return False
    return any(
        ("prototypes" in key or "proto_strength" in key or "raw_gamma" in key)
        for key in state_dict.keys()
    )


def build_edl_proto_model(args, checkpoint_path=None):
    """
    Build MIL backbone, load optional checkpoint, and wrap with Prototype+EDL.

    Returns:
        model, loaded_proto_checkpoint
    """
    args.n_class = 1
    if args.feature_extraction == "online" and not getattr(args, "clip_chk_pt_path", None):
        raise ValueError(
            "--clip_chk_pt_path is required when --feature_extraction online "
            "so the Mammo-CLIP image encoder/backbone can be initialized."
        )

    mil_model = build_model(args)

    checkpoint_state = None
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_file():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            checkpoint_state = _get_checkpoint_state_dict(checkpoint)
        else:
            print(f"[EDL_PROTO] Warning: checkpoint not found at {checkpoint_path}; training from scratch.")

    is_wrapped_checkpoint = _looks_like_wrapped_state_dict(checkpoint_state)
    is_proto_checkpoint = _looks_like_edl_proto_state_dict(checkpoint_state)

    if checkpoint_state is not None and not is_wrapped_checkpoint:
        load_msg = mil_model.load_state_dict(checkpoint_state, strict=False)
        print(f"[EDL_PROTO] Loaded pretrained MIL backbone from: {checkpoint_path}")
        _print_load_summary("[EDL_PROTO][MIL load]", load_msg)

    model = MIL_EDL_Prototype_Wrapper(
        mil_model,
        edl_dropout=args.edl_dropout,
        proto_k=args.edl_proto_k,
        proto_topk=args.edl_proto_topk,
        proto_normalize=args.edl_proto_normalize,
        proto_gamma_init=args.edl_proto_gamma_init,
    )

    if checkpoint_state is not None and is_wrapped_checkpoint:
        load_msg = model.load_state_dict(checkpoint_state, strict=False)
        if is_proto_checkpoint:
            print(f"[EDL_PROTO] Loaded Prototype+EDL checkpoint from: {checkpoint_path}")
        else:
            print(
                "[EDL_PROTO] Loaded wrapped EDL backbone weights; "
                "prototype heads will be initialized separately."
            )
        _print_load_summary("[EDL_PROTO][wrapped load]", load_msg)

    return model, is_proto_checkpoint


def _move_inputs_to_device(data, device, non_blocking=True):
    if isinstance(data["x"], dict):
        return {scale: tensor.to(device, non_blocking=non_blocking) for scale, tensor in data["x"].items()}
    if isinstance(data["x"], list):
        return [tensor.to(device, non_blocking=non_blocking) for tensor in data["x"]]
    return data["x"].to(device, non_blocking=non_blocking)


@torch.no_grad()
def initialize_prototypes_from_train_split(model, train_df, args, device, fold):
    """Initialize every PrototypeEDLHead from current-fold training embeddings."""
    if args.edl_proto_init != "kmeans":
        print("[EDL_PROTO] Prototype KMeans initialization skipped; using random init.")
        return

    heads = model.prototype_heads()
    if not heads:
        print("[EDL_PROTO] No prototype heads found; skipping initialization.")
        return

    print("[EDL_PROTO] Initializing prototypes with train-fold KMeans embeddings...")
    init_loader = MIL_dataloader(train_df, "test", args)
    buckets = {
        name: {"embeddings": [], "labels": []}
        for name in heads.keys()
    }
    current_labels = None

    def make_hook(head_name):
        def hook(module, inputs):
            if current_labels is None:
                return
            buckets[head_name]["embeddings"].append(inputs[0].detach().float().cpu())
            buckets[head_name]["labels"].append(current_labels.detach().cpu())
        return hook

    handles = [
        head.register_forward_pre_hook(make_hook(name))
        for name, head in heads.items()
    ]

    was_training = model.training
    model.eval()
    model.is_training = False
    amp_enabled = bool(args.apex) and device.type == "cuda"

    try:
        progress_iter = tqdm(init_loader, desc=f"[fold {fold} EDL_PROTO init]", total=len(init_loader))
        for data in progress_iter:
            current_labels = data["y"].long().to(device)
            inputs = _move_inputs_to_device(data, device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                _ = model(inputs)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    for name, head in heads.items():
        embeddings = torch.cat(buckets[name]["embeddings"], dim=0)
        labels = torch.cat(buckets[name]["labels"], dim=0)
        warnings_list = head.initialize_from_embeddings(
            embeddings,
            labels,
            random_state=args.seed + fold,
        )
        print(
            f"[EDL_PROTO] {name}: initialized {head.num_classes * head.prototypes_per_class} "
            f"prototypes from {len(labels)} training embeddings."
        )
        for warning_text in warnings_list:
            print(f"[EDL_PROTO] Warning: {name}: {warning_text}")


def expected_proto_columns(args):
    topk = max(0, min(int(args.edl_proto_topk), int(args.edl_proto_k)))
    columns = []
    for class_idx in range(2):
        for rank in range(1, topk + 1):
            prefix = f"proto_c{class_idx}_top{rank}"
            columns.extend([
                f"{prefix}_idx",
                f"{prefix}_evidence",
                f"{prefix}_similarity",
            ])
    return columns


def _append_proto_batch(proto_buffers, edl_out):
    if "topk_proto_idx" not in edl_out:
        return

    top_idx = edl_out["topk_proto_idx"].detach().cpu().numpy()
    top_evidence = edl_out["topk_proto_evidence"].detach().cpu().numpy()
    top_similarity = edl_out["topk_proto_similarity"].detach().cpu().numpy()

    for class_idx in range(top_idx.shape[1]):
        for rank_idx in range(top_idx.shape[2]):
            prefix = f"proto_c{class_idx}_top{rank_idx + 1}"
            proto_buffers.setdefault(f"{prefix}_idx", []).append(top_idx[:, class_idx, rank_idx])
            proto_buffers.setdefault(f"{prefix}_evidence", []).append(top_evidence[:, class_idx, rank_idx])
            proto_buffers.setdefault(f"{prefix}_similarity", []).append(top_similarity[:, class_idx, rank_idx])


@torch.no_grad()
def edl_proto_predict(loader, model, args, device, desc="EDL_PROTO predict"):
    """Run Prototype+EDL inference and return sample-level predictions."""
    model.eval()
    model.is_training = False

    targs = []
    probs_list = []
    preds_list = []
    uncertainty_list = []
    evidence_list = []
    alpha_list = []
    proto_buffers = {}
    sample_patient_ids = []
    sample_image_ids = []

    progress_iter = tqdm(enumerate(loader), total=len(loader), desc=desc)
    for _, data in progress_iter:
        inputs = _move_inputs_to_device(data, device, non_blocking=True)
        labels = data["y"].long().to(device)

        if isinstance(data["x"], dict):
            batch_size = next(iter(data["x"].values())).size(0)
        elif isinstance(data["x"], list):
            batch_size = data["x"][0].size(0)
        else:
            batch_size = data["x"].size(0)

        sample_patient_ids.extend(data.get("patient_id", [None] * batch_size))
        sample_image_ids.extend(data.get("image_id", [None] * batch_size))

        amp_enabled = bool(args.apex) and device.type == "cuda"
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            edl_out = model(inputs)

        prob = edl_out["prob"].detach().cpu()
        evidence = edl_out["evidence"].detach().cpu()
        alpha = edl_out["alpha"].detach().cpu()
        uncertainty = edl_out["uncertainty"].detach().cpu()
        pred_class = torch.argmax(prob, dim=-1)

        targs.append(labels.cpu().numpy())
        probs_list.append(prob[:, 1].numpy())
        preds_list.append(pred_class.numpy())
        uncertainty_list.append(uncertainty.numpy())
        evidence_list.append(evidence.numpy())
        alpha_list.append(alpha.numpy())
        _append_proto_batch(proto_buffers, edl_out)

    evidence_array = np.concatenate(evidence_list)
    alpha_array = np.concatenate(alpha_list)
    results = {
        "patient_id": sample_patient_ids,
        "image_id": sample_image_ids,
        "label": np.concatenate(targs).tolist(),
        "score": np.concatenate(probs_list).tolist(),
        "pred": np.concatenate(preds_list).tolist(),
        "uncertainty": np.concatenate(uncertainty_list).tolist(),
        "evidence_0": evidence_array[:, 0].tolist(),
        "evidence_1": evidence_array[:, 1].tolist(),
        "alpha_0": alpha_array[:, 0].tolist(),
        "alpha_1": alpha_array[:, 1].tolist(),
    }

    for key, chunks in proto_buffers.items():
        results[key] = np.concatenate(chunks).tolist()

    return results


def build_prediction_df(split_df, sample_results, split_name, fold, args):
    label_col = args.label.lower()
    pred_df = split_df.copy().reset_index(drop=True)
    pred_df["prediction_score"] = sample_results["score"]
    pred_df["predicted_class"] = sample_results["pred"]
    pred_df[label_col] = sample_results["label"]
    pred_df["evidence_0"] = sample_results["evidence_0"]
    pred_df["evidence_1"] = sample_results["evidence_1"]
    pred_df["alpha_0"] = sample_results["alpha_0"]
    pred_df["alpha_1"] = sample_results["alpha_1"]
    pred_df["uncertainty"] = sample_results["uncertainty"]
    pred_df["fold"] = fold
    pred_df["split"] = split_name

    for key, values in sample_results.items():
        if key.startswith("proto_"):
            pred_df[key] = values

    if "cohort_num" not in pred_df.columns and "cohert_num" in pred_df.columns:
        pred_df["cohort_num"] = pred_df["cohert_num"]
    for col in ["patient_id", "image_id", "cohort_num"]:
        if col not in pred_df.columns:
            pred_df[col] = None

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
    proto_cols = [col for col in expected_proto_columns(args) if col in pred_df.columns]
    extra_proto_cols = sorted(
        col for col in pred_df.columns
        if col.startswith("proto_") and col not in proto_cols
    )
    keep_cols = [col for col in base_cols + proto_cols + extra_proto_cols if col in pred_df.columns]
    return pred_df[keep_cols]


def do_edl_proto_training(args, device):
    """Main Prototype+EDL training function with k-fold cross-validation."""
    args.n_class = 1
    args.data_dir = Path(args.data_dir)
    args.df = pd.read_csv(args.data_dir / args.csv_file).fillna(0)

    print(f"df shape: {args.df.shape}")
    print(args.df.columns)

    _, dev_df, test_df = split_df_by_cohorts(
        args.df,
        train_cohorts=args.train_cohorts,
        test_cohorts=args.test_cohorts,
    )

    if args.data_frac < 1.0:
        dev_df = dev_df.sample(frac=args.data_frac, random_state=1, ignore_index=True)

    now = datetime.now().strftime("%Y-%m-%d")
    args.output_path = Path(f"{args.output_dir}/EDL_PROTO/{args.dataset}_{args.label}/fold_{args.n_folds}/{now}")
    os.makedirs(args.output_path, exist_ok=True)
    print(f"Output path: {args.output_path}")

    args_dict = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in args.__dict__.items()
        if key != "df"
    }
    with open(args.output_path / "args.yaml", "w") as f:
        yaml.safe_dump(args_dict, f, default_flow_style=False)

    single_internal_val = args.n_folds == 0
    if single_internal_val:
        print(
            "[Auto-Config] n_folds=0 detected. Creating an internal validation "
            "split from train cohorts; test cohorts remain held out."
        )
        train_df, val_df = adaptive_stratified_train_val_split(
            dev_df,
            val_frac=args.kfold0_val_frac,
            max_val_frac=args.kfold0_val_max_frac,
            args=args,
            context="EDL_PROTO n_folds=0 internal train/val split",
        )
        split_iter = [(train_df, val_df)]
        total_folds = 1
    else:
        split_iter = generator_cross_val_folds(
            dev_df,
            args.n_folds,
            args.label,
            random_state=args.seed,
        )
        total_folds = args.n_folds

    all_val_results = []
    fold_assignments = []

    for fold, (train_df, val_df) in enumerate(split_iter):
        if fold < args.start_fold:
            continue

        print(f'\n{"=" * 60}')
        print(f"  EDL_PROTO Fold {fold} / {total_folds}")
        print(f'{"=" * 60}')

        args.cur_fold = fold
        seed_all(args.seed + fold)

        path_results_fold = args.output_path / f"fold_{fold}"
        path_results_fold.mkdir(parents=True, exist_ok=True)

        valid_split_name = "val"
        print(f"Train: {len(train_df)}, {valid_split_name.capitalize()}: {len(val_df)}")

        train_loader = MIL_dataloader(train_df, "train", args)
        valid_loader = MIL_dataloader(val_df, valid_split_name, args)

        pretrained_checkpoint = resolve_mil_checkpoint(args.resume, fold)
        if args.resume is not None and pretrained_checkpoint is None:
            print(f"[EDL_PROTO] Warning: no checkpoint found under {args.resume} for fold {fold}; training from scratch.")

        model, loaded_proto_checkpoint = build_edl_proto_model(args, pretrained_checkpoint)
        model.to(device)

        if not loaded_proto_checkpoint:
            initialize_prototypes_from_train_split(model, train_df, args, device, fold)

        if args.train_edl_only:
            freeze_mil_backbone_train_edl_only(model)
            print("[EDL_PROTO] Freeze mode enabled: training only prototype EDL head(s); MIL backbone is frozen.")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

        trainable_parameters = [p for p in model.parameters() if p.requires_grad]
        if not trainable_parameters:
            raise RuntimeError("No trainable parameters found. Check the Prototype+EDL freeze configuration.")

        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=args.lr,
            weight_decay=args.weight_decay,
        )

        total_steps = len(train_loader) * args.epochs
        warmup_steps = len(train_loader) if args.warmup_epochs == 1 else 10
        warmup_steps = 0 if total_steps <= 1 else min(warmup_steps, total_steps - 1)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
        )
        scaler = torch.cuda.amp.GradScaler(enabled=bool(args.apex) and device.type == "cuda")

        class_weights = None
        if getattr(args, "weighted_BCE", "n") == "y":
            class_weights = get_edl_class_weights(train_df, args.label)

        criterion = EDLCombinedLoss(
            num_classes=2,
            kl_weight=args.edl_kl_weight,
            annealing_start=args.edl_annealing_start,
            annealing_epochs=args.edl_annealing_epochs,
            class_weights=class_weights,
        )

        val_stats, best_checkpoint_path = edl_train_loop(
            train_loader,
            valid_loader,
            model,
            optimizer,
            scheduler,
            scaler,
            criterion,
            path_results_fold,
            args,
            device,
            valid_split_name=valid_split_name,
        )

        fold_summary = {
            "fold": fold,
            "auc_roc": val_stats["auc_roc"],
            "f1": val_stats["f1"],
            "bacc": val_stats["bacc"],
            "loss": val_stats["loss"],
            "eval_source": "internal_val" if single_internal_val else "cross_val",
        }
        all_val_results.append(fold_summary)

        print(f"\nGenerating Prototype+EDL predictions with best model for fold {fold}...")
        checkpoint = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        all_split_dfs = []
        split_specs = [("train", train_df), ("val", val_df), ("test", test_df)]
        for split_name, split_df in split_specs:
            if split_df is None or len(split_df) == 0:
                continue

            loader = MIL_dataloader(split_df, "test", args)
            sample_results = edl_proto_predict(
                loader,
                model,
                args,
                device,
                desc=f"EDL_PROTO {split_name} predict",
            )
            pred_df = build_prediction_df(split_df, sample_results, split_name, fold, args)
            all_split_dfs.append(pred_df)

            if split_name == "val":
                for _, row in pred_df.iterrows():
                    fold_assignments.append(row.to_dict())

        if all_split_dfs:
            fold_pred_df = pd.concat(all_split_dfs, ignore_index=True)
            fold_pred_df.to_csv(
                path_results_fold / f"{args.dataset}_edl_proto_predictions_fold_{fold}.csv",
                index=False,
            )
            print(f"Saved fold {fold} Prototype+EDL predictions: {len(fold_pred_df)} samples")

        del model
        clear_memory()

    summary_df = pd.DataFrame(all_val_results)
    if len(summary_df) > 1:
        metric_cols = [col for col in summary_df.columns if col not in ["fold", "eval_source"]]
        mean_std = summary_df[metric_cols].agg(["mean", "std"]).reset_index(drop=True)
        mean_std["fold"] = ["mean", "std"]
        mean_std["eval_source"] = "summary"
        summary_df = pd.concat([summary_df, mean_std], ignore_index=True)

    summary_df.to_csv(args.output_path / "edl_proto_results_summary.csv", index=False)
    print(f"\nResults summary saved to {args.output_path / 'edl_proto_results_summary.csv'}")
    print(summary_df.to_string())

    if fold_assignments:
        fold_df = pd.DataFrame(fold_assignments)
        fold_df.to_csv(
            args.output_path / f"{args.dataset}_edl_proto_val_fold_assignments.csv",
            index=False,
        )
        print(f"Fold assignments saved ({len(fold_df)} validation samples)")

    return args.output_path


def main():
    args = config()
    args.edl_proto_normalize = args.edl_proto_normalize == "y"

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print(f"[INFO] Using GPU {args.gpu_id}")

    seed_all(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    args.apex = True if args.apex == "y" else False

    if hasattr(args, "df"):
        del args.df
    torch.cuda.empty_cache()

    output_path = do_edl_proto_training(args, device)

    print("\n" + "=" * 60)
    print("  Training complete. Starting automatic Prototype+EDL testing...")
    print("=" * 60)

    from edl_proto_test import run_edl_proto_test

    test_output_dir = output_path / "edl_proto_test_results"
    run_edl_proto_test(args, device, checkpoint_dir=output_path, output_dir=test_output_dir)

    print("\n===== Prototype+EDL Training + Testing Pipeline Complete =====")


if __name__ == "__main__":
    main()
