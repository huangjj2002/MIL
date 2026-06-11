import argparse
import gc
import math
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml
from tqdm import tqdm

from Datasets.dataset_utils import MIL_dataloader
from MIL import build_model
from MIL.edl_losses import EDLCombinedLoss
from MIL.edl_proto_dirichlet_models import (
    BagEmbeddingPrototypeEDLModel,
    MIL_PrototypeDirichletEDL_Wrapper,
)
from edl_train import (
    LinearWarmupCosineAnnealingLR,
    _get_checkpoint_state_dict,
    _infer_bag_embedding_dim,
    _print_load_summary,
    build_train_eval_loader,
    config as base_edl_config,
    format_fold_label,
    freeze_mil_backbone_train_edl_only,
    get_train_curve_metadata,
    resolve_mil_checkpoint,
)
from edl_proto_train import (
    _add_proto_args,
    _move_inputs_to_device,
    prototype_regularization_loss,
)
from utils.data_split_utils import (
    adaptive_stratified_train_val_split,
    generator_cross_val_folds,
    split_df_by_cohorts,
)
from utils.generic_utils import AverageMeter, clear_memory, seed_all
from utils.metrics import auroc, evaluate_metrics


DIRICHLET_DIAGNOSTIC_KEYS = [
    "ce_loss",
    "data_loss",
    "unweighted_ce_loss",
    "class_weighted_ce_loss",
    "focal_ce_loss",
    "kl_loss",
    "annealing",
    "wrong_evidence_penalty",
    "margin_violation_mean",
    "total_evidence_mean",
    "focal_factor_mean",
    "sample_weight_mean",
    "focal_weighted_denominator",
    "total_loss",
    "class0_ce_loss_mean",
    "class0_weighted_ce_loss_mean",
    "class0_focal_weighted_ce_loss_mean",
    "class0_focal_factor_mean",
    "class0_wrong_evidence_penalty_mean",
    "class1_ce_loss_mean",
    "class1_weighted_ce_loss_mean",
    "class1_focal_weighted_ce_loss_mean",
    "class1_focal_factor_mean",
    "class1_wrong_evidence_penalty_mean",
]


def config():
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


class PrototypeDirichletLoss(torch.nn.Module):
    def __init__(self, args, class_weights=None):
        super().__init__()
        wrong_balanced = getattr(args, "edl_wrong_evidence_class_balanced", "y")
        if isinstance(wrong_balanced, str):
            wrong_balanced = wrong_balanced == "y"
        self.loss = EDLCombinedLoss(
            num_classes=2,
            kl_weight=float(args.edl_kl_weight),
            annealing_start=int(args.edl_annealing_start),
            annealing_epochs=int(args.edl_annealing_epochs),
            class_weights=class_weights,
            focal_gamma=float(args.edl_focal_gamma),
            wrong_evidence_penalty_weight=float(args.edl_wrong_evidence_penalty_weight),
            wrong_evidence_margin=float(args.edl_wrong_evidence_margin),
            wrong_evidence_class_balanced=wrong_balanced,
            loss_weight_normalization=args.edl_loss_weight_normalization,
        )

    def get_annealing_coeff(self, epoch):
        return self.loss.get_annealing_coeff(epoch)

    def forward(self, head_output, target, epoch=0):
        return self.loss(head_output["alpha"], target, epoch=epoch)


def build_dirichlet_criterion(args, class_weights=None):
    return PrototypeDirichletLoss(args, class_weights=class_weights)


def get_dirichlet_class_weights(train_df, label_col):
    labels = train_df[label_col].astype(int)
    num_pos = int((labels == 1).sum())
    num_neg = int((labels == 0).sum())
    if num_pos <= 0:
        print("[EDL_PROTO] Warning: no positive samples in this training fold; using unweighted EDL loss.")
        return None

    pos_weight = float(num_neg / num_pos)
    print(
        f"[EDL_PROTO] Weighted EDL enabled: "
        f"neg={num_neg}, pos={num_pos}, pos_weight={pos_weight:.4f}"
    )
    return [1.0, pos_weight]


def _new_loss_meters():
    return {key: AverageMeter() for key in DIRICHLET_DIAGNOSTIC_KEYS}


def _is_finite_number(value):
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _update_loss_meters(meters, loss_dict, batch_size):
    for key, meter in meters.items():
        value = loss_dict.get(key)
        if value is None or not _is_finite_number(value):
            continue
        weight = batch_size
        if key.startswith("class0_"):
            weight = int(loss_dict.get("class0_n", 0))
        elif key.startswith("class1_"):
            weight = int(loss_dict.get("class1_n", 0))
        if weight > 0:
            meter.update(float(value), weight)


def _loss_meter_averages(meters):
    return {key: meter.avg for key, meter in meters.items() if meter.count > 0}


def _append_epoch_stats(history, stats):
    for key in history:
        if key in stats:
            history[key].append(stats[key])
        elif key not in {"loss", "f1", "bacc", "auc_roc", "lr"}:
            history[key].append(float("nan"))


def _init_epoch_history(include_lr=False):
    history = {"loss": [], "f1": [], "bacc": [], "auc_roc": []}
    if include_lr:
        history["lr"] = []
    for key in DIRICHLET_DIAGNOSTIC_KEYS:
        history[key] = []
    return history


def save_dirichlet_loss_curve(train_results, val_results, output_path, train_prefix="train", train_label="Train", plot_title="Prototype-EDL Loss Curve"):
    n_epochs = len(train_results["loss"])
    curve_data = {
        "epoch": list(range(1, n_epochs + 1)),
        f"{train_prefix}_loss": train_results["loss"],
        "val_loss": val_results["loss"],
        f"{train_prefix}_auc_roc": train_results["auc_roc"],
        "val_auc_roc": val_results["auc_roc"],
        f"{train_prefix}_f1": train_results["f1"],
        "val_f1": val_results["f1"],
        f"{train_prefix}_bacc": train_results["bacc"],
        "val_bacc": val_results["bacc"],
        "lr": train_results["lr"],
    }
    base_keys = {"loss", "auc_roc", "f1", "bacc", "lr"}
    for key, values in train_results.items():
        if key not in base_keys and len(values) == n_epochs:
            curve_data[f"{train_prefix}_{key}"] = values
    for key, values in val_results.items():
        if key not in base_keys and len(values) == n_epochs:
            curve_data[f"val_{key}"] = values

    curve_df = pd.DataFrame(curve_data)
    curve_df.to_csv(output_path / "edl_proto_loss_curve.csv", index=False)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5.75))
        ax.plot(
            curve_df["epoch"],
            curve_df[f"{train_prefix}_loss"],
            color="#1f77b4",
            linewidth=2.2,
            label=f"{train_label.lower()} loss",
        )
        ax.plot(
            curve_df["epoch"],
            curve_df["val_loss"],
            color="#d62728",
            linewidth=2.2,
            label="val loss",
        )
        ax.set_xlabel("epoch")
        ax.set_ylabel("loss")
        ax.set_title(plot_title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc="upper right", frameon=False)
        fig.tight_layout()
        fig.savefig(output_path / "edl_proto_loss_curve.png", dpi=200)
        plt.close(fig)
    except Exception as exc:
        print(f"[EDL_PROTO] Warning: failed to save loss curve plot: {exc}")


def _looks_like_wrapped_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        return False
    return any(key.startswith("mil_model.") for key in state_dict.keys())


def _looks_like_dirichlet_proto_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        return False
    return any(
        ("prototypes" in key or "proto_strength" in key or "raw_gamma" in key)
        for key in state_dict.keys()
    )


def build_edl_proto_dirichlet_model(args, checkpoint_path=None):
    args.n_class = 1
    if args.feature_extraction == "online" and not getattr(args, "clip_chk_pt_path", None):
        raise ValueError(
            "--clip_chk_pt_path is required when --feature_extraction online "
            "so the Mammo-CLIP image encoder/backbone can be initialized."
        )

    checkpoint_state = None
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_file():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            checkpoint_state = _get_checkpoint_state_dict(checkpoint)
        else:
            print(f"[EDL_PROTO] Warning: checkpoint not found at {checkpoint_path}; training from scratch.")

    is_wrapped_checkpoint = _looks_like_wrapped_state_dict(checkpoint_state)
    is_proto_checkpoint = _looks_like_dirichlet_proto_state_dict(checkpoint_state)

    if args.feature_extraction == "bag_embedding":
        model = BagEmbeddingPrototypeEDLModel(
            in_features=_infer_bag_embedding_dim(args),
            edl_dropout=args.edl_dropout,
            proto_k=args.edl_proto_k,
            proto_topk=args.edl_proto_topk,
            proto_normalize=args.edl_proto_normalize,
            proto_gamma_init=args.edl_proto_gamma_init,
        )
        if checkpoint_state is not None and is_proto_checkpoint:
            load_msg = model.load_state_dict(checkpoint_state, strict=False)
            print(f"[EDL_PROTO] Loaded Prototype-EDL checkpoint from: {checkpoint_path}")
            _print_load_summary("[EDL_PROTO][load]", load_msg)
        elif checkpoint_state is not None:
            print(
                f"[EDL_PROTO] Warning: {checkpoint_path} is not a Prototype-EDL "
                "checkpoint for bag_embedding; starting the head from scratch."
            )
        return model, checkpoint_state is not None and is_proto_checkpoint

    mil_model = build_model(args)

    if checkpoint_state is not None and not is_wrapped_checkpoint:
        load_msg = mil_model.load_state_dict(checkpoint_state, strict=False)
        print(f"[EDL_PROTO] Loaded pretrained MIL backbone from: {checkpoint_path}")
        _print_load_summary("[EDL_PROTO][MIL load]", load_msg)

    model = MIL_PrototypeDirichletEDL_Wrapper(
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
            print(f"[EDL_PROTO] Loaded Prototype-EDL checkpoint from: {checkpoint_path}")
        else:
            print(
                "[EDL_PROTO] Loaded wrapped model weights; "
                "prototype heads will be initialized separately."
            )
        _print_load_summary("[EDL_PROTO][wrapped load]", load_msg)

    return model, is_proto_checkpoint


def edl_proto_dirichlet_train_fn(train_loader, model, criterion, optimizer, epoch, args, scheduler, scaler, device):
    model.train()
    model.is_training = True

    losses = AverageMeter()
    ce_losses = AverageMeter()
    kl_losses = AverageMeter()
    proto_losses = AverageMeter()
    proto_attract_losses = AverageMeter()
    proto_separation_losses = AverageMeter()
    proto_diversity_losses = AverageMeter()
    loss_meters = _new_loss_meters()

    progress_iter = tqdm(
        enumerate(train_loader),
        desc=f"[{epoch + 1:03d}/{args.epochs:03d} EDL_PROTO train]",
        total=len(train_loader),
    )

    targs = []
    probs_list = []
    preds_list = []

    for _, data in progress_iter:
        inputs = _move_inputs_to_device(data, device, non_blocking=False)
        labels = data["y"].long().to(device)
        batch_size = labels.size(0)

        amp_enabled = bool(args.apex) and device.type == "cuda"
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            edl_out = model(inputs)
            loss, loss_dict = criterion(edl_out, labels, epoch=epoch)
            for side_out in edl_out.get("side_outputs", {}).values():
                side_loss, _ = criterion(side_out, labels, epoch=epoch)
                loss = loss + side_loss

            proto_reg, proto_dict = prototype_regularization_loss(model, edl_out, labels, args)
            loss = loss + proto_reg

        losses.update(loss.item(), batch_size)
        ce_losses.update(loss_dict["ce_loss"], batch_size)
        kl_losses.update(loss_dict["kl_loss"], batch_size)
        _update_loss_meters(loss_meters, loss_dict, batch_size)
        proto_losses.update(float(proto_dict["proto_reg"].item()), batch_size)
        proto_attract_losses.update(float(proto_dict["proto_attract"].item()), batch_size)
        proto_separation_losses.update(float(proto_dict["proto_separate"].item()), batch_size)
        proto_diversity_losses.update(float(proto_dict["proto_diverse"].item()), batch_size)

        scaler.scale(loss).backward()
        if args.clip_grad > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

        prob = edl_out["prob"].detach()
        pred_class = torch.argmax(prob, dim=-1)
        targs.append(labels.cpu().numpy())
        probs_list.append(prob[:, 1].cpu().numpy())
        preds_list.append(pred_class.cpu().numpy())

        cuda_mem = torch.cuda.memory_usage(device) if device.type == "cuda" and torch.cuda.is_available() else 0
        progress_iter.set_postfix({
            "loss": f"{losses.avg:.4f}",
            "ce": f"{ce_losses.avg:.4f}",
            "kl": f"{kl_losses.avg:.4f}",
            "ann": f"{loss_meters['annealing'].avg:.3f}",
            "ev": f"{loss_meters['total_evidence_mean'].avg:.3f}",
            "proto": f"{proto_losses.avg:.4f}",
            "CUDA-Mem": f"{cuda_mem}%",
        })

    targs = np.concatenate(targs)
    probs = np.concatenate(probs_list)
    preds = np.concatenate(preds_list)
    auc = auroc(targs, probs)
    f1, bacc = evaluate_metrics(targs, preds)

    train_stats = {
        "loss": losses.avg,
        "ce_loss": ce_losses.avg,
        "kl_loss": kl_losses.avg,
        "proto_reg_loss": proto_losses.avg,
        "proto_attract_loss": proto_attract_losses.avg,
        "proto_separation_loss": proto_separation_losses.avg,
        "proto_diversity_loss": proto_diversity_losses.avg,
        "auc_roc": auc,
        "f1": f1,
        "bacc": bacc,
        "lr": optimizer.param_groups[0]["lr"],
    }
    train_stats.update(_loss_meter_averages(loss_meters))
    return train_stats


@torch.no_grad()
def edl_proto_dirichlet_valid_fn(
    valid_loader,
    model,
    args,
    device,
    split="val",
    epoch=1,
    criterion_eval=None,
):
    model.eval()
    model.is_training = False

    losses = AverageMeter()
    if criterion_eval is None:
        criterion_eval = build_dirichlet_criterion(args, class_weights=None)
    loss_meters = _new_loss_meters()
    proto_losses = AverageMeter()
    proto_attract_losses = AverageMeter()
    proto_separation_losses = AverageMeter()
    proto_diversity_losses = AverageMeter()

    targs = []
    probs_list = []
    preds_list = []
    uncertainty_list = []
    evidence_list = []
    alpha_list = []
    sample_patient_ids = []
    sample_image_ids = []

    desc = f"[{epoch + 1:03d}/{args.epochs:03d} EDL_PROTO valid]" if split == "val" else None
    progress_iter = tqdm(enumerate(valid_loader), total=len(valid_loader), desc=desc)

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
            loss, loss_dict = criterion_eval(edl_out, labels, epoch=epoch)
            for side_out in edl_out.get("side_outputs", {}).values():
                side_loss, _ = criterion_eval(side_out, labels, epoch=epoch)
                loss = loss + side_loss
            proto_reg, proto_dict = prototype_regularization_loss(model, edl_out, labels, args)
            loss = loss + proto_reg

        losses.update(loss.item(), batch_size)
        _update_loss_meters(loss_meters, loss_dict, batch_size)
        proto_losses.update(float(proto_dict["proto_reg"].item()), batch_size)
        proto_attract_losses.update(float(proto_dict["proto_attract"].item()), batch_size)
        proto_separation_losses.update(float(proto_dict["proto_separate"].item()), batch_size)
        proto_diversity_losses.update(float(proto_dict["proto_diverse"].item()), batch_size)

        prob = edl_out["prob"].detach()
        pred_class = torch.argmax(prob, dim=-1)
        uncertainty = edl_out["uncertainty"].detach()
        evidence = edl_out["evidence"].detach()
        alpha = edl_out["alpha"].detach()

        targs.append(labels.cpu().numpy())
        probs_list.append(prob[:, 1].cpu().numpy())
        preds_list.append(pred_class.cpu().numpy())
        uncertainty_list.append(uncertainty.cpu().numpy())
        evidence_list.append(evidence.cpu().numpy())
        alpha_list.append(alpha.cpu().numpy())

    targs = np.concatenate(targs)
    probs = np.concatenate(probs_list)
    preds = np.concatenate(preds_list)
    evidence_array = np.concatenate(evidence_list)
    alpha_array = np.concatenate(alpha_list)

    auc_val = auroc(targs, probs)
    f1, bacc = evaluate_metrics(targs, preds)
    stats = {
        "loss": losses.avg,
        "auc_roc": auc_val,
        "f1": f1,
        "bacc": bacc,
        "evidence_0_mean": float(evidence_array[:, 0].mean()),
        "evidence_1_mean": float(evidence_array[:, 1].mean()),
        "alpha_0_mean": float(alpha_array[:, 0].mean()),
        "alpha_1_mean": float(alpha_array[:, 1].mean()),
        "proto_reg_loss": proto_losses.avg,
        "proto_attract_loss": proto_attract_losses.avg,
        "proto_separation_loss": proto_separation_losses.avg,
        "proto_diversity_loss": proto_diversity_losses.avg,
    }
    stats.update(_loss_meter_averages(loss_meters))

    sample_results = {
        "patient_id": sample_patient_ids,
        "image_id": sample_image_ids,
        "label": targs.tolist(),
        "score": probs.tolist(),
        "pred": preds.tolist(),
        "uncertainty": np.concatenate(uncertainty_list).tolist(),
        "evidence_0": evidence_array[:, 0].tolist(),
        "evidence_1": evidence_array[:, 1].tolist(),
        "alpha_0": alpha_array[:, 0].tolist(),
        "alpha_1": alpha_array[:, 1].tolist(),
    }

    return targs, preds, probs, stats, sample_results


def edl_proto_dirichlet_train_loop(train_loader, valid_loader, model, optimizer, scheduler, scaler,
                                   criterion, output_path, args, device, valid_split_name="val",
                                   train_eval_loader=None):
    best_aucroc = -float("inf")
    best_val_loss = float("inf")
    best_epoch = 0
    best_val_stats = None
    best_checkpoint_path = output_path / "best_model.pth"
    epochs_without_improvement = 0
    early_stop_patience = max(0, int(getattr(args, "early_stop_patience", 0)))
    early_stop_min_delta = max(0.0, float(getattr(args, "early_stop_min_delta", 0.0)))

    train_results = _init_epoch_history(include_lr=True)
    val_results = _init_epoch_history(include_lr=False)
    train_prefix, train_label = get_train_curve_metadata(train_eval_loader)
    for key in [
        "proto_reg_loss",
        "proto_attract_loss",
        "proto_separation_loss",
        "proto_diversity_loss",
    ]:
        train_results[key] = []
        val_results[key] = []

    for epoch in range(args.epochs):
        print(f"\n-------- Epoch {epoch + 1}/{args.epochs} --------")
        _ = time.time()

        train_stats = edl_proto_dirichlet_train_fn(
            train_loader, model, criterion, optimizer, epoch, args, scheduler, scaler, device
        )
        curve_train_stats = train_stats
        if train_eval_loader is not None:
            _, _, _, curve_train_stats, _ = edl_proto_dirichlet_valid_fn(
                train_eval_loader,
                model,
                args,
                device,
                split="train_eval",
                epoch=epoch,
                criterion_eval=criterion,
            )
            curve_train_stats["lr"] = train_stats["lr"]
        _, _, _, val_stats, _ = edl_proto_dirichlet_valid_fn(
            valid_loader,
            model,
            args,
            device,
            split=valid_split_name,
            epoch=epoch,
            criterion_eval=criterion,
        )

        valid_display_name = "Test" if valid_split_name == "test" else "Val"
        print(
            f"\n{train_label} Loss: {curve_train_stats['loss']:.4f} | "
            f"ProtoReg: {curve_train_stats['proto_reg_loss']:.4f} | "
            f"F1: {curve_train_stats['f1']:.4f} | BAcc: {curve_train_stats['bacc']:.4f} | "
            f"AUC: {curve_train_stats['auc_roc']:.4f}"
        )
        print(
            f"{valid_display_name}   Loss: {val_stats['loss']:.4f} | "
            f"ProtoReg: {val_stats['proto_reg_loss']:.4f} | "
            f"F1: {val_stats['f1']:.4f} | BAcc: {val_stats['bacc']:.4f} | "
            f"AUC: {val_stats['auc_roc']:.4f}"
        )

        _append_epoch_stats(train_results, curve_train_stats)
        _append_epoch_stats(val_results, val_stats)
        plot_title = f"EDL k={getattr(args, 'edl_proto_k', 0)} - {format_fold_label(output_path)}"
        save_dirichlet_loss_curve(
            train_results,
            val_results,
            output_path,
            train_prefix=train_prefix,
            train_label=train_label,
            plot_title=plot_title,
        )

        val_auc = val_stats["auc_roc"]
        val_auc_is_valid = np.isfinite(val_auc)
        annealing_coeff = criterion.get_annealing_coeff(epoch)
        annealing_complete = annealing_coeff >= 1.0
        should_save = (
            val_stats["loss"] < best_val_loss - early_stop_min_delta
            or best_val_stats is None
        )

        if should_save:
            epochs_without_improvement = 0
            if val_auc_is_valid:
                best_aucroc = val_auc
            best_val_loss = val_stats["loss"]
            best_val_stats = val_stats
            best_epoch = epoch + 1
            best_checkpoint_path = output_path / "best_model.pth"
            print(
                f"Epoch {epoch + 1} - Save best validation loss: {best_val_loss:.4f} "
                f"(AUC: {val_stats['auc_roc']:.4f})"
            )
            torch.save({
                "model": model.state_dict(),
                "epoch": epoch,
                "auroc": val_stats["auc_roc"],
                "f1": val_stats["f1"],
                "bacc": val_stats["bacc"],
                "dir_path": output_path,
            }, best_checkpoint_path)
        elif early_stop_patience > 0 and not annealing_complete:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(f"\nBest validation loss at epoch {best_epoch}: {best_val_loss:.4f}")

        if early_stop_patience > 0:
            if not annealing_complete:
                print(
                    "Early stopping paused until EDL annealing completes "
                    f"(annealing={annealing_coeff:.3f})."
                )
            else:
                print(
                    f"Early stopping: {epochs_without_improvement}/"
                    f"{early_stop_patience} epochs without improvement"
                )
                if epochs_without_improvement >= early_stop_patience:
                    print(
                        f"Early stopping triggered at epoch {epoch + 1}. "
                        f"Best epoch: {best_epoch}."
                    )
                    break

    torch.cuda.empty_cache()
    gc.collect()
    return best_val_stats, best_checkpoint_path


@torch.no_grad()
def initialize_prototypes_from_train_split(model, train_df, args, device, fold):
    if args.edl_proto_init != "kmeans":
        print("[EDL_PROTO] Prototype KMeans initialization skipped; using random init.")
        return

    heads = model.prototype_heads() if hasattr(model, "prototype_heads") else {}
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
                f"{prefix}_distance",
            ])
    return columns


def _append_proto_batch(proto_buffers, edl_out):
    if "topk_proto_idx" not in edl_out:
        return

    top_idx = edl_out["topk_proto_idx"].detach().cpu().numpy()
    top_evidence = edl_out["topk_proto_evidence"].detach().cpu().numpy()
    top_similarity = edl_out["topk_proto_similarity"].detach().cpu().numpy()
    top_distances = edl_out["topk_proto_distances"].detach().cpu().numpy()

    for class_idx in range(top_idx.shape[1]):
        for rank_idx in range(top_idx.shape[2]):
            prefix = f"proto_c{class_idx}_top{rank_idx + 1}"
            proto_buffers.setdefault(f"{prefix}_idx", []).append(top_idx[:, class_idx, rank_idx])
            proto_buffers.setdefault(f"{prefix}_evidence", []).append(top_evidence[:, class_idx, rank_idx])
            proto_buffers.setdefault(f"{prefix}_similarity", []).append(top_similarity[:, class_idx, rank_idx])
            proto_buffers.setdefault(f"{prefix}_distance", []).append(top_distances[:, class_idx, rank_idx])


@torch.no_grad()
def edl_proto_dirichlet_predict(loader, model, args, device, desc="EDL_PROTO predict"):
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
        "predicted_class": lambda x: int(x.mean() >= 0.5),
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


def run_edl_proto_dirichlet_test(args, device, checkpoint_dir, output_dir):
    checkpoint_dir = Path(checkpoint_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    args.data_dir = Path(args.data_dir)
    args.n_class = 1
    df = pd.read_csv(args.data_dir / args.csv_file).fillna(0)

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

    print("\n===== Predicting on development data with Prototype-EDL =====")
    all_dev_results = []
    dev_results_df = None
    test_ensemble = None

    for fold in range(total_folds):
        print(f"\n--- Fold {fold} ---")
        ckpt_path = checkpoint_dir / f"fold_{fold}" / "best_model.pth"
        if not ckpt_path.exists():
            print(f"Warning: checkpoint not found at {ckpt_path}, skipping fold {fold}")
            continue

        model, _ = build_edl_proto_dirichlet_model(args, ckpt_path)
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
        val_results = edl_proto_dirichlet_predict(
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
        print("\n===== Predicting on test data with Prototype-EDL =====")
        test_all_fold_results = []

        for fold in range(total_folds):
            print(f"\n--- Test with Fold {fold} model ---")
            ckpt_path = checkpoint_dir / f"fold_{fold}" / "best_model.pth"
            if not ckpt_path.exists():
                print(f"Warning: checkpoint not found at {ckpt_path}, skipping")
                continue

            model, _ = build_edl_proto_dirichlet_model(args, ckpt_path)
            model.to(device)

            test_loader = MIL_dataloader(test_df, "test", args)
            test_results = edl_proto_dirichlet_predict(
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

    print("\n===== Prototype-EDL Test Complete =====")
    return output_dir


def do_edl_proto_dirichlet_training(args, device):
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
    args.output_path = Path(
        f"{args.output_dir}/EDL_PROTO_DIRICHLET/{args.dataset}_{args.label}/fold_{args.n_folds}/{now}"
    )
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
        train_eval_loader = build_train_eval_loader(train_df, args)
        valid_loader = MIL_dataloader(val_df, valid_split_name, args)

        pretrained_checkpoint = resolve_mil_checkpoint(args.resume, fold)
        if args.resume is not None and pretrained_checkpoint is None:
            print(f"[EDL_PROTO] Warning: no checkpoint found under {args.resume} for fold {fold}; training from scratch.")

        model, loaded_proto_checkpoint = build_edl_proto_dirichlet_model(args, pretrained_checkpoint)
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
            raise RuntimeError("No trainable parameters found. Check the Prototype-EDL freeze configuration.")

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
            class_weights = get_dirichlet_class_weights(train_df, args.label)

        criterion = build_dirichlet_criterion(args, class_weights=class_weights)

        val_stats, best_checkpoint_path = edl_proto_dirichlet_train_loop(
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
            train_eval_loader=train_eval_loader,
        )

        fold_summary = {
            "fold": fold,
            "auc_roc": val_stats["auc_roc"],
            "f1": val_stats["f1"],
            "bacc": val_stats["bacc"],
            "loss": val_stats["loss"],
            "eval_source": "internal_val" if single_internal_val else "cross_val",
        }
        for key in DIRICHLET_DIAGNOSTIC_KEYS:
            if key in val_stats:
                fold_summary[key] = val_stats[key]
        for key in [
            "proto_reg_loss",
            "proto_attract_loss",
            "proto_separation_loss",
            "proto_diversity_loss",
        ]:
            if key in val_stats:
                fold_summary[key] = val_stats[key]
        for key in ["evidence_0_mean", "evidence_1_mean", "alpha_0_mean", "alpha_1_mean"]:
            if key in val_stats:
                fold_summary[key] = val_stats[key]
        all_val_results.append(fold_summary)

        print(f"\nGenerating Prototype-EDL predictions with best model for fold {fold}...")
        checkpoint = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        model.eval()

        all_split_dfs = []
        split_specs = [("train", train_df), ("val", val_df), ("test", test_df)]
        for split_name, split_df in split_specs:
            if split_df is None or len(split_df) == 0:
                continue

            loader = MIL_dataloader(split_df, "test", args)
            sample_results = edl_proto_dirichlet_predict(
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
            print(f"Saved fold {fold} Prototype-EDL predictions: {len(fold_pred_df)} samples")

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

    output_path = do_edl_proto_dirichlet_training(args, device)

    print("\n" + "=" * 60)
    print("  Training complete. Starting automatic Prototype-EDL testing...")
    print("=" * 60)

    test_output_dir = output_path / "edl_proto_test_results"
    run_edl_proto_dirichlet_test(args, device, checkpoint_dir=output_path, output_dir=test_output_dir)

    print("\n===== Prototype-EDL Training + Testing Pipeline Complete =====")


if __name__ == "__main__":
    main()
