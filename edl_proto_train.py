

import argparse
import gc
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
from MIL.edl_proto_models import BagEmbeddingPrototypeDSTModel, MIL_EDL_Prototype_Wrapper
from edl_train import (
    LinearWarmupCosineAnnealingLR,
    EDL_LOSS_DIAGNOSTIC_KEYS,
    _append_epoch_stats,
    _get_checkpoint_state_dict,
    _infer_bag_embedding_dim,
    _init_epoch_history,
    _loss_meter_averages,
    _new_loss_meters,
    _print_load_summary,
    _update_loss_meters,
    build_edl_criterion,
    build_train_eval_loader,
    config as base_edl_config,
    edl_valid_fn,
    freeze_mil_backbone_train_edl_only,
    get_train_curve_metadata,
    get_edl_class_weights,
    format_fold_label,
    resolve_mil_checkpoint,
    save_loss_curve,
)
from utils.data_split_utils import (
    adaptive_stratified_train_val_split,
    generator_cross_val_folds,
    split_df_by_cohorts,
)
from utils.generic_utils import AverageMeter, clear_memory, seed_all
from utils.metrics import auroc, evaluate_metrics


def _add_proto_args(parser):
    parser.add_argument("--edl_proto_k", default=4, type=int,
                        help="Number of prototypes per class.")
    parser.add_argument("--edl_proto_topk", default=3, type=int,
                        help="Number of top prototypes exported per class.")
    parser.add_argument("--edl_proto_gamma_init", default=1.0, type=float,
                        help=(
                            "Initial trainable DST distance sharpness for prototype similarity; "
                            "this is not the separation/diversity regularization margin."
                        ))
    parser.add_argument("--edl_proto_alpha_init", default=0.0, type=float,
                        help="Initial DST prototype reliability logit.")
    parser.add_argument("--edl_proto_normalize", default="y", choices=["y", "n"],
                        help="Normalize embeddings and prototypes before distance computation.")
    parser.add_argument("--edl_proto_init", default="embedding_bank",
                        choices=[
                            "fold_best_scores",
                            "embedding_bank",
                            "best_samples",
                            "kmeans",
                            "random",
                        ],
                        help=(
                            "Prototype initialization method. fold_best_scores selects real samples "
                            "from the current fold's training patients using cached original-MIL "
                            "scores; embedding_bank loads a legacy global prototype_bank.csv; "
                            "best_samples scores the current model before training."
                        ))
    parser.add_argument("--edl_proto_attract_weight", default=0.1, type=float,
                        help="Weight for pulling samples toward same-class prototypes.")
    parser.add_argument("--edl_proto_separation_weight", default=0.1, type=float,
                        help="Weight for pushing samples away from opposite-class prototypes.")
    parser.add_argument("--edl_proto_diversity_weight", default=0.01, type=float,
                        help="Weight for keeping same-class prototypes diverse.")
    parser.add_argument("--edl_proto_gamma_sep", "--edl-proto-gamma-sep",
                        default=None, type=float,
                        help="Hinge distance margin gamma_sep used only by prototype separation loss.")
    parser.add_argument("--edl_proto_gamma_div", "--edl-proto-gamma-div",
                        default=None, type=float,
                        help="Hinge distance margin gamma_div used only by prototype diversity loss.")
    parser.add_argument("--edl_proto_margin", default=None, type=float,
                        help=(
                            "Legacy shared separation/diversity margin. It is used as the fallback "
                            "for either new gamma argument that is not supplied; default fallback is 1.0."
                        ))
    parser.add_argument("--edl_proto_balance_classes", default="y", choices=["y", "n"],
                        help="Average prototype attraction/separation by class instead of by sample.")
    parser.add_argument("--edl_proto_allow_patient_overlap", default="n", choices=["y", "n"],
                        help=(
                            "Emergency exploratory override only: permit prototype-source patient IDs "
                            "to overlap validation or held-out test patients. This invalidates leakage-free "
                            "evaluation and is recorded in args.yaml."
                        ))


def _resolve_proto_margins(args):
    legacy_margin = getattr(args, "edl_proto_margin", None)
    shared_fallback = 1.0 if legacy_margin is None else float(legacy_margin)
    separation_margin = getattr(args, "edl_proto_gamma_sep", None)
    diversity_margin = getattr(args, "edl_proto_gamma_div", None)
    separation_margin = shared_fallback if separation_margin is None else float(separation_margin)
    diversity_margin = shared_fallback if diversity_margin is None else float(diversity_margin)
    if separation_margin < 0.0 or diversity_margin < 0.0:
        raise ValueError("Prototype separation/diversity margins must be non-negative.")
    return separation_margin, diversity_margin


def config():

    proto_parser = argparse.ArgumentParser(add_help=False)
    _add_proto_args(proto_parser)
    if any(flag in sys.argv[1:] for flag in ("-h", "--help")):
        print("Prototype-DST specific arguments:")
        proto_parser.print_help()
        print("\nShared DST/MIL training arguments:")
    proto_args, remaining = proto_parser.parse_known_args()

    original_argv = sys.argv[:]
    try:
        sys.argv = [original_argv[0]] + remaining
        args = base_edl_config()
    finally:
        sys.argv = original_argv

    for key, value in vars(proto_args).items():
        setattr(args, key, value)
    args.edl_proto_gamma_sep, args.edl_proto_gamma_div = _resolve_proto_margins(args)
    return args


def _looks_like_wrapped_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        return False
    return any(key.startswith("mil_model.") for key in state_dict.keys())


def _looks_like_edl_proto_state_dict(state_dict):
    if not isinstance(state_dict, dict):
        return False
    return any(
        (
            "ds_module" in key
            or "prototype" in key
            or "prototypes" in key
            or "proto_strength" in key
            or "raw_gamma" in key
        )
        for key in state_dict.keys()
    )


def build_edl_proto_model(args, checkpoint_path=None):

    args.n_class = 1
    if args.feature_extraction == "online" and not getattr(args, "clip_chk_pt_path", None):
        raise ValueError(
            "--clip_chk_pt_path is required when --feature_extraction online "
            "so the Mammo-CLIP image encoder/backbone can be initialized."
        )

    checkpoint_payload = None
    checkpoint_state = None
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_file():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            checkpoint_payload = checkpoint if isinstance(checkpoint, dict) else None
            checkpoint_state = _get_checkpoint_state_dict(checkpoint)
        else:
            print(f"[DST_PROTO] Warning: checkpoint not found at {checkpoint_path}; training from scratch.")

    is_wrapped_checkpoint = _looks_like_wrapped_state_dict(checkpoint_state)
    is_proto_checkpoint = _looks_like_edl_proto_state_dict(checkpoint_state)

    if args.feature_extraction == "bag_embedding":
        model = BagEmbeddingPrototypeDSTModel(
            in_features=_infer_bag_embedding_dim(args),
            edl_dropout=args.edl_dropout,
            proto_k=args.edl_proto_k,
            proto_topk=args.edl_proto_topk,
            proto_normalize=args.edl_proto_normalize,
            proto_gamma_init=args.edl_proto_gamma_init,
            proto_alpha_init=args.edl_proto_alpha_init,
        )
        if checkpoint_state is not None:
            load_msg = model.load_state_dict(checkpoint_state, strict=False)
            print(f"[DST_PROTO] Loaded Prototype-DST checkpoint from: {checkpoint_path}")
            _print_load_summary("[DST_PROTO][load]", load_msg)
            if checkpoint_payload is not None:
                _attach_prototype_bank(model, checkpoint_payload.get("prototype_bank"))
        return model, checkpoint_state is not None and is_proto_checkpoint

    mil_model = build_model(args)

    if checkpoint_state is not None and not is_wrapped_checkpoint:
        load_msg = mil_model.load_state_dict(checkpoint_state, strict=False)
        print(f"[DST_PROTO] Loaded pretrained MIL backbone from: {checkpoint_path}")
        _print_load_summary("[DST_PROTO][MIL load]", load_msg)

    model = MIL_EDL_Prototype_Wrapper(
        mil_model,
        edl_dropout=args.edl_dropout,
        proto_k=args.edl_proto_k,
        proto_topk=args.edl_proto_topk,
        proto_normalize=args.edl_proto_normalize,
        proto_gamma_init=args.edl_proto_gamma_init,
        proto_alpha_init=args.edl_proto_alpha_init,
    )

    if checkpoint_state is not None and is_wrapped_checkpoint:
        load_msg = model.load_state_dict(checkpoint_state, strict=False)
        if is_proto_checkpoint:
            print(f"[DST_PROTO] Loaded Prototype-DST checkpoint from: {checkpoint_path}")
        else:
            print(
                "[DST_PROTO] Loaded wrapped DST backbone weights; "
                "prototype heads will be initialized separately."
            )
        _print_load_summary("[DST_PROTO][wrapped load]", load_msg)
        if checkpoint_payload is not None:
            _attach_prototype_bank(model, checkpoint_payload.get("prototype_bank"))

    return model, is_proto_checkpoint


def _move_inputs_to_device(data, device, non_blocking=True):
    if isinstance(data["x"], dict):
        return {scale: tensor.to(device, non_blocking=non_blocking) for scale, tensor in data["x"].items()}
    if isinstance(data["x"], list):
        return [tensor.to(device, non_blocking=non_blocking) for tensor in data["x"]]
    return data["x"].to(device, non_blocking=non_blocking)


def _zero_like_loss(edl_out):
    if "dst_mass" in edl_out:
        return edl_out["dst_mass"].sum() * 0.0
    return edl_out["prob"].sum() * 0.0


def _single_output_proto_reg(edl_out, labels, separation_margin, balance_classes=True):
    """Sample-prototype attraction and opposite-class separation for one head."""
    if "prototype_distances" not in edl_out:
        zero = _zero_like_loss(edl_out)
        return zero, zero, zero

    distances = edl_out["prototype_distances"]
    labels = labels.long()
    batch_idx = torch.arange(labels.size(0), device=labels.device)
    class_attractions = []
    class_separations = []

    for class_idx in range(distances.size(1)):
        class_mask = labels == class_idx
        if not torch.any(class_mask):
            continue

        class_distances = distances[class_mask]
        same_distances = class_distances[:, class_idx]
        class_attractions.append(same_distances.min(dim=-1).values.mean())

        other_class_mask = torch.ones(distances.size(1), dtype=torch.bool, device=distances.device)
        other_class_mask[class_idx] = False
        nearest_opposite = class_distances[:, other_class_mask, :].reshape(
            class_distances.size(0), -1
        ).min(dim=-1).values
        class_separations.append(F.relu(float(separation_margin) - nearest_opposite).mean())

    if not class_attractions:
        zero = _zero_like_loss(edl_out)
        return zero, zero, zero

    if balance_classes:
        attraction = torch.stack(class_attractions).mean()
        separation = torch.stack(class_separations).mean()
    else:
        same_distances = distances[batch_idx, labels]
        attraction = same_distances.min(dim=-1).values.mean()
        if distances.size(1) == 2:
            opposite_labels = 1 - labels
            opposite_distances = distances[batch_idx, opposite_labels]
            nearest_opposite = opposite_distances.min(dim=-1).values
        else:
            sample_class_mask = torch.ones(
                distances.size(0),
                distances.size(1),
                dtype=torch.bool,
                device=distances.device,
            )
            sample_class_mask[batch_idx, labels] = False
            nearest_opposite = distances[sample_class_mask].view(distances.size(0), -1).min(dim=-1).values
        separation = F.relu(float(separation_margin) - nearest_opposite).mean()

    return attraction, separation, attraction + separation


def _prototype_diversity_loss(model, diversity_margin):
    heads = model.prototype_heads() if hasattr(model, "prototype_heads") else {}
    losses = []

    for head in heads.values():
        prototypes = head.prototypes
        if head.normalize:
            prototypes = F.normalize(prototypes, dim=-1)

        for class_idx in range(prototypes.size(0)):
            class_prototypes = prototypes[class_idx]
            if class_prototypes.size(0) < 2:
                continue
            pairwise_distances = torch.cdist(
                class_prototypes.unsqueeze(0),
                class_prototypes.unsqueeze(0),
                p=2,
            ).squeeze(0).pow(2)
            mask = ~torch.eye(
                class_prototypes.size(0),
                dtype=torch.bool,
                device=class_prototypes.device,
            )
            losses.append(F.relu(float(diversity_margin) - pairwise_distances[mask]).mean())

    if losses:
        return torch.stack(losses).mean()

    first_param = next(model.parameters())
    return first_param.sum() * 0.0


def prototype_regularization_loss(model, edl_out, labels, args):
    separation_margin, diversity_margin = _resolve_proto_margins(args)
    attract_weight = float(getattr(args, "edl_proto_attract_weight", 0.0))
    separation_weight = float(getattr(args, "edl_proto_separation_weight", 0.0))
    diversity_weight = float(getattr(args, "edl_proto_diversity_weight", 0.0))
    balance_classes = getattr(args, "edl_proto_balance_classes", True)
    if isinstance(balance_classes, str):
        balance_classes = balance_classes == "y"

    head_outputs = [
        head_out
        for head_out in [edl_out] + list(edl_out.get("side_outputs", {}).values())
        if "prototype_distances" in head_out
    ]
    if not head_outputs:
        zero = _zero_like_loss(edl_out)
        diversity_loss = _prototype_diversity_loss(model, diversity_margin)
        total = diversity_weight * diversity_loss
        return total, {
            "proto_attract": zero.detach(),
            "proto_separate": zero.detach(),
            "proto_diverse": diversity_loss.detach(),
            "proto_reg": total.detach(),
        }

    attractions = []
    separations = []
    for head_out in head_outputs:
        attraction, separation, _ = _single_output_proto_reg(
            head_out,
            labels,
            separation_margin,
            balance_classes=balance_classes,
        )
        attractions.append(attraction)
        separations.append(separation)

    attraction_loss = torch.stack(attractions).mean()
    separation_loss = torch.stack(separations).mean()
    diversity_loss = _prototype_diversity_loss(model, diversity_margin)

    total = (
        attract_weight * attraction_loss
        + separation_weight * separation_loss
        + diversity_weight * diversity_loss
    )
    return total, {
        "proto_attract": attraction_loss.detach(),
        "proto_separate": separation_loss.detach(),
        "proto_diverse": diversity_loss.detach(),
        "proto_reg": total.detach(),
    }


def keep_frozen_mil_backbone_in_eval(model, args):

    if getattr(args, "train_edl_only", False) and hasattr(model, "mil_model"):
        model.mil_model.eval()


def edl_proto_train_fn(train_loader, model, criterion, optimizer, epoch, args, scheduler, scaler, device):

    model.train()
    keep_frozen_mil_backbone_in_eval(model, args)
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
        desc=f"[{epoch + 1:03d}/{args.epochs:03d} DST_PROTO train]",
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
            "wep": f"{loss_meters['wrong_evidence_penalty'].avg:.4f}",
            "viol": f"{loss_meters['margin_violation_mean'].avg:.4f}",
            "omega": f"{loss_meters['mass_omega_mean'].avg:.4f}",
            "proto": f"{proto_losses.avg:.4f}",
            "attr": f"{proto_attract_losses.avg:.4f}",
            "sep": f"{proto_separation_losses.avg:.4f}",
            "div": f"{proto_diversity_losses.avg:.4f}",
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
def evaluate_proto_regularization(loader, model, args, device, desc="DST_PROTO val proto-reg"):
    model.eval()
    model.is_training = False

    proto_losses = AverageMeter()
    proto_attract_losses = AverageMeter()
    proto_separation_losses = AverageMeter()
    proto_diversity_losses = AverageMeter()

    progress_iter = tqdm(enumerate(loader), desc=desc, total=len(loader))
    for _, data in progress_iter:
        inputs = _move_inputs_to_device(data, device, non_blocking=True)
        labels = data["y"].long().to(device)
        batch_size = labels.size(0)

        amp_enabled = bool(args.apex) and device.type == "cuda"
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            edl_out = model(inputs)
            proto_reg, proto_dict = prototype_regularization_loss(model, edl_out, labels, args)

        proto_losses.update(float(proto_dict["proto_reg"].item()), batch_size)
        proto_attract_losses.update(float(proto_dict["proto_attract"].item()), batch_size)
        proto_separation_losses.update(float(proto_dict["proto_separate"].item()), batch_size)
        proto_diversity_losses.update(float(proto_dict["proto_diverse"].item()), batch_size)

    return {
        "proto_reg_loss": proto_losses.avg,
        "proto_attract_loss": proto_attract_losses.avg,
        "proto_separation_loss": proto_separation_losses.avg,
        "proto_diversity_loss": proto_diversity_losses.avg,
    }


def edl_proto_train_loop(train_loader, valid_loader, model, optimizer, scheduler, scaler,
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
        start_time = time.time()

        train_stats = edl_proto_train_fn(
            train_loader, model, criterion, optimizer, epoch, args, scheduler, scaler, device
        )
        curve_train_stats = train_stats
        if train_eval_loader is not None:
            _, _, _, curve_train_stats, _ = edl_valid_fn(
                train_eval_loader,
                model,
                args,
                device,
                split="train_eval",
                epoch=epoch,
                criterion_eval=criterion,
            )
            train_eval_proto_stats = evaluate_proto_regularization(
                train_eval_loader,
                model,
                args,
                device,
                desc=f"[{epoch + 1:03d}/{args.epochs:03d} DST_PROTO train-eval proto-reg]",
            )
            curve_train_stats.update(train_eval_proto_stats)
            curve_train_stats["loss"] = curve_train_stats["loss"] + train_eval_proto_stats["proto_reg_loss"]
            curve_train_stats["lr"] = train_stats["lr"]
        _, _, _, val_stats, _ = edl_valid_fn(
            valid_loader,
            model,
            args,
            device,
            split=valid_split_name,
            epoch=epoch,
            criterion_eval=criterion,
        )
        val_proto_stats = evaluate_proto_regularization(
            valid_loader,
            model,
            args,
            device,
            desc=f"[{epoch + 1:03d}/{args.epochs:03d} DST_PROTO val proto-reg]",
        )
        val_stats.update(val_proto_stats)
        val_stats["loss"] = val_stats["loss"] + val_proto_stats["proto_reg_loss"]

        _ = time.time() - start_time
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
        plot_title = f"DST k={getattr(args, 'edl_proto_k', 0)} - {format_fold_label(output_path)}"
        save_loss_curve(
            train_results,
            val_results,
            output_path,
            train_prefix=train_prefix,
            train_label=train_label,
            plot_title=plot_title,
        )

        val_auc = val_stats["auc_roc"]
        val_auc_is_valid = np.isfinite(val_auc)
        annealing_coeff = (
            criterion.get_annealing_coeff(epoch)
            if hasattr(criterion, "get_annealing_coeff")
            else 1.0
        )
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
                "prototype_bank": _get_prototype_bank_records(model),
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


def _attach_prototype_bank(model, prototype_bank):
    if prototype_bank is None:
        return None

    if isinstance(prototype_bank, pd.DataFrame):
        bank_df = prototype_bank.copy()
    else:
        bank_df = pd.DataFrame(prototype_bank)

    if bank_df.empty:
        return None

    for col in ["head_name", "source_patient_id", "source_image_id", "prototype_id", "selection_method"]:
        if col in bank_df.columns:
            bank_df[col] = bank_df[col].astype(str)

    model._prototype_bank_df = bank_df.reset_index(drop=True)
    return model._prototype_bank_df


def _get_prototype_bank_df(model):
    bank_df = getattr(model, "_prototype_bank_df", None)
    if bank_df is None or len(bank_df) == 0:
        return None
    return bank_df.copy()


def _get_prototype_bank_records(model):
    bank_df = _get_prototype_bank_df(model)
    if bank_df is None:
        return None
    return bank_df.to_dict(orient="records")


def save_prototype_bank_csv(output_path, model):
    bank_df = _get_prototype_bank_df(model)
    if bank_df is None:
        return None
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    bank_df.to_csv(output_path, index=False)
    print(f"[DST_PROTO] Prototype bank saved: {output_path}")
    return output_path


def _normalize_prototype_matrix(values):
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.clip(norms, 1e-12, None)


def _embedding_bank_row_value(row, *keys, default=None):
    for key in keys:
        if key in row and not pd.isna(row[key]):
            value = row[key]
            if isinstance(value, np.generic):
                value = value.item()
            return value
    return default


def _load_head_prototypes_from_embedding_bank(head, head_name, bank_df, embeddings):
    if "head_name" in bank_df.columns:
        head_rows = bank_df[bank_df["head_name"].astype(str) == str(head_name)].copy()
        if head_rows.empty:
            head_rows = bank_df.copy()
    else:
        head_rows = bank_df.copy()

    prototype_vectors = []
    normalized_rows = []
    warnings_list = []

    for class_idx in range(head.num_classes):
        class_df = head_rows[head_rows["prototype_class"].astype(int) == class_idx].copy()
        if class_df.empty:
            raise RuntimeError(f"Prototype bank has no class {class_idx} rows for {head_name}.")

        sort_cols = [
            col
            for col in ["prototype_rank", "prototype_global_idx", "selection_rank"]
            if col in class_df.columns
        ]
        if sort_cols:
            class_df = class_df.sort_values(sort_cols)
        class_df = class_df.reset_index(drop=True)

        if len(class_df) < head.prototypes_per_class:
            warnings_list.append(
                f"class {class_idx} has {len(class_df)} bank prototypes; "
                f"repeating rows to fill {head.prototypes_per_class}"
            )

        for rank_idx in range(head.prototypes_per_class):
            row = class_df.iloc[rank_idx % len(class_df)].to_dict()
            embedding_row = int(_embedding_bank_row_value(
                row,
                "embedding_row",
                "embedding_start",
                default=-1,
            ))
            if embedding_row < 0 or embedding_row >= int(embeddings.shape[0]):
                raise RuntimeError(
                    f"Prototype bank row points to invalid embedding_row={embedding_row} "
                    f"for {head_name} class {class_idx} rank {rank_idx}."
                )
            prototype_vectors.append(np.asarray(embeddings[embedding_row], dtype=np.float32))
            global_idx = class_idx * head.prototypes_per_class + rank_idx
            row.update({
                "head_name": head_name,
                "prototype_global_idx": global_idx,
                "prototype_class": class_idx,
                "prototype_rank": rank_idx,
                "prototype_id": f"c{class_idx}_p{rank_idx}",
                "selection_rank": rank_idx,
                "selection_method": str(row.get("selection_method", "embedding_bank")),
                "embedding_row": embedding_row,
            })
            normalized_rows.append(row)

    prototype_array = np.asarray(prototype_vectors, dtype=np.float32)
    if head.normalize:
        prototype_array = _normalize_prototype_matrix(prototype_array)
    prototype_array = prototype_array.reshape(
        head.num_classes,
        head.prototypes_per_class,
        head.in_features,
    )
    head.set_prototypes_from_embeddings(prototype_array)
    return normalized_rows, warnings_list


def initialize_prototypes_from_embedding_bank(model, args):
    if getattr(args, "feature_extraction", None) != "bag_embedding":
        return None
    cache_dir = getattr(args, "embedding_cache_dir", None)
    if cache_dir is None:
        return None

    cache_dir = Path(cache_dir)
    bank_path = cache_dir / "prototype_bank.csv"
    embeddings_path = cache_dir / "embeddings.npy"
    if not bank_path.exists() or not embeddings_path.exists():
        return None

    heads = model.prototype_heads()
    if not heads:
        return None

    bank_df = pd.read_csv(bank_path)
    embeddings = np.load(embeddings_path, mmap_mode="r")
    bank_rows = []

    print(f"[DST_PROTO] Loading prototypes from embedding bank: {bank_path}")
    for head_name, head in heads.items():
        selected_rows, warnings_list = _load_head_prototypes_from_embedding_bank(
            head,
            head_name,
            bank_df,
            embeddings,
        )
        bank_rows.extend(selected_rows)
        print(
            f"[DST_PROTO] {head_name}: loaded {head.num_classes * head.prototypes_per_class} "
            f"prototypes from embedding bank."
        )
        for warning_text in warnings_list:
            print(f"[DST_PROTO] Warning: {head_name}: {warning_text}")

    return _attach_prototype_bank(model, bank_rows)


def _normalize_sample_id(value):
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    return str(value)


def _patient_id_set(frame):
    if frame is None or len(frame) == 0:
        return set()
    if "patient_id" not in frame.columns:
        raise ValueError("Fold-safe prototype selection requires a patient_id column.")
    return {_normalize_sample_id(value) for value in frame["patient_id"]}


def _select_fold_best_score_rows(
    metadata_df,
    train_df,
    val_df,
    test_df,
    label_col,
    prototypes_per_class,
):
    key_cols = ["patient_id", "image_id"]
    required_metadata = set(key_cols + ["embedding_start", "embedding_end", "origin_prediction_score"])
    missing_metadata = sorted(required_metadata.difference(metadata_df.columns))
    if missing_metadata:
        raise ValueError(
            "fold_best_scores metadata is missing columns: " + ", ".join(missing_metadata)
        )
    required_train = set(key_cols + [label_col])
    missing_train = sorted(required_train.difference(train_df.columns))
    if missing_train:
        raise ValueError(
            "fold_best_scores training data is missing columns: " + ", ".join(missing_train)
        )
    if int(prototypes_per_class) < 1:
        raise ValueError("prototypes_per_class must be positive for fold_best_scores.")

    train_patients = _patient_id_set(train_df)
    val_patients = _patient_id_set(val_df)
    test_patients = _patient_id_set(test_df)
    allow_overlap = getattr(train_df, "attrs", {}).get("edl_proto_allow_patient_overlap", False)
    overlap_messages = []
    train_val_overlap = train_patients.intersection(val_patients)
    train_test_overlap = train_patients.intersection(test_patients)
    if train_val_overlap:
        overlap_messages.append(
            "training/validation="
            f"{len(train_val_overlap)} (examples={sorted(train_val_overlap)[:5]})"
        )
    if train_test_overlap:
        overlap_messages.append(
            "training/held-out-test="
            f"{len(train_test_overlap)} (examples={sorted(train_test_overlap)[:5]})"
        )
    if overlap_messages and not allow_overlap:
        raise ValueError(
            "Patient ID overlap during fold-safe prototype selection: "
            + "; ".join(overlap_messages)
            + ". Refuse to continue unless the explicit exploratory override is enabled."
        )
    if overlap_messages:
        print(
            "[DST_PROTO][WARNING] Patient-overlap override is enabled; prototype "
            "selection is NOT leakage-free (" + "; ".join(overlap_messages) + "). "
            "Do not use this run for paper results."
        )

    train_keys = train_df[key_cols + [label_col]].copy()
    metadata = metadata_df.drop(columns=[label_col], errors="ignore").copy()
    for frame in (train_keys, metadata):
        frame["patient_id"] = frame["patient_id"].map(_normalize_sample_id)
        frame["image_id"] = frame["image_id"].map(_normalize_sample_id)

    label_counts = train_keys.groupby(key_cols, dropna=False)[label_col].nunique()
    if (label_counts > 1).any():
        raise ValueError("Training data contains conflicting labels for a patient/image key.")
    train_keys = train_keys.drop_duplicates(key_cols, keep="first")
    if metadata.duplicated(key_cols).any():
        raise ValueError("Embedding metadata contains duplicate patient/image keys.")

    candidates = train_keys.merge(
        metadata,
        on=key_cols,
        how="left",
        validate="one_to_one",
        indicator=True,
    )
    missing_embeddings = candidates[candidates["_merge"] != "both"]
    if not missing_embeddings.empty:
        example = missing_embeddings.iloc[0]
        raise ValueError(
            "No cached embedding metadata for fold-training sample "
            f"patient_id={example['patient_id']}, image_id={example['image_id']}."
        )
    candidates = candidates.drop(columns="_merge")
    candidates[label_col] = pd.to_numeric(candidates[label_col], errors="raise").astype(int)
    if not candidates[label_col].isin([0, 1]).all():
        raise ValueError("fold_best_scores currently supports binary labels 0/1 only.")
    candidates["origin_prediction_score"] = pd.to_numeric(
        candidates["origin_prediction_score"], errors="raise"
    ).astype(float)
    if not np.isfinite(candidates["origin_prediction_score"]).all():
        raise ValueError("origin_prediction_score contains NaN or infinite values.")
    if not candidates["origin_prediction_score"].between(0.0, 1.0).all():
        raise ValueError("origin_prediction_score must lie in [0, 1].")

    labels = candidates[label_col].to_numpy(dtype=int)
    positive_scores = candidates["origin_prediction_score"].to_numpy(dtype=float)
    candidates["source_true_class_score"] = np.where(
        labels == 1,
        positive_scores,
        1.0 - positive_scores,
    )
    if "origin_predicted_class" in candidates.columns:
        predicted = pd.to_numeric(
            candidates["origin_predicted_class"], errors="raise"
        ).astype(int).to_numpy()
    else:
        predicted = (positive_scores >= 0.5).astype(int)
    candidates["source_predicted_class"] = predicted
    candidates["source_correct"] = predicted == labels

    selected = []
    for class_idx in (0, 1):
        class_rows = candidates[candidates[label_col] == class_idx].copy()
        if class_rows.empty:
            raise ValueError(f"No class {class_idx} fold-training samples are available for prototypes.")
        class_rows = class_rows.sort_values(
            ["source_correct", "source_true_class_score", "patient_id", "image_id"],
            ascending=[False, False, True, True],
            kind="mergesort",
        ).reset_index(drop=True)
        for rank_idx in range(int(prototypes_per_class)):
            row = class_rows.iloc[rank_idx % len(class_rows)].copy()
            row["prototype_class"] = class_idx
            row["prototype_rank"] = rank_idx
            row["selection_rank"] = rank_idx
            row["selection_method"] = "fold_best_scores"
            row["source_label"] = int(row[label_col])
            row["source_prediction_score"] = float(row["origin_prediction_score"])
            row["source_selection_score"] = float(row["source_true_class_score"])
            row["source_selection_class"] = class_idx
            row["source_split"] = "train"
            selected.append(row)

    selected_df = pd.DataFrame(selected).reset_index(drop=True)
    selected_patients = set(selected_df["patient_id"].map(_normalize_sample_id))
    if not selected_patients.issubset(train_patients):
        raise AssertionError("A selected prototype did not originate from the current fold training set.")
    if selected_patients.intersection(val_patients | test_patients) and not allow_overlap:
        raise AssertionError("A selected prototype patient appears in validation or held-out test data.")
    return selected_df


def initialize_prototypes_from_fold_best_scores(model, train_df, val_df, test_df, args):
    if getattr(args, "feature_extraction", None) != "bag_embedding":
        raise ValueError("fold_best_scores requires --feature_extraction bag_embedding.")
    cache_dir = Path(getattr(args, "embedding_cache_dir", ""))
    metadata_path = cache_dir / "metadata.csv"
    embeddings_path = cache_dir / "embeddings.npy"
    if not metadata_path.is_file() or not embeddings_path.is_file():
        raise FileNotFoundError(
            "fold_best_scores requires metadata.csv and embeddings.npy under "
            f"--embedding_cache_dir; got {cache_dir}."
        )

    # _select_fold_best_score_rows is intentionally dataframe-only for testing;
    # pass the opt-in leakage override through a dataframe attribute rather than
    # making it a default argument that could be enabled accidentally.
    train_df = train_df.copy()
    train_df.attrs["edl_proto_allow_patient_overlap"] = (
        getattr(args, "edl_proto_allow_patient_overlap", "n") == "y"
    )

    metadata = pd.read_csv(metadata_path, dtype={"patient_id": str, "image_id": str})
    embeddings = np.load(embeddings_path, mmap_mode="r")
    heads = model.prototype_heads()
    bank_rows = []
    for head_name, head in heads.items():
        selected = _select_fold_best_score_rows(
            metadata,
            train_df,
            val_df,
            test_df,
            args.label,
            head.prototypes_per_class,
        )
        vectors = []
        for _, row in selected.iterrows():
            start = int(row["embedding_start"])
            end = int(row["embedding_end"])
            if end - start != 1 or start < 0 or end > int(embeddings.shape[0]):
                raise ValueError(
                    "fold_best_scores expects one valid embedding row per image; "
                    f"got [{start}, {end}) for patient_id={row['patient_id']}, "
                    f"image_id={row['image_id']}."
                )
            vectors.append(np.asarray(embeddings[start], dtype=np.float32))

        prototype_array = np.asarray(vectors, dtype=np.float32)
        if prototype_array.shape[1] != head.in_features:
            raise ValueError(
                f"Embedding dimension {prototype_array.shape[1]} does not match "
                f"prototype head dimension {head.in_features}."
            )
        if head.normalize:
            prototype_array = _normalize_prototype_matrix(prototype_array)
        head.set_prototypes_from_embeddings(
            prototype_array.reshape(head.num_classes, head.prototypes_per_class, head.in_features)
        )

        for _, row in selected.iterrows():
            class_idx = int(row["prototype_class"])
            rank_idx = int(row["prototype_rank"])
            bank_rows.append({
                "head_name": head_name,
                "prototype_global_idx": class_idx * head.prototypes_per_class + rank_idx,
                "prototype_class": class_idx,
                "prototype_rank": rank_idx,
                "prototype_id": f"c{class_idx}_p{rank_idx}",
                "source_patient_id": str(row["patient_id"]),
                "source_image_id": str(row["image_id"]),
                "source_label": int(row["source_label"]),
                "source_prediction_score": float(row["source_prediction_score"]),
                "source_true_class_score": float(row["source_true_class_score"]),
                "source_selection_score": float(row["source_selection_score"]),
                "source_selection_class": int(row["source_selection_class"]),
                "source_predicted_class": int(row["source_predicted_class"]),
                "source_correct": bool(row["source_correct"]),
                "source_split": "train",
                "selection_rank": rank_idx,
                "selection_method": "fold_best_scores",
                "embedding_row": int(row["embedding_start"]),
            })
        print(
            f"[DST_PROTO] {head_name}: initialized {len(selected)} prototypes from "
            "current-fold training samples using original-MIL true-class scores."
        )
    return _attach_prototype_bank(model, bank_rows)


def _to_str_list(values):
    return ["" if value is None else str(value) for value in values]


def _select_best_sample_prototypes(head, bucket, head_name):
    embeddings = torch.cat(bucket["embeddings"], dim=0).float().cpu().numpy()
    labels = torch.cat(bucket["labels"], dim=0).long().cpu().numpy().astype(int)
    prediction_scores = torch.cat(bucket["prediction_scores"], dim=0).float().cpu().numpy()
    true_class_scores = torch.cat(bucket["true_class_scores"], dim=0).float().cpu().numpy()
    preds = torch.cat(bucket["preds"], dim=0).long().cpu().numpy().astype(int)
    patient_ids = _to_str_list(bucket["patient_id"])
    image_ids = _to_str_list(bucket["image_id"])
    source_rows = np.concatenate(bucket["source_row"], axis=0).astype(int)

    if embeddings.ndim != 2 or embeddings.shape[1] != head.in_features:
        raise ValueError(
            f"{head_name}: expected embeddings with shape (N, {head.in_features}), "
            f"got {embeddings.shape}."
        )
    if embeddings.shape[0] == 0:
        raise ValueError(f"{head_name}: cannot select prototypes from an empty embedding set.")

    prototype_vectors = []
    bank_rows = []
    warnings_list = []

    working_embeddings = embeddings.astype(np.float32, copy=True)
    if head.normalize:
        norms = np.linalg.norm(working_embeddings, axis=1, keepdims=True)
        working_embeddings = working_embeddings / np.clip(norms, 1e-12, None)

    global_order = np.lexsort((source_rows, -true_class_scores, -(preds == labels).astype(int)))

    for class_idx in range(head.num_classes):
        class_indices = np.where(labels == class_idx)[0]
        if len(class_indices) == 0:
            ordered = global_order
            warnings_list.append(
                f"class {class_idx} has no samples; reusing global high-confidence samples"
            )
        else:
            correct = (preds[class_indices] == labels[class_indices]).astype(int)
            ordered = class_indices[
                np.lexsort((source_rows[class_indices], -true_class_scores[class_indices], -correct))
            ]

        if len(ordered) < head.prototypes_per_class:
            warnings_list.append(
                f"class {class_idx} has {len(ordered)} samples; repeating selected samples "
                f"to fill {head.prototypes_per_class} prototypes"
            )

        for rank_idx in range(head.prototypes_per_class):
            source_idx = int(ordered[rank_idx % len(ordered)])
            prototype_vectors.append(working_embeddings[source_idx])
            global_idx = class_idx * head.prototypes_per_class + rank_idx
            bank_rows.append({
                "head_name": head_name,
                "prototype_global_idx": global_idx,
                "prototype_class": class_idx,
                "prototype_rank": rank_idx,
                "prototype_id": f"c{class_idx}_p{rank_idx}",
                "source_patient_id": patient_ids[source_idx],
                "source_image_id": image_ids[source_idx],
                "source_label": int(labels[source_idx]),
                "source_prediction_score": float(prediction_scores[source_idx]),
                "source_true_class_score": float(true_class_scores[source_idx]),
                "source_selection_score": float(true_class_scores[source_idx]),
                "source_selection_class": int(class_idx),
                "source_predicted_class": int(preds[source_idx]),
                "source_correct": bool(preds[source_idx] == labels[source_idx]),
                "source_train_row": int(source_rows[source_idx]),
                "selection_rank": rank_idx,
                "selection_method": "best_samples",
            })

    prototype_array = np.asarray(prototype_vectors, dtype=np.float32).reshape(
        head.num_classes,
        head.prototypes_per_class,
        head.in_features,
    )
    head.set_prototypes_from_embeddings(prototype_array)
    return bank_rows, warnings_list


@torch.no_grad()
def initialize_prototypes_from_train_split(
    model,
    train_df,
    args,
    device,
    fold,
    val_df=None,
    test_df=None,
):
    init_method = getattr(args, "edl_proto_init", "embedding_bank")
    if init_method == "random":
        print("[DST_PROTO] Prototype initialization skipped; using random init.")
        return None

    heads = model.prototype_heads()
    if not heads:
        print("[DST_PROTO] No prototype heads found; skipping initialization.")
        return None

    if init_method == "fold_best_scores":
        print(
            "[DST_PROTO] Initializing prototypes from current-fold training samples "
            "ranked by original-MIL true-class score..."
        )
        return initialize_prototypes_from_fold_best_scores(
            model,
            train_df,
            val_df,
            test_df,
            args,
        )

    if init_method == "embedding_bank":
        bank_df = initialize_prototypes_from_embedding_bank(model, args)
        if bank_df is not None:
            return bank_df
        print(
            "[DST_PROTO] Prototype embedding bank not found or not applicable; "
            "falling back to best_samples initialization."
        )
        init_method = "best_samples"

    if init_method == "kmeans":
        print("[DST_PROTO] Initializing prototypes with train-fold KMeans embeddings...")
    elif init_method == "best_samples":
        print(
            "[DST_PROTO] Initializing prototypes with best train samples "
            "(highest true-class confidence, correct predictions first)..."
        )
    else:
        raise ValueError(f"Unsupported --edl_proto_init: {init_method}")

    init_loader = MIL_dataloader(train_df, "test", args)
    buckets = {
        name: {
            "embeddings": [],
            "labels": [],
            "prediction_scores": [],
            "true_class_scores": [],
            "preds": [],
            "patient_id": [],
            "image_id": [],
            "source_row": [],
        }
        for name in heads.keys()
    }
    current_labels = None
    current_patient_ids = None
    current_image_ids = None
    current_source_rows = None
    seen_heads = []

    def make_hook(head_name):
        def hook(module, inputs):
            if current_labels is None:
                return
            seen_heads.append(head_name)
            buckets[head_name]["embeddings"].append(inputs[0].detach().float().cpu())
            buckets[head_name]["labels"].append(current_labels.detach().cpu())
            buckets[head_name]["patient_id"].extend(current_patient_ids)
            buckets[head_name]["image_id"].extend(current_image_ids)
            buckets[head_name]["source_row"].append(current_source_rows.copy())
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
        progress_iter = tqdm(init_loader, desc=f"[fold {fold} DST_PROTO init]", total=len(init_loader))
        source_offset = 0
        for data in progress_iter:
            current_labels = data["y"].long().to(device)
            batch_size = current_labels.size(0)
            current_patient_ids = data.get("patient_id", [None] * batch_size)
            current_image_ids = data.get("image_id", [None] * batch_size)
            current_source_rows = np.arange(source_offset, source_offset + batch_size, dtype=np.int64)
            source_offset += batch_size
            seen_heads = []
            inputs = _move_inputs_to_device(data, device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=amp_enabled):
                edl_out = model(inputs)

            prob = edl_out["prob"].detach().float().cpu()
            preds = torch.argmax(prob, dim=-1)
            labels_cpu = current_labels.detach().cpu().long()
            true_class_scores = prob.gather(1, labels_cpu.view(-1, 1)).squeeze(1)
            prediction_scores = prob[:, 1]
            for head_name in seen_heads:
                buckets[head_name]["prediction_scores"].append(prediction_scores)
                buckets[head_name]["true_class_scores"].append(true_class_scores)
                buckets[head_name]["preds"].append(preds)
    finally:
        for handle in handles:
            handle.remove()
        model.train(was_training)

    bank_rows = []
    for name, head in heads.items():
        embeddings = torch.cat(buckets[name]["embeddings"], dim=0)
        labels = torch.cat(buckets[name]["labels"], dim=0)
        if init_method == "kmeans":
            warnings_list = head.initialize_from_embeddings(
                embeddings,
                labels,
                random_state=args.seed + fold,
            )
        else:
            selected_rows, warnings_list = _select_best_sample_prototypes(head, buckets[name], name)
            bank_rows.extend(selected_rows)
        print(
            f"[DST_PROTO] {name}: initialized {head.num_classes * head.prototypes_per_class} "
            f"prototypes from {len(labels)} training embeddings using {init_method}."
        )
        for warning_text in warnings_list:
            print(f"[DST_PROTO] Warning: {name}: {warning_text}")

    bank_df = _attach_prototype_bank(model, bank_rows) if bank_rows else None
    return bank_df


def expected_proto_columns(args):
    topk = max(0, min(int(args.edl_proto_topk), int(args.edl_proto_k)))
    columns = []
    for class_idx in range(2):
        for rank in range(1, topk + 1):
            prefix = f"proto_c{class_idx}_top{rank}"
            columns.extend([
                f"{prefix}_idx",
                f"{prefix}_global_idx",
                f"{prefix}_evidence",
                f"{prefix}_mass",
                f"{prefix}_similarity",
                f"{prefix}_distance",
                f"{prefix}_source_patient_id",
                f"{prefix}_source_image_id",
                f"{prefix}_source_label",
                f"{prefix}_source_prediction_score",
                f"{prefix}_source_true_class_score",
                f"{prefix}_source_selection_score",
                f"{prefix}_source_selection_class",
                f"{prefix}_source_predicted_class",
                f"{prefix}_source_correct",
            ])
    columns.extend([
        "proto_nearest_global_idx",
        "proto_nearest_class",
        "proto_nearest_idx",
        "proto_nearest_similarity",
        "proto_nearest_distance",
        "proto_nearest_mass",
        "proto_nearest_source_patient_id",
        "proto_nearest_source_image_id",
        "proto_nearest_source_label",
        "proto_nearest_source_prediction_score",
        "proto_nearest_source_true_class_score",
        "proto_nearest_source_selection_score",
        "proto_nearest_source_selection_class",
        "proto_nearest_source_predicted_class",
        "proto_nearest_source_correct",
    ])
    return columns


def _append_proto_batch(proto_buffers, edl_out):
    if "topk_proto_idx" in edl_out:
        top_idx = edl_out["topk_proto_idx"].detach().cpu().numpy()
        top_evidence = edl_out["topk_proto_evidence"].detach().cpu().numpy()
        top_similarity = edl_out["topk_proto_similarity"].detach().cpu().numpy()
        top_distances = edl_out["topk_proto_distances"].detach().cpu().numpy()
        prototypes_per_class = (
            int(edl_out["prototype_similarity"].shape[2])
            if "prototype_similarity" in edl_out
            else int(top_idx.shape[2])
        )

        for class_idx in range(top_idx.shape[1]):
            for rank_idx in range(top_idx.shape[2]):
                prefix = f"proto_c{class_idx}_top{rank_idx + 1}"
                proto_buffers.setdefault(f"{prefix}_idx", []).append(top_idx[:, class_idx, rank_idx])
                proto_buffers.setdefault(f"{prefix}_global_idx", []).append(
                    class_idx * prototypes_per_class + top_idx[:, class_idx, rank_idx]
                )
                proto_buffers.setdefault(f"{prefix}_evidence", []).append(top_evidence[:, class_idx, rank_idx])
                proto_buffers.setdefault(f"{prefix}_mass", []).append(top_evidence[:, class_idx, rank_idx])
                proto_buffers.setdefault(f"{prefix}_similarity", []).append(top_similarity[:, class_idx, rank_idx])
                proto_buffers.setdefault(f"{prefix}_distance", []).append(top_distances[:, class_idx, rank_idx])

    if "prototype_similarity" in edl_out and "prototype_distances" in edl_out:
        similarity = edl_out["prototype_similarity"].detach().cpu().numpy()
        distances = edl_out["prototype_distances"].detach().cpu().numpy()
        prototype_mass = edl_out.get("prototype_mass", edl_out.get("prototype_evidence"))
        mass_values = prototype_mass.detach().cpu().numpy() if prototype_mass is not None else None
        batch_idx = np.arange(similarity.shape[0])
        prototypes_per_class = similarity.shape[2]
        flat_similarity = similarity.reshape(similarity.shape[0], -1)
        nearest_global = np.argmax(flat_similarity, axis=1)
        nearest_class = nearest_global // prototypes_per_class
        nearest_idx = nearest_global % prototypes_per_class
        proto_buffers.setdefault("proto_nearest_global_idx", []).append(nearest_global)
        proto_buffers.setdefault("proto_nearest_class", []).append(nearest_class)
        proto_buffers.setdefault("proto_nearest_idx", []).append(nearest_idx)
        proto_buffers.setdefault("proto_nearest_similarity", []).append(
            similarity[batch_idx, nearest_class, nearest_idx]
        )
        proto_buffers.setdefault("proto_nearest_distance", []).append(
            distances[batch_idx, nearest_class, nearest_idx]
        )
        if mass_values is not None:
            proto_buffers.setdefault("proto_nearest_mass", []).append(
                mass_values[batch_idx, nearest_class, nearest_idx]
            )


PROTOTYPE_SOURCE_FIELD_SPECS = [
    ("source_patient_id", "source_patient_id", ""),
    ("source_image_id", "source_image_id", ""),
    ("source_label", "source_label", np.nan),
    ("source_prediction_score", "source_prediction_score", np.nan),
    ("source_true_class_score", "source_true_class_score", np.nan),
    ("source_selection_score", "source_selection_score", np.nan),
    ("source_selection_class", "source_selection_class", np.nan),
    ("source_predicted_class", "source_predicted_class", np.nan),
    ("source_correct", "source_correct", np.nan),
]


def _primary_prototype_bank(model):
    bank_df = _get_prototype_bank_df(model)
    if bank_df is None:
        return None, None

    head_name = "edl_head"
    if "head_name" in bank_df.columns and head_name in set(bank_df["head_name"].astype(str)):
        bank_df = bank_df[bank_df["head_name"].astype(str) == head_name].copy()
    elif "head_name" in bank_df.columns and len(bank_df) > 0:
        head_name = str(bank_df["head_name"].iloc[0])
        bank_df = bank_df[bank_df["head_name"].astype(str) == head_name].copy()

    if bank_df.empty:
        return None, None
    return head_name, bank_df


def _prototype_bank_lookup(model):
    _, bank_df = _primary_prototype_bank(model)
    if bank_df is None:
        return {}

    lookup = {}
    for _, row in bank_df.iterrows():
        if "prototype_global_idx" not in row:
            continue
        try:
            global_idx = int(row["prototype_global_idx"])
        except (TypeError, ValueError):
            continue
        lookup[global_idx] = row
    return lookup


def _bank_value(row, field, default):
    if row is None or field not in row:
        return default
    value = row[field]
    if pd.isna(value):
        return default
    if isinstance(value, np.generic):
        value = value.item()
    return value


def _rows_from_global_indices(global_indices, lookup):
    rows = []
    for value in global_indices:
        try:
            if pd.isna(value):
                rows.append(None)
                continue
            global_idx = int(value)
        except (TypeError, ValueError):
            rows.append(None)
            continue
        rows.append(lookup.get(global_idx))
    return rows


def _append_source_columns_for_prefix(results, prefix, global_indices, lookup):
    rows = _rows_from_global_indices(global_indices, lookup)
    for bank_field, suffix, default in PROTOTYPE_SOURCE_FIELD_SPECS:
        results[f"{prefix}_{suffix}"] = [
            _bank_value(row, bank_field, default)
            for row in rows
        ]


def _add_prototype_source_columns(results, model, args):
    lookup = _prototype_bank_lookup(model)
    if not lookup:
        return

    topk = max(0, min(int(args.edl_proto_topk), int(args.edl_proto_k)))
    for class_idx in range(2):
        for rank in range(1, topk + 1):
            prefix = f"proto_c{class_idx}_top{rank}"
            global_key = f"{prefix}_global_idx"
            if global_key not in results:
                continue
            _append_source_columns_for_prefix(results, prefix, results[global_key], lookup)

    if "proto_nearest_global_idx" in results:
        _append_source_columns_for_prefix(
            results,
            "proto_nearest",
            results["proto_nearest_global_idx"],
            lookup,
        )


INTERPRETABILITY_TENSOR_KEYS = [
    "prototype_mass",
    "prototype_similarity",
    "prototype_distances",
    "topk_proto_idx",
    "topk_proto_mass",
    "topk_proto_similarity",
    "topk_proto_distances",
]


def _append_interpretability_batch(interpretability_buffers, edl_out):
    for key in INTERPRETABILITY_TENSOR_KEYS:
        if key in edl_out:
            interpretability_buffers.setdefault(key, []).append(
                edl_out[key].detach().cpu().numpy()
            )


def _string_array(values):
    return np.asarray(["" if value is None else str(value) for value in values], dtype=str)


def _concat_result_field(results_by_split, key, dtype=None):
    chunks = []
    for _, sample_results in results_by_split:
        if key not in sample_results:
            continue
        chunks.append(np.asarray(sample_results[key]))
    if not chunks:
        return None

    values = np.concatenate(chunks, axis=0)
    if dtype is not None:
        values = values.astype(dtype, copy=False)
    return values


def _extract_primary_prototype_state(model):
    if not hasattr(model, "prototype_heads"):
        return {}

    heads = model.prototype_heads()
    if not heads:
        return {}

    head_name = "edl_head" if "edl_head" in heads else next(iter(heads.keys()))
    head = heads[head_name]
    arrays = {
        "prototype_head": np.asarray(head_name),
        "prototypes": head.prototypes.detach().cpu().numpy(),
    }

    activation = getattr(getattr(head, "ds_module", None), "ds1_activate", None)
    if activation is not None:
        reliability = torch.sigmoid(activation.xi).detach().cpu()
        gamma = activation.eta.detach().cpu().pow(2)
        shape = (head.num_classes, head.prototypes_per_class)
        arrays["prototype_reliability"] = reliability.view(shape).numpy()
        arrays["prototype_gamma"] = gamma.view(shape).numpy()

    _, bank_df = _primary_prototype_bank(model)
    if bank_df is not None:
        shape = (head.num_classes, head.prototypes_per_class)
        source_patient_id = np.full(shape, "", dtype="<U256")
        source_image_id = np.full(shape, "", dtype="<U256")
        source_label = np.full(shape, -1, dtype=np.int64)
        source_prediction_score = np.full(shape, np.nan, dtype=np.float32)
        source_true_class_score = np.full(shape, np.nan, dtype=np.float32)
        source_selection_score = np.full(shape, np.nan, dtype=np.float32)
        source_selection_class = np.full(shape, -1, dtype=np.int64)
        source_predicted_class = np.full(shape, -1, dtype=np.int64)
        source_correct = np.full(shape, False, dtype=bool)

        for _, row in bank_df.iterrows():
            try:
                class_idx = int(row["prototype_class"])
                rank_idx = int(row["prototype_rank"])
            except (KeyError, TypeError, ValueError):
                continue
            if class_idx < 0 or class_idx >= shape[0] or rank_idx < 0 or rank_idx >= shape[1]:
                continue
            source_patient_id[class_idx, rank_idx] = str(_bank_value(row, "source_patient_id", ""))
            source_image_id[class_idx, rank_idx] = str(_bank_value(row, "source_image_id", ""))
            source_label[class_idx, rank_idx] = int(_bank_value(row, "source_label", -1))
            source_prediction_score[class_idx, rank_idx] = float(
                _bank_value(row, "source_prediction_score", np.nan)
            )
            source_true_class_score[class_idx, rank_idx] = float(
                _bank_value(row, "source_true_class_score", np.nan)
            )
            source_selection_score[class_idx, rank_idx] = float(
                _bank_value(row, "source_selection_score", np.nan)
            )
            source_selection_class[class_idx, rank_idx] = int(
                _bank_value(row, "source_selection_class", class_idx)
            )
            source_predicted_class[class_idx, rank_idx] = int(
                _bank_value(row, "source_predicted_class", -1)
            )
            source_correct[class_idx, rank_idx] = bool(_bank_value(row, "source_correct", False))

        arrays.update({
            "prototype_source_patient_id": source_patient_id,
            "prototype_source_image_id": source_image_id,
            "prototype_source_label": source_label,
            "prototype_source_prediction_score": source_prediction_score,
            "prototype_source_true_class_score": source_true_class_score,
            "prototype_source_selection_score": source_selection_score,
            "prototype_source_selection_class": source_selection_class,
            "prototype_source_predicted_class": source_predicted_class,
            "prototype_source_correct": source_correct,
        })

    return arrays


def save_dst_proto_interpretability_npz(output_path, model, results_by_split):
    results_by_split = [
        (split_name, sample_results)
        for split_name, sample_results in results_by_split
        if sample_results is not None and len(sample_results.get("label", [])) > 0
    ]
    if not results_by_split:
        return None

    split_values = []
    for split_name, sample_results in results_by_split:
        split_values.extend([split_name] * len(sample_results["label"]))

    arrays = {
        "patient_id": _string_array(_concat_result_field(results_by_split, "patient_id")),
        "image_id": _string_array(_concat_result_field(results_by_split, "image_id")),
        "split": _string_array(split_values),
        "label": _concat_result_field(results_by_split, "label", dtype=np.int64),
        "prediction_score": _concat_result_field(results_by_split, "score", dtype=np.float32),
        "predicted_class": _concat_result_field(results_by_split, "pred", dtype=np.int64),
        "dst_mass": _concat_result_field(results_by_split, "dst_mass", dtype=np.float32),
        "uncertainty": _concat_result_field(results_by_split, "uncertainty", dtype=np.float32),
    }

    for key in INTERPRETABILITY_TENSOR_KEYS:
        values = _concat_result_field(results_by_split, key)
        if values is not None:
            arrays[key] = values

    arrays.update(_extract_primary_prototype_state(model))

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    print(f"[DST_PROTO] Interpretability NPZ saved: {output_path}")
    return output_path


@torch.no_grad()
def edl_proto_predict(loader, model, args, device, desc="DST_PROTO predict"):
    model.eval()
    model.is_training = False

    targs = []
    probs_list = []
    preds_list = []
    uncertainty_list = []
    mass_list = []
    proto_buffers = {}
    interpretability_buffers = {}
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
        mass = edl_out["dst_mass"].detach().cpu()
        uncertainty = edl_out["uncertainty"].detach().cpu()
        pred_class = torch.argmax(prob, dim=-1)

        targs.append(labels.cpu().numpy())
        probs_list.append(prob[:, 1].numpy())
        preds_list.append(pred_class.numpy())
        uncertainty_list.append(uncertainty.numpy())
        mass_list.append(mass.numpy())
        _append_proto_batch(proto_buffers, edl_out)
        _append_interpretability_batch(interpretability_buffers, edl_out)

    mass_array = np.concatenate(mass_list)
    results = {
        "patient_id": sample_patient_ids,
        "image_id": sample_image_ids,
        "label": np.concatenate(targs).tolist(),
        "score": np.concatenate(probs_list).tolist(),
        "pred": np.concatenate(preds_list).tolist(),
        "uncertainty": np.concatenate(uncertainty_list).tolist(),
        "dst_mass": mass_array,
        "mass_0": mass_array[:, 0].tolist(),
        "mass_1": mass_array[:, 1].tolist(),
        "mass_omega": mass_array[:, 2].tolist(),
    }

    for key, chunks in proto_buffers.items():
        results[key] = np.concatenate(chunks).tolist()

    for key, chunks in interpretability_buffers.items():
        results[key] = np.concatenate(chunks, axis=0)

    _add_prototype_source_columns(results, model, args)

    return results


def build_prediction_df(split_df, sample_results, split_name, fold, args):
    label_col = args.label.lower()
    pred_df = split_df.copy().reset_index(drop=True)
    pred_df["prediction_score"] = sample_results["score"]
    pred_df["predicted_class"] = sample_results["pred"]
    pred_df[label_col] = sample_results["label"]
    pred_df["mass_0"] = sample_results["mass_0"]
    pred_df["mass_1"] = sample_results["mass_1"]
    pred_df["mass_omega"] = sample_results["mass_omega"]
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
        "mass_0",
        "mass_1",
        "mass_omega",
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
    args.output_path = Path(f"{args.output_dir}/DST_PROTO/{args.dataset}_{args.label}/fold_{args.n_folds}/{now}")
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
            context="DST_PROTO n_folds=0 internal train/val split",
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
    # Keep the raw predictions from every fold so the reviewer-analysis script
    # can reconstruct out-of-fold validation performance and the five-model
    # held-out test ensemble without rerunning inference.
    all_fold_prediction_dfs = []

    for fold, (train_df, val_df) in enumerate(split_iter):
        if fold < args.start_fold:
            continue

        print(f'\n{"=" * 60}')
        print(f"  DST_PROTO Fold {fold} / {total_folds}")
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
            print(f"[DST_PROTO] Warning: no checkpoint found under {args.resume} for fold {fold}; training from scratch.")

        model, loaded_proto_checkpoint = build_edl_proto_model(args, pretrained_checkpoint)
        model.to(device)

        if not loaded_proto_checkpoint:
            prototype_bank_df = initialize_prototypes_from_train_split(
                model,
                train_df,
                args,
                device,
                fold,
                val_df=val_df,
                test_df=test_df,
            )
            if prototype_bank_df is not None:
                save_prototype_bank_csv(
                    path_results_fold / f"{args.dataset}_dst_proto_prototype_bank_fold_{fold}.csv",
                    model,
                )
        else:
            save_prototype_bank_csv(
                path_results_fold / f"{args.dataset}_dst_proto_prototype_bank_fold_{fold}.csv",
                model,
            )

        if args.train_edl_only:
            freeze_mil_backbone_train_edl_only(model)
            print("[DST_PROTO] Freeze mode enabled: training only prototype DST head(s); MIL backbone is frozen.")

        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")

        trainable_parameters = [p for p in model.parameters() if p.requires_grad]
        if not trainable_parameters:
            raise RuntimeError("No trainable parameters found. Check the Prototype-DST freeze configuration.")

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

        criterion = build_edl_criterion(args, class_weights=class_weights)

        val_stats, best_checkpoint_path = edl_proto_train_loop(
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
        for key in EDL_LOSS_DIAGNOSTIC_KEYS:
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
        all_val_results.append(fold_summary)

        print(f"\nGenerating Prototype-DST predictions with best model for fold {fold}...")
        checkpoint = torch.load(best_checkpoint_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        _attach_prototype_bank(model, checkpoint.get("prototype_bank"))
        save_prototype_bank_csv(
            path_results_fold / f"{args.dataset}_dst_proto_prototype_bank_fold_{fold}.csv",
            model,
        )
        model.eval()

        all_split_dfs = []
        fold_interpretability_results = []
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
                desc=f"DST_PROTO {split_name} predict",
            )
            fold_interpretability_results.append((split_name, sample_results))
            pred_df = build_prediction_df(split_df, sample_results, split_name, fold, args)
            all_split_dfs.append(pred_df)

            if split_name == "val":
                for _, row in pred_df.iterrows():
                    fold_assignments.append(row.to_dict())

        if all_split_dfs:
            fold_pred_df = pd.concat(all_split_dfs, ignore_index=True)
            all_fold_prediction_dfs.append(fold_pred_df)
            fold_pred_df.to_csv(
                path_results_fold / f"{args.dataset}_dst_proto_predictions_fold_{fold}.csv",
                index=False,
            )
            print(f"Saved fold {fold} Prototype-DST predictions: {len(fold_pred_df)} samples")
            save_dst_proto_interpretability_npz(
                path_results_fold / f"{args.dataset}_dst_proto_interpretability_fold_{fold}.npz",
                model,
                fold_interpretability_results,
            )

        del model
        clear_memory()

    summary_df = pd.DataFrame(all_val_results)
    if len(summary_df) > 1:
        metric_cols = [col for col in summary_df.columns if col not in ["fold", "eval_source"]]
        mean_std = summary_df[metric_cols].agg(["mean", "std"]).reset_index(drop=True)
        mean_std["fold"] = ["mean", "std"]
        mean_std["eval_source"] = "summary"
        summary_df = pd.concat([summary_df, mean_std], ignore_index=True)

    summary_df.to_csv(args.output_path / "dst_proto_results_summary.csv", index=False)
    print(f"\nResults summary saved to {args.output_path / 'dst_proto_results_summary.csv'}")
    print(summary_df.to_string())

    if fold_assignments:
        fold_df = pd.DataFrame(fold_assignments)
        fold_df.to_csv(
            args.output_path / f"{args.dataset}_dst_proto_val_fold_assignments.csv",
            index=False,
        )
        print(f"Fold assignments saved ({len(fold_df)} validation samples)")

    if all_fold_prediction_dfs:
        all_predictions_df = pd.concat(all_fold_prediction_dfs, ignore_index=True)
        all_predictions_df.to_csv(
            args.output_path / f"{args.dataset}_dst_proto_all_predictions.csv",
            index=False,
        )

        # Exactly one validation prediction per development sample: the model
        # from the fold where that sample was held out.  These are the scores
        # used to choose operating thresholds, never the held-out test scores.
        dev_predictions_df = all_predictions_df[
            all_predictions_df["split"] == "val"
        ].reset_index(drop=True)
        dev_predictions_df.to_csv(
            args.output_path / f"{args.dataset}_dst_proto_dev_predictions.csv",
            index=False,
        )

        # One held-out test prediction per model/fold.  The analysis code first
        # aggregates image scores within patient and then averages over folds.
        test_all_folds_df = all_predictions_df[
            all_predictions_df["split"] == "test"
        ].reset_index(drop=True)
        test_all_folds_df.to_csv(
            args.output_path / f"{args.dataset}_dst_proto_test_all_folds.csv",
            index=False,
        )
        print(
            "Saved analysis-ready predictions: "
            f"{len(dev_predictions_df)} validation OOF rows and "
            f"{len(test_all_folds_df)} held-out test fold rows"
        )

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
    print("  Training complete. Starting automatic Prototype-DST testing...")
    print("=" * 60)

    from edl_proto_test import run_edl_proto_test

    test_output_dir = output_path / "dst_proto_test_results"
    run_edl_proto_test(args, device, checkpoint_dir=output_path, output_dir=test_output_dir)

    print("\n===== Prototype-DST Training + Testing Pipeline Complete =====")


if __name__ == "__main__":
    main()
