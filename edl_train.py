"""
EDL Training Script - Evidential Deep Learning with MIL

This script performs fine-tuning of a pretrained MIL model with an EDL head,
using 5-fold cross-validation. It loads pretrained MIL weights, replaces the
classification head with an EDL head, and can either train all parameters or
freeze the MIL backbone and train only the EDL module.

Usage:
    python edl_train.py --resume path/to/pretrained_mil --data_dir datasets/... --csv_file grouped_df.csv
    
All arguments mirror the original main.py to ensure compatibility with the MIL model builder.
Additional EDL-specific arguments are provided for loss configuration.
"""

import os
import sys
import gc
import time
import warnings
import argparse
import yaml
import math
from pathlib import Path
from datetime import datetime

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from torch.optim.lr_scheduler import LambdaLR

# Internal imports
from Datasets.dataset_utils import MIL_dataloader
from MIL import build_model
from MIL.edl_models import EDLHead, MIL_EDL_Wrapper
from MIL.edl_losses import EDLCombinedLoss, edl_crossentropy_loss, kl_divergence_dirichlet
from utils.metrics import auroc, evaluate_metrics
from utils.generic_utils import seed_all, AverageMeter, clear_memory
from utils.data_split_utils import (
    adaptive_stratified_train_val_split,
    generator_cross_val_folds,
    split_df_by_cohorts,
)


def config():
    parser = argparse.ArgumentParser(description="EDL Training Script")
    
    # ===== EDL-specific arguments =====
    parser.add_argument('--edl_kl_weight', type=float, default=0.1, help='Weight for KL divergence regularization')
    parser.add_argument('--edl_annealing_epochs', type=int, default=10, help='Number of epochs for KL annealing')
    parser.add_argument('--edl_annealing_start', type=int, default=0, help='Epoch to start KL annealing')
    parser.add_argument('--edl_dropout', type=float, default=0.0, help='Dropout for EDL head')
    parser.add_argument('--train_edl_only', '--freeze_backbone', action='store_true', default=False,
                        help='Freeze the MIL backbone and train only EDL head(s)')
    
    # ===== Folder arguments =====
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument('--output_dir', default='Mammo-CLIP-output/edl_experiments', help='Path to output logs')
    parser.add_argument("--data_dir", default="datasets/Vindir-mammoclip", type=str)
    parser.add_argument("--clip_chk_pt_path", default=None, type=str,
                        help="Path to Mammo-CLIP checkpoint; required when --feature_extraction online")
    parser.add_argument("--csv_file", default="grouped_df.csv", type=str)
    parser.add_argument('--feat_dir', default='new_extracted_features', type=str)
    parser.add_argument("--img_dir", default="VinDir_preprocessed_mammoclip/images_png", type=str)
    parser.add_argument('--train', action='store_true', default=True)
    
    # ===== Data settings =====
    parser.add_argument("--img-size", nargs='+', default=[1520, 912])
    parser.add_argument("--dataset", default="ViNDr", type=str)
    parser.add_argument("--data_frac", default=1.0, type=float)
    parser.add_argument("--label", default="Mass", type=str)
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--n_folds", default=5, type=int)
    parser.add_argument("--start-fold", default=0, type=int)
    parser.add_argument(
        "--kfold0-val-frac",
        "--kfold0_val_frac",
        dest="kfold0_val_frac",
        default=0.2,
        type=float,
        help="Validation fraction split from train cohorts when --n_folds 0.",
    )
    parser.add_argument(
        "--kfold0-val-max-frac",
        "--kfold0_val_max_frac",
        dest="kfold0_val_max_frac",
        default=0.5,
        type=float,
        help=(
            "Maximum validation fraction allowed when --n_folds 0 needs a larger "
            "validation split to contain both classes."
        ),
    )
    parser.add_argument(
        "--train-cohorts", "--train_cohorts",
        dest="train_cohorts",
        default="1-8",
        type=str,
        help="Cohorts to use for training / cross-validation, e.g. '1-8' or '1,2,3'.",
    )
    parser.add_argument(
        "--test-cohorts", "--test_cohorts",
        dest="test_cohorts",
        default="9-10",
        type=str,
        help="Cohorts to use for held-out testing, e.g. '9-10' or '9,10'.",
    )
    parser.add_argument("--mean", default=0.3089279, type=float)
    parser.add_argument("--std", default=0.25053555408335154, type=float)
    
    # ===== Feature extraction =====
    parser.add_argument("--feature_extraction", default='offline', type=str)
    parser.add_argument("--feat_dim", default=352, type=int)
    parser.add_argument("--arch", default="upmc_breast_clip_det_b5_period_n_ft", type=str)
    parser.add_argument('--model-type', default="Classifier", type=str)
    
    # ===== MIL model parameters =====
    parser.add_argument('--mil_type', default=None, choices=[None, 'instance', 'embedding', 'pyramidal_mil'], type=str)
    parser.add_argument('--pooling_type', default='mean', choices=['max', 'mean', 'attention', 'gated-attention', 'pma'], type=str)
    parser.add_argument('--type_mil_encoder', default='mlp', choices=['mlp', 'sab', 'isab'], type=str)
    parser.add_argument('--fcl_attention_dim', type=int, default=128)
    parser.add_argument('--map_prob_func', type=str, default='softmax', choices=['softmax', 'sparsemax', 'entmax', 'alpha_entmax'])
    parser.add_argument('--fcl_encoder_dim', type=int, default=256)
    parser.add_argument('--sab_num_heads', type=int, default=4)
    parser.add_argument('--isab_num_heads', type=int, default=4)
    parser.add_argument('--pma_num_heads', type=int, default=1)
    parser.add_argument('--num_encoder_blocks', type=int, default=2)
    parser.add_argument('--trans_layer_norm', type=bool, default=False)
    parser.add_argument('--fcl_dropout', type=float, default=0.0)
    
    # ===== Multi-scale MIL =====
    parser.add_argument('--multi_scale_model', type=str, choices=['fpn', 'backbone_pyramid', 'msp'], default=None)
    parser.add_argument('--scales', type=int, nargs='*', default=(16, 32, 64, 128))
    parser.add_argument('--deep_supervision', action='store_true', default=False)
    parser.add_argument('--type_scale_aggregator', type=str, choices=['concatenation', 'max_p', 'mean_p', 'attention', 'gated-attention'], default=None)
    
    # ===== Patching =====
    parser.add_argument('--patch_size', type=int, default=512)
    parser.add_argument('--overlap', type=float, nargs='*', default=[0.0])
    
    # ===== FPN =====
    parser.add_argument('--fpn_dim', type=int, default=256)
    parser.add_argument('--upsample_method', type=str, choices=['bilinear', 'nearest'], default='nearest')
    parser.add_argument('--norm_fpn', type=bool, default=False)
    
    # ===== Nested MIL =====
    parser.add_argument('--nested_model', action='store_true', default=False)
    parser.add_argument('--type_region_aggregator', type=str, default=None)
    parser.add_argument('--type_region_encoder', default=None, choices=['mlp', 'sab', 'isab'], type=str)
    parser.add_argument('--type_region_pooling', default=None, choices=['max', 'mean', 'attention', 'gated-attention', 'pma'], type=str)
    
    # ===== Training parameters =====
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--epochs", default=20, type=int)
    parser.add_argument("--lr", default=5.0e-5, type=float)
    parser.add_argument("--weight-decay", default=1e-4, type=float)
    parser.add_argument("--clip_grad", type=float, default=0.0)
    parser.add_argument("--warmup-epochs", default=1, type=float)
    parser.add_argument("--num_cycles", default=0.5, type=float)
    parser.add_argument("--data_aug", action='store_true', default=False)
    parser.add_argument("--weighted-BCE", "--weighted_BCE", dest="weighted_BCE",
                        default="y", choices=["y", "n"],
                        help="Enable automatic positive-class weighting using neg/pos in each training fold.")
    parser.add_argument("--early_stop_patience", default=0, type=int,
                        help="Early stopping patience in epochs. Set > 0 to enable; 0 disables early stopping.")
    parser.add_argument("--early_stop_min_delta", default=0.0, type=float,
                        help="Minimum validation metric improvement required to reset early stopping.")
    
    # ===== Regularization =====
    parser.add_argument('--drop_classhead', type=float, default=0.0)
    parser.add_argument('--drop_attention_pool', type=float, default=0.0)
    parser.add_argument('--drop_mha', type=float, default=0.0)
    
    # ===== Misc =====
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--apex", default="y", type=str)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument('--roi_eval', action='store_true', default=False)
    parser.add_argument('--resume', default=None, type=str, help='Path to pretrained MIL checkpoint or directory')
    
    return parser.parse_args()


def resolve_mil_checkpoint(resume_path, fold=None):
    """Resolve a pretrained MIL checkpoint from a file or experiment directory."""
    if resume_path is None:
        return None

    resume_path = Path(resume_path)
    if resume_path.is_file():
        return resume_path

    if not resume_path.exists():
        return None

    candidates = []
    if fold is not None:
        candidates.extend([
            resume_path / f'fold_{fold}' / 'best_model.pth',
            resume_path / f'run_{fold}' / 'best_model.pth',
        ])

    candidates.extend([
        resume_path / 'best_model.pth',
        resume_path / 'checkpoint.pth',
    ])

    return next((path for path in candidates if path.exists()), None)


def _get_checkpoint_state_dict(checkpoint):
    """Return the model state dict from a torch checkpoint or raw state dict."""
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        return checkpoint['model']
    return checkpoint


def _looks_like_edl_state_dict(state_dict):
    """EDL checkpoints are saved from MIL_EDL_Wrapper and have wrapper prefixes."""
    if not isinstance(state_dict, dict):
        return False
    return any(
        key.startswith(('mil_model.', 'edl_head.', 'edl_side_heads.'))
        for key in state_dict.keys()
    )


def _print_load_summary(prefix, load_msg):
    missing = list(getattr(load_msg, 'missing_keys', []))
    unexpected = list(getattr(load_msg, 'unexpected_keys', []))
    print(f"{prefix} missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    if missing[:5]:
        print(f"{prefix} first missing keys: {missing[:5]}")
    if unexpected[:5]:
        print(f"{prefix} first unexpected keys: {unexpected[:5]}")


class LinearWarmupCosineAnnealingLR(LambdaLR):
    """Linear warmup + cosine annealing learning rate scheduler."""
    def __init__(self, optimizer, total_steps, warmup_steps, last_epoch=-1):
        assert warmup_steps < total_steps, "Warmup steps should be less than total steps."
        self.tsteps = total_steps
        self.wsteps = int(warmup_steps) if not isinstance(warmup_steps, float) else math.ceil(total_steps * warmup_steps)
        super().__init__(optimizer, self._lr_multiplier, last_epoch)

    def _lr_multiplier(self, step):
        if step < self.wsteps:
            return step / float(max(1, self.wsteps))
        cos_factor = (step - self.wsteps) / (self.tsteps - self.wsteps)
        return max(0, math.cos(cos_factor * (math.pi / 2)) ** 2)


def build_edl_model(args, checkpoint_path=None):
    """
    Build MIL model, optionally load pretrained weights, wrap with EDL head.
    
    Args:
        args: configuration namespace
        checkpoint_path: path to pretrained MIL checkpoint
    
    Returns:
        MIL_EDL_Wrapper model
    """
    args.n_class = 1
    if args.feature_extraction == 'online' and not getattr(args, 'clip_chk_pt_path', None):
        raise ValueError(
            "--clip_chk_pt_path is required when --feature_extraction online "
            "so the Mammo-CLIP image encoder/backbone can be initialized."
        )

    mil_model = build_model(args)

    checkpoint_state = None
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_file():
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            checkpoint_state = _get_checkpoint_state_dict(checkpoint)
        else:
            print(f"[EDL] Warning: checkpoint not found at {checkpoint_path}, training from scratch.")

    # Plain MIL checkpoints are loaded before wrapping because their keys match
    # the raw MIL model. The EDL head is then initialized from scratch.
    is_edl_checkpoint = _looks_like_edl_state_dict(checkpoint_state)
    if checkpoint_state is not None and not is_edl_checkpoint:
        load_msg = mil_model.load_state_dict(checkpoint_state, strict=False)
        print(f"[EDL] Loaded pretrained MIL backbone from: {checkpoint_path}")
        _print_load_summary("[EDL][MIL load]", load_msg)

    # Wrap with EDL head
    edl_model = MIL_EDL_Wrapper(mil_model, edl_dropout=args.edl_dropout)
    if checkpoint_state is not None and is_edl_checkpoint:
        # Trained EDL checkpoints already include mil_model.* and edl_head.*
        # weights, so load them after the wrapper exists.
        load_msg = edl_model.load_state_dict(checkpoint_state, strict=False)
        print(f"[EDL] Loaded EDL training checkpoint from: {checkpoint_path}")
        _print_load_summary("[EDL][EDL load]", load_msg)

    return edl_model


def freeze_mil_backbone_train_edl_only(model):
    """
    Freeze the wrapped MIL model and keep only EDL head parameters trainable.

    The original MIL classifier stays inside model.mil_model but is bypassed by
    MIL_EDL_Wrapper, so the trainable surface is the main EDL head plus optional
    scale-specific EDL side heads.
    """
    for param in model.mil_model.parameters():
        param.requires_grad = False

    edl_head = getattr(model, 'edl_head', None)
    if edl_head is not None:
        for param in edl_head.parameters():
            param.requires_grad = True

    edl_side_heads = getattr(model, 'edl_side_heads', None)
    if edl_side_heads is not None:
        for param in edl_side_heads.parameters():
            param.requires_grad = True


def keep_frozen_mil_backbone_in_eval(model, args):
    """Keep frozen backbone modules deterministic while EDL heads remain trainable."""
    if getattr(args, 'train_edl_only', False):
        model.mil_model.eval()


def edl_train_fn(train_loader, model, criterion, optimizer, epoch, args, scheduler, scaler, device):
    """Training loop for one epoch with EDL loss."""
    
    model.train()
    keep_frozen_mil_backbone_in_eval(model, args)
    model.is_training = True
    
    losses = AverageMeter()
    ce_losses = AverageMeter()
    kl_losses = AverageMeter()
    
    progress_iter = tqdm(enumerate(train_loader),
                         desc=f"[{epoch + 1:03d}/{args.epochs:03d} EDL train]",
                         total=len(train_loader))
    
    targs = []
    probs_list = []
    preds_list = []
    uncertainty_list = []
    
    for step, data in progress_iter:
        # Move data to device
        if isinstance(data['x'], dict):
            inputs = {scale: tensor.to(device) for scale, tensor in data['x'].items()}
        elif isinstance(data['x'], list):
            inputs = [tensor.to(device) for tensor in data['x']]
        else:
            inputs = data['x'].to(device)
        
        labels = data['y'].long().to(device)  # EDL needs integer labels
        batch_size = labels.size(0)
        
        amp_enabled = args.apex and device.type == 'cuda'
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            # Forward pass through EDL wrapper
            edl_out = model(inputs)
            
            alpha = edl_out['alpha']  # (B, K)
            
            # Compute EDL loss
            loss, loss_dict = criterion(alpha, labels, epoch=epoch)
            for side_out in edl_out.get('side_outputs', {}).values():
                side_loss, _ = criterion(side_out['alpha'], labels, epoch=epoch)
                loss = loss + side_loss
        
        losses.update(loss.item(), batch_size)
        ce_losses.update(loss_dict['ce_loss'], batch_size)
        kl_losses.update(loss_dict['kl_loss'], batch_size)
        
        # Backprop
        scaler.scale(loss).backward()
        
        if args.clip_grad > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        
        # Collect predictions
        prob = edl_out['prob'].detach()  # (B, K)
        pred_class = torch.argmax(prob, dim=-1)  # (B,)
        uncertainty = edl_out['uncertainty'].detach()  # (B,)
        
        targs.append(labels.cpu().numpy())
        probs_list.append(prob[:, 1].cpu().numpy())  # positive class probability
        preds_list.append(pred_class.cpu().numpy())
        uncertainty_list.append(uncertainty.cpu().numpy())
        cuda_mem = torch.cuda.memory_usage(device) if device.type == 'cuda' and torch.cuda.is_available() else 0
        
        progress_iter.set_postfix({
            "loss": f"{losses.avg:.4f}",
            "ce": f"{ce_losses.avg:.4f}",
            "kl": f"{kl_losses.avg:.4f}",
            "CUDA-Mem": f"{cuda_mem}%",
        })
    
    # Compute metrics
    targs = np.concatenate(targs)
    probs = np.concatenate(probs_list)
    preds = np.concatenate(preds_list)
    
    auc = auroc(targs, probs)
    f1, bacc = evaluate_metrics(targs, preds)
    
    train_stats = {
        'loss': losses.avg,
        'ce_loss': ce_losses.avg,
        'kl_loss': kl_losses.avg,
        'auc_roc': auc,
        'f1': f1,
        'bacc': bacc,
        'lr': optimizer.param_groups[0]['lr'],
    }
    
    return train_stats


@torch.no_grad()
def edl_valid_fn(valid_loader, model, args, device, split='val', epoch=1):
    """Validation loop for EDL model."""
    
    model.eval()
    model.is_training = False
    
    losses = AverageMeter()
    criterion_eval = EDLCombinedLoss(
        num_classes=2,
        kl_weight=args.edl_kl_weight,
        annealing_start=args.edl_annealing_start,
        annealing_epochs=args.edl_annealing_epochs,
    )
    
    targs = []
    probs_list = []
    preds_list = []
    uncertainty_list = []
    evidence_list = []
    alpha_list = []
    
    sample_patient_ids = []
    sample_image_ids = []
    
    if split == 'val':
        progress_iter = tqdm(enumerate(valid_loader),
                             desc=f"[{epoch + 1:03d}/{args.epochs:03d} EDL valid]",
                             total=len(valid_loader))
    else:
        progress_iter = tqdm(enumerate(valid_loader), total=len(valid_loader))
    
    for step, data in progress_iter:
        if isinstance(data['x'], dict):
            inputs = {scale: tensor.to(device, non_blocking=True) for scale, tensor in data['x'].items()}
            batch_size = inputs[list(inputs.keys())[0]].size(0)
        elif isinstance(data['x'], list):
            inputs = [tensor.to(device, non_blocking=True) for tensor in data['x']]
            batch_size = inputs[0].size(0)
        else:
            inputs = data['x'].to(device, non_blocking=True)
            batch_size = inputs.size(0)
        
        labels = data['y'].long().to(device)
        sample_patient_ids.extend(data.get('patient_id', [None] * batch_size))
        sample_image_ids.extend(data.get('image_id', [None] * batch_size))
        
        amp_enabled = args.apex and device.type == 'cuda'
        with torch.cuda.amp.autocast(enabled=amp_enabled):
            edl_out = model(inputs)
            alpha = edl_out['alpha']
            loss, _ = criterion_eval(alpha, labels, epoch=epoch)
            for side_out in edl_out.get('side_outputs', {}).values():
                side_loss, _ = criterion_eval(side_out['alpha'], labels, epoch=epoch)
                loss = loss + side_loss
        
        losses.update(loss.item(), batch_size)
        
        prob = edl_out['prob'].detach()
        pred_class = torch.argmax(prob, dim=-1)
        uncertainty = edl_out['uncertainty'].detach()
        evidence = edl_out['evidence'].detach()
        
        targs.append(labels.cpu().numpy())
        probs_list.append(prob[:, 1].cpu().numpy())
        preds_list.append(pred_class.cpu().numpy())
        uncertainty_list.append(uncertainty.cpu().numpy())
        evidence_list.append(evidence.cpu().numpy())
        alpha_list.append(alpha.detach().cpu().numpy())
    
    targs = np.concatenate(targs)
    probs = np.concatenate(probs_list)
    preds = np.concatenate(preds_list)
    
    auc_val = auroc(targs, probs)
    f1, bacc = evaluate_metrics(targs, preds)
    
    val_stats = {
        'loss': losses.avg,
        'auc_roc': auc_val,
        'f1': f1,
        'bacc': bacc,
    }
    
    sample_results = {
        'patient_id': sample_patient_ids,
        'image_id': sample_image_ids,
        'label': targs.tolist(),
        'score': probs.tolist(),
        'pred': preds.tolist(),
        'uncertainty': np.concatenate(uncertainty_list).tolist(),
        'evidence_0': np.concatenate(evidence_list)[:, 0].tolist(),
        'evidence_1': np.concatenate(evidence_list)[:, 1].tolist(),
        'alpha_0': np.concatenate(alpha_list)[:, 0].tolist(),
        'alpha_1': np.concatenate(alpha_list)[:, 1].tolist(),
    }
    
    return targs, preds, probs, val_stats, sample_results


def save_loss_curve(train_results, val_results, output_path):
    """Save epoch-wise loss history as CSV and PNG in the fold output folder."""
    if not train_results['loss'] or not val_results['loss']:
        return

    output_path = Path(output_path)
    curve_df = pd.DataFrame({
        'epoch': np.arange(1, len(train_results['loss']) + 1),
        'train_loss': train_results['loss'],
        'val_loss': val_results['loss'],
        'train_auc_roc': train_results['auc_roc'],
        'val_auc_roc': val_results['auc_roc'],
        'train_f1': train_results['f1'],
        'val_f1': val_results['f1'],
        'train_bacc': train_results['bacc'],
        'val_bacc': val_results['bacc'],
        'lr': train_results['lr'],
    })
    curve_df.to_csv(output_path / 'edl_loss_curve.csv', index=False)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(curve_df['epoch'], curve_df['train_loss'], marker='o', label='Train Loss')
        ax.plot(curve_df['epoch'], curve_df['val_loss'], marker='o', label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('EDL Loss Curve')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path / 'edl_loss_curve.png', dpi=200)
        plt.close(fig)
    except Exception as exc:
        print(f"[EDL] Warning: failed to save loss curve plot: {exc}")


def get_edl_class_weights(train_df, label_col):
    """Return [negative_weight, positive_weight] using neg/pos for class imbalance."""
    labels = train_df[label_col].astype(int)
    num_pos = int((labels == 1).sum())
    num_neg = int((labels == 0).sum())
    if num_pos <= 0:
        print("[EDL] Warning: no positive samples in this training fold; using unweighted EDL CE.")
        return None

    pos_weight = float(num_neg / num_pos)
    print(f"[EDL] Weighted CE enabled: neg={num_neg}, pos={num_pos}, pos_weight={pos_weight:.4f}")
    return [1.0, pos_weight]


def edl_train_loop(train_loader, valid_loader, model, optimizer, scheduler, scaler, 
                   criterion, output_path, args, device, valid_split_name='val'):
    """Full training loop across all epochs."""
    
    best_aucroc = -float('inf')
    best_val_loss = float('inf')
    best_epoch = 0
    best_val_stats = None
    best_checkpoint_path = output_path / 'best_model.pth'
    epochs_without_improvement = 0
    early_stop_patience = max(0, int(getattr(args, 'early_stop_patience', 0)))
    early_stop_min_delta = max(0.0, float(getattr(args, 'early_stop_min_delta', 0.0)))
    
    train_results = {'loss': [], 'f1': [], 'bacc': [], 'auc_roc': [], 'lr': []}
    val_results = {'loss': [], 'f1': [], 'bacc': [], 'auc_roc': []}
    
    for epoch in range(args.epochs):
        print(f"\n-------- Epoch {epoch + 1}/{args.epochs} --------")
        start_time = time.time()
        
        # Train
        train_stats = edl_train_fn(train_loader, model, criterion, optimizer, epoch, args, scheduler, scaler, device)
        
        # Validate
        val_targs, val_preds, val_probs, val_stats, _ = edl_valid_fn(
            valid_loader, model, args, device, split=valid_split_name, epoch=epoch)
        
        elapsed = time.time() - start_time
        
        valid_display_name = 'Test' if valid_split_name == 'test' else 'Val'
        print(f"\nTrain Loss: {train_stats['loss']:.4f} | F1: {train_stats['f1']:.4f} | BAcc: {train_stats['bacc']:.4f} | AUC: {train_stats['auc_roc']:.4f}")
        print(f"{valid_display_name}   Loss: {val_stats['loss']:.4f} | F1: {val_stats['f1']:.4f} | BAcc: {val_stats['bacc']:.4f} | AUC: {val_stats['auc_roc']:.4f}")
        
        train_results['loss'].append(train_stats['loss'])
        train_results['f1'].append(train_stats['f1'])
        train_results['bacc'].append(train_stats['bacc'])
        train_results['auc_roc'].append(train_stats['auc_roc'])
        train_results['lr'].append(train_stats['lr'])
        
        val_results['loss'].append(val_stats['loss'])
        val_results['f1'].append(val_stats['f1'])
        val_results['bacc'].append(val_stats['bacc'])
        val_results['auc_roc'].append(val_stats['auc_roc'])
        save_loss_curve(train_results, val_results, output_path)
        
        # Save best model. AUC is undefined when the validation fold contains
        # only one class, which can happen with very rare positives and grouped
        # k-fold splitting. Fall back to validation loss in that case so the
        # fold still produces a usable checkpoint.
        val_auc = val_stats['auc_roc']
        val_auc_is_valid = np.isfinite(val_auc)
        annealing_coeff = (
            criterion.get_annealing_coeff(epoch)
            if hasattr(criterion, 'get_annealing_coeff')
            else 1.0
        )
        annealing_complete = annealing_coeff >= 1.0
        should_save = (
            (val_auc_is_valid and val_auc > best_aucroc + early_stop_min_delta)
            or (not val_auc_is_valid and val_stats['loss'] < best_val_loss - early_stop_min_delta)
            or best_val_stats is None
        )

        if should_save:
            epochs_without_improvement = 0
            if val_auc_is_valid:
                best_aucroc = val_stats['auc_roc']
            best_val_loss = val_stats['loss']
            best_val_stats = val_stats
            best_epoch = epoch + 1
            
            best_checkpoint_path = output_path / 'best_model.pth'
            if val_auc_is_valid:
                print(f'Epoch {epoch + 1} - Save best AUC: {best_aucroc:.4f}')
            else:
                print(f"Epoch {epoch + 1} - {valid_display_name} AUC is undefined; save best validation loss: {best_val_loss:.4f}")
            
            torch.save({
                'model': model.state_dict(),
                'epoch': epoch,
                'auroc': val_stats['auc_roc'],
                'f1': val_stats['f1'],
                'bacc': val_stats['bacc'],
                'dir_path': output_path,
            }, best_checkpoint_path)
        elif early_stop_patience > 0 and not annealing_complete:
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
        
        if np.isfinite(best_aucroc):
            print(f'\nBest AUC-ROC at epoch {best_epoch}: {best_aucroc:.4f}')
        else:
            print(f'\nBest validation loss at epoch {best_epoch}: {best_val_loss:.4f} (AUC undefined)')

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


def do_edl_training(args, device):
    """Main EDL training function with k-fold cross-validation."""
    
    args.n_class = 1
    
    # Data setup
    args.data_dir = Path(args.data_dir)
    args.df = pd.read_csv(args.data_dir / args.csv_file)
    args.df = args.df.fillna(0)
    
    print(f"df shape: {args.df.shape}")
    print(args.df.columns)
    
    _, dev_df, test_df = split_df_by_cohorts(
        args.df,
        train_cohorts=args.train_cohorts,
        test_cohorts=args.test_cohorts,
    )
    
    if args.data_frac < 1.0:
        dev_df = dev_df.sample(frac=args.data_frac, random_state=1, ignore_index=True)
    
    # Output setup
    now = datetime.now().strftime('%Y-%m-%d')
    args.output_path = Path(f"{args.output_dir}/EDL/{args.dataset}_{args.label}/fold_{args.n_folds}/{now}")
    os.makedirs(args.output_path, exist_ok=True)
    print(f"Output path: {args.output_path}")
    
    # Save config
    args_dict = {
        k: str(v) if isinstance(v, Path) else v
        for k, v in args.__dict__.items()
        if k != 'df'
    }
    with open(args.output_path / "args.yaml", 'w') as f:
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
            context="EDL n_folds=0 internal train/val split",
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

    # Track fold results
    all_val_results = []
    
    # Store fold info for each validation sample
    fold_assignments = []
    
    for fold, (train_df, val_df) in enumerate(split_iter):
        if fold < args.start_fold:
            continue

        print(f'\n{"="*60}')
        print(f'  EDL Fold {fold} / {total_folds}')
        print(f'{"="*60}')
        
        args.cur_fold = fold
        seed_all(args.seed + fold)
        
        path_results_fold = args.output_path / f'fold_{fold}'
        Path(path_results_fold).mkdir(parents=True, exist_ok=True)
        
        valid_split_name = 'val'
        print(f"Train: {len(train_df)}, {valid_split_name.capitalize()}: {len(val_df)}")

        train_loader = MIL_dataloader(train_df, 'train', args)
        valid_loader = MIL_dataloader(val_df, valid_split_name, args)
        
        # Build EDL model (MIL + EDL head)
        pretrained_checkpoint = resolve_mil_checkpoint(args.resume, fold)
        if args.resume is not None and pretrained_checkpoint is None:
            print(f"[EDL] Warning: no checkpoint found under {args.resume} for fold {fold}; training from scratch.")
        model = build_edl_model(args, pretrained_checkpoint)
        if args.train_edl_only:
            freeze_mil_backbone_train_edl_only(model)
            print("[EDL] Freeze mode enabled: training only EDL head(s); MIL backbone is frozen.")
        model.to(device)
        
        # Print model info
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")
        
        # Optimizer (full fine-tuning by default, or only EDL heads with --train_edl_only)
        trainable_parameters = [p for p in model.parameters() if p.requires_grad]
        if not trainable_parameters:
            raise RuntimeError("No trainable parameters found. Check the EDL freeze configuration.")
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
        # Scheduler
        total_steps = len(train_loader) * args.epochs
        warmup_steps = len(train_loader) if args.warmup_epochs == 1 else 10
        warmup_steps = 0 if total_steps <= 1 else min(warmup_steps, total_steps - 1)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps
        )
        
        # Scaler
        scaler = torch.cuda.amp.GradScaler(enabled=args.apex and device.type == 'cuda')
        
        # EDL Loss. Match the original weighted-BCE behavior by up-weighting
        # positive samples in the EDL cross-entropy term.
        class_weights = None
        if getattr(args, 'weighted_BCE', 'n') == 'y':
            class_weights = get_edl_class_weights(train_df, args.label)

        criterion = EDLCombinedLoss(
            num_classes=2,
            kl_weight=args.edl_kl_weight,
            annealing_start=args.edl_annealing_start,
            annealing_epochs=args.edl_annealing_epochs,
            class_weights=class_weights,
        )
        
        # Train
        val_stats, best_checkpoint_path = edl_train_loop(
            train_loader, valid_loader, model, optimizer, scheduler, scaler,
            criterion, path_results_fold, args, device, valid_split_name=valid_split_name
        )

        fold_summary = {
            'fold': fold,
            'auc_roc': val_stats['auc_roc'],
            'f1': val_stats['f1'],
            'bacc': val_stats['bacc'],
            'loss': val_stats['loss'],
            'eval_source': 'internal_val' if single_internal_val else 'cross_val',
        }
        all_val_results.append(fold_summary)
        
        # Load best model and generate predictions for all splits
        print(f"\nGenerating predictions with best model for fold {fold}...")
        checkpoint = torch.load(best_checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        
        # Generate predictions for all data (train + val + test)
        all_split_dfs = []
        split_specs = [('train', train_df), ('val', val_df), ('test', test_df)]
        for split_name, split_df in split_specs:
            if split_df is None or len(split_df) == 0:
                continue
            
            loader = MIL_dataloader(split_df, 'test', args)
            _, _, _, _, sample_results = edl_valid_fn(
                loader, model, args, device, split=split_name, epoch=args.epochs
            )
            
            pred_df = split_df.copy().reset_index(drop=True)
            label_col = args.label.lower()
            
            pred_df['prediction_score'] = sample_results['score']
            pred_df['predicted_class'] = sample_results['pred']
            pred_df[label_col] = sample_results['label']
            pred_df['evidence_0'] = sample_results['evidence_0']
            pred_df['evidence_1'] = sample_results['evidence_1']
            pred_df['alpha_0'] = sample_results['alpha_0']
            pred_df['alpha_1'] = sample_results['alpha_1']
            pred_df['uncertainty'] = sample_results['uncertainty']
            pred_df['fold'] = fold
            pred_df['split'] = split_name
            if 'cohort_num' not in pred_df.columns and 'cohert_num' in pred_df.columns:
                pred_df['cohort_num'] = pred_df['cohert_num']
            
            for col in ['patient_id', 'image_id', 'cohort_num']:
                if col not in pred_df.columns:
                    pred_df[col] = None
            
            keep_cols = ['patient_id', 'image_id', 'split', 'cohort_num', label_col,
                        'prediction_score', 'predicted_class',
                        'evidence_0', 'evidence_1', 'alpha_0', 'alpha_1', 'uncertainty', 'fold']
            keep_cols = [c for c in keep_cols if c in pred_df.columns]
            all_split_dfs.append(pred_df[keep_cols])
            
            # Record fold assignment for val samples
            if split_name == 'val':
                for _, row in pred_df.iterrows():
                    fold_assignments.append(row.to_dict())
        
        # Save fold predictions
        if all_split_dfs:
            fold_pred_df = pd.concat(all_split_dfs, ignore_index=True)
            fold_pred_df.to_csv(path_results_fold / f'{args.dataset}_edl_predictions_fold_{fold}.csv', index=False)
            print(f"Saved fold {fold} predictions: {len(fold_pred_df)} samples")
        
        del model
        clear_memory()
    
    # Save summary results
    summary_df = pd.DataFrame(all_val_results)
    if len(summary_df) > 1:
        metric_cols = [col for col in summary_df.columns if col not in ['fold', 'eval_source']]
        mean_std = summary_df[metric_cols].agg(['mean', 'std']).reset_index(drop=True)
        mean_std['fold'] = ['mean', 'std']
        mean_std['eval_source'] = 'summary'
        summary_df = pd.concat([summary_df, mean_std], ignore_index=True)
    
    summary_df.to_csv(args.output_path / 'edl_results_summary.csv', index=False)
    print(f"\nResults summary saved to {args.output_path / 'edl_results_summary.csv'}")
    print(summary_df.to_string())
    
    # Save fold assignments
    if fold_assignments:
        fold_df = pd.DataFrame(fold_assignments)
        fold_df.to_csv(args.output_path / f'{args.dataset}_edl_val_fold_assignments.csv', index=False)
        print(f"Fold assignments saved ({len(fold_df)} validation samples)")
    
    return args.output_path


def main():
    args = config()
    
    # Setup
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print(f"[INFO] Using GPU {args.gpu_id}")
    
    seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    args.apex = True if args.apex == "y" else False
    
    # Clean up
    if hasattr(args, 'df'):
        del args.df
    torch.cuda.empty_cache()
    
    # ===== Training =====
    output_path = do_edl_training(args, device)
    
    # ===== Auto Test =====
    print("\n" + "=" * 60)
    print("  Training complete. Starting automatic EDL testing...")
    print("=" * 60)
    
    from edl_test import run_edl_test
    
    test_output_dir = output_path / 'edl_test_results'
    run_edl_test(args, device, checkpoint_dir=output_path, output_dir=test_output_dir)
    
    print("\n===== EDL Training + Testing Pipeline Complete =====")


if __name__ == "__main__":
    main()
