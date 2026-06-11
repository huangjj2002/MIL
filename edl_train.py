

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
from MIL.edl_models import BagEmbeddingDSTModel, MIL_EDL_Wrapper
from MIL.edl_losses import DSTNLLLoss
from utils.metrics import auroc, evaluate_metrics
from utils.generic_utils import seed_all, AverageMeter, clear_memory
from utils.data_split_utils import (
    adaptive_stratified_train_val_split,
    generator_cross_val_folds,
    split_df_by_cohorts,
)


def config():
    parser = argparse.ArgumentParser(description="DST Training Script")
    
    # ===== DST-specific arguments =====
    parser.add_argument('--edl_kl_weight', type=float, default=0.1, help='Legacy EDL argument ignored by DST loss.')
    parser.add_argument('--edl_annealing_epochs', type=int, default=10, help='Legacy EDL argument ignored by DST loss.')
    parser.add_argument('--edl_annealing_start', type=int, default=0, help='Legacy EDL argument ignored by DST loss.')
    parser.add_argument('--edl_dropout', type=float, default=0.0, help='Dropout for DST head')
    parser.add_argument('--dst_k', '--dst-k', dest='dst_k', default=4, type=int,
                        help='Number of DST prototypes per class for the vanilla DST head.')
    parser.add_argument('--dst_topk', '--dst-topk', dest='dst_topk', default=0, type=int,
                        help='Number of top DST prototypes to keep in the vanilla head output.')
    parser.add_argument('--dst_gamma_init', '--dst-gamma-init', dest='dst_gamma_init',
                        default=1.0, type=float,
                        help='Initial DST distance sharpness.')
    parser.add_argument('--dst_alpha_init', '--dst-alpha-init', dest='dst_alpha_init',
                        default=0.0, type=float,
                        help='Initial DST prototype reliability logit.')
    parser.add_argument('--dst_normalize', '--dst-normalize', dest='dst_normalize',
                        default='y', choices=['y', 'n'],
                        help='Normalize bag vectors and prototypes before DST distance computation.')
    parser.add_argument('--edl_focal_gamma', '--edl-focal-gamma', dest='edl_focal_gamma',
                        type=float, default=0.0,
                        help='Legacy EDL argument ignored by DST loss.')
    parser.add_argument('--edl_wrong_evidence_penalty_weight', '--edl-wrong-evidence-penalty-weight',
                        dest='edl_wrong_evidence_penalty_weight', type=float, default=0.0,
                        help='Legacy EDL argument ignored by DST loss.')
    parser.add_argument('--edl_wrong_evidence_margin', '--edl-wrong-evidence-margin',
                        dest='edl_wrong_evidence_margin', type=float, default=0.05,
                        help='Legacy EDL argument ignored by DST loss.')
    parser.add_argument('--edl_wrong_evidence_class_balanced', '--edl-wrong-evidence-class-balanced',
                        dest='edl_wrong_evidence_class_balanced', default='y', choices=['y', 'n'],
                        help='Legacy EDL argument ignored by DST loss.')
    parser.add_argument('--edl_loss_weight_normalization', '--edl-loss-weight-normalization',
                        dest='edl_loss_weight_normalization', default='legacy_mean',
                        choices=['legacy_mean', 'weighted_mean'],
                        help='Legacy EDL argument ignored by DST loss.')
    parser.add_argument('--train_edl_only', '--freeze_backbone', action='store_true', default=False,
                        help='Freeze the MIL backbone and train only DST head(s)')
    
    # ===== Folder arguments =====
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument('--output_dir', default='Mammo-CLIP-output/edl_experiments', help='Path to output logs')
    parser.add_argument("--data_dir", default="datasets/Vindir-mammoclip", type=str)
    parser.add_argument("--clip_chk_pt_path", default=None, type=str,
                        help="Path to Mammo-CLIP checkpoint; required when --feature_extraction online")
    parser.add_argument("--csv_file", default="grouped_df.csv", type=str)
    parser.add_argument('--feat_dir', default='new_extracted_features', type=str)
    parser.add_argument('--embedding_cache_dir', '--embedding-cache-dir',
                        dest='embedding_cache_dir', default=None, type=str,
                        help='Path to extract_origin_embeddings.py bag_origin cache.')
    parser.add_argument("--img_dir", default="VinDir_preprocessed_mammoclip/images_png", type=str)
    parser.add_argument('--train', action='store_true', default=True)
    
    # ===== Data settings =====
    parser.add_argument("--img-size", "--img_size", dest="img_size", nargs='+',
                        type=int, default=[1520, 912])
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

    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        return checkpoint['model']
    return checkpoint


def _infer_bag_embedding_dim(args):
    if getattr(args, 'embedding_cache_dir', None) is None:
        raise ValueError("--embedding_cache_dir is required when --feature_extraction bag_embedding.")
    embeddings_path = Path(args.embedding_cache_dir) / "embeddings.npy"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Bag embeddings not found: {embeddings_path}")
    embeddings = np.load(embeddings_path, mmap_mode='r')
    if embeddings.ndim != 2:
        raise ValueError(f"Expected embeddings.npy to be 2D, got shape {embeddings.shape}.")
    return int(embeddings.shape[1])


def _looks_like_edl_state_dict(state_dict):

    if not isinstance(state_dict, dict):
        return False
    return any(
        key.startswith(('mil_model.', 'edl_head.', 'edl_side_heads.', 'dst_head.'))
        or 'ds_module' in key
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

    args.n_class = 1
    if args.feature_extraction == 'online' and not getattr(args, 'clip_chk_pt_path', None):
        raise ValueError(
            "--clip_chk_pt_path is required when --feature_extraction online "
            "so the Mammo-CLIP image encoder/backbone can be initialized."
        )

    checkpoint_state = None
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.is_file():
            checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
            checkpoint_state = _get_checkpoint_state_dict(checkpoint)
        else:
            print(f"[DST] Warning: checkpoint not found at {checkpoint_path}, training from scratch.")

    is_dst_checkpoint = _looks_like_edl_state_dict(checkpoint_state)
    if args.feature_extraction == 'bag_embedding':
        model = BagEmbeddingDSTModel(
            in_features=_infer_bag_embedding_dim(args),
            edl_dropout=args.edl_dropout,
            dst_k=args.dst_k,
            dst_topk=args.dst_topk,
            dst_normalize=args.dst_normalize,
            dst_gamma_init=args.dst_gamma_init,
            dst_alpha_init=args.dst_alpha_init,
        )
        if checkpoint_state is not None:
            load_msg = model.load_state_dict(checkpoint_state, strict=False)
            print(f"[DST] Loaded DST checkpoint from: {checkpoint_path}")
            _print_load_summary("[DST][load]", load_msg)
        return model

    mil_model = build_model(args)

    if checkpoint_state is not None and not is_dst_checkpoint:
        load_msg = mil_model.load_state_dict(checkpoint_state, strict=False)
        print(f"[DST] Loaded pretrained MIL backbone from: {checkpoint_path}")
        _print_load_summary("[DST][MIL load]", load_msg)

    dst_model = MIL_EDL_Wrapper(
        mil_model,
        edl_dropout=args.edl_dropout,
        dst_k=args.dst_k,
        dst_topk=args.dst_topk,
        dst_normalize=args.dst_normalize,
        dst_gamma_init=args.dst_gamma_init,
        dst_alpha_init=args.dst_alpha_init,
    )
    if checkpoint_state is not None and is_dst_checkpoint:

        load_msg = dst_model.load_state_dict(checkpoint_state, strict=False)
        print(f"[DST] Loaded DST training checkpoint from: {checkpoint_path}")
        _print_load_summary("[DST][DST load]", load_msg)

    return dst_model


def freeze_mil_backbone_train_edl_only(model):

    if hasattr(model, 'mil_model'):
        for param in model.mil_model.parameters():
            param.requires_grad = False

    edl_head = getattr(model, 'edl_head', None)
    if edl_head is not None:
        for param in edl_head.parameters():
            param.requires_grad = True

    dst_head = getattr(model, 'dst_head', None)
    if dst_head is not None:
        for param in dst_head.parameters():
            param.requires_grad = True

    edl_side_heads = getattr(model, 'edl_side_heads', None)
    if edl_side_heads is not None:
        for param in edl_side_heads.parameters():
            param.requires_grad = True


def keep_frozen_mil_backbone_in_eval(model, args):

    if getattr(args, 'train_edl_only', False) and hasattr(model, 'mil_model'):
        model.mil_model.eval()


EDL_LOSS_DIAGNOSTIC_KEYS = [
    'nll_loss',
    'ce_loss',
    'data_loss',
    'kl_loss',
    'annealing',
    'wrong_evidence_penalty',
    'margin_violation_mean',
    'total_evidence_mean',
    'focal_factor_mean',
    'sample_weight_mean',
    'focal_weighted_denominator',
    'mass_0_mean',
    'mass_1_mean',
    'mass_omega_mean',
    'total_loss',
    'class0_nll_loss_mean',
    'class0_mass_mean',
    'class1_nll_loss_mean',
    'class1_mass_mean',
]


def build_edl_criterion(args, class_weights=None):
    return DSTNLLLoss(class_weights=class_weights)


def _new_loss_meters():
    return {key: AverageMeter() for key in EDL_LOSS_DIAGNOSTIC_KEYS}


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
        if key.startswith('class0_'):
            weight = int(loss_dict.get('class0_n', 0))
        elif key.startswith('class1_'):
            weight = int(loss_dict.get('class1_n', 0))
        if weight > 0:
            meter.update(float(value), weight)


def _loss_meter_averages(meters):
    return {key: meter.avg for key, meter in meters.items() if meter.count > 0}


def _append_epoch_stats(history, stats):
    for key in history:
        if key in stats:
            history[key].append(stats[key])
        elif key not in {'loss', 'f1', 'bacc', 'auc_roc', 'lr'}:
            history[key].append(float('nan'))


def _init_epoch_history(include_lr=False):
    history = {'loss': [], 'f1': [], 'bacc': [], 'auc_roc': []}
    if include_lr:
        history['lr'] = []
    for key in EDL_LOSS_DIAGNOSTIC_KEYS:
        history[key] = []
    return history


def use_train_eval_metrics(args):
    return getattr(args, 'feature_extraction', None) == 'bag_embedding'


def build_train_eval_loader(train_df, args):
    if not use_train_eval_metrics(args):
        return None
    return MIL_dataloader(train_df, 'test', args)


def get_train_curve_metadata(train_eval_loader=None):
    if train_eval_loader is not None:
        return 'train_eval', 'Train Eval'
    return 'train', 'Train'


def format_fold_label(output_path):
    fold_name = Path(output_path).name.replace('_', ' ')
    return fold_name


def edl_train_fn(train_loader, model, criterion, optimizer, epoch, args, scheduler, scaler, device):

    
    model.train()
    keep_frozen_mil_backbone_in_eval(model, args)
    model.is_training = True
    
    losses = AverageMeter()
    ce_losses = AverageMeter()
    kl_losses = AverageMeter()
    loss_meters = _new_loss_meters()
    
    progress_iter = tqdm(enumerate(train_loader),
                         desc=f"[{epoch + 1:03d}/{args.epochs:03d} DST train]",
                         total=len(train_loader))
    
    targs = []
    probs_list = []
    preds_list = []
    uncertainty_list = []
    
    for step, data in progress_iter:
      
        if isinstance(data['x'], dict):
            inputs = {scale: tensor.to(device) for scale, tensor in data['x'].items()}
        elif isinstance(data['x'], list):
            inputs = [tensor.to(device) for tensor in data['x']]
        else:
            inputs = data['x'].to(device)
        
        labels = data['y'].long().to(device)  
        batch_size = labels.size(0)
        
        amp_enabled = args.apex and device.type == 'cuda'
        with torch.cuda.amp.autocast(enabled=amp_enabled):
        
            edl_out = model(inputs)
            loss, loss_dict = criterion(edl_out, labels, epoch=epoch)
            for side_out in edl_out.get('side_outputs', {}).values():
                side_loss, _ = criterion(side_out, labels, epoch=epoch)
                loss = loss + side_loss
        
        losses.update(loss.item(), batch_size)
        ce_losses.update(loss_dict['ce_loss'], batch_size)
        kl_losses.update(loss_dict['kl_loss'], batch_size)
        _update_loss_meters(loss_meters, loss_dict, batch_size)
        
       
        scaler.scale(loss).backward()
        
        if args.clip_grad > 0.0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
        
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()
        
   
        prob = edl_out['prob'].detach()  # (B, K)
        pred_class = torch.argmax(prob, dim=-1)  # (B,)
        uncertainty = edl_out['uncertainty'].detach()  # (B,)
        
        targs.append(labels.cpu().numpy())
        probs_list.append(prob[:, 1].cpu().numpy())  
        preds_list.append(pred_class.cpu().numpy())
        uncertainty_list.append(uncertainty.cpu().numpy())
        cuda_mem = torch.cuda.memory_usage(device) if device.type == 'cuda' and torch.cuda.is_available() else 0
        
        progress_iter.set_postfix({
            "loss": f"{losses.avg:.4f}",
            "ce": f"{ce_losses.avg:.4f}",
            "kl": f"{kl_losses.avg:.4f}",
            "wep": f"{loss_meters['wrong_evidence_penalty'].avg:.4f}",
            "viol": f"{loss_meters['margin_violation_mean'].avg:.4f}",
            "omega": f"{loss_meters['mass_omega_mean'].avg:.4f}",
            "CUDA-Mem": f"{cuda_mem}%",
        })
    

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
    train_stats.update(_loss_meter_averages(loss_meters))
    
    return train_stats


@torch.no_grad()
def edl_valid_fn(valid_loader, model, args, device, split='val', epoch=1, criterion_eval=None):
 
    
    model.eval()
    model.is_training = False
    
    losses = AverageMeter()
    if criterion_eval is None:
        criterion_eval = build_edl_criterion(args, class_weights=None)
    loss_meters = _new_loss_meters()
    
    targs = []
    probs_list = []
    preds_list = []
    uncertainty_list = []
    mass_list = []
    
    sample_patient_ids = []
    sample_image_ids = []
    
    if split == 'val':
        progress_iter = tqdm(enumerate(valid_loader),
                             desc=f"[{epoch + 1:03d}/{args.epochs:03d} DST valid]",
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
            loss, loss_dict = criterion_eval(edl_out, labels, epoch=epoch)
            for side_out in edl_out.get('side_outputs', {}).values():
                side_loss, _ = criterion_eval(side_out, labels, epoch=epoch)
                loss = loss + side_loss
        
        losses.update(loss.item(), batch_size)
        _update_loss_meters(loss_meters, loss_dict, batch_size)
        
        prob = edl_out['prob'].detach()
        pred_class = torch.argmax(prob, dim=-1)
        uncertainty = edl_out['uncertainty'].detach()
        mass = edl_out['dst_mass'].detach()
        
        targs.append(labels.cpu().numpy())
        probs_list.append(prob[:, 1].cpu().numpy())
        preds_list.append(pred_class.cpu().numpy())
        uncertainty_list.append(uncertainty.cpu().numpy())
        mass_list.append(mass.cpu().numpy())
    
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
    val_stats.update(_loss_meter_averages(loss_meters))
    
    sample_results = {
        'patient_id': sample_patient_ids,
        'image_id': sample_image_ids,
        'label': targs.tolist(),
        'score': probs.tolist(),
        'pred': preds.tolist(),
        'uncertainty': np.concatenate(uncertainty_list).tolist(),
        'mass_0': np.concatenate(mass_list)[:, 0].tolist(),
        'mass_1': np.concatenate(mass_list)[:, 1].tolist(),
        'mass_omega': np.concatenate(mass_list)[:, 2].tolist(),
    }
    
    return targs, preds, probs, val_stats, sample_results


def save_loss_curve(train_results, val_results, output_path, train_prefix='train', train_label='Train', plot_title='DST Loss Curve'):

    if not train_results['loss'] or not val_results['loss']:
        return

    output_path = Path(output_path)
    n_epochs = len(train_results['loss'])
    curve_data = {
        'epoch': np.arange(1, len(train_results['loss']) + 1),
        f'{train_prefix}_loss': train_results['loss'],
        'val_loss': val_results['loss'],
        f'{train_prefix}_auc_roc': train_results['auc_roc'],
        'val_auc_roc': val_results['auc_roc'],
        f'{train_prefix}_f1': train_results['f1'],
        'val_f1': val_results['f1'],
        f'{train_prefix}_bacc': train_results['bacc'],
        'val_bacc': val_results['bacc'],
        'lr': train_results['lr'],
    }
    base_keys = {'loss', 'auc_roc', 'f1', 'bacc', 'lr'}
    for key, values in train_results.items():
        if key not in base_keys and len(values) == n_epochs:
            curve_data[f'{train_prefix}_{key}'] = values
    for key, values in val_results.items():
        if key not in base_keys and len(values) == n_epochs:
            curve_data[f'val_{key}'] = values
    curve_df = pd.DataFrame(curve_data)
    curve_df.to_csv(output_path / 'dst_loss_curve.csv', index=False)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(10, 5.75))
        ax.plot(
            curve_df['epoch'],
            curve_df[f'{train_prefix}_loss'],
            color='#1f77b4',
            linewidth=2.2,
            label=f'{train_label.lower()} loss',
        )
        ax.plot(
            curve_df['epoch'],
            curve_df['val_loss'],
            color='#d62728',
            linewidth=2.2,
            label='val loss',
        )
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.set_title(plot_title)
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right', frameon=False)
        fig.tight_layout()
        fig.savefig(output_path / 'dst_loss_curve.png', dpi=200)
        plt.close(fig)
    except Exception as exc:
        print(f"[DST] Warning: failed to save loss curve plot: {exc}")


def get_edl_class_weights(train_df, label_col):

    labels = train_df[label_col].astype(int)
    num_pos = int((labels == 1).sum())
    num_neg = int((labels == 0).sum())
    if num_pos <= 0:
        print("[DST] Warning: no positive samples in this training fold; using unweighted DST NLL.")
        return None

    pos_weight = float(num_neg / num_pos)
    print(f"[DST] Weighted NLL enabled: neg={num_neg}, pos={num_pos}, pos_weight={pos_weight:.4f}")
    return [1.0, pos_weight]


def edl_train_loop(train_loader, valid_loader, model, optimizer, scheduler, scaler,
                   criterion, output_path, args, device, valid_split_name='val',
                   train_eval_loader=None):
 
    
    best_aucroc = -float('inf')
    best_val_loss = float('inf')
    best_epoch = 0
    best_val_stats = None
    best_checkpoint_path = output_path / 'best_model.pth'
    epochs_without_improvement = 0
    early_stop_patience = max(0, int(getattr(args, 'early_stop_patience', 0)))
    early_stop_min_delta = max(0.0, float(getattr(args, 'early_stop_min_delta', 0.0)))
    
    train_results = _init_epoch_history(include_lr=True)
    val_results = _init_epoch_history(include_lr=False)
    train_prefix, train_label = get_train_curve_metadata(train_eval_loader)
    
    for epoch in range(args.epochs):
        print(f"\n-------- Epoch {epoch + 1}/{args.epochs} --------")
        start_time = time.time()
        
        
        train_stats = edl_train_fn(train_loader, model, criterion, optimizer, epoch, args, scheduler, scaler, device)
        curve_train_stats = train_stats
        if train_eval_loader is not None:
            _, _, _, curve_train_stats, _ = edl_valid_fn(
                train_eval_loader,
                model,
                args,
                device,
                split='train_eval',
                epoch=epoch,
                criterion_eval=criterion,
            )
            curve_train_stats['lr'] = train_stats['lr']
        
     
        val_targs, val_preds, val_probs, val_stats, _ = edl_valid_fn(
            valid_loader,
            model,
            args,
            device,
            split=valid_split_name,
            epoch=epoch,
            criterion_eval=criterion,
        )
        
        elapsed = time.time() - start_time
        
        valid_display_name = 'Test' if valid_split_name == 'test' else 'Val'
        print(f"\n{train_label} Loss: {curve_train_stats['loss']:.4f} | F1: {curve_train_stats['f1']:.4f} | BAcc: {curve_train_stats['bacc']:.4f} | AUC: {curve_train_stats['auc_roc']:.4f}")
        print(f"{valid_display_name}   Loss: {val_stats['loss']:.4f} | F1: {val_stats['f1']:.4f} | BAcc: {val_stats['bacc']:.4f} | AUC: {val_stats['auc_roc']:.4f}")
        
        _append_epoch_stats(train_results, curve_train_stats)
        _append_epoch_stats(val_results, val_stats)
        plot_title = f"DST k={getattr(args, 'dst_k', 0)} - {format_fold_label(output_path)}"
        save_loss_curve(
            train_results,
            val_results,
            output_path,
            train_prefix=train_prefix,
            train_label=train_label,
            plot_title=plot_title,
        )
        
  
        val_auc = val_stats['auc_roc']
        val_auc_is_valid = np.isfinite(val_auc)
        annealing_coeff = (
            criterion.get_annealing_coeff(epoch)
            if hasattr(criterion, 'get_annealing_coeff')
            else 1.0
        )
        annealing_complete = annealing_coeff >= 1.0
        should_save = (
            val_stats['loss'] < best_val_loss - early_stop_min_delta
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
            print(
                f"Epoch {epoch + 1} - Save best validation loss: {best_val_loss:.4f} "
                f"(AUC: {val_stats['auc_roc']:.4f})"
            )
            
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
        
        print(f'\nBest validation loss at epoch {best_epoch}: {best_val_loss:.4f}')

        if early_stop_patience > 0:
            if not annealing_complete:
                print(
                    "Early stopping paused until DST warmup completes "
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

    
    args.n_class = 1
    
   
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
    
  
    now = datetime.now().strftime('%Y-%m-%d')
    args.output_path = Path(f"{args.output_dir}/DST/{args.dataset}_{args.label}/fold_{args.n_folds}/{now}")
    os.makedirs(args.output_path, exist_ok=True)
    print(f"Output path: {args.output_path}")
    
 
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
            context="DST n_folds=0 internal train/val split",
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

        print(f'\n{"="*60}')
        print(f'  DST Fold {fold} / {total_folds}')
        print(f'{"="*60}')
        
        args.cur_fold = fold
        seed_all(args.seed + fold)
        
        path_results_fold = args.output_path / f'fold_{fold}'
        Path(path_results_fold).mkdir(parents=True, exist_ok=True)
        
        valid_split_name = 'val'
        print(f"Train: {len(train_df)}, {valid_split_name.capitalize()}: {len(val_df)}")

        train_loader = MIL_dataloader(train_df, 'train', args)
        train_eval_loader = build_train_eval_loader(train_df, args)
        valid_loader = MIL_dataloader(val_df, valid_split_name, args)
        
       
        pretrained_checkpoint = resolve_mil_checkpoint(args.resume, fold)
        if args.resume is not None and pretrained_checkpoint is None:
            print(f"[DST] Warning: no checkpoint found under {args.resume} for fold {fold}; training from scratch.")
        model = build_edl_model(args, pretrained_checkpoint)
        if args.train_edl_only:
            freeze_mil_backbone_train_edl_only(model)
            print("[DST] Freeze mode enabled: training only DST head(s); MIL backbone is frozen.")
        model.to(device)
        
        
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Total params: {total_params:,}, Trainable: {trainable_params:,}")
        

        trainable_parameters = [p for p in model.parameters() if p.requires_grad]
        if not trainable_parameters:
            raise RuntimeError("No trainable parameters found. Check the EDL freeze configuration.")
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=args.lr,
            weight_decay=args.weight_decay
        )
        
     
        total_steps = len(train_loader) * args.epochs
        warmup_steps = len(train_loader) if args.warmup_epochs == 1 else 10
        warmup_steps = 0 if total_steps <= 1 else min(warmup_steps, total_steps - 1)
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            total_steps=total_steps,
            warmup_steps=warmup_steps
        )
        
     
        scaler = torch.cuda.amp.GradScaler(enabled=args.apex and device.type == 'cuda')
        
     
        class_weights = None
        if getattr(args, 'weighted_BCE', 'n') == 'y':
            class_weights = get_edl_class_weights(train_df, args.label)

        criterion = build_edl_criterion(args, class_weights=class_weights)
        
     
        val_stats, best_checkpoint_path = edl_train_loop(
            train_loader, valid_loader, model, optimizer, scheduler, scaler,
            criterion, path_results_fold, args, device,
            valid_split_name=valid_split_name,
            train_eval_loader=train_eval_loader,
        )

        fold_summary = {
            'fold': fold,
            'auc_roc': val_stats['auc_roc'],
            'f1': val_stats['f1'],
            'bacc': val_stats['bacc'],
            'loss': val_stats['loss'],
            'eval_source': 'internal_val' if single_internal_val else 'cross_val',
        }
        for key in EDL_LOSS_DIAGNOSTIC_KEYS:
            if key in val_stats:
                fold_summary[key] = val_stats[key]
        all_val_results.append(fold_summary)
        
 
        print(f"\nGenerating predictions with best model for fold {fold}...")
        checkpoint = torch.load(best_checkpoint_path, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['model'])
        model.eval()
        
        
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
            pred_df['mass_0'] = sample_results['mass_0']
            pred_df['mass_1'] = sample_results['mass_1']
            pred_df['mass_omega'] = sample_results['mass_omega']
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
                        'mass_0', 'mass_1', 'mass_omega', 'uncertainty', 'fold']
            keep_cols = [c for c in keep_cols if c in pred_df.columns]
            all_split_dfs.append(pred_df[keep_cols])
            
          
            if split_name == 'val':
                for _, row in pred_df.iterrows():
                    fold_assignments.append(row.to_dict())
        
       
        if all_split_dfs:
            fold_pred_df = pd.concat(all_split_dfs, ignore_index=True)
            fold_pred_df.to_csv(path_results_fold / f'{args.dataset}_dst_predictions_fold_{fold}.csv', index=False)
            print(f"Saved fold {fold} predictions: {len(fold_pred_df)} samples")
        
        del model
        clear_memory()
    
    
    summary_df = pd.DataFrame(all_val_results)
    if len(summary_df) > 1:
        metric_cols = [col for col in summary_df.columns if col not in ['fold', 'eval_source']]
        mean_std = summary_df[metric_cols].agg(['mean', 'std']).reset_index(drop=True)
        mean_std['fold'] = ['mean', 'std']
        mean_std['eval_source'] = 'summary'
        summary_df = pd.concat([summary_df, mean_std], ignore_index=True)
    
    summary_df.to_csv(args.output_path / 'dst_results_summary.csv', index=False)
    print(f"\nResults summary saved to {args.output_path / 'dst_results_summary.csv'}")
    print(summary_df.to_string())
    
 
    if fold_assignments:
        fold_df = pd.DataFrame(fold_assignments)
        fold_df.to_csv(args.output_path / f'{args.dataset}_dst_val_fold_assignments.csv', index=False)
        print(f"Fold assignments saved ({len(fold_df)} validation samples)")
    
    return args.output_path


def main():
    args = config()
    args.dst_normalize = args.dst_normalize == 'y'
    
  
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print(f"[INFO] Using GPU {args.gpu_id}")
    
    seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    args.apex = True if args.apex == "y" else False
  
    if hasattr(args, 'df'):
        del args.df
    torch.cuda.empty_cache()
    

    output_path = do_edl_training(args, device)
    

    print("\n" + "=" * 60)
    print("  Training complete. Starting automatic DST testing...")
    print("=" * 60)
    
    from edl_test import run_edl_test
    
    test_output_dir = output_path / 'dst_test_results'
    run_edl_test(args, device, checkpoint_dir=output_path, output_dir=test_output_dir)
    
    print("\n===== DST Training + Testing Pipeline Complete =====")


if __name__ == "__main__":
    main()
