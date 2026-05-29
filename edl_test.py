"""
EDL Test Script - Evidential Deep Learning Inference

Loads the trained EDL models from each fold, performs predictions on all data,
and generates a unified CSV with evidence, alpha, uncertainty, and fold columns.

The script reproduces the same k-fold splits used during training to determine
which fold model should be used for each validation sample. For test data, 
all fold models predict and results are aggregated.

Usage:
    python edl_test.py --checkpoint_dir path/to/edl_output --data_dir datasets/... --csv_file grouped_df.csv
"""

import os
import gc
import warnings
import argparse
import yaml
import math
from pathlib import Path

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from tqdm import tqdm

# Internal imports
from Datasets.dataset_utils import MIL_dataloader
from MIL import build_model
from MIL.edl_models import EDLHead, MIL_EDL_Wrapper
from MIL.edl_losses import EDLCombinedLoss
from utils.metrics import auroc, evaluate_metrics
from utils.generic_utils import seed_all, clear_memory
from utils.data_split_utils import (
    adaptive_stratified_train_val_split,
    generator_cross_val_folds,
    split_df_by_cohorts,
)
from sklearn.metrics import confusion_matrix


def config():
    parser = argparse.ArgumentParser(description="EDL Test Script")
    
    # ===== Checkpoint =====
    parser.add_argument('--checkpoint_dir', type=str, required=True,
                        help='Path to EDL training output directory containing fold_0, fold_1, ... subdirs')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Output directory for test results (default: checkpoint_dir/edl_test_results)')
    
    # ===== EDL-specific =====
    parser.add_argument('--edl_kl_weight', type=float, default=0.1)
    parser.add_argument('--edl_annealing_epochs', type=int, default=10)
    parser.add_argument('--edl_annealing_start', type=int, default=0)
    parser.add_argument('--edl_dropout', type=float, default=0.0)
    
    # ===== Data settings =====
    parser.add_argument("--gpu_id", type=str, default="0")
    parser.add_argument("--data_dir", default="datasets/Vindir-mammoclip", type=str)
    parser.add_argument("--csv_file", default="grouped_df.csv", type=str)
    parser.add_argument("--img_dir", default="VinDir_preprocessed_mammoclip/images_png", type=str)
    parser.add_argument("--clip_chk_pt_path", default=None, type=str,
                        help="Path to Mammo-CLIP checkpoint; required when --feature_extraction online")
    parser.add_argument('--train', action='store_true', default=False)
    parser.add_argument("--img-size", nargs='+', default=[1520, 912])
    parser.add_argument("--dataset", default="ViNDr", type=str)
    parser.add_argument("--label", default="Mass", type=str)
    parser.add_argument("--num-classes", default=1, type=int)
    parser.add_argument("--mean", default=0.3089279, type=float)
    parser.add_argument("--std", default=0.25053555408335154, type=float)
    parser.add_argument("--data_frac", default=1.0, type=float)
    
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
    
    # ===== Inference =====
    parser.add_argument("--batch-size", default=8, type=int)
    parser.add_argument("--apex", default="y", type=str)
    parser.add_argument("--num-workers", default=4, type=int)
    parser.add_argument("--seed", default=10, type=int)
    parser.add_argument("--n_folds", default=5, type=int)
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
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--data_aug", action='store_true', default=False)
    parser.add_argument('--roi_eval', action='store_true', default=False)
    parser.add_argument('--drop_classhead', type=float, default=0.0)
    parser.add_argument('--drop_attention_pool', type=float, default=0.0)
    parser.add_argument('--drop_mha', type=float, default=0.0)
    
    # ===== Resume (for MIL pretrained weights, optional) =====
    parser.add_argument('--resume', default=None, type=str)
    
    return parser.parse_args()


def _get_checkpoint_state_dict(checkpoint):
    """Return the model state dict from a torch checkpoint or raw state dict."""
    if isinstance(checkpoint, dict) and 'model' in checkpoint:
        return checkpoint['model']
    return checkpoint


def _print_load_summary(prefix, load_msg):
    missing = list(getattr(load_msg, 'missing_keys', []))
    unexpected = list(getattr(load_msg, 'unexpected_keys', []))
    print(f"{prefix} missing keys: {len(missing)}, unexpected keys: {len(unexpected)}")
    if missing[:5]:
        print(f"{prefix} first missing keys: {missing[:5]}")
    if unexpected[:5]:
        print(f"{prefix} first unexpected keys: {unexpected[:5]}")


def build_edl_model(args, checkpoint_path=None):
    """Build and load EDL model."""
    args.n_class = 1
    if args.feature_extraction == 'online' and not getattr(args, 'clip_chk_pt_path', None):
        raise ValueError(
            "--clip_chk_pt_path is required when --feature_extraction online "
            "so the Mammo-CLIP image encoder/backbone can be initialized."
        )

    mil_model = build_model(args)
    edl_model = MIL_EDL_Wrapper(mil_model, edl_dropout=args.edl_dropout)
    
    if checkpoint_path is not None:
        checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
        load_msg = edl_model.load_state_dict(_get_checkpoint_state_dict(checkpoint), strict=False)
        print(f"[EDL Test] Loaded EDL model from: {checkpoint_path}")
        _print_load_summary("[EDL Test][load]", load_msg)
    
    return edl_model


@torch.no_grad()
def edl_predict(loader, model, args, device):
    """
    Run EDL inference on a data loader.
    
    Returns:
        dict with sample-level predictions
    """
    model.eval()
    model.is_training = False
    
    targs = []
    probs_list = []
    preds_list = []
    uncertainty_list = []
    evidence_list = []
    alpha_list = []
    
    sample_patient_ids = []
    sample_image_ids = []
    
    progress_iter = tqdm(enumerate(loader), total=len(loader), desc="EDL predict")
    
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
        
        prob = edl_out['prob'].detach().cpu()
        evidence = edl_out['evidence'].detach().cpu()
        alpha = edl_out['alpha'].detach().cpu()
        uncertainty = edl_out['uncertainty'].detach().cpu()
        pred_class = torch.argmax(prob, dim=-1)
        
        targs.append(labels.cpu().numpy())
        probs_list.append(prob[:, 1].numpy())  # positive class prob
        preds_list.append(pred_class.numpy())
        uncertainty_list.append(uncertainty.numpy())
        evidence_list.append(evidence.numpy())
        alpha_list.append(alpha.numpy())
    
    results = {
        'patient_id': sample_patient_ids,
        'image_id': sample_image_ids,
        'label': np.concatenate(targs).tolist(),
        'score': np.concatenate(probs_list).tolist(),
        'pred': np.concatenate(preds_list).tolist(),
        'uncertainty': np.concatenate(uncertainty_list).tolist(),
        'evidence_0': np.concatenate(evidence_list)[:, 0].tolist(),
        'evidence_1': np.concatenate(evidence_list)[:, 1].tolist(),
        'alpha_0': np.concatenate(alpha_list)[:, 0].tolist(),
        'alpha_1': np.concatenate(alpha_list)[:, 1].tolist(),
    }
    
    return results


def run_edl_test(args, device, checkpoint_dir=None, output_dir=None):
    """
    EDL test function that can be called from training script or standalone.
    
    Args:
        args: configuration namespace
        device: torch device
        checkpoint_dir: path to checkpoints (overrides args.checkpoint_dir)
        output_dir: path for output (overrides args.output_dir)
    
    Returns:
        output_dir: Path to test results directory
    """
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
    elif hasattr(args, 'checkpoint_dir'):
        checkpoint_dir = Path(args.checkpoint_dir)
    else:
        raise ValueError("checkpoint_dir must be provided via argument or args.checkpoint_dir")
    
    # Set output dir
    if output_dir is not None:
        output_dir = Path(output_dir)
    elif hasattr(args, 'output_dir') and args.output_dir is not None:
        output_dir = Path(args.output_dir)
    else:
        output_dir = checkpoint_dir / 'edl_test_results'
    output_dir.mkdir(parents=True, exist_ok=True)
    seed_all(args.seed)
    
    # Load data
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

    # Prefer the fold assignment produced by training. If absent, recreate the
    # same split path after resetting the seed.
    assignment_path = checkpoint_dir / f'{args.dataset}_edl_val_fold_assignments.csv'
    fold_val_dfs = {}
    if assignment_path.exists():
        assignment_df = pd.read_csv(assignment_path)
        for fold in range(total_folds):
            fold_val_dfs[fold] = assignment_df[assignment_df['fold'] == fold].reset_index(drop=True)
    else:
        if single_internal_val:
            _, val_df_fold = adaptive_stratified_train_val_split(
                dev_df,
                val_frac=args.kfold0_val_frac,
                max_val_frac=args.kfold0_val_max_frac,
                args=args,
                context="EDL test n_folds=0 internal train/val split",
            )
            fold_val_dfs[0] = val_df_fold.reset_index(drop=True)
        else:
            for fold_idx, (_, val_df_fold) in enumerate(
                generator_cross_val_folds(dev_df, args.n_folds, args.label, random_state=args.seed)
            ):
                fold_val_dfs[fold_idx] = val_df_fold.reset_index(drop=True)
    
    label_col = args.label.lower()
    
    # ========== Predict on development data ==========
    print("\n===== Predicting on development data =====")
    
    all_dev_results = []
    dev_results_df = None
    test_ensemble = None
    
    for fold in range(total_folds):
        print(f"\n--- Fold {fold} ---")
        
        ckpt_path = checkpoint_dir / f'fold_{fold}' / 'best_model.pth'
        if not ckpt_path.exists():
            print(f"Warning: checkpoint not found at {ckpt_path}, skipping fold {fold}")
            continue
        
        model = build_edl_model(args, ckpt_path)
        model.to(device)
        
        val_df = fold_val_dfs.get(fold, pd.DataFrame()).reset_index(drop=True)
        if len(val_df) == 0:
            print(f"Warning: no validation rows found for fold {fold}, skipping")
            continue
        if args.label not in val_df.columns and label_col in val_df.columns:
            val_df[args.label] = val_df[label_col]
        
        val_loader = MIL_dataloader(val_df, 'test', args)
        val_results = edl_predict(val_loader, model, args, device)
        
        val_result_df = pd.DataFrame({
            'patient_id': val_results['patient_id'],
            'image_id': val_results['image_id'],
            'split': 'val',
            label_col: val_results['label'],
            'prediction_score': val_results['score'],
            'predicted_class': val_results['pred'],
            'evidence_0': val_results['evidence_0'],
            'evidence_1': val_results['evidence_1'],
            'alpha_0': val_results['alpha_0'],
            'alpha_1': val_results['alpha_1'],
            'uncertainty': val_results['uncertainty'],
            'fold': fold,
        })
        if 'cohort_num' not in val_result_df.columns and 'cohert_num' in val_df.columns:
            val_result_df['cohort_num'] = val_df['cohert_num'].values
        elif 'cohort_num' in val_df.columns:
            val_result_df['cohort_num'] = val_df['cohort_num'].values
        
        all_dev_results.append(val_result_df)
        
        targs = np.array(val_results['label'])
        probs = np.array(val_results['score'])
        preds = np.array(val_results['pred'])
        
        try:
            fold_auc = auroc(targs, probs)
            fold_f1, fold_bacc = evaluate_metrics(targs, preds)
            print(f"  Fold {fold} Val - AUC: {fold_auc:.4f}, F1: {fold_f1:.4f}, BAcc: {fold_bacc:.4f}")
        except Exception as e:
            print(f"  Fold {fold} metrics error: {e}")
        
        del model
        clear_memory()
    
    if all_dev_results:
        dev_results_df = pd.concat(all_dev_results, ignore_index=True)
        dev_results_df.to_csv(output_dir / f'{args.dataset}_edl_dev_predictions.csv', index=False)
        print(f"\nDev predictions saved: {len(dev_results_df)} samples")
    
    # ========== Predict on test data ==========
    if len(test_df) > 0:
        print("\n===== Predicting on test data =====")
        
        test_all_fold_results = []
        
        for fold in range(total_folds):
            print(f"\n--- Test with Fold {fold} model ---")
            
            ckpt_path = checkpoint_dir / f'fold_{fold}' / 'best_model.pth'
            if not ckpt_path.exists():
                print(f"Warning: checkpoint not found at {ckpt_path}, skipping")
                continue
            
            model = build_edl_model(args, ckpt_path)
            model.to(device)
            
            test_loader = MIL_dataloader(test_df, 'test', args)
            test_results = edl_predict(test_loader, model, args, device)
            
            test_result_df = pd.DataFrame({
                'patient_id': test_results['patient_id'],
                'image_id': test_results['image_id'],
                'split': 'test',
                label_col: test_results['label'],
                'prediction_score': test_results['score'],
                'predicted_class': test_results['pred'],
                'evidence_0': test_results['evidence_0'],
                'evidence_1': test_results['evidence_1'],
                'alpha_0': test_results['alpha_0'],
                'alpha_1': test_results['alpha_1'],
                'uncertainty': test_results['uncertainty'],
                'fold': fold,
            })
            if 'cohort_num' not in test_result_df.columns and 'cohert_num' in test_df.columns:
                test_result_df['cohort_num'] = test_df['cohert_num'].values
            elif 'cohort_num' in test_df.columns:
                test_result_df['cohort_num'] = test_df['cohort_num'].values
            
            test_all_fold_results.append(test_result_df)
            
            targs = np.array(test_results['label'])
            probs = np.array(test_results['score'])
            preds = np.array(test_results['pred'])
            
            try:
                fold_auc = auroc(targs, probs)
                fold_f1, fold_bacc = evaluate_metrics(targs, preds)
                print(f"  Fold {fold} Test - AUC: {fold_auc:.4f}, F1: {fold_f1:.4f}, BAcc: {fold_bacc:.4f}")
            except Exception as e:
                print(f"  Fold {fold} metrics error: {e}")
            
            del model
            clear_memory()
        
        if test_all_fold_results:
            test_all_df = pd.concat(test_all_fold_results, ignore_index=True)
            test_all_df.to_csv(output_dir / f'{args.dataset}_edl_test_all_folds.csv', index=False)
            
            test_ensemble = test_all_df.groupby(['patient_id', 'image_id']).agg({
                'prediction_score': 'mean',
                'predicted_class': lambda x: (x.mean() >= 0.5).astype(int),
                'evidence_0': 'mean',
                'evidence_1': 'mean',
                'alpha_0': 'mean',
                'alpha_1': 'mean',
                'uncertainty': 'mean',
                label_col: 'first',
                'cohort_num': 'first',
                'split': 'first',
            }).reset_index()
            test_ensemble['fold'] = 'ensemble'
            
            test_ensemble.to_csv(output_dir / f'{args.dataset}_edl_test_ensemble.csv', index=False)
            
            targs_ens = test_ensemble[label_col].values
            probs_ens = test_ensemble['prediction_score'].values.astype(float)
            preds_ens = test_ensemble['predicted_class'].values.astype(int)
            
            try:
                ens_auc = auroc(targs_ens, probs_ens)
                ens_f1, ens_bacc = evaluate_metrics(targs_ens.astype(int), preds_ens)
                print(f"\nEnsemble Test - AUC: {ens_auc:.4f}, F1: {ens_f1:.4f}, BAcc: {ens_bacc:.4f}")
            except Exception as e:
                print(f"Ensemble metrics error: {e}")
    
    # ========== Combined output ==========
    if dev_results_df is not None and test_ensemble is not None:
        combined_df = pd.concat([dev_results_df, test_ensemble], ignore_index=True)
    elif dev_results_df is not None:
        combined_df = dev_results_df
    else:
        combined_df = None
    
    if combined_df is not None:
        combined_df.to_csv(output_dir / f'{args.dataset}_edl_all_predictions.csv', index=False)
        print(f"\nCombined predictions saved: {len(combined_df)} samples -> {output_dir / f'{args.dataset}_edl_all_predictions.csv'}")
    
    print("\n===== EDL Test Complete =====")
    return output_dir


def main():
    args = config()
    
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    print(f"[INFO] Using GPU {args.gpu_id}")
    
    seed_all(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    args.apex = True if args.apex == "y" else False
    
    run_edl_test(args, device)


if __name__ == "__main__":
    main()
