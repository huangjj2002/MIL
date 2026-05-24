#external imports 
import gc
import time
from pathlib import Path
import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# internal imports 
from Datasets.dataset_utils import MIL_dataloader
from MIL import build_model 

from utils.metrics import auroc, evaluate_metrics
from utils.generic_utils import seed_all, AverageMeter, timeSince, print_network, clear_memory 
from utils.training_setup_utils import initialize_training_setup, Training_Stage_Config
from utils.plot_utils import plot_loss_and_acc_curves, plot_lrs_scheduler, plot_confusion_matrix, ROC_curves
from utils.data_split_utils import (
    generator_cross_val_folds,
    split_df_by_cohorts,
    stratified_train_val_split,
)

def resolve_checkpoint_path(resume_path, checkpoint_idx=None):
    if resume_path is None:
        return None

    resume_path = Path(resume_path)

    if resume_path.is_file():
        return resume_path

    if not resume_path.exists():
        return None

    candidate_paths = []

    if checkpoint_idx is not None:
        candidate_paths.extend([
            resume_path / f'fold_{checkpoint_idx}' / 'best_model.pth',
            resume_path / f'run_{checkpoint_idx}' / 'best_model.pth',
        ])

    candidate_paths.extend([
        resume_path / 'best_model.pth',
        resume_path / 'checkpoint.pth',
    ])

    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path

    return None

def get_class_names(args):
    if args.label.lower() == 'mass':
        return 'not_mass', 'mass'
    if args.label.lower() == 'suspicious_calcification':
        return 'not_calcification', 'calcification'
    if args.label.lower() == 'cancer':
        return 'not_cancer', 'cancer'
    return 'negative', 'positive'

def build_sample_results_df(df, sample_results, split_name, args):
    sample_df = df.copy().reset_index(drop=True)

    for column, values in sample_results.items():
        if column in ['patient_id', 'image_id']:
            continue
        sample_df[column] = values

    if args.mil_type == 'pyramidal_mil':
        if 'score_aggregated' in sample_df.columns:
            sample_df['score'] = sample_df['score_aggregated']
        if 'pred_aggregated' in sample_df.columns:
            sample_df['pred'] = sample_df['pred_aggregated']

    label_col = args.label.lower()

    if 'score' in sample_df.columns:
        sample_df['prediction_score'] = sample_df['score'].astype(float)
    else:
        sample_df['prediction_score'] = np.nan

    if 'pred' in sample_df.columns:
        sample_df['predicted_class'] = sample_df['pred'].astype(float).round().astype(int)
    else:
        sample_df['predicted_class'] = (sample_df['prediction_score'] >= 0.5).astype(int)

    if 'label' in sample_df.columns:
        sample_df[label_col] = sample_df['label'].astype(float).round().astype(int)
    elif label_col in sample_df.columns:
        sample_df[label_col] = sample_df[label_col].astype(float).round().astype(int)
    else:
        sample_df[label_col] = np.nan

    sample_df['split'] = split_name
    if 'cohort_num' not in sample_df.columns and 'cohert_num' in sample_df.columns:
        sample_df['cohort_num'] = sample_df['cohert_num']

    for required_col in ['patient_id', 'image_id', 'split', 'cohort_num']:
        if required_col not in sample_df.columns:
            sample_df[required_col] = None

    # Keep only required export columns; runs is appended later in evaluation flow.
    keep_cols = ['patient_id', 'image_id', 'split', 'cohort_num', label_col, 'prediction_score', 'predicted_class']
    return sample_df[keep_cols]

def predict_and_build_results(df, split_name, model, args, device):
    loader = MIL_dataloader(df, 'test', args)
    _, _, _, _, sample_results = valid_fn(
        loader,
        model,
        criterion=torch.nn.BCEWithLogitsLoss(reduction='mean'),
        args=args,
        device=device,
        split=split_name,
        return_sample_results=True,
    )
    return build_sample_results_df(df, sample_results, split_name, args)

def save_full_split_predictions(model, split_dfs, output_dir, file_stem, args, device):
    prediction_frames = []

    for split_name, split_df in split_dfs:
        if split_df is None or len(split_df) == 0:
            continue
        prediction_frames.append(predict_and_build_results(split_df, split_name, model, args, device))

    if prediction_frames:
        predictions_df = pd.concat(prediction_frames, ignore_index=True)
        if hasattr(args, 'cur_fold'):
            predictions_df['fold'] = int(args.cur_fold)
        if hasattr(args, 'checkpoint_index'):
            predictions_df['runs'] = int(args.checkpoint_index)
        output_path = Path(output_dir) / f'{file_stem}.csv'
        predictions_df.to_csv(output_path, index=False)
        print(f"Sample-level predictions saved to {output_path}")

def save_mil_loss_curve(train_results, val_results, output_path):
    """Save epoch-wise MIL loss history in the same shape as EDL curves."""
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
    curve_df.to_csv(output_path / 'mil_loss_curve.csv', index=False)

    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.plot(curve_df['epoch'], curve_df['train_loss'], marker='o', label='Train Loss')
        ax.plot(curve_df['epoch'], curve_df['val_loss'], marker='o', label='Val Loss')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('MIL Loss Curve')
        ax.grid(True, alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(output_path / 'mil_loss_curve.png', dpi=200)
        plt.close(fig)
    except Exception as exc:
        print(f"[MIL] Warning: failed to save loss curve plot: {exc}")

def get_primary_stats(stats, args):
    """Return the aggregated branch metrics when using a multi-scale model."""
    if args.multi_scale_model is not None:
        return stats['aggregated']
    return stats

def do_experiments(args, device):
        
    args.n_class = 1 # Binary classification setup (single output neuron)
        
    # Define class labels based on selected task
    if args.label.lower() == 'mass':
        class0 = 'not_mass'
        class1 = 'mass'
    elif args.label.lower() == 'suspicious_calcification':
        class0 = 'not_calcification'
        class1 = 'calcification'   
    elif args.label.lower() == 'cancer':
        class0 = 'not_cancer'
        class1 = 'cancer'  

    label_dict = {class0: 0, class1: 1}

    ############################ Data Setup ############################
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

    # reduce dataset size for debugging/experiments if desired
    if args.data_frac < 1.0:
        dev_df = dev_df.sample(frac=args.data_frac, random_state=1, ignore_index=True) 

    if args.eval_scheme == 'kfold_cv+test' and args.n_folds == 0:
        print("[Auto-Config] n_folds=0 detected. Using train cohorts for training and test cohorts for validation.")

    # repeated k runs using fixed data splits 
    if args.eval_scheme == 'kruns_train+val+test': 
        use_test_as_validation = args.n_folds == 0

        # split development set into training and validation sets
        if use_test_as_validation:
            train_df = dev_df.reset_index(drop=True)
            val_df = test_df.reset_index(drop=True)
        else:
            train_df, val_df = stratified_train_val_split(dev_df, args.val_split, args=args)

        # initialize results dictionary based on model type
        if args.multi_scale_model is not None: 

            # track results for each scale if required by model configuration
            if (args.type_scale_aggregator in ['concatenation', 'gated-attention'] and args.deep_supervision) or args.type_scale_aggregator in ['max_p', 'mean_p']:  
                all_val_results = {scale: {'f1': [], 'bacc': [], 'auc_roc': []} for scale in args.scales}
                all_test_results = {scale: {'f1': [], 'bacc': [], 'auc_roc': []} for scale in args.scales}

            else: 
                all_val_results = {}
                all_test_results = {}

            # track aggregated results
            all_val_results['aggregated'] = {'f1': [], 'bacc': [], 'auc_roc': []}
            all_test_results['aggregated'] = {'f1': [], 'bacc': [], 'auc_roc': []}
        
        else: 
            # track results for non multi-scale models 
            all_val_results = {'f1': [], 'bacc': [], 'auc_roc': []}
            all_test_results = {'f1': [], 'bacc': [], 'auc_roc': []} 
            
        # set test data loader
        test_loader = MIL_dataloader(test_df ,'test', args)

        # perform multiple runs (kruns) of training and testing
        for idx_run in range(args.n_runs):
            print(f'\n================== run nº: {idx_run} ======================')
            args.cur_fold = idx_run  
            args.checkpoint_index = args.start_run + idx_run

            # set seed for reproducibility
            seed_all(args.seed+args.start_run+idx_run)

            # create directory for saving results for this run
            path_results_run = args.output_path / f'run_{args.start_run+idx_run}'
            Path(path_results_run).mkdir(parents=True, exist_ok=True)

            # train and validate model
            val_results, best_checkpoint_path = k_experiment(
                train_df,
                val_df,
                output_path=path_results_run,
                args=args,
                device=device,
                valid_split_name='test' if use_test_as_validation else 'val',
            )

            # load the best model checkpoint
            checkpoint = torch.load(best_checkpoint_path, map_location='cpu',weights_only=False)
            fold_model = build_model(args)
            fold_model.load_state_dict(checkpoint['model'])
            fold_model.to(device)
            fold_model.eval()

            # evaluate model on test set
            test_targs, test_preds, test_probs, test_results, _ = valid_fn(
                test_loader, fold_model, criterion = torch.nn.BCEWithLogitsLoss(reduction='mean'), args = args, device = device, split = 'test')

            save_full_split_predictions(
                fold_model,
                [('train', train_df), ('test', test_df)] if use_test_as_validation else [('train', train_df), ('val', val_df), ('test', test_df)],
                path_results_run,
                f'{args.dataset}_all_predictions_run_{args.start_run + idx_run}',
                args,
                device,
            )

            # free GPU memory
            del fold_model; clear_memory()

            # report and store test results
            if args.multi_scale_model is not None: 
                print(f"\nTest Loss: {test_results['loss']:.4f}") 

                # Print results for individual scales if applicable
                if (args.type_scale_aggregator in ['concatenation', 'gated-attention'] and args.deep_supervision) or args.type_scale_aggregator in ['max_p', 'mean_p']:  
                    for s in args.scales:
                        print(f"Scale: {s} --> Test F1-Score: {test_results[s]['f1']:.4f} | Test Bacc: {test_results[s]['bacc']:.4f} | Test ROC-AUC: {test_results[s]['auc_roc']:.4f}")            

                # Print aggregated results
                print(f"Aggregated Results --> Test F1-Score: {test_results['aggregated']['f1']:.4f} | Test Bacc: {test_results['aggregated']['bacc']:.4f} | Test ROC-AUC: {test_results['aggregated']['auc_roc']:.4f}")

                # Generate confusion matrix and ROC curves
                plot_confusion_matrix(test_results['aggregated']['cf_matrix'], label_dict, '', path_results_run)
                ROC_curves(test_targs, test_probs, '', path_results_run)

                # Append results per scale
                if (args.type_scale_aggregator in ['concatenation', 'gated-attention'] and args.deep_supervision) or args.type_scale_aggregator in ['max_p', 'mean_p']:  
                
                    for s in args.scales:
                        all_val_results[s]['f1'].append(val_results[s]['f1'])
                        all_val_results[s]['bacc'].append(val_results[s]['bacc'])
                        all_val_results[s]['auc_roc'].append(val_results[s]['auc_roc'])
        
                        all_test_results[s]['f1'].append(test_results[s]['f1'])
                        all_test_results[s]['bacc'].append(test_results[s]['bacc'])
                        all_test_results[s]['auc_roc'].append(test_results[s]['auc_roc'])

                # Append aggregated results
                all_test_results['aggregated']['f1'].append(test_results['aggregated']['f1'])
                all_test_results['aggregated']['bacc'].append(test_results['aggregated']['bacc'])
                all_test_results['aggregated']['auc_roc'].append(test_results['aggregated']['auc_roc'])
                
                all_val_results['aggregated']['f1'].append(val_results['aggregated']['f1'])
                all_val_results['aggregated']['bacc'].append(val_results['aggregated']['bacc'])
                all_val_results['aggregated']['auc_roc'].append(val_results['aggregated']['auc_roc'])

            else: 
                # Log and store results for non-multiscale models
                
                print(f"Test F1-Score: {test_results['f1']:.4f} | Test Bacc: {test_results['bacc']:.4f} | Test ROC-AUC: {test_results['auc_roc']:.4f}")           

                plot_confusion_matrix(test_results['cf_matrix'], label_dict, '', path_results_run)
                ROC_curves(test_targs, test_probs, '', path_results_run)
                
                # Append Results 
                all_val_results['f1'].append(val_results['f1'])
                all_val_results['bacc'].append(val_results['bacc'])
                all_val_results['auc_roc'].append(val_results['auc_roc'])
    
                all_test_results['f1'].append(test_results['f1'])
                all_test_results['bacc'].append(test_results['bacc'])
                all_test_results['auc_roc'].append(test_results['auc_roc'])

        # Collect all results into structured format
        val_results_data = {'runs': np.arange(args.n_runs)}
        test_results_data = {'runs': np.arange(args.n_runs)}

        if args.multi_scale_model is not None: 

            if (args.type_scale_aggregator in ['concatenation', 'gated-attention'] and args.deep_supervision) or args.type_scale_aggregator in ['max_p', 'mean_p']: 
                
                # Append metrics for all scales
                for s in args.scales:
                    val_results_data[f'bacc_{s}'] = all_val_results[s]['bacc']
                    val_results_data[f'f1_{s}'] = all_val_results[s]['f1']
                    val_results_data[f'auc_roc_{s}'] = all_val_results[s]['auc_roc']
        
                    test_results_data[f'bacc_{s}'] = all_test_results[s]['bacc']
                    test_results_data[f'f1_{s}'] = all_test_results[s]['f1']
                    test_results_data[f'auc_roc_{s}'] = all_test_results[s]['auc_roc']
            
            # Append metrics for aggregated results
            val_results_data['bacc_aggregated'] = all_val_results['aggregated']['bacc']
            val_results_data['f1_aggregated'] = all_val_results['aggregated']['f1']
            val_results_data['auc_roc_aggregated'] = all_val_results['aggregated']['auc_roc']
    
            test_results_data['bacc_aggregated'] = all_test_results['aggregated']['bacc']
            test_results_data['f1_aggregated'] = all_test_results['aggregated']['f1']
            test_results_data['auc_roc_aggregated'] = all_test_results['aggregated']['auc_roc']
            
        else: 
            val_results_data['bacc'] = all_val_results['bacc']
            val_results_data['f1'] = all_val_results['f1']
            val_results_data['auc'] = all_val_results['auc_roc']
    
            test_results_data['bacc'] = all_test_results['bacc']
            test_results_data['f1'] = all_test_results['f1']
            test_results_data['auc'] = all_test_results['auc_roc']
            
        # Create the final DataFrame
        val_results_data = pd.DataFrame(val_results_data)
        test_results_data = pd.DataFrame(test_results_data)
        
        if args.n_runs > 1: 
            
            # Calculate mean and std for specific columns
            val_mean_std = val_results_data.drop('runs', axis=1).agg(['mean', 'std']).reset_index(drop=True)
            test_mean_std = test_results_data.drop('runs', axis=1).agg(['mean', 'std']).reset_index(drop=True)
            val_mean_std['runs'] = ['mean', 'std']
            test_mean_std['runs'] = ['mean', 'std']

            # Append mean and std to the original DataFrame
            val_results_data = pd.concat([val_results_data, val_mean_std]).reset_index(drop=True)
            test_results_data = pd.concat([test_results_data, test_mean_std]).reset_index(drop=True)

        # Combine validation and test results
        if use_test_as_validation:
            metrics_data = pd.concat([test_results_data], keys=['test'], names=['split', 'index'])
        else:
            metrics_data = pd.concat([val_results_data, test_results_data], keys=['validation', 'test'], names=['split', 'index'])
        metrics_data = metrics_data.reset_index(level='split') # Reset index to turn the keys into columns
        metrics_data.to_csv(args.output_path / 'results_summary.csv', index=False)


    elif args.eval_scheme == 'kfold_cv+test':

        use_test_as_validation = args.n_folds == 0
        if use_test_as_validation:
            split_iter = [(0, (dev_df.reset_index(drop=True), test_df.reset_index(drop=True)))]
            total_folds = 1
        else:
            split_iter = enumerate(
                generator_cross_val_folds(
                    dev_df,
                    args.n_folds,
                    args.label,
                    random_state=args.seed,
                )
            )
            total_folds = args.n_folds

        track_side_results = (
            args.multi_scale_model is not None
            and (
                (args.type_scale_aggregator in ['concatenation', 'gated-attention'] and args.deep_supervision)
                or args.type_scale_aggregator in ['max_p', 'mean_p']
            )
        )

        if args.multi_scale_model is not None:
            all_val_results = {scale: {'f1': [], 'bacc': [], 'auc_roc': []} for scale in args.scales} if track_side_results else {}
            all_test_results = {scale: {'f1': [], 'bacc': [], 'auc_roc': []} for scale in args.scales} if track_side_results else {}
            all_val_results['aggregated'] = {'f1': [], 'bacc': [], 'auc_roc': []}
            all_test_results['aggregated'] = {'f1': [], 'bacc': [], 'auc_roc': []}
        else:
            all_val_results = {'f1': [], 'bacc': [], 'auc_roc': []}
            all_test_results = {'f1': [], 'bacc': [], 'auc_roc': []}

        test_loader = MIL_dataloader(test_df, 'test', args) if len(test_df) > 0 else None
        evaluated_folds = []
        fold_summaries = []
        fold_assignments = []

        def append_results(store, stats):
            if args.multi_scale_model is not None:
                if track_side_results:
                    for scale in args.scales:
                        store[scale]['f1'].append(stats[scale]['f1'])
                        store[scale]['bacc'].append(stats[scale]['bacc'])
                        store[scale]['auc_roc'].append(stats[scale]['auc_roc'])
                primary_stats = stats['aggregated']
                store['aggregated']['f1'].append(primary_stats['f1'])
                store['aggregated']['bacc'].append(primary_stats['bacc'])
                store['aggregated']['auc_roc'].append(primary_stats['auc_roc'])
            else:
                store['f1'].append(stats['f1'])
                store['bacc'].append(stats['bacc'])
                store['auc_roc'].append(stats['auc_roc'])

        def print_split_results(split_name, stats):
            print(f"\n{split_name} Loss: {stats['loss']:.4f}")
            if args.multi_scale_model is not None:
                if track_side_results:
                    for scale in args.scales:
                        print(
                            f"Scale: {scale} --> {split_name} F1-Score: {stats[scale]['f1']:.4f} | "
                            f"{split_name} Bacc: {stats[scale]['bacc']:.4f} | "
                            f"{split_name} ROC-AUC: {stats[scale]['auc_roc']:.4f}"
                        )
                primary_stats = stats['aggregated']
                print(
                    f"Aggregated Results --> {split_name} F1-Score: {primary_stats['f1']:.4f} | "
                    f"{split_name} Bacc: {primary_stats['bacc']:.4f} | "
                    f"{split_name} ROC-AUC: {primary_stats['auc_roc']:.4f}"
                )
            else:
                print(
                    f"{split_name} F1-Score: {stats['f1']:.4f} | "
                    f"{split_name} Bacc: {stats['bacc']:.4f} | "
                    f"{split_name} ROC-AUC: {stats['auc_roc']:.4f}"
                )

        for fold, (train_df, val_df) in split_iter:
            if fold < args.start_fold:
                continue

            print(f'\n================== fold: {fold} / {total_folds} training ======================')

            args.cur_fold = fold
            args.checkpoint_index = fold
            seed_all(args.seed + fold)

            path_results_fold = args.output_path / f'fold_{fold}'
            Path(path_results_fold).mkdir(parents=True, exist_ok=True)

            valid_split_name = 'test' if use_test_as_validation else 'val'
            print(f"Train: {len(train_df)}, {valid_split_name.capitalize()}: {len(val_df)}")

            val_results, best_checkpoint_path = k_experiment(
                train_df,
                val_df,
                path_results_fold,
                args,
                device,
                valid_split_name=valid_split_name,
            )

            evaluated_folds.append(fold)
            print_split_results('Test' if use_test_as_validation else 'Val', val_results)
            append_results(all_val_results, val_results)

            primary_val_stats = get_primary_stats(val_results, args)
            fold_summaries.append({
                'fold': fold,
                'auc_roc': primary_val_stats['auc_roc'],
                'f1': primary_val_stats['f1'],
                'bacc': primary_val_stats['bacc'],
                'loss': val_results['loss'],
                'eval_source': 'test_cohorts' if use_test_as_validation else 'cross_val',
            })

            if not use_test_as_validation:
                val_assignment_df = val_df.copy().reset_index(drop=True)
                val_assignment_df['fold'] = fold
                val_assignment_df['split'] = 'val'
                fold_assignments.extend(val_assignment_df.to_dict('records'))

            checkpoint = torch.load(best_checkpoint_path, map_location='cpu', weights_only=False)

            fold_model = build_model(args)
            fold_model.load_state_dict(checkpoint['model'])
            fold_model.to(device)
            fold_model.eval()

            split_specs = (
                [('train', train_df), ('test', test_df)]
                if use_test_as_validation
                else [('train', train_df), ('val', val_df), ('test', test_df)]
            )
            save_full_split_predictions(
                fold_model,
                split_specs,
                path_results_fold,
                f'{args.dataset}_mil_predictions_fold_{fold}',
                args,
                device,
            )

            if test_loader is not None:
                test_targs, test_preds, test_probs, test_results, _ = valid_fn(
                    test_loader,
                    fold_model,
                    criterion=torch.nn.BCEWithLogitsLoss(reduction='mean'),
                    args=args,
                    device=device,
                    split='test',
                )

                print_split_results('Test', test_results)
                append_results(all_test_results, test_results)

                primary_test_stats = get_primary_stats(test_results, args)
                cf_matrix = primary_test_stats.get('cf_matrix')
                if cf_matrix is not None:
                    plot_confusion_matrix(cf_matrix, label_dict, '', path_results_fold)
                ROC_curves(test_targs, test_probs, '', path_results_fold)
            else:
                print("No test data found. Skipping test evaluation.")

            del fold_model
            clear_memory()

        val_results_data = {'folds': evaluated_folds}
        has_test_results = (
            len(all_test_results['aggregated']['bacc']) > 0
            if args.multi_scale_model is not None
            else len(all_test_results['bacc']) > 0
        )
        if has_test_results:
            test_results_data = {'folds': evaluated_folds}

        if args.multi_scale_model is not None:
            if track_side_results:
                for scale in args.scales:
                    val_results_data[f'bacc_{scale}'] = all_val_results[scale]['bacc']
                    val_results_data[f'f1_{scale}'] = all_val_results[scale]['f1']
                    val_results_data[f'auc_roc_{scale}'] = all_val_results[scale]['auc_roc']

                    if has_test_results:
                        test_results_data[f'bacc_{scale}'] = all_test_results[scale]['bacc']
                        test_results_data[f'f1_{scale}'] = all_test_results[scale]['f1']
                        test_results_data[f'auc_roc_{scale}'] = all_test_results[scale]['auc_roc']

            val_results_data['bacc_aggregated'] = all_val_results['aggregated']['bacc']
            val_results_data['f1_aggregated'] = all_val_results['aggregated']['f1']
            val_results_data['auc_roc_aggregated'] = all_val_results['aggregated']['auc_roc']

            if has_test_results:
                test_results_data['bacc_aggregated'] = all_test_results['aggregated']['bacc']
                test_results_data['f1_aggregated'] = all_test_results['aggregated']['f1']
                test_results_data['auc_roc_aggregated'] = all_test_results['aggregated']['auc_roc']
        else:
            val_results_data['bacc'] = all_val_results['bacc']
            val_results_data['f1'] = all_val_results['f1']
            val_results_data['auc'] = all_val_results['auc_roc']

            if has_test_results:
                test_results_data['bacc'] = all_test_results['bacc']
                test_results_data['f1'] = all_test_results['f1']
                test_results_data['auc'] = all_test_results['auc_roc']

        val_results_data = pd.DataFrame(val_results_data)
        if has_test_results:
            test_results_data = pd.DataFrame(test_results_data)

        if len(evaluated_folds) > 1:
            val_mean_std = val_results_data.drop('folds', axis=1).agg(['mean', 'std']).reset_index(drop=True)
            val_mean_std['folds'] = ['mean', 'std']
            val_results_data = pd.concat([val_results_data, val_mean_std]).reset_index(drop=True)

            if has_test_results:
                test_mean_std = test_results_data.drop('folds', axis=1).agg(['mean', 'std']).reset_index(drop=True)
                test_mean_std['folds'] = ['mean', 'std']
                test_results_data = pd.concat([test_results_data, test_mean_std]).reset_index(drop=True)

        if has_test_results:
            if use_test_as_validation:
                metrics_data = pd.concat([test_results_data], keys=['test'], names=['split', 'index'])
            else:
                metrics_data = pd.concat([val_results_data, test_results_data], keys=['validation', 'test'], names=['split', 'index'])
        else:
            metrics_data = pd.concat([val_results_data], keys=['validation'], names=['split', 'index'])

        metrics_data = metrics_data.reset_index(level='split')
        metrics_data.to_csv(args.output_path / 'results_summary.csv', index=False)

        summary_df = pd.DataFrame(fold_summaries)
        if len(summary_df) > 1:
            metric_cols = [col for col in summary_df.columns if col not in ['fold', 'eval_source']]
            mean_std = summary_df[metric_cols].agg(['mean', 'std']).reset_index(drop=True)
            mean_std['fold'] = ['mean', 'std']
            mean_std['eval_source'] = 'summary'
            summary_df = pd.concat([summary_df, mean_std], ignore_index=True)
        summary_df.to_csv(args.output_path / 'mil_results_summary.csv', index=False)

        if fold_assignments and not use_test_as_validation:
            fold_df = pd.DataFrame(fold_assignments)
            fold_df.to_csv(args.output_path / f'{args.dataset}_mil_val_fold_assignments.csv', index=False)


def k_experiment(train_df, val_df, output_path, args, device, valid_split_name='val'): 
    """
    Executes a single train/validation experiment.
    
    Args:
        train_df (DataFrame): Training data.
        val_df (DataFrame): Validation data.
        output_path (Path): Directory to save results and checkpoints.
        args (Namespace): Configuration and hyperparameters.
        device (torch.device): Device to run model on.

    Returns:
        Tuple:
            - best_val_stats (dict): Best evaluation metrics on validation set.
            - best_model_path (str): Path to the best model checkpoint.
    """
        
    if args.running_interactive:
        # test on small subsets of data on interactive mode
        train_df = train_df.sample(1000)
        val_df = val_df.sample(n=1000)

    # Initialize data loaders
    train_loader = MIL_dataloader(train_df, 'train', args)
    valid_loader = MIL_dataloader(val_df, valid_split_name, args)
    print(f'train_loader: {len(train_loader)}, valid_loader: {len(valid_loader)}')

    # Build and load model
    model = build_model(args)
    checkpoint_path = resolve_checkpoint_path(args.resume, getattr(args, 'checkpoint_index', None))
    if checkpoint_path is not None:
        print(f"Loading checkpoint for fine-tuning from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location='cpu',weights_only=False)
        
        msg = model.load_state_dict(checkpoint['model'], strict=False)
        print(f"Checkpoint loaded. Message: {msg}")
    print("Model is loaded")

    # Setup training stage manager if online feature extraction is enabled
    training_stage_manager = Training_Stage_Config(model=model, training_mode=args.training_mode, warmup_epochs=args.warmup_stage_epochs) if args.feature_extraction == 'online' else None 

    model = model.to(device)
    print_network(model)

    optimizer, scheduler, scaler, train_criterion, eval_criterion = initialize_training_setup(train_loader, model, device, args)

    best_val_stats, best_model = train_loop(
        train_loader,
        valid_loader,
        model,
        training_stage_manager,
        train_criterion,
        eval_criterion,
        optimizer,
        scheduler,
        scaler,
        output_path,
        args,
        device,
        valid_split_name=valid_split_name,
    )
    
    return best_val_stats, best_model
    

def train_loop(train_loader, valid_loader, model, training_stage_manager, train_criterion, eval_criterion, optimizer, scheduler, scaler, output_path, args, device, valid_split_name='val'):

    best_aucroc = -float('inf')
    best_val_loss = float('inf')
    best_epoch = 0
    best_val_stats = None
    best_checkpoint_path = output_path / 'best_model.pth'
    epochs_without_improvement = 0
    early_stop_patience = max(0, int(getattr(args, 'early_stop_patience', 0)))
    early_stop_min_delta = max(0.0, float(getattr(args, 'early_stop_min_delta', 0.0)))

    # Dictionaries to keep track of training and validation metrics per epoch
    train_results = {'loss': [], 'f1': [], 'bacc': [], 'auc_roc':[], 'lr':[]}
    val_results = {'loss': [], 'f1': [], 'bacc': [], 'auc_roc':[]}
        
    for epoch in range(args.epochs):

        print(f"\n-------- Epoch {epoch + 1}/{args.epochs} --------")
        
        start_time = time.time()

        if training_stage_manager is not None:
            training_stage_manager(model, optimizer, epoch, optimizer.param_groups[0]['lr'])

        # training for one epoch
        train_stats = train_fn(train_loader, model, train_criterion, optimizer, epoch, args, scheduler, scaler, device)

        # validation after the epoch
        val_output = valid_fn(valid_loader, model, eval_criterion, args, device, split=valid_split_name, epoch=epoch)
        if isinstance(val_output, tuple):
            _, _, _, val_stats, _ = val_output
        else:
            val_stats = val_output
    
        elapsed = time.time() - start_time

        valid_display_name = 'Test' if valid_split_name == 'test' else 'Val'

        primary_train_stats = get_primary_stats(train_stats, args)
        primary_val_stats = get_primary_stats(val_stats, args)

        # If using multi-scale model, report scale-specific and aggregated results
        if args.multi_scale_model is not None: 
            print(f"\nTrain Loss: {train_stats['loss']:.4f}")

            if (args.type_scale_aggregator in ['concatenation', 'gated-attention'] and args.deep_supervision) or args.type_scale_aggregator in ['max_p', 'mean_p']: 
                for s in args.scales:
                    print(f"Scale: {s} --> Train F1-Score: {train_stats[s]['f1']:.4f} | Train Bacc: {train_stats[s]['bacc']:.4f} | Train ROC-AUC: {train_stats[s]['auc_roc']:.4f}")
                
            print(f"Aggregated Results --> Train F1-Score: {primary_train_stats['f1']:.4f} | Train Bacc: {primary_train_stats['bacc']:.4f} | Train ROC-AUC: {primary_train_stats['auc_roc']:.4f}")
        
            print(f"\n{valid_display_name} Loss: {val_stats['loss']:.4f}") 

            if (args.type_scale_aggregator in ['concatenation', 'gated-attention'] and args.deep_supervision) or args.type_scale_aggregator in ['max_p', 'mean_p']: 
                for s in args.scales:
                    print(f"Scale: {s} --> {valid_display_name} F1-Score: {val_stats[s]['f1']:.4f} | {valid_display_name} Bacc: {val_stats[s]['bacc']:.4f} | {valid_display_name} ROC-AUC: {val_stats[s]['auc_roc']:.4f}")            
            
            print(f"Aggregated Results --> {valid_display_name} F1-Score: {primary_val_stats['f1']:.4f} | {valid_display_name} Bacc: {primary_val_stats['bacc']:.4f} | {valid_display_name} ROC-AUC: {primary_val_stats['auc_roc']:.4f}")

        else: 
            # Single scale MIL models
            print(f"\nTrain Loss: {train_stats['loss']:.4f} | Train F1-Score: {primary_train_stats['f1']:.4f} | Train Bacc: {primary_train_stats['bacc']:.4f} | Train ROC-AUC: {primary_train_stats['auc_roc']:.4f}")
            
            print(f"\n{valid_display_name} Loss: {val_stats['loss']:.4f} | {valid_display_name} F1-Score: {primary_val_stats['f1']:.4f} | {valid_display_name} Bacc: {primary_val_stats['bacc']:.4f} | {valid_display_name} ROC-AUC: {primary_val_stats['auc_roc']:.4f}\n")

        train_results['loss'].append(train_stats['loss'])
        train_results['f1'].append(primary_train_stats['f1'])
        train_results['bacc'].append(primary_train_stats['bacc'])
        train_results['auc_roc'].append(primary_train_stats['auc_roc'])
        train_results['lr'].append(train_stats['lr'])
            
        val_results['loss'].append(val_stats['loss'])
        val_results['f1'].append(primary_val_stats['f1'])
        val_results['bacc'].append(primary_val_stats['bacc'])
        val_results['auc_roc'].append(primary_val_stats['auc_roc'])
        save_mil_loss_curve(train_results, val_results, output_path)

        val_auc = primary_val_stats['auc_roc']
        val_auc_is_valid = np.isfinite(val_auc)
        should_save = (
            (val_auc_is_valid and val_auc > best_aucroc + early_stop_min_delta)
            or (not val_auc_is_valid and val_stats['loss'] < best_val_loss - early_stop_min_delta)
            or best_val_stats is None
        )

        if should_save:
            epochs_without_improvement = 0
            if val_auc_is_valid:
                best_aucroc = val_auc
            best_val_loss = val_stats['loss']
            best_val_stats = val_stats 
            best_epoch = epoch + 1

            if val_auc_is_valid:
                print(f'\nEpoch {epoch + 1} - Save aucroc: {best_aucroc:.4f} Model')
            else:
                print(f"\nEpoch {epoch + 1} - {valid_display_name} AUC is undefined; save best validation loss: {best_val_loss:.4f}")
                
            torch.save(
                { 
                    'model': model.state_dict(),
                    'epoch': epoch,
                    'auroc': primary_val_stats['auc_roc'],
                    'f1': primary_val_stats['f1'], 
                    'bacc': primary_val_stats['bacc'],
                    'dir_path': output_path
                }, best_checkpoint_path
            )
        else:
            epochs_without_improvement += 1

        if np.isfinite(best_aucroc):
            print(f'\nbest AUC-ROC Score at epoch {best_epoch}: {best_aucroc:.4f}')
        else:
            print(f'\nbest validation loss at epoch {best_epoch}: {best_val_loss:.4f} (AUC undefined)')

        if early_stop_patience > 0:
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

    # Plot learning rate scheduler curve and training/validation metrics curves
    plot_lrs_scheduler(train_results['lr'], output_path)
    try:
        plot_loss_and_acc_curves(train_results, val_results, 'auc_roc', output_path)
    except Exception as exc:
        print(f"[MIL] Warning: failed to save legacy loss curves: {exc}")

    # Clear GPU memory cache and garbage collect
    torch.cuda.empty_cache()
    gc.collect()
    
    return best_val_stats, best_checkpoint_path

def train_fn(train_loader, model, criterion, optimizer, epoch, args, scheduler, scaler, device):
    """
    Training loop for one epoch.
    """
        
    model.train() # Set model to training mode
    model.is_training = True 
    
    losses = AverageMeter()

    progress_iter = tqdm(enumerate(train_loader), 
                         desc=f"[{epoch + 1:03d}/{args.epochs:03d} epoch train]",
                         total=len(train_loader)
                        )
    
    targs = []

    if args.mil_type == 'pyramidal_mil':
        preds = {}
        probs = {}

        if (args.type_scale_aggregator in ['concatenation', 'gated-attention'] and args.deep_supervision) or args.type_scale_aggregator in ['max_p', 'mean_p']: 
            
            for s in args.scales: 
                preds[s] = []
                probs[s] = []
            
        preds['aggregated'] = []
        probs['aggregated'] = []

    else:
        preds = []
        probs = []
        
    start = time.time()

    # Iterate over batches
    for step, data in progress_iter:

        # Send data to device
        if isinstance(data['x'], dict): 
            inputs = {scale: tensor.to(device) for scale, tensor in data['x'].items()}
            batch_size = inputs[args.scales[0]].size(0)
        elif isinstance(data['x'], list): 
            inputs = [tensor.to(device) for tensor in data['x']]
            batch_size = inputs[0].size(0)
        else: 
            inputs = data['x'].to(device) 
            batch_size = inputs.size(0)

        labels = data['y'].float().to(device)
        
        # Wrap forward pass with autocast
        with torch.cuda.amp.autocast(enabled=args.apex):

            if args.mil_type == 'pyramidal_mil':
                if args.type_scale_aggregator in ['concatenation', 'gated-attention']:  

                    # Model returns logits for the scale-specific and multi-scale branches if deep supervision enabled
                    if args.deep_supervision: 
                        logits, side_logits = model(inputs) 
                    else: # Model returns logits for the multi-scale branch 
                        logits= model(inputs) 
                    
                    logits = logits.nan_to_num()
                    
                    loss = criterion(logits.view(-1, 1), labels.view(-1, 1))
                    
                elif args.type_scale_aggregator in ['max_p', 'mean_p']: 
                    side_logits = model(inputs)
                    
                    loss = 0.0 
            
            else: 
                # single-scale mil models 
                logits = model(inputs)

                loss = criterion(logits.view(-1, 1), labels.view(-1, 1))

        if (args.type_scale_aggregator in ['concatenation', 'gated-attention'] and args.deep_supervision) or args.type_scale_aggregator in ['max_p', 'mean_p']: 
            
            for idx, side_logit in enumerate(side_logits): 
                side_logit = side_logit.nan_to_num()
                    
                loss += criterion(side_logit.view(-1, 1), labels.view(-1, 1))
        
        losses.update(loss.item(), batch_size)

        # Backprop w/ gradient scaling
        scaler.scale(loss).backward()

        # Gradient clipping if enabled
        if args.clip_grad > 0.0:
            # Unscales the gradients of optimizer's assigned params in-place
            scaler.unscale_(optimizer)

            # Since the gradients of optimizer's assigned params are unscaled, clips as usual
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)

        # Step optimizer and update scaler
        scaler.step(optimizer)
        scaler.update() 
        
        optimizer.zero_grad() # Clear gradients for next step

        # Step learning rate scheduler per batch
        scheduler.step()

        targs.append(labels.cpu().numpy()) 

        
        if args.mil_type == 'pyramidal_mil': # store predictions and probabilities for multi-scale MIL models

            # Store predictions and probabilities depending on multi-scale aggregator 
            if (args.type_scale_aggregator in ['concatenation', 'gated-attention'] and args.deep_supervision) or args.type_scale_aggregator in ['max_p', 'mean_p']: 

                # Store scale-specific predictions and probabilities
                for idx, s in enumerate(args.scales): 
                    y_probs = side_logits[idx].sigmoid().detach()
                    y_probs = y_probs.nan_to_num()
                    
                    y_preds = (y_probs > 0.5).float()
                    
                    probs[s].append(y_probs.cpu().numpy())
                    preds[s].append(y_preds.cpu().numpy())

            # Store multi-scale aggregated predictions and probabilities depending on multi-scale aggregator 
            if args.type_scale_aggregator in ['concatenation', 'gated-attention']:
                y_probs = logits.sigmoid().detach()
                y_probs = y_probs.nan_to_num()
                
                y_preds = (y_probs > 0.5).float()
    
                probs['aggregated'].append(y_probs.cpu().numpy())
                preds['aggregated'].append(y_preds.cpu().numpy())
    
            elif args.type_scale_aggregator in ['max_p', 'mean_p']:
                # mean or max pooling over side logits 
                y_probs_aggregated = torch.zeros_like(y_probs)
    
                for idx, s in enumerate(args.scales): 
                    y_probs = side_logits[idx].sigmoid().detach()
                    y_probs = y_probs.nan_to_num()  # Ensure no NaNs
    
                    if args.type_scale_aggregator == 'mean_p': 
                        y_probs_aggregated += y_probs/len(args.scales)
                        
                    if args.type_scale_aggregator == 'max_p': 
                        y_probs_aggregated = torch.maximum(y_probs_aggregated, y_probs)
                        
                y_preds_aggregated = (y_probs_aggregated > 0.5).float()
                    
                probs['aggregated'].append(y_probs_aggregated.cpu().numpy())
                preds['aggregated'].append(y_preds_aggregated.cpu().numpy()) 
        
        else: # store predictions and probabilities for single-scale mil models 
            y_probs = logits.sigmoid().detach()
            y_preds = (y_probs > 0.5).float()
    
            probs.append(y_probs.cpu().numpy())
            preds.append(y_preds.cpu().numpy())

            
        progress_iter.set_postfix(
            {
                "lr": [optimizer.param_groups[0]['lr']],
                "loss": f"{losses.avg:.4f}",
                #"loss": f"{train_loss:.4f}",
                "CUDA-Mem": f"{torch.cuda.memory_usage(device)}%",
                "CUDA-Util": f"{torch.cuda.utilization(device)}%",
            }
        )

    train_stats = {
        'loss': losses.avg, 
        'lr': optimizer.param_groups[0]['lr']
    }

    targs = np.concatenate(targs)

    # Compute and store metrics depending on MIL model type
    if args.mil_type == 'pyramidal_mil':

        # Metrics per scale if applicable
        if (args.type_scale_aggregator in ['concatenation', 'gated-attention'] and args.deep_supervision) or args.type_scale_aggregator in ['max_p', 'mean_p']: 
            for s in args.scales:
                
                preds_s = np.concatenate(preds[s])
                probs_s = np.concatenate(probs[s])
        
                aucroc = auroc(targs, probs_s)
                f1, bacc = evaluate_metrics(targs, preds_s)
        
                train_stats[s] = {'auc_roc': aucroc, 'bacc': bacc, 'f1': f1}

        # Metrics on aggregated predictions
        preds = np.concatenate(preds['aggregated'])
        probs = np.concatenate(probs['aggregated'])

        aucroc = auroc(targs, probs)
        f1, bacc = evaluate_metrics(targs, preds) 
    
        train_stats['aggregated'] = {'auc_roc': aucroc, 'bacc': bacc, 'f1': f1}

    else: # single-scale mil models 
        preds = np.concatenate(preds)
        probs = np.concatenate(probs)
    
        aucroc = auroc(targs, probs)
        f1, bacc = evaluate_metrics(targs, preds)

        train_stats.update({'auc_roc': aucroc, 'bacc': bacc, 'f1': f1})
    
    return train_stats 

@torch.no_grad()
def valid_fn(valid_loader, model, criterion, args, device, split = 'val', epoch=1, return_sample_results=False):
    
    model.eval() # Set model to evaluation mode
    model.is_training = False 
    
    losses = AverageMeter() 

    targs = []
    sample_patient_ids = []
    sample_image_ids = []

    if args.mil_type == 'pyramidal_mil':
        preds = {}
        probs = {}

        if (args.type_scale_aggregator in ['concatenation', 'gated-attention'] and args.deep_supervision) or args.type_scale_aggregator in ['max_p', 'mean_p']:
    
            for s in args.scales: 
                preds[s] = []
                probs[s] = []
            
        preds['aggregated'] = []
        probs['aggregated'] = []

    else: 
        preds = []
        probs = []
            
    start = time.time()

    if split == 'val': 
        progress_iter = tqdm(enumerate(valid_loader), 
                             desc=f"[{epoch + 1:03d}/{args.epochs:03d} epoch valid]",
                             total=len(valid_loader)
                            )
    else:
        progress_iter = tqdm(enumerate(valid_loader), 
                             total=len(valid_loader)
                            )
    
    for step, data in progress_iter:

        # Send data to device
        if isinstance(data['x'], dict): 
            inputs = {scale: tensor.to(device, non_blocking=True) for scale, tensor in data['x'].items()}
            batch_size = inputs[args.scales[0]].size(0)
        elif isinstance(data['x'], list): 
            inputs = [tensor.to(device, non_blocking=True) for tensor in data['x']]
            batch_size = inputs[0].size(0)
        else: 
            inputs = data['x'].to(device, non_blocking=True)
            batch_size = inputs.size(0)

        labels = data['y'].float().to(device)
        sample_patient_ids.extend(data.get('patient_id', [None] * batch_size))
        sample_image_ids.extend(data.get('image_id', [None] * batch_size))
        
        # Wrap forward pass with autocast
        with torch.cuda.amp.autocast(enabled=args.apex):

            if args.mil_type == 'pyramidal_mil': 
                
                if args.type_scale_aggregator in ['concatenation', 'gated-attention']:

                    # Model returns logits for the scale-specific and multi-scale branches if deep supervision enabled
                    if args.deep_supervision: 
                        logits, side_logits = model(inputs) 
                    else: # Model returns logits only for the multi-scale branch 
                        logits = model(inputs) 
                        
                    logits = logits.nan_to_num()
                    
                    loss = criterion(logits.view(-1, 1), labels.view(-1, 1))
                    
                elif args.type_scale_aggregator in ['mean_p', 'max_p']:
                    side_logits = model(inputs)
                    
                    loss = 0.0 

            else: # single-scale mil models 
                logits = model(inputs)

                loss = criterion(logits.view(-1, 1), labels.view(-1, 1))

        if (args.type_scale_aggregator in ['concatenation', 'gated-attention'] and args.deep_supervision) or args.type_scale_aggregator in ['max_p', 'mean_p']:
            for idx, side_logit in enumerate(side_logits): 
                side_logit = side_logit.nan_to_num()
                    
                loss += criterion(side_logit.view(-1, 1), labels.view(-1, 1))
                
        losses.update(loss.item(), batch_size)

        targs.append(labels.cpu().numpy()) 

        if args.mil_type == 'pyramidal_mil':

            if (args.type_scale_aggregator in ['concatenation', 'gated-attention'] and args.deep_supervision) or args.type_scale_aggregator in ['max_p', 'mean_p']:
                
                # Store scale-specific predictions and probabilities 
                for idx, s in enumerate(args.scales): 
                    y_probs = side_logits[idx].sigmoid().detach()
                    y_probs = y_probs.nan_to_num()
                    
                    y_preds = (y_probs > 0.5).float()
                    
                    probs[s].append(y_probs.cpu().numpy())
                    preds[s].append(y_preds.cpu().numpy())

            # store multi-scale aggregated probabilities and predictions depending on multi-scale aggregator type 
            if args.type_scale_aggregator in ['concatenation', 'gated-attention']:
                y_probs = logits.sigmoid().detach()
                y_probs = y_probs.nan_to_num()
                
                y_preds = (y_probs > 0.5).float()
    
                probs['aggregated'].append(y_probs.cpu().numpy())
                preds['aggregated'].append(y_preds.cpu().numpy())
    
            elif args.type_scale_aggregator in ['mean_p', 'max_p']:
                # multi-scale aggregated results --> mean or max pooling over scale-specific probabilities and predictions 
                y_probs_aggregated = torch.zeros_like(y_probs)
    
                for idx, s in enumerate(args.scales): 
                    y_probs = side_logits[idx].sigmoid().detach()
                    y_probs = y_probs.nan_to_num()
    
                    if args.type_scale_aggregator == 'mean_p': 
                        y_probs_aggregated += y_probs/len(args.scales)
                        
                    if args.type_scale_aggregator == 'max_p': 
                        y_probs_aggregated = torch.maximum(y_probs_aggregated, y_probs)
                        
                y_preds_aggregated = (y_probs_aggregated > 0.5).float()
                    
                probs['aggregated'].append(y_probs_aggregated.cpu().numpy())
                preds['aggregated'].append(y_preds_aggregated.cpu().numpy()) 

        else: # store predictions and probabilities for single-scale mil models 

            y_probs = logits.sigmoid().detach()
            y_preds = (y_probs > 0.5).float()
    
            probs.append(y_probs.cpu().numpy())
            preds.append(y_preds.cpu().numpy())
        
        progress_iter.set_postfix(
            {
                "loss": f"{losses.avg:.4f}",
                "CUDA-Mem": f"{torch.cuda.memory_usage(device)}%",
                "CUDA-Util": f"{torch.cuda.utilization(device)}%",
            }
        )

    val_stats = {
        'loss': losses.avg, 
    }

    targs = np.concatenate(targs)

    # Preserve raw sample-wise predictions/probabilities before metric aggregation
    sample_preds_raw = preds
    sample_probs_raw = probs

    # Compute and store metrics depending on MIL model type
    if args.mil_type == 'pyramidal_mil':

        if (args.type_scale_aggregator in ['concatenation', 'gated-attention'] and args.deep_supervision) or args.type_scale_aggregator in ['max_p', 'mean_p']:

            # Metrics per scale if applicable
            for s in args.scales:
    
                preds_s = np.concatenate(preds[s])
                probs_s = np.concatenate(probs[s])
        
                aucroc = auroc(targs, probs_s)
                f1, bacc = evaluate_metrics(targs, preds_s)
                
                val_stats[s] = {'auc_roc': aucroc, 'bacc': bacc, 'f1': f1}

        # Metrics on aggregated predictions
        preds = np.concatenate(preds['aggregated'])
        probs = np.concatenate(probs['aggregated'])
    
        aucroc = auroc(targs, probs)
        f1, bacc = evaluate_metrics(targs, preds) 
        cf_matrix = confusion_matrix(targs, preds) if split == 'test' else None
            
        val_stats['aggregated'] = {'auc_roc': aucroc, 'bacc': bacc, 'f1': f1, 'cf_matrix': cf_matrix}

    else: # single-scale mil models

        preds = np.concatenate(preds)
        probs = np.concatenate(probs)
    
        aucroc = auroc(targs, probs)
        f1, bacc = evaluate_metrics(targs, preds) 
        cf_matrix = confusion_matrix(targs, preds) if split == 'test' else None
        
        val_stats.update({'auc_roc': aucroc, 'bacc': bacc, 'f1': f1, 'cf_matrix': cf_matrix})
    
    if split == 'test' or return_sample_results: 
        sample_results = {
            'patient_id': sample_patient_ids,
            'image_id': sample_image_ids,
            'label': targs.tolist() if isinstance(targs, np.ndarray) else list(targs),
        }

        if args.mil_type == 'pyramidal_mil':
            if isinstance(sample_preds_raw, dict) and isinstance(sample_probs_raw, dict):
                for key, value in sample_preds_raw.items():
                    pred_values = np.concatenate(value) if isinstance(value, list) else np.asarray(value)
                    sample_results[f'pred_{key}'] = pred_values.reshape(-1).tolist()
                for key, value in sample_probs_raw.items():
                    score_values = np.concatenate(value) if isinstance(value, list) else np.asarray(value)
                    sample_results[f'score_{key}'] = score_values.reshape(-1).tolist()
        else:
            pred_values = np.concatenate(sample_preds_raw) if isinstance(sample_preds_raw, list) else np.asarray(sample_preds_raw)
            score_values = np.concatenate(sample_probs_raw) if isinstance(sample_probs_raw, list) else np.asarray(sample_probs_raw)
            sample_results['pred'] = pred_values.reshape(-1).tolist()
            sample_results['score'] = score_values.reshape(-1).tolist()

        return targs, preds, probs, val_stats, sample_results

    return val_stats 
