import pathlib
if not hasattr(pathlib, 'WindowsPath'):
    pathlib.WindowsPath = pathlib.PosixPath
import numpy as np
import pandas as pd
from pathlib import Path
import os 

import torch

from Datasets.dataset_utils import MIL_dataloader
from MIL import build_model 
from MIL.MIL_experiment import valid_fn, build_sample_results_df, resolve_checkpoint_path
from utils.generic_utils import seed_all, print_network
from utils.plot_utils import plot_confusion_matrix, ROC_curves
from utils.data_split_utils import stratified_train_val_split

def run_eval(checkpoint_path, args, device):

    if args.feature_extraction == 'online': 
        if 'efficientnetv2' in args.arch:
            args.model_base_name = 'efficientv2_s'
        elif 'efficientnet_b5_ns' in args.arch:
            args.model_base_name = 'efficientnetb5'
        else:
            args.model_base_name = args.arch
        
    args.n_class = 1 # Binary classification task

    # Define class labels 
    if args.label.lower() == 'mass':
        class0 = 'not_mass'
        class1 = 'mass'
    elif args.label.lower() == 'suspicious_calcification':
        class0 = 'not_calcification'
        class1 = 'calcification'   
    elif args.label.lower()=='cancer':
        class0='not_cancer'
        class1='cancer'
    label_dict = {class0: 0, class1: 1}

    args.resume= Path(args.resume)
    
    ############################ Data Setup ############################
    args.data_dir = Path(args.data_dir)
    
    args.df = pd.read_csv(args.data_dir / args.csv_file)
    args.df = args.df.fillna(0)
    
    print(f"df shape: {args.df.shape}")
    print(args.df.columns)

    if args.eval_set == 'val': 
        dev_df = args.df[args.df['split'] == "training"].reset_index(drop=True)
        _, test_df = stratified_train_val_split(dev_df, 0.2, args = args)
    
    elif args.eval_set == 'test': # Use official test split
        test_df = args.df[args.df['split'] == "test"].reset_index(drop=True)
    elif args.eval_set == 'all':
        # Ignore split values and evaluate all rows from input CSV
        test_df = args.df.reset_index(drop=True)
    else:
        raise ValueError(f"Unsupported eval_set: {args.eval_set}")

    # Create DataLoader for MIL evaluation on test set
    test_loader = MIL_dataloader(test_df ,'test', args)

    # Build model
    model = build_model(args)
    model.is_training = False # Set model mode for evaluation
    
    model.to(device)
    print_network(model)

    # Load best model checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu',weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=False)
    
    # Set the model to evaluation mode
    model.eval()

    test_targs, test_preds, test_probs, test_results, sample_results = valid_fn(
        test_loader, model, criterion = torch.nn.BCEWithLogitsLoss(reduction='mean'), args = args, device = device, split = 'test'
    )
    
    # Print overall test loss
    print(f"\nTest Loss: {test_results['loss']:.4f}")     

    # Print metrics per scale
    for s in args.scales:
        print(f"Scale: {s} --> Test F1-Score: {test_results[s]['f1']:.4f} | Test Bacc: {test_results[s]['bacc']:.4f} | Test ROC-AUC: {test_results[s]['auc_roc']:.4f}")            

    # Print aggregated metrics across scales
    print(f"Aggregated Results --> Test F1-Score: {test_results['aggregated']['f1']:.4f} | Test Bacc: {test_results['aggregated']['bacc']:.4f} | Test ROC-AUC: {test_results['aggregated']['auc_roc']:.4f}")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    run_name = checkpoint_path.parent.name 
    
    print(f"Saving ROC curve and Confusion Matrix to {output_path} ...")
    
    try:

        ROC_curves(test_targs, test_probs, run_name, output_path)
        
  
        if 'aggregated' in test_results and 'cf_matrix' in test_results['aggregated']:
            cm = test_results['aggregated']['cf_matrix']
        elif 'cf_matrix' in test_results:
            cm = test_results['cf_matrix']
        else:
            cm = None
            
        if cm is not None:
            plot_confusion_matrix(cm, label_dict, run_name, output_path)
            
    except Exception as e:
        print(f"Warning: Failed to plot curves. Error: {e}")
    final_results_data = {}
    
    # Append metrics for all scales
    for s in args.scales:
        final_results_data[f'{args.eval_set}_bacc_{s}'] = test_results[s]['bacc']
        final_results_data[f'{args.eval_set}_f1_{s}'] = test_results[s]['f1']
        final_results_data[f'{args.eval_set}_auc_roc_{s}'] = test_results[s]['auc_roc']
        
    # Append metrics for aggregated results
    final_results_data[f'{args.eval_set}_bacc_aggregated'] = test_results['aggregated']['bacc']
    final_results_data[f'{args.eval_set}_f1_aggregated'] = test_results['aggregated']['f1']
    final_results_data[f'{args.eval_set}_auc_roc_aggregated'] = test_results['aggregated']['auc_roc']
        
    # Create the final DataFrame
    df_final_results = pd.DataFrame(final_results_data, index=[0])

    sample_results_df = build_sample_results_df(test_df, sample_results, args.eval_set, args)

    return df_final_results, sample_results_df


def Eval(args, device):

    all_results = []  # Store results from all runs

    for run_idx in range(args.n_runs):
        seed_all(args.seed)
        
        print(f'\nRunning eval for model run nº{run_idx + args.start_run}....')
        
        checkpoint_path = resolve_checkpoint_path(args.resume, args.start_run + run_idx)
        if checkpoint_path is None:
            raise FileNotFoundError(
                f"Could not find checkpoint for index {args.start_run + run_idx} under {args.resume}. "
                "Expected run_i/best_model.pth, fold_i/best_model.pth, or a direct checkpoint file."
            )
        
        # Run the evaluation and get results as DataFrame
        run_results_df, sample_results_df = run_eval(checkpoint_path, args, device) 
        
        # Add column to track the run
        run_results_df["runs"] = args.start_run + run_idx
        sample_results_df["runs"] = args.start_run + run_idx
        
        all_results.append(run_results_df)
        sample_output_path = os.path.join(args.output_dir, f'{args.dataset}_{args.eval_set}_predictions_run_{args.start_run + run_idx}.csv')
        sample_results_df.to_csv(sample_output_path, index=False)
        print(f"Sample-level predictions saved to {sample_output_path}")
    
    

    # Combine all runs into a single DataFrame
    combined_df = pd.concat(all_results, ignore_index=True)
    if args.n_runs > 1:  
        # Calculate mean and std for specific columns
        mean_std = combined_df.drop('runs', axis=1).agg(['mean', 'std']).reset_index(drop=True)
        mean_std['runs'] = ['mean', 'std']

        # Append mean and std to the original DataFrame
        combined_df = pd.concat([combined_df, mean_std]).reset_index(drop=True)

    print(combined_df)
    os.makedirs(args.output_dir, exist_ok=True)
    output_path = os.path.join(args.output_dir, f'{args.dataset}_eval_summary.csv')
    combined_df.to_csv(output_path, index=False)
    print(f"Results saved to {output_path}")
    
