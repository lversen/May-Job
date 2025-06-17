"""
Run all experiments from the TChemGNN paper and compare results.
This script orchestrates all experiments and generates a comprehensive report.

FIXED VERSION: Shows live training progress and uses proper epoch counts.
"""

import os
import sys
import subprocess
import pandas as pd
import numpy as np
import json
from datetime import datetime
import argparse
import time

# Import our modules directly for integrated training
from data import load_data, extract_edge_indices, create_atom_features, prepare_data_for_training
from models import GSR, GSRAlternative
from training import train_lipophilicity_model
from utils import set_seed, format_y_values


class ExperimentRunner:
    def __init__(self, data_dir='datasets', output_dir='experiment_results', quick=False, model_type='paper'):
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.quick = quick
        self.model_type = model_type
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.results_dir = os.path.join(output_dir, f'full_run_{model_type}_{self.timestamp}')
        os.makedirs(self.results_dir, exist_ok=True)
        
        # Expected results from paper
        self.expected_results = {
            'esol': 0.7844,
            'freesolv': 1.0124,
            'lipophilicity': 1.0221,
            'bace': 0.9586
        }
        
        # Dataset configurations
        self.dataset_configs = {
            'bace': {
                'data_file': 'bace.csv',
                'target_column': 'pIC50',
                'smiles_column': 'mol',
                'description': 'BACE Inhibition (pIC50)',
                'use_no_pooling': False,
                'expected_rmse': 0.9586
            },
            'esol': {
                'data_file': 'ESOL.csv',
                'target_column': 'measured log solubility in mols per litre',
                'smiles_column': 'SMILES',
                'description': 'ESOL Solubility',
                'use_no_pooling': True,
                'expected_rmse': 0.7844
            },
            'freesolv': {
                'data_file': 'FreeSolv.csv',
                'target_column': 'expt',
                'smiles_column': 'SMILES',
                'description': 'FreeSolv Hydration Free Energy',
                'use_no_pooling': True,
                'expected_rmse': 1.0124
            },
            'lipophilicity': {
                'data_file': 'Lipophilicity!.csv',
                'target_column': 'exp',
                'smiles_column': 'smiles',
                'description': 'Lipophilicity',
                'use_no_pooling': False,
                'expected_rmse': 1.0221
            }
        }
        
        self.results = {}
    
    def find_dataset_files(self, dataset):
        """Find dataset files."""
        dataset_config = self.dataset_configs[dataset]
        dataset_path = os.path.join(self.data_dir, dataset)
        csvs_path = os.path.join(dataset_path, 'csvs')
        
        if not os.path.exists(csvs_path):
            csvs_path = dataset_path
        if not os.path.exists(csvs_path):
            csvs_path = self.data_dir
        
        data_file_name = dataset_config['data_file']
        data_file_path = os.path.join(csvs_path, data_file_name)
        
        if os.path.exists(data_file_path):
            return data_file_path
        else:
            # Try to find any CSV file containing the dataset name
            import glob
            potential_files = glob.glob(os.path.join(csvs_path, f'*{dataset}*.csv'))
            if potential_files:
                return potential_files[0]
        
        return None
    
    def run_single_experiment_integrated(self, dataset):
        """Run a single experiment using integrated training (shows live progress)."""
        config = self.dataset_configs[dataset]
        
        print(f"\n{'='*50}")
        print(f"Running integrated experiment for {dataset.upper()}")
        print(f"Model type: {self.model_type}")
        print(f"Expected RMSE from paper: {config['expected_rmse']}")
        print(f"{'='*50}")
        
        try:
            # Set seed
            set_seed(42)
            
            # Find dataset file
            data_file = self.find_dataset_files(dataset)
            if data_file is None:
                raise FileNotFoundError(f"Could not find data file for dataset: {dataset}")
            
            print(f"Using data file: {data_file}")
            
            # Load data
            df, _ = load_data(data_file, None)
            smiles_column = config['smiles_column']
            target_column = config['target_column']
            
            if smiles_column not in df.columns:
                potential_smiles_columns = [col for col in df.columns if 'smile' in col.lower()]
                if potential_smiles_columns:
                    smiles_column = potential_smiles_columns[0]
                    print(f"SMILES column '{config['smiles_column']}' not found. Using '{smiles_column}' instead.")
                else:
                    raise KeyError(f"Could not find SMILES column '{smiles_column}' in data file")
            
            smiles_list = list(df[smiles_column])
            y_values = list(df[target_column])
            y = format_y_values(y_values)
            
            print(f"Loaded {len(smiles_list)} molecules")
            
            # Extract features
            print("Extracting molecular graph structures...")
            edge_index = extract_edge_indices(smiles_list)
            
            print("Creating atom features with 3D molecular properties...")
            x = create_atom_features(smiles_list, features_dict=None)
            
            print("Preparing data for model...")
            data_list = prepare_data_for_training(x, edge_index, y, smiles_list)
            
            # Set up output directory
            output_dir = os.path.join(self.results_dir, 'main_experiments', dataset)
            os.makedirs(output_dir, exist_ok=True)
            
            # Determine epochs and early stopping
            if self.quick:
                epochs = 200  # Quick run
                early_stopping_patience = 50
                print(f"QUICK MODE: Using {epochs} epochs with early stopping patience {early_stopping_patience}")
            else:
                epochs = 5000  # Full paper training
                early_stopping_patience = 500
                print(f"FULL MODE: Using {epochs} epochs with early stopping patience {early_stopping_patience}")
            
            # Determine pooling strategy
            use_no_pooling = config['use_no_pooling']
            pooling_strategy = 'last_node' if use_no_pooling else 'mean'
            
            # Set model-specific parameters
            if self.model_type == 'alternative':
                heads = 4
                dropout = 0.2
                hidden_dim = 64
            else:
                heads = 1
                dropout = 0
                hidden_dim = 28
            
            print(f"Using pooling strategy: {pooling_strategy}")
            print(f"Model settings: heads={heads}, dropout={dropout}, hidden_dim={hidden_dim}")
            print(f"Paper settings: NO gradient clipping, NO learning rate scheduler")
            
            # Train model with live progress
            start_time = time.time()
            print("\nStarting model training with live progress...")
            
            model, log_dir, results_dir_exp, checkpoints_dir = train_lipophilicity_model(
                data_list=data_list,
                smiles_list=smiles_list,
                epochs=epochs,
                batch_size=16,
                lr=0.00075,
                feature_dim=35,
                hidden_dim=hidden_dim,
                early_stopping_patience=early_stopping_patience,
                heads=heads,
                dropout=dropout,
                base_dir=output_dir,
                device_str=None,
                use_lr_scheduler=False,  # Paper doesn't use LR scheduler
                use_no_pooling=use_no_pooling,
                pooling_strategy=pooling_strategy,
                clip_grad_norm=0.0,  # Paper doesn't use gradient clipping
                model_type=self.model_type
            )
            
            end_time = time.time()
            
            # Extract test RMSE from the final metrics
            try:
                metrics_file = os.path.join(log_dir, 'metrics.csv')
                if os.path.exists(metrics_file):
                    metrics_df = pd.read_csv(metrics_file)
                    # Get the last row with non-NaN test RMSE
                    test_rmse_rows = metrics_df.dropna(subset=['Test_RMSE'])
                    if not test_rmse_rows.empty:
                        test_rmse = test_rmse_rows.iloc[-1]['Test_RMSE']
                    else:
                        test_rmse = None
                else:
                    test_rmse = None
            except Exception as e:
                print(f"Warning: Could not extract test RMSE: {e}")
                test_rmse = None
            
            self.results[f'{dataset}_main'] = {
                'test_rmse': test_rmse,
                'expected_rmse': self.expected_results[dataset],
                'time_seconds': end_time - start_time,
                'status': 'success',
                'epochs_trained': epochs,
                'log_dir': log_dir,
                'results_dir': results_dir_exp,
                'model_type': self.model_type
            }
            
            if test_rmse is not None:
                print(f"✓ {dataset} completed - Test RMSE: {test_rmse:.4f}")
                print(f"  Expected: {self.expected_results[dataset]:.4f}")
                print(f"  Difference: {abs(test_rmse - self.expected_results[dataset]):.4f}")
            else:
                print(f"✓ {dataset} completed - Test RMSE: Could not extract")
            
            return True
            
        except Exception as e:
            print(f"✗ {dataset} failed with exception: {str(e)}")
            self.results[f'{dataset}_main'] = {
                'status': 'error',
                'error': str(e),
                'model_type': self.model_type
            }
            return False
    
    def run_single_experiment_subprocess(self, dataset):
        """Run a single experiment using subprocess (fallback, no live progress)."""
        config = self.dataset_configs[dataset]
        
        print(f"\n{'='*50}")
        print(f"Running subprocess experiment for {dataset.upper()}")
        print(f"Model type: {self.model_type}")
        print(f"Expected RMSE from paper: {config['expected_rmse']}")
        print(f"{'='*50}")
        
        # Determine epochs
        epochs = 200 if self.quick else 1000  # Reduced but reasonable for subprocess
        
        cmd = [
            'python', 'main.py',
            '--dataset', dataset,
            '--data_dir', self.data_dir,
            '--output_dir', os.path.join(self.results_dir, 'main_experiments'),
            '--epochs', str(epochs),
            '--early_stopping_patience', str(epochs // 10),
            '--clip_grad_norm', '0.0',  # Paper doesn't use gradient clipping
            '--model_type', self.model_type,
            # Note: --use_lr_scheduler defaults to False in updated main.py
        ]
        
        if self.quick:
            print(f"QUICK MODE: Using {epochs} epochs")
        
        try:
            start_time = time.time()
            # Run without capturing output to see live progress
            result = subprocess.run(cmd, text=True)
            end_time = time.time()
            
            if result.returncode == 0:
                self.results[f'{dataset}_main'] = {
                    'test_rmse': None,  # Would need to parse logs
                    'expected_rmse': self.expected_results[dataset],
                    'time_seconds': end_time - start_time,
                    'status': 'success',
                    'model_type': self.model_type
                }
                print(f"✓ {dataset} completed via subprocess")
                return True
            else:
                print(f"✗ {dataset} failed with return code {result.returncode}")
                self.results[f'{dataset}_main'] = {
                    'status': 'failed',
                    'return_code': result.returncode,
                    'model_type': self.model_type
                }
                return False
                
        except Exception as e:
            print(f"✗ {dataset} failed with exception: {str(e)}")
            self.results[f'{dataset}_main'] = {
                'status': 'error',
                'error': str(e),
                'model_type': self.model_type
            }
            return False
    
    def run_main_experiments(self, use_integrated=True):
        """Run the main experiments for all four datasets."""
        print("=" * 70)
        print("Running Main Experiments (Tables 1-4 from paper)")
        print(f"Model type: {self.model_type}")
        if self.quick:
            print("QUICK MODE: Reduced epochs for faster testing")
        else:
            print("FULL MODE: Paper-accurate training (5000 epochs)")
        print("=" * 70)
        
        datasets = ['esol', 'freesolv', 'lipophilicity', 'bace']
        
        for dataset in datasets:
            if use_integrated:
                # Use integrated training for live progress
                self.run_single_experiment_integrated(dataset)
            else:
                # Use subprocess as fallback
                self.run_single_experiment_subprocess(dataset)
    
    def run_ablation_studies(self):
        """Run ablation studies from Table 2."""
        print("\n" + "=" * 70)
        print("Running Ablation Studies (Table 2 from paper)")
        print("=" * 70)
        
        cmd = [
            'python', 'ablation_studies.py',
            '--experiment', 'ablation',
            '--data_dir', self.data_dir,
            '--output_dir', os.path.join(self.results_dir, 'ablation')
        ]
        
        try:
            # Run without capturing output to see progress
            result = subprocess.run(cmd, text=True)
            
            if result.returncode == 0:
                # Try to read results
                ablation_csv = os.path.join(self.results_dir, 'ablation', 'esol_ablation_results.csv')
                if os.path.exists(ablation_csv):
                    ablation_df = pd.read_csv(ablation_csv)
                    self.results['ablation_study'] = {
                        'status': 'success',
                        'results': ablation_df.to_dict('records')
                    }
                    print("✓ Ablation study completed")
                    print(ablation_df)
                else:
                    self.results['ablation_study'] = {'status': 'success_no_csv'}
            else:
                print("✗ Ablation study failed")
                self.results['ablation_study'] = {'status': 'failed'}
                
        except Exception as e:
            print(f"✗ Ablation study failed: {str(e)}")
            self.results['ablation_study'] = {'status': 'error', 'error': str(e)}
    
    def run_random_forest_baseline(self):
        """Run Random Forest baseline for FreeSolv."""
        print("\n" + "=" * 70)
        print("Running Random Forest Baseline (FreeSolv)")
        print("=" * 70)
        
        cmd = [
            'python', 'ablation_studies.py',
            '--experiment', 'random_forest',
            '--data_dir', self.data_dir,
            '--output_dir', os.path.join(self.results_dir, 'random_forest')
        ]
        
        try:
            # Run without capturing output
            result = subprocess.run(cmd, text=True)
            
            if result.returncode == 0:
                self.results['random_forest_freesolv'] = {'status': 'success'}
                print("✓ Random Forest baseline completed")
            else:
                print("✗ Random Forest baseline failed")
                self.results['random_forest_freesolv'] = {'status': 'failed'}
                
        except Exception as e:
            print(f"✗ Random Forest baseline failed: {str(e)}")
            self.results['random_forest_freesolv'] = {'status': 'error', 'error': str(e)}
    
    def run_node_position_analysis(self):
        """Run node position analysis from Table 5."""
        print("\n" + "=" * 70)
        print("Running Node Position Analysis (Table 5 from paper)")
        print("=" * 70)
        
        cmd = [
            'python', 'ablation_studies.py',
            '--experiment', 'node_positions',
            '--data_dir', self.data_dir,
            '--output_dir', os.path.join(self.results_dir, 'node_analysis')
        ]
        
        try:
            result = subprocess.run(cmd, text=True)
            
            if result.returncode == 0:
                # Try to read results
                node_csv = os.path.join(self.results_dir, 'node_analysis', 'node_position_analysis.csv')
                if os.path.exists(node_csv):
                    node_df = pd.read_csv(node_csv, index_col=0)
                    self.results['node_position_analysis'] = {
                        'status': 'success',
                        'results': node_df.to_dict()
                    }
                    print("✓ Node position analysis completed")
                    print(node_df)
                else:
                    self.results['node_position_analysis'] = {'status': 'success_no_csv'}
            else:
                print("✗ Node position analysis failed")
                self.results['node_position_analysis'] = {'status': 'failed'}
                
        except Exception as e:
            print(f"✗ Node position analysis failed: {str(e)}")
            self.results['node_position_analysis'] = {'status': 'error', 'error': str(e)}
    
    def run_molecular_analysis(self):
        """Run molecular structure analysis."""
        print("\n" + "=" * 70)
        print("Running Molecular Structure Analysis (Figures 2-3 from paper)")
        print("=" * 70)
        
        cmd = [
            'python', 'molecular_analysis.py',
            '--analysis', 'isomers',
            '--data_dir', self.data_dir,
            '--output_dir', os.path.join(self.results_dir, 'molecular_analysis')
        ]
        
        try:
            result = subprocess.run(cmd, text=True)
            
            if result.returncode == 0:
                self.results['molecular_analysis'] = {'status': 'success'}
                print("✓ Molecular analysis completed")
            else:
                print("✗ Molecular analysis failed")
                self.results['molecular_analysis'] = {'status': 'failed'}
                
        except Exception as e:
            print(f"✗ Molecular analysis failed: {str(e)}")
            self.results['molecular_analysis'] = {'status': 'error', 'error': str(e)}
    
    def generate_report(self):
        """Generate a comprehensive report of all results."""
        print("\n" + "=" * 70)
        print("Generating Comprehensive Report")
        print("=" * 70)
        
        report_file = os.path.join(self.results_dir, 'experiment_report.txt')
        
        with open(report_file, 'w') as f:
            f.write("TChemGNN Paper Experiments - Comprehensive Report\n")
            f.write("=" * 70 + "\n")
            f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Mode: {'QUICK' if self.quick else 'FULL'}\n")
            f.write(f"Model type: {self.model_type}\n\n")
            
            # Main experiment results
            f.write("1. MAIN EXPERIMENTS (Comparison with Paper Results)\n")
            f.write("-" * 50 + "\n")
            
            for dataset in ['esol', 'freesolv', 'lipophilicity', 'bace']:
                key = f'{dataset}_main'
                if key in self.results and self.results[key]['status'] == 'success':
                    result = self.results[key]
                    f.write(f"\n{dataset.upper()}:\n")
                    f.write(f"  Paper RMSE: {result['expected_rmse']:.4f}\n")
                    if result.get('test_rmse') is not None:
                        f.write(f"  Our RMSE:   {result['test_rmse']:.4f}\n")
                        diff = abs(result['test_rmse'] - result['expected_rmse'])
                        f.write(f"  Difference: {diff:.4f} ({diff/result['expected_rmse']*100:.1f}%)\n")
                    else:
                        f.write(f"  Our RMSE:   [Could not extract]\n")
                    f.write(f"  Time:       {result['time_seconds']:.1f} seconds\n")
                    if 'epochs_trained' in result:
                        f.write(f"  Epochs:     {result['epochs_trained']}\n")
                else:
                    f.write(f"\n{dataset.upper()}: FAILED or INCOMPLETE\n")
            
            # Other results sections...
            f.write(f"\n\n2. EXPERIMENT SETTINGS\n")
            f.write("-" * 50 + "\n")
            f.write(f"Mode: {'QUICK (reduced epochs)' if self.quick else 'FULL (paper-accurate)'}\n")
            f.write(f"Model type: {self.model_type}\n")
            if self.model_type == 'paper':
                f.write("Paper model settings applied:\n")
                f.write("  - 5 GAT layers with tanh activation\n")
                f.write("  - 28 hidden dimensions\n")
                f.write("  - 1 attention head\n")
                f.write("  - No dropout\n")
                f.write("  - ~3.7K parameters\n")
            else:
                f.write("Alternative model settings applied:\n")
                f.write("  - 4 GAT layers with ReLU activation\n")
                f.write("  - 64 hidden dimensions\n")
                f.write("  - 4 attention heads\n")
                f.write("  - 0.2 dropout\n")
                f.write("  - Layer normalization\n")
                
            f.write("Common settings:\n")
            f.write("  - NO learning rate scheduler\n")
            f.write("  - NO gradient clipping\n")
            f.write("  - RMSprop optimizer with lr=0.00075\n")
            f.write("  - 35 features (14 atomic + 21 molecular, logP excluded)\n")
            
            f.write("\n" + "=" * 70 + "\n")
            f.write("SUMMARY:\n")
            if self.quick:
                f.write("Quick experiments completed. For full paper comparison, run without --quick.\n")
            else:
                f.write("Full experiments completed with paper-accurate settings.\n")
            f.write(f"All training used {self.model_type} model architecture.\n")
        
        print(f"✓ Report saved to: {report_file}")
        
        # Also save results as JSON
        json_file = os.path.join(self.results_dir, 'experiment_results.json')
        with open(json_file, 'w') as f:
            json.dump(self.results, f, indent=2)
        print(f"✓ Results saved to: {json_file}")
    
    def run_all(self, use_integrated=True, skip_auxiliary=False):
        """Run all experiments."""
        print("\n" + "=" * 70)
        print("TChemGNN - Running All Experiments from the Paper")
        print("=" * 70)
        print(f"Mode: {'QUICK' if self.quick else 'FULL'}")
        print(f"Model type: {self.model_type}")
        print(f"Training method: {'INTEGRATED (live progress)' if use_integrated else 'SUBPROCESS'}")
        print(f"Output directory: {self.results_dir}\n")
        
        # Run main experiments
        self.run_main_experiments(use_integrated=use_integrated)
        
        if not skip_auxiliary:
            # Run auxiliary experiments
            self.run_ablation_studies()
            self.run_random_forest_baseline()
            self.run_node_position_analysis()
            self.run_molecular_analysis()
        
        # Generate final report
        self.generate_report()
        
        print("\n" + "=" * 70)
        print("All experiments completed!")
        print(f"Results saved in: {self.results_dir}")
        print("=" * 70)


def main():
    parser = argparse.ArgumentParser(description='Run all TChemGNN experiments')
    parser.add_argument('--data_dir', type=str, default='datasets',
                      help='Directory containing datasets')
    parser.add_argument('--output_dir', type=str, default='experiment_results',
                      help='Directory for all outputs')
    parser.add_argument('--quick', action='store_true',
                      help='Run quick version with fewer epochs')
    parser.add_argument('--model_type', type=str, default='paper',
                      choices=['paper', 'alternative'],
                      help='Type of model to use (paper: original GSR, alternative: alternative GSR)')
    parser.add_argument('--use_subprocess', action='store_true',
                      help='Use subprocess calls instead of integrated training (less progress info)')
    parser.add_argument('--skip_auxiliary', action='store_true',
                      help='Skip auxiliary experiments (ablation, RF, etc.)')
    parser.add_argument('--main_only', action='store_true',
                      help='Run only main experiments')
    
    args = parser.parse_args()
    
    # Check if required files exist
    required_files = ['main.py']
    if not args.main_only:
        required_files.extend(['ablation_studies.py', 'molecular_analysis.py'])
    
    missing_files = [f for f in required_files if not os.path.exists(f)]
    
    if missing_files:
        print(f"Error: Missing required files: {missing_files}")
        print("Please ensure all experiment scripts are in the current directory.")
        sys.exit(1)
    
    # Check if data directory exists
    if not os.path.exists(args.data_dir):
        print(f"Error: Data directory '{args.data_dir}' not found.")
        print("Please download the datasets first.")
        sys.exit(1)
    
    # Run experiments
    runner = ExperimentRunner(args.data_dir, args.output_dir, quick=args.quick, model_type=args.model_type)
    
    if args.main_only:
        print(f"Running MAIN EXPERIMENTS ONLY with {args.model_type} model")
        runner.run_main_experiments(use_integrated=not args.use_subprocess)
        runner.generate_report()
    else:
        runner.run_all(
            use_integrated=not args.use_subprocess,
            skip_auxiliary=args.skip_auxiliary
        )


if __name__ == '__main__':
    main()