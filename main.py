import argparse
import os
import time
import glob
from datetime import datetime

from data import load_data, extract_edge_indices, create_atom_features, prepare_data_for_training
from models import GSR
from training import train_lipophilicity_model
from visualization.gif_generator import create_training_animation
from utils import set_seed, format_y_values


# Dataset configurations with specific column names
DATASET_CONFIG = {
    'bace': {
        'data_file': 'bace.csv',
        'target_column': 'pIC50',
        'smiles_column': 'mol',
        'description': 'BACE Inhibition (pIC50)',
        'use_no_pooling': False,  # Paper uses pooling for BACE
        'expected_rmse': 0.9586
    },
    'esol': {
        'data_file': 'ESOL.csv',
        'target_column': 'measured log solubility in mols per litre',
        'smiles_column': 'SMILES',
        'description': 'ESOL Solubility',
        'use_no_pooling': True,  # Paper uses last node for ESOL
        'expected_rmse': 0.7844
    },
    'freesolv': {  # Note: paper uses "FreeSolv" but we keep consistent naming
        'data_file': 'FreeSolv.csv',
        'target_column': 'expt',
        'smiles_column': 'SMILES',
        'description': 'FreeSolv Hydration Free Energy',
        'use_no_pooling': True,  # Paper uses last node for FreeSolv
        'expected_rmse': 1.0124
    },
    'lipophilicity': {
        'data_file': 'Lipophilicity!.csv',
        'target_column': 'exp',
        'smiles_column': 'smiles',
        'description': 'Lipophilicity',
        'use_no_pooling': False,  # Paper mentions mean pooling works slightly better
        'expected_rmse': 1.0221
    }
}


def parse_args():
    """
    Parse command line arguments aligned with paper specifications.
    
    Returns:
    - args: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='TChemGNN: Molecular Property Prediction with Graph Neural Networks')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, required=True, 
                      choices=list(DATASET_CONFIG.keys()),
                      help=f'Dataset to use: {", ".join(DATASET_CONFIG.keys())}')
    parser.add_argument('--data_dir', type=str, default='datasets',
                      help='Base directory containing dataset folders')
    parser.add_argument('--smiles_column', type=str, default=None,
                      help='Name of the SMILES column in the data file (overrides default)')
    
    # Model parameters (aligned with paper)
    parser.add_argument('--feature_dim', type=int, default=36,
                      help='Dimension of atom features (36 in paper: 14 atomic + 22 molecular)')
    parser.add_argument('--hidden_dim', type=int, default=28,
                      help='Dimension of hidden layers (28 gives best results in paper)')
    parser.add_argument('--heads', type=int, default=1,
                      help='Number of attention heads in GAT (1 in paper)')
    parser.add_argument('--dropout', type=float, default=0,
                      help='Dropout probability (0 in paper)')
    
    # Pooling strategy
    parser.add_argument('--pooling_strategy', type=str, default='auto',
                      choices=['auto', 'mean', 'last_node', 'first_node'],
                      help='Pooling strategy: auto (dataset-specific), mean, last_node, first_node')
    
    # Training parameters (aligned with paper)
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=5000,
                      help='Maximum number of epochs (5000 in paper)')
    parser.add_argument('--lr', type=float, default=0.00075,
                      help='Learning rate (paper mentions RMSprop optimizer)')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    parser.add_argument('--clip_grad_norm', type=float, default=1.0,
                      help='Maximum gradient norm for gradient clipping (0 to disable)')
    parser.add_argument('--use_lr_scheduler', action='store_true', default=True,
                      help='Use learning rate scheduler (ReduceLROnPlateau)')
    parser.add_argument('--no_lr_scheduler', action='store_false', dest='use_lr_scheduler',
                      help='Disable learning rate scheduler')
    
    # Device parameter
    parser.add_argument('--device', type=str, default=None,
                      help='Device to use (e.g. "cpu", "cuda", "cuda:0"). If not specified, will use CUDA if available.')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='output',
                      help='Base directory for outputs')
    
    # Visualization parameters
    parser.add_argument('--create_gifs', action='store_true',
                      help='Create animated GIFs from training visualizations')
    
    return parser.parse_args()


def find_dataset_files(dataset, data_dir):
    """
    Find all relevant files for the specified dataset.
    
    Parameters:
    - dataset: Name of the dataset
    - data_dir: Base directory containing dataset folders
    
    Returns:
    - dict: Dictionary containing all file paths
    """
    dataset_config = DATASET_CONFIG[dataset]
    dataset_path = os.path.join(data_dir, dataset)
    csvs_path = os.path.join(dataset_path, 'csvs')
    
    # If csvs subdirectory doesn't exist, try the dataset directory itself
    if not os.path.exists(csvs_path):
        csvs_path = dataset_path
    
    # If dataset directory doesn't exist, try the base directory
    if not os.path.exists(csvs_path):
        csvs_path = data_dir
    
    result = {
        'data_file': None,
        'feature_files': {}  # Paper uses RDKit to compute 3D features on-the-fly
    }
    
    # Find the main data file
    data_file_name = dataset_config['data_file']
    data_file_path = os.path.join(csvs_path, data_file_name)
    
    if os.path.exists(data_file_path):
        result['data_file'] = data_file_path
    else:
        # Try to find any CSV file containing the dataset name
        potential_files = glob.glob(os.path.join(csvs_path, f'*{dataset}*.csv'))
        if potential_files:
            result['data_file'] = potential_files[0]
    
    return result


def main():
    """
    Main function to run the TChemGNN model aligned with the paper.
    """
    # Parse arguments
    args = parse_args()
    
    # Get dataset configuration
    dataset_config = DATASET_CONFIG[args.dataset]
    
    # Set random seed
    set_seed(args.seed)
    
    # Start timing
    start_time = time.time()
    
    # Set up output directory
    output_dir = os.path.join(args.output_dir, args.dataset)
    
    # Add configuration info to output directory name
    output_dir += f"_hidden{args.hidden_dim}"
    if args.pooling_strategy != 'auto':
        output_dir += f"_{args.pooling_strategy}"
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 50)
    print(f"TChemGNN - {dataset_config['description']} Prediction")
    print(f"Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Expected RMSE from paper: {dataset_config['expected_rmse']}")
    print("=" * 50)
    
    # Determine pooling strategy
    if args.pooling_strategy == 'auto':
        use_no_pooling = dataset_config['use_no_pooling']
        pooling_strategy = 'last_node' if use_no_pooling else 'mean'
        print(f"Using dataset-specific pooling strategy: {pooling_strategy}")
    else:
        pooling_strategy = args.pooling_strategy
        use_no_pooling = pooling_strategy != 'mean'
        print(f"Using user-specified pooling strategy: {pooling_strategy}")
    
    # Find dataset files
    files = find_dataset_files(args.dataset, args.data_dir)
    
    if files['data_file'] is None:
        raise FileNotFoundError(f"Could not find data file for dataset: {args.dataset}")
    
    print(f"Using data file: {files['data_file']}")
    
    # Display model configuration
    print(f"\nModel Configuration (TChemGNN):")
    print(f"  Feature dimension: {args.feature_dim} (14 atomic + 22 molecular)")
    print(f"  Hidden dimension: {args.hidden_dim}")
    print(f"  Number of GAT layers: 5")
    print(f"  Attention heads: {args.heads}")
    print(f"  Activation: tanh")
    print(f"  Dropout: {args.dropout}")
    print(f"  Pooling: {pooling_strategy}")
    
    # Display training configuration
    print(f"\nTraining Configuration:")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Learning rate: {args.lr}")
    print(f"  Optimizer: RMSprop")
    print(f"  Max epochs: {args.epochs}")
    print(f"  LR scheduler: {'Enabled' if args.use_lr_scheduler else 'Disabled'}")
    print(f"  Device: {args.device if args.device else 'Auto (will use CUDA if available)'}")
    
    print("\nLoading data...")
    
    # Load data (no external feature files - all computed from SMILES)
    df, _ = load_data(files['data_file'], None)
    
    # Determine SMILES column - command line arg overrides config
    smiles_column = args.smiles_column if args.smiles_column else dataset_config['smiles_column']
    
    # Display column info
    print(f"Target column: '{dataset_config['target_column']}'")
    print(f"SMILES column: '{smiles_column}'")
    
    # Check if the SMILES column exists
    if smiles_column not in df.columns:
        # If not, try to automatically detect it
        potential_smiles_columns = [col for col in df.columns if 'smile' in col.lower()]
        if potential_smiles_columns:
            smiles_column = potential_smiles_columns[0]
            print(f"SMILES column '{dataset_config['smiles_column']}' not found. Using '{smiles_column}' instead.")
        else:
            # Show available columns and raise error
            print(f"Available columns: {', '.join(df.columns)}")
            raise KeyError(f"Could not find SMILES column '{smiles_column}' in data file")
    
    # Extract SMILES and target values
    smiles_list = list(df[smiles_column])
    target_column = dataset_config['target_column']
    y_values = list(df[target_column])
    y = format_y_values(y_values)
    
    print(f"Loaded {len(smiles_list)} molecules")
    
    # For lipophilicity dataset, remove problematic molecules as mentioned in paper
    if args.dataset == 'lipophilicity':
        print("Note: Paper mentions removing 2 molecules where RDKit couldn't compute 3D structure")
    
    # Extract edge indices for graph construction
    print("Extracting molecular graph structures...")
    edge_index = extract_edge_indices(smiles_list)
    
    # Create atom features (with 3D features computed on-the-fly)
    print("Creating atom features with 3D molecular properties...")
    x = create_atom_features(smiles_list, features_dict=None)
    
    # Verify feature dimension
    if len(x[0][0]) != args.feature_dim:
        print(f"Warning: Expected {args.feature_dim} features but got {len(x[0][0])}")
    
    # Prepare data for training
    print("Preparing data for model...")
    data_list = prepare_data_for_training(x, edge_index, y, smiles_list)
    
    # Train model
    print("\nStarting model training...")
    model, log_dir, results_dir, checkpoints_dir = train_lipophilicity_model(
        data_list=data_list,
        smiles_list=smiles_list,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        early_stopping_patience=int(args.epochs/10),
        heads=args.heads,
        dropout=args.dropout,
        base_dir=output_dir,
        device_str=args.device,
        use_lr_scheduler=args.use_lr_scheduler,
        use_no_pooling=use_no_pooling,
        pooling_strategy=pooling_strategy,
        clip_grad_norm=args.clip_grad_norm
    )
        
    # Create animated GIFs from the training visualizations if requested
    if args.create_gifs:
        print("\nCreating animated GIFs from training visualizations...")
        gif_paths = create_training_animation(log_dir, results_dir)
        
        # Print paths to created GIFs
        print("\nCreated GIFs:")
        for viz_type, path in gif_paths.items():
            print(f"  - {viz_type}: {path}")
    
    # Done
    elapsed_time = time.time() - start_time
    print("\n" + "=" * 50)
    print(f"Training completed in {elapsed_time/60:.2f} minutes")
    print(f"Results saved to {results_dir}")
    print(f"Logs saved to {log_dir}")
    print(f"Model checkpoints saved to {checkpoints_dir}")
    print(f"Expected RMSE from paper: {dataset_config['expected_rmse']}")
    print("=" * 50)


if __name__ == '__main__':
    main()