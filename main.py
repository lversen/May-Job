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
        'smiles_column': 'mol',  # Add SMILES column name
        'description': 'BACE Inhibition (pIC50)'
    },
    'esol': {
        'data_file': 'ESOL.csv',
        'target_column': 'measured log solubility in mols per litre',
        'smiles_column': 'SMILES',  # Different SMILES column name for ESOL
        'description': 'ESOL Solubility'
    },
    'fsolv': {
        'data_file': 'Fsolv.csv',
        'target_column': 'expt',
        'smiles_column': 'smiles',  # Assuming FreeSolv uses 'SMILES'
        'description': 'FreeSolv Hydration Free Energy'
    },
    'lipophilicity': {
        'data_file': 'Lipophilicity!.csv',
        'target_column': 'exp',
        'smiles_column': 'smiles',
        'description': 'Lipophilicity'
    }
}

# Feature file names to look for
FEATURE_FILES = [
    'angles.csv',
    'dipole_momentum.csv',
    'widths.csv',
    'lengths.csv',
    'heights.csv',
    'volumes.csv'
]

# Global features that can be disabled
GLOBAL_FEATURES = [
    'angles',
    'dipole_momentum',
    'widths',
    'lengths',
    'heights',
    'volumes'
]


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
    - args: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Molecular Property Prediction with Graph Neural Networks')
    
    # Dataset parameters
    parser.add_argument('--dataset', type=str, required=True, 
                      choices=list(DATASET_CONFIG.keys()),
                      help=f'Dataset to use: {", ".join(DATASET_CONFIG.keys())}')
    parser.add_argument('--data_dir', type=str, default='datasets',
                      help='Base directory containing dataset folders')
    parser.add_argument('--smiles_column', type=str, default=None,
                      help='Name of the SMILES column in the data file (overrides default)')
    
    # Feature control parameters
    parser.add_argument('--disable_global_features', action='store_true',
                      help='Disable global molecular features (angles, dipole_momentum, widths, lengths, heights, volumes)')
    parser.add_argument('--exclude_features', nargs='*', 
                      choices=GLOBAL_FEATURES,
                      help='Specific global features to exclude (e.g., --exclude_features angles volumes)')
    
    # Model parameters
    parser.add_argument('--feature_dim', type=int, default=35,
                      help='Dimension of atom features (will be automatically adjusted if global features are disabled)')
    parser.add_argument('--hidden_dim', type=int, default=64,
                      help='Dimension of hidden layers')
    parser.add_argument('--heads', type=int, default=1,
                      help='Number of attention heads in GAT')
    parser.add_argument('--dropout', type=float, default=0,
                      help='Dropout probability')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=16,
                      help='Batch size for training (0 means no batching - full dataset)')
    parser.add_argument('--epochs', type=int, default=5000,
                      help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=0.00075,
                      help='Learning rate')
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
    
    # Add new parameter for creating GIFs (optional feature)
    parser.add_argument('--create_gifs', action='store_true',
                      help='Create animated GIFs from training visualizations')
    
    return parser.parse_args()


def find_dataset_files(dataset, data_dir, disable_global_features=False, exclude_features=None):
    """
    Find all relevant files for the specified dataset.
    
    Parameters:
    - dataset: Name of the dataset
    - data_dir: Base directory containing dataset folders
    - disable_global_features: Whether to disable all global features
    - exclude_features: List of specific features to exclude
    
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
        'feature_files': {}
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
    
    # Find feature files (skip if global features are disabled)
    if not disable_global_features:
        exclude_features = exclude_features or []
        
        for feature_file in FEATURE_FILES:
            feature_name = os.path.splitext(feature_file)[0]  # Remove the .csv extension
            
            # Skip excluded features
            if feature_name in exclude_features:
                continue
                
            feature_path = os.path.join(csvs_path, feature_file)
            if os.path.exists(feature_path):
                result['feature_files'][feature_name] = feature_path
    
    return result


def calculate_feature_dimension(features_dict):
    """
    Calculate the actual feature dimension based on loaded features.
    
    Parameters:
    - features_dict: Dictionary of loaded features
    
    Returns:
    - int: Total feature dimension
    """
    from data.preprocessing import get_base_feature_dimension
    
    # Get base features dimension
    base_features = get_base_feature_dimension()
    
    # Add additional global features count
    additional_features_count = len(features_dict) if features_dict else 0
    
    # The actual dimension includes base atom features + additional global features
    return base_features + additional_features_count


def main():
    """
    Main function to run the molecular property prediction model.
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
    
    # Add feature configuration to output directory name
    if args.disable_global_features:
        output_dir += "_no_global_features"
    elif args.exclude_features:
        excluded_str = "_exclude_" + "_".join(args.exclude_features)
        output_dir += excluded_str
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 50)
    print(f"{dataset_config['description']} Prediction - Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    if args.disable_global_features:
        print("RUNNING WITHOUT GLOBAL FEATURES")
    elif args.exclude_features:
        print(f"EXCLUDING FEATURES: {', '.join(args.exclude_features)}")
    print("=" * 50)
    
    # Find dataset files
    files = find_dataset_files(args.dataset, args.data_dir, 
                             args.disable_global_features, 
                             args.exclude_features)
    
    if files['data_file'] is None:
        raise FileNotFoundError(f"Could not find data file for dataset: {args.dataset}")
    
    print(f"Using data file: {files['data_file']}")
    
    if args.disable_global_features:
        print("Global features disabled - using only atom-level features")
    else:
        print(f"Found {len(files['feature_files'])} feature files:")
        for feature, file_path in files['feature_files'].items():
            print(f"  - {feature}: {file_path}")
        
        if args.exclude_features:
            print(f"Excluded features: {', '.join(args.exclude_features)}")
    
    # Display batch size setting
    if args.batch_size <= 0:
        print("\nRunning with NO batching (using full dataset)")
    else:
        print(f"\nBatch size: {args.batch_size}")
    
    # Display device setting
    if args.device:
        print(f"Device: {args.device}")
    else:
        print("Device: Auto (will use CUDA if available)")
    
    # Display gradient clipping setting
    print(f"Gradient clipping norm: {args.clip_grad_norm}")
    
    print("\nLoading data...")
    
    # Load data with additional feature files
    df, features_dict = load_data(files['data_file'], files['feature_files'])
    
    # Calculate actual feature dimension
    actual_feature_dim = calculate_feature_dimension(features_dict)
    
    # Override user-specified feature_dim if it doesn't match
    if args.feature_dim != actual_feature_dim:
        print(f"Adjusting feature dimension from {args.feature_dim} to {actual_feature_dim} based on loaded features")
        feature_dim = actual_feature_dim
    else:
        feature_dim = args.feature_dim
    
    # Determine SMILES column - command line arg overrides config
    smiles_column = args.smiles_column if args.smiles_column else dataset_config['smiles_column']
    
    # Display column info
    print(f"Target column: '{dataset_config['target_column']}'")
    print(f"SMILES column: '{smiles_column}'")
    print(f"Feature dimension: {feature_dim}")
    
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
    
    # Extract edge indices for graph construction
    print("Extracting molecular graph structures...")
    edge_index = extract_edge_indices(smiles_list)
    
    # Create atom features
    print("Creating atom features...")
    x = create_atom_features(smiles_list, features_dict)
    
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
        feature_dim=feature_dim,  # Use calculated feature dimension
        hidden_dim=args.hidden_dim,
        early_stopping_patience=int(args.epochs/10),
        heads=args.heads,
        dropout=args.dropout,
        base_dir=output_dir,
        device_str=args.device,
        use_lr_scheduler=args.use_lr_scheduler
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
    print("=" * 50)


if __name__ == '__main__':
    main()