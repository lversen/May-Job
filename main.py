import argparse
import os
import time
from datetime import datetime

from data import load_data, extract_edge_indices, create_atom_features, prepare_data_for_training
from models import GSR
from training import train_lipophilicity_model
from visualization.gif_generator import create_training_animation
from utils import set_seed, format_y_values


def parse_args():
    """
    Parse command line arguments.
    
    Returns:
    - args: Parsed arguments
    """
    parser = argparse.ArgumentParser(description='Lipophilicity Prediction with Graph Neural Networks')
    
    # Data parameters
    parser.add_argument('--data_file', type=str, default='datasets/Lipophilicity!.csv',
                      help='Path to the main data file')
    parser.add_argument('--angles_file', type=str, default='datasets/angles.csv',
                      help='Path to the angles file')
    parser.add_argument('--dipole_momentum_file', type=str, default='datasets/dipole_momentum.csv',
                      help='Path to the dipole momentum file')
    parser.add_argument('--widths_file', type=str, default='datasets/widths.csv',
                      help='Path to the widths file')
    parser.add_argument('--lengths_file', type=str, default='datasets/lengths.csv',
                      help='Path to the lengths file')
    parser.add_argument('--heights_file', type=str, default='datasets/heights.csv',
                      help='Path to the heights file')
    parser.add_argument('--volumes_file', type=str, default='datasets/volumes.csv',
                      help='Path to the volumes file')
    
    # Model parameters
    parser.add_argument('--feature_dim', type=int, default=35,
                      help='Dimension of atom features')
    parser.add_argument('--hidden_dim', type=int, default=64,
                      help='Dimension of hidden layers')
    parser.add_argument('--heads', type=int, default=4,
                      help='Number of attention heads in GAT')
    parser.add_argument('--dropout', type=float, default=0.2,
                      help='Dropout probability')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                      help='Batch size for training (0 means no batching - full dataset)')
    parser.add_argument('--epochs', type=int, default=5000,
                      help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--early_stopping', type=int, default=500,
                      help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
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


def main():
    """
    Main function to run the lipophilicity prediction model.
    """
    # Parse arguments
    args = parse_args()
    
    # Set random seed
    set_seed(args.seed)
    
    # Start timing
    start_time = time.time()
    
    # Set up output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 50)
    print(f"Lipophilicity Prediction - Starting at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 50)
    
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
    
    print("\nLoading data...")
    
    # Load data
    additional_features_files = {
        'angles': args.angles_file,
        'dipole_momentum': args.dipole_momentum_file,
        'widths': args.widths_file,
        'lengths': args.lengths_file,
        'heights': args.heights_file,
        'volumes': args.volumes_file
    }
    
    df, features_dict = load_data(args.data_file, additional_features_files)
    
    # Extract SMILES and target values
    smiles_list = list(df['smiles'])
    y_values = list(df['exp'])
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
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        early_stopping_patience=args.early_stopping,
        heads=args.heads,
        dropout=args.dropout,
        base_dir=output_dir,  # Pass the base directory to the training function
        device_str=args.device  # Pass the device string
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