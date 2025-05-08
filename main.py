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
                      help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=1000,
                      help='Maximum number of epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                      help='Learning rate')
    parser.add_argument('--early_stopping', type=int, default=100,
                      help='Patience for early stopping')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results',
                      help='Base directory for outputs')
    
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
    
    # Set up output directories with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_base = f"{args.output_dir}_{timestamp}"
    
    print("=" * 50)
    print(f"Lipophilicity Prediction - Starting at {timestamp}")
    print("=" * 50)
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
        dropout=args.dropout
    )
    
    # Create animated GIFs from the training visualizations
    print("\nCreating animated GIFs from training visualizations...")
    gif_paths = create_training_animation(log_dir, results_dir)
    
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