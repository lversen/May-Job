# Fix for visualization/visualize.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def visualize_results(log_dir, results_dir, epoch=None):
    """
    Create visualization plots for training results with standardized plot ranges.
    
    Parameters:
    - log_dir: Directory containing training logs
    - results_dir: Directory to save visualization results
    - epoch: Optional specific epoch to visualize
    """
    print(f"Creating visualizations in {results_dir}")
    
    # Load metrics
    metrics_file = os.path.join(log_dir, 'metrics.csv')
    if not os.path.exists(metrics_file):
        metrics_file = os.path.join(log_dir, 'training_metrics.csv')
    
    metrics_df = pd.read_csv(metrics_file)
    
    # Create results directory
    os.makedirs(results_dir, exist_ok=True)
    
    # Create visualization progress bar
    viz_steps = 2  # Base steps (loss and RMSE curves)
    
    viz_pbar = tqdm(total=viz_steps, desc="Creating visualizations", leave=False)
    
    # Define standardized ranges for plots based on domain knowledge
    # These are default values that can be overridden if the actual data exceeds them
    
    # Standard ranges for lipophilicity prediction
    value_min, value_max = -3.0, 7.0  # Typical logP range for drug-like molecules
    
    # Standard ranges for metrics
    loss_min, loss_max = 0.0, 1.0  # Typical MSE range
    rmse_min, rmse_max = 0.0, 1.0  # Typical RMSE range
    error_min, error_max = 0.0, 2.0  # Typical error range
    
    # Check if we need to adjust the ranges based on actual data
    loss_max = max(loss_max, metrics_df['Train_Loss'].max() * 1.1, 
                  metrics_df['Val_Loss'].max() * 1.1)
    
    if 'Test_Loss' in metrics_df.columns:
        test_loss_max = metrics_df['Test_Loss'].dropna().max() if not metrics_df['Test_Loss'].empty else 0
        loss_max = max(loss_max, test_loss_max * 1.1)
    
    rmse_max = max(rmse_max, metrics_df['Train_RMSE'].max() * 1.1, 
                  metrics_df['Val_RMSE'].max() * 1.1)
    
    if 'Test_RMSE' in metrics_df.columns:
        test_rmse_max = metrics_df['Test_RMSE'].dropna().max() if not metrics_df['Test_RMSE'].empty else 0
        rmse_max = max(rmse_max, test_rmse_max * 1.1)
    
    # Plot loss curves with logarithmic scale
    plt.figure(figsize=(12, 8))
    
    # Ensure all loss values are positive for log scale (add small epsilon to zeros or negative values)
    epsilon = 1e-8
    train_loss = metrics_df['Train_Loss'].apply(lambda x: max(x, epsilon))
    val_loss = metrics_df['Val_Loss'].apply(lambda x: max(x, epsilon))
    
    plt.plot(metrics_df['Epoch'], train_loss, label='Training Loss')
    plt.plot(metrics_df['Epoch'], val_loss, label='Validation Loss')
    
    # Only plot test loss if it exists and has data points
    if 'Test_Loss' in metrics_df.columns:
        # Filter epochs where test loss is actually measured
        test_df = metrics_df.dropna(subset=['Test_Loss'])
        if not test_df.empty:
            test_loss = test_df['Test_Loss'].apply(lambda x: max(x, epsilon))
            plt.plot(test_df['Epoch'], test_loss, label='Test Loss', 
                    marker='o', markersize=4, linestyle='-', linewidth=1.5)
    
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves (Log Scale)')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    
    # Set logarithmic scale for y-axis
    plt.yscale('log')
    
    # Add grid lines for the log scale
    plt.grid(True, which="minor", ls="--", alpha=0.4)
    
    # Save the figure
    plt.savefig(os.path.join(results_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    viz_pbar.update(1)
    
    # Plot RMSE curves with logarithmic scale
    plt.figure(figsize=(12, 8))
    
    # Ensure all RMSE values are positive for log scale (add small epsilon to zeros or negative values)
    epsilon = 1e-8
    train_rmse = metrics_df['Train_RMSE'].apply(lambda x: max(x, epsilon))
    val_rmse = metrics_df['Val_RMSE'].apply(lambda x: max(x, epsilon))
    
    plt.plot(metrics_df['Epoch'], train_rmse, label='Training RMSE')
    plt.plot(metrics_df['Epoch'], val_rmse, label='Validation RMSE')
    
    # Only plot test RMSE if it exists and has data points
    if 'Test_RMSE' in metrics_df.columns:
        # Filter epochs where test RMSE is actually measured
        test_df = metrics_df.dropna(subset=['Test_RMSE'])
        if not test_df.empty:
            test_rmse = test_df['Test_RMSE'].apply(lambda x: max(x, epsilon))
            plt.plot(test_df['Epoch'], test_rmse, label='Test RMSE', 
                    marker='o', markersize=4, linestyle='-', linewidth=1.5)
    
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('RMSE Curves (Log Scale)')
    plt.legend()
    plt.grid(True, which="both", ls="-")
    
    # Set logarithmic scale for y-axis
    plt.yscale('log')
    
    # Add grid lines for the log scale
    plt.grid(True, which="minor", ls="--", alpha=0.4)
    
    # Save the figure
    plt.savefig(os.path.join(results_dir, 'rmse_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    viz_pbar.update(1)
    
    # If a specific epoch is provided, visualize predictions for that epoch
    if epoch is not None:
        epoch_dir = os.path.join(log_dir, f'epoch_{epoch}')
        
        # Check if prediction files exist for this epoch
        train_pred_file = os.path.join(epoch_dir, 'train_predictions.csv')
        val_pred_file = os.path.join(epoch_dir, 'val_predictions.csv')
        test_pred_file = os.path.join(epoch_dir, 'test_predictions.csv')
        
        if os.path.exists(train_pred_file) and os.path.exists(val_pred_file) and os.path.exists(test_pred_file):
            viz_pbar.total += 3  # Add steps for epoch-specific visualizations
            viz_pbar.refresh()
            
            # Load prediction data
            train_df = pd.read_csv(train_pred_file)
            val_df = pd.read_csv(val_pred_file)
            test_df = pd.read_csv(test_pred_file)
            
            # Update value ranges based on actual data if necessary
            actual_min = min(train_df['Actual'].min(), val_df['Actual'].min(), test_df['Actual'].min())
            actual_max = max(train_df['Actual'].max(), val_df['Actual'].max(), test_df['Actual'].max())
            pred_min = min(train_df['Predicted'].min(), val_df['Predicted'].min(), test_df['Predicted'].min())
            pred_max = max(train_df['Predicted'].max(), val_df['Predicted'].max(), test_df['Predicted'].max())
            
            value_min = min(value_min, actual_min, pred_min) - 0.5  # Add some margin
            value_max = max(value_max, actual_max, pred_max) + 0.5  # Add some margin
            
            # Update error range if necessary
            max_error = max(train_df['Error'].max(), val_df['Error'].max(), test_df['Error'].max())
            error_max = max(error_max, max_error * 1.1)  # Add 10% margin
            
            # Combined scatter plot with standardized range
            plt.figure(figsize=(12, 10))
            plt.scatter(train_df['Actual'], train_df['Predicted'], alpha=0.6, label='Train', color='blue')
            plt.scatter(val_df['Actual'], val_df['Predicted'], alpha=0.6, label='Validation', color='green')
            plt.scatter(test_df['Actual'], test_df['Predicted'], alpha=0.6, label='Test', color='red')
            
            # Add perfect prediction line
            plt.plot([value_min, value_max], [value_min, value_max], 'k--', alpha=0.5)
            
            plt.xlim(value_min, value_max)
            plt.ylim(value_min, value_max)
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title(f'Predicted vs Actual Values (Epoch {epoch})')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(results_dir, f'predictions_epoch_{epoch}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            viz_pbar.update(1)
            
            # Error histogram with standardized range
            plt.figure(figsize=(12, 8))
            combined_errors = pd.concat([
                train_df[['Error']].assign(Dataset='Train'),
                val_df[['Error']].assign(Dataset='Validation'),
                test_df[['Error']].assign(Dataset='Test')
            ])
            
            sns.histplot(data=combined_errors, x='Error', hue='Dataset', bins=30, alpha=0.6)
            plt.xlim(error_min, error_max)  # Set standardized x-axis limits
            plt.title(f'Error Distribution (Epoch {epoch})')
            plt.xlabel('Absolute Error')
            plt.ylabel('Count')
            plt.grid(True)
            plt.savefig(os.path.join(results_dir, f'error_histogram_epoch_{epoch}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            viz_pbar.update(1)
            
            # Visualize node predictions if files exist
            node_pred_files_exist = all([
                os.path.exists(os.path.join(epoch_dir, f'{dataset}_node_predictions.csv')) 
                for dataset in ['train', 'val', 'test']
            ])
            
            if node_pred_files_exist:
                # Pass the standardized ranges to the node prediction visualization function
                visualize_node_predictions(log_dir, results_dir, epoch, 
                                         value_min=value_min, value_max=value_max,
                                         error_min=error_min, error_max=error_max)
                viz_pbar.update(1)
            else:
                print(f"Skipping node prediction visualization for epoch {epoch} as files are missing")
        else:
            print(f"Skipping detailed visualizations for epoch {epoch} as prediction files are missing")
            print(f"Consider setting logging interval to include this epoch or using the nearest epoch multiple of 50")
    
    viz_pbar.close()
    print(f"Visualizations saved to {results_dir}")


def visualize_node_predictions(log_dir, results_dir, epoch, value_min=None, value_max=None, 
                              error_min=0.0, error_max=2.0):
    """
    Create visualizations for node predictions with standardized ranges.
    
    Parameters:
    - log_dir: Directory containing training logs
    - results_dir: Directory to save visualization results
    - epoch: Epoch to visualize
    - value_min: Minimum value for plot ranges (if None, will be determined from data)
    - value_max: Maximum value for plot ranges (if None, will be determined from data)
    - error_min: Minimum error value for plot ranges
    - error_max: Maximum error value for plot ranges
    """
    epoch_dir = os.path.join(log_dir, f'epoch_{epoch}')
    
    # Check if node prediction files exist
    train_node_file = os.path.join(epoch_dir, 'train_node_predictions.csv')
    val_node_file = os.path.join(epoch_dir, 'val_node_predictions.csv')
    test_node_file = os.path.join(epoch_dir, 'test_node_predictions.csv')
    
    if not (os.path.exists(train_node_file) and os.path.exists(val_node_file) and os.path.exists(test_node_file)):
        return
    
    # Create a progress bar for node visualization
    node_viz_pbar = tqdm(total=3, desc="Creating node visualizations", leave=False)
    
    # Load node predictions
    train_nodes = pd.read_csv(train_node_file)
    val_nodes = pd.read_csv(val_node_file)
    test_nodes = pd.read_csv(test_node_file)
    
    # Add dataset identifier
    train_nodes['dataset'] = 'Train'
    val_nodes['dataset'] = 'Validation'
    test_nodes['dataset'] = 'Test'
    
    # Combine all node predictions
    all_nodes = pd.concat([train_nodes, val_nodes, test_nodes])
    
    # Default plot ranges if not provided
    if value_min is None or value_max is None:
        # Determine ranges based on the data
        node_min = all_nodes['node_prediction'].min()
        node_max = all_nodes['node_prediction'].max()
        target_min = all_nodes['molecule_target'].min()
        target_max = all_nodes['molecule_target'].max()
        
        # Set ranges with some margin
        margin = 0.5
        data_min = min(node_min, target_min) - margin
        data_max = max(node_max, target_max) + margin
        
        # Use these if not provided
        value_min = value_min if value_min is not None else data_min
        value_max = value_max if value_max is not None else data_max
    
    # Create directory for node visualizations
    node_viz_dir = os.path.join(results_dir, 'node_visualizations')
    os.makedirs(node_viz_dir, exist_ok=True)
    
    # Distribution of node predictions
    plt.figure(figsize=(12, 8))
    sns.histplot(data=all_nodes, x='node_prediction', hue='dataset', bins=30, alpha=0.6)
    plt.xlim(value_min, value_max)  # Set standardized x-axis limits
    plt.title(f'Distribution of Node Predictions (Epoch {epoch})')
    plt.xlabel('Node Prediction Value')
    plt.ylabel('Count')
    plt.grid(True)
    plt.savefig(os.path.join(node_viz_dir, f'node_prediction_distribution_epoch_{epoch}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    node_viz_pbar.update(1)
    
    # Node predictions vs molecule targets
    plt.figure(figsize=(12, 8))
    for dataset, color in zip(['Train', 'Validation', 'Test'], ['blue', 'green', 'red']):
        dataset_nodes = all_nodes[all_nodes['dataset'] == dataset]
        plt.scatter(dataset_nodes['molecule_target'], dataset_nodes['node_prediction'], 
                  alpha=0.1, label=dataset, color=color, s=20)
    
    # Add diagonal line for reference
    plt.plot([value_min, value_max], [value_min, value_max], 'k--', alpha=0.5)
    
    plt.xlim(value_min, value_max)
    plt.ylim(value_min, value_max)
    plt.xlabel('Molecule Target (Lipophilicity)')
    plt.ylabel('Node Prediction')
    plt.title(f'Node Predictions vs Molecule Targets (Epoch {epoch})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(node_viz_dir, f'node_vs_target_epoch_{epoch}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    node_viz_pbar.update(1)
    
    # Determine y-axis limits for the molecule index plot
    # This is a special case as we plot both predictions and targets
    mol_y_min = min(value_min, all_nodes['molecule_target'].min()) - 0.5
    mol_y_max = max(value_max, all_nodes['molecule_target'].max()) + 0.5
    
    # Node predictions by molecule index
    plt.figure(figsize=(12, 8))
    for dataset, color in zip(['Train', 'Validation', 'Test'], ['blue', 'green', 'red']):
        dataset_nodes = all_nodes[all_nodes['dataset'] == dataset]
        # Check if dataset has any nodes
        if len(dataset_nodes) > 0:
            # Calculate mean node prediction per molecule
            mol_means = dataset_nodes.groupby('molecule_index')['node_prediction'].mean()
            # Get corresponding targets
            mol_targets = dataset_nodes.groupby('molecule_index')['molecule_target'].first()
            
            plt.scatter(mol_means.index, mol_means.values, alpha=0.6, label=f'{dataset} (Pred)', color=color, s=20)
            plt.scatter(mol_targets.index, mol_targets.values, alpha=0.6, label=f'{dataset} (Target)', 
                      color=color, marker='x', s=30)
    
    plt.ylim(mol_y_min, mol_y_max)  # Set standardized y-axis limits
    plt.xlabel('Molecule Index')
    plt.ylabel('Value')
    plt.title(f'Mean Node Predictions vs Molecule Targets (Epoch {epoch})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(node_viz_dir, f'prediction_vs_target_by_molecule_epoch_{epoch}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    node_viz_pbar.update(1)
    
    node_viz_pbar.close()