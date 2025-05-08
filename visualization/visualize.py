# Fix for visualization/visualize.py
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm


def visualize_results(log_dir, results_dir, epoch=None):
    """
    Create visualization plots for training results.
    
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
    
    # Plot loss curves
    plt.figure(figsize=(12, 8))
    plt.plot(metrics_df['Epoch'], metrics_df['Train_Loss'], label='Training Loss')
    plt.plot(metrics_df['Epoch'], metrics_df['Val_Loss'], label='Validation Loss')
    plt.plot(metrics_df['Epoch'], metrics_df['Test_Loss'], label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (MSE)')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(results_dir, 'loss_curves.png'), dpi=300, bbox_inches='tight')
    plt.close()
    viz_pbar.update(1)
    
    # Plot RMSE curves
    plt.figure(figsize=(12, 8))
    plt.plot(metrics_df['Epoch'], metrics_df['Train_RMSE'], label='Training RMSE')
    plt.plot(metrics_df['Epoch'], metrics_df['Val_RMSE'], label='Validation RMSE')
    plt.plot(metrics_df['Epoch'], metrics_df['Test_RMSE'], label='Test RMSE')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE')
    plt.title('RMSE Curves')
    plt.legend()
    plt.grid(True)
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
            
            # Visualization code continues as before...
            # Combined scatter plot
            plt.figure(figsize=(12, 10))
            plt.scatter(train_df['Actual'], train_df['Predicted'], alpha=0.6, label='Train', color='blue')
            plt.scatter(val_df['Actual'], val_df['Predicted'], alpha=0.6, label='Validation', color='green')
            plt.scatter(test_df['Actual'], test_df['Predicted'], alpha=0.6, label='Test', color='red')
            
            # Add perfect prediction line
            min_val = min(train_df['Actual'].min(), val_df['Actual'].min(), test_df['Actual'].min())
            max_val = max(train_df['Actual'].max(), val_df['Actual'].max(), test_df['Actual'].max())
            plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
            
            plt.xlabel('Actual')
            plt.ylabel('Predicted')
            plt.title(f'Predicted vs Actual Values (Epoch {epoch})')
            plt.legend()
            plt.grid(True)
            plt.savefig(os.path.join(results_dir, f'predictions_epoch_{epoch}.png'), dpi=300, bbox_inches='tight')
            plt.close()
            viz_pbar.update(1)
            
            # Error histogram
            plt.figure(figsize=(12, 8))
            combined_errors = pd.concat([
                train_df[['Error']].assign(Dataset='Train'),
                val_df[['Error']].assign(Dataset='Validation'),
                test_df[['Error']].assign(Dataset='Test')
            ])
            
            sns.histplot(data=combined_errors, x='Error', hue='Dataset', bins=30, alpha=0.6)
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
                visualize_node_predictions(log_dir, results_dir, epoch)
                viz_pbar.update(1)
            else:
                print(f"Skipping node prediction visualization for epoch {epoch} as files are missing")
        else:
            print(f"Skipping detailed visualizations for epoch {epoch} as prediction files are missing")
            print(f"Consider setting logging interval to include this epoch or using the nearest epoch multiple of 50")
    
    viz_pbar.close()
    print(f"Visualizations saved to {results_dir}")


def visualize_node_predictions(log_dir, results_dir, epoch):
    """
    Create visualizations for node predictions.
    
    Parameters:
    - log_dir: Directory containing training logs
    - results_dir: Directory to save visualization results
    - epoch: Epoch to visualize
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
    
    # Create directory for node visualizations
    node_viz_dir = os.path.join(results_dir, 'node_visualizations')
    os.makedirs(node_viz_dir, exist_ok=True)
    
    # Distribution of node predictions
    plt.figure(figsize=(12, 8))
    sns.histplot(data=all_nodes, x='node_prediction', hue='dataset', bins=30, alpha=0.6)
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
    min_val = min(all_nodes['molecule_target'].min(), all_nodes['node_prediction'].min())
    max_val = max(all_nodes['molecule_target'].max(), all_nodes['node_prediction'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5)
    
    plt.xlabel('Molecule Target (Lipophilicity)')
    plt.ylabel('Node Prediction')
    plt.title(f'Node Predictions vs Molecule Targets (Epoch {epoch})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(node_viz_dir, f'node_vs_target_epoch_{epoch}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    node_viz_pbar.update(1)
    
    # Node predictions by molecule index
    plt.figure(figsize=(12, 8))
    for dataset, color in zip(['Train', 'Validation', 'Test'], ['blue', 'green', 'red']):
        dataset_nodes = all_nodes[all_nodes['dataset'] == dataset]
        # Calculate mean node prediction per molecule
        mol_means = dataset_nodes.groupby('molecule_index')['node_prediction'].mean()
        # Get corresponding targets
        mol_targets = dataset_nodes.groupby('molecule_index')['molecule_target'].first()
        
        plt.scatter(mol_means.index, mol_means.values, alpha=0.6, label=f'{dataset} (Pred)', color=color, s=20)
        plt.scatter(mol_targets.index, mol_targets.values, alpha=0.6, label=f'{dataset} (Target)', 
                  color=color, marker='x', s=30)
    
    plt.xlabel('Molecule Index')
    plt.ylabel('Value')
    plt.title(f'Mean Node Predictions vs Molecule Targets (Epoch {epoch})')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(node_viz_dir, f'prediction_vs_target_by_molecule_epoch_{epoch}.png'), dpi=300, bbox_inches='tight')
    plt.close()
    node_viz_pbar.update(1)
    
    node_viz_pbar.close()