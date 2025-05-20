import torch
import pandas as pd
import os
import csv  # Add this import for CSV writer
from tqdm import tqdm  # Import tqdm

def compute_rmse(predictions, targets):
    """
    Compute Root Mean Square Error between predictions and targets.
    
    Parameters:
    - predictions: Predicted values
    - targets: Target values
    
    Returns:
    - rmse: Root Mean Square Error
    """
    mse = ((predictions - targets) ** 2).mean()
    rmse = torch.sqrt(mse)
    return rmse

def log_molecule_atom_predictions_to_csv(node_predictions, node_batch_indices, targets, indices, epoch_dir, epoch):
    """
    Save molecule-level predictions with all atom predictions to CSV.
    
    Parameters:
    - node_predictions: List of node (atom) predictions
    - node_batch_indices: List of batch indices for each node
    - targets: List of target values for each molecule
    - indices: Indices of molecules in the dataset
    - epoch_dir: Directory to save the CSV files
    - epoch: Current epoch number
    """
    import csv
    from tqdm import tqdm
    
    # Create a mapping between batch indices and molecule indices
    unique_batch_indices = set(node_batch_indices)
    batch_to_mol_idx = {batch_idx: idx for idx, batch_idx in enumerate(sorted(unique_batch_indices))}
    
    # Create a dictionary to store atom predictions by molecule
    molecule_atoms = {}
    
    # Group predictions by molecule
    for node_idx, (pred, batch_idx) in enumerate(tqdm(
        zip(node_predictions, node_batch_indices), 
        total=len(node_predictions),
        desc=f"Processing molecule atom predictions",
        leave=False
    )):
        mol_idx = batch_to_mol_idx[batch_idx]
        if mol_idx < len(indices):
            orig_mol_idx = indices[mol_idx]
            
            if orig_mol_idx not in molecule_atoms:
                molecule_atoms[orig_mol_idx] = []
            
            molecule_atoms[orig_mol_idx].append(pred[0])  # Assuming prediction is a 1D array with one value
    
    # Create y_pred CSV with atom predictions for each molecule
    y_pred_file = os.path.join(epoch_dir, f'y_pred_epoch_{epoch}.csv')
    with open(y_pred_file, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # For each molecule, write a row with all its atom predictions
        for orig_mol_idx, atom_preds in molecule_atoms.items():
            row = [orig_mol_idx] + atom_preds
            writer.writerow(row)
    
    # Create y_actual CSV with ground truth values
    y_actual_file = os.path.join(epoch_dir, f'y_actual_epoch_{epoch}.csv')
    with open(y_actual_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Molecule_ID', 'Actual_Value'])
        
        # Create a mapping from original molecule index to target value
        mol_idx_to_target = {}
        for i, orig_idx in enumerate(indices):
            if i < len(targets):
                mol_idx_to_target[orig_idx] = targets[i][0]
        
        # Write each molecule's target value
        for orig_mol_idx in molecule_atoms.keys():
            if orig_mol_idx in mol_idx_to_target:
                writer.writerow([orig_mol_idx, mol_idx_to_target[orig_mol_idx]])
    
    print(f"Saved atom predictions for {len(molecule_atoms)} molecules")
def evaluate_model_with_nodes(model, data_loader, criterion, device):
    """
    Evaluate model on a dataset, returning both node-level and graph-level predictions.
    
    Parameters:
    - model: Trained GSR model
    - data_loader: DataLoader containing test data
    - criterion: Loss function
    - device: Device to run evaluation on (CPU or GPU)
    
    Returns:
    - avg_loss: Average loss
    - avg_rmse: Average RMSE
    - all_graph_predictions: Graph-level predictions
    - all_node_predictions: Node-level predictions
    - all_node_batch_indices: Batch indices for nodes
    - all_targets: Target values
    """
    model.eval()
    total_loss = 0
    total_rmse = 0
    all_graph_predictions = []
    all_node_predictions = []
    all_node_batch_indices = []
    all_targets = []
    
    with torch.no_grad():
        # Add progress bar for evaluation
        for batch in tqdm(data_loader, desc="Evaluating", leave=False):
            batch = batch.to(device)
            node_outputs, graph_outputs = model(batch)
            targets = batch.y
            
            loss = criterion(graph_outputs, targets)
            rmse = compute_rmse(graph_outputs, targets)
            
            total_loss += loss.item() * batch.num_graphs
            total_rmse += rmse.item() * batch.num_graphs
            
            all_graph_predictions.extend(graph_outputs.cpu().numpy())
            all_node_predictions.extend(node_outputs.cpu().numpy())
            
            # Check if batch is the entire dataset (no batching mode)
            if hasattr(batch, 'batch') and batch.batch is not None:
                all_node_batch_indices.extend(batch.batch.cpu().numpy())
            else:
                # If no batching is used, create batch indices manually
                # Assign each node to its corresponding graph index
                graph_sizes = []
                idx = 0
                for graph_idx, _ in enumerate(targets):
                    # Get number of nodes in this graph
                    if idx < len(node_outputs):
                        # Count forward until we find the start of the next graph
                        nodes_count = 0
                        while idx + nodes_count < len(node_outputs):
                            nodes_count += 1
                        graph_sizes.append(nodes_count)
                        idx += nodes_count
                
                # Create batch indices based on graph sizes
                batch_indices = []
                for graph_idx, size in enumerate(graph_sizes):
                    batch_indices.extend([graph_idx] * size)
                all_node_batch_indices.extend(batch_indices)
                
            all_targets.extend(targets.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader.dataset)
    avg_rmse = total_rmse / len(data_loader.dataset)
    
    return avg_loss, avg_rmse, all_graph_predictions, all_node_predictions, all_node_batch_indices, all_targets


def log_to_csv(epoch, train_loss, train_rmse, val_loss, val_rmse, test_loss, test_rmse, 
              train_predictions, train_targets, val_predictions, val_targets, 
              test_predictions, test_targets, smiles_list, log_dir):
    """
    Log training results to CSV files similar to the stable version.
    
    Parameters:
    - epoch: Current epoch
    - train_loss, val_loss, test_loss: Loss values for each dataset
    - train_rmse, val_rmse, test_rmse: RMSE values for each dataset
    - train_predictions, val_predictions, test_predictions: Predicted values
    - train_targets, val_targets, test_targets: Target values
    - smiles_list: List of SMILES strings
    - log_dir: Directory to save logs
    """
    # Create epoch directory
    epoch_dir = os.path.join(log_dir, f'epoch_{epoch}')
    os.makedirs(epoch_dir, exist_ok=True)
    
    # Log metrics (for consistency, though they should already be logged every epoch)
    log_metrics_to_csv(epoch, train_loss, train_rmse, val_loss, val_rmse, test_loss, test_rmse, log_dir)
    
    # Create separate CSV files for y_test and y_pred (stable version style)
    y_test_file = open(os.path.join(epoch_dir, f'y_test_epoch_{epoch}.csv'), 'w', newline='')
    y_pred_file = open(os.path.join(epoch_dir, f'y_pred_epoch_{epoch}.csv'), 'w', newline='')
    
    y_test_writer = csv.writer(y_test_file)
    y_pred_writer = csv.writer(y_pred_file)
    
    # Write headers
    y_test_writer.writerow(['Sample_ID', 'y_test'])
    y_pred_writer.writerow(['Sample_ID', 'y_pred'])
    
    # Write test predictions to CSV files
    for i, (y_t, y_p) in enumerate(zip(test_targets, test_predictions)):
        sample_id = f'test_{i}'
        y_test_writer.writerow([sample_id, y_t[0]])
        y_pred_writer.writerow([sample_id, y_p[0]])
    
    # Close the y_test and y_pred files
    y_test_file.close()
    y_pred_file.close()
    
    # Also log predictions for train, val, and test sets as before
    train_indices = list(range(len(train_targets)))
    train_results_df = pd.DataFrame({
        'Index': train_indices,
        'SMILES': [smiles_list[i] for i in train_indices],
        'Actual': [t[0] for t in train_targets],
        'Predicted': [p[0] for p in train_predictions],
        'Error': [abs(p[0] - t[0]) for p, t in zip(train_predictions, train_targets)]
    })
    train_results_df.to_csv(os.path.join(epoch_dir, 'train_predictions.csv'), index=False)
    
    val_indices = list(range(len(train_targets), len(train_targets) + len(val_targets)))
    val_results_df = pd.DataFrame({
        'Index': val_indices,
        'SMILES': [smiles_list[i] for i in val_indices],
        'Actual': [t[0] for t in val_targets],
        'Predicted': [p[0] for p in val_predictions],
        'Error': [abs(p[0] - t[0]) for p, t in zip(val_predictions, val_targets)]
    })
    val_results_df.to_csv(os.path.join(epoch_dir, 'val_predictions.csv'), index=False)
    
    test_indices = list(range(len(train_targets) + len(val_targets), len(smiles_list)))
    test_results_df = pd.DataFrame({
        'Index': test_indices,
        'SMILES': [smiles_list[i] for i in test_indices],
        'Actual': [t[0] for t in test_targets],
        'Predicted': [p[0] for p in test_predictions],
        'Error': [abs(p[0] - t[0]) for p, t in zip(test_predictions, test_targets)]
    })
    test_results_df.to_csv(os.path.join(epoch_dir, 'test_predictions.csv'), index=False)


def log_node_predictions_to_csv(node_predictions, node_batch_indices, targets, smiles_list, indices, epoch_dir, dataset_name):
    """
    Save node-level predictions to CSV.
    
    Parameters:
    - node_predictions: List of node predictions
    - node_batch_indices: List of batch indices for each node
    - targets: List of target values (one per molecule)
    - smiles_list: List of SMILES strings
    - indices: Indices of molecules in the dataset
    - epoch_dir: Directory to save the CSV file
    - dataset_name: Name of the dataset (train, val, test)
    """
    # Create a dictionary to store node predictions for each molecule
    molecule_nodes = {}
    
    # Group nodes by molecule (batch index)
    unique_batch_indices = set(node_batch_indices)
    batch_to_mol_idx = {batch_idx: idx for idx, batch_idx in enumerate(sorted(unique_batch_indices))}
    
    # For each node, record its molecule index, prediction, and the molecule's target value
    rows = []
    
    # Use tqdm for progress visualization
    for node_idx, (pred, batch_idx) in enumerate(tqdm(
        zip(node_predictions, node_batch_indices), 
        total=len(node_predictions),
        desc=f"Processing {dataset_name} node predictions",
        leave=False
    )):
        mol_idx = batch_to_mol_idx[batch_idx]
        if mol_idx < len(indices):
            orig_mol_idx = indices[mol_idx]
            smiles = smiles_list[orig_mol_idx]
            # Get the target value for this molecule
            if mol_idx < len(targets):
                target_value = targets[mol_idx][0]  # Assuming target is a 1D array with one value
            else:
                target_value = float('nan')  # Handle case where target might be missing
                
            rows.append({
                'molecule_index': orig_mol_idx,
                'SMILES': smiles,
                'node_index': node_idx,
                'node_prediction': pred[0],  # Assuming prediction is a 1D array with one value
                'molecule_target': target_value,  # Add the molecule's target value
                'batch_index': batch_idx
            })
    
    # Create DataFrame and save to CSV
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(epoch_dir, f'{dataset_name}_node_predictions.csv'), index=False)
    
    print(f"Saved {len(rows)} node predictions for {dataset_name} dataset")

# First, modify the log_metrics_to_csv function in training/evaluation.py:

def log_metrics_to_csv(epoch, train_loss, train_rmse, val_loss, val_rmse, test_loss=None, test_rmse=None, log_dir=None):
    """
    Log only the metrics to CSV files, without saving detailed predictions.
    Called every epoch to create continuous data for loss curves.
    
    Parameters:
    - epoch: Current epoch
    - train_loss, val_loss, test_loss: Loss values for each dataset
    - train_rmse, val_rmse, test_rmse: RMSE values for each dataset
    - log_dir: Directory to save logs
    """
    import numpy as np
    
    # Create the DataFrame for the new row - ALWAYS include all columns
    metrics_row = {
        'Epoch': epoch,
        'Train_Loss': train_loss,
        'Train_RMSE': train_rmse,
        'Val_Loss': val_loss,
        'Val_RMSE': val_rmse,
        'Test_Loss': test_loss if test_loss is not None else np.nan,
        'Test_RMSE': test_rmse if test_rmse is not None else np.nan
    }
    
    metrics_df = pd.DataFrame([metrics_row])
    
    # Save to metrics.csv
    metrics_file_path = os.path.join(log_dir, 'metrics.csv')
    file_exists = os.path.exists(metrics_file_path)
    
    if file_exists:
        # Append the new row without header
        metrics_df.to_csv(metrics_file_path, mode='a', header=False, index=False)
    else:
        # First time creating the file
        metrics_df.to_csv(metrics_file_path, mode='w', header=True, index=False)
    
    # Also save to training_metrics.csv for backward compatibility
    training_metrics_file = os.path.join(log_dir, 'training_metrics.csv')
    
    if os.path.exists(training_metrics_file):
        metrics_df.to_csv(training_metrics_file, mode='a', header=False, index=False)
    else:
        metrics_df.to_csv(training_metrics_file, mode='w', header=True, index=False)