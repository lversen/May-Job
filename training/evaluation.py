import torch
import pandas as pd
import os
import csv  # Add this import for CSV writer

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
        for batch in data_loader:
            batch = batch.to(device)
            node_outputs, graph_outputs = model(batch)
            targets = batch.y
            
            loss = criterion(graph_outputs, targets)
            rmse = compute_rmse(graph_outputs, targets)
            
            total_loss += loss.item() * batch.num_graphs
            total_rmse += rmse.item() * batch.num_graphs
            
            all_graph_predictions.extend(graph_outputs.cpu().numpy())
            all_node_predictions.extend(node_outputs.cpu().numpy())
            all_node_batch_indices.extend(batch.batch.cpu().numpy())
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
    
    # Save metrics in both formats for compatibility
    
    # 1. Save to metrics.csv using CSV writer (stable version style)
    metrics_file_path = os.path.join(log_dir, 'metrics.csv')
    file_exists = os.path.exists(metrics_file_path)
    
    with open(metrics_file_path, 'a', newline='') as metrics_file:
        metrics_writer = csv.writer(metrics_file)
        
        # Write headers if file doesn't exist
        if not file_exists:
            metrics_writer.writerow(['Epoch', 'Train_Loss', 'Train_RMSE', 'Val_Loss', 'Val_RMSE', 'Test_Loss', 'Test_RMSE'])
        
        # Write metrics row
        metrics_writer.writerow([epoch, train_loss, train_rmse, val_loss, val_rmse, test_loss, test_rmse])
        
        # Ensure data is written to file
        metrics_file.flush()
    
    # 2. ALSO save to training_metrics.csv using pandas for backward compatibility
    metrics_df = pd.DataFrame({
        'Epoch': [epoch],
        'Train_Loss': [train_loss],
        'Train_RMSE': [train_rmse],
        'Val_Loss': [val_loss],
        'Val_RMSE': [val_rmse],
        'Test_Loss': [test_loss],
        'Test_RMSE': [test_rmse]
    })
    
    training_metrics_file = os.path.join(log_dir, 'training_metrics.csv')
    
    if os.path.exists(training_metrics_file):
        metrics_df.to_csv(training_metrics_file, mode='a', header=False, index=False)
    else:
        metrics_df.to_csv(training_metrics_file, mode='w', header=True, index=False)
    
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
    for node_idx, (pred, batch_idx) in enumerate(zip(node_predictions, node_batch_indices)):
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