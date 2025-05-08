import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch_geometric.loader import DataLoader
import numpy as np

from .evaluation import evaluate_model_with_nodes, log_to_csv, log_node_predictions_to_csv
from visualization.visualize import visualize_results


def train_lipophilicity_model(data_list, smiles_list, 
                             epochs=1000, 
                             batch_size=32,
                             lr=0.001,
                             feature_dim=35, 
                             hidden_dim=64, 
                             early_stopping_patience=100,
                             heads=4,
                             dropout=0.2):
    """
    Train a lipophilicity prediction model.
    
    Parameters:
    - data_list: List of PyTorch Geometric Data objects
    - smiles_list: List of SMILES strings
    - epochs: Maximum number of training epochs
    - batch_size: Batch size
    - lr: Learning rate
    - feature_dim: Dimension of input features
    - hidden_dim: Dimension of hidden layers
    - early_stopping_patience: Number of epochs to wait before early stopping
    - heads: Number of attention heads in GAT
    - dropout: Dropout probability
    
    Returns:
    - model: Trained GSR model
    - log_dir: Directory containing training logs
    - results_dir: Directory containing results
    - checkpoints_dir: Directory containing model checkpoints
    """
    # Set seeds for reproducibility
    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create directories for logs and results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = f'logs_{timestamp}'
    results_dir = f'results_{timestamp}'
    checkpoints_dir = f'checkpoints_{timestamp}'
    
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)
    
    # Split data into training, validation and testing sets
    train_size = int(0.8 * len(data_list))
    val_size = int(0.1 * len(data_list))
    
    # Shuffle the data with fixed seed for reproducibility
    indices = list(range(len(data_list)))
    random.shuffle(indices)
    
    train_indices = indices[:train_size]
    val_indices = indices[train_size:train_size + val_size]
    test_indices = indices[train_size + val_size:]
    
    train_data = [data_list[i] for i in train_indices]
    val_data = [data_list[i] for i in val_indices]
    test_data = [data_list[i] for i in test_indices]
    
    # Create DataLoaders with batching
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    # Create model, optimizer, and loss function
    from models.gsr import GSR
    model = GSR(in_channels=feature_dim, 
                hidden_channels=hidden_dim, 
                out_channels=1, 
                heads=heads, 
                dropout=dropout).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=50)
    
    criterion = nn.MSELoss()
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    early_stopping_counter = 0
    best_epoch = 0
    
    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start_time = time.time()
        train_loss = 0
        train_rmse = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            _, graph_output = model(batch)  # Ignore node predictions during training
            loss = criterion(graph_output, batch.y)
            
            loss.backward()
            optimizer.step()
            
            # Calculate RMSE
            from .evaluation import compute_rmse
            batch_rmse = compute_rmse(graph_output, batch.y)
            
            train_loss += loss.item() * batch.num_graphs
            train_rmse += batch_rmse.item() * batch.num_graphs
            
        # Calculate average training metrics
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_rmse = train_rmse / len(train_loader.dataset)
        
        # Evaluate on validation set
        val_loss, val_rmse, val_graph_preds, _, _, _ = evaluate_model_with_nodes(model, val_loader, criterion, device)
        
        # Update learning rate scheduler
        scheduler.step(val_loss)
        
        # Check for early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            early_stopping_counter = 0
            best_epoch = epoch
            
            # Save best model
            torch.save(model.state_dict(), os.path.join(checkpoints_dir, 'best_model.pt'))
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch} epochs")
                break
        
        # Calculate elapsed time for the epoch
        epoch_time = time.time() - epoch_start_time
        
        # Print progress
        if epoch % 10 == 0 or epoch == 1:
            # Evaluate on test set
            test_loss, test_rmse, test_graph_preds, _, _, _ = evaluate_model_with_nodes(model, test_loader, criterion, device)
            
            print(f"Epoch {epoch}/{epochs} | Time: {epoch_time:.2f}s")
            print(f"  Train Loss: {avg_train_loss:.4f} | Train RMSE: {avg_train_rmse:.4f}")
            print(f"  Val Loss: {val_loss:.4f} | Val RMSE: {val_rmse:.4f}")
            print(f"  Test Loss: {test_loss:.4f} | Test RMSE: {test_rmse:.4f}")
            print(f"  Best Val Loss: {best_val_loss:.4f} (Epoch {best_epoch})")
            print(f"  Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")
            
            # Create epoch directory
            epoch_dir = os.path.join(log_dir, f'epoch_{epoch}')
            os.makedirs(epoch_dir, exist_ok=True)
            
            # Get predictions for all datasets including node predictions
            _, _, train_graph_preds, train_node_preds, train_node_batch, train_targets = evaluate_model_with_nodes(
                model, DataLoader(train_data, batch_size=batch_size), criterion, device
            )
            _, _, val_graph_preds, val_node_preds, val_node_batch, val_targets = evaluate_model_with_nodes(
                model, DataLoader(val_data, batch_size=batch_size), criterion, device
            )
            _, _, test_graph_preds, test_node_preds, test_node_batch, test_targets = evaluate_model_with_nodes(
                model, DataLoader(test_data, batch_size=batch_size), criterion, device
            )
            
            # Log graph-level results to CSV
            log_to_csv(
                epoch, avg_train_loss, avg_train_rmse, val_loss, val_rmse, test_loss, test_rmse,
                train_graph_preds, train_targets, val_graph_preds, val_targets,
                test_graph_preds, test_targets, smiles_list, log_dir
            )
            
            # Log node-level predictions to CSV
            log_node_predictions_to_csv(train_node_preds, train_node_batch, train_targets, smiles_list, train_indices, epoch_dir, 'train')
            log_node_predictions_to_csv(val_node_preds, val_node_batch, val_targets, smiles_list, val_indices, epoch_dir, 'val')
            log_node_predictions_to_csv(test_node_preds, test_node_batch, test_targets, smiles_list, test_indices, epoch_dir, 'test')
            
            # Visualize results
            visualize_results(log_dir, results_dir, epoch)
    
    # Load the best model
    model.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'best_model.pt')))
    
    # Final evaluation with node predictions
    train_loss, train_rmse, train_graph_preds, train_node_preds, train_node_batch, train_targets = evaluate_model_with_nodes(
        model, DataLoader(train_data, batch_size=batch_size), criterion, device
    )
    val_loss, val_rmse, val_graph_preds, val_node_preds, val_node_batch, val_targets = evaluate_model_with_nodes(
        model, DataLoader(val_data, batch_size=batch_size), criterion, device
    )
    test_loss, test_rmse, test_graph_preds, test_node_preds, test_node_batch, test_targets = evaluate_model_with_nodes(
        model, DataLoader(test_data, batch_size=batch_size), criterion, device
    )
    
    print("\nFinal Evaluation (Best Model):")
    print(f"  Train Loss: {train_loss:.4f} | Train RMSE: {train_rmse:.4f}")
    print(f"  Val Loss: {val_loss:.4f} | Val RMSE: {val_rmse:.4f}")
    print(f"  Test Loss: {test_loss:.4f} | Test RMSE: {test_rmse:.4f}")
    
    # Create final epoch directory and save final node predictions
    final_dir = os.path.join(log_dir, f'epoch_final')
    os.makedirs(final_dir, exist_ok=True)
    
    log_node_predictions_to_csv(train_node_preds, train_node_batch, train_targets, smiles_list, train_indices, final_dir, 'train')
    log_node_predictions_to_csv(val_node_preds, val_node_batch, val_targets, smiles_list, val_indices, final_dir, 'val')
    log_node_predictions_to_csv(test_node_preds, test_node_batch, test_targets, smiles_list, test_indices, final_dir, 'test')
    
    # Create final visualizations
    visualize_results(log_dir, results_dir, best_epoch)
    
    return model, log_dir, results_dir, checkpoints_dir