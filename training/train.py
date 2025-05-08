import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from datetime import datetime
from torch_geometric.loader import DataLoader
import numpy as np 
from tqdm import tqdm  # Import tqdm for progress bars

from .evaluation import evaluate_model_with_nodes, log_to_csv, log_node_predictions_to_csv
from visualization.visualize import visualize_results
from utils.helpers import get_device


def train_lipophilicity_model(data_list, smiles_list, 
                             epochs=1000, 
                             batch_size=32,
                             lr=0.001,
                             feature_dim=35, 
                             hidden_dim=64, 
                             early_stopping_patience=100,
                             heads=4,
                             dropout=0.2,
                             base_dir="",
                             device_str=None):
    """
    Train a lipophilicity prediction model with logging similar to the stable version.
    
    Parameters:
    - data_list: List of PyTorch Geometric Data objects
    - smiles_list: List of SMILES strings
    - epochs: Maximum number of training epochs
    - batch_size: Batch size for training (0 means no batching - use full dataset)
    - lr: Learning rate
    - feature_dim: Dimension of atom features
    - hidden_dim: Dimension of hidden layers
    - early_stopping_patience: Number of epochs to wait before early stopping
    - heads: Number of attention heads in GAT
    - dropout: Dropout probability
    - base_dir: Base directory for outputs
    - device_str: Optional string to specify device ('cpu', 'cuda', 'cuda:0', etc.)
    
    Returns:
    - model: Trained model
    - log_dir: Directory containing logs
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
    device = get_device(device_str)
    print(f"Using device: {device}")
    
    # Create directories for logs and results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create parent directories if they don't exist
    logs_parent = os.path.join(base_dir, "logs")
    results_parent = os.path.join(base_dir, "results")
    checkpoints_parent = os.path.join(base_dir, "checkpoints")
    
    os.makedirs(logs_parent, exist_ok=True)
    os.makedirs(results_parent, exist_ok=True)
    os.makedirs(checkpoints_parent, exist_ok=True)
    
    # Create timestamped subdirectories
    log_dir = os.path.join(logs_parent, f'logs_{timestamp}')
    results_dir = os.path.join(results_parent, f'results_{timestamp}')
    checkpoints_dir = os.path.join(checkpoints_parent, f'checkpoints_{timestamp}')
    
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
    
    # Handle batch_size=0 (no batching) case
    if batch_size <= 0:
        print("Batch size set to 0 or negative value - using full dataset (no batching)")
        train_batch_size = len(train_data)
        val_batch_size = len(val_data)
        test_batch_size = len(test_data)
    else:
        train_batch_size = batch_size
        val_batch_size = batch_size
        test_batch_size = batch_size
    
    # Create DataLoaders with appropriate batch sizes
    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=val_batch_size)
    test_loader = DataLoader(test_data, batch_size=test_batch_size)
    
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
    
    # Training loop with tqdm progress bar
    print(f"Starting training for {epochs} epochs...")
    print(f"Using {'full dataset (no batching)' if batch_size <= 0 else f'batch size of {batch_size}'}")
    pbar = tqdm(range(1, epochs + 1), desc="Training Progress", position=0)
    
    for epoch in pbar:
        model.train()
        epoch_start_time = time.time()
        train_loss = 0
        train_rmse = 0
        
        # Progress bar for batches (optional, can be disabled for cleaner output)
        # batch_pbar = tqdm(train_loader, desc=f"Epoch {epoch}", leave=False, position=1) 
        # for batch in batch_pbar:
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
        
        # Update progress bar with metrics
        pbar.set_postfix({
            'train_loss': f'{avg_train_loss:.4f}',
            'train_rmse': f'{avg_train_rmse:.4f}',
            'val_loss': f'{val_loss:.4f}',
            'val_rmse': f'{val_rmse:.4f}',
            'best_epoch': best_epoch,
            'lr': f'{optimizer.param_groups[0]["lr"]:.6f}',
            'patience': f'{early_stopping_counter}/{early_stopping_patience}'
        })
        
        # Early stopping check
        if early_stopping_counter >= early_stopping_patience:
            pbar.write(f"Early stopping triggered after {epoch} epochs")
            break
        
        # Log detailed metrics every 50 epochs (to match stable version)
        if epoch % 50 == 0:
            model.eval()
            
            # Create epoch directory
            epoch_dir = os.path.join(log_dir, f'epoch_{epoch}')
            os.makedirs(epoch_dir, exist_ok=True)
            
            # Get predictions for all datasets
            with torch.no_grad():
                # Testing
                test_loss = 0
                total_test_rmse = 0
                all_test_preds = []
                all_test_targets = []
                
                test_pbar = tqdm(enumerate(test_loader), total=len(test_loader), 
                                desc=f"Testing (Epoch {epoch})", leave=False)
                for i, data_batch in test_pbar:
                    data_batch = data_batch.to(device)
                    _, y_pred = model(data_batch)
                    batch_test_loss = criterion(y_pred, data_batch.y)
                    test_loss += batch_test_loss.item() * data_batch.num_graphs
                    
                    # Calculate RMSE for test
                    test_rmse = compute_rmse(y_pred, data_batch.y)
                    total_test_rmse += test_rmse.item() * data_batch.num_graphs
                    
                    # Print RMSE and loss for each element in the test set (like stable version)
                    for j in range(data_batch.num_graphs):
                        sample_test_pred = y_pred[j].cpu().numpy()
                        sample_test_target = data_batch.y[j].cpu().numpy()
                        all_test_preds.append(sample_test_pred)
                        all_test_targets.append(sample_test_target)
                        
                        # Calculate individual test loss and RMSE for this sample
                        ind_test_loss = ((sample_test_pred - sample_test_target) ** 2).mean()
                        ind_test_rmse = np.sqrt(ind_test_loss)
                        
                        if j == 0:  # Only print for the first element to avoid clutter
                            test_pbar.write(f"element {i*test_batch_size+j}, Epoch {epoch}: "
                                          f"Test Loss: {ind_test_loss:.4f}, Test RMSE: {ind_test_rmse:.4f}")
                
                avg_test_loss = test_loss / len(test_loader.dataset)
                avg_test_rmse = total_test_rmse / len(test_loader.dataset)
                pbar.write(f"Epoch {epoch} - Test Loss: {avg_test_loss:.4f}, Test RMSE: {avg_test_rmse:.4f}")
            
            # Get full dataset predictions for logging and visualization
            pbar.write(f"Generating visualizations for epoch {epoch}...")
            
            # Create progress bars for the evaluation steps
            with tqdm(total=3, desc="Evaluating datasets", leave=False) as eval_pbar:
                # Use same batch_size logic for evaluation
                _, _, train_graph_preds, train_node_preds, train_node_batch, train_targets = evaluate_model_with_nodes(
                    model, DataLoader(train_data, batch_size=train_batch_size), criterion, device
                )
                eval_pbar.update(1)
                
                _, _, val_graph_preds, val_node_preds, val_node_batch, val_targets = evaluate_model_with_nodes(
                    model, DataLoader(val_data, batch_size=val_batch_size), criterion, device
                )
                eval_pbar.update(1)
                
                _, _, test_graph_preds, test_node_preds, test_node_batch, test_targets = evaluate_model_with_nodes(
                    model, DataLoader(test_data, batch_size=test_batch_size), criterion, device
                )
                eval_pbar.update(1)
            
            # Log graph-level results to CSV
            log_to_csv(
                epoch, avg_train_loss, avg_train_rmse, val_loss, val_rmse, avg_test_loss, avg_test_rmse,
                train_graph_preds, train_targets, val_graph_preds, val_targets,
                test_graph_preds, test_targets, smiles_list, log_dir
            )
            
            # Log node-level predictions to CSV
            log_node_predictions_to_csv(train_node_preds, train_node_batch, train_targets, smiles_list, train_indices, epoch_dir, 'train')
            log_node_predictions_to_csv(val_node_preds, val_node_batch, val_targets, smiles_list, val_indices, epoch_dir, 'val')
            log_node_predictions_to_csv(test_node_preds, test_node_batch, test_targets, smiles_list, test_indices, epoch_dir, 'test')
            
            # Visualize results
            visualize_results(log_dir, results_dir, epoch)
            pbar.write(f"Completed visualizations for epoch {epoch}")
    
    pbar.close()
    
    # Load the best model
    model.load_state_dict(torch.load(os.path.join(checkpoints_dir, 'best_model.pt')))
    
    # Final evaluation with node predictions
    print("\nPerforming final evaluation on the best model...")
    with tqdm(total=3, desc="Final Evaluation", position=0) as final_pbar:
        train_loss, train_rmse, train_graph_preds, train_node_preds, train_node_batch, train_targets = evaluate_model_with_nodes(
            model, DataLoader(train_data, batch_size=train_batch_size), criterion, device
        )
        final_pbar.update(1)
        
        val_loss, val_rmse, val_graph_preds, val_node_preds, val_node_batch, val_targets = evaluate_model_with_nodes(
            model, DataLoader(val_data, batch_size=val_batch_size), criterion, device
        )
        final_pbar.update(1)
        
        test_loss, test_rmse, test_graph_preds, test_node_preds, test_node_batch, test_targets = evaluate_model_with_nodes(
            model, DataLoader(test_data, batch_size=test_batch_size), criterion, device
        )
        final_pbar.update(1)
    
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