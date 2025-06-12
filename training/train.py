import os
import time
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
from datetime import datetime
from torch_geometric.loader import DataLoader
import numpy as np 
from tqdm import tqdm

from .evaluation import *
from visualization.visualize import visualize_results
from utils.helpers import get_device


def get_node_predictions_for_graphs(node_preds, batch, node_selection='last'):
    """
    Extract graph-level predictions from node predictions based on selection strategy.
    
    Parameters:
    - node_preds: Node-level predictions
    - batch: Batch indices for nodes
    - node_selection: Strategy for selecting nodes ('last', 'first', 'mean')
    
    Returns:
    - graph_preds: Graph-level predictions
    """
    batch_size = batch.max().item() + 1
    graph_preds = []
    
    for i in range(batch_size):
        # Get nodes for this graph
        mask = batch == i
        graph_node_preds = node_preds[mask]
        
        if node_selection == 'last':
            # Use the last node (as per paper for ESOL/FreeSolv)
            selected_pred = graph_node_preds[-1]
        elif node_selection == 'first':
            # Use the first node
            selected_pred = graph_node_preds[0]
        elif node_selection == 'mean':
            # Use mean pooling
            selected_pred = graph_node_preds.mean(dim=0)
        else:
            # Default to mean pooling
            selected_pred = graph_node_preds.mean(dim=0)
        
        graph_preds.append(selected_pred)
    
    return torch.stack(graph_preds)


def train_lipophilicity_model(data_list, smiles_list, 
                             epochs=5000, 
                             batch_size=128,
                             lr=0.001,
                             feature_dim=35, 
                             hidden_dim=28, 
                             early_stopping_patience=500,
                             heads=1,
                             dropout=0,
                             base_dir="",
                             device_str=None,
                             use_lr_scheduler=True,
                             use_no_pooling=False,
                             pooling_strategy='mean',
                             clip_grad_norm=1.0):
    """
    Train the TChemGNN model aligned with the paper specifications.
    
    Parameters:
    - data_list: List of PyTorch Geometric Data objects
    - smiles_list: List of SMILES strings
    - epochs: Maximum number of training epochs (5000 in paper)
    - batch_size: Batch size for training
    - lr: Learning rate (paper uses RMSprop optimizer)
    - feature_dim: Dimension of atom features (35 in paper, logP excluded)
    - hidden_dim: Dimension of hidden layers (28 in paper)
    - early_stopping_patience: Number of epochs to wait before early stopping
    - heads: Number of attention heads in GAT (1 in paper)
    - dropout: Dropout probability (0 in paper)
    - base_dir: Base directory for outputs
    - device_str: Optional string to specify device
    - use_lr_scheduler: Whether to use learning rate scheduler
    - use_no_pooling: Whether to use no-pooling approach (True for ESOL/FreeSolv)
    - pooling_strategy: Strategy for pooling ('mean', 'last', 'first')
    - clip_grad_norm: Maximum gradient norm for clipping (0 to disable)
    
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

    # Split data into training, validation and testing sets (80/10/10 as per paper)
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
    
    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)
    
    # Create model aligned with paper architecture
    from models.gsr import GSR
    model = GSR(in_channels=feature_dim, 
                hidden_channels=hidden_dim, 
                out_channels=1, 
                heads=heads, 
                dropout=dropout).to(device)
    
    # Use RMSprop optimizer as specified in the paper
    optimizer = optim.RMSprop(model.parameters(), lr=lr)
    
    # Learning rate scheduler
    if use_lr_scheduler:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=int(epochs/30))
        print("Using ReduceLROnPlateau learning rate scheduler")
    else:
        scheduler = None
        print("Learning rate scheduler disabled")
    
    criterion = nn.MSELoss()
    
    # Determine node selection strategy
    if pooling_strategy == 'last':
        node_selection = 'last'
        print("Using last node prediction (no pooling)")
    elif pooling_strategy == 'first':
        node_selection = 'first'
        print("Using first node prediction")
    else:
        node_selection = 'mean'
        print("Using mean pooling")
    
    # Initialize variables for early stopping
    best_val_loss = float('inf')
    early_stopping_counter = 0
    best_epoch = 0
    
    # Training loop
    print(f"Starting training for {epochs} epochs...")
    print(f"Pooling strategy: {pooling_strategy}")
    pbar = tqdm(range(1, epochs + 1), desc="Training Progress", position=0)
    
    for epoch in pbar:
        model.train()
        epoch_start_time = time.time()
        train_loss = 0
        train_rmse = 0
        
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            
            node_outputs, graph_outputs = model(batch)
            
            # Use appropriate pooling strategy
            if node_selection != 'mean':
                # For no-pooling approaches, extract predictions differently
                graph_outputs = get_node_predictions_for_graphs(
                    node_outputs, batch.batch, node_selection
                )
            
            loss = criterion(graph_outputs, batch.y)
            
            loss.backward()
            
            # Gradient clipping if specified
            if clip_grad_norm > 0:
                clip_grad_norm_(model.parameters(), clip_grad_norm)
            
            optimizer.step()
            
            # Calculate RMSE
            from .evaluation import compute_rmse
            batch_rmse = compute_rmse(graph_outputs, batch.y)
            
            train_loss += loss.item() * batch.num_graphs
            train_rmse += batch_rmse.item() * batch.num_graphs
            
        # Calculate average training metrics
        avg_train_loss = train_loss / len(train_loader.dataset)
        avg_train_rmse = train_rmse / len(train_loader.dataset)

        # Evaluate on validation set
        model.eval()
        val_loss = 0
        val_rmse = 0
        
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                node_outputs, graph_outputs = model(batch)
                
                # Use appropriate pooling strategy
                if node_selection != 'mean':
                    graph_outputs = get_node_predictions_for_graphs(
                        node_outputs, batch.batch, node_selection
                    )
                
                loss = criterion(graph_outputs, batch.y)
                rmse = compute_rmse(graph_outputs, batch.y)
                
                val_loss += loss.item() * batch.num_graphs
                val_rmse += rmse.item() * batch.num_graphs
        
        val_loss = val_loss / len(val_loader.dataset)
        val_rmse = val_rmse / len(val_loader.dataset)
        
        # Update learning rate scheduler
        if use_lr_scheduler:
            scheduler.step(val_loss)
            
        # Log metrics at every epoch
        log_metrics_to_csv(epoch, avg_train_loss, avg_train_rmse, val_loss, val_rmse, log_dir=log_dir)
        
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
        
        # Log detailed metrics every 50 epochs
        if epoch % 50 == 0:
            model.eval()
            
            # Create epoch directory
            epoch_dir = os.path.join(log_dir, f'epoch_{epoch}')
            os.makedirs(epoch_dir, exist_ok=True)
            
            # Get predictions for all datasets
            with torch.no_grad():
                # Testing
                test_loss = 0
                test_rmse = 0
                
                for batch in test_loader:
                    batch = batch.to(device)
                    node_outputs, graph_outputs = model(batch)
                    
                    # Use appropriate pooling strategy
                    if node_selection != 'mean':
                        graph_outputs = get_node_predictions_for_graphs(
                            node_outputs, batch.batch, node_selection
                        )
                    
                    batch_test_loss = criterion(graph_outputs, batch.y)
                    test_loss += batch_test_loss.item() * batch.num_graphs
                    
                    # Calculate RMSE for test
                    test_rmse_val = compute_rmse(graph_outputs, batch.y)
                    test_rmse += test_rmse_val.item() * batch.num_graphs
                
                avg_test_loss = test_loss / len(test_loader.dataset)
                avg_test_rmse = test_rmse / len(test_loader.dataset)
                pbar.write(f"Epoch {epoch} - Test Loss: {avg_test_loss:.4f}, Test RMSE: {avg_test_rmse:.4f}")
                
                # Update metrics CSV with test metrics
                log_metrics_to_csv(epoch, avg_train_loss, avg_train_rmse, val_loss, val_rmse, 
                                  avg_test_loss, avg_test_rmse, log_dir)
            
            # Get full dataset predictions for logging and visualization
            pbar.write(f"Generating visualizations for epoch {epoch}...")
            
            # Evaluate and log predictions
            _, _, train_graph_preds, train_node_preds, train_node_batch, train_targets = evaluate_model_with_nodes(
                model, DataLoader(train_data, batch_size=batch_size), criterion, device
            )
            
            _, _, val_graph_preds, val_node_preds, val_node_batch, val_targets = evaluate_model_with_nodes(
                model, DataLoader(val_data, batch_size=batch_size), criterion, device
            )
            
            _, _, test_graph_preds, test_node_preds, test_node_batch, test_targets = evaluate_model_with_nodes(
                model, DataLoader(test_data, batch_size=batch_size), criterion, device
            )
            
            # Log detailed predictions to CSV
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
    
    # Final evaluation
    print("\nPerforming final evaluation on the best model...")
    
    # Final evaluation code similar to above...
    train_loss, train_rmse, _, _, _, _ = evaluate_model_with_nodes(
        model, DataLoader(train_data, batch_size=batch_size), criterion, device
    )
    
    val_loss, val_rmse, _, _, _, _ = evaluate_model_with_nodes(
        model, DataLoader(val_data, batch_size=batch_size), criterion, device
    )
    
    test_loss, test_rmse, _, _, _, _ = evaluate_model_with_nodes(
        model, DataLoader(test_data, batch_size=batch_size), criterion, device
    )
    
    print("\nFinal Evaluation (Best Model):")
    print(f"  Train Loss: {train_loss:.4f} | Train RMSE: {train_rmse:.4f}")
    print(f"  Val Loss: {val_loss:.4f} | Val RMSE: {val_rmse:.4f}")
    print(f"  Test Loss: {test_loss:.4f} | Test RMSE: {test_rmse:.4f}")
    
    # Create final visualizations
    visualize_results(log_dir, results_dir, best_epoch)
    
    return model, log_dir, results_dir, checkpoints_dir