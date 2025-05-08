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
                             dropout=0.2,
                             base_dir=""):
    """
    Train a lipophilicity prediction model with logging similar to the stable version.
    
    Parameters remain the same as in the original function.
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
        
        # Print basic epoch progress (similar to stable version)
        print(epoch, train_loss, train_rmse)
        
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
                
                for i, data_batch in enumerate(test_loader):
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
                        
                        print(f"element {i*batch_size+j}, Epoch {epoch}: Test Loss: {ind_test_loss:.4f}")
                        print(f"element {i*batch_size+j}, Epoch {epoch}: Test RMSE: {ind_test_rmse:.4f}")
                        print(f'y_test: {sample_test_target.tolist()}, y_pred: {sample_test_pred.tolist()}')
                
                avg_test_loss = test_loss / len(test_loader.dataset)
                avg_test_rmse = total_test_rmse / len(test_loader.dataset)
            
            # Get full dataset predictions for logging and visualization
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