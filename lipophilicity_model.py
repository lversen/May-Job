#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Lipophilicity Prediction Model using Graph Attention Networks

This script implements a Graph Neural Network model for predicting the lipophilicity
of molecules represented as SMILES strings. The model uses Graph Attention Networks (GAT)
from PyTorch Geometric.
"""

import os
import csv
import argparse
from typing import List, Tuple, Dict, Any, Optional

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import Data, Dataset
from torch_geometric.nn import GATConv
from torch_geometric.utils import add_self_loops
from sklearn.model_selection import train_test_split

# RDKit imports for molecule processing
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, AllChem, Descriptors, Draw
from rdkit.Chem.rdMolDescriptors import CalcMolFormula


# Define a dictionary mapping atomic numbers to electronegativity values (Pauling scale)
ELECTRONEGATIVITY_DICT = {
    1: 2.20,   # Hydrogen
    6: 2.55,   # Carbon
    7: 3.04,   # Nitrogen
    8: 3.44,   # Oxygen
    9: 3.98,   # Fluorine
    16: 2.58,  # Sulfur
    17: 2.96,  # Chlorine
    35: 2.66,  # Bromine
    53: 2.66,  # Iodine
    11: 0.93,  # Sodium
    12: 1.31,  # Magnesium
}


class MoleculeDataset(Dataset):
    """Custom dataset class for molecular graph data."""
    
    def __init__(self, smiles_list: List[str], y_values: List[float], 
                 additional_features: Dict[str, List[float]]):
        """
        Initialize the dataset.
        
        Args:
            smiles_list: List of SMILES strings representing molecules
            y_values: Target lipophilicity values
            additional_features: Dictionary of additional features (volumes, widths, etc.)
        """
        super().__init__(None, None)
        
        # Validate input data
        self._validate_data(smiles_list, y_values, additional_features)
        
        self.smiles_list = smiles_list
        self.y_values = y_values
        self.additional_features = additional_features
        self.data_list = self._prepare_data()
    
    def _validate_data(self, smiles_list: List[str], y_values: List[float], 
                      additional_features: Dict[str, List[float]]):
        """
        Validate that all data has consistent lengths.
        
        Args:
            smiles_list: List of SMILES strings
            y_values: List of target values
            additional_features: Dictionary of additional features
        
        Raises:
            ValueError: If data lengths don't match
        """
        # Check if SMILES and y values have the same length
        if len(smiles_list) != len(y_values):
            raise ValueError(f"Number of SMILES strings ({len(smiles_list)}) does not match "
                             f"number of target values ({len(y_values)})")
        
        # Check if all additional feature lists have the same length as SMILES list
        for feature_name, feature_list in additional_features.items():
            if len(feature_list) != len(smiles_list):
                raise ValueError(f"Length of {feature_name} ({len(feature_list)}) does not match "
                                 f"number of SMILES strings ({len(smiles_list)})")
    
    def _prepare_data(self) -> List[Data]:
        """Process SMILES strings and create graph data objects."""
        x_features = self._extract_node_features()
        edge_indices = self._extract_edge_indices()
        
        data_list = []
        for i in range(len(self.smiles_list)):
            x_graph = torch.tensor(x_features[i], dtype=torch.float32)
            edge_index_graph = torch.tensor(edge_indices[i], dtype=torch.long).t().contiguous()
            y_graph = torch.tensor([[self.y_values[i]]], dtype=torch.float32)
            
            data_graph = Data(x=x_graph, edge_index=edge_index_graph, y=y_graph)
            data_list.append(data_graph)
            
        return data_list
    
    def _extract_edge_indices(self) -> List[List[List[int]]]:
        """Extract edge indices from molecules for graph construction."""
        all_edge_indexes = []
        
        for smiles in self.smiles_list:
            # Convert SMILES string to RDKit Mol object
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Could not convert SMILES string to molecule: {smiles}")
                
            # Get bonds information
            bonds = mol.GetBonds()
            
            # Initialize edge indexes list for this SMILES
            edge_indexes = []
            
            # Iterate through bonds to get the edge indexes
            for bond in bonds:
                begin_idx = bond.GetBeginAtomIdx()
                end_idx = bond.GetEndAtomIdx()
                edge_indexes.append([begin_idx, end_idx])
                edge_indexes.append([end_idx, begin_idx])  # Add reverse direction
                
            # Append the edge indexes for this SMILES to the overall list
            all_edge_indexes.append(edge_indexes)
            
        return all_edge_indexes
    
    def _extract_node_features(self) -> List[List[List[float]]]:
        """Extract atom features for each molecule."""
        x_features = []
        
        for idx, smiles in enumerate(self.smiles_list):
            # Convert SMILES to an RDKit molecule object
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Could not convert SMILES string to molecule: {smiles}")
                
            # Calculate global features for the molecule
            global_features = self._calculate_global_features(mol)
            
            # Initialize an empty list to store atom features
            atom_features = []
            
            # Iterate through each atom in the molecule
            for atom in mol.GetAtoms():
                # Get atom-specific features
                features = self._get_atom_features(atom, mol)
                
                # Add global molecular features to each atom's features
                features.extend(global_features)
                
                # Add additional features from provided lists
                for feature_name in self.additional_features:
                    features.append(self.additional_features[feature_name][idx])
                
                # Append the atom's features to the list
                atom_features.append(features)
                
            # Append features for this molecule
            x_features.append(atom_features)
            
        return x_features
    
    def _calculate_global_features(self, mol) -> List[float]:
        """Calculate global molecular features."""
        # Basic molecular properties
        min_degree = min(atom.GetDegree() for atom in mol.GetAtoms())
        num_hbond_donors = sum(1 for atom in mol.GetAtoms() 
                              if atom.GetSymbol() == 'O' and atom.GetTotalNumHs() > 0)
        num_rings = len(mol.GetRingInfo().AtomRings())
        num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        polar_surface_area = rdMolDescriptors.CalcTPSA(mol)
        
        # Additional molecular properties
        molecular_weight = rdMolDescriptors.CalcExactMolWt(mol)
        num_atoms = mol.GetNumAtoms()
        hba = rdMolDescriptors.CalcNumHBA(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        fraction_sp2 = sum(1 for atom in mol.GetAtoms() 
                         if atom.GetHybridization() == Chem.HybridizationType.SP2) / num_atoms
        
        # Global properties related to atom characteristics
        valence = sum(atom.GetTotalValence() for atom in mol.GetAtoms())
        
        # General electronegativity: Calculate the average electronegativity of all atoms
        total_electronegativity = sum(ELECTRONEGATIVITY_DICT.get(atom.GetAtomicNum(), 0) 
                                     for atom in mol.GetAtoms())
        general_electronegativity = total_electronegativity / num_atoms if num_atoms > 0 else 0
        
        return [
            min_degree,
            num_hbond_donors,
            num_rings,
            num_rotatable_bonds,
            polar_surface_area,
            molecular_weight,
            num_atoms,
            hba,
            hbd,
            fraction_sp2,
            valence,
            general_electronegativity,
        ]
    
    def _get_atom_features(self, atom, mol) -> List[float]:
        """Extract features for a specific atom."""
        # Global properties related to atom's characteristics
        is_in_ring = int(len(mol.GetRingInfo().AtomRings()) > 0)
        is_aromatic = int(any(a.GetIsAromatic() for a in mol.GetAtoms()))
        formal_charge = sum(a.GetFormalCharge() for a in mol.GetAtoms())
        
        # Define atom-specific features
        features = [
            is_in_ring,            # Whether the molecule contains any rings
            is_aromatic,           # Whether the molecule is aromatic
            formal_charge,         # Global formal charge of the molecule
            int(atom.GetDegree()), # Atomic degree (number of bonds)
            int(atom.GetAtomicNum()),  # Atomic number
            int(atom.GetTotalNumHs()), # Number of hydrogens
            int(atom.GetTotalValence()), # Atomic valence
            int(atom.GetNumRadicalElectrons()),  # Number of radical electrons
            int(atom.GetFormalCharge()), # Formal charge on the atom
            int(atom.GetHybridization()), # Hybridization of the atom
            int((atom.GetMass() - 10.812) / 116.092),  # Normalized mass feature
            int((atom.GetAtomicNum() - 1.5) / 0.6),
            int((atom.GetAtomicNum() - 0.64) / 0.76),
            ELECTRONEGATIVITY_DICT.get(atom.GetAtomicNum(), 0),  # Electronegativity
            int(atom.GetAtomicNum() in ELECTRONEGATIVITY_DICT),  # Whether the atom has defined electronegativity
            int(atom.GetNumImplicitHs() > 0),  # If the atom has implicit hydrogens
        ]
        
        # Add hydroxyl group feature
        if atom.GetSymbol() == 'O' and any(neighbor.GetSymbol() == 'H' for neighbor in atom.GetNeighbors()):
            features.append(1)  # Hydroxyl group (-OH) presence
        else:
            features.append(0)
            
        return features
    
    def len(self) -> int:
        """Return the number of graphs in the dataset."""
        return len(self.data_list)
    
    def get(self, idx: int) -> Data:
        """Get a specific graph by index."""
        return self.data_list[idx]


class GSR(nn.Module):
    """Graph Structural Regression (GSR) model using Graph Attention Networks."""
    
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int):
        """
        Initialize the GSR model.
        
        Args:
            in_channels: Number of input features per node
            hidden_channels: Number of hidden features per node
            out_channels: Number of output features per node
        """
        super(GSR, self).__init__()
        self.conv1 = GATConv(in_channels, hidden_channels)
        self.conv2 = GATConv(hidden_channels, hidden_channels)
        self.conv3 = GATConv(hidden_channels, hidden_channels)
        self.conv4 = GATConv(hidden_channels, hidden_channels)
        self.conv5 = GATConv(hidden_channels, out_channels)  # Output layer
        
    def forward(self, data: Data) -> torch.Tensor:
        """
        Forward pass through the network.
        
        Args:
            data: Graph data object containing node features and edge indices
            
        Returns:
            Node feature tensor after message passing
        """
        x, edge_index = data.x, data.edge_index
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
        
        # Apply the graph convolutional layers with activation functions
        x = torch.tanh(self.conv1(x, edge_index))
        x = torch.tanh(self.conv2(x, edge_index))
        x = torch.tanh(self.conv3(x, edge_index))
        x = torch.tanh(self.conv4(x, edge_index))
        x = self.conv5(x, edge_index)  # No activation on final layer
        
        return x


def load_data(csv_path: str) -> Tuple[List[str], List[float]]:
    """
    Load molecular data from CSV file.
    
    Args:
        csv_path: Path to the CSV file containing SMILES and experimental values
        
    Returns:
        Tuple of (SMILES strings list, experimental values list)
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Check if required columns exist
        if 'smiles' not in df.columns or 'exp' not in df.columns:
            raise ValueError("CSV file must contain 'smiles' and 'exp' columns")
            
        # Remove problematic rows if needed (like in the notebook where row 1562 was removed)
        # This is now optional and controlled by parameters
        
        # Extract SMILES and experimental values
        smiles_list = list(df['smiles'])
        exp_values = list(df['exp'])
        
        return smiles_list, exp_values
    
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        raise


def load_additional_features(feature_paths: Dict[str, str], expected_length: int) -> Dict[str, List[float]]:
    """
    Load additional molecular features from CSV files and validate their length.
    
    Args:
        feature_paths: Dictionary mapping feature names to file paths
        expected_length: Expected length of feature lists (should match SMILES list length)
        
    Returns:
        Dictionary mapping feature names to feature value lists
    """
    features = {}
    
    for feature_name, file_path in feature_paths.items():
        try:
            if not os.path.exists(file_path):
                print(f"Warning: Feature file {file_path} not found. Using default values.")
                # Create a list of zeros with the expected length as fallback
                features[feature_name] = [0.0] * expected_length
                continue
            
            feature_df = pd.read_csv(file_path, header=None)
            feature_values = list(feature_df.to_numpy().flatten())
            
            # Validate length
            if len(feature_values) < expected_length:
                print(f"Warning: {feature_name} has fewer values ({len(feature_values)}) than SMILES strings ({expected_length}).")
                print(f"Padding with zeros to match length.")
                # Pad with zeros to match expected length
                feature_values = feature_values + [0.0] * (expected_length - len(feature_values))
            elif len(feature_values) > expected_length:
                print(f"Warning: {feature_name} has more values ({len(feature_values)}) than SMILES strings ({expected_length}).")
                print(f"Truncating to match length.")
                # Truncate to match expected length
                feature_values = feature_values[:expected_length]
                
            features[feature_name] = feature_values
        except Exception as e:
            print(f"Error loading {feature_name} from {file_path}: {str(e)}")
            print(f"Using default values instead.")
            # Create a list of zeros with the expected length as fallback
            features[feature_name] = [0.0] * expected_length
            
    return features


def compute_rmse(predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """
    Compute Root Mean Square Error between predictions and targets.
    
    Args:
        predictions: Predicted values
        targets: Target values
        
    Returns:
        RMSE value as a tensor
    """
    mse = ((predictions - targets) ** 2).mean()
    rmse = torch.sqrt(mse)
    return rmse


# Add to Data class for proper device handling
def move_to_device(data, device):
    """Helper function to move a PyG Data object to a specific device."""
    for key, item in data:
        if torch.is_tensor(item):
            data[key] = item.to(device)
    return data


def evaluate_model(model: nn.Module, 
                data_loader: List[Data] or torch.utils.data.DataLoader, 
                criterion: nn.Module,
                device: torch.device,
                use_batching: bool = False) -> Tuple[float, float]:
    """
    Evaluate the model on a dataset.
    
    Args:
        model: The model to evaluate
        data_loader: Data to evaluate on (either a list of Data objects or a DataLoader)
        criterion: Loss function
        device: Device to use for evaluation
        use_batching: Whether the data_loader uses batching
        
    Returns:
        Tuple of (average loss, average RMSE)
    """
    model.eval()
    total_loss = 0
    total_rmse = 0
    total_samples = 0
    
    with torch.no_grad():
        if use_batching:
            # Batched evaluation
            for batch in data_loader:
                # Move batch to device
                batch = batch.to(device)
                
                # Forward pass
                output = model(batch)
                loss = criterion(output, batch.y)
                
                # Calculate metrics
                total_loss += loss.item() * batch.num_graphs
                rmse = compute_rmse(output, batch.y)
                total_rmse += rmse.item() * batch.num_graphs
                total_samples += batch.num_graphs
        else:
            # Non-batched evaluation
            for data_graph in data_loader:
                # Move data to device
                data_graph = data_graph.to(device)
                
                # Forward pass
                output = model(data_graph)
                loss = criterion(output, data_graph.y)
                
                # Calculate metrics
                total_loss += loss.item()
                rmse = compute_rmse(output, data_graph.y)
                total_rmse += rmse.item()
                total_samples += 1
    
    # Calculate averages
    avg_loss = total_loss / total_samples
    avg_rmse = total_rmse / total_samples
    
    return avg_loss, avg_rmse


def evaluate_model_with_predictions(model: nn.Module, 
                                   data_loader: List[Data] or torch.utils.data.DataLoader, 
                                   criterion: nn.Module,
                                   device: torch.device,
                                   use_batching: bool = False) -> Tuple[float, float, List[Tuple[float, float]]]:
    """
    Evaluate the model and return predictions.
    
    Args:
        model: The model to evaluate
        data_loader: Data to evaluate on
        criterion: Loss function
        device: Device to use for evaluation
        use_batching: Whether the data_loader uses batching
        
    Returns:
        Tuple of (average loss, average RMSE, list of (y_true, y_pred) pairs)
    """
    model.eval()
    total_loss = 0
    total_rmse = 0
    total_samples = 0
    predictions = []
    
    with torch.no_grad():
        if use_batching:
            # Batched evaluation
            for batch in data_loader:
                # Move batch to device
                batch = batch.to(device)
                
                # Forward pass
                output = model(batch)
                loss = criterion(output, batch.y)
                
                # Calculate metrics
                total_loss += loss.item() * batch.num_graphs
                rmse = compute_rmse(output, batch.y)
                total_rmse += rmse.item() * batch.num_graphs
                total_samples += batch.num_graphs
                
                # Store predictions
                for y_true, y_pred in zip(batch.y.cpu().numpy().flatten(), output.cpu().numpy().flatten()):
                    predictions.append((float(y_true), float(y_pred)))
        else:
            # Non-batched evaluation
            for data_graph in data_loader:
                # Move data to device
                data_graph = data_graph.to(device)
                
                # Forward pass
                output = model(data_graph)
                loss = criterion(output, data_graph.y)
                
                # Calculate metrics
                total_loss += loss.item()
                rmse = compute_rmse(output, data_graph.y)
                total_rmse += rmse.item()
                total_samples += 1
                
                # Store predictions
                for y_true, y_pred in zip(data_graph.y.cpu().numpy().flatten(), output.cpu().numpy().flatten()):
                    predictions.append((float(y_true), float(y_pred)))
    
    # Calculate averages
    avg_loss = total_loss / total_samples
    avg_rmse = total_rmse / total_samples
    
    return avg_loss, avg_rmse, predictions


def train_model(model: nn.Module, 
                train_loader: List[Data] or torch.utils.data.DataLoader, 
                val_loader: List[Data] or torch.utils.data.DataLoader, 
                test_loader: List[Data] or torch.utils.data.DataLoader, 
                optimizer: optim.Optimizer, 
                criterion: nn.Module, 
                num_epochs: int, 
                device: torch.device,
                log_dir: str = 'logs', 
                eval_interval: int = 50,
                use_batching: bool = False,
                mixed_precision: bool = False,
                gradient_accumulation_steps: int = 1,
                early_stopping_patience: int = 20):
    """
    Train the model and evaluate at intervals.
    
    Args:
        model: The model to train
        train_loader: Training data (either a list of Data objects or a DataLoader)
        val_loader: Validation data (either a list of Data objects or a DataLoader)
        test_loader: Test data (either a list of Data objects or a DataLoader)
        optimizer: Optimizer
        criterion: Loss function
        num_epochs: Number of epochs to train for
        device: Device to use for training (CPU or GPU)
        log_dir: Directory to save logs
        eval_interval: Interval (in epochs) to evaluate and log metrics
        use_batching: Whether to use batching for training
        mixed_precision: Whether to use mixed precision training (only for GPU)
        gradient_accumulation_steps: Number of steps to accumulate gradients
        early_stopping_patience: Number of evaluation intervals without improvement before stopping
    """
    # Create directory for logs if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Setup CSV file for logging metrics
    metrics_file = open(os.path.join(log_dir, 'metrics.csv'), 'w', newline='')
    metrics_writer = csv.writer(metrics_file)
    metrics_writer.writerow(['Epoch', 'Train_Loss', 'Train_RMSE', 
                            'Val_Loss', 'Val_RMSE', 
                            'Test_Loss', 'Test_RMSE'])
    
    # Move model to the selected device
    model = model.to(device)
    
    # Set up mixed precision training if requested (only for GPU)
    scaler = torch.cuda.amp.GradScaler() if mixed_precision and device.type == 'cuda' else None
    
    # Early stopping variables
    best_val_rmse = float('inf')
    best_model_state = None
    patience_counter = 0
    
    # Training loop
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        total_train_rmse = 0
        train_samples = 0
        
        # Reset gradients at the beginning of each epoch
        optimizer.zero_grad()
        
        # Training
        if use_batching:
            # Batched training loop
            for batch_idx, batch in enumerate(train_loader):
                # Move batch to device
                batch = batch.to(device)
                
                # Mixed precision context if enabled
                with torch.cuda.amp.autocast() if mixed_precision and device.type == 'cuda' else contextlib.nullcontext():
                    output = model(batch)
                    loss = criterion(output, batch.y) / gradient_accumulation_steps
                
                # Mixed precision backward pass if enabled
                if mixed_precision and device.type == 'cuda':
                    scaler.scale(loss).backward()
                    
                    # Gradient accumulation
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()
                else:
                    loss.backward()
                    
                    # Gradient accumulation
                    if (batch_idx + 1) % gradient_accumulation_steps == 0:
                        optimizer.step()
                        optimizer.zero_grad()
                
                # Calculate metrics (non-reduced loss)
                total_loss += loss.item() * gradient_accumulation_steps * batch.num_graphs
                rmse = compute_rmse(output, batch.y)
                total_train_rmse += rmse.item() * batch.num_graphs
                train_samples += batch.num_graphs
        else:
            # Non-batched training loop (one sample at a time)
            for i, data_graph in enumerate(train_loader):
                # Move data to device
                data_graph = data_graph.to(device)
                
                # Reset gradients for each sample
                optimizer.zero_grad()
                
                # Forward pass
                with torch.cuda.amp.autocast() if mixed_precision and device.type == 'cuda' else contextlib.nullcontext():
                    output = model(data_graph)
                    loss = criterion(output, data_graph.y)
                
                # Backward pass
                if mixed_precision and device.type == 'cuda':
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                
                # Calculate metrics
                total_loss += loss.item()
                rmse = compute_rmse(output, data_graph.y)
                total_train_rmse += rmse.item()
                train_samples += 1
        
        # Calculate average metrics for the epoch
        avg_train_loss = total_loss / train_samples
        avg_train_rmse = total_train_rmse / train_samples
        
        # Log progress
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_train_loss:.4f}, RMSE: {avg_train_rmse:.4f}")
        
        # Evaluate on validation and test data at intervals
        if (epoch + 1) % eval_interval == 0:
            # Create epoch-specific CSV files for predictions
            y_test_file = open(os.path.join(log_dir, f'y_test_epoch_{epoch+1}.csv'), 'w', newline='')
            y_pred_file = open(os.path.join(log_dir, f'y_pred_epoch_{epoch+1}.csv'), 'w', newline='')
            y_test_writer = csv.writer(y_test_file)
            y_pred_writer = csv.writer(y_pred_file)
            
            # Write headers
            y_test_writer.writerow(['Sample_ID', 'y_test'])
            y_pred_writer.writerow(['Sample_ID', 'y_pred'])
            
            # Evaluate on validation set
            val_loss, val_rmse = evaluate_model(model, val_loader, criterion, device, use_batching)
            
            # Evaluate on test set and log predictions
            test_loss, test_rmse, test_predictions = evaluate_model_with_predictions(
                model, test_loader, criterion, device, use_batching
            )
            
            # Write predictions to CSV
            for i, (y_test, y_pred) in enumerate(test_predictions):
                sample_id = f'test_{i}'
                y_test_writer.writerow([sample_id, y_test])
                y_pred_writer.writerow([sample_id, y_pred])
            
            # Close prediction files
            y_test_file.close()
            y_pred_file.close()
            
            # Write metrics to CSV
            metrics_writer.writerow([epoch+1, avg_train_loss, avg_train_rmse, 
                                    val_loss, val_rmse, 
                                    test_loss, test_rmse])
            metrics_file.flush()  # Ensure data is written to file
            
            # Print evaluation metrics
            print(f"Epoch {epoch+1} Evaluation:")
            print(f"  Training:    Loss = {avg_train_loss:.4f}, RMSE = {avg_train_rmse:.4f}")
            print(f"  Validation:  Loss = {val_loss:.4f}, RMSE = {val_rmse:.4f}")
            print(f"  Test:        Loss = {test_loss:.4f}, RMSE = {test_rmse:.4f}")
            
            # Early stopping check
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                best_model_state = model.state_dict().copy()
                patience_counter = 0
                
                # Save the best model
                torch.save(model.state_dict(), os.path.join(log_dir, 'best_model.pt'))
            else:
                patience_counter += 1
                
            if patience_counter >= early_stopping_patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load the best model if early stopping was triggered
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Close the metrics file
    metrics_file.close()
    
    print("Training completed!")
    
    return model


def main():
    """Main function to run the training pipeline."""
    parser = argparse.ArgumentParser(description='Train a Graph Neural Network for lipophilicity prediction')
    
    # Data parameters
    parser.add_argument('--data', type=str, default='Lipophilicity!.csv',
                       help='Path to the CSV file containing SMILES and experimental values')
    parser.add_argument('--volumes', type=str, default='volumes.csv',
                       help='Path to the CSV file containing volume features')
    parser.add_argument('--widths', type=str, default='widths.csv',
                       help='Path to the CSV file containing width features')
    parser.add_argument('--lengths', type=str, default='lengths.csv',
                       help='Path to the CSV file containing length features')
    parser.add_argument('--heights', type=str, default='heights.csv',
                       help='Path to the CSV file containing height features')
    parser.add_argument('--dipole_momentum', type=str, default='dipole_momentum.csv',
                       help='Path to the CSV file containing dipole momentum features')
    parser.add_argument('--angles', type=str, default='angles.csv',
                       help='Path to the CSV file containing angle features')
    
    # Model parameters
    parser.add_argument('--in_channels', type=int, default=35,
                       help='Number of input channels (feature dimensions)')
    parser.add_argument('--hidden_channels', type=int, default=28,
                       help='Number of hidden channels in the model')
    parser.add_argument('--out_channels', type=int, default=1,
                       help='Number of output channels (prediction dimensions)')
    
    # Training parameters
    parser.add_argument('--lr', type=float, default=0.00075,
                       help='Learning rate for the optimizer')
    parser.add_argument('--epochs', type=int, default=1250,
                       help='Number of epochs to train for')
    parser.add_argument('--eval_interval', type=int, default=50,
                       help='Interval (in epochs) to evaluate the model')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed for reproducibility')
    parser.add_argument('--train_ratio', type=float, default=0.8,
                       help='Ratio of data to use for training')
    parser.add_argument('--val_ratio', type=float, default=0.1,
                       help='Ratio of data to use for validation')
    
    # Output parameters
    parser.add_argument('--log_dir', type=str, default='logs',
                       help='Directory to save logs and results')
    
    # Device and performance parameters
    parser.add_argument('--device', type=str, default='auto',
                       help='Device to use for training (cpu, cuda, or auto for automatic selection)')
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training. Set to 0 for no batching.')
    parser.add_argument('--mixed_precision', action='store_true',
                       help='Use mixed precision training (only effective with CUDA)')
    parser.add_argument('--gradient_accumulation', type=int, default=1,
                       help='Number of steps to accumulate gradients over')
    parser.add_argument('--num_workers', type=int, default=0,
                       help='Number of worker processes for data loading')
    
    # Early stopping parameters
    parser.add_argument('--early_stopping', action='store_true',
                       help='Enable early stopping')
    parser.add_argument('--patience', type=int, default=5,
                       help='Number of evaluations without improvement before early stopping')
    
    # Data handling options
    parser.add_argument('--remove_problematic_rows', action='store_true',
                       help='Remove problematic rows from the dataset (like row 1562 in the notebook)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print verbose output during processing')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Load data
    print(f"Loading data from {args.data}...")
    try:
        df = pd.read_csv(args.data)
        
        # Check if required columns exist
        if 'smiles' not in df.columns or 'exp' not in df.columns:
            raise ValueError(f"CSV file must contain 'smiles' and 'exp' columns. Found columns: {df.columns.tolist()}")
        
        # Remove problematic rows if specified
        if args.remove_problematic_rows and len(df) > 1561:
            print("Removing problematic row 1562 (index 1561)...")
            df = df.drop(1561, axis=0)
            df = df.reset_index(drop=True)
        
        # Extract SMILES and experimental values
        smiles_list = list(df['smiles'])
        exp_values = list(df['exp'])
        
        print(f"Loaded {len(smiles_list)} molecules with experimental values.")
    except Exception as e:
        print(f"Error loading data: {str(e)}")
        return
    
    # Load additional features
    print("Loading additional molecular features...")
    feature_paths = {
        'volumes': args.volumes,
        'widths': args.widths,
        'lengths': args.lengths,
        'heights': args.heights,
        'dipole_momentum': args.dipole_momentum,
        'angles': args.angles
    }
    additional_features = load_additional_features(feature_paths, len(smiles_list))
    
    # Create dataset
    print("Creating molecular dataset...")
    try:
        dataset = MoleculeDataset(smiles_list, exp_values, additional_features)
        print(f"Successfully created dataset with {len(dataset)} molecules.")
    except Exception as e:
        print(f"Error creating dataset: {str(e)}")
        return
    
    # Split data into train, validation, and test sets
    train_size = int(args.train_ratio * len(dataset))
    val_size = int(args.val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size
    
    try:
        indices = list(range(len(dataset)))
        # Shuffle indices for random split
        np.random.shuffle(indices)
        
        train_indices = indices[:train_size]
        val_indices = indices[train_size:train_size + val_size]
        test_indices = indices[train_size + val_size:]
        
        train_data = [dataset[i] for i in train_indices]
        val_data = [dataset[i] for i in val_indices]
        test_data = [dataset[i] for i in test_indices]
        
        print(f"Data split: {len(train_data)} training, {len(val_data)} validation, {len(test_data)} test")
    except Exception as e:
        print(f"Error splitting dataset: {str(e)}")
        return
    
    # Initialize model, optimizer, and loss function
    print("Initializing model...")
    model = GSR(in_channels=args.in_channels, 
               hidden_channels=args.hidden_channels, 
               out_channels=args.out_channels)
    
    optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()
    
    # Create log directory
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Train the model
    print(f"Starting training for {args.epochs} epochs...")
    try:
        train_model(model, train_data, val_data, test_data, 
                   optimizer, criterion, args.epochs, 
                   log_dir=args.log_dir, eval_interval=args.eval_interval)
    except Exception as e:
        print(f"Error during training: {str(e)}")
        return
    
    # Save the trained model
    try:
        torch.save(model.state_dict(), os.path.join(args.log_dir, 'lipophilicity_model.pt'))
        print(f"Model saved to {os.path.join(args.log_dir, 'lipophilicity_model.pt')}")
    except Exception as e:
        print(f"Error saving model: {str(e)}")
        return
    
    print("Training completed successfully!")


if __name__ == '__main__':
    main()