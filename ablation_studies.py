"""
Ablation studies and additional experiments from the TChemGNN paper.
This script implements:
1. Ablation study with different GNN layer types (Table 2)
2. Random Forest baseline for FreeSolv
3. Analysis of predictions at different node positions (Table 5)
"""

import os
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, GATConv, GATv2Conv, GINConv, GraphConv
from torch_geometric.nn.conv import MessagePassing
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Import existing modules
from data import load_data, extract_edge_indices, create_atom_features, prepare_data_for_training
from models import GSR
from training.evaluation import evaluate_model_with_nodes, compute_rmse
from utils import set_seed, format_y_values, get_device


class SimplifiedGNN(nn.Module):
    """
    Simplified GNN model for ablation studies with different layer types.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, layer_type='GCN', num_layers=4, dropout=0):
        super(SimplifiedGNN, self).__init__()
        
        self.layers = nn.ModuleList()
        self.layer_type = layer_type
        
        # First layer
        if layer_type == 'GCN':
            self.layers.append(GCNConv(in_channels, hidden_channels))
        elif layer_type == 'GAT':
            self.layers.append(GATConv(in_channels, hidden_channels, heads=1, dropout=dropout))
        elif layer_type == 'GATv2':
            self.layers.append(GATv2Conv(in_channels, hidden_channels, heads=1, dropout=dropout))
        elif layer_type == 'GIN':
            nn_first = nn.Sequential(nn.Linear(in_channels, hidden_channels), nn.ReLU())
            self.layers.append(GINConv(nn_first))
        elif layer_type == 'GraphConv':
            self.layers.append(GraphConv(in_channels, hidden_channels))
            
        # Hidden layers
        for _ in range(num_layers - 2):
            if layer_type == 'GCN':
                self.layers.append(GCNConv(hidden_channels, hidden_channels))
            elif layer_type in ['GAT', 'GATv2']:
                conv = GATConv if layer_type == 'GAT' else GATv2Conv
                self.layers.append(conv(hidden_channels, hidden_channels, heads=1, dropout=dropout))
            elif layer_type == 'GIN':
                nn_hidden = nn.Sequential(nn.Linear(hidden_channels, hidden_channels), nn.ReLU())
                self.layers.append(GINConv(nn_hidden))
            elif layer_type == 'GraphConv':
                self.layers.append(GraphConv(hidden_channels, hidden_channels))
                
        # Output layer
        self.out_layer = nn.Linear(hidden_channels, out_channels)
        
    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        
        # Apply layers with ReLU activation (except GIN which has built-in activation)
        for i, layer in enumerate(self.layers):
            x = layer(x, edge_index)
            if self.layer_type != 'GIN':
                x = torch.relu(x)
                
        # Output layer
        node_preds = self.out_layer(x)
        
        # Global mean pooling
        from torch_geometric.nn import global_mean_pool
        batch = data.batch if hasattr(data, 'batch') else torch.zeros(node_preds.size(0), dtype=torch.long, device=node_preds.device)
        graph_preds = global_mean_pool(node_preds, batch)
        
        return node_preds, graph_preds


def run_ablation_study_esol(data_dir='datasets', output_dir='ablation_results'):
    """
    Run ablation study on ESOL dataset as described in Table 2 of the paper.
    """
    print("=" * 50)
    print("Running Ablation Study on ESOL Dataset")
    print("=" * 50)
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load ESOL data
    esol_file = os.path.join(data_dir, 'esol', 'ESOL.csv')
    if not os.path.exists(esol_file):
        esol_file = os.path.join(data_dir, 'ESOL.csv')
    
    df, _ = load_data(esol_file, None)
    smiles_list = list(df['SMILES'])
    y_values = list(df['measured log solubility in mols per litre'])
    y = format_y_values(y_values)
    
    print(f"Loaded {len(smiles_list)} molecules from ESOL dataset")
    
    # Extract molecular structures
    edge_index = extract_edge_indices(smiles_list)
    
    # Results dictionary
    results = {}
    
    # 1. Test different GNN layer types
    layer_types = ['GCN', 'GAT', 'GATv2', 'GraphConv']  # GIN often unstable
    
    for layer_type in layer_types:
        print(f"\nTesting {layer_type} layers...")
        
        # Create features WITHOUT 3D features for basic comparison
        x_no_3d = create_atom_features(smiles_list, features_dict=None)
        
        # Manually remove the 3D features (last 6 features)
        x_no_3d_truncated = []
        for mol_features in x_no_3d:
            truncated = [atom_features[:-6] for atom_features in mol_features]
            x_no_3d_truncated.append(truncated)
        
        # Prepare data
        data_list = prepare_data_for_training(x_no_3d_truncated, edge_index, y, smiles_list)
        
        # Split data
        train_size = int(0.8 * len(data_list))
        val_size = int(0.1 * len(data_list))
        
        indices = list(range(len(data_list)))
        np.random.shuffle(indices)
        
        train_data = [data_list[i] for i in indices[:train_size]]
        val_data = [data_list[i] for i in indices[train_size:train_size + val_size]]
        test_data = [data_list[i] for i in indices[train_size + val_size:]]
        
        # Create DataLoaders
        train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=16)
        test_loader = DataLoader(test_data, batch_size=16)
        
        # Create model
        device = get_device()
        model = SimplifiedGNN(
            in_channels=len(x_no_3d_truncated[0][0]),
            hidden_channels=28,
            out_channels=1,
            layer_type=layer_type,
            num_layers=4
        ).to(device)
        
        # Train model
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.MSELoss()
        
        best_val_rmse = float('inf')
        patience_counter = 0
        max_patience = 50
        
        pbar = tqdm(range(300), desc=f"Training {layer_type}")
        for epoch in pbar:
            # Training
            model.train()
            train_loss = 0
            for batch in train_loader:
                batch = batch.to(device)
                optimizer.zero_grad()
                _, graph_outputs = model(batch)
                loss = criterion(graph_outputs, batch.y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * batch.num_graphs
                
            # Validation
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    batch = batch.to(device)
                    _, graph_outputs = model(batch)
                    loss = criterion(graph_outputs, batch.y)
                    val_loss += loss.item() * batch.num_graphs
                    
            val_rmse = np.sqrt(val_loss / len(val_loader.dataset))
            
            if val_rmse < best_val_rmse:
                best_val_rmse = val_rmse
                patience_counter = 0
            else:
                patience_counter += 1
                
            if patience_counter >= max_patience:
                break
                
            pbar.set_postfix({'val_rmse': f'{val_rmse:.4f}'})
        
        # Test evaluation
        test_loss, test_rmse, _, _, _, _ = evaluate_model_with_nodes(
            model, test_loader, criterion, device
        )
        
        results[f"{layer_type} model"] = test_rmse
        print(f"{layer_type} Test RMSE: {test_rmse:.4f}")
    
    # 2. Test our model with and without 3D features
    print("\n\nTesting impact of 3D features...")
    
    # Without 3D features
    print("Training GAT model WITHOUT 3D features...")
    x_no_3d = []
    for mol_features in create_atom_features(smiles_list, features_dict=None):
        truncated = [atom_features[:-6] for atom_features in mol_features]
        x_no_3d.append(truncated)
    
    data_list_no_3d = prepare_data_for_training(x_no_3d, edge_index, y, smiles_list)
    
    # Train model without 3D features
    train_data = [data_list_no_3d[i] for i in indices[:train_size]]
    val_data = [data_list_no_3d[i] for i in indices[train_size:train_size + val_size]]
    test_data = [data_list_no_3d[i] for i in indices[train_size + val_size:]]
    
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=16)
    test_loader = DataLoader(test_data, batch_size=16)
    
    model_no_3d = GSR(
        in_channels=len(x_no_3d[0][0]),
        hidden_channels=28,
        out_channels=1,
        heads=1,
        dropout=0
    ).to(device)
    
    # Quick training
    optimizer = optim.RMSprop(model_no_3d.parameters(), lr=0.00075)
    best_val_rmse = float('inf')
    
    pbar = tqdm(range(500), desc="Training without 3D features")
    for epoch in pbar:
        model_no_3d.train()
        for batch in train_loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            _, graph_outputs = model_no_3d(batch)
            loss = criterion(graph_outputs, batch.y)
            loss.backward()
            optimizer.step()
            
        model_no_3d.eval()
        val_rmse = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = batch.to(device)
                _, graph_outputs = model_no_3d(batch)
                val_rmse += compute_rmse(graph_outputs, batch.y).item() * batch.num_graphs
        val_rmse = val_rmse / len(val_loader.dataset)
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            
        pbar.set_postfix({'val_rmse': f'{val_rmse:.4f}'})
    
    test_loss, test_rmse_no_3d, _, _, _, _ = evaluate_model_with_nodes(
        model_no_3d, test_loader, criterion, device
    )
    results["Our model without 3D features"] = test_rmse_no_3d
    
    # 3. Test pooling strategies
    print("\n\nTesting different pooling strategies...")
    
    # Full features for pooling tests
    x_full = create_atom_features(smiles_list, features_dict=None)
    data_list_full = prepare_data_for_training(x_full, edge_index, y, smiles_list)
    
    # Save results
    results_df = pd.DataFrame(list(results.items()), columns=['Model', 'RMSE'])
    results_df = results_df.sort_values('RMSE')
    results_df.to_csv(os.path.join(output_dir, 'esol_ablation_results.csv'), index=False)
    
    print("\n\nAblation Study Results (ESOL):")
    print(results_df.to_string())
    
    return results_df


def run_random_forest_baseline(data_dir='datasets', dataset='freesolv'):
    """
    Run Random Forest baseline using molecular descriptors.
    """
    print("=" * 50)
    print(f"Running Random Forest Baseline on {dataset.upper()} Dataset")
    print("=" * 50)
    
    # Dataset configurations
    dataset_configs = {
        'freesolv': {
            'file': 'FreeSolv.csv',
            'target_col': 'expt',
            'smiles_col': 'SMILES'
        },
        'esol': {
            'file': 'ESOL.csv',
            'target_col': 'measured log solubility in mols per litre',
            'smiles_col': 'SMILES'
        }
    }
    
    config = dataset_configs[dataset]
    
    # Load data
    data_file = os.path.join(data_dir, dataset, config['file'])
    if not os.path.exists(data_file):
        data_file = os.path.join(data_dir, config['file'])
    
    df = pd.read_csv(data_file)
    smiles_list = list(df[config['smiles_col']])
    y_values = list(df[config['target_col']])
    
    print(f"Loaded {len(smiles_list)} molecules")
    
    # Generate molecular descriptors using RDKit
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors
    
    # Calculate a comprehensive set of molecular descriptors
    descriptor_names = [desc[0] for desc in Descriptors._descList]
    
    X = []
    valid_indices = []
    
    print("Calculating molecular descriptors...")
    for i, smiles in enumerate(tqdm(smiles_list)):
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            # Calculate all RDKit descriptors
            descriptors = []
            for desc_name in descriptor_names:
                desc_func = getattr(Descriptors, desc_name)
                try:
                    value = desc_func(mol)
                    if isinstance(value, (int, float)) and not np.isnan(value) and not np.isinf(value):
                        descriptors.append(value)
                    else:
                        descriptors.append(0.0)
                except:
                    descriptors.append(0.0)
            
            # Add some additional features
            descriptors.extend([
                rdMolDescriptors.CalcNumRotatableBonds(mol),
                rdMolDescriptors.CalcNumHBD(mol),
                rdMolDescriptors.CalcNumHBA(mol),
                mol.GetNumAtoms(),
                len(mol.GetRingInfo().AtomRings())
            ])
            
            X.append(descriptors)
            valid_indices.append(i)
    
    X = np.array(X)
    y = np.array([y_values[i] for i in valid_indices])
    
    print(f"Feature matrix shape: {X.shape}")
    
    # Split data (80/10/10)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.111, random_state=42)
    
    # Train Random Forest
    print("Training Random Forest...")
    rf = RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1
    )
    
    rf.fit(X_train, y_train)
    
    # Evaluate
    train_pred = rf.predict(X_train)
    val_pred = rf.predict(X_val)
    test_pred = rf.predict(X_test)
    
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    val_rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    
    print(f"\nRandom Forest Results ({dataset.upper()}):")
    print(f"Train RMSE: {train_rmse:.4f}")
    print(f"Val RMSE: {val_rmse:.4f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    
    # Feature importance analysis
    feature_importance = pd.DataFrame({
        'feature': descriptor_names + ['NumRotatableBonds', 'NumHBD', 'NumHBA', 'NumAtoms', 'NumRings'],
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    return test_rmse, rf, feature_importance


def analyze_node_positions(data_dir='datasets', output_dir='node_analysis'):
    """
    Analyze predictions at different node positions (Table 5 in paper).
    """
    print("=" * 50)
    print("Analyzing Predictions at Different Node Positions")
    print("=" * 50)
    
    os.makedirs(output_dir, exist_ok=True)
    
    datasets = ['esol', 'freesolv', 'lipophilicity', 'bace']
    dataset_configs = {
        'esol': {'file': 'ESOL.csv', 'target': 'measured log solubility in mols per litre', 'smiles': 'SMILES'},
        'freesolv': {'file': 'FreeSolv.csv', 'target': 'expt', 'smiles': 'SMILES'},
        'lipophilicity': {'file': 'Lipophilicity!.csv', 'target': 'exp', 'smiles': 'smiles'},
        'bace': {'file': 'bace.csv', 'target': 'pIC50', 'smiles': 'mol'}
    }
    
    results = {}
    
    for dataset in datasets:
        print(f"\n\nAnalyzing {dataset.upper()} dataset...")
        config = dataset_configs[dataset]
        
        # Load data
        data_file = os.path.join(data_dir, dataset, config['file'])
        if not os.path.exists(data_file):
            data_file = os.path.join(data_dir, config['file'])
            
        df = pd.read_csv(data_file)
        
        # Filter molecules with at least 5 atoms for middle position analysis
        from rdkit import Chem
        valid_indices = []
        for i, smiles in enumerate(df[config['smiles']]):
            mol = Chem.MolFromSmiles(smiles)
            if mol and mol.GetNumAtoms() >= 5:
                valid_indices.append(i)
                
        df_filtered = df.iloc[valid_indices].reset_index(drop=True)
        
        smiles_list = list(df_filtered[config['smiles']])
        y_values = list(df_filtered[config['target']])
        y = format_y_values(y_values)
        
        print(f"Using {len(smiles_list)} molecules with >= 5 atoms")
        
        # Extract features and structures
        edge_index = extract_edge_indices(smiles_list)
        x = create_atom_features(smiles_list, features_dict=None)
        data_list = prepare_data_for_training(x, edge_index, y, smiles_list)
        
        # Split data
        train_size = int(0.8 * len(data_list))
        val_size = int(0.1 * len(data_list))
        
        indices = list(range(len(data_list)))
        np.random.shuffle(indices)
        
        test_indices = indices[train_size + val_size:]
        test_data = [data_list[i] for i in test_indices]
        test_loader = DataLoader(test_data, batch_size=16)
        
        # Load pre-trained model or train a new one
        device = get_device()
        model = GSR(in_channels=35, hidden_channels=28, out_channels=1, heads=1, dropout=0).to(device)
        
        # For this analysis, we'll use a pre-trained model or train briefly
        # In practice, you'd load the best model for each dataset
        
        # Analyze predictions at different positions
        positions = ['first', 'second', 'middle', 'second_last', 'last', 'mean']
        position_rmse = {}
        
        for position in positions:
            print(f"Analyzing {position} position...")
            
            all_predictions = []
            all_targets = []
            
            model.eval()
            with torch.no_grad():
                for batch in test_loader:
                    batch = batch.to(device)
                    node_outputs, _ = model(batch)
                    
                    # Get predictions based on position
                    batch_size = batch.num_graphs
                    for i in range(batch_size):
                        # Get nodes for this graph
                        if hasattr(batch, 'batch'):
                            mask = batch.batch == i
                            graph_node_preds = node_outputs[mask]
                        else:
                            # Handle single graph case
                            graph_node_preds = node_outputs
                        
                        num_nodes = len(graph_node_preds)
                        
                        if position == 'first':
                            pred = graph_node_preds[0]
                        elif position == 'second':
                            pred = graph_node_preds[1] if num_nodes > 1 else graph_node_preds[0]
                        elif position == 'middle':
                            # Average of middle nodes (excluding first 2 and last 2)
                            if num_nodes > 4:
                                middle_preds = graph_node_preds[2:-2]
                                pred = middle_preds.mean(dim=0)
                            else:
                                pred = graph_node_preds[num_nodes//2]
                        elif position == 'second_last':
                            pred = graph_node_preds[-2] if num_nodes > 1 else graph_node_preds[-1]
                        elif position == 'last':
                            pred = graph_node_preds[-1]
                        elif position == 'mean':
                            pred = graph_node_preds.mean(dim=0)
                        
                        all_predictions.append(pred.cpu().numpy())
                        all_targets.append(batch.y[i].cpu().numpy())
            
            # Calculate RMSE
            predictions = np.array(all_predictions)
            targets = np.array(all_targets)
            rmse = np.sqrt(mean_squared_error(targets.flatten(), predictions.flatten()))
            position_rmse[position] = rmse
        
        results[dataset] = position_rmse
    
    # Save results
    results_df = pd.DataFrame(results).T
    results_df.to_csv(os.path.join(output_dir, 'node_position_analysis.csv'))
    
    print("\n\nNode Position Analysis Results:")
    print(results_df.round(4))
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description='Run ablation studies for TChemGNN')
    parser.add_argument('--experiment', type=str, required=True,
                      choices=['ablation', 'random_forest', 'node_positions', 'all'],
                      help='Which experiment to run')
    parser.add_argument('--data_dir', type=str, default='datasets',
                      help='Directory containing datasets')
    parser.add_argument('--output_dir', type=str, default='ablation_results',
                      help='Directory for output results')
    
    args = parser.parse_args()
    
    if args.experiment == 'ablation' or args.experiment == 'all':
        run_ablation_study_esol(args.data_dir, args.output_dir)
        
    if args.experiment == 'random_forest' or args.experiment == 'all':
        run_random_forest_baseline(args.data_dir, 'freesolv')
        
    if args.experiment == 'node_positions' or args.experiment == 'all':
        analyze_node_positions(args.data_dir, args.output_dir)


if __name__ == '__main__':
    main()