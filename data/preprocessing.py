import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors
import torch
from torch_geometric.data import Data


def load_data(csv_file, additional_features_files=None):
    """
    Load data from CSV file and additional feature files.
    
    Parameters:
    - csv_file: Path to main CSV file with SMILES and experimental values
    - additional_features_files: Dictionary mapping feature names to file paths
    
    Returns:
    - df: DataFrame containing SMILES and target values
    - features_dict: Dictionary containing additional molecular features
    """
    df = pd.read_csv(csv_file)
    
    # Clean the data if needed
    if 1561 in df.index:
        df = df.drop(1561, axis=0)
        df = df.reset_index(drop=True)
    
    features_dict = {}
    
    # Load additional feature files if provided
    if additional_features_files:
        for feature_name, file_path in additional_features_files.items():
            try:
                feature_data = pd.read_csv(file_path, header=None)
                features_dict[feature_name] = list(feature_data.to_numpy().flatten())
                print(f"Loaded {feature_name} data: {len(features_dict[feature_name])} values")
            except Exception as e:
                print(f"Error loading {feature_name} from {file_path}: {e}")
    
    return df, features_dict


def extract_edge_indices(smiles_list):
    """
    Extract edge indices from SMILES for graph construction.
    
    Parameters:
    - smiles_list: List of SMILES strings
    
    Returns:
    - all_edge_indexes: List of edge indices for each molecule
    """
    all_edge_indexes = []
    
    for smiles in smiles_list:
        # Convert SMILES string to RDKit Mol object
        mol = Chem.MolFromSmiles(smiles)
        
        # Get atom and bond information
        bonds = mol.GetBonds()
        
        # Initialize edge indexes list for this SMILES
        edge_indexes = []
        
        # Iterate through bonds to get the edge indexes
        for bond in bonds:
            begin_idx = bond.GetBeginAtomIdx()
            end_idx = bond.GetEndAtomIdx()
            # Add both directions for undirected graph
            edge_indexes.append([begin_idx, end_idx])
            edge_indexes.append([end_idx, begin_idx])  
        
        # Append the edge indexes for this SMILES to the overall list
        all_edge_indexes.append(edge_indexes)
    
    return all_edge_indexes


def create_atom_features(smiles_list, features_dict=None):
    """
    Create atom-level features for all molecules.
    
    Parameters:
    - smiles_list: List of SMILES strings
    - features_dict: Dictionary of additional molecular features
    
    Returns:
    - x: List of atom feature matrices for each molecule
    """
    # A dictionary mapping atomic numbers to electronegativity values (Pauling scale)
    electronegativity_dict = {
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

    additional_features = features_dict.keys() if features_dict else []
    
    x = []
    for di, smiles in enumerate(smiles_list):
        # Convert SMILES to an RDKit molecule object
        mol = Chem.MolFromSmiles(smiles)
        
        # Initialize an empty list to store atom features
        result = []
        
        # Calculate global features for the molecule
        min_degree = min(atom.GetDegree() for atom in mol.GetAtoms())  
        num_hbond_donors = sum(1 for atom in mol.GetAtoms() if atom.GetSymbol() == 'O' and atom.GetTotalNumHs() > 0)
        num_rings = len(mol.GetRingInfo().AtomRings())
        num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        polar_surface_area = rdMolDescriptors.CalcTPSA(mol)
        
        # Additional chemical properties
        molecular_weight = rdMolDescriptors.CalcExactMolWt(mol)
        logP = rdMolDescriptors.CalcCrippenDescriptors(mol)[0]
        
        # Additional properties for the molecule
        num_atoms = mol.GetNumAtoms()
        hba = rdMolDescriptors.CalcNumHBA(mol)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        fraction_sp2 = sum(1 for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP2) / num_atoms
        
        # Global molecule properties
        is_in_ring = int(len(mol.GetRingInfo().AtomRings()) > 0)
        is_aromatic = int(any(atom.GetIsAromatic() for atom in mol.GetAtoms()))
        formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        valence = sum(atom.GetTotalValence() for atom in mol.GetAtoms())
        
        # General electronegativity
        total_electronegativity = sum(electronegativity_dict.get(atom.GetAtomicNum(), 0) for atom in mol.GetAtoms())
        general_electronegativity = total_electronegativity / num_atoms if num_atoms > 0 else 0
        
        # Combine global features
        global_features = [
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
        
        # Iterate through each atom in the molecule
        for atom in mol.GetAtoms():
            # Define atom features
            features = [
                is_in_ring,
                is_aromatic,
                formal_charge,
                int(atom.GetDegree()),
                int(atom.GetAtomicNum()),
                int(atom.GetTotalNumHs()),
                int(atom.GetTotalValence()),
                int(atom.GetNumRadicalElectrons()),
                int(atom.GetFormalCharge()),
                int(atom.GetHybridization()),
                int((atom.GetMass() - 10.812) / 116.092),
                int((atom.GetAtomicNum() - 1.5) / 0.6),
                int((atom.GetAtomicNum() - 0.64) / 0.76),
                electronegativity_dict.get(atom.GetAtomicNum(), 0),
                int(atom.GetAtomicNum() in electronegativity_dict),
                int(atom.GetNumImplicitHs() > 0),
            ]
            
            # Add global features (these are the same for every atom in the molecule)
            features.extend(global_features)
            
            # Add hydroxyl group feature
            if atom.GetSymbol() == 'O' and any(neighbor.GetSymbol() == 'H' for neighbor in atom.GetNeighbors()):
                features.append(1)  # Hydroxyl group (-OH) presence
            else:
                features.append(0)
            
            # Add additional features from external data if available
            if features_dict:
                for feature_name in additional_features:
                    if di < len(features_dict[feature_name]):
                        features.append(features_dict[feature_name][di])
                    else:
                        features.append(0)  # Default value if missing
            
            # Append the atom features to the result
            result.append(features)
        
        # Append the result for the current molecule
        x.append(result)
    
    return x


def prepare_data_for_training(x, edge_index, y, smiles_list):
    """
    Prepare PyTorch Geometric Data objects for training.
    
    Parameters:
    - x: List of atom feature matrices for each molecule
    - edge_index: List of edge indices for each molecule
    - y: List of target values for each molecule
    - smiles_list: List of SMILES strings
    
    Returns:
    - data_list: List of PyTorch Geometric Data objects
    """
    data_list = []
    
    for i in range(len(x)):
        x_graph = torch.tensor(x[i], dtype=torch.float32)
        edge_index_graph = torch.tensor(edge_index[i], dtype=torch.long).t().contiguous()
        y_graph = torch.tensor(y[i], dtype=torch.float32)
        
        data_graph = Data(x=x_graph, edge_index=edge_index_graph, y=y_graph)
        data_list.append(data_graph)
    
    return data_list