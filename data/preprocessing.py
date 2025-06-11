import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors, AllChem, rdPartialCharges
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


def get_base_feature_dimension():
    """
    Calculate the base feature dimension according to the paper.
    
    Returns:
    - int: Base feature dimension (36 = 14 atomic + 22 molecular)
    """
    # According to the paper:
    # 14 atomic-level features (per atom)
    # 22 molecular-level features (16 molecular + 6 global 3D)
    return 36


def compute_3d_features(mol):
    """
    Compute 3D molecular features as described in the paper.
    
    Parameters:
    - mol: RDKit molecule object
    
    Returns:
    - dict: Dictionary containing 3D features
    """
    # Generate 3D coordinates
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.UFFOptimizeMolecule(mol)
    
    # Get conformer
    conf = mol.GetConformer()
    
    # Calculate molecule dimensions
    positions = []
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        positions.append([pos.x, pos.y, pos.z])
    
    positions = np.array(positions)
    
    # Width, length, height (max extent in each dimension)
    width = positions[:, 0].max() - positions[:, 0].min()
    length = positions[:, 1].max() - positions[:, 1].min()
    height = positions[:, 2].max() - positions[:, 2].min()
    
    # Volume approximation (using bounding box)
    volume = width * length * height
    
    # Calculate dipole moment
    # First, compute partial charges
    rdPartialCharges.ComputeGasteigerCharges(mol)
    
    # Calculate dipole moment components
    dipole_x, dipole_y, dipole_z = 0.0, 0.0, 0.0
    for i in range(mol.GetNumAtoms()):
        charge = float(mol.GetAtomWithIdx(i).GetProp('_GasteigerCharge'))
        if not np.isnan(charge) and not np.isinf(charge):
            pos = conf.GetAtomPosition(i)
            dipole_x += charge * pos.x
            dipole_y += charge * pos.y
            dipole_z += charge * pos.z
    
    # Dipole magnitude
    dipole_momentum = np.sqrt(dipole_x**2 + dipole_y**2 + dipole_z**2)
    
    # Angle of general molecular orientation (angle of principal axis)
    # Using the vector from center of mass to farthest atom
    center_of_mass = positions.mean(axis=0)
    distances = np.sqrt(((positions - center_of_mass)**2).sum(axis=1))
    farthest_idx = distances.argmax()
    
    # Direction vector
    direction = positions[farthest_idx] - center_of_mass
    direction_norm = direction / (np.linalg.norm(direction) + 1e-8)
    
    # Angle with z-axis (in radians)
    angle = np.arccos(np.clip(direction_norm[2], -1.0, 1.0))
    
    # Remove hydrogens for the rest of the calculation
    mol = Chem.RemoveHs(mol)
    
    return {
        'volume': volume,
        'width': width,
        'length': length,
        'height': height,
        'dipole_momentum': dipole_momentum,
        'angle': angle
    }


def create_atom_features(smiles_list, features_dict=None):
    """
    Create atom-level features for all molecules according to the paper.
    
    Features according to the paper:
    - 14 Atomic-Level Features (per atom)
    - 22 Molecular-Level Features (16 molecular + 6 global 3D)
    Total: 36 features per atom
    
    Parameters:
    - smiles_list: List of SMILES strings
    - features_dict: Dictionary of additional molecular features (can be None/empty)
    
    Returns:
    - x: List of atom feature matrices for each molecule
    """
    # Electronegativity dictionary (Pauling scale) as in the paper
    electronegativity_dict = {
        1: 2.20,   # Hydrogen
        6: 2.55,   # Carbon
        7: 3.04,   # Nitrogen
        8: 3.44,   # Oxygen
        9: 3.98,   # Fluorine
        15: 2.19,  # Phosphorus
        16: 2.58,  # Sulfur
        17: 2.96,  # Chlorine
        35: 2.66,  # Bromine
        53: 2.66,  # Iodine
        11: 0.93,  # Sodium
        12: 1.31,  # Magnesium
    }

    # Print feature information
    print(f"Feature dimensions: Total=36 (14 atomic + 22 molecular)")
    
    x = []
    computed_3d_features = []
    
    # First pass: compute all 3D features
    print("Computing 3D molecular features...")
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        try:
            features_3d = compute_3d_features(mol)
            computed_3d_features.append(features_3d)
        except:
            # Fallback values if 3D generation fails
            computed_3d_features.append({
                'volume': 0.0,
                'width': 0.0,
                'length': 0.0,
                'height': 0.0,
                'dipole_momentum': 0.0,
                'angle': 0.0
            })
    
    # Second pass: create feature vectors
    for di, smiles in enumerate(smiles_list):
        # Convert SMILES to an RDKit molecule object
        mol = Chem.MolFromSmiles(smiles)
        
        # Initialize an empty list to store atom features
        result = []
        
        # === Calculate 16 Molecular-Level Features (same for all atoms) ===
        
        # 1. has_ring
        has_ring = int(len(mol.GetRingInfo().AtomRings()) > 0)
        
        # 2. is_aromatic
        is_aromatic = int(any(atom.GetIsAromatic() for atom in mol.GetAtoms()))
        
        # 3. formal_charge (global)
        formal_charge = sum(atom.GetFormalCharge() for atom in mol.GetAtoms())
        
        # 4. min_degree
        min_degree = min(atom.GetDegree() for atom in mol.GetAtoms())
        
        # 5. num_hbond_donors
        num_hbond_donors = rdMolDescriptors.CalcNumHBD(mol)
        
        # 6. num_rings
        num_rings = len(mol.GetRingInfo().AtomRings())
        
        # 7. num_rotatable_bonds
        num_rotatable_bonds = rdMolDescriptors.CalcNumRotatableBonds(mol)
        
        # 8. polar_surface_area
        polar_surface_area = rdMolDescriptors.CalcTPSA(mol)
        
        # 9. molecular_weight
        molecular_weight = rdMolDescriptors.CalcExactMolWt(mol)
        
        # 10. logP
        logP = Descriptors.MolLogP(mol)
        
        # 11. num_atoms
        num_atoms = mol.GetNumAtoms()
        
        # 12. hba (hydrogen bond acceptors)
        hba = rdMolDescriptors.CalcNumHBA(mol)
        
        # 13. hbd (hydrogen bond donors)
        hbd = rdMolDescriptors.CalcNumHBD(mol)
        
        # 14. fraction_sp2
        fraction_sp2 = sum(1 for atom in mol.GetAtoms() if atom.GetHybridization() == Chem.HybridizationType.SP2) / num_atoms
        
        # 15. valence (global)
        valence = sum(atom.GetTotalValence() for atom in mol.GetAtoms())
        
        # 16. general_electronegativity (average)
        total_electronegativity = sum(electronegativity_dict.get(atom.GetAtomicNum(), 0) for atom in mol.GetAtoms())
        general_electronegativity = total_electronegativity / num_atoms if num_atoms > 0 else 0
        
        # Get 3D features
        features_3d = computed_3d_features[di]
        
        # Iterate through each atom in the molecule
        for atom in mol.GetAtoms():
            # === 14 Atomic-Level Features (per atom) ===
            
            # 1. atom_degree
            atom_degree = atom.GetDegree()
            
            # 2. atomic_number
            atomic_number = atom.GetAtomicNum()
            
            # 3. num_hydrogens
            num_hydrogens = atom.GetTotalNumHs()
            
            # 4. atomic_valence
            atomic_valence = atom.GetTotalValence()
            
            # 5. num_radical_electrons
            num_radical_electrons = atom.GetNumRadicalElectrons()
            
            # 6. atom_formal_charge
            atom_formal_charge = atom.GetFormalCharge()
            
            # 7. atom_hybridization
            atom_hybridization = int(atom.GetHybridization())
            
            # 8. electronegativity
            electronegativity = electronegativity_dict.get(atom.GetAtomicNum(), 0)
            
            # 9. has_electronegativity
            has_electronegativity = int(atom.GetAtomicNum() in electronegativity_dict)
            
            # 10. has_implicit_hydrogens
            has_implicit_hydrogens = int(atom.GetNumImplicitHs() > 0)
            
            # 11. hydroxyl_group (-OH presence)
            hydroxyl_group = int(atom.GetSymbol() == 'O' and any(neighbor.GetSymbol() == 'H' for neighbor in atom.GetNeighbors()))
            
            # 12. atomic_mass_scaled: As = (A - 10.812)/116.092
            atomic_mass = atom.GetMass()
            atomic_mass_scaled = (atomic_mass - 10.812) / 116.092
            
            # 13. van_der_waals_radius_scaled: Rvdw,s = (Rvdw - 1.5)/0.6
            # Approximate Van der Waals radii (in Angstroms)
            vdw_radii = {1: 1.2, 6: 1.7, 7: 1.55, 8: 1.52, 9: 1.47, 
                        15: 1.8, 16: 1.8, 17: 1.75, 35: 1.85, 53: 1.98}
            vdw_radius = vdw_radii.get(atom.GetAtomicNum(), 1.5)
            vdw_radius_scaled = (vdw_radius - 1.5) / 0.6
            
            # 14. covalent_radius_scaled: Rcov,s = (Rcov - 0.64)/0.76
            # Approximate covalent radii (in Angstroms)
            cov_radii = {1: 0.31, 6: 0.76, 7: 0.71, 8: 0.66, 9: 0.57,
                        15: 1.07, 16: 1.05, 17: 1.02, 35: 1.2, 53: 1.39}
            cov_radius = cov_radii.get(atom.GetAtomicNum(), 0.7)
            cov_radius_scaled = (cov_radius - 0.64) / 0.76
            
            # Combine all features in the exact order from the paper
            features = [
                # 14 Atomic-Level Features
                atom_degree,                # 1
                atomic_number,              # 2
                num_hydrogens,              # 3
                atomic_valence,             # 4
                num_radical_electrons,      # 5
                atom_formal_charge,         # 6
                atom_hybridization,         # 7
                electronegativity,          # 8
                has_electronegativity,      # 9
                has_implicit_hydrogens,     # 10
                hydroxyl_group,             # 11
                atomic_mass_scaled,         # 12
                vdw_radius_scaled,          # 13
                cov_radius_scaled,          # 14
                
                # 16 Molecular-Level Features
                has_ring,                   # 15
                is_aromatic,                # 16
                formal_charge,              # 17
                min_degree,                 # 18
                num_hbond_donors,           # 19
                num_rings,                  # 20
                num_rotatable_bonds,        # 21
                polar_surface_area,         # 22
                molecular_weight,           # 23
                logP,                       # 24
                num_atoms,                  # 25
                hba,                        # 26
                hbd,                        # 27
                fraction_sp2,               # 28
                valence,                    # 29
                general_electronegativity,  # 30
                
                # 6 Global 3D Features
                features_3d['volume'],          # 31
                features_3d['width'],           # 32
                features_3d['length'],          # 33
                features_3d['height'],          # 34
                features_3d['dipole_momentum'], # 35
                features_3d['angle']            # 36
            ]
            
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