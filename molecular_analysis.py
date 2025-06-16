"""
Molecular structure analysis and visualization for TChemGNN paper.
This script analyzes molecular properties and creates visualizations
comparing isomers as shown in Figures 2 and 3 of the paper.

FIXED VERSION: Handles RDKit Draw module import issues
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import argparse

# RDKit imports with comprehensive error handling
try:
    from rdkit import Chem
    from rdkit.Chem import AllChem, rdMolDescriptors, rdPartialCharges
    
    # Try to import Draw module with fallback
    try:
        from rdkit.Chem import Draw
        from rdkit.Chem.Draw import IPythonConsole
        DRAW_AVAILABLE = True
        print("✓ RDKit Draw module loaded successfully")
    except ImportError as e:
        print(f"Warning: RDKit Draw module not available: {e}")
        print("Molecule visualization will be limited.")
        DRAW_AVAILABLE = False
        
    RDKIT_AVAILABLE = True
    print("✓ RDKit core modules loaded successfully")
    
except ImportError as e:
    print(f"Error importing RDKit: {e}")
    print("Please install RDKit with: conda install -c conda-forge rdkit")
    RDKIT_AVAILABLE = False
    DRAW_AVAILABLE = False


def safe_mol_to_image(mol, size=(300, 300)):
    """Safely convert molecule to image with fallback."""
    if not DRAW_AVAILABLE or mol is None:
        print("Cannot generate molecule image - Draw module not available or invalid molecule")
        # Return a placeholder or None
        return None
    
    try:
        return Draw.MolToImage(mol, size=size)
    except Exception as e:
        print(f"Error drawing molecule: {e}")
        return None


def safe_mol_draw_2d(mol, output_file=None, highlight_atoms=None, highlight_colors=None):
    """Safely draw molecule with 2D drawer."""
    if not DRAW_AVAILABLE or mol is None:
        print("Cannot draw molecule - Draw module not available or invalid molecule")
        return None
    
    try:
        drawer = Draw.MolDraw2D(400, 400)
        if highlight_atoms and highlight_colors:
            drawer.DrawMolecule(mol, highlightAtoms=highlight_atoms, 
                              highlightAtomColors=highlight_colors)
        else:
            drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        
        if output_file:
            with open(output_file, 'wb') as f:
                f.write(drawer.GetDrawingText().encode())
            print(f"Saved molecule drawing to {output_file}")
            
        return drawer
    except Exception as e:
        print(f"Error in 2D drawing: {e}")
        return None


def compute_molecular_features(smiles):
    """
    Compute all molecular features for a given SMILES string.
    Returns a dictionary with all features used in the paper.
    """
    if not RDKIT_AVAILABLE:
        print("RDKit not available - cannot compute molecular features")
        return None, None, None
        
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print(f"Invalid SMILES: {smiles}")
        return None, None, None
    
    # Add hydrogens for 3D calculation
    mol_h = Chem.AddHs(mol)
    
    try:
        AllChem.EmbedMolecule(mol_h, randomSeed=42)
        AllChem.UFFOptimizeMolecule(mol_h)
    except Exception as e:
        print(f"Warning: Could not generate 3D structure for {smiles}: {e}")
        # Create default 3D features
        features = {
            'molecular_weight': rdMolDescriptors.CalcExactMolWt(mol),
            'num_atoms': mol.GetNumAtoms(),
            'num_rings': len(mol.GetRingInfo().AtomRings()),
            'aromatic_rings': sum(1 for ring in mol.GetRingInfo().AtomRings() 
                                if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)),
            'volume': 0.0,
            'width': 0.0,
            'length': 0.0,
            'height': 0.0,
            'dipole_momentum': 0.0,
            'angle': 0.0,
            'logP': rdMolDescriptors.CalcCrippenDescriptors(mol)[0],
            'tpsa': rdMolDescriptors.CalcTPSA(mol),
            'num_rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
            'num_hbd': rdMolDescriptors.CalcNumHBD(mol),
            'num_hba': rdMolDescriptors.CalcNumHBA(mol),
            'formula': rdMolDescriptors.CalcMolFormula(mol),
            'smiles': smiles
        }
        return features, mol, mol_h
    
    # Get conformer
    conf = mol_h.GetConformer()
    
    # Calculate 3D dimensions
    positions = []
    for i in range(mol_h.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        positions.append([pos.x, pos.y, pos.z])
    
    positions = np.array(positions)
    
    # Basic 3D features
    width = positions[:, 0].max() - positions[:, 0].min()
    length = positions[:, 1].max() - positions[:, 1].min()
    height = positions[:, 2].max() - positions[:, 2].min()
    volume = width * length * height
    
    # Calculate dipole moment
    try:
        rdPartialCharges.ComputeGasteigerCharges(mol_h)
        dipole_x, dipole_y, dipole_z = 0.0, 0.0, 0.0
        for i in range(mol_h.GetNumAtoms()):
            charge = float(mol_h.GetAtomWithIdx(i).GetProp('_GasteigerCharge'))
            if not np.isnan(charge) and not np.isinf(charge):
                pos = conf.GetAtomPosition(i)
                dipole_x += charge * pos.x
                dipole_y += charge * pos.y
                dipole_z += charge * pos.z
        
        dipole_momentum = np.sqrt(dipole_x**2 + dipole_y**2 + dipole_z**2)
    except Exception as e:
        print(f"Warning: Could not compute dipole moment: {e}")
        dipole_momentum = 0.0
    
    # Angle calculation
    center_of_mass = positions.mean(axis=0)
    distances = np.sqrt(((positions - center_of_mass)**2).sum(axis=1))
    farthest_idx = distances.argmax()
    direction = positions[farthest_idx] - center_of_mass
    direction_norm = direction / (np.linalg.norm(direction) + 1e-8)
    angle = np.arccos(np.clip(direction_norm[2], -1.0, 1.0))
    
    # Remove hydrogens for other calculations
    mol = Chem.RemoveHs(mol_h)
    
    # Molecular features
    features = {
        # Basic properties
        'molecular_weight': rdMolDescriptors.CalcExactMolWt(mol),
        'num_atoms': mol.GetNumAtoms(),
        'num_rings': len(mol.GetRingInfo().AtomRings()),
        'aromatic_rings': sum(1 for ring in mol.GetRingInfo().AtomRings() 
                            if all(mol.GetAtomWithIdx(idx).GetIsAromatic() for idx in ring)),
        
        # 3D features
        'volume': volume,
        'width': width,
        'length': length,
        'height': height,
        'dipole_momentum': dipole_momentum,
        'angle': angle * 180 / np.pi,  # Convert to degrees
        
        # Other molecular descriptors
        'logP': rdMolDescriptors.CalcCrippenDescriptors(mol)[0],
        'tpsa': rdMolDescriptors.CalcTPSA(mol),
        'num_rotatable_bonds': rdMolDescriptors.CalcNumRotatableBonds(mol),
        'num_hbd': rdMolDescriptors.CalcNumHBD(mol),
        'num_hba': rdMolDescriptors.CalcNumHBA(mol),
        
        # Chemical formula
        'formula': rdMolDescriptors.CalcMolFormula(mol),
        'smiles': smiles
    }
    
    return features, mol, mol_h


def find_isomers_in_dataset(dataset_path, formula, smiles_col='SMILES'):
    """
    Find all molecules with the same chemical formula in a dataset.
    """
    if not RDKIT_AVAILABLE:
        print("RDKit not available - cannot find isomers")
        return []
        
    df = pd.read_csv(dataset_path)
    
    isomers = []
    for idx, row in df.iterrows():
        smiles = row[smiles_col]
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            mol_formula = rdMolDescriptors.CalcMolFormula(mol)
            if mol_formula == formula:
                isomers.append({
                    'index': idx,
                    'smiles': smiles,
                    'formula': mol_formula,
                    'target': row.get('measured log solubility in mols per litre', None)
                })
    
    return isomers


def visualize_isomer_comparison(smiles1, smiles2, name1="Molecule 1", name2="Molecule 2", output_file=None):
    """
    Create a comparison visualization of two isomers showing their structures and properties.
    """
    if not RDKIT_AVAILABLE:
        print("RDKit not available - cannot visualize isomers")
        return None, None
        
    # Compute features for both molecules
    features1, mol1, mol1_h = compute_molecular_features(smiles1)
    features2, mol2, mol2_h = compute_molecular_features(smiles2)
    
    if features1 is None or features2 is None:
        print("Error: Could not process one or both molecules")
        return None, None
    
    # Create figure
    fig = plt.figure(figsize=(16, 10))
    
    # 2D structures (if Draw is available)
    if DRAW_AVAILABLE:
        ax1 = plt.subplot(2, 3, 1)
        img1 = safe_mol_to_image(mol1, size=(300, 300))
        if img1:
            ax1.imshow(img1)
        ax1.set_title(name1)
        ax1.axis('off')
        
        ax2 = plt.subplot(2, 3, 2)
        img2 = safe_mol_to_image(mol2, size=(300, 300))
        if img2:
            ax2.imshow(img2)
        ax2.set_title(name2)
        ax2.axis('off')
    else:
        # Text-based representation if Draw not available
        ax1 = plt.subplot(2, 3, 1)
        ax1.text(0.5, 0.5, f"{name1}\n{smiles1}", 
                horizontalalignment='center', verticalalignment='center',
                fontsize=10, wrap=True)
        ax1.set_title(name1)
        ax1.axis('off')
        
        ax2 = plt.subplot(2, 3, 2)
        ax2.text(0.5, 0.5, f"{name2}\n{smiles2}", 
                horizontalalignment='center', verticalalignment='center',
                fontsize=10, wrap=True)
        ax2.set_title(name2)
        ax2.axis('off')
    
    # 3D structure visualization (simplified)
    ax3 = plt.subplot(2, 3, 3, projection='3d')
    
    # Plot atoms for molecule 1 (if 3D coordinates available)
    if mol1_h and mol1_h.GetNumConformers() > 0:
        conf1 = mol1_h.GetConformer()
        for i in range(mol1_h.GetNumAtoms()):
            pos = conf1.GetAtomPosition(i)
            atom = mol1_h.GetAtomWithIdx(i)
            color = 'gray' if atom.GetSymbol() == 'C' else 'red' if atom.GetSymbol() == 'O' else 'blue'
            ax3.scatter(pos.x, pos.y, pos.z, c=color, s=100, alpha=0.7)
    
    ax3.set_title(f"3D Structure: {name1}")
    ax3.set_xlabel('X')
    ax3.set_ylabel('Y')
    ax3.set_zlabel('Z')
    
    # Feature comparison table
    ax4 = plt.subplot(2, 3, 4)
    ax4.axis('off')
    
    # Create comparison data
    comparison_data = []
    for key in ['molecular_weight', 'volume', 'width', 'length', 'height', 'dipole_momentum', 'angle']:
        comparison_data.append([
            key.replace('_', ' ').title(),
            f"{features1[key]:.3f}",
            f"{features2[key]:.3f}"
        ])
    
    table = ax4.table(cellText=comparison_data,
                     colLabels=['Property', name1, name2],
                     cellLoc='left',
                     loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    
    # Chemical formula
    ax5 = plt.subplot(2, 3, 5)
    ax5.text(0.5, 0.5, f"Chemical Formula: {features1['formula']}", 
            horizontalalignment='center', verticalalignment='center',
            fontsize=16, weight='bold')
    ax5.axis('off')
    
    # Additional properties
    ax6 = plt.subplot(2, 3, 6)
    ax6.axis('off')
    
    additional_data = []
    for key in ['logP', 'tpsa', 'num_rotatable_bonds', 'num_hbd', 'num_hba']:
        additional_data.append([
            key.replace('_', ' ').title(),
            f"{features1[key]:.2f}" if isinstance(features1[key], float) else str(features1[key]),
            f"{features2[key]:.2f}" if isinstance(features2[key], float) else str(features2[key])
        ])
    
    table2 = ax6.table(cellText=additional_data,
                      colLabels=['Property', name1, name2],
                      cellLoc='left',
                      loc='center')
    table2.auto_set_font_size(False)
    table2.set_fontsize(10)
    table2.scale(1, 2)
    
    plt.suptitle(f'Isomer Comparison: {features1["formula"]}', fontsize=16, weight='bold')
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Saved visualization to {output_file}")
    else:
        plt.show()
    
    return features1, features2


def analyze_esol_isomers(data_dir='datasets', output_dir='isomer_analysis'):
    """
    Analyze isomers from ESOL dataset as shown in the paper.
    """
    if not RDKIT_AVAILABLE:
        print("RDKit not available - cannot analyze isomers")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    # Examples from the paper
    isomer_examples = [
        {
            'formula': 'C7H7NO2',
            'smiles1': 'COC(=O)c1ccncc1',  # Methyl nicotinate
            'smiles2': 'Cc1ccc(cc1)[N+](=O)[O-]',  # p-Nitrotoluene
            'name1': 'Methyl nicotinate',
            'name2': 'p-Nitrotoluene'
        },
        {
            'formula': 'C6H14O',
            'smiles1': 'CC(C)CC(C)O',  # 3-Methyl-2-pentanol
            'smiles2': 'CCCCCCO',  # 1-Hexanol
            'name1': '3-Methyl-2-pentanol',
            'name2': '1-Hexanol'
        }
    ]
    
    # Load ESOL dataset
    esol_file = os.path.join(data_dir, 'esol', 'ESOL.csv')
    if not os.path.exists(esol_file):
        esol_file = os.path.join(data_dir, 'ESOL.csv')
    
    if not os.path.exists(esol_file):
        print(f"ESOL dataset not found at {esol_file}")
        return
    
    print("Analyzing isomers from ESOL dataset...")
    
    for i, example in enumerate(isomer_examples):
        print(f"\nAnalyzing {example['formula']} isomers...")
        
        # Find all isomers in dataset
        isomers = find_isomers_in_dataset(esol_file, example['formula'])
        print(f"Found {len(isomers)} molecules with formula {example['formula']}")
        
        # Visualize the specific pair from the paper
        output_file = os.path.join(output_dir, f'isomer_comparison_{example["formula"]}.png')
        features1, features2 = visualize_isomer_comparison(
            example['smiles1'], 
            example['smiles2'],
            example['name1'],
            example['name2'],
            output_file
        )
        
        if features1 and features2:
            # Print detailed comparison
            print(f"\nDetailed comparison for {example['formula']}:")
            print(f"{example['name1']}:")
            print(f"  SMILES: {example['smiles1']}")
            print(f"  3D Volume: {features1['volume']:.3f}")
            print(f"  Dipole moment: {features1['dipole_momentum']:.3f}")
            print(f"  Angle: {features1['angle']:.1f}°")
            
            print(f"\n{example['name2']}:")
            print(f"  SMILES: {example['smiles2']}")
            print(f"  3D Volume: {features2['volume']:.3f}")
            print(f"  Dipole moment: {features2['dipole_momentum']:.3f}")
            print(f"  Angle: {features2['angle']:.1f}°")
            
            # Calculate differences
            print(f"\nDifferences:")
            print(f"  Volume difference: {abs(features1['volume'] - features2['volume']):.3f}")
            print(f"  Dipole difference: {abs(features1['dipole_momentum'] - features2['dipole_momentum']):.3f}")
            print(f"  Angle difference: {abs(features1['angle'] - features2['angle']):.1f}°")


def analyze_smiles_encoding(smiles, output_file=None):
    """
    Analyze and visualize SMILES encoding to show first and last atoms.
    """
    if not RDKIT_AVAILABLE:
        print("RDKit not available - cannot analyze SMILES encoding")
        return None
        
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        print("Error: Invalid SMILES")
        return None
    
    # Highlight first and last atoms
    highlight_atoms = [0, mol.GetNumAtoms() - 1]  # First and last
    highlight_colors = {0: (0.8, 0.2, 0.8), mol.GetNumAtoms() - 1: (0.8, 0.8, 0.2)}  # Purple and yellow
    
    # Draw molecule with highlights
    drawer = safe_mol_draw_2d(mol, output_file, highlight_atoms, highlight_colors)
    
    # Print analysis
    print(f"SMILES: {smiles}")
    print(f"Number of atoms: {mol.GetNumAtoms()}")
    print(f"First atom (position 0): {mol.GetAtomWithIdx(0).GetSymbol()}")
    print(f"Last atom (position {mol.GetNumAtoms()-1}): {mol.GetAtomWithIdx(mol.GetNumAtoms()-1).GetSymbol()}")
    
    return mol


def main():
    parser = argparse.ArgumentParser(description='Molecular structure analysis for TChemGNN')
    parser.add_argument('--analysis', type=str, required=True,
                      choices=['isomers', 'smiles', 'features'],
                      help='Type of analysis to perform')
    parser.add_argument('--data_dir', type=str, default='datasets',
                      help='Directory containing datasets')
    parser.add_argument('--output_dir', type=str, default='molecular_analysis',
                      help='Directory for output')
    parser.add_argument('--smiles', type=str, default=None,
                      help='SMILES string to analyze (for smiles/features analysis)')
    
    args = parser.parse_args()
    
    if not RDKIT_AVAILABLE:
        print("Error: RDKit is not available. Please install it with:")
        print("  conda install -c conda-forge rdkit")
        return
    
    if args.analysis == 'isomers':
        analyze_esol_isomers(args.data_dir, args.output_dir)
        
    elif args.analysis == 'smiles':
        if args.smiles is None:
            # Analyze examples from paper
            examples = [
                ('C1=NC2=C(N1)C(=O)NC=N2', 'Hypoxanthine'),
                ('CCCCCCCCO', 'Octanol')
            ]
            os.makedirs(args.output_dir, exist_ok=True)
            for smiles, name in examples:
                print(f"\nAnalyzing {name}:")
                output_file = os.path.join(args.output_dir, f'smiles_encoding_{name}.png')
                analyze_smiles_encoding(smiles, output_file)
        else:
            analyze_smiles_encoding(args.smiles)
            
    elif args.analysis == 'features':
        if args.smiles is None:
            print("Please provide a SMILES string with --smiles")
        else:
            features, _, _ = compute_molecular_features(args.smiles)
            if features:
                print(f"\nMolecular features for {args.smiles}:")
                for key, value in features.items():
                    print(f"  {key}: {value}")


if __name__ == '__main__':
    main()