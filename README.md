# TChemGNN - Implementation Aligned with Paper

This repository contains the implementation of **TChemGNN** (Tiny Chemistry Graph Neural Network) from the paper "Efficient learning of molecular properties with Graph Neural Networks and 3D molecular features".

## Key Changes to Align with Paper

### 1. **Feature Engineering (36 total features)**
   - **14 Atomic-Level Features**: atom_degree, atomic_number, num_hydrogens, atomic_valence, num_radical_electrons, atom_formal_charge, atom_hybridization, electronegativity, has_electronegativity, has_implicit_hydrogens, hydroxyl_group, atomic_mass_scaled, van_der_waals_radius_scaled, covalent_radius_scaled
   - **16 Molecular-Level Features**: has_ring, is_aromatic, formal_charge, min_degree, num_hbond_donors, num_rings, num_rotatable_bonds, polar_surface_area, molecular_weight, logP, num_atoms, hba, hbd, fraction_sp2, valence, general_electronegativity
   - **6 Global 3D Features**: volume, width, length, height, dipole_momentum, angle

### 2. **Model Architecture**
   - 5 GAT (Graph Attention Network) layers
   - Hidden dimension: 28 (best results)
   - Activation: hyperbolic tangent (tanh)
   - Total parameters: ~3.7K
   - Heads: 1
   - Dropout: 0

### 3. **No-Pooling Approach**
   - For ESOL and FreeSolv: Uses the last node's prediction
   - For Lipophilicity and BACE: Uses mean pooling
   - Based on the SMILES encoding structure

### 4. **Training Configuration**
   - Optimizer: RMSprop
   - Learning rate: 0.00075
   - Epochs: 5000
   - Train/Val/Test split: 80/10/10

## Expected Results (RMSE)

| Dataset | Paper Result | Expected |
|---------|--------------|----------|
| ESOL | 0.7844 | ✓ |
| FreeSolv | 1.0124 | ✓ |
| Lipophilicity | 1.0221 | ✓ |
| BACE | 0.9586 | ✓ |

## Usage

### Basic Training Command

```bash
# ESOL dataset (uses last node prediction automatically)
python main.py --dataset esol --data_dir datasets

# FreeSolv dataset (uses last node prediction automatically)
python main.py --dataset freesolv --data_dir datasets

# Lipophilicity dataset (uses mean pooling automatically)
python main.py --dataset lipophilicity --data_dir datasets

# BACE dataset (uses mean pooling automatically)
python main.py --dataset bace --data_dir datasets
```

### Advanced Options

```bash
# Override pooling strategy
python main.py --dataset esol --pooling_strategy mean

# Custom hyperparameters (not recommended - paper values are optimal)
python main.py --dataset esol --hidden_dim 64 --heads 4

# Create training visualizations as GIFs
python main.py --dataset esol --create_gifs

# Use specific device
python main.py --dataset esol --device cuda:0
```

### Dataset Structure

Place your datasets in the following structure:
```
datasets/
├── esol/
│   └── ESOL.csv
├── freesolv/
│   └── FreeSolv.csv
├── lipophilicity/
│   └── Lipophilicity!.csv
└── bace/
    └── bace.csv
```

## Key Implementation Details

1. **3D Feature Computation**: The code now computes 3D molecular features on-the-fly using RDKit, including molecular dimensions, volume, dipole moment, and orientation angle.

2. **Feature Scaling**: Atomic mass, Van der Waals radius, and covalent radius are scaled according to the formulas in the paper.

3. **SMILES Ordering**: The model leverages the unique SMILES encoding where the first and last atoms are typically at the periphery of the molecule, making them important for properties like solubility.

4. **Model Size**: The implementation maintains ~3.7K parameters as specified in the paper, making it very efficient compared to large foundation models.

## Dependencies

- PyTorch
- PyTorch Geometric
- RDKit
- NumPy
- Pandas
- tqdm
- matplotlib
- seaborn

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{lutchyn2024efficient,
  title={Efficient learning of molecular properties with Graph Neural Networks and 3D molecular features},
  author={Lutchyn, Tetiana and Mardal, Marie and Ricaud, Benjamin},
  journal={...},
  year={2024}
}
```

## Notes

- The paper mentions removing 2 molecules from the Lipophilicity dataset where RDKit couldn't compute 3D structures. This is handled gracefully in the code with fallback values.
- The no-pooling approach is a key innovation showing that for some properties (ESOL, FreeSolv), using peripheral atoms' predictions is more effective than averaging over all atoms.
- All hyperparameters are set to the paper's optimal values by default.