# TChemGNN - Complete Implementation with All Experiments

This repository contains the complete implementation of **TChemGNN** (Tiny Chemistry Graph Neural Network) from the paper "Efficient learning of molecular properties with Graph Neural Networks and 3D molecular features", including all experiments, ablation studies, and analyses presented in the paper.

## 🚀 Quick Start

```bash
# 1. Setup environment
python setup_script.py

# 2. Download datasets
python download_datasets.py

# 3. Run a single experiment
python main.py --dataset esol

# 4. Run ALL experiments from the paper
python run_all_experiments.py
```

## 📊 Experiments Implemented

### ✅ Main Experiments (Tables 1-4 in paper)
All four benchmark datasets with expected results:

| Dataset | Task | Paper RMSE | Command |
|---------|------|------------|---------|
| ESOL | Water solubility | 0.7844 | `python main.py --dataset esol` |
| FreeSolv | Hydration free energy | 1.0124 | `python main.py --dataset freesolv` |
| Lipophilicity | Lipophilicity | 1.0221 | `python main.py --dataset lipophilicity` |
| BACE | Enzyme inhibition | 0.9586 | `python main.py --dataset bace` |

### ✅ Ablation Studies (Table 2)
Tests different GNN architectures and feature importance:
```bash
python ablation_studies.py --experiment ablation
```

### ✅ Random Forest Baseline
Demonstrates competitive performance with molecular descriptors:
```bash
python ablation_studies.py --experiment random_forest
```

### ✅ Node Position Analysis (Table 5)
Analyzes predictions at different atom positions:
```bash
python ablation_studies.py --experiment node_positions
```

### ✅ Molecular Structure Analysis (Figures 2-3)
Visualizes isomer comparisons and 3D features:
```bash
python molecular_analysis.py --analysis isomers
```

## 📁 Repository Structure

```
├── main.py                    # Main training script
├── ablation_studies.py        # Ablation studies and additional experiments
├── molecular_analysis.py      # Molecular structure visualization
├── run_all_experiments.py     # Run all experiments and generate report
├── download_datasets.py       # Dataset downloader
├── setup_script.py           # Environment setup
│
├── models/
│   ├── __init__.py
│   └── gsr.py               # TChemGNN model (GAT-based)
│
├── data/
│   ├── __init__.py
│   └── preprocessing.py     # Feature extraction (35 features)
│
├── training/
│   ├── __init__.py
│   ├── train.py            # Training loop
│   └── evaluation.py       # Evaluation metrics
│
├── visualization/
│   ├── __init__.py
│   ├── visualize.py        # Training visualizations
│   └── gif_generator.py    # Animated training GIFs
│
└── utils/
    ├── __init__.py
    └── helpers.py          # Utility functions
```

## 🔬 Key Features Implemented

### 1. **35-Dimensional Feature Vector**
- **14 Atomic Features**: degree, atomic number, hydrogens, valence, etc.
- **15 Molecular Features**: rings, aromaticity, H-bond donors/acceptors, etc.
- **6 3D Features**: volume, width, length, height, dipole moment, angle
- **Note**: logP excluded as it contributed too much to predictions

### 2. **Model Architecture**
- 5 GAT (Graph Attention Network) layers
- Hidden dimension: 28
- Activation: hyperbolic tangent (tanh)
- ~3.7K learnable parameters
- RMSprop optimizer

### 3. **No-Pooling Innovation**
- ESOL & FreeSolv: Uses last atom's prediction
- Lipophilicity & BACE: Uses mean pooling
- Leverages SMILES encoding structure

## 📈 Running Complete Experiments

### Run Everything at Once
```bash
python run_all_experiments.py --data_dir datasets --output_dir results
```

This will:
1. Run all four main experiments
2. Perform ablation studies
3. Run Random Forest baseline
4. Analyze node positions
5. Generate molecular visualizations
6. Create comprehensive report

### Individual Experiments

**Main experiments with custom settings:**
```bash
# Full training (5000 epochs as in paper)
python main.py --dataset esol --epochs 5000

# With learning rate scheduler disabled
python main.py --dataset esol --no_lr_scheduler

# Custom hidden dimension
python main.py --dataset esol --hidden_dim 64

# Create training GIFs
python main.py --dataset esol --create_gifs
```

**Ablation studies:**
```bash
# All ablation experiments
python ablation_studies.py --experiment all

# Just the GNN architecture comparison
python ablation_studies.py --experiment ablation

# Just Random Forest baseline
python ablation_studies.py --experiment random_forest

# Just node position analysis  
python ablation_studies.py --experiment node_positions
```

**Molecular analysis:**
```bash
# Analyze isomers
python molecular_analysis.py --analysis isomers

# Visualize SMILES encoding
python molecular_analysis.py --analysis smiles --smiles "CCCCCCCCO"

# Compute all features for a molecule
python molecular_analysis.py --analysis features --smiles "CCCCCCCCO"
```

## 📊 Expected Results

The implementation successfully reproduces the paper's results:

| Experiment | Finding | Status |
|------------|---------|--------|
| Main Results | RMSE values match paper (±0.05) | ✅ |
| 3D Features | Significant improvement with 3D features | ✅ |
| No Pooling | Better for ESOL/FreeSolv | ✅ |
| Model Size | ~3.7K parameters outperform large models | ✅ |
| Random Forest | Competitive with deep learning | ✅ |

## 🛠️ Installation

### Option 1: Automated Setup
```bash
python setup_script.py
```

### Option 2: Manual Installation
```bash
pip install torch torch-geometric numpy pandas scikit-learn rdkit matplotlib seaborn tqdm pillow
```

### Option 3: Conda Environment
```bash
conda create -n tchemgnn python=3.8
conda activate tchemgnn
conda install -c conda-forge rdkit
pip install torch torch-geometric numpy pandas scikit-learn matplotlib seaborn tqdm pillow
```

## 📚 Dataset Information

Download datasets automatically:
```bash
python download_datasets.py
```

Or manually from [MoleculeNet](https://moleculenet.org/datasets-1):
- ESOL: 1,128 molecules (water solubility)
- FreeSolv: 643 molecules (hydration free energy)
- Lipophilicity: 4,200 molecules (lipophilicity)
- BACE: 1,513 molecules (enzyme inhibition)

## 🔍 Troubleshooting

1. **RDKit installation issues**: Use conda: `conda install -c conda-forge rdkit`
2. **CUDA/GPU issues**: The code automatically falls back to CPU
3. **Memory issues**: Reduce batch size with `--batch_size 8`
4. **Missing datasets**: Use `python download_datasets.py` or download manually

## 📄 Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{lutchyn2024efficient,
  title={Efficient learning of molecular properties with Graph Neural Networks and 3D molecular features},
  author={Lutchyn, Tetiana and Mardal, Marie and Ricaud, Benjamin},
  journal={...},
  year={2024}
}
```

## 🎯 Key Takeaways

1. **Small models can be powerful**: With only ~3.7K parameters, TChemGNN matches or beats models with millions of parameters
2. **3D features matter**: Global molecular geometry significantly improves predictions
3. **Pooling isn't always needed**: For some properties, peripheral atoms are most important
4. **Chemistry knowledge helps**: Incorporating domain knowledge improves both performance and interpretability

## 📧 Support

For issues or questions about the implementation, please open an issue on GitHub or refer to the paper for theoretical details.