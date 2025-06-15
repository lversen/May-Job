"""
Setup script for TChemGNN - installs all required dependencies.
"""

import subprocess
import sys
import os


def install_requirements():
    """Install all required packages for TChemGNN."""
    
    print("=" * 50)
    print("Setting up TChemGNN Environment")
    print("=" * 50)
    
    # Core requirements
    requirements = [
        'torch>=1.12.0',
        'torch-geometric',
        'torch-scatter',
        'torch-sparse',
        'numpy',
        'pandas',
        'scikit-learn',
        'rdkit',
        'matplotlib',
        'seaborn',
        'tqdm',
        'pillow'  # For GIF generation
    ]
    
    # First, upgrade pip
    print("\nUpgrading pip...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
    
    # Install PyTorch first (CPU version for compatibility)
    print("\nInstalling PyTorch...")
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', 
        'torch', 'torchvision', 'torchaudio', 
        '--index-url', 'https://download.pytorch.org/whl/cpu'
    ])
    
    # Install PyTorch Geometric
    print("\nInstalling PyTorch Geometric...")
    subprocess.check_call([
        sys.executable, '-m', 'pip', 'install', 
        'torch-geometric'
    ])
    
    # Install other requirements
    print("\nInstalling other requirements...")
    for req in requirements:
        if req not in ['torch>=1.12.0', 'torch-geometric']:  # Already installed
            print(f"Installing {req}...")
            try:
                subprocess.check_call([sys.executable, '-m', 'pip', 'install', req])
            except:
                print(f"Warning: Could not install {req}")
    
    # Special handling for RDKit if it fails
    try:
        import rdkit
        print("✓ RDKit installed successfully")
    except ImportError:
        print("\nRDKit installation failed. Trying conda...")
        try:
            subprocess.check_call(['conda', 'install', '-c', 'conda-forge', 'rdkit', '-y'])
        except:
            print("\n⚠ Could not install RDKit automatically.")
            print("Please install RDKit manually:")
            print("  conda install -c conda-forge rdkit")
            print("  or")
            print("  pip install rdkit")
    
    print("\n" + "=" * 50)
    print("Setup complete!")
    print("=" * 50)
    
    # Verify installation
    print("\nVerifying installation...")
    verify_installation()


def verify_installation():
    """Verify that all required packages are installed."""
    
    required_modules = [
        ('torch', 'PyTorch'),
        ('torch_geometric', 'PyTorch Geometric'),
        ('rdkit', 'RDKit'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('tqdm', 'tqdm')
    ]
    
    all_good = True
    
    for module_name, display_name in required_modules:
        try:
            __import__(module_name)
            print(f"✓ {display_name} is installed")
        except ImportError:
            print(f"✗ {display_name} is NOT installed")
            all_good = False
    
    if all_good:
        print("\n✓ All dependencies are installed correctly!")
        print("\nNext steps:")
        print("1. Download datasets: python download_datasets.py")
        print("2. Run experiments: python main.py --dataset esol")
        print("3. Run all experiments: python run_all_experiments.py")
    else:
        print("\n⚠ Some dependencies are missing.")
        print("Please install them manually or use a conda environment.")


def create_requirements_txt():
    """Create a requirements.txt file."""
    
    requirements_content = """# TChemGNN Requirements
torch>=1.12.0
torch-geometric
torch-scatter
torch-sparse
numpy
pandas
scikit-learn
rdkit
matplotlib
seaborn
tqdm
pillow
"""
    
    with open('requirements.txt', 'w') as f:
        f.write(requirements_content)
    
    print("Created requirements.txt file")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup TChemGNN environment')
    parser.add_argument('--verify-only', action='store_true',
                      help='Only verify installation without installing packages')
    parser.add_argument('--create-requirements', action='store_true',
                      help='Create requirements.txt file')
    
    args = parser.parse_args()
    
    if args.create_requirements:
        create_requirements_txt()
    elif args.verify_only:
        verify_installation()
    else:
        install_requirements()