"""
Improved setup script for TChemGNN - handles Windows compilation issues.
Replace your existing setup_script.py with this version.
"""

import subprocess
import sys
import os
import platform
import torch


def get_pytorch_info():
    """Get PyTorch version and CUDA info for wheel selection."""
    try:
        import torch
        torch_version = torch.__version__.split('+')[0]  # Remove +cu128 suffix
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            cuda_version = torch.version.cuda
            if cuda_version:
                cuda_suffix = f"cu{cuda_version.replace('.', '')}"
            else:
                cuda_suffix = "cpu"
        else:
            cuda_suffix = "cpu"
            
        return torch_version, cuda_suffix
    except ImportError:
        return None, None


def install_pytorch_geometric_windows():
    """Install PyTorch Geometric components with pre-built wheels for Windows."""
    print("\nDetected Windows - using pre-built wheels for PyTorch Geometric...")
    
    # Get PyTorch info
    torch_version, cuda_suffix = get_pytorch_info()
    
    if torch_version is None:
        print("PyTorch not found. Installing PyTorch first...")
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', 
            'torch', 'torchvision', 'torchaudio',
            '--index-url', 'https://download.pytorch.org/whl/cpu'
        ])
        torch_version, cuda_suffix = get_pytorch_info()
    
    print(f"PyTorch version: {torch_version}, CUDA: {cuda_suffix}")
    
    # Construct wheel URL
    base_url = f"https://data.pyg.org/whl/torch-{torch_version}+{cuda_suffix}.html"
    
    # Install PyTorch Geometric components with pre-built wheels
    pyg_packages = [
        'torch-scatter',
        'torch-sparse', 
        'torch-cluster',
        'torch-spline-conv',
        'torch-geometric'
    ]
    
    for package in pyg_packages:
        print(f"Installing {package}...")
        try:
            subprocess.check_call([
                sys.executable, '-m', 'pip', 'install', package,
                '-f', base_url
            ])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError as e:
            print(f"Warning: Failed to install {package} with wheels")
            print(f"Error: {e}")
            
            # Fallback: try conda if available
            if package in ['torch-scatter', 'torch-sparse']:
                try:
                    print(f"Trying conda for {package}...")
                    subprocess.check_call([
                        'conda', 'install', '-c', 'pyg', '-c', 'conda-forge', 
                        package, '-y'
                    ])
                    print(f"✓ {package} installed via conda")
                except:
                    print(f"✗ Could not install {package}")


def install_pytorch_geometric_unix():
    """Install PyTorch Geometric for Unix-like systems."""
    print("\nInstalling PyTorch Geometric for Unix/Linux/Mac...")
    
    packages = [
        'torch-geometric',
        'torch-scatter', 
        'torch-sparse'
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', package])
            print(f"✓ {package} installed successfully")
        except subprocess.CalledProcessError:
            print(f"Warning: Could not install {package}")


def install_requirements():
    """Install all required packages for TChemGNN."""
    
    print("=" * 50)
    print("Setting up TChemGNN Environment (Improved)")
    print("=" * 50)
    
    # Detect OS
    is_windows = platform.system() == 'Windows'
    
    # First, upgrade pip
    print("\nUpgrading pip...")
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'])
    
    # Install PyTorch first (CPU version for compatibility)
    print("\nInstalling PyTorch...")
    try:
        import torch
        print(f"✓ PyTorch already installed: {torch.__version__}")
    except ImportError:
        subprocess.check_call([
            sys.executable, '-m', 'pip', 'install', 
            'torch', 'torchvision', 'torchaudio', 
            '--index-url', 'https://download.pytorch.org/whl/cpu'
        ])
    
    # Install PyTorch Geometric with OS-specific method
    if is_windows:
        install_pytorch_geometric_windows()
    else:
        install_pytorch_geometric_unix()
    
    # Core scientific packages
    core_requirements = [
        'numpy',
        'pandas', 
        'scikit-learn',
        'matplotlib',
        'seaborn',
        'tqdm',
        'pillow'
    ]
    
    print("\nInstalling core scientific packages...")
    for req in core_requirements:
        print(f"Installing {req}...")
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', req])
        except subprocess.CalledProcessError:
            print(f"Warning: Could not install {req}")
    
    # Install RDKit with special handling
    print("\nInstalling RDKit...")
    try:
        import rdkit
        print("✓ RDKit already installed")
    except ImportError:
        # Try pip first
        try:
            subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'rdkit'])
            print("✓ RDKit installed via pip")
        except subprocess.CalledProcessError:
            # Try conda
            try:
                print("Pip failed, trying conda...")
                subprocess.check_call(['conda', 'install', '-c', 'conda-forge', 'rdkit', '-y'])
                print("✓ RDKit installed via conda")
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
        ('torch_scatter', 'PyTorch Scatter'),
        ('torch_sparse', 'PyTorch Sparse'),
        ('rdkit', 'RDKit'),
        ('numpy', 'NumPy'),
        ('pandas', 'Pandas'),
        ('sklearn', 'Scikit-learn'),
        ('matplotlib', 'Matplotlib'),
        ('tqdm', 'tqdm')
    ]
    
    all_good = True
    critical_missing = []
    
    for module_name, display_name in required_modules:
        try:
            __import__(module_name)
            print(f"✓ {display_name} is installed")
        except ImportError:
            print(f"✗ {display_name} is NOT installed")
            all_good = False
            if module_name in ['torch', 'torch_geometric', 'rdkit']:
                critical_missing.append(display_name)
    
    if all_good:
        print("\n✓ All dependencies are installed correctly!")
        print("\nNext steps:")
        print("1. Download datasets: python download_datasets.py")
        print("2. Run experiments: python main.py --dataset esol")
        print("3. Run all experiments: python run_all_experiments.py")
    else:
        print(f"\n⚠ Some dependencies are missing.")
        if critical_missing:
            print(f"Critical missing packages: {', '.join(critical_missing)}")
            print("\nTroubleshooting suggestions:")
            print("1. If on Windows, ensure you have Visual Studio Build Tools")
            print("2. Try using conda environment instead:")
            print("   conda create -n tchemgnn python=3.9")
            print("   conda activate tchemgnn")
            print("   conda install pytorch-geometric pytorch-sparse pytorch-scatter -c pyg")
            print("   conda install rdkit -c conda-forge")
        else:
            print("Run the script again or install missing packages manually.")


def create_conda_environment_file():
    """Create environment.yml for conda users."""
    
    env_content = """name: tchemgnn
channels:
  - pytorch
  - pyg
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - pytorch
  - torchvision
  - torchaudio
  - cpuonly  # Remove this line if you want GPU support
  - pytorch-geometric
  - pytorch-scatter
  - pytorch-sparse
  - rdkit
  - numpy
  - pandas
  - scikit-learn
  - matplotlib
  - seaborn
  - tqdm
  - pillow
"""
    
    with open('environment.yml', 'w') as f:
        f.write(env_content)
    
    print("Created environment.yml for conda users")
    print("To use: conda env create -f environment.yml")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Setup TChemGNN environment')
    parser.add_argument('--verify-only', action='store_true',
                      help='Only verify installation without installing packages')
    parser.add_argument('--create-conda-env', action='store_true',
                      help='Create conda environment.yml file')
    parser.add_argument('--force-conda', action='store_true',
                      help='Prefer conda over pip for installations')
    
    args = parser.parse_args()
    
    if args.create_conda_env:
        create_conda_environment_file()
    elif args.verify_only:
        verify_installation()
    else:
        if args.force_conda:
            print("Note: --force-conda not fully implemented yet")
        install_requirements()


if __name__ == '__main__':
    main()