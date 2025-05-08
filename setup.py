from setuptools import setup, find_packages

setup(
    name="lipophilicity_prediction",
    version="0.1",
    packages=find_packages(),
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "torch>=1.9.0",
        "torch_geometric>=2.0.0",
        "rdkit>=2021.09.1",
        "matplotlib>=3.4.0",
        "seaborn>=0.11.0",
        "pillow>=8.0.0",
        "scikit-learn>=0.24.0",
    ],
)