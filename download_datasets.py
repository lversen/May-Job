"""
Download datasets for TChemGNN experiments.
This script downloads the required datasets from MoleculeNet.
"""

import os
import urllib.request
import zipfile
import pandas as pd
import argparse
from tqdm import tqdm


class DatasetDownloader:
    def __init__(self, data_dir='datasets'):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Dataset URLs and information
        self.datasets = {
            'esol': {
                'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/delaney-processed.csv',
                'filename': 'ESOL.csv',
                'columns': {
                    'smiles': 'smiles',
                    'target': 'measured log solubility in mols per litre'
                }
            },
            'freesolv': {
                'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/SAMPL.csv',
                'filename': 'FreeSolv.csv',
                'columns': {
                    'smiles': 'smiles',
                    'target': 'expt'
                }
            },
            'lipophilicity': {
                'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/Lipophilicity.csv',
                'filename': 'Lipophilicity!.csv',
                'columns': {
                    'smiles': 'smiles',
                    'target': 'exp'
                }
            },
            'bace': {
                'url': 'https://deepchemdata.s3-us-west-1.amazonaws.com/datasets/bace.csv',
                'filename': 'bace.csv',
                'columns': {
                    'smiles': 'mol',
                    'target': 'pIC50'
                }
            }
        }
    
    def download_file(self, url, filepath):
        """Download a file with progress bar."""
        print(f"Downloading {os.path.basename(filepath)}...")
        
        # Get file size
        response = urllib.request.urlopen(url)
        file_size = int(response.headers.get('Content-Length', 0))
        
        # Download with progress bar
        with tqdm(total=file_size, unit='B', unit_scale=True) as pbar:
            def hook(block_num, block_size, total_size):
                downloaded = block_num * block_size
                pbar.update(downloaded - pbar.n)
            
            urllib.request.urlretrieve(url, filepath, reporthook=hook)
    
    def process_dataset(self, dataset_name):
        """Download and process a dataset."""
        info = self.datasets[dataset_name]
        dataset_dir = os.path.join(self.data_dir, dataset_name)
        os.makedirs(dataset_dir, exist_ok=True)
        
        filepath = os.path.join(dataset_dir, info['filename'])
        
        # Check if file already exists
        if os.path.exists(filepath):
            print(f"✓ {dataset_name} already exists at {filepath}")
            return True
        
        try:
            # Download the file
            self.download_file(info['url'], filepath)
            
            # Verify the download by reading the CSV
            df = pd.read_csv(filepath)
            
            # Check for required columns
            if info['columns']['smiles'] not in df.columns:
                # Try to find SMILES column
                smiles_cols = [col for col in df.columns if 'smile' in col.lower()]
                if smiles_cols:
                    print(f"  Note: SMILES column '{info['columns']['smiles']}' not found, but found '{smiles_cols[0]}'")
                else:
                    print(f"  Warning: Could not find SMILES column")
            
            if info['columns']['target'] not in df.columns:
                print(f"  Warning: Target column '{info['columns']['target']}' not found")
                print(f"  Available columns: {', '.join(df.columns)}")
            
            print(f"✓ Downloaded {dataset_name}: {len(df)} molecules")
            return True
            
        except Exception as e:
            print(f"✗ Failed to download {dataset_name}: {str(e)}")
            # Try alternative approach or provide instructions
            print(f"\nPlease download {dataset_name} manually:")
            print(f"  1. Go to https://moleculenet.org/datasets-1")
            print(f"  2. Download the {dataset_name} dataset")
            print(f"  3. Place it in {dataset_dir}/{info['filename']}")
            return False
    
    def download_all(self):
        """Download all datasets."""
        print("=" * 50)
        print("Downloading TChemGNN Datasets")
        print("=" * 50)
        
        success_count = 0
        
        for dataset_name in self.datasets:
            print(f"\nProcessing {dataset_name}...")
            if self.process_dataset(dataset_name):
                success_count += 1
        
        print("\n" + "=" * 50)
        print(f"Downloaded {success_count}/{len(self.datasets)} datasets successfully")
        
        if success_count < len(self.datasets):
            print("\nSome datasets failed to download.")
            print("You can download them manually from:")
            print("https://moleculenet.org/datasets-1")
        
        return success_count == len(self.datasets)
    
    def verify_datasets(self):
        """Verify that all datasets are present and valid."""
        print("\nVerifying datasets...")
        all_valid = True
        
        for dataset_name, info in self.datasets.items():
            filepath = os.path.join(self.data_dir, dataset_name, info['filename'])
            
            if os.path.exists(filepath):
                try:
                    df = pd.read_csv(filepath)
                    print(f"✓ {dataset_name}: {len(df)} molecules")
                except Exception as e:
                    print(f"✗ {dataset_name}: Invalid file - {str(e)}")
                    all_valid = False
            else:
                print(f"✗ {dataset_name}: Not found")
                all_valid = False
        
        return all_valid


def main():
    parser = argparse.ArgumentParser(description='Download datasets for TChemGNN')
    parser.add_argument('--data_dir', type=str, default='datasets',
                      help='Directory to store datasets')
    parser.add_argument('--verify_only', action='store_true',
                      help='Only verify existing datasets without downloading')
    
    args = parser.parse_args()
    
    downloader = DatasetDownloader(args.data_dir)
    
    if args.verify_only:
        if downloader.verify_datasets():
            print("\n✓ All datasets are present and valid!")
        else:
            print("\n✗ Some datasets are missing or invalid.")
            print("Run without --verify_only to download them.")
    else:
        if downloader.download_all():
            print("\n✓ All datasets downloaded successfully!")
            print(f"You can now run experiments with: python main.py --dataset esol --data_dir {args.data_dir}")
        else:
            print("\n⚠ Some datasets need to be downloaded manually.")


if __name__ == '__main__':
    main()