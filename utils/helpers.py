import os
import random
import torch
import numpy as np


def set_seed(seed=42):
    """
    Set random seeds for reproducibility.
    
    Parameters:
    - seed: Random seed
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


def create_directory(directory):
    """
    Create directory if it doesn't exist.
    
    Parameters:
    - directory: Directory path to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")


def format_y_values(values):
    """
    Format target values for model input.
    
    Parameters:
    - values: List of target values
    
    Returns:
    - formatted_values: List of formatted target values [[y_1], [y_2], ...]
    """
    # Convert to the desired format
    formatted_values = [[[y_t]] for y_t in values]
    return formatted_values


def get_device():
    """
    Get the appropriate device (CUDA or CPU).
    
    Returns:
    - device: PyTorch device
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return device