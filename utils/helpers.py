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


def get_device(device_str=None):
    """
    Get the appropriate device (CUDA or CPU).
    
    Parameters:
    - device_str: Optional string to specify device ('cpu', 'cuda', 'cuda:0', etc.)
    
    Returns:
    - device: PyTorch device
    """
    if device_str is None:
        # Auto-detect
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        # User specified device
        if device_str.startswith('cuda') and not torch.cuda.is_available():
            print(f"Warning: CUDA requested but not available. Falling back to CPU.")
            device = torch.device('cpu')
        else:
            device = torch.device(device_str)
    
    return device