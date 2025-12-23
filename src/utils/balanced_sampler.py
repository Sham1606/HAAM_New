"""
Balanced sampler for training to handle class imbalance.
Calculates weights based on inverse frequency of emotion classes.
"""

import torch
from torch.utils.data import WeightedRandomSampler
import numpy as np

def get_balanced_sampler(dataset_labels):
    """
    Create a WeightedRandomSampler for balanced training
    Args:
        dataset_labels (list/array): List of integer labels for the dataset
    Returns:
        WeightedRandomSampler
    """
    labels = np.array(dataset_labels)
    class_counts = np.bincount(labels)
    
    # Avoid division by zero
    class_counts = np.maximum(class_counts, 1)
    
    # Inverse frequency weights
    class_weights = 1. / class_counts
    
    # Assign weight to each sample
    sample_weights = class_weights[labels]
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(sample_weights).double(),
        num_samples=len(sample_weights),
        replacement=True
    )
    
    return sampler, class_weights
