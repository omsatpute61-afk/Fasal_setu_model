"""
PyTorch Loss Functions and Samplers for Imbalanced Agricultural Datasets
Handles the long-tailed distribution of IP102 and real-world field data.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import WeightedRandomSampler

class FocalLoss(nn.Module):
    """
    Focal Loss heavily penalizes the model when it misses a rare disease or pest,
    while dampening the loss from easily classified dominant classes (like Healthy leaves).
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        # inputs: [batch_size, num_classes]
        # targets: [batch_size]
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss) # Prevents nans when probability 0
        
        focal_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return torch.mean(focal_loss)
        elif self.reduction == 'sum':
            return torch.sum(focal_loss)
        else:
            return focal_loss

def create_class_aware_sampler(dataset, class_counts):
    """
    Creates a WeightedRandomSampler to ensure rare classes (e.g. rare pests in IP102) 
    are passed to the GPU more frequently during training.
    
    Args:
        dataset: PyTorch dataset
        class_counts: list or tensor containing the number of samples for each class
    """
    # Calculate weights for each class: inverse of class frequency
    class_weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
    
    # Assign weight to each sample in the dataset based on its class
    # Assumes dataset has a .labels attribute
    sample_weights = [class_weights[label] for label in dataset.labels]
    
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    return sampler

# Example usage test
if __name__ == "__main__":
    print("Testing Focal Loss Initialization...")
    criterion = FocalLoss(gamma=2.0)
    dummy_outputs = torch.randn(10, 115) # 10 samples, 115 PlantSeg classes
    dummy_targets = torch.randint(0, 115, (10,))
    loss = criterion(dummy_outputs, dummy_targets)
    print(f"Focal Loss successfully computed: {loss.item():.4f}")
