import torch
import torch.nn as nn
import torchvision.models as models
from torchvision.models import EfficientNet_B0_Weights

def get_disease_model(num_classes=3):
    """
    Loads pre-trained EfficientNet, freezes base layers, and replaces classification head.
    """
    # Load EfficientNet-B0 pre-trained on ImageNet
    model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
    
    # Freeze all base feature-extraction layers
    for param in model.features.parameters():
        param.requires_grad = False
        
    # Replace the classification head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    return model

def dummy_train_loop():
    """
    Simulates the training loop.
    """
    model = get_disease_model()
    print("Disease Model Architecture initialized for Transfer Learning.")
    print("Base layers frozen. Classifier head replaced.")
    
    # In a real scenario, we would load a DataLoader with PlantDoc datasets 
    # and use torch.optim to train only the classifier parameters.
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    
    print("Dummy training loop executed successfully.")
    return model

if __name__ == "__main__":
    dummy_train_loop()
