import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms
import timm
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np

# Dummy Dataset for structure, since real data isn't provided here
class AgriculturalDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.image_paths = []
        self.labels = []
        # In a real scenario, this would load from a directory structure (e.g. ImageFolder)
        
    def __len__(self):
        return len(self.image_paths) if len(self.image_paths) > 0 else 100 # Dummy length
        
    def __getitem__(self, idx):
        # Create a dummy image mimicking a camera frame
        img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
        label = np.random.randint(0, 10)
        
        if self.transform:
            augmented = self.transform(image=img)
            img = augmented['image']
            
        return img, label

def get_train_transforms():
    """
    Data Augmentation simulating harsh field conditions from budget smartphones.
    """
    return A.Compose([
        A.Resize(224, 224),
        A.RandomBrightnessContrast(p=0.5),
        A.GaussNoise(p=0.3),
        A.MotionBlur(p=0.3),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_val_transforms():
    """
    Validation transforms (resize and normalize only).
    """
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def train():
    print("Initializing MobileNetV4 Disease Classifier (SOTA Pivot)...")
    
    # Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 1. MODEL INITIALIZATION
    num_classes = 10
    model = timm.create_model('mobilenetv4_conv_small', pretrained=True, num_classes=num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    # 2. DATA AUGMENTATION (BUDGET CAMERA SIMULATION)
    train_dataset = AgriculturalDataset("data/train", transform=get_train_transforms())
    val_dataset = AgriculturalDataset("data/val", transform=get_val_transforms())
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    
    # 3. TRAINING LOOP
    epochs = 30
    best_val_acc = 0.0
    save_dir = "src/weights"
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, "best_disease_model.pth")
    
    print(f"Starting training for {epochs} epochs...")
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        train_loss /= len(train_loader)
        
        # Validation Loop
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
                
        val_loss /= len(val_loader)
        val_acc = 100. * correct / total
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        # Save Best Weights
        if val_acc > best_val_acc:
            print(f"Validation accuracy improved ({best_val_acc:.2f}% -> {val_acc:.2f}%). Saving model...")
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            
    print(f"Training complete. Best model saved to {save_path} with {best_val_acc:.2f}% accuracy.")

if __name__ == "__main__":
    train()
