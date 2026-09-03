import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split, Subset
import timm
import albumentations as A  # pyright: ignore[reportMissingTypeStubs]
from albumentations.pytorch import ToTensorV2  # pyright: ignore[reportMissingTypeStubs]
import numpy as np
from PIL import Image
from typing import Any, Dict, List, Tuple, cast

# --- 1. FLATTENED DATASET WRAPPER (Custom built for Kaggle Nested Folders) ---
class FlattenedCropDataset(Dataset[Tuple[np.ndarray, int]]):
    def __init__(self, root_dir: str) -> None:
        self.image_paths: List[str] = []
        self.labels: List[int] = []
        
        # Scan for leaf-node folders that actually contain images (the disease-level folders)
        class_folders: List[str] = []
        for dirpath, _, filenames in os.walk(root_dir):
            if any(f.lower().endswith(('.jpg', '.jpeg', '.png')) for f in filenames):
                class_folders.append(dirpath)
                
        # Sort and map the disease folder names as our classes
        self.classes: List[str] = sorted([os.path.basename(folder) for folder in class_folders])
        self.class_to_idx: Dict[str, int] = {cls: i for i, cls in enumerate(self.classes)}
        
        # Map every single image to its correct disease class index
        for folder in class_folders:
            cls_name: str = os.path.basename(folder)
            idx: int = self.class_to_idx[cls_name]
            for f in os.listdir(folder):
                if f.lower().endswith(('.jpg', '.jpeg', '.png')):
                    self.image_paths.append(os.path.join(folder, f))
                    self.labels.append(idx)

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, int]:
        # Return a raw numpy array and the label so Albumentations can modify it later
        img = Image.open(self.image_paths[idx]).convert('RGB')
        return np.array(img), self.labels[idx]

# Wrapper to apply different transforms to Train vs Valid splits
class TransformSubsetWrapper(Dataset[Tuple[torch.Tensor, int]]):
    def __init__(self, subset: Subset[Tuple[np.ndarray, int]], transform: Any) -> None:
        self.subset = subset
        self.transform = transform
        
    def __len__(self) -> int:
        return len(self.subset)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        img_array, label = self.subset[idx]
        if self.transform:
            transformed = self.transform(image=img_array)
            img_array = transformed['image']
        # Albumentations ToTensorV2 returns a PyTorch tensor
        return cast(torch.Tensor, img_array), label

# --- 2. TRANSFORMS ---
def get_train_transforms() -> Any:
    return A.Compose([
        A.Resize(224, 224),
        A.RandomBrightnessContrast(p=0.5),
        A.HorizontalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

def get_val_transforms() -> Any:
    return A.Compose([
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])

# --- 3. MAIN TRAINING LOOP ---
def train() -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Fresh Start: Using device: {device}")
    
    # EXACT HARDCODED PATH FROM YOUR X-RAY SCAN
    data_dir: str = "/kaggle/input/datasets/salmasyed1360/plant-diseases-100k-labelled-images/crops"
    
    if not os.path.exists(data_dir):
        print(f"CRITICAL: Could not find {data_dir}. Path is incorrect!")
        return

    print("Scanning and Flattening the Dataset...")
    master_dataset = FlattenedCropDataset(data_dir)
    
    num_classes: int = len(master_dataset.classes)
    total_images: int = len(master_dataset)
    
    if total_images == 0:
        print("CRITICAL: Found 0 images. Stop and verify.")
        return
        
    print(f"✅ Successfully detected {num_classes} disease classes across {total_images} images!")
    
    # 80/20 Split mathematically
    train_size: int = int(0.8 * total_images)
    val_size: int = total_images - train_size
    print(f"📊 Splitting into {train_size} Train | {val_size} Validation")
    
    train_subset: Subset[Tuple[np.ndarray, int]]
    val_subset: Subset[Tuple[np.ndarray, int]]
    train_subset, val_subset = random_split(master_dataset, [train_size, val_size])
    
    # Apply Augmentations
    train_dataset = TransformSubsetWrapper(train_subset, transform=get_train_transforms())
    val_dataset = TransformSubsetWrapper(val_subset, transform=get_val_transforms())
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=2)

    print("Initializing MobileNetV4...")
    model: nn.Module = timm.create_model('mobilenetv4_conv_small', pretrained=True, num_classes=num_classes)
    model = model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    
    epochs: int = 20
    best_val_acc: float = 0.0
    save_dir: str = "/kaggle/working/weights"
    os.makedirs(save_dir, exist_ok=True)
    save_path: str = os.path.join(save_dir, "best_disease_model.pth")
    
    print("Starting Epoch 1...")
    for epoch in range(epochs):
        model.train()
        train_loss: float = 0.0
        
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]
            train_loss += float(loss.item())
            
        train_loss /= len(train_loader)
        
        model.eval()
        val_loss: float = 0.0
        correct: int = 0
        total: int = 0
        
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                
                val_loss += float(loss.item())
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += int(predicted.eq(labels).sum().item())
                
        val_loss /= len(val_loader)
        val_acc: float = 100. * correct / total
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%")
        
        if val_acc > best_val_acc:
            print(f"⭐ New Best Accuracy ({best_val_acc:.2f}% -> {val_acc:.2f}%). Saving weights...")
            best_val_acc = val_acc
            torch.save(model.state_dict(), save_path)
            
    print(f"Training completed successfully! Best model stored at {save_path}")

if __name__ == "__main__":
    train()