import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms  # pyright: ignore[reportMissingTypeStubs]
from PIL import Image
import timm
from sklearn.model_selection import train_test_split  # pyright: ignore[reportUnknownVariableType]
from typing import Any, Dict, List, Tuple, cast

class PestopiaDataset(Dataset[Tuple[torch.Tensor, int]]):
    def __init__(self, image_paths: List[str], labels: List[int], transform: Any = None) -> None:
        self.image_paths: List[str] = image_paths
        self.labels: List[int] = labels
        self.transform: Any = transform

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image: Image.Image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        # Ensure we return a tensor
        return cast(torch.Tensor, image), self.labels[idx]

def train_sota_pest_classifier() -> None:
    device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing High-Res Pestopia Training on: {device}")

    data_dir: str = "/kaggle/input/datasets/shruthisindhura/pestopia/Datasets/Pest_Dataset"
    
    all_paths: List[str] = []
    all_labels: List[int] = []
    classes: List[str] = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx: Dict[str, int] = {cls_name: i for i, cls_name in enumerate(classes)}
    
    for cls_name in classes:
        cls_dir: str = os.path.join(data_dir, cls_name)
        for img_name in os.listdir(cls_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_paths.append(os.path.join(cls_dir, img_name))
                all_labels.append(class_to_idx[cls_name])

    print(f"Loaded {len(all_paths)} images across {len(classes)} classes.")

    split_result = cast(
        Tuple[List[str], List[str], List[int], List[int]],
        train_test_split(all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels)
    )
    train_paths, val_paths, train_labels, val_labels = split_result

    # 1. High-Resolution & Advanced Augmentation
    transform_train: Any = transforms.Compose([
        transforms.RandomResizedCrop(384, scale=(0.75, 1.0)), 
        transforms.TrivialAugmentWide(), # SOTA augmentation
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val: Any = transforms.Compose([
        transforms.Resize(400),
        transforms.CenterCrop(384), # High-res inference
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(PestopiaDataset(train_paths, train_labels, transform_train), batch_size=16, shuffle=True, num_workers=2)
    val_loader = DataLoader(PestopiaDataset(val_paths, val_labels, transform_val), batch_size=16, shuffle=False, num_workers=2)

    # 2. Capacity Upgrade to MobileNetV4 Large
    print("Loading MobileNetV4-Large at 384x384...")
    model: nn.Module = timm.create_model('mobilenetv4_conv_large', pretrained=True, num_classes=len(classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=2e-4, weight_decay=0.01)
    
    epochs: int = 15
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc: float = 0.0
    os.makedirs('/kaggle/working/weights', exist_ok=True)
    save_path: str = '/kaggle/working/weights/best_pest_model_sota.pth'

    for epoch in range(epochs):
        model.train()
        running_loss: float = 0.0
        for inputs, labels in train_loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()  # pyright: ignore[reportUnknownMemberType]
            running_loss += float(loss.item())

        scheduler.step()

        model.eval()
        correct: int = 0
        total: int = 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs = inputs.to(device)
                labels = labels.to(device)
                outputs = model(inputs)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += int((predicted == labels).sum().item())

        val_acc: float = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)
            print(f"⭐ New Best: {best_acc:.2f}% saved.")
            
    print(f"Download your weights from: {save_path}")

if __name__ == "__main__":
    train_sota_pest_classifier()