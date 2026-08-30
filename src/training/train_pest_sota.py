import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import timm
from sklearn.model_selection import train_test_split

class PestopiaDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform

    def __len__(self): return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform: image = self.transform(image)
        return image, self.labels[idx]

def train_high_acc_classifier():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing High-Accuracy Pestopia Training on: {device}")

    data_dir = "/kaggle/input/datasets/shruthisindhura/pestopia/Datasets/Pest_Dataset"
    
    # 1. Parse Directory
    all_paths, all_labels = [], []
    classes = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    
    for cls_name in classes:
        cls_dir = os.path.join(data_dir, cls_name)
        for img_name in os.listdir(cls_dir):
            if img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
                all_paths.append(os.path.join(cls_dir, img_name))
                all_labels.append(class_to_idx[cls_name])

    # 2. Stratified Split (Ensures equal class representation)
    train_paths, val_paths, train_labels, val_labels = train_test_split(
        all_paths, all_labels, test_size=0.2, random_state=42, stratify=all_labels
    )

    # 3. Aggressive Data Augmentation for Training
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    # Clean Transforms for Validation
    transform_val = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    train_loader = DataLoader(PestopiaDataset(train_paths, train_labels, transform_train), batch_size=32, shuffle=True, num_workers=2)
    val_loader = DataLoader(PestopiaDataset(val_paths, val_labels, transform_val), batch_size=32, shuffle=False, num_workers=2)

    # 4. Upgraded Architecture
    print("Loading MobileNetV4-Medium...")
    model = timm.create_model('mobilenetv4_conv_medium', pretrained=True, num_classes=len(classes))
    model = model.to(device)

    # 5. Advanced Optimization
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.01)
    
    epochs = 20
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    best_acc = 0.0
    os.makedirs('/kaggle/working/weights', exist_ok=True)
    save_path = '/kaggle/working/weights/best_pest_model.pth'

    print("Commencing Training Loop...")
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Step the scheduler
        scheduler.step()

        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                outputs = model(inputs.to(device))
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels.to(device)).sum().item()

        val_acc = 100 * correct / total
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Epoch {epoch+1}/{epochs} | LR: {current_lr:.6f} | Loss: {running_loss/len(train_loader):.4f} | Val Acc: {val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), save_path)

    print(f"\n✅ Training Complete! Peak Validation Accuracy: {best_acc:.2f}%")
    print(f"Model saved to: {save_path}")

if __name__ == "__main__":
    train_high_acc_classifier()