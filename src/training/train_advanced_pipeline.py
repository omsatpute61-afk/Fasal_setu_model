import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
import timm
import onnx
import onnxruntime as ort
import numpy as np
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from typing import Tuple, Dict, Any, List

# --- 1. DATA AUGMENTATION ---
def get_train_transforms(img_size: int) -> T.Compose:
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0), interpolation=T.InterpolationMode.BILINEAR),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.2),
        T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.25, hue=0.05),
        T.RandomAffine(degrees=15, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        T.RandomApply([T.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 2.0))], p=0.3),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def get_val_transforms(img_size: int) -> T.Compose:
    return T.Compose([
        T.Resize((img_size, img_size)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

# --- 2. FOCAL LOSS WITH LABEL SMOOTHING & WEIGHTING ---
class FocalLoss(nn.Module):
    def __init__(self, alpha: torch.Tensor, gamma: float = 2.0, label_smoothing: float = 0.10):
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Shape: (C,)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Cross entropy loss with label smoothing
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', label_smoothing=self.label_smoothing)
        pt = torch.exp(-ce_loss)
        
        # Get alpha for each target
        alpha_t = self.alpha[targets].to(inputs.device)
        
        focal_loss = alpha_t * ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# --- 3. TEMPERATURE SCALING & CALIBRATION ---
class TemperatureScaler(nn.Module):
    def __init__(self):
        super(TemperatureScaler, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.5)

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.temperature

    def set_temperature(self, valid_loader: DataLoader, model: nn.Module, device: str):
        self.to(device)
        model.eval()
        nll_criterion = nn.CrossEntropyLoss().to(device)
        
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                inputs = inputs.to(device)
                logits = model(inputs)
                logits_list.append(logits)
                labels_list.append(labels)
                
        logits = torch.cat(logits_list).to(device)
        labels = torch.cat(labels_list).to(device)
        
        optimizer = torch.optim.LBFGS([self.temperature], lr=0.01, max_iter=50)
        
        def eval():
            optimizer.zero_grad()
            loss = nll_criterion(self.forward(logits), labels)
            loss.backward()
            return loss
            
        optimizer.step(eval)
        print(f"Optimal Temperature: {self.temperature.item():.4f}")

def expected_calibration_error(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true)
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        if prop_in_bin > 0:
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    return ece

# --- 4. OOD & REJECTION LOGIC ---
class InferenceRejector:
    def __init__(self, msp_threshold: float = 0.80, entropy_threshold: float = 0.35, healthy_override: float = 0.85):
        self.msp_threshold = msp_threshold
        self.entropy_threshold = entropy_threshold
        self.healthy_override = healthy_override
        
    def evaluate(self, probs: np.ndarray, healthy_idx: int) -> Tuple[bool, str]:
        C = probs.shape[0]
        max_prob = np.max(probs)
        pred_idx = np.argmax(probs)
        
        # 1. Maximum Softmax Probability
        if max_prob < self.msp_threshold:
            return True, "ESCALATE_TO_KVK: Low Confidence (MSP)"
            
        # 2. Normalized Entropy
        entropy = -np.sum(probs * np.log(probs + 1e-9)) / np.log(C)
        if entropy > self.entropy_threshold:
            return True, "ESCALATE_TO_KVK: High Entropy (OOD)"
            
        # 3. Healthy Override Trap
        if pred_idx == healthy_idx and max_prob < self.healthy_override:
            return True, "ESCALATE_TO_KVK: Unconfident Healthy Override"
            
        return False, "VALID"

# --- 5. MODEL TRAINER ---
class ModelTrainer:
    def __init__(self, model: nn.Module, device: str, alpha_weights: torch.Tensor, num_epochs: int = 50):
        self.model = model.to(device)
        self.device = device
        self.num_epochs = num_epochs
        
        self.criterion = FocalLoss(alpha=alpha_weights, gamma=2.0, label_smoothing=0.10)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-4, betas=(0.9, 0.999), weight_decay=1e-2)
        
        # Simplified scheduler setup
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=num_epochs, eta_min=1e-6)
        self.scaler = torch.cuda.amp.GradScaler()
        
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> nn.Module:
        best_f1 = 0.0
        
        for epoch in range(self.num_epochs):
            self.model.train()
            train_loss = 0.0
            
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                
                with torch.cuda.amp.autocast():
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                train_loss += loss.item()
                
            self.scheduler.step()
            
            # Validation
            val_loss, y_true, y_pred, y_prob = self.validate(val_loader)
            
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            b_acc = balanced_accuracy_score(y_true, y_pred)
            
            print(f"Epoch {epoch+1}/{self.num_epochs} - Loss: {train_loss/len(train_loader):.4f} | Val F1: {f1:.4f} | B-Acc: {b_acc:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                torch.save(self.model.state_dict(), "best_model.pth")
                
        self.model.load_state_dict(torch.load("best_model.pth"))
        return self.model

    def validate(self, val_loader: DataLoader):
        self.model.eval()
        total_loss = 0.0
        y_true, y_pred, y_prob = [], [], []
        
        with torch.no_grad():
            for inputs, targets in val_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
                
                probs = F.softmax(outputs, dim=1)
                preds = torch.argmax(probs, dim=1)
                
                y_true.extend(targets.cpu().numpy())
                y_pred.extend(preds.cpu().numpy())
                y_prob.extend(probs.cpu().numpy())
                
        return total_loss/len(val_loader), np.array(y_true), np.array(y_pred), np.array(y_prob)

# --- 6. ONNX EXPORT ENGINE ---
class ONNXExporter:
    @staticmethod
    def export(model: nn.Module, temperature_scaler: TemperatureScaler, img_size: int, output_path: str):
        # Create a combined calibrated model wrapper
        class CalibratedModel(nn.Module):
            def __init__(self, base_model, scaler):
                super().__init__()
                self.base = base_model
                self.scaler = scaler
                
            def forward(self, x):
                logits = self.base(x)
                calibrated_logits = self.scaler(logits)
                return F.softmax(calibrated_logits, dim=1)
                
        calibrated_model = CalibratedModel(model, temperature_scaler)
        calibrated_model.eval()
        
        dummy_input = torch.randn(1, 3, img_size, img_size, device="cpu")
        calibrated_model.to("cpu")
        
        torch.onnx.export(
            calibrated_model,
            dummy_input,
            output_path,
            export_params=True,
            opset_version=17,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['probabilities'],
            dynamic_axes={
                'input': {0: 'batch_size'},
                'probabilities': {0: 'batch_size'}
            }
        )
        print(f"Exported ONNX model to {output_path}")
        
    @staticmethod
    def verify(onnx_path: str, img_size: int):
        session = ort.InferenceSession(onnx_path, providers=['CPUExecutionProvider'])
        dummy_input = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
        
        # Single batch verification
        out_single = session.run(['probabilities'], {'input': dummy_input})[0]
        print(f"ONNX Verification (Single): Probabilities shape {out_single.shape} | Sum: {np.sum(out_single):.4f}")
        
        # Dynamic batch verification
        dummy_batch = np.random.randn(4, 3, img_size, img_size).astype(np.float32)
        out_batch = session.run(['probabilities'], {'input': dummy_batch})[0]
        print(f"ONNX Verification (Batch 4): Probabilities shape {out_batch.shape}")

# --- MAIN ENTRY POINT ---
def main():
    # 1. Configuration (Example setup for Disease Stream)
    num_classes = 68
    img_size = 224
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print(f"Starting Training Pipeline on {device}...")
    
    # 2. Model Instantiation
    model = timm.create_model('mobilenetv4_conv_small', pretrained=True, num_classes=num_classes)
    
    # 3. Dummy Data setup (Replace with real datasets)
    # Using reciprocal weights for FocalLoss
    dummy_class_counts = np.random.randint(10, 1000, size=num_classes)
    alpha_weights = 1.0 / np.sqrt(dummy_class_counts)
    alpha_weights = torch.tensor(alpha_weights / np.sum(alpha_weights), dtype=torch.float32)
    
    print("Pipeline architecture and logic structured. Plug in Dataset/DataLoaders to execute.")
    # Excluded actual DataLoader boilerplates for brevity, logic is complete.
    
    # Example execution flow:
    # trainer = ModelTrainer(model, device, alpha_weights)
    # trained_model = trainer.train(train_loader, val_loader)
    
    # scaler = TemperatureScaler()
    # scaler.set_temperature(val_loader, trained_model, device)
    
    # ONNXExporter.export(trained_model, scaler, img_size, "disease_calibrated.onnx")
    # ONNXExporter.verify("disease_calibrated.onnx", img_size)
    
if __name__ == "__main__":
    main()

