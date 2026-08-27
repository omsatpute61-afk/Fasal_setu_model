import os
import cv2
import numpy as np

try:
    import torch
    import torch.nn as nn
    import torchvision.models as models
    from torchvision.models import MobileNet_V3_Small_Weights
except ImportError:
    torch = None

class PlantValidator:
    """
    Stage 1 Gatekeeper: Ultra-lightweight binary classifier (Plant vs Non-Plant).
    Designed for sub-15ms inference on Edge Android devices.
    """
    def __init__(self, model_path="models/plant_validator_int8.onnx", threshold=0.85):
        self.model_path = model_path
        self.threshold = threshold
        self.model = self._load_model()
        
    def _get_pytorch_model(self):
        """Returns the PyTorch MobileNetV3-Small architecture for validation/export."""
        if not torch:
            return None
        model = models.mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        # Modify classification head for binary output
        in_features = model.classifier[3].in_features
        model.classifier[3] = nn.Sequential(
            nn.Linear(in_features, 1),
            nn.Sigmoid()
        )
        return model

    def _load_model(self):
        """Mock loading the ONNX/TFLite edge model."""
        if os.path.exists(self.model_path):
            return "loaded_edge_model"
        return None

    def preprocess(self, frame):
        """Lightweight preprocessing (resize & normalize)"""
        resized = cv2.resize(frame, (224, 224))
        # Simple normalization to avoid heavy compute
        normalized = resized.astype(np.float32) / 255.0
        return normalized

    def validate(self, frame):
        """
        Runs the binary classification.
        Returns dict with is_plant bool, confidence, and message.
        """
        # _ = self.preprocess(frame)
        
        # If we had the real model, we would run it here.
        # For the prototype, we will simulate the validation based on the image's dominant color.
        # If it's mostly black/dark, or lacks green/yellow (like soil/hands), we reject it.
        
        if self.model is not None:
            # Execute real ONNX/TFLite inference
            pass
            
        # Fallback Mock Logic: Check if there's sufficient "green/yellow" pixel mass to simulate a leaf
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # Broad range for foliage (yellow and green)
        mask = cv2.inRange(hsv, np.array([25, 30, 30]), np.array([90, 255, 255]))
        foliage_ratio = cv2.countNonZero(mask) / (frame.shape[0] * frame.shape[1] + 1e-5)
        
        # We simulate a confidence score based on the foliage ratio
        confidence = min(foliage_ratio * 2.0, 0.99) # arbitrary scaling for mock
        
        if confidence >= self.threshold:
            return {
                "is_plant": True,
                "confidence": round(confidence, 4),
                "message": "Valid crop foliage detected."
            }
        else:
            return {
                "is_plant": False,
                "confidence": round(confidence, 4),
                "message": '["Object is not in the context to model"]'
            }

if __name__ == "__main__":
    validator = PlantValidator()
    
    print("Testing with dummy empty frame...")
    empty_frame = np.zeros((224, 224, 3), dtype=np.uint8)
    print(validator.validate(empty_frame))
    
    print("\nTesting with dummy green frame...")
    green_frame = np.zeros((224, 224, 3), dtype=np.uint8)
    green_frame[:, :] = [0, 200, 0] # Green
    print(validator.validate(green_frame))
