import os
import cv2
import json
import numpy as np

try:
    import onnxruntime as ort
except ImportError:
    ort = None

class DiseaseModel:
    """
    Crop Disease Detection Model
    
    Training & Export Strategy:
    - 2-stage transfer learning approach: 
      1. Pretraining on PlantVillage for morphological features.
      2. Fine-tuning on PlantDoc for real-world field robustness and complex backgrounds.
    - Model architecture: MobileNetV3 or EfficientNet-Lite (INT8 Quantized).
    - Export format: ONNX or TFLite.
    """
    def __init__(self, model_path="models/disease_model_int8.onnx"):
        self.model_path = model_path
        self.model = self._load_model()

    def _load_model(self):
        if ort and os.path.exists(self.model_path):
            try:
                # Load optimized ONNX model for edge
                session_options = ort.SessionOptions()
                session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                return ort.InferenceSession(self.model_path, session_options)
            except Exception as e:
                print(f"Error loading disease model: {e}")
                return None
        return None

    def preprocess(self, image):
        """
        Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) 
        to improve contrast in field conditions.
        """
        # Convert to LAB color space
        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Apply CLAHE to L-channel
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        
        # Merge channels
        limg = cv2.merge((cl, a, b))
        enhanced_img = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        
        # Resize to expected input size (e.g., 224x224)
        resized = cv2.resize(enhanced_img, (224, 224))
        
        # Normalize to float32 or int8 depending on the model, assuming float32 for fallback
        normalized = resized.astype(np.float32) / 255.0
        normalized = np.transpose(normalized, (2, 0, 1)) # HWC to CHW
        normalized = np.expand_dims(normalized, axis=0)
        return normalized

    def predict(self, image):
        """
        Runs inference on the image and returns a JSON dictionary.
        """
        preprocessed_img = self.preprocess(image)
        
        if self.model is not None:
            # ONNX inference
            input_name = self.model.get_inputs()[0].name
            outputs = self.model.run(None, {input_name: preprocessed_img})
            
            # Mock parsing logic based on output
            predicted_class_idx = np.argmax(outputs[0])
            confidence = float(np.max(outputs[0]))
            
            # Dummy map
            classes = ["Healthy", "Early Blight", "Late Blight"]
            disease_class = classes[predicted_class_idx]
            requires_action = disease_class != "Healthy" and confidence > 0.8
            
            return {
                "model": "disease",
                "crop_detected": "Tomato",
                "disease_class": disease_class,
                "confidence": confidence,
                "requires_immediate_action": requires_action
            }
        else:
            # Fallback mock heuristic using color analysis
            # We check the original image for brown spots
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Count brown pixels (blight) vs green pixels (healthy)
            lower_brown = np.array([10, 50, 20])
            upper_brown = np.array([25, 255, 200])
            mask_brown = cv2.inRange(hsv, lower_brown, upper_brown)
            
            lower_green = np.array([25, 40, 40])
            upper_green = np.array([90, 255, 255])
            mask_green = cv2.inRange(hsv, lower_green, upper_green)
            
            brown_pixels = cv2.countNonZero(mask_brown)
            green_pixels = cv2.countNonZero(mask_green)
            
            total = brown_pixels + green_pixels + 1e-5
            brown_ratio = brown_pixels / total
            
            if brown_ratio > 0.15:
                disease_class = "Early Blight"
                confidence = round(0.75 + brown_ratio, 2)
                requires_action = True
            elif brown_ratio > 0.05:
                disease_class = "Late Blight"
                confidence = round(0.60 + brown_ratio, 2)
                requires_action = True
            else:
                disease_class = "Healthy"
                confidence = 0.95
                requires_action = False
                
            return {
                "model": "disease",
                "crop_detected": "Leaf",
                "disease_class": disease_class + " (Mock)",
                "confidence": min(confidence, 0.99),
                "requires_immediate_action": requires_action
            }

if __name__ == "__main__":
    model = DiseaseModel()
    dummy_image = np.zeros((500, 500, 3), dtype=np.uint8)
    print(json.dumps(model.predict(dummy_image), indent=2))
