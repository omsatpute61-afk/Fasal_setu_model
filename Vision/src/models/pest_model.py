import os
import cv2
import json
import numpy as np

class PestModel:
    """
    Pest Identification Model
    
    Training & Export Strategy:
    - Dataset: IP102 or Pest24 (specialized for bounding-box detection of small pests).
    - Architecture: YOLOv8 Nano or YOLOv11 Nano (INT8 Quantized).
    - Inference: Slicing Aided Hyper Inference (SAHI) or OpenCV frame-tiling to detect 
      very small insects without losing them to image resizing.
    """
    def __init__(self, model_path="models/yolo_pest_int8.onnx", tile_size=640, overlap=0.2):
        self.model_path = model_path
        self.tile_size = tile_size
        self.overlap = overlap
        self.economic_threshold = 5 # Arbitrary threshold for pest count requiring action
        self.model = self._load_model()

    def _load_model(self):
        # Mock load YOLO model
        if os.path.exists(self.model_path):
            return "loaded_model"
        return None

    def _tile_image(self, image):
        """
        Manually tile image using OpenCV for SAHI-like small object detection
        """
        h, w = image.shape[:2]
        tiles = []
        coordinates = []
        stride = int(self.tile_size * (1 - self.overlap))
        
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                y2 = min(y + self.tile_size, h)
                x2 = min(x + self.tile_size, w)
                
                # Adjust if tile is smaller than expected (at edges)
                y1 = max(0, y2 - self.tile_size)
                x1 = max(0, x2 - self.tile_size)
                
                tile = image[y1:y2, x1:x2]
                tiles.append(tile)
                coordinates.append((x1, y1, x2, y2))
                
        return tiles, coordinates

    def predict(self, image):
        """
        Runs inference with frame-tiling for small insects.
        """
        tiles, coordinates = self._tile_image(image)
        all_bboxes = []
        
        if self.model is not None:
            # Perform inference on each tile and map bbox back to original image
            pass
        else:
            # Fallback OpenCV dynamic contour detection for pests (mocking YOLO)
            # Find small dark spots on the leaf
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            blurred = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Adaptive threshold to find high-contrast spots
            thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for cnt in contours:
                area = cv2.contourArea(cnt)
                # A typical small pest might be between 10 and 150 pixels in area
                if 15 < area < 120:
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Ensure it is somewhat bug-shaped (not a long line)
                    if 0.3 < w/float(h) < 3.0:
                        all_bboxes.append({
                            "class": "Pest (Mock)",
                            "confidence": round(0.5 + (min(area, 100)/200.0), 2),
                            "bbox": [x, y, x+w, y+h]
                        })
                        
            # Limit to top 15 to avoid spam from texture
            all_bboxes = sorted(all_bboxes, key=lambda x: x['confidence'], reverse=True)[:15]
            
        total_pest_count = len(all_bboxes)
        threshold_exceeded = total_pest_count >= self.economic_threshold
        
        return {
            "model": "pest",
            "total_pest_count": total_pest_count,
            "pest_bounding_boxes": all_bboxes,
            "economic_threshold_exceeded": threshold_exceeded
        }

if __name__ == "__main__":
    model = PestModel()
    dummy_image = np.zeros((1080, 1920, 3), dtype=np.uint8)
    print(json.dumps(model.predict(dummy_image), indent=2))
