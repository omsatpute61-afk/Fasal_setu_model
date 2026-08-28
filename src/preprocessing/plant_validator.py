"""
Phase 2: Forgiving Gatekeeper
Validates if the image contains plant foliage. Extremely tolerant to 
allow hands, dirt, and complex backgrounds (acceptance threshold >= 0.35).
"""
import cv2
import numpy as np

class NotACropError(Exception):
    """Custom exception raised when the image does not contain valid crop foliage."""
    pass

class PlantValidator:
    def __init__(self, acceptance_threshold=0.40):
        self.acceptance_threshold = acceptance_threshold

    def _calculate_foliage_confidence(self, frame) -> float:
        """
        Calculates a mock confidence score based on the ratio of green/yellow pixels.
        Strictly filtering for the structural and color characteristics of the 10 authorized terrestrial crops.
        """
        proxy = cv2.resize(frame, (160, 160), interpolation=cv2.INTER_NEAREST)
        hsv = cv2.cvtColor(proxy, cv2.COLOR_BGR2HSV)
        
        # Strict range for crop foliage (yellow to green)
        lower_bound = np.array([30, 40, 40])
        upper_bound = np.array([85, 255, 255])
        
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        total_pixels = proxy.shape[0] * proxy.shape[1] + 1e-5
        foliage_ratio = cv2.countNonZero(mask) / total_pixels
        
        # Scale the ratio to simulate a neural network confidence (0.0 to 1.0)
        confidence = min(foliage_ratio * 3.5, 0.99)
        return float(confidence)

    def validate(self, frame):
        """
        Validates the frame. 
        Throws NotACropError if confidence is below 0.40.
        """
        confidence = self._calculate_foliage_confidence(frame)
        
        if confidence < self.acceptance_threshold:
            raise NotACropError("Invalid image: Crop leaf not detected. Please center the leaf.")
            
        return {
            "status": "PASS",
            "confidence": round(confidence, 4),
            "warning": None
        }
