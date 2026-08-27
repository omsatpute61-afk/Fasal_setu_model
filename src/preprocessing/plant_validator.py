"""
Phase 2: Forgiving Gatekeeper
Validates if the image contains plant foliage. Extremely tolerant to 
allow hands, dirt, and complex backgrounds (acceptance threshold >= 0.35).
"""
import cv2
import numpy as np

class PlantValidator:
    def __init__(self, acceptance_threshold=0.35, soft_pass_threshold=0.55):
        self.acceptance_threshold = acceptance_threshold
        self.soft_pass_threshold = soft_pass_threshold

    def _calculate_foliage_confidence(self, frame) -> float:
        """
        Calculates a mock confidence score based on the ratio of green/yellow pixels.
        In production, this would be a lightweight MobileNetV3 binary classifier.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Broad range for foliage (yellow and green)
        lower_bound = np.array([25, 30, 30])
        upper_bound = np.array([90, 255, 255])
        
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Calculate ratio of foliage to total pixels
        total_pixels = frame.shape[0] * frame.shape[1] + 1e-5
        foliage_ratio = cv2.countNonZero(mask) / total_pixels
        
        # Scale the ratio to simulate a neural network confidence (0.0 to 1.0)
        confidence = min(foliage_ratio * 3.0, 0.99)
        return float(confidence)

    def validate(self, frame):
        """
        Validates the frame. 
        NEVER throws a hard blocking error if hands/dirt are in the frame as long as
        confidence >= 0.35.
        """
        confidence = self._calculate_foliage_confidence(frame)
        
        if confidence >= self.soft_pass_threshold:
            return {
                "status": "PASS",
                "confidence": round(confidence, 4),
                "warning": None
            }
        elif confidence >= self.acceptance_threshold:
            return {
                "status": "SOFT_PASS",
                "confidence": round(confidence, 4),
                "warning": "Partial foliage. Scanning with adapted sensitivity."
            }
        else:
            return {
                "status": "REJECT",
                "confidence": round(confidence, 4),
                "warning": "No foliage detected. Please aim at a crop."
            }
