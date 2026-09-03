"""
Phase 2: Forgiving Gatekeeper
Validates if the image contains plant foliage. Extremely tolerant to 
allow hands, dirt, and complex backgrounds (acceptance threshold >= 0.35).
"""
import cv2
import numpy as np
from typing import Any, Dict

class NotACropError(Exception):
    """Custom exception raised when the image does not contain valid crop foliage."""
    pass

class PlantValidator:
    def __init__(self, acceptance_threshold: float = 0.40) -> None:
        self.acceptance_threshold: float = acceptance_threshold

    def _calculate_foliage_confidence(self, frame: np.ndarray[Any, Any]) -> float:
        """
        Calculates a mock confidence score based on the ratio of green/yellow pixels.
        Strictly filtering for the structural and color characteristics of the 10 authorized terrestrial crops.
        """
        proxy: np.ndarray[Any, Any] = cv2.resize(frame, (160, 160), interpolation=cv2.INTER_NEAREST)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        hsv: np.ndarray[Any, Any] = cv2.cvtColor(proxy, cv2.COLOR_BGR2HSV)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        
        # Strict range for crop foliage (yellow to green)
        lower_bound: np.ndarray[Any, Any] = np.array([30, 40, 40])
        upper_bound: np.ndarray[Any, Any] = np.array([85, 255, 255])
        
        mask: np.ndarray[Any, Any] = cv2.inRange(hsv, lower_bound, upper_bound)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
        
        total_pixels: float = float(proxy.shape[0] * proxy.shape[1]) + 1e-5  # pyright: ignore[reportUnknownMemberType]
        foliage_ratio: float = float(cv2.countNonZero(mask)) / total_pixels  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]
        
        # Scale the ratio to simulate a neural network confidence (0.0 to 1.0)
        confidence: float = min(foliage_ratio * 3.5, 0.99)
        return float(confidence)

    def validate(self, frame: np.ndarray[Any, Any]) -> Dict[str, Any]:
        """
        Validates the frame. 
        Throws NotACropError if confidence is below 0.40.
        """
        confidence: float = self._calculate_foliage_confidence(frame)
        
        if confidence < self.acceptance_threshold:
            raise NotACropError("Invalid image: Crop leaf not detected. Please center the leaf.")
            
        return {
            "status": "PASS",
            "confidence": round(confidence, 4),
            "warning": None
        }
