import cv2
import numpy as np
import logging
from typing import Dict, Any, Tuple

# Configure production logger (bypasses Render's print buffering)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("PlantValidator")

class NotACropError(Exception):
    """Custom exception raised when the image does not contain valid crop foliage."""
    pass

class ProductionPlantValidator:
    def __init__(self, color_threshold: float = 0.15, structure_threshold: float = 0.10) -> None:
        """
        :param color_threshold: At least 15% of the total image must be biological green.
        :param structure_threshold: The single largest green object must occupy at least 10% of the image.
        """
        self.color_threshold: float = color_threshold
        self.structure_threshold: float = structure_threshold
        
        # Tightened biological foliage color bounds (Hue 35 to 77)
        self.lower_green: np.ndarray[Any, Any] = np.array([35, 40, 40])
        self.upper_green: np.ndarray[Any, Any] = np.array([77, 255, 255])

    def _analyze_structure(self, frame: np.ndarray[Any, Any]) -> Tuple[float, float, bool]:
        """
        Analyzes color density and structural integrity (contours) of the foliage.
        """
        # Resize for fast, standardized processing
        img: np.ndarray[Any, Any] = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        hsv: np.ndarray[Any, Any] = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # 1. Color Extraction
        mask: np.ndarray[Any, Any] = cv2.inRange(hsv, self.lower_green, self.upper_green)
        total_pixels: float = float(img.shape[0] * img.shape[1])
        color_density: float = float(cv2.countNonZero(mask)) / total_pixels
        
        # 2. Noise Reduction (Morphological Math)
        # This erases thin lines (stitching) and small dots (fabric patterns)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel)
        
        # 3. Contour Analysis (Finding physical objects)
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        max_contour_area: float = 0.0
        if contours:
            # Find the largest single green object in the image
            largest_contour = max(contours, key=cv2.contourArea)
            max_contour_area = float(cv2.contourArea(largest_contour)) / total_pixels
            
        # Image is valid ONLY if it has enough color AND a large, solid leaf-like structure
        is_valid: bool = (color_density >= self.color_threshold) and (max_contour_area >= self.structure_threshold)
        
        return color_density, max_contour_area, is_valid

    def validate(self, frame: np.ndarray[Any, Any]) -> Dict[str, Any]:
        """
        Validates the frame for production inference.
        """
        try:
            color_density, max_contour_area, is_valid = self._analyze_structure(frame)
            
            logger.info(f"Diagnostics | Density: {color_density:.3f} | Largest Structure: {max_contour_area:.3f}")
            
            if not is_valid:
                logger.warning(f"REJECTED: Failed structural biological check. Score: {max_contour_area:.3f}")
                raise NotACropError("Invalid image. Please center the leaf and ensure it fills the frame.")
                
            # Realistic confidence scaling based on the physical size of the leaf
            confidence: float = min((max_contour_area / 0.50), 0.99)
            
            logger.info("ACCEPTED: Biological structure verified. Passing to ML model.")
            return {
                "status": "PASS",
                "confidence": round(confidence, 4),
                "warning": None
            }
            
        except Exception as e:
            if isinstance(e, NotACropError):
                raise
            logger.error(f"Validator crashed during processing: {str(e)}")
            raise NotACropError("Image processing error. Please upload a clear photo.")
