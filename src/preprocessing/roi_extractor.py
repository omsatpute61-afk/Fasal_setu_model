"""
Phase 6: High-Detail Leaf ROI Auto-Zoom
Extracts the primary foliage region from the frame, applies a 10% context margin,
and scales it to 512x512. This prevents critical small details (tiny pests, early lesions) 
from being destroyed by standard bilinear downsampling.
"""
import cv2
import numpy as np

class LeafROIExtractor:
    def __init__(self, target_size=(512, 512), context_margin=0.1):
        self.target_size = target_size
        self.context_margin = context_margin

    def extract_roi(self, frame):
        """
        Uses HSV masking and contour detection to find the tight bounding box
        of the primary foliage cluster.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Broad spectrum for foliage (Green & Yellow)
        lower_bound = np.array([20, 30, 30])
        upper_bound = np.array([90, 255, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Clean up the mask using morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            # If no contours found, return the resized original frame as fallback
            return cv2.resize(frame, self.target_size)
            
        # Find the largest contour assuming it's the primary leaf
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Get bounding box
        x, y, w, h = cv2.boundingRect(largest_contour)
        
        # Add 10% context margin
        margin_x = int(w * self.context_margin)
        margin_y = int(h * self.context_margin)
        
        x_min = max(0, x - margin_x)
        y_min = max(0, y - margin_y)
        x_max = min(frame.shape[1], x + w + margin_x)
        y_max = min(frame.shape[0], y + h + margin_y)
        
        # Crop the high-resolution patch
        cropped_roi = frame[y_min:y_max, x_min:x_max]
        
        # Scale ONLY the leaf ROI to input dimensions
        final_roi = cv2.resize(cropped_roi, self.target_size, interpolation=cv2.INTER_AREA)
        
        return final_roi
