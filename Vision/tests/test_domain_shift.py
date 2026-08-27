import pytest
import numpy as np
import cv2

from src.data.augmentations import get_field_simulation_pipeline, apply_augmentation
from src.models.alert_engine import DecisionEngine

def test_domain_shift_robustness():
    """
    Tests if the DecisionEngine can handle images passed through the 
    heaviest Albumentations noise pipeline without breaking JSON output schemas.
    """
    engine = DecisionEngine()
    pipeline = get_field_simulation_pipeline(image_size=(640, 640))
    
    # Create a dummy healthy-looking image
    dummy_image = np.ones((1080, 1920, 3), dtype=np.uint8) * 100
    dummy_image[400:600, 800:1000] = [0, 200, 0] # Green patch
    
    # Apply severe domain-shift augmentations
    augmented_image = apply_augmentation(dummy_image, pipeline)
    
    # Make sure augmented image is valid
    assert augmented_image is not None
    assert augmented_image.shape == (640, 640, 3)
    
    # Because our engine currently falls back to mocks if .onnx aren't present,
    # this will test the engine's resilience and parsing schema.
    health_card = engine.analyze_crop(augmented_image)
    
    # Assert JSON Schema constraints
    assert "timestamp" in health_card
    assert "inference_time_seconds" in health_card
    assert "overall_status" in health_card
    assert "diagnostics" in health_card
    
    diagnostics = health_card["diagnostics"]
    
    assert "disease_analysis" in diagnostics
    assert "model" in diagnostics["disease_analysis"]
    assert diagnostics["disease_analysis"]["model"] == "disease"
    
    assert "pest_analysis" in diagnostics
    assert "economic_threshold_exceeded" in diagnostics["pest_analysis"]
    
    assert "nutrient_analysis" in diagnostics
    assert "chlorosis_percentage" in diagnostics["nutrient_analysis"]
    
    print("Domain shift schema validation passed successfully!")
