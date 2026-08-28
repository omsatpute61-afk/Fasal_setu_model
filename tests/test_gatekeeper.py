import pytest
import numpy as np
from src.preprocessing.plant_validator import PlantValidator, NotACropError

def test_gatekeeper_rejects_non_crop():
    """
    Ensures the strict gatekeeper raises NotACropError when no foliage is present.
    """
    validator = PlantValidator()
    
    # Create a noisy blank image (definitely not a leaf)
    blank_noise = np.random.randint(0, 256, (640, 640, 3), dtype=np.uint8)
    
    with pytest.raises(NotACropError) as exc_info:
        validator.validate(blank_noise)
        
    assert "Invalid image: Crop leaf not detected. Please center the leaf." in str(exc_info.value)
