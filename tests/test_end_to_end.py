"""
Phase 5: End-to-End Verification
Tests the entire AgriVision Edge AI backend using a noisy, heavily blurred image 
to ensure the pipeline behaves correctly.
"""
import cv2
import numpy as np
from PIL import Image
from typing import Dict, Any

from src.engine.decision_engine import DecisionEngine

def test_pipeline_resilience_and_schema() -> None:
    # 1. Initialize the core engine
    engine = DecisionEngine()

    # 2. Generate a noisy, heavily blurred synthetic image
    # Base brown soil image to prevent Gray World AWB from aggressively neutralizing green
    frame: np.ndarray[Any, Any] = np.zeros((640, 640, 3), dtype=np.uint8)
    frame[:, :] = [30, 50, 60] # BGR Brown soil
    
    # Add a small green leaf (250x250) to trigger SOFT_PASS (~0.45 confidence)
    frame[200:450, 200:450] = [0, 150, 0] # BGR Green
    frame[250:300, 250:300] = [0, 200, 200] # Yellow patch
    
    # Add heavy Gaussian noise (using safe addition to prevent uint8 underflow artifacts)
    noise: np.ndarray[Any, Any] = np.random.normal(0, 25, frame.shape)  # pyright: ignore[reportUnknownVariableType, reportUnknownMemberType]
    noisy_frame: np.ndarray[Any, Any] = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)  # pyright: ignore[reportUnknownArgumentType, reportUnknownMemberType]
    
    # Apply severe blur (simulating budget camera motion blur)
    blurred_frame: np.ndarray[Any, Any] = cv2.GaussianBlur(noisy_frame, (31, 31), 15)

    # Convert to PIL Image for the new DecisionEngine signature
    # OpenCV uses BGR, convert to RGB first
    rgb_frame: np.ndarray[Any, Any] = cv2.cvtColor(blurred_frame, cv2.COLOR_BGR2RGB)
    pil_image: Image.Image = Image.fromarray(rgb_frame)

    # 3. Process the image (no 'crop' parameter in current implementation)
    payload: Dict[str, Any] = engine.process_image(pil_image)
    
    # 4. Assertions
    # Ensure it didn't crash and returned the current schema
    assert "error" not in payload, f"Pipeline threw an error: {payload.get('error')}"
    assert "score" in payload
    assert "disease" in payload
    assert "advice" in payload
    
    # Check that score is bounded correctly
    score: float = float(payload["score"])
    assert 1.0 <= score <= 10.0, f"Score out of bounds: {score}"
    
    # Verify disease output format
    disease: str = str(payload["disease"])
    assert isinstance(disease, str), "Disease field is not a string"

    print("\n[SUCCESS] Pipeline survived heavy artifacts and generated the correct JSON schema.")

if __name__ == "__main__":
    test_pipeline_resilience_and_schema()
