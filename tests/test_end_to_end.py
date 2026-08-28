"""
Phase 5: End-to-End Verification
Tests the entire AgriVision Edge AI backend using a noisy, heavily blurred image 
to ensure the Enhancer and Gatekeeper behave correctly.
"""
import cv2
import numpy as np
import pytest

from src.engine.decision_engine import DecisionEngine

def test_pipeline_resilience_and_schema():
    # 1. Initialize the core engine
    engine = DecisionEngine()

    # 2. Generate a noisy, heavily blurred synthetic image
    # Base brown soil image to prevent Gray World AWB from aggressively neutralizing green
    frame = np.zeros((640, 640, 3), dtype=np.uint8)
    frame[:, :] = [30, 50, 60] # BGR Brown soil
    
    # Add a small green leaf (250x250) to trigger SOFT_PASS (~0.45 confidence)
    frame[200:450, 200:450] = [0, 150, 0] # BGR Green
    frame[250:300, 250:300] = [0, 200, 200] # Yellow patch
    
    # Add heavy Gaussian noise (using safe addition to prevent uint8 underflow artifacts)
    noise = np.random.normal(0, 25, frame.shape)
    noisy_frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Apply severe blur (simulating budget camera motion blur)
    blurred_frame = cv2.GaussianBlur(noisy_frame, (31, 31), 15)

    # 3. Process the image
    payload = engine.process_image(blurred_frame, crop="Tomato")
    
    # 4. Assertions
    # Ensure it didn't crash and returned the 4-tab schema
    assert "error" not in payload, f"Pipeline threw an error: {payload.get('error')}"
    assert "tab_1_overview" in payload
    assert "tab_2_disease" in payload
    assert "tab_3_pests" in payload
    assert "tab_4_treatment" in payload
    
    # Ensure Gatekeeper triggered a SOFT_PASS warning due to the heavy blur/noise
    overview = payload["tab_1_overview"]
    assert overview["gatekeeper_warning"] is not None, "Gatekeeper failed to issue a soft pass warning for the blurred image."
    assert "Partial foliage" in overview["gatekeeper_warning"]

    # Ensure Taxonomy Registry successfully populated the exact scientific names
    disease = payload["tab_2_disease"]
    assert disease["scientific_name"] == "Alternaria solani", "Taxonomy mapping failed for Tomato Early Blight."
    
    pests = payload["tab_3_pests"]
    assert pests["scientific_name"] == "Aphis gossypii", "Taxonomy mapping failed for Aphids."
    
    print("\n[SUCCESS] Pipeline survived heavy artifacts, passed the Gatekeeper, and generated the strict 4-Tab JSON schema.")

if __name__ == "__main__":
    test_pipeline_resilience_and_schema()
