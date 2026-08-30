import json
import time
from typing import Any, Dict
import numpy as np
from PIL import Image

from src.engine.decision_engine import DecisionEngine

def test_engine() -> None:
    """
    CLI Test Script for the Crop Diagnostics Engine.
    Runs the engine against a dummy array to verify the pipeline.
    """
    print("Initializing Decision Engine...")
    start_time: float = time.time()
    
    try:
        engine = DecisionEngine()
    except Exception as e:
        print(f"Failed to initialize engine: {e}")
        return
        
    print(f"Engine Initialized in {time.time() - start_time:.2f} seconds.")

    print("\n--- TEST: VALID FRAME (MOCK LEAF) ---")
    
    # Create a dummy image (simulating a 640x640 RGB image)
    valid_frame: np.ndarray[Any, Any] = np.zeros((640, 640, 3), dtype=np.uint8)
    valid_frame[:, :] = [0, 200, 0] 
    valid_frame[0:320, :] = [0, 255, 255] 
    
    # Convert numpy array to PIL Image
    image: Image.Image = Image.fromarray(valid_frame)
    
    inference_start: float = time.time()
    result: Dict[str, Any] = engine.process_image(image)
    inference_time: float = time.time() - inference_start
    
    # Print the result
    print(f"Inference Time: {inference_time:.4f} seconds")
    print(json.dumps(result, indent=2))

if __name__ == "__main__":
    test_engine()
