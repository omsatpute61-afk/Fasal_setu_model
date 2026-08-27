import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.models.plant_validator import PlantValidator
from src.models.disease_model import DiseaseModel
from src.models.pest_model import PestModel
from src.models.nutrient_model import NutrientModel

class DiagnosticEngine:
    """
    Two-Stage Conditional Router for Edge AI Inference.
    - Stage 1: Ultra-fast binary validation (Plant vs OOD).
    - Stage 2: Heavy multi-model diagnostic inference.
    """
    def __init__(self):
        # Stage 1
        self.plant_validator = PlantValidator()
        
        # Stage 2
        self.disease_model = DiseaseModel()
        self.pest_model = PestModel()
        self.nutrient_model = NutrientModel()
        
        # Thread pool for concurrent heavy inference
        self.executor = ThreadPoolExecutor(max_workers=3)

    def trigger_sms_alert(self, reason):
        """Mock function to send SMS alert to farmer."""
        print(f"\n[SMS ALERT TRIGGERED] Sending to Farmer: {reason}\n")

    async def _run_heavy_inference(self, frame):
        loop = asyncio.get_running_loop()
        
        task_disease = loop.run_in_executor(self.executor, self.disease_model.predict, frame)
        task_pest = loop.run_in_executor(self.executor, self.pest_model.predict, frame)
        task_nutrient = loop.run_in_executor(self.executor, self.nutrient_model.predict, frame)
        
        return await asyncio.gather(task_disease, task_pest, task_nutrient)

    def analyze_frame(self, frame):
        start_time = time.time()
        
        # ==========================================
        # STAGE 1: ULTRA-FAST VALIDATION GATEKEEPER
        # ==========================================
        validation = self.plant_validator.validate(frame)
        stage1_time = round(time.time() - start_time, 4)
        
        if not validation["is_plant"]:
            # EARLY EXIT: Save CPU/Battery
            return {
                "timestamp": time.time(),
                "inference_time_seconds": stage1_time,
                "overall_status": "REJECTED",
                "error": validation["message"],
                "gatekeeper_confidence": validation["confidence"]
            }
            
        # ==========================================
        # STAGE 2: HEAVY MULTI-MODEL INFERENCE
        # ==========================================
        # Only runs if Stage 1 passes
        heavy_start_time = time.time()
        results = asyncio.run(self._run_heavy_inference(frame))
        disease_res, pest_res, nutrient_res = results
        
        alerts = []
        if disease_res.get("requires_immediate_action"):
            alerts.append(f"Critical Disease Detected: {disease_res['disease_class']}")
        if pest_res.get("economic_threshold_exceeded"):
            alerts.append(f"Pest Threshold Exceeded ({pest_res['total_pest_count']} found)")
        if nutrient_res.get("severity") == "High":
            alerts.append(f"Severe Nutrient Deficiency: {nutrient_res['primary_deficiency']}")

        if alerts:
            self.trigger_sms_alert(" | ".join(alerts))

        total_inference_time = round(time.time() - start_time, 4)

        return {
            "timestamp": time.time(),
            "inference_time_seconds": total_inference_time,
            "overall_status": "CRITICAL" if alerts else "HEALTHY",
            "gatekeeper_confidence": validation["confidence"],
            "diagnostics": {
                "disease_analysis": disease_res,
                "pest_analysis": pest_res,
                "nutrient_analysis": nutrient_res
            },
            "recommended_actions": alerts if alerts else ["Continue standard care."]
        }

if __name__ == "__main__":
    import numpy as np
    engine = DiagnosticEngine()
    
    print("--- TEST 1: INVALID FRAME (OOD / SOIL) ---")
    # Pure black/brown frame simulating soil or pocket
    invalid_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    invalid_frame[:, :] = [30, 40, 50]
    res1 = engine.analyze_frame(invalid_frame)
    print(json.dumps(res1, indent=2))
    
    print("\n--- TEST 2: VALID FRAME (LEAF) ---")
    # Green frame simulating a healthy leaf
    valid_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    valid_frame[:, :] = [0, 200, 0] # BGR Green
    valid_frame[0:320, :] = [0, 255, 255] # Some yellow to trigger nutrient alert
    res2 = engine.analyze_frame(valid_frame)
    print(json.dumps(res2, indent=2))
