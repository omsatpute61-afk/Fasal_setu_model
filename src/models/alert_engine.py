import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

from .disease_model import DiseaseModel
from .pest_model import PestModel
from .nutrient_model import NutrientModel

class DecisionEngine:
    """
    Alert & Recommendation Engine
    
    Aggregates the 3 model outputs concurrently using threading to save time on edge devices.
    Triggers mock SMS alerts for critical situations.
    """
    def __init__(self):
        self.disease_model = DiseaseModel()
        self.pest_model = PestModel()
        self.nutrient_model = NutrientModel()
        
        # Edge devices (like Raspberry Pi) typically have 4 cores. 
        # ThreadPoolExecutor is effective here since OpenCV releases the GIL during heavy image operations.
        self.executor = ThreadPoolExecutor(max_workers=3)

    def trigger_sms_alert(self, reason):
        """Mock function to send SMS alert to farmer via GSM module or API."""
        print(f"\n[SMS ALERT TRIGGERED] Sending to Farmer: {reason}\n")

    async def _run_inference_concurrently(self, image):
        loop = asyncio.get_running_loop()
        
        # Run the three CPU/IO bound inference tasks in the thread pool concurrently
        task_disease = loop.run_in_executor(self.executor, self.disease_model.predict, image)
        task_pest = loop.run_in_executor(self.executor, self.pest_model.predict, image)
        task_nutrient = loop.run_in_executor(self.executor, self.nutrient_model.predict, image)
        
        results = await asyncio.gather(task_disease, task_pest, task_nutrient)
        return results

    def analyze_crop(self, image):
        """
        Synchronous wrapper to run the async pipeline and return a FarmerHealthCard.
        """
        start_time = time.time()
        
        # Run asynchronous inference loop
        results = asyncio.run(self._run_inference_concurrently(image))
        disease_res, pest_res, nutrient_res = results
        
        # Evaluate alert conditions
        alerts = []
        if disease_res.get("requires_immediate_action"):
            alerts.append(f"Critical Disease Detected: {disease_res['disease_class']}")
            
        if pest_res.get("economic_threshold_exceeded"):
            alerts.append(f"Pest Economic Threshold Exceeded ({pest_res['total_pest_count']} found)")
            
        if nutrient_res.get("severity") == "High":
            alerts.append(f"Severe Nutrient Deficiency: {nutrient_res['primary_deficiency']}")

        # Trigger SMS if any critical alerts exist
        if alerts:
            self.trigger_sms_alert(" | ".join(alerts))

        inference_time = round(time.time() - start_time, 4)

        # Synthesize Farmer Health Card
        farmer_health_card = {
            "timestamp": time.time(),
            "inference_time_seconds": inference_time,
            "overall_status": "CRITICAL" if alerts else "HEALTHY",
            "diagnostics": {
                "disease_analysis": disease_res,
                "pest_analysis": pest_res,
                "nutrient_analysis": nutrient_res
            },
            "recommended_actions": alerts if alerts else ["Continue standard care."]
        }
        
        return farmer_health_card

if __name__ == "__main__":
    import numpy as np
    engine = DecisionEngine()
    
    print("Capturing dummy image from edge camera...")
    # Mock image simulation
    dummy_image = np.zeros((640, 640, 3), dtype=np.uint8)
    dummy_image[0:320, :] = [0, 255, 255] # Mock chlorosis (yellow)
    
    print("Running Edge AI Pipeline...\n")
    health_card = engine.analyze_crop(dummy_image)
    
    print("--- FARMER HEALTH CARD ---")
    print(json.dumps(health_card, indent=2))
