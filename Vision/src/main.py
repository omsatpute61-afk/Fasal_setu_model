import json
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor

from src.models.plant_validator import PlantValidator
from src.models.disease_model import DiseaseModel
from src.models.pest_model import PestModel
from src.models.nutrient_model import NutrientModel
from src.data.taxonomy_db import TaxonomyDatabase

class DiagnosticEngine:
    """
    Two-Stage Conditional Router for Edge AI Inference.
    - Stage 1: Ultra-fast binary validation (Plant vs OOD).
    - Stage 2: Heavy multi-model diagnostic inference enriched with Fine-Grained Taxonomy.
    """
    def __init__(self):
        self.plant_validator = PlantValidator()
        self.disease_model = DiseaseModel()
        self.pest_model = PestModel()
        self.nutrient_model = NutrientModel()
        self.taxonomy_db = TaxonomyDatabase()
        
        self.executor = ThreadPoolExecutor(max_workers=3)

    def trigger_sms_alert(self, reason):
        print(f"\n[SMS ALERT TRIGGERED] Sending to Farmer: {reason}\n")

    async def _run_heavy_inference(self, frame):
        loop = asyncio.get_running_loop()
        
        task_disease = loop.run_in_executor(self.executor, self.disease_model.predict, frame)
        task_pest = loop.run_in_executor(self.executor, self.pest_model.predict, frame)
        task_nutrient = loop.run_in_executor(self.executor, self.nutrient_model.predict, frame)
        
        return await asyncio.gather(task_disease, task_pest, task_nutrient)

    def analyze_frame(self, frame):
        start_time = time.time()
        
        # STAGE 1: GATEKEEPER
        validation = self.plant_validator.validate(frame)
        stage1_time = round(time.time() - start_time, 4)
        
        if not validation["is_plant"]:
            return {
                "overall_status": "REJECTED",
                "error": validation["message"],
                "inference_time_seconds": stage1_time
            }
            
        # STAGE 2: HEAVY INFERENCE
        heavy_start_time = time.time()
        results = asyncio.run(self._run_heavy_inference(frame))
        disease_res, pest_res, nutrient_res = results
        
        detections = []
        alerts = []
        
        # Enrich Disease Output
        if disease_res.get("disease_class") and disease_res["disease_class"] != "Healthy (Mock)":
            tax_info = self.taxonomy_db.get_disease_info(disease_res["disease_class"])
            detections.append({
                "status": "success",
                "detection_type": "disease",
                "diagnosis": {
                    "common_name": tax_info["common_name"],
                    "scientific_name": tax_info["scientific_name"],
                    "confidence_score": disease_res["confidence"],
                    "bounding_box_coordinates": None
                },
                "treatment_plan": tax_info["treatment_plan"]
            })
            if tax_info["treatment_plan"]["urgency_level"] in ["High", "Critical"]:
                alerts.append(f"Disease: {tax_info['common_name']}")

        # Enrich Pest Output
        for pest in pest_res.get("pest_bounding_boxes", []):
            tax_info = self.taxonomy_db.get_pest_info(pest["class"])
            detections.append({
                "status": "success",
                "detection_type": "pest",
                "diagnosis": {
                    "common_name": tax_info["common_name"],
                    "scientific_name": tax_info["scientific_name"],
                    "confidence_score": pest["confidence"],
                    "bounding_box_coordinates": pest["bbox"]
                },
                "treatment_plan": tax_info["treatment_plan"]
            })
            if tax_info["treatment_plan"]["urgency_level"] in ["High", "Critical"]:
                alerts.append(f"Pest: {tax_info['common_name']}")

        # Nutrient Output (Mock enrichment)
        if nutrient_res.get("severity") in ["Low", "Medium", "High"]:
            detections.append({
                "status": "success",
                "detection_type": "nutrient",
                "diagnosis": {
                    "common_name": f"{nutrient_res['primary_deficiency']} Deficiency",
                    "scientific_name": "Nutrient Imbalance",
                    "confidence_score": 0.85,
                    "bounding_box_coordinates": None
                },
                "treatment_plan": {
                    "organic_control": "Apply compost tea or targeted organic amendments.",
                    "chemical_control": f"Apply NPK fertilizer targeted for {nutrient_res['primary_deficiency']}.",
                    "urgency_level": nutrient_res["severity"]
                }
            })
            if nutrient_res["severity"] == "High":
                alerts.append(f"Nutrient: {nutrient_res['primary_deficiency']} Deficiency")

        if alerts:
            # Deduplicate alerts for SMS
            self.trigger_sms_alert(" | ".join(list(set(alerts))))

        total_inference_time = round(time.time() - start_time, 4)

        return {
            "overall_status": "CRITICAL" if alerts else "HEALTHY",
            "inference_time_seconds": total_inference_time,
            "gatekeeper_confidence": validation["confidence"],
            "detections": detections
        }

if __name__ == "__main__":
    import numpy as np
    engine = DiagnosticEngine()
    
    print("--- TEST 1: INVALID FRAME (OOD / SOIL) ---")
    invalid_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    invalid_frame[:, :] = [30, 40, 50]
    res1 = engine.analyze_frame(invalid_frame)
    print(json.dumps(res1, indent=2))
    
    print("\n--- TEST 2: VALID FRAME (LEAF) ---")
    valid_frame = np.zeros((640, 640, 3), dtype=np.uint8)
    valid_frame[:, :] = [0, 200, 0] 
    valid_frame[0:320, :] = [0, 255, 255] 
    res2 = engine.analyze_frame(valid_frame)
    print(json.dumps(res2, indent=2))
