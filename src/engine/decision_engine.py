"""
Phase 4: Diagnostic Engine & UI Payload
Orchestrates the entire Edge AI pipeline (Enhancer -> Gatekeeper -> Diagnostic Models).
Generates the strict 4-Tab JSON schema for the Android UI.
"""
import time
import json
from src.preprocessing.image_enhancer import ImageEnhancer
from src.preprocessing.plant_validator import PlantValidator
from src.data.taxonomy_registry import TaxonomyRegistry

class DecisionEngine:
    def __init__(self):
        self.enhancer = ImageEnhancer()
        self.validator = PlantValidator()
        self.registry = TaxonomyRegistry()

    def analyze_disease(self, enhanced_frame, crop):
        """Mock method representing the YOLO/EfficientNet disease model"""
        # Simulate detecting a disease based on the crop type
        if crop == "Tomato":
            return {"name": "Tomato Early Blight", "affected_area_pct": 22.5}
        elif crop == "Maize":
            return {"name": "Maize Blight", "affected_area_pct": 15.0}
        return {"name": "Healthy", "affected_area_pct": 0.0}

    def detect_pests(self, enhanced_frame, crop):
        """Mock method representing the SAHI YOLO pest detection model"""
        # Simulate finding aphids
        return {
            "name": "Aphid",
            "count": 12,
            "bboxes": [[10, 20, 30, 40], [50, 60, 70, 80]],
            "etl_exceeded": True # Economic Threshold Level
        }

    def colorimetric_nutrients(self, enhanced_frame):
        """Mock method representing the OpenCV nutrient chlorosis/necrosis logic"""
        return {"deficiency": "Nitrogen", "severity": "Medium"}

    def process_image(self, frame, crop):
        """
        Main orchestration flow.
        Returns the strict 4-Tab JSON Schema payload.
        """
        start_time = time.perf_counter()
        
        # 1. Enforce Crop Support
        if not self.registry.is_crop_supported(crop):
            return {"error": f"Crop '{crop}' is not supported in the Top 10 Terrestrial Registry."}

        # 2. Hardware-Resilient Enhancement
        enhancement_res = self.enhancer.process(frame)
        enhanced_frame = enhancement_res["enhanced_frame"]

        # 3. Forgiving Gatekeeper
        validation = self.validator.validate(enhanced_frame)
        if validation["status"] == "REJECT":
            return {"error": validation["warning"]}
            
        is_soft_pass = validation["status"] == "SOFT_PASS"

        # 4. Run Diagnostic Models
        disease_res = self.analyze_disease(enhanced_frame, crop)
        pest_res = self.detect_pests(enhanced_frame, crop)
        nutrient_res = self.colorimetric_nutrients(enhanced_frame)

        # 5. Registry Lookups
        disease_tax = self.registry.get_disease_info(disease_res["name"])
        pest_tax = self.registry.get_pest_info(pest_res["name"])

        # 6. Construct 4-Tab JSON Schema
        health_index = max(0, 100 - (disease_res["affected_area_pct"] * 2) - (pest_res["count"] * 1.5))
        urgency = "Critical" if disease_tax.get("scientific_name") != "N/A" or pest_res["etl_exceeded"] else "Normal"
        
        # Merge treatments
        organic_treatments = []
        chemical_treatments = []
        if disease_res["name"] != "Healthy":
            organic_treatments.append(f"Disease: {disease_tax['organic']}")
            chemical_treatments.append(f"Disease: {disease_tax['chemical']}")
        if pest_res["count"] > 0:
            organic_treatments.append(f"Pest: {pest_tax['organic']}")
            chemical_treatments.append(f"Pest: {pest_tax['chemical']}")

        exec_time_ms = (time.perf_counter() - start_time) * 1000

        payload = {
            "tab_1_overview": {
                "crop": crop,
                "health_index_score": round(health_index, 1),
                "primary_urgency": urgency,
                "gatekeeper_warning": validation["warning"] if is_soft_pass else None,
                "execution_time_ms": round(exec_time_ms, 2)
            },
            "tab_2_disease": {
                "common_name": disease_res["name"],
                "scientific_name": disease_tax["scientific_name"],
                "affected_area_percentage": disease_res["affected_area_pct"]
            },
            "tab_3_pests": {
                "primary_pest": pest_res["name"],
                "scientific_name": pest_tax["scientific_name"],
                "insect_count": pest_res["count"],
                "bounding_boxes": pest_res["bboxes"],
                "economic_threshold_warning": pest_res["etl_exceeded"]
            },
            "tab_4_treatment": {
                "organic_advisory": organic_treatments if organic_treatments else ["Continue standard organic regime."],
                "chemical_advisory": chemical_treatments if chemical_treatments else ["No chemicals required."]
            }
        }

        return payload
