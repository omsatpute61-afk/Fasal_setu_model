"""
Phase 4: Diagnostic Engine & UI Payload
Orchestrates the entire Edge AI pipeline (Enhancer -> Gatekeeper -> Diagnostic Models).
Generates the strict 4-Tab JSON schema for the Android UI.
"""
import time
import json
from src.preprocessing.image_enhancer import ImageEnhancer
from src.preprocessing.plant_validator import PlantValidator, NotACropError
from src.preprocessing.roi_extractor import LeafROIExtractor
from src.data.taxonomy_registry import TaxonomyRegistry
from src.engine.scoring import CropHealthScorer
from src.engine.treatment_agent import TreatmentAdvisor
import os
import time
from ultralytics import YOLO

class DecisionEngine:
    def __init__(self):
        self.enhancer = ImageEnhancer()
        self.validator = PlantValidator()
        self.roi_extractor = LeafROIExtractor()
        self.registry = TaxonomyRegistry()
        self.treatment_advisor = TreatmentAdvisor()
        
        # Load actual trained YOLO models
        # Using a try-except to fallback to base models if weights aren't present locally yet
        try:
            self.disease_model = YOLO("src/weights/disease_model.pt")
        except:
            self.disease_model = YOLO("yolov8n.pt") # Fallback for demo
            
        try:
            self.pest_model = YOLO("src/weights/pest_model.pt")
        except:
            self.pest_model = YOLO("yolov8n.pt") # Fallback for demo

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
        try:
            validation = self.validator.validate(enhanced_frame)
        except NotACropError as e:
            return {"error": str(e)}
            
        is_soft_pass = validation["status"] == "SOFT_PASS"

        # 4. Leaf ROI Extraction (Auto-Zoom)
        roi_frame = self.roi_extractor.extract_roi(enhanced_frame)

        # 5. DYNAMIC REAL-TIME INFERENCE (YOLO)
        # Disease Prediction
        disease_results = self.disease_model.predict(roi_frame, conf=0.45, verbose=False)
        if disease_results and len(disease_results[0].boxes) > 0:
            top_class_idx = int(disease_results[0].boxes.cls[0])
            detected_disease = self.disease_model.names[top_class_idx]
            
            box = disease_results[0].boxes.xyxy[0].cpu().numpy()
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            roi_area = roi_frame.shape[0] * roi_frame.shape[1]
            affected_area_pct = min(100.0, (box_area / roi_area) * 100)
        else:
            detected_disease = "Healthy"
            affected_area_pct = 0.0

        # Pest Prediction
        pest_results = self.pest_model.predict(roi_frame, conf=0.35, verbose=False)
        pest_count = len(pest_results[0].boxes) if pest_results else 0
        pest_bboxes = []
        if pest_count > 0:
            top_pest_idx = int(pest_results[0].boxes.cls[0])
            detected_pest = self.pest_model.names[top_pest_idx]
            pest_bboxes = pest_results[0].boxes.xyxy.cpu().numpy().tolist()
        else:
            detected_pest = "None"
            
        disease_res = {"name": detected_disease, "affected_area_pct": affected_area_pct}
        pest_res = {"name": detected_pest, "count": pest_count, "bboxes": pest_bboxes, "etl_exceeded": pest_count > 5}

        # 6. Registry Lookups
        disease_tax = self.registry.get_disease_info(disease_res["name"])
        pest_tax = self.registry.get_pest_info(pest_res["name"])

        # 7. PHASE 2: 1-10 Health Scoring Engine
        scoring_result = CropHealthScorer.calculate_score(disease_res["affected_area_pct"], pest_res["count"])
        health_index = scoring_result["score"]
        health_category = scoring_result["category"]
        
        urgency = "Critical" if disease_tax.get("scientific_name") != "N/A" or pest_res["etl_exceeded"] else "Normal"
        
        # 8. Merge baseline treatments
        organic_treatments = []
        chemical_treatments = []
        if disease_res["name"] != "Healthy":
            organic_treatments.append(f"Disease: {disease_tax['organic']}")
            chemical_treatments.append(f"Disease: {disease_tax['chemical']}")
        if pest_res["count"] > 0:
            organic_treatments.append(f"Pest: {pest_tax['organic']}")
            chemical_treatments.append(f"Pest: {pest_tax['chemical']}")

        # 9. PHASE 3: RAG Knowledge Retrieval
        rag_advisory = self.treatment_advisor.get_treatment(disease_res["name"], pest_res["name"], health_category)
        if rag_advisory:
            # We append the RAG insights to the organic advisory for display in the UI
            organic_treatments.append(rag_advisory)

        exec_time_ms = (time.perf_counter() - start_time) * 1000

        # 10. Construct 4-Tab JSON Schema
        payload = {
            "tab_1_overview": {
                "crop": crop,
                "health_index_score": health_index,
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
