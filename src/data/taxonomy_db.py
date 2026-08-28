"""
Fine-Grained Taxonomy Database
Acts as a lookup dictionary to enrich the model's raw output with scientific names and treatment plans.
"""

class TaxonomyDatabase:
    def __init__(self):
        # Database mapping for Diseases (Pathogens)
        self.disease_db = {
            "Early Blight": {
                "common_name": "Tomato Early Blight",
                "scientific_name": "Alternaria solani",
                "treatment_plan": {
                    "organic_control": "Prune lower leaves to improve airflow, apply copper-based fungicide or Bacillus subtilis.",
                    "chemical_control": "Apply Chlorothalonil or Mancozeb every 7-10 days.",
                    "urgency_level": "High"
                }
            },
            "Late Blight": {
                "common_name": "Tomato Late Blight",
                "scientific_name": "Phytophthora infestans",
                "treatment_plan": {
                    "organic_control": "Destroy infected plants immediately. Ensure wide spacing.",
                    "chemical_control": "Apply Copper Octanoate or specific anti-oomycete fungicides like Mefenoxam.",
                    "urgency_level": "Critical"
                }
            },
            "Healthy": {
                "common_name": "Healthy Crop",
                "scientific_name": "N/A",
                "treatment_plan": {
                    "organic_control": "Maintain current watering and fertilization schedules.",
                    "chemical_control": "None required.",
                    "urgency_level": "None"
                }
            }
        }

        # Database mapping for Pests (Insects)
        self.pest_db = {
            "Pest (Mock)": {
                "common_name": "Asiatic Rice Borer / General Pest",
                "scientific_name": "Chilo suppressalis",
                "treatment_plan": {
                    "organic_control": "Introduce natural predators like Trichogramma wasps or spray Neem oil.",
                    "chemical_control": "Apply Chlorantraniliprole or Fipronil granules.",
                    "urgency_level": "Medium"
                }
            },
            "Class 4": {
                "common_name": "Asiatic Rice Borer",
                "scientific_name": "Chilo suppressalis",
                "treatment_plan": {
                    "organic_control": "Pheromone traps, Trichogramma wasps.",
                    "chemical_control": "Chlorantraniliprole.",
                    "urgency_level": "High"
                }
            },
            "Class 8": {
                "common_name": "Brown Plant Hopper",
                "scientific_name": "Nilaparvata lugens",
                "treatment_plan": {
                    "organic_control": "Avoid excessive nitrogen fertilizers, introduce spiders/mirid bugs.",
                    "chemical_control": "Pymetrozine, Buprofezin.",
                    "urgency_level": "High"
                }
            }
        }

    def get_disease_info(self, raw_class_name):
        """
        Takes the raw output from the disease model and returns the enriched taxonomic info.
        """
        # Clean up the mock suffix if present
        clean_name = raw_class_name.replace(" (Mock)", "").strip()
        
        # Return fallback if not found
        return self.disease_db.get(clean_name, {
            "common_name": clean_name,
            "scientific_name": "Unknown Pathogen",
            "treatment_plan": {
                "organic_control": "Consult local agricultural extension for unknown pathology.",
                "chemical_control": "Isolate plant and observe.",
                "urgency_level": "Unknown"
            }
        })

    def get_pest_info(self, raw_class_name):
        """
        Takes the raw output from the pest model and returns enriched taxonomic info.
        """
        clean_name = raw_class_name.replace(" (Mock)", "").strip()
        return self.pest_db.get(clean_name, {
            "common_name": clean_name,
            "scientific_name": "Unknown Insect",
            "treatment_plan": {
                "organic_control": "General insecticidal soap or Neem oil.",
                "chemical_control": "Broad-spectrum insecticide if damage is severe.",
                "urgency_level": "Medium"
            }
        })
