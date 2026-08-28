"""
Phase 3: Taxonomy Registry
Provides exact scientific binomials and tailored agronomic treatment plans 
for the top 10 terrestrial crops.
"""

class TaxonomyRegistry:
    def __init__(self):
        # Top 10 Supported Terrestrial Crops
        self.supported_crops = [
            "Wheat", "Maize", "Cotton", "Sugarcane", "Soybean", 
            "Potato", "Tomato", "Groundnut", "Mustard", "Chickpea"
        ]

        # Disease Database (Pathogens)
        self.disease_registry = {
            "Wheat Rust": {
                "scientific_name": "Puccinia graminis",
                "organic": "Use resistant cultivars (e.g., PBW 343). Ensure proper spacing for aeration.",
                "chemical": "Spray Propiconazole (0.1%) or Tebuconazole (0.1%)."
            },
            "Maize Blight": {
                "scientific_name": "Bipolaris maydis",
                "organic": "Crop rotation and residue destruction.",
                "chemical": "Apply Mancozeb or Chlorothalonil."
            },
            "Cotton Wilt": {
                "scientific_name": "Fusarium oxysporum f. sp. vasinfectum",
                "organic": "Apply Trichoderma viride in soil. Use crop rotation.",
                "chemical": "Soil drenching with Carbendazim."
            },
            "Potato Late Blight": {
                "scientific_name": "Phytophthora infestans",
                "organic": "Destroy cull piles and infected tubers. Ensure good drainage.",
                "chemical": "Apply Copper Oxychloride or Metalaxyl."
            },
            "Tomato Early Blight": {
                "scientific_name": "Alternaria solani",
                "organic": "Prune lower leaves, apply Bacillus subtilis.",
                "chemical": "Apply Chlorothalonil or Mancozeb every 7-10 days."
            },
            "Healthy": {
                "scientific_name": "N/A",
                "organic": "Maintain standard organic compost regime.",
                "chemical": "None required."
            }
        }

        # Pest Database (Insects)
        self.pest_registry = {
            "Bollworm": {
                "scientific_name": "Helicoverpa armigera",
                "organic": "Install pheromone traps (5/ha). Release Trichogramma wasps.",
                "chemical": "Apply Spinosad (0.3ml/L) or Emamectin Benzoate (0.4g/L)."
            },
            "Aphid": {
                "scientific_name": "Aphis gossypii",
                "organic": "Spray 5% Neem Seed Kernel Extract (NSKE) or insecticidal soap.",
                "chemical": "Spray Imidacloprid (0.3ml/L) or Thiamethoxam."
            },
            "Fall Armyworm": {
                "scientific_name": "Spodoptera frugiperda",
                "organic": "Hand-pick egg masses. Apply Neem oil.",
                "chemical": "Apply Chlorantraniliprole."
            }
        }

    def is_crop_supported(self, crop_name: str) -> bool:
        return crop_name in self.supported_crops

    def get_disease_info(self, common_name: str):
        return self.disease_registry.get(common_name, {
            "scientific_name": "Unknown Pathogen",
            "organic": "Isolate plant and consult local agricultural extension.",
            "chemical": "Do not apply chemicals without specific diagnosis."
        })

    def get_pest_info(self, common_name: str):
        return self.pest_registry.get(common_name, {
            "scientific_name": "Unknown Insecta",
            "organic": "Apply broad-spectrum Neem oil.",
            "chemical": "Consult extension officer before applying insecticides."
        })
