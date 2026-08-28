import os
from pathlib import Path

def setup_rag_data():
    """
    Autonomously creates the required directories and generates a 
    synthetic agricultural extension manual text file for RAG ingestion.
    """
    base_dir = Path(__file__).parent.parent
    docs_dir = base_dir / "data" / "documents"
    chroma_dir = base_dir / "data" / "chroma_db"
    
    # Create directories
    docs_dir.mkdir(parents=True, exist_ok=True)
    chroma_dir.mkdir(parents=True, exist_ok=True)
    
    doc_path = docs_dir / "mock_agricultural_extension.txt"
    
    mock_content = """AGRIVISION KVK EXTENSION MANUAL (SYNTHETIC RAG DATA)

Tomato Early Blight: 
Optimal to Vulnerable Health: Organic control includes Copper-based fungicides (0.2%) applied cautiously to avoid foliage burn.
Critical Health: Chemical dosage requires Mancozeb 75 WP at 2.5g/liter of water applied immediately.

Maize Blight:
Optimal Health: Maintain 60cm row spacing to improve canopy airflow.
Critical Health: Chemical dosage requires Propiconazole 25 EC at 1ml/liter of water.

Whitefly Infestation: 
Organic control requires Neem oil 10000 ppm at 2ml/liter to disrupt the nymphal stages. 
Chemical dosage: Imidacloprid 17.8 SL at 0.5ml/liter applied directly to the underside of the leaves.

Aphid Infestation:
Organic control: Release Ladybird beetles (Coccinellidae) or apply insecticidal soap.
Chemical dosage: Spray Thiamethoxam 25 WG at 0.2g/liter of water if pest count exceeds the Economic Threshold Level (ETL).

Cotton Wilt:
Vulnerable Health: Soil drenching with Trichoderma viride enriched compost.
Critical Health: Chemical dosage requires Carbendazim 50 WP at 2g/liter in the root zone.
"""
    
    with open(doc_path, "w", encoding="utf-8") as f:
        f.write(mock_content)
        
    print(f"[SUCCESS] Successfully created synthetic RAG data at: {doc_path}")
    print(f"[SUCCESS] ChromaDB directory ready at: {chroma_dir}")

if __name__ == "__main__":
    setup_rag_data()
