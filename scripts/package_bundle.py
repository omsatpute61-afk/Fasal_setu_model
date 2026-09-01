import os
import shutil
import zipfile
import json
from datetime import datetime

def package_bundle() -> None:
    print("Starting Pestopia Mobile Bundle packaging...")
    
    export_dir = os.path.join("export", "Pestopia_Mobile_Bundle")
    zip_path = os.path.join("export", "Pestopia_Mobile_Bundle.zip")
    
    disease_onnx = os.path.join("src", "weights", "disease_model.onnx")
    disease_onnx_data = os.path.join("src", "weights", "disease_model.onnx.data")
    pest_onnx = os.path.join("src", "weights", "pest_model.onnx")
    pest_onnx_data = os.path.join("src", "weights", "pest_model.onnx.data")
    disease_labels = os.path.join("src", "weights", "disease_labels.txt")
    pest_labels = os.path.join("src", "weights", "pest_labels.txt")
    chroma_db_src = os.path.join("src", "data", "chroma_db")
    
    assets_dir = os.path.join(export_dir, "assets")
    db_dir = os.path.join(export_dir, "database")
    
    if os.path.exists(export_dir):
        print(f"Clearing existing bundle directory: {export_dir}")
        shutil.rmtree(export_dir)
        
    if os.path.exists(zip_path):
        os.remove(zip_path)
        
    os.makedirs(assets_dir, exist_ok=True)
    os.makedirs(db_dir, exist_ok=True)
    
    files_to_copy = [
        (disease_onnx, "disease_model.onnx"),
        (disease_onnx_data, "disease_model.onnx.data"),
        (pest_onnx, "pest_model.onnx"),
        (pest_onnx_data, "pest_model.onnx.data"),
        (disease_labels, "disease_labels.txt"),
        (pest_labels, "pest_labels.txt")
    ]
    
    included_files = []
    
    print("\nCopying model assets and labels...")
    for src_path, file_name in files_to_copy:
        if os.path.exists(src_path):
            target_path = os.path.join(assets_dir, file_name)
            shutil.copy2(src_path, target_path)
            included_files.append(f"assets/{file_name}")
            print(f"Copied: {file_name}")
        else:
            print(f"Warning: Missing required file - {src_path}")
            
    print("\nCopying Chroma Vector Database...")
    if os.path.exists(chroma_db_src):
        for item in os.listdir(chroma_db_src):
            s = os.path.join(chroma_db_src, item)
            d = os.path.join(db_dir, item)
            if os.path.isdir(s):
                shutil.copytree(s, d)
            else:
                shutil.copy2(s, d)
        included_files.append("database/chroma_db_contents")
        print("Copied: chroma_db folder contents")
    else:
        print(f"Warning: Missing Vector DB - {chroma_db_src}")
        
    metadata_path = os.path.join(export_dir, "metadata.json")
    print("\nGenerating metadata.json...")
    metadata = {
        "bundle_creation_time": datetime.utcnow().isoformat() + "Z",
        "models": {
            "disease_model": {
                "input_tensor": "[1, 3, 224, 224]",
                "classes": 67
            },
            "pest_model": {
                "input_tensor": "[1, 3, 384, 384]",
                "classes": 132
            }
        },
        "included_assets": included_files
    }
    
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=4)
        
    print("Copied: metadata.json")
    
    print("\nCompressing bundle into .zip archive...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(export_dir):
            for file in files:
                file_path = os.path.join(root, file)
                arcname = os.path.relpath(file_path, os.path.dirname(export_dir))
                zipf.write(file_path, arcname)
                
    print(f"\nSuccess: Bundle fully packaged and saved to -> {zip_path}")

if __name__ == "__main__":
    package_bundle()
