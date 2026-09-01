# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false
import os
import shutil
import zipfile
import json
from datetime import datetime

def package_mobile_bundle() -> None:
    root_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    export_dir: str = os.path.join(root_dir, 'export')
    bundle_name: str = 'Pestopia_Mobile_Bundle'
    bundle_dir: str = os.path.join(export_dir, bundle_name)
    
    # 1. Clear existing bundle directory to prevent stale files
    if os.path.exists(bundle_dir):
        print(f"Clearing existing directory: {bundle_dir}")
        shutil.rmtree(bundle_dir)
        
    os.makedirs(bundle_dir, exist_ok=True)
    
    # 2. Define source paths for ONNX models and labels
    weights_dir: str = os.path.join(root_dir, 'src', 'weights')
    
    disease_onnx: str = os.path.join(weights_dir, 'disease_model.onnx')
    disease_onnx_data: str = os.path.join(weights_dir, 'disease_model.onnx.data')
    pest_onnx: str = os.path.join(weights_dir, 'pest_model.onnx')
    disease_labels: str = os.path.join(weights_dir, 'disease_labels.txt')
    pest_labels: str = os.path.join(weights_dir, 'pest_labels.txt')
    
    required_files: list[str] = [disease_onnx, pest_onnx, disease_labels, pest_labels]
    
    # Check if required files exist
    for filepath in required_files:
        if not os.path.exists(filepath):
            print(f"ERROR: Required file missing: {filepath}")
            print("Cannot generate mobile bundle.")
            return

    # 3. Create assets subfolder and copy files safely
    assets_dir: str = os.path.join(bundle_dir, 'assets')
    os.makedirs(assets_dir, exist_ok=True)
    
    print("Copying core ONNX assets...")
    shutil.copy2(disease_onnx, assets_dir)
    shutil.copy2(pest_onnx, assets_dir)
    shutil.copy2(disease_labels, assets_dir)
    shutil.copy2(pest_labels, assets_dir)
    
    if os.path.exists(disease_onnx_data):
        print("Copying external ONNX weights (disease_model.onnx.data)...")
        shutil.copy2(disease_onnx_data, assets_dir)
        
    # 4. Generate metadata.json
    metadata = {
        "bundle_name": "Pestopia_Edge_AI",
        "version": "2.0.0",
        "build_date": datetime.utcnow().isoformat() + "Z",
        "target_platform": "Android API 26+ (Kotlin/Jetpack Compose)",
        "models": {
            "disease_model": {
                "filename": "disease_model.onnx",
                "framework": "ONNXRuntime",
                "input_resolution": "224x224x3",
                "classes": 67
            },
            "pest_model": {
                "filename": "pest_model.onnx",
                "framework": "ONNXRuntime",
                "input_resolution": "384x384x3",
                "classes": 132
            }
        },
        "notes": "RAG completely removed. Pure ONNX execution pipeline."
    }
    
    metadata_path: str = os.path.join(bundle_dir, 'metadata.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=4)
    print("Generated metadata.json")
        
    # 5. Zip the bundle
    zip_filename: str = os.path.join(export_dir, f"{bundle_name}.zip")
    print(f"Archiving bundle to {zip_filename}...")
    
    with zipfile.ZipFile(zip_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(bundle_dir):
            for file in files:
                file_path: str = os.path.join(root, file)
                arcname: str = os.path.relpath(file_path, bundle_dir)
                zipf.write(file_path, arcname)
                
    print("\n✅ Mobile bundle successfully packaged for Handoff!")
    print(f"Location: {zip_filename}")
    
if __name__ == "__main__":
    package_mobile_bundle()
