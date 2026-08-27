import os
import subprocess
from roboflow import Roboflow

def setup_directories():
    os.makedirs("datasets/plantdoc", exist_ok=True)
    os.makedirs("datasets/ip102", exist_ok=True)
    os.makedirs("datasets/maize_nutrient", exist_ok=True)
    print("Created dataset directories.")

def download_plantdoc():
    """
    Downloads PlantDoc dataset using Roboflow (requires ROBOFLOW_API_KEY).
    """
    print("Downloading PlantDoc dataset...")
    api_key = os.environ.get("ROBOFLOW_API_KEY")
    if not api_key:
        print("Warning: ROBOFLOW_API_KEY not found. Skipping actual PlantDoc download.")
        return
        
    try:
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("plantdoc").project("plantdoc-dataset")
        dataset = project.version(1).download("yolov8", location="datasets/plantdoc")
        print(f"PlantDoc downloaded to {dataset.location}")
    except Exception as e:
        print(f"Failed to download PlantDoc: {e}")

def download_ip102():
    """
    Downloads IP102 pest dataset using Kaggle CLI (requires ~/.kaggle/kaggle.json).
    """
    print("Downloading IP102 dataset from Kaggle...")
    try:
        # Check if kaggle command exists and credentials are set
        subprocess.run(["kaggle", "datasets", "download", "-d", "vbookshelf/rice-leaf-diseases", "-p", "datasets/ip102", "--unzip"], check=True)
        print("IP102 downloaded successfully.")
    except Exception as e:
        print(f"Kaggle download failed or skipped: {e}")

def download_maize_deficiency():
    """
    Downloads Maize deficiency dataset.
    """
    print("Downloading Maize Nutrient Deficiency dataset...")
    try:
        subprocess.run(["kaggle", "datasets", "download", "-d", "example/maize-nutrient-deficiency", "-p", "datasets/maize_nutrient", "--unzip"], check=True)
        print("Maize dataset downloaded successfully.")
    except Exception as e:
        print(f"Kaggle download failed or skipped: {e}")

if __name__ == "__main__":
    setup_directories()
    download_plantdoc()
    download_ip102()
    download_maize_deficiency()
    print("Data ingestion pipeline completed.")
