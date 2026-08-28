import os
from ultralytics import RTDETR

def train_pest_model():
    """
    Phase 2 SOTA Pivot: RT-DETR Pest Object Detection.
    Trains a Vision Transformer (RT-DETR) model specifically tuned for dense agricultural pest detection.
    """
    print("Initializing RT-DETR Pest Classifier (SOTA Pivot)...")
    
    # 1. MODEL INITIALIZATION
    # Utilizing the large ('l') checkpoint for robust feature extraction before fine-tuning
    model = RTDETR('rtdetr-l.pt')
    
    # Define your dataset YAML path here.
    # e.g., data_path = "datasets/pest_dataset/data.yaml"
    data_path = "path/to/pest/data.yaml"
    
    print(f"Starting RT-DETR anti-overfitting training loop on dataset: {data_path}")
    
    # 2. THE ANTI-OVERFITTING TRAINING LOOP
    # - mosaic=1.0 is critical to prevent the transformer from memorizing background leaves.
    # - batch=8 prevents CUDA OOM issues on consumer GPUs with transformer architectures.
    try:
        results = model.train(
            data=data_path,
            epochs=50,
            imgsz=640,
            batch=8,
            mosaic=1.0,
            project="runs/detect",
            name="rtdetr_pest_training"
        )
        print("Training successfully completed.")
    except Exception as e:
        print(f"Training interrupted or failed. Ensure your data.yaml is correctly configured.\nError: {e}")

    # 3. WEIGHT EXPORT
    print("\n" + "="*50)
    print("WEIGHT EXPORT INSTRUCTIONS:")
    print("Once training fully completes successfully, your best model weights will be saved to:")
    print("runs/detect/rtdetr_pest_training/weights/best.pt")
    print("Please copy 'best.pt' and rename it into your production directory:")
    print("cp runs/detect/rtdetr_pest_training/weights/best.pt src/weights/pest_model.pt")
    print("="*50 + "\n")

if __name__ == "__main__":
    train_pest_model()
