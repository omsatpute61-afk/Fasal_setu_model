import torch
import os

def export_to_onnx(model, input_tensor, output_path):
    """
    Exports a PyTorch model to ONNX for TFLite/Edge deployment.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    torch.onnx.export(
        model, 
        input_tensor, 
        output_path, 
        export_params=True, 
        opset_version=12,
        do_constant_folding=True,
        input_names=['input'],
        output_names=['output'],
        dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}}
    )
    print(f"Model exported to {output_path}")

def export_pipeline():
    print("Starting Edge export pipeline...")
    
    # 1. Export Disease Model
    from src.training.train_disease_realworld import get_disease_model
    disease_model = get_disease_model()
    disease_model.eval()
    dummy_input = torch.randn(1, 3, 224, 224)
    export_to_onnx(disease_model, dummy_input, "models/disease_model_int8.onnx")
    
    # 2. Export Nutrient Model
    from src.training.train_nutrient_hybrid import get_nutrient_model
    nutrient_model = get_nutrient_model()
    nutrient_model.eval()
    export_to_onnx(nutrient_model, dummy_input, "models/nutrient_cnn_int8.onnx")
    
    # Note: YOLO to TFLite export is typically handled via Ultralytics CLI:
    # yolo export model=runs/pest_detection/weights/best.pt format=tflite int8=True
    print("Export pipeline completed.")

if __name__ == "__main__":
    export_pipeline()
