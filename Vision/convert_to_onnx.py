import os
import torch
import torch.nn as nn
import timm

def export_to_onnx() -> None:
    print("Starting ONNX conversion...")
    
    # Define exact paths
    disease_pth: str = "src/weights/best_disease_model.pth"
    pest_pth: str = "src/weights/best_pest_model.pth"
    
    disease_onnx: str = "src/weights/disease_model.onnx"
    pest_onnx: str = "src/weights/pest_model.onnx"
    
    # 1. Convert Disease Model (67 classes, 224x224)
    if os.path.exists(disease_pth):
        print("\nTranslating Disease Model...")
        model_d: nn.Module = timm.create_model('mobilenetv4_conv_small', pretrained=False, num_classes=67)
        model_d.load_state_dict(torch.load(disease_pth, map_location='cpu', weights_only=True))
        model_d.eval()
        
        dummy_input: torch.Tensor = torch.randn(1, 3, 224, 224)
        torch.onnx.export(model_d, (dummy_input,), disease_onnx, input_names=['input'], output_names=['output'])  # pyright: ignore[reportUnknownMemberType]
        print(f"Success: {disease_onnx} created!")
    else:
        print(f"Error: Could not find {disease_pth}. Please check the file name.")

    # 2. Convert Pest Model (132 classes, 384x384)
    if os.path.exists(pest_pth):
        print("\nTranslating Pest Model...")
        model_p: nn.Module = timm.create_model('mobilenetv4_conv_large', pretrained=False, num_classes=132)
        model_p.load_state_dict(torch.load(pest_pth, map_location='cpu', weights_only=True))
        model_p.eval()
        
        dummy_input_p: torch.Tensor = torch.randn(1, 3, 384, 384)
        torch.onnx.export(model_p, (dummy_input_p,), pest_onnx, input_names=['input'], output_names=['output'])  # pyright: ignore[reportUnknownMemberType]
        print(f"Success: {pest_onnx} created!")
    else:
        print(f"Error: Could not find {pest_pth}. (If it is named 'best_pest_model_sota.pth', please rename it to 'best_pest_model.pth')")

if __name__ == "__main__":
    export_to_onnx()