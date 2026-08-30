import os
import torch
import torch.nn as nn
from PIL import Image
import timm
from torchvision import transforms # pyright: ignore[reportMissingTypeStubs]
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from typing import Any, Dict, List, Optional, cast

class DecisionEngine:
    def __init__(self) -> None:
        self.device: torch.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Define paths
        self.src_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.disease_weights: str = os.path.join(self.src_dir, 'weights', 'best_disease_model.pth')
        self.pest_weights: str = os.path.join(self.src_dir, 'weights', 'best_pest_model.pth')
        self.db_dir: str = os.path.join(self.src_dir, 'data', 'chroma_db')

        # Standard MobileNetV4 Image Transforms
        self.transform: Any = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        # Load AI Components
        self.disease_model: Optional[nn.Module] = self._load_vision_model(self.disease_weights, 14)
        self.pest_model: Optional[nn.Module] = self._load_vision_model(self.pest_weights, 132)
        self.rag_db: Optional[Chroma] = self._load_rag_agent()

    def _load_vision_model(self, path: str, num_classes: int) -> Optional[nn.Module]:
        try:
            model: nn.Module = timm.create_model('mobilenetv4_conv_small', pretrained=False, num_classes=num_classes)
            if os.path.exists(path):
                model.load_state_dict(torch.load(path, map_location=self.device, weights_only=True))
                model = model.to(self.device)
                model.eval()
                return model
            return None
        except Exception as e:
            print(f"Vision model error at {path}: {e}")
            return None

    def _load_rag_agent(self) -> Optional[Chroma]:
        try:
            if os.path.exists(self.db_dir):
                embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
                return Chroma(persist_directory=self.db_dir, embedding_function=embeddings)
            return None
        except Exception as e:
            print(f"RAG Database error: {e}")
            return None

    def process_image(self, image: Image.Image) -> Dict[str, Any]:
        issues_detected: List[str] = []
        health_score: int = 10
        
        # Preprocess image
        # We cast the transform output to a torch.Tensor to satisfy Pylance strict mode
        input_tensor: torch.Tensor = cast(torch.Tensor, self.transform(image))
        input_tensor = input_tensor.unsqueeze(0).to(self.device)

        # 1. Pest Inference
        if self.pest_model:
            with torch.no_grad():
                pest_out: torch.Tensor = self.pest_model(input_tensor)
                pest_idx: int = int(torch.argmax(pest_out, dim=1).item())
                # Note: You will map pest_idx to your actual class names list later
                issues_detected.append(f"Pest ID {pest_idx}")
                health_score -= 3

        # 2. Disease Inference
        if self.disease_model:
            with torch.no_grad():
                disease_out: torch.Tensor = self.disease_model(input_tensor)
                disease_idx: int = int(torch.argmax(disease_out, dim=1).item())
                issues_detected.append(f"Disease ID {disease_idx}")
                health_score -= 2

        detected_issue_string: str = " and ".join(issues_detected) if issues_detected else "Healthy Crop"

        # 3. Autonomous RAG Advisory
        advice: str = "General monitoring recommended. No immediate chemical intervention required."
        if issues_detected and self.rag_db:
            query: str = f"What is the best pesticide and organic treatment for {detected_issue_string}?"
            docs = self.rag_db.similarity_search(query, k=2)
            if docs:
                advice = "\n\n".join([doc.page_content for doc in docs])

        return {
            'disease': detected_issue_string,
            'score': max(1, health_score), 
            'advice': advice
        }