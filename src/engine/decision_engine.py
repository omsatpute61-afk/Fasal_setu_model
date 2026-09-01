# pyright: reportMissingImports=false, reportMissingTypeStubs=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportMissingTypeArgument=false, reportIndexIssue=false
import os
import numpy as np
import onnxruntime as ort
from PIL import Image
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

class DecisionEngine:
    def __init__(self) -> None:
        self.src_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        self.disease_onnx = os.path.join(self.src_dir, 'weights', 'disease_model.onnx')
        self.pest_onnx = os.path.join(self.src_dir, 'weights', 'pest_model.onnx')
        self.db_dir = os.path.join(self.src_dir, 'data', 'chroma_db')
        
        # New: Paths to your dictionaries
        self.disease_labels_path = os.path.join(self.src_dir, 'weights', 'disease_labels.txt')
        self.pest_labels_path = os.path.join(self.src_dir, 'weights', 'pest_labels.txt')

        self.disease_session = self._load_onnx(self.disease_onnx)
        self.pest_session = self._load_onnx(self.pest_onnx)
        self.rag_db = self._load_rag()
        
        # Load the plain-English lists into memory
        self.disease_classes = self._load_labels(self.disease_labels_path)
        self.pest_classes = self._load_labels(self.pest_labels_path)

    def _load_onnx(self, path: str):
        if not os.path.exists(path):
            return None
        try:
            return ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        except Exception as e:
            print(f"Failed to load ONNX: {e}")
            return None

    def _load_rag(self):
        if not os.path.exists(self.db_dir):
            return None
            
        # Optimization: Suppress the HuggingFace token warnings for cleaner terminal output
        os.environ["TOKENIZERS_PARALLELISM"] = "false" 
        embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        return Chroma(persist_directory=self.db_dir, embedding_function=embeddings)

    def _load_labels(self, path: str):
        """Reads the text file and returns a list of class names."""
        if not os.path.exists(path):
            print(f"Warning: Label file missing at {path}")
            return []
        with open(path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f.readlines()]

    def _preprocess_image(self, image: Image.Image, img_size: int) -> np.ndarray:
        img = image.resize((img_size, img_size), Image.Resampling.BILINEAR)
        img_data = np.array(img, dtype=np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img_data = (img_data - mean) / std
        img_data = np.transpose(img_data, (2, 0, 1))
        return np.expand_dims(img_data, axis=0)

    def _get_confidence(self, logits):
        """Converts raw AI output into a Softmax percentage"""
        e_x = np.exp(logits - np.max(logits))
        return e_x / e_x.sum()

    def process_image(self, image: Image.Image) -> dict:
        issues_detected = []
        health_score = 10
        
        disease_display = "Healthy Crop"
        pest_display = "No pests detected."
        rag_search_terms = []
        
        # 1. Disease Inference
        if self.disease_session and self.disease_classes:
            input_tensor = self._preprocess_image(image, 224)
            outputs = self.disease_session.run(None, {'input': input_tensor})[0][0]
            
            probs = self._get_confidence(outputs)
            disease_idx = int(np.argmax(probs))
            confidence = probs[disease_idx] * 100
            
            # Map index to English name
            disease_name = self.disease_classes[disease_idx] if disease_idx < len(self.disease_classes) else f"Unknown ID {disease_idx}"
            
            # The "Healthy" Override
            if "healthy" not in disease_name.lower():
                disease_display = f"Disease: {disease_name} (Conf: {confidence:.1f}%)"
                issues_detected.append(disease_display)
                rag_search_terms.append(disease_name)
                health_score -= 4
            else:
                disease_display = f"Healthy Crop (Conf: {confidence:.1f}%)"

        # 2. Pest Inference
        if self.pest_session and self.pest_classes:
            input_tensor = self._preprocess_image(image, 384)
            outputs = self.pest_session.run(None, {'input': input_tensor})[0][0]
            
            probs = self._get_confidence(outputs)
            pest_idx = int(np.argmax(probs))
            confidence = probs[pest_idx] * 100
            
            pest_name = self.pest_classes[pest_idx] if pest_idx < len(self.pest_classes) else f"Unknown ID {pest_idx}"
            
            # Confidence Threshold & "Healthy/None" override
            if confidence > 60.0 and "healthy" not in pest_name.lower() and "none" not in pest_name.lower():
                pest_display = f"Pest: {pest_name} (Conf: {confidence:.1f}%)"
                issues_detected.append(pest_display)
                rag_search_terms.append(pest_name)
                health_score -= 4

        # 3. Vector DB Retrieval (RAG)
        advice = "No immediate chemical intervention required. Continue regular crop monitoring."
        if rag_search_terms and self.rag_db:
            # Querying the database using ONLY the English names (no percentages or IDs)
            query = f"Recommended pesticide and treatment for {' and '.join(rag_search_terms)}"
            results = self.rag_db.similarity_search(query, k=2)
            if results:
                advice = "\n\n".join([doc.page_content for doc in results])

        return {
            'disease_text': disease_display,
            'pest_text': pest_display,
            'score': max(1, health_score),
            'advice': advice
        }