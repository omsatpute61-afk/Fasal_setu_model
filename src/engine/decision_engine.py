# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false
import os
import numpy as np
import onnxruntime as ort # type: ignore
from PIL import Image
import concurrent.futures
from typing import Any, Optional

class DecisionEngine:
    def __init__(self) -> None:
        self.src_dir: str = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        
        self.disease_onnx: str = os.path.join(self.src_dir, 'weights', 'disease_model.onnx')
        self.pest_onnx: str = os.path.join(self.src_dir, 'weights', 'pest_model.onnx')
        self.disease_labels_path: str = os.path.join(self.src_dir, 'weights', 'disease_labels.txt')
        self.pest_labels_path: str = os.path.join(self.src_dir, 'weights', 'pest_labels.txt')

        self.disease_session: Any = self._load_onnx(self.disease_onnx)
        self.pest_session: Any = self._load_onnx(self.pest_onnx)
        
        self.disease_classes: list[str] = self._load_labels(self.disease_labels_path)
        self.pest_classes: list[str] = self._load_labels(self.pest_labels_path)

    def _load_onnx(self, path: str) -> Any:
        if not os.path.exists(path):
            return None
        try:
            opts: Any = ort.SessionOptions()  # type: ignore
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL  # type: ignore
            opts.intra_op_num_threads = int(os.cpu_count() or 4)
            return ort.InferenceSession(path, sess_options=opts, providers=['CPUExecutionProvider'])  # type: ignore
        except Exception as e:
            print(f"Failed to load ONNX: {e}")
            return None

    def _load_labels(self, path: str) -> list[str]:
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
        return np.expand_dims(np.transpose(img_data, (2, 0, 1)), axis=0)

    def _get_confidence(self, logits: np.ndarray) -> np.ndarray:
        e_x = np.exp(logits - np.max(logits))
        return e_x / e_x.sum()

    def process_image(self, image: Image.Image) -> dict[str, Any]:
        health_score: int = 10
        escalate_kvk: bool = False
        
        disease_display: str = "Healthy Crop"
        pest_display: str = "No pests detected."

        def run_disease() -> Optional[tuple[int, float]]:
            if not (self.disease_session and self.disease_classes):
                return None
            input_tensor = self._preprocess_image(image, 224)
            outputs: Any = self.disease_session.run(None, {'input': input_tensor})[0][0]
            probs: np.ndarray = self._get_confidence(outputs)
            idx: int = int(np.argmax(probs))
            return idx, float(probs[idx] * 100)

        def run_pest() -> Optional[tuple[int, float]]:
            if not (self.pest_session and self.pest_classes):
                return None
            input_tensor = self._preprocess_image(image, 384)
            outputs: Any = self.pest_session.run(None, {'input': input_tensor})[0][0]
            probs: np.ndarray = self._get_confidence(outputs)
            idx: int = int(np.argmax(probs))
            return idx, float(probs[idx] * 100)

        # Execute both models in parallel
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_d = executor.submit(run_disease)
            future_p = executor.submit(run_pest)
            disease_res = future_d.result()
            pest_res = future_p.result()

        # 1. Disease Processing
        if disease_res:
            d_idx, d_conf = disease_res
            raw_name: str = self.disease_classes[d_idx] if d_idx < len(self.disease_classes) else f"Unknown ID {d_idx}"
            d_name: str = raw_name.replace("_", " ")

            if d_conf < 85.0:
                disease_display = f"Uncertain Diagnosis ({d_conf:.1f}%)"
                escalate_kvk = True
                health_score -= 3
            else:
                if "healthy" in d_name.lower():
                    disease_display = f"Healthy Crop (Conf: {d_conf:.1f}%)"
                else:
                    disease_display = f"Disease: {d_name} (Conf: {d_conf:.1f}%)"
                    health_score -= 4

        # 2. Pest Processing
        if pest_res:
            p_idx, p_conf = pest_res
            raw_pname: str = self.pest_classes[p_idx] if p_idx < len(self.pest_classes) else f"Unknown ID {p_idx}"
            p_name: str = raw_pname.replace("_", " ")

            if p_conf < 80.0:
                pest_display = f"Uncertain Diagnosis ({p_conf:.1f}%)"
                escalate_kvk = True
                health_score -= 2
            else:
                if "healthy" not in p_name.lower() and "none" not in p_name.lower():
                    pest_display = f"Pest: {p_name} (Conf: {p_conf:.1f}%)"
                    health_score -= 4
                else:
                    pest_display = f"No pests detected (Conf: {p_conf:.1f}%)"

        return {
            'disease_text': disease_display,
            'pest_text': pest_display,
            'score': max(1, health_score),
            'escalate_kvk': escalate_kvk
        }