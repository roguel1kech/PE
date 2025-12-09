from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from typing import Dict


class SentimentAnalyzer:
    def __init__(self, model_name: str = "blanchefort/rubert-base-cased-sentiment"):
        """
        Русскоязычная модель анализа тональности.
        Классы:
        0 - NEGATIVE
        1 - NEUTRAL
        2 - POSITIVE
        """
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

        self.id2label = {
            0: "NEGATIVE",
            1: "NEUTRAL",
            2: "POSITIVE"
        }

    @torch.inference_mode()
    def predict(self, text: str) -> Dict:
        if not text or not text.strip():
            return {"label": "NEUTRAL", "score": 0.0, "raw": None}

        inputs = self.tokenizer(
            text,
            padding=True,
            max_length=256,
            truncation=True,
            return_tensors="pt"
        ).to(self.device)

        logits = self.model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        label_id = int(probs.argmax())
        label = self.id2label[label_id]
        score = float(probs[label_id])

        return {"label": label, "score": score, "raw": probs.tolist()}


sentiment_analyzer = SentimentAnalyzer()
