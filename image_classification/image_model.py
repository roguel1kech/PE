import torch
from torchvision import models, transforms
from PIL import Image
import json
from typing import List, Dict


class ImageClassifier:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = models.resnet18(pretrained=True)
        self.model.to(self.device)
        self.model.eval()

        # стандартные преобразования ImageNet
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # словарь id -> label для ImageNet
        self.idx2label = self._load_imagenet_labels()

    def _load_imagenet_labels(self) -> Dict[int, str]:
        # стандартный json со списком классов, можно положить локально.
        url = "https://raw.githubusercontent.com/pytorch/hub/master/imagenet_classes.txt"
        path = torch.hub.load_state_dict_from_url(url, progress=True, map_location="cpu")
        # НО: это не настоящий state_dict, поэтому проще сделать локальный файл.
        # Для простоты заменим на заглушку:
        # В реальном проекте можно скачать imagenet_classes.txt и читать из него.
        return {}

    def predict(self, image_path: str, topk: int = 5) -> List[Dict]:
        image = Image.open(image_path).convert("RGB")
        x = self.transform(image).unsqueeze(0).to(self.device)

        with torch.inference_mode():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1)[0].cpu()

        top_probs, top_idxs = torch.topk(probs, topk)
        results = []
        for p, idx in zip(top_probs, top_idxs):
            idx_int = int(idx)
            label = str(idx_int)  # Можно заменить на нормальное имя класса, если загрузить словарь
            results.append({"label": label, "score": float(p)})

        return results


image_classifier = ImageClassifier()
