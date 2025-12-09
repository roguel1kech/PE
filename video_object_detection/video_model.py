import cv2
import torch
from transformers import AutoImageProcessor, DetrForObjectDetection
from typing import List, Dict


class VideoObjectDetector:
    def __init__(self, model_name: str = "facebook/detr-resnet-50"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = DetrForObjectDetection.from_pretrained(model_name)
        self.model.to(self.device)
        self.model.eval()

    @torch.inference_mode()
    def detect_in_frame(self, frame) -> List[Dict]:
        inputs = self.processor(images=frame, return_tensors="pt").to(self.device)
        outputs = self.model(**inputs)

        target_sizes = torch.tensor([frame.shape[:2]])  # (h, w)
        results = self.processor.post_process_object_detection(
            outputs, threshold=0.7, target_sizes=target_sizes
        )[0]

        detections = []
        for score, label, box in zip(
            results["scores"], results["labels"], results["boxes"]
        ):
            detections.append({
                "label_id": int(label),
                "score": float(score),
                "box": [float(x) for x in box]
            })

        return detections

    def process_video(self, video_path: str, frame_step: int = 30):
        """
        frame_step=30 => берем примерно 1 кадр в секунду при 30 fps.
        """
        cap = cv2.VideoCapture(video_path)
        frame_idx = 0

        all_results = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_step == 0:
                detections = self.detect_in_frame(frame)
                all_results.append({
                    "frame": frame_idx,
                    "detections": detections
                })

            frame_idx += 1

        cap.release()
        return all_results


video_detector = VideoObjectDetector()
