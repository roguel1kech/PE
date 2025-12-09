import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import csv
from typing import Dict, Tuple


class AudioClassifier:
    def __init__(
        self,
        model_url: str = "https://tfhub.dev/google/yamnet/1",
        class_map_url: str = "https://raw.githubusercontent.com/tensorflow/models/master/research/audioset/yamnet/yamnet_class_map.csv"
    ):
        """
        Audio event classifier YAMNet.
        model_url - TF Hub URL
        class_map_url - CSV с метками классов
        """
        self.model = hub.load(model_url)
        self.class_names = self._load_class_map(class_map_url)

    def _load_class_map(self, url: str):
        # Очень грубо: скачиваем файл через tf.io, можно заменить на локальный csv
        path = tf.keras.utils.get_file("yamnet_class_map.csv", url)
        class_names = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                class_names.append(row["display_name"])
        return class_names

    def _load_audio(self, wav_path: str) -> Tuple[np.ndarray, int]:
        file_contents = tf.io.read_file(wav_path)
        audio, sample_rate = tf.audio.decode_wav(file_contents, desired_channels=1)
        waveform = tf.squeeze(audio, axis=-1)

        sample_rate = int(sample_rate)
        if sample_rate != 16000:
            # ресемплим до 16кГц
            waveform = tf.audio.resample(waveform, sample_rate, 16000)
            sample_rate = 16000

        return waveform.numpy(), sample_rate

    def predict(self, wav_path: str) -> Dict:
        waveform, sr = self._load_audio(wav_path)

        # YAMNet принимает 16кГц моно
        scores, embeddings, spectrogram = self.model(waveform)

        scores_np = scores.numpy()
        mean_scores = scores_np.mean(axis=0)
        top_class = int(np.argmax(mean_scores))
        score = float(mean_scores[top_class])
        label = self.class_names[top_class] if top_class < len(self.class_names) else str(top_class)

        return {
            "label": label,
            "score": score
        }


audio_classifier = AudioClassifier()
