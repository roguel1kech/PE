from audio_model import audio_classifier

def main():
    path = input("Укажи путь к WAV-файлу: ")
    result = audio_classifier.predict(path)
    print(f"Класс: {result['label']} (score={result['score']:.4f})")

if __name__ == "__main__":
    main()
