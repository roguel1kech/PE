from sentiment_model import sentiment_analyzer

def main():
    print("Введите текст для анализа (пустая строка — выход):")
    while True:
        text = input("> ")
        if not text.strip():
            print("Выход.")
            break

        result = sentiment_analyzer.predict(text)
        print(f"Тональность: {result['label']} (score={result['score']:.4f})")

if __name__ == "__main__":
    main()
