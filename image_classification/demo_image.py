from image_model import image_classifier

def main():
    path = input("Укажи путь к изображению: ")
    results = image_classifier.predict(path, topk=5)
    print("Top-5 классов:")
    for r in results:
        print(f"{r['label']}: {r['score']:.4f}")

if __name__ == "__main__":
    main()
