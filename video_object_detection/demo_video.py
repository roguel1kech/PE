from video_model import video_detector

def main():
    path = input("Укажи путь к видео (.mp4): ")
    results = video_detector.process_video(path, frame_step=30)

    for frame_info in results:
        print(f"\nКадр {frame_info['frame']}:")
        for det in frame_info["detections"]:
            print(f"  label_id={det['label_id']}, score={det['score']:.3f}, box={det['box']}")

if __name__ == "__main__":
    main()
