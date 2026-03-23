from src.pipeline import Pipeline

def test_pipeline():
    video_path = "/home/jembo/AVATAR/data/raw/-FaXLcSFjUI_trimmed.mp4"
    output_dir = "/home/jembo/AVATAR/data/processed/-FaXLcSFjUI_trimmed"
    pipeline = Pipeline(video_path, output_dir)
    results = pipeline.run()
    for entry in results:
        print(f"[{entry['start']:.2f}s - {entry['end']:.2f}s] {entry['speaker']}: {entry['transcription']}")

if __name__ == "__main__":
    test_pipeline()