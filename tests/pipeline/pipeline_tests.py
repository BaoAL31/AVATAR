from src.pipeline import Pipeline
import torch

DEVICE_DEFAULT = torch.device("cuda" if torch.cuda.is_available() else "cpu")
def test_pipeline():
    video_path = "/home/hoangbng/AVATAR/AVATAR/data/raw/-FaXLcSFjUI_trimmed.mp4"
    output_dir = "/home/hoangbng/AVATAR/AVATAR/data/processed/-FaXLcSFjUI_trimmed"
    pipeline = Pipeline(video_path, output_dir, device=DEVICE_DEFAULT, visualize=True)
    results = pipeline.run()
    for entry in results:
        print(f"[{entry['start']:.2f}s - {entry['end']:.2f}s] {entry['speaker']}: {entry['transcription']}")

if __name__ == "__main__":
    test_pipeline()