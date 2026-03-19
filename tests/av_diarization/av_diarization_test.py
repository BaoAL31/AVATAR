from src.diarization.run_diarization import diarize

if __name__ == "__main__":
    video_name = "-FaXLcSFjUI_trimmed"
    diarize(video_name=video_name, visualize=True)