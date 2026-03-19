from src.transcription.run_usr import load_and_preprocess_track, load_model

VIDEO_NAME = "-FaXLcSFjUI_trimmed"

def test_load_and_preprocess_track():
    track_idx = 0
    video, audio = load_and_preprocess_track(VIDEO_NAME, track_idx)
    print("Video tensor shape:", video.shape)
    print("Audio tensor shape:", audio.shape)

def test_load_usr_model():
    model, cfg = load_model('/home/jembo/AVATAR/models/usr/checkpoints/baseplus_high_resource_lrs3vox2.pth')
    print("Model loaded!")
    print(model)

if __name__ == "__main__":
    test_load_and_preprocess_track()
    test_load_usr_model()
