from src.preprocessing.mouth_crop import process_video

if __name__ == "__main__":
    video_name = "-FaXLcSFjUI_trimmed"
    base_folder = "/home/jembo/AVATAR/data/processed"
    process_video(video_name, base_folder)