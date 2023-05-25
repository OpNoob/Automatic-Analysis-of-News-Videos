import json
import time

from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter
import os
from tqdm import tqdm
from utils import secondsStr, parseTimestamp2
from validation import getTranscriptData


def extactFramesDir(video_dir, output_dir, limit: int = None, num_frames=5, log_name="log.json", skip_existing=True):
    """
    extracts the keyframes from a video directory

    :param video_dir: The video directory
    :param output_dir: The output directory for the key frames
    :param limit: Limit the number of videos to process
    :param num_frames: The number of frames to extract per video
    :param log_name: The log file name to show extra information about the extraction
    :param skip_existing: Boolean to skip existing folder, meaning that extraction was already complete
    :return:
    """
    os.makedirs(output_dir, exist_ok=True)

    videos = [f for f in os.listdir(video_dir) if f.endswith(".mp4")]
    if limit is not None:
        videos = videos[:limit]

    vd = Video()

    pbar = tqdm(total=len(videos), desc="Extracting key frames from videos")
    for video_name in videos:
        time_start = time.time()

        name = os.path.splitext(video_name)[0]
        video_path = os.path.join(video_dir, video_name)
        video_output_dir = os.path.join(output_dir, name)
        transcript_path = os.path.join(video_dir, name + ".json")

        # Skip if directory already exists
        if skip_existing and os.path.exists(video_output_dir):
            print(f"'{video_output_dir}' already exists, skipping")
            pbar.update()
            continue

        os.makedirs(video_output_dir, exist_ok=True)

        # Loads validation
        with open(transcript_path, "r") as fv:
            transcript = json.load(fv)

        diskwriter = KeyFrameDiskWriter(location=video_output_dir)
        vd.extract_video_keyframes(
            no_of_frames=num_frames, file_path=video_path,
            writer=diskwriter
        )

        time_end = time.time()
        duration = time_end - time_start

        log = {
            "Duration_s": duration,
            "Duration": secondsStr(duration),
            "valid_info": getTranscriptData(transcript, name=name, time_stamp_parser=parseTimestamp2),
            "Transcript": transcript,
        }

        log_path = os.path.join(video_output_dir, log_name)
        with open(log_path, "w") as fl:
            json.dump(log, fl, indent=4)

        pbar.update()
    pbar.close()


if __name__ == "__main__":
    extactFramesDir("data/NewsSegments", "data/comparing/keyframes", num_frames=10, limit=None)
