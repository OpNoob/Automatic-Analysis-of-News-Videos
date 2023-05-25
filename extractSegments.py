import json
import os
import subprocess
import time

from tqdm import tqdm
from utils import parseTimestamp, secondsStr, getDuplicateName
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.video.io.VideoFileClip import VideoFileClip

BASE_VIDEO_DIR = "data/extract_full_news"
BASE_ARTICLE_DIR = "data/Article_Links_Extraction"


def extractSegments(extract_dir):
    time_start = time.time()
    os.makedirs(extract_dir, exist_ok=True)

    video_files = [f for f in os.listdir(BASE_VIDEO_DIR) if f.endswith(".mp4")]

    files = [f for f in os.listdir(BASE_ARTICLE_DIR) if f.endswith(".json")]  # get json files
    pbar = tqdm(total=len(files), desc="Extracting segments")
    for json_file_name in files:
        name = os.path.splitext(json_file_name)[0]

        if name + ".mp4" not in video_files:
            print(f"No video file for '{name}'")
            continue

        json_path = os.path.join(BASE_ARTICLE_DIR, json_file_name)
        with open(json_path, "r", encoding="utf-8") as f:
            json_dict = json.load(f)

            # Verification that names match
            news_date = json_dict["News date"]
            if news_date != name:
                print(f"Name '{name}' and News date '{news_date}' did not Match!")
            transcripts = json_dict["Transcripts"]
            for transcript in transcripts:
                # Check if no transcripts
                if not transcript["transcript mlt"] and not transcript["transcript eng"]:
                    continue

                # Get timestamp in seconds
                time_stamp = transcript["Time stamp"]
                time_stamp_s = parseTimestamp(time_stamp, name=name)
                if time_stamp_s is False:
                    continue

                # Get end timestamp
                duration = transcript["Duration"]
                duration_s = parseTimestamp(duration, name=name)
                if duration_s is False:
                    continue

                time_stamp_end_s = time_stamp_s + duration_s  # Calculate end timestap

                transcript["Time stamp"] = secondsStr(time_stamp_s)
                transcript["Time stamp end"] = secondsStr(time_stamp_end_s)
                transcript["Duration"] = secondsStr(duration_s)
                transcript["News date"] = news_date

                # Segment video
                output_video_path = getDuplicateName(extract_dir, name, extension=".mp4", convention=" segment #?", include_index=True)
                output_json_path = output_video_path.replace(".mp4", ".json")
                intput_video_path = os.path.join(BASE_VIDEO_DIR, name + ".mp4")
                print(intput_video_path, time_stamp_s, time_stamp_end_s, output_video_path)

                try:
                    ffmpeg_extract_subclip(intput_video_path, time_stamp_s, time_stamp_end_s, targetname=output_video_path)
                except Exception as e:
                    try:
                        clip = VideoFileClip(intput_video_path)
                        duration_div = clip.duration
                        if time_stamp_end_s > duration_div:
                            time_stamp_end_s = duration_div
                        subclip = clip.subclip(time_stamp_s, time_stamp_end_s)
                        subclip.write_videofile(output_video_path)
                    except Exception as e2:
                        print(f"Error in '{output_video_path}' with error {str(e2)}")

                # Save transcript to json
                # output_json_path = os.path.join(extract_dir, name + ".json")
                with open(output_json_path, "w") as fo:
                    json.dump(transcript, fo, indent=4)

        pbar.update()
    pbar.close()

    time_end = time.time()
    duration_extract = time_end - time_start
    print("Duration: ", secondsStr(duration_extract))


if __name__ == "__main__":
    extractSegments("data/NewsSegments")
