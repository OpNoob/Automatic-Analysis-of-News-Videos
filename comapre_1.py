import os
from video_analysis_1 import *
from validation2 import validateVideos

"""
Script to analyse News Video Segments with different configurations
"""

NEWS_VIDEO_DIR = "data/NewsSegments"
OUTPUT_DIR = "data/comparing_segments/"
RESULTS_DIR = "data/comparing_segments_results"
VALIDATION_DIR = "data/Validation2"

FOLDERS_PATH = list()


def getTimeScene(s: NewsAnalysis.Scene):
    return s.scene_timestamps[0].get_seconds(), s.scene_timestamps[1].get_seconds()


def skip_seconds_2_2():
    output_folder = os.path.join(OUTPUT_DIR, "skip_seconds_2_2")
    FOLDERS_PATH.append(output_folder)
    analyseDirMulti(NEWS_VIDEO_DIR, output_folder,
                    skip_seconds_check=2,
                    skip_scene_seconds_check=2,
                    resolution=(640, 360),
                    processes=3)


def skip_seconds_3_2():
    output_folder = os.path.join(OUTPUT_DIR, "skip_seconds_3_2")
    FOLDERS_PATH.append(output_folder)
    analyseDirMulti(NEWS_VIDEO_DIR, output_folder,
                    skip_seconds_check=3,
                    skip_scene_seconds_check=2,
                    resolution=(640, 360),
                    processes=3)


def skip_seconds_1_2():
    output_folder = os.path.join(OUTPUT_DIR, "skip_seconds_1_2")
    FOLDERS_PATH.append(output_folder)
    analyseDirMulti(NEWS_VIDEO_DIR, output_folder,
                    skip_seconds_check=1,
                    skip_scene_seconds_check=2,
                    resolution=(640, 360),
                    processes=3)


def skip_seconds_2_1():
    output_folder = os.path.join(OUTPUT_DIR, "skip_seconds_2_1")
    FOLDERS_PATH.append(output_folder)
    analyseDirMulti(NEWS_VIDEO_DIR, output_folder,
                    skip_seconds_check=2,
                    skip_scene_seconds_check=1,
                    resolution=(640, 360),
                    processes=3)


def skip_seconds_2_3():
    output_folder = os.path.join(OUTPUT_DIR, "skip_seconds_2_3")
    FOLDERS_PATH.append(output_folder)
    analyseDirMulti(NEWS_VIDEO_DIR, output_folder,
                    skip_seconds_check=2,
                    skip_scene_seconds_check=3,
                    resolution=(640, 360),
                    processes=3)


def skip_seconds_default_resolution():
    output_folder = os.path.join(OUTPUT_DIR, "skip_seconds_default_resolution")
    FOLDERS_PATH.append(output_folder)
    analyseDirMulti(NEWS_VIDEO_DIR, output_folder,
                    skip_seconds_check=2,
                    skip_scene_seconds_check=2,
                    processes=3)


def validateDir():
    pbar = tqdm(total=len(FOLDERS_PATH), desc=f"Getting results in directory '{OUTPUT_DIR}'")
    for output_analysis_dir in FOLDERS_PATH:
        # results_dir = os.path.join(RESULTS_DIR, folder)
        # output_analysis_dir = os.path.join(OUTPUT_DIR, folder)

        validateVideos(RESULTS_DIR, output_analysis_dir, VALIDATION_DIR, getTimeScene=getTimeScene)

        pbar.update()
    pbar.close()


def main():
    skip_seconds_2_2()
    skip_seconds_3_2()
    skip_seconds_1_2()
    skip_seconds_2_1()
    skip_seconds_2_3()
    skip_seconds_default_resolution()

    validateDir()


if __name__ == "__main__":
    main()
