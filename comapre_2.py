import os
from video_analysis_2 import *
from validation2 import validateVideos, mainValidate
from validation3 import compareResults, databaseChart

"""
Script to analyse News Video Segments with different configurations
"""

NEWS_VIDEO_DIR = "data/NewsSegments"
OUTPUT_DIR = "data/comparing_segments/"
RESULTS_DIR = "data/comparing_segments_results"
VALIDATION_DIR = "data/Validation2"

FOLDERS_PATH = list()


def getTimeScene(s: NewsAnalysis.Scene):
    return s.time_start_s, s.time_end_s


def tracking_skip_2():
    output_folder = os.path.join(OUTPUT_DIR, "tracking_skip_2")
    FOLDERS_PATH.append(output_folder)
    analyseDirMulti(NEWS_VIDEO_DIR, output_folder,
                    skip_seconds_check=2,
                    resolution=(640, 360),
                    processes=4,
                    show_frame=True,
                    overwrite_tracker=True)


def tracking_skip_0_5():
    output_folder = os.path.join(OUTPUT_DIR, "tracking_skip_0_5")
    FOLDERS_PATH.append(output_folder)
    analyseDirMulti(NEWS_VIDEO_DIR, output_folder,
                    skip_seconds_check=0.5,
                    resolution=(640, 360),
                    processes=4,
                    show_frame=True,
                    overwrite_tracker=True)


def tracking_database():
    output_folder = os.path.join(OUTPUT_DIR, "tracking_database")
    FOLDERS_PATH.append(output_folder)
    analyseDir(NEWS_VIDEO_DIR, output_folder,
               skip_seconds_check=2,
               resolution=(640, 360),
               processes=7,
               show_frame=False,
               overwrite_tracker=True,
               export_db=os.path.join(output_folder, "database"))


def tracking_database_2():
    output_folder = os.path.join(OUTPUT_DIR, "tracking_database_2")
    FOLDERS_PATH.append(output_folder)
    analyseDir(NEWS_VIDEO_DIR, output_folder,
               # skip_seconds_check=2,
               skip_seconds_check=1,

               # min_scene_duration=0,
               min_scene_duration=0,

               # tolerance_match=0.6,
               tolerance_match=0.5,

               resolution=(640, 360),
               overwrite_tracker=True,
               tracker_type="KCF",

               processes=7,
               show_frame=False,
               export_db=os.path.join(output_folder, "database"))


def tracking_skip_2_average_encodings_min_scene_0():
    output_folder = os.path.join(OUTPUT_DIR, "tracking_skip_2_average_encodings_min_scene_0")
    FOLDERS_PATH.append(output_folder)
    analyseDirMulti(NEWS_VIDEO_DIR, output_folder,
                    skip_seconds_check=2,
                    resolution=(640, 360),
                    processes=4,
                    show_frame=True,
                    min_scene_duration=0,
                    overwrite_tracker=True)


def tracking_skip_0_5_average_encodings_min_scene_0():
    output_folder = os.path.join(OUTPUT_DIR, "tracking_skip_0_5_average_encodings_min_scene_0")
    FOLDERS_PATH.append(output_folder)
    analyseDirMulti(NEWS_VIDEO_DIR, output_folder,
                    skip_seconds_check=0.5,
                    resolution=(640, 360),
                    processes=4,
                    show_frame=True,
                    min_scene_duration=0,
                    overwrite_tracker=True)


def tracking_skip_1_average_encodings():
    output_folder = os.path.join(OUTPUT_DIR, "tracking_skip_1_average_encodings")
    FOLDERS_PATH.append(output_folder)
    analyseDirMulti(NEWS_VIDEO_DIR, output_folder,
                    skip_seconds_check=1,
                    resolution=(640, 360),
                    processes=4,
                    show_frame=True,
                    min_scene_duration=0,
                    overwrite_tracker=True)


def tracking_skip_2_average_encodings_default_res():
    output_folder = os.path.join(OUTPUT_DIR, "tracking_skip_2_average_encodings_default_res")
    FOLDERS_PATH.append(output_folder)
    analyseDirMulti(NEWS_VIDEO_DIR, output_folder,
                    skip_seconds_check=2,
                    # resolution=(640, 360),
                    processes=4,
                    show_frame=True,
                    min_scene_duration=0,
                    overwrite_tracker=True)


def tracking_skip_1_average_encodings_default_res():
    output_folder = os.path.join(OUTPUT_DIR, "tracking_skip_1_average_encodings_default_res")
    FOLDERS_PATH.append(output_folder)
    analyseDirMulti(NEWS_VIDEO_DIR, output_folder,
                    skip_seconds_check=1,
                    # resolution=(640, 360),
                    processes=4,
                    show_frame=True,
                    min_scene_duration=0,
                    overwrite_tracker=True)


def tracking_skip_2_first_encodings():
    output_folder = os.path.join(OUTPUT_DIR, "tracking_skip_2_first_encodings")
    FOLDERS_PATH.append(output_folder)
    analyseDirMulti(NEWS_VIDEO_DIR, output_folder,
                    skip_seconds_check=2,
                    resolution=(640, 360),
                    processes=7,
                    show_frame=True,
                    min_scene_duration=0,
                    overwrite_tracker=True,
                    encodings_selection="first",
                    )


def tracking_database_1_0_5():
    output_folder = os.path.join(OUTPUT_DIR, "tracking_database_1_0_5")
    FOLDERS_PATH.append(output_folder)
    analyseDir(NEWS_VIDEO_DIR, output_folder,
               skip_seconds_check=1,
               min_scene_duration=0,

               # tolerance_match=0.6,
               tolerance_match=0.5,

               resolution=(640, 360),
               overwrite_tracker=True,
               tracker_type="KCF",

               processes=7,
               show_frame=False,
               export_db=os.path.join(output_folder, "database"))


def tracking_database_1_0_6():
    output_folder = os.path.join(OUTPUT_DIR, "tracking_database_1_0_6")
    FOLDERS_PATH.append(output_folder)
    analyseDir(NEWS_VIDEO_DIR, output_folder,
               skip_seconds_check=1,
               min_scene_duration=0,

               # tolerance_match=0.6,
               tolerance_match=0.6,

               resolution=(640, 360),
               overwrite_tracker=True,
               tracker_type="KCF",

               processes=7,
               show_frame=False,
               export_db=os.path.join(output_folder, "database"))


def validateDir():
    pbar = tqdm(total=len(FOLDERS_PATH), desc=f"Getting results in directory '{OUTPUT_DIR}'")
    for output_analysis_dir in FOLDERS_PATH:
        # results_dir = os.path.join(RESULTS_DIR, folder)
        # output_analysis_dir = os.path.join(OUTPUT_DIR, folder)

        # validateVideos(RESULTS_DIR, output_analysis_dir, VALIDATION_DIR, getTimeScene=getTimeScene)
        validateVideos(RESULTS_DIR, output_analysis_dir, VALIDATION_DIR, getTimeScene=getTimeScene, show=False)

        pbar.update()
    pbar.close()


def main():
    # tracking_skip_2()
    # tracking_skip_0_5()
    # tracking_database()
    # tracking_database_2()
    # tracking_skip_2_average_encodings_min_scene_0()
    # tracking_skip_0_5_average_encodings_min_scene_0()

    # tracking_skip_1_average_encodings()
    # tracking_skip_2_average_encodings_default_res()

    # tracking_skip_2_first_encodings()
    # tracking_skip_1_average_encodings_default_res()

    tracking_database_1_0_5()
    tracking_database_1_0_6()
    databaseChart("tracking_database_1_0_5")
    databaseChart("tracking_database_1_0_6")

    # validateDir()
    # compareResults()


if __name__ == "__main__":
    main()
