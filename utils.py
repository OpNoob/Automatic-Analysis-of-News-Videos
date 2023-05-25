import os
import pickle
import sys
import time
import json

import pandas as pd

VALIDATION_DIR = "data/Validation2"

NEWS_VIDEOS_DIR = "data/extract_full_news"
NEWS_VIDEO_SEGMENTS_DIR = "data/NewsSegments"

if getattr(sys, 'frozen', False):
    # running as a PyInstaller bundle
    SCRIPT_DIR_PATH = os.path.dirname(sys.executable)
else:
    # running as a normal Python script
    SCRIPT_DIR_PATH = os.path.dirname(os.path.abspath(__file__))


def get_absolute_path(file_path):
    """
    Constructs an absolute path from a file path.

    If the file path is relative, it is interpreted as relative to the
    directory path of the script file. If the file path is already absolute,
    it is returned as is.
    """
    if os.path.isabs(file_path):
        # the path is already absolute
        return file_path
    else:
        # the path is relative, construct an absolute path relative to the script file
        return os.path.join(SCRIPT_DIR_PATH, file_path)


def getVideoNames(directory: str):
    return [x for x in os.listdir(directory) if x.endswith(".mp4")]


# Loads
def loadPickle(path):
    with open(path, "rb") as f:
        return pickle.load(f)


# Saves
def savePickle(obj, path, show_error=True):
    try:
        abs_path = os.path.abspath(path)
        with open(abs_path, "wb") as f:
            pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    except PermissionError as e:
        if show_error:
            print(str(e))
        return False

    return True


def saveJson(obj: dict, path, show_error=True):
    try:
        abs_path = os.path.abspath(path)
        with open(abs_path, "w") as f:
            json.dump(obj, f, indent=4)
    except PermissionError as e:
        if show_error:
            print(str(e))
        return False

    return True


# Time
def calcTime(function, *args):
    time_start = time.time()
    function(*args)
    time_end = time.time()


def min_secs(time_secs, string_format=False):
    m, s = divmod(time_secs, 60)
    m = round(m, 2)
    s = round(s, 2)

    if string_format:
        return "{}mins, {}secs".format(m, s)

    return m, s


def secondsStr(seconds):
    return_str = ""
    if seconds // (60 * 60) > 0:
        hours = seconds // (60 * 60)
        return_str += f"{hours} hours, "
        seconds -= hours * (60 * 60)

        minutes = seconds // 60
        return_str += f"{minutes} minutes, "
        seconds -= minutes * 60

        return_str += f"{seconds:.2f} seconds"

    elif seconds // 60 > 0:
        minutes = seconds // 60
        return_str += f"{minutes} minutes, "
        seconds -= minutes * 60

        return_str += f"{seconds:.2f} seconds"

    else:
        return_str += f"{seconds:.2f} seconds"

    return return_str


def secondsStrMini(seconds):
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = (seconds % 3600) % 60

    time_format = '{:02d}:{:02d}:{:02d}'.format(hours, minutes, seconds)
    return time_format


# IO
def getDuplicateName(folder, file_name, extension, convention="(?)", starting_index=0, include_index=False):
    """
    Get new name of file already exists.  Basic usage: name = getDuplicateName(folder_dir, )

    :param folder: Directory of file
    :param file_name: File name without extension
    :param extension: Desired extension of file
    :param convention: Convention goes at the end, where the '?' signifies the duplicate index
    :param starting_index: Starting index of duplicate files
    :param include_index: If True, then immediately check file path with index (index would be innate)
    :return:
    """

    def getIndexStr():
        return convention.replace('?', str(duplicate_index))

    duplicate_index = starting_index
    if include_index:
        basename = file_name + getIndexStr() + extension
    else:
        basename = file_name + extension
    file_path = os.path.join(folder, basename)
    while os.path.exists(file_path):
        img_name = f"{file_name}{getIndexStr()}{extension}"
        file_path = os.path.join(folder, img_name)
        duplicate_index += 1
    return file_path


def genericName(name: str, index: int, key: str = "?"):
    return name.replace(key, str(index))


# fp = getDuplicateName("dat/test", "test1", ".json", include_index=True)
# print(fp)

def parseTimestamp(ts: str, name=None):
    """
    Returns the timestamp in seconds format

    :param ts: Timestamp to parse
    :param name: For error output to show name of file
    :return:
    """

    if ts == "N/A":
        return False

    if ":" in ts:
        minute, second = ts.split(":")
    elif ";" in ts:
        minute, second = ts.split(";")
    else:
        print(f"Time stamp broken for '{name}', with timestamp '{ts}'")
        return False
    minute = int(minute)
    second = int(second)
    time_stamp_s = minute * 60 + second
    return time_stamp_s


def parseTimestamp2(ts: str, name=None):
    components = ts.split(",")
    time_s = 0
    for c in components:
        if "hours" in c:
            c = c.replace("hours", "").replace(" ", "")
            time_s += int(c) * 60 * 60
        elif "minutes" in c:
            c = c.replace("minutes", "").replace(" ", "")
            time_s += int(c) * 60
        elif "seconds" in c:
            c = c.replace("seconds", "").replace(" ", "")
            time_s += int(c)
    return time_s


def parseTimestamp3(ts: str, name=None):
    """
    Returns the timestamp in seconds format ()

    :param ts: Timestamp to parse
    :param name: For error output to show name of file
    :return:
    """

    hour, minute, second, frames = ts.split(":")

    hour = int(hour)
    minute = int(minute)
    second = int(second)
    time_stamp_s = hour * 60 * 60 + minute * 60 + second
    return time_stamp_s


def getTimeStamps(scene):
    if hasattr(scene, "time_start_s") and hasattr(scene, "time_end_s"):
        time_start_s, time_end_s = scene.time_start_s, scene.time_end_s
    elif hasattr(scene, "scene_timestamps") and hasattr(scene, "scene_timestamps"):
        time_start_s, time_end_s = scene.scene_timestamps[0].get_seconds(), scene.scene_timestamps[
            1].get_seconds()
    else:
        print("ERROR IN GETTING TIME", scene.__dict__.keys())
        time_start_s = 0
        time_end_s = 0

    return time_start_s, time_end_s


def nameEmpty(name: str):
    if name is None or isinstance(name, float) and pd.isna(name) or isinstance(name, str) and name.isspace():
        return True
    return False


def getRowData(r):
    [time_start, time_end, name, name_shown, presenter] = r

    # Ensure empty string is None
    if nameEmpty(name):
        name = None

    # Parse time stamps
    time_start_s = parseTimestamp3(time_start)
    time_end_s = parseTimestamp3(time_end)

    return time_start_s, time_end_s, name, name_shown, presenter
