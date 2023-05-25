from PyQt5 import QtWidgets, QtGui, QtCore
from PyQt5.QtWidgets import QFileDialog, QMessageBox, QProgressBar, QCheckBox
from PyQt5.QtCore import Qt, QThread, pyqtSignal

import os.path
import random
import pandas as pd
from matplotlib import pyplot as plt

import utils
from utils import secondsStr, getTimeStamps
from video_analysis_2 import *


def nameEmpty(name: str):
    if name is None or isinstance(name, float) and pd.isna(name) or name.isspace():
        return True
    return False


def generateColor():
    r = random.randint(0, 255)
    g = random.randint(0, 255)
    b = random.randint(0, 255)
    color = (r / 255, g / 255, b / 255)
    return color


def generateTimelineValidationAX(validation_path, ax, color_dict=None):
    if color_dict is None:
        color_dict = dict()

    df = pd.read_excel(validation_path, header=0)
    rows = [tuple(row) for row in df.values]

    names_durations = dict()
    for row in rows:
        time_start_s, time_end_s, name, name_shown, presenter = getRowData(row)
        if name is None:
            continue

        duration = time_end_s - time_start_s
        if name in names_durations:
            names_durations[name] += duration
        else:
            names_durations[name] = duration

    labels = set()
    for row in rows:
        time_start_s, time_end_s, name, name_shown, presenter = getRowData(row)
        if name is None:
            continue

        duration = time_end_s - time_start_s

        total_duration = names_durations[name]

        if name not in color_dict:
            color_dict[name] = generateColor()

        label = f'{name} ({secondsStr(total_duration)})'
        if label in labels:
            label = None
        else:
            labels.add(label)
        ax.barh(0, duration, left=time_start_s, color=color_dict[name],
                label=label)
        ax.text(time_start_s + duration / 2, 0, name, ha='center', va='center', rotation=90)

    ax.set_yticks([])
    ax.set_xlabel('Timestamps')
    ax.set_title('Validation')
    ax.legend()


def generateTimelineAnalysis(na,
                             show=False, save_path=None,
                             validation_path=None
                             ):
    graph_name = os.path.basename(na.video_name)

    # na = NewsAnalysis.staticLoad(analysis_dir, analysis_data_name=analysis_data_name)

    # na = NewsAnalysis(None, analysis_dir, analysis_data_name=analysis_data_name, load_previous=True)
    durations_data = na.getDurations()
    color_dict = dict()

    if validation_path is None:
        fig, ax1 = plt.subplots(figsize=(15, 5))
    else:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))

        generateTimelineValidationAX(validation_path, ax2, color_dict=color_dict)
        ax2.set_xlim(0, na.duration)
    ax1.set_xlim(0, na.duration)

    labels = set()
    for scene in na.scenes:
        time_start_s, time_end_s = getTimeStamps(scene)
        duration = time_end_s - time_start_s

        face_info = scene.face_info
        name = face_info.name
        if name is None:
            continue

        total_duration = durations_data[face_info].duration

        if name not in color_dict:
            color_dict[name] = generateColor()

        label = f'{name} ({secondsStr(total_duration)})'
        if label in labels:
            label = None
        else:
            labels.add(label)
        ax1.barh(0, duration, left=time_start_s, color=color_dict[name],
                 label=label)
        ax1.text(time_start_s + duration / 2, 0, name, ha='center', va='center', rotation=90)

    ax1.set_yticks([])
    ax1.set_xlabel('Timestamps (seconds)')
    ax1.set_title('Analysis')
    ax1.legend()

    fig.subplots_adjust(hspace=1)
    fig.suptitle(graph_name)

    if save_path is not None:
        fig.savefig(save_path)
    if show:
        plt.show()

    return fig


def generateTimelineDirectory(directory):
    timeline_folder = "timeline"
    timeline_validation_folder = "timeline_validation"

    folders = os.listdir(directory)

    os.makedirs(os.path.join(directory, timeline_folder), exist_ok=True)
    os.makedirs(os.path.join(directory, timeline_validation_folder), exist_ok=True)

    for folder in folders:
        if folder in ["database", timeline_folder, timeline_validation_folder]:
            continue

        folder_path = os.path.join(directory, folder)

        na = NewsAnalysis.staticLoad(folder_path)
        fig = generateTimelineAnalysis(na,
                                 validation_path=os.path.join(utils.VALIDATION_DIR, folder + ".xlsx"),
                                 show=False,
                                 save_path=os.path.join(directory, timeline_validation_folder, folder + ".png")
                                 )
        plt.close(fig)

        fig = generateTimelineAnalysis(na,
                                 # validation_path=os.path.join(utils.VALIDATION_DIR, folder + ".xlsx"),
                                 show=False,
                                 save_path=os.path.join(directory, timeline_folder, folder + ".png")
                                 )
        plt.close(fig)


def test():
    na = NewsAnalysis.staticLoad("data/comparing_segments/tracking_database_2/01.08.2022 segment #0")
    generateTimelineAnalysis(na,
                             validation_path="data/Validation2/01.08.2022 segment #0.xlsx",
                             show=True)


def main():
    # generateTimelineDirectory("data/comparing_segments/tracking_skip_1_average_encodings_default_res")

    generateTimelineDirectory("data/comparing_segments/tracking_skip_1_average_encodings")
    generateTimelineDirectory("data/comparing_segments/tracking_skip_0_5_average_encodings_min_scene_0")


if __name__ == "__main__":
    # test()
    main()
