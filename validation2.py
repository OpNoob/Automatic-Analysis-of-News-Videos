import csv
import os
import pickle
import json
from enum import Enum

from Levenshtein import distance
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

import utils
from utils import parseTimestamp3
from video_analysis_1 import NewsAnalysis
import plots

"""
Validation from annotated videos
"""


def calculate_iou(start1, end1, start2, end2):
    """
    Calculates the Intersection over Union (IoU) of two intervals.

    :param start1: Start point of the first interval.
    :param end1: End point of the first interval.
    :param start2: Start point of the second interval.
    :param end2: End point of the second interval.
    :return: The IoU of the two intervals.
    """
    intersection = max(0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)

    if union == 0:
        return 1

    return intersection / union


def intersection_and_iou(list1, get_time_list1, list2, get_time_list2, iou_threshold: float = 0.0):
    """
    Computes the intersection and IoU values of intervals in two lists, as well as
    non-intersecting intervals in each list.

    :param iou_threshold: IOU threshold for intersections
    :param list1: The first list of intervals.
    :param list2: The second list of intervals.
    :param get_time_list1: Function to get time from list1, in seconds as a tuple (time_start, time_end)
    :param get_time_list2: Function to get time from list2, in seconds as a tuple (time_start, time_end)
    :return: A tuple of three lists:
             1. A list of tuples, where each tuple contains the intersecting intervals and their IoU value.
             2. A list of intervals from list1 that do not intersect with any interval in list2.
             3. A list of intervals from list2 that do not intersect with any interval in list1.
    """
    intersections = []
    non_intersections1 = []
    non_intersections2 = []

    # Initialize dictionaries to keep track of the highest IoU values of each interval in each list
    list_1_highest_iou = {i: 0 for i in range(len(list1))}
    list_2_highest_iou = {i: 0 for i in range(len(list2))}

    # Iterate over all pairs of intervals and calculate their IoU
    for i1, x1 in enumerate(list1):
        for i2, x2 in enumerate(list2):
            time_start_1, time_end_1 = get_time_list1(x1)
            time_start_2, time_end_2 = get_time_list2(x2)
            iou = calculate_iou(time_start_1, time_end_1, time_start_2, time_end_2)

            # Update the highest IoU values for each interval
            list_1_highest_iou[i1] = max(list_1_highest_iou[i1], iou)
            list_2_highest_iou[i2] = max(list_2_highest_iou[i2], iou)

            # If the IoU is greater than 0, add the intersecting intervals and their IoU to the list
            if iou > iou_threshold:
                intersections.append((x1, x2, iou))

    # Add intervals from list1 that do not intersect with any interval in list2 to non_intersections1
    for i, max_iou in list_1_highest_iou.items():
        if max_iou <= iou_threshold:
            non_intersections1.append(list1[i])

    # Add intervals from list2 that do not intersect with any interval in list1 to non_intersections2
    for i, max_iou in list_2_highest_iou.items():
        if max_iou <= iou_threshold:
            non_intersections2.append(list2[i])

    return intersections, non_intersections1, non_intersections2


class Validation:
    class VideoStatus:
        class SceneFlags(Enum):
            ERROR = 0
            Match = 1
            Extra = 2
            Missing = 3
            Mismatch = 4
            MatchNone = 5  # Counts with 'Match'
            DetectionMissing = 6
            DetectionExtra = 7

        class SceneStatus:
            """
            Case Scenario

            Match
            Actual: Person A, Predicted: Person A

            Extra
            Actual: None, Predicted: Person A

            Missing
            Actual: Person A, Predicted: None

            Mismatch
            Actual: Person B, Predicted: Person A

            MatchNone (Counts with Match)
            Actual: None, Predicted: None

            DetectionMissing
            Actual: Person detected, Predicted: No scene information

            DetectionExtra
            Actual: No scene information, Predicted: Person detected
            """

            def __init__(self, actual_name=None, actual_timestamp=None, detection_missing=False, detection_extra=False,
                         predicted_name=None, predicted_time_stamp=None, iou: float = None):
                self.actual_name = actual_name
                self.predicted_name = predicted_name

                self.flag = Validation.VideoStatus.SceneFlags.ERROR  # The type of match
                self.name_distance = None  # Calculated if names are found in both cases
                self.iou = iou

                self.actual_timestamp = actual_timestamp
                self.predicted_time_stamp = predicted_time_stamp

                # Time IoU
                # intersection = min(actual_timestamp[1], predicted_time_stamp[1]) - max(actual_timestamp[0], predicted_time_stamp[0])
                # union = max(actual_timestamp[1], predicted_time_stamp[1]) - min(actual_timestamp[0], predicted_time_stamp[0])
                # self.iou = intersection / union

                def nameEmpty(name: str):
                    if name is None or isinstance(name, float) and pd.isna(name):
                        return True
                    return False

                if detection_missing:  # Detection is missing
                    self.flag = self.flag.DetectionMissing
                elif detection_extra:  # Detection is extra
                    self.flag = self.flag.DetectionExtra
                else:
                    if nameEmpty(actual_name):
                        if nameEmpty(predicted_name):  # Both names empty
                            self.flag = self.flag.MatchNone
                        else:  # Found name where there was not
                            self.flag = self.flag.Extra
                    else:
                        if nameEmpty(predicted_name):  # Did not find name, where there should
                            self.flag = self.flag.Missing
                        else:  # Names found in both cases
                            if actual_name.upper() == predicted_name.upper():
                                self.flag = self.flag.Match
                            else:
                                self.flag = self.flag.Mismatch

                            name_distance = distance(actual_name.upper(), predicted_name.upper())
                            max_length = max(len(actual_name), len(predicted_name))
                            ratio_difference = name_distance / max_length
                            self.name_distance = ratio_difference

            def toDict(self):
                return {
                    "actual_timestamp": self.actual_timestamp,
                    "predicted_time_stamp": self.predicted_time_stamp,
                    "actual_name": self.actual_name,
                    "predicted_name": self.predicted_name,
                    "name_distance": self.name_distance,
                    "flag": self.flag.name,
                    "iou": self.iou,
                }

        def __init__(self, name, analysis_time: int = None):
            self.name = name

            self.scene_statuses = list()
            self.analysis_time = analysis_time

        def add(self, scene_status: SceneStatus):
            self.scene_statuses.append(scene_status)

        def toDict(self):
            return {
                "name": self.name,
                "analysis_time": utils.secondsStr(self.analysis_time),
                "scene_statuses": [s.toDict() for s in self.scene_statuses],
            }

    """
    Metrics: IOU (time), Confusion matrix, OCR accuracy
    """

    def __init__(self, directory, name=None):
        # os.makedirs(directory, exist_ok=True)

        self.directory = directory
        self.name = name

        self.num_annotated_scenes = 0
        self.num_predicted_scenes = 0

        self.video_statuses = list()

    def save(self):
        file_name = "results" if self.name is None else f"{self.name} Validation Results"

        save_pickle_path = os.path.join(self.directory, f"{file_name}.pkl")
        with open(save_pickle_path, "wb") as f:
            pickle.dump(self, f)

        save_json_path = os.path.join(self.directory, f"{file_name}.json")
        with open(save_json_path, "w") as f:
            json.dump(self.toDict(), f, indent=4)

    def load(self, path):
        with open(path, "rb") as f:
            self.__dict__ = pickle.load(f).__dict__

    def add(self, video_status: VideoStatus):
        self.video_statuses.append(video_status)

    def toDict(self):
        return {
            "video_statuses": [s.toDict() for s in self.video_statuses],
        }

    def __iter__(self):
        for vs in self.video_statuses:
            for ss in vs.scene_statuses:
                yield ss

    def getAverageIOU(self):
        iou_sum = 0
        iou_count = 0

        for x in self:
            if x.iou is not None:
                iou_sum += x.iou
                iou_count += 1

        if iou_count == 0:
            return 0
        iou_avg = iou_sum / iou_count
        return iou_avg

    def getAverageNameDistance(self):
        s = 0
        c = 0

        for x in self:
            if x.name_distance is not None:
                s += x.name_distance
                c += 1

        avg = s / c
        return avg

    def getAverageAnalysisTime(self):
        s = 0
        c = 0

        for x in self.video_statuses:
            s += x.analysis_time
            c += 1

        avg = s / c
        return avg

    def getCM(self, graph_path=None, name: str = None, show=False):
        if graph_path is None:
            graph_path = os.path.join(self.directory, "Confusion Matrix.png")
        cm = plots.ConfusionMatrix("Confusion Matrix" if name is None else name, save_location=graph_path)

        flags = Validation.VideoStatus.SceneFlags
        for x in self:
            flag = x.flag
            if flag == flags.Match:
                cm.tp += 1
            elif flag == flags.Mismatch:  # only question here <-
                cm.fp += 1
            elif flag == flags.MatchNone:
                cm.tn += 1
            elif flag == flags.Missing:
                cm.fn += 1
            elif flag == flags.Extra:
                cm.fp += 1
            elif flag == flags.DetectionMissing:
                cm.fn += 1
            elif flag == flags.DetectionExtra:
                cm.fp += 1

        cm.genPlot(show=show)

    def genStats(self):
        self.getPlot()
        self.genClassConfusionMatrix()

    def getPlot(self, show=False, show_general_info=True):
        file_name = "results" if self.name is None else f"{self.name} Validation Results"
        graph_path = os.path.join(self.directory, f"{file_name}.png")

        # Get data
        labels = [f.name for f in self.VideoStatus.SceneFlags]
        count_dict = {flag: 0 for flag in self.VideoStatus.SceneFlags}
        for x in self:
            flag = x.flag
            count_dict[flag] += 1
        data = [count_dict[flag] for flag in self.VideoStatus.SceneFlags]

        # Skipping ERROR flag
        labels = labels[1:]
        data = data[1:]

        # Define the width ratios for the subplots
        if show_general_info:
            width_ratios = [1, 2, 2, len(labels), 5]
            fig, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(1, 5, figsize=(18, 6),
                                                          gridspec_kw={'width_ratios': width_ratios})
        else:
            width_ratios = [1, 2, 2, len(labels)]
            fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(15, 6),
                                                     gridspec_kw={'width_ratios': width_ratios})

        # Create the first graphic for the average values
        avg_time = self.getAverageAnalysisTime()
        time_labels = ["Time Taken"]
        time_data = [avg_time]
        ax1.bar(
            time_labels,
            time_data
        )
        ax1.annotate(f"{avg_time:.2f}", xy=(0, avg_time), ha='center', va='bottom')
        # ax1.set_ylabel("Value")
        ax1.set_title("Time Taken")
        ax1.set_ylim([0, 200])
        ax1.set_xticklabels(time_labels, rotation=45, ha='right')  # Rotate labels by 45 degrees

        # Create the second graphic for the average values
        scene_labels = ["Scenes annotated", "Scenes Predicted"]
        scene_data = [self.num_annotated_scenes, self.num_predicted_scenes]
        ax2.bar(
            scene_labels,
            scene_data
        )
        for i, val in enumerate(scene_data):
            ax2.annotate(str(val), xy=(i, val), ha='center', va='bottom')
        # ax1.set_ylabel("Value")
        ax2.set_title("Scenes detected")
        ax2.set_ylim([0, 10000])
        ax2.set_xticklabels(scene_labels, rotation=45, ha='right')  # Rotate labels by 45 degrees

        # Create the third graphic for the average values
        avg_iou = self.getAverageIOU()
        avg_name_distance = self.getAverageNameDistance()
        general_labels = ["Temporal IoU", "Name Distance"]
        general_data = [avg_iou, avg_name_distance]
        ax3.bar(
            general_labels,
            general_data
        )
        ax3.annotate(f"{avg_iou:.2f}", xy=(0, avg_iou), ha='center', va='bottom')
        ax3.annotate(f"{avg_name_distance:.2f}", xy=(1, avg_name_distance), ha='center', va='bottom')
        # ax2.set_ylabel("Value")
        ax3.set_title("Average Values")
        ax3.set_ylim([0, 1])
        ax3.set_xticklabels(general_labels, rotation=45, ha='right')  # Rotate labels by 45 degrees

        # Create the fourth graphic for the bar chart
        for i, val in enumerate(data):
            ax4.annotate(str(val), xy=(i, val), ha='center', va='bottom')
        ax4.bar(labels, data)
        ax4.set_xlabel("Flag Name")
        # ax3.set_ylabel("Value")
        ax4.set_title("Match Flags")
        ax4.set_ylim([0, 2000])
        ax4.set_xticklabels(labels, rotation=45, ha='right')  # Rotate labels by 45 degrees

        if show_general_info:
            # Add a general information subplot
            general_info = f"Time is in Seconds.\n\n" \
                           f"Scenes are a timeperiod where a person occurs.\n\n" \
                           f"Average Temporal IoU is measure by \n(timeframe intersection / timeframe union).\n\n" \
                           f"Average Name Distance is measured by \n(Levenshtein / max_length).\n\n" \
                           f"For the matching flags, each annotated scene was \nwas compared to a predicted scene, \nif the timestamps overlapped, then \nand IoU was calculated along with an appropriate flag\n\n" \
                           f"== Flags ==\n" \
                           f"Match: Instances where names match.\n" \
                           f"Extra: Instances where a name was predicted but in \nannotations there was no name.\n" \
                           f"Missing: Instances where there was no predicted name \nbut in annotations there way.\n" \
                           f"Mismatch: Instances where predicted and annotated \nnames did not match.\n" \
                           f"MatchNone: Instances where both predicted and \nannotated names were empty.\n" \
                           f"DetectionMissing: Instances there was no face was \npredicted, but in annotated there was a face.\n" \
                           f"DetectionExtra: Instances where a face is predicted, \nbut there was no face annotated within that timeframe.\n"
            ax5.set_title("General Information")
            ax5.text(0.05, 0.95, general_info, fontsize=7, ha='left', va='top')

        # Add the title to both graphics
        fig.suptitle(file_name, fontsize=16, fontweight='bold')

        # Save the combined figure as a single file
        fig.savefig(graph_path, bbox_inches='tight')

        # Show the graphics
        if show:
            plt.show()

    def genClassConfusionMatrix(self):
        file_name = "results" if self.name is None else f"{self.name} Class Confusion Matrix"
        graph_path = os.path.join(self.directory, f"{file_name}.png")


def iterNewsSegments(val_dir, analysis_dir, analysis_data_name="analysis_data"):
    # Get filenames of validation files
    validation_filenames = [f for f in os.listdir(val_dir) if f.endswith(".xlsx")]

    # Loop through each validation file
    for file in validation_filenames:
        # Extract file name without extension
        file_name = os.path.splitext(file)[0]

        # Set analysis directory path
        current_analysis_dir = os.path.join(analysis_dir, file_name)

        # Initialize NewsAnalysis object and load analysis data
        na = NewsAnalysis(None, current_analysis_dir, analysis_data_name=analysis_data_name, load_previous=True)

        # Read validation file into pandas dataframe
        v_path = os.path.join(val_dir, file)
        df = pd.read_excel(v_path, header=0)

        # Convert dataframe rows to tuples and store in rows list
        rows = [tuple(row) for row in df.values]

        yield rows, na


def validateVideos(res_dir, analysis_dir, val_dir, analysis_data_name="analysis_data", iou_threshold=0.0,
                   getTimeScene=None, show=False):
    # Create results directory if it doesn't already exist
    os.makedirs(res_dir, exist_ok=True)

    # Initialize Validation object
    valid = Validation(res_dir, name=os.path.basename(analysis_dir))

    # Get filenames of validation files
    validation_filenames = [f for f in os.listdir(val_dir) if f.endswith(".xlsx")]

    # Loop through each validation file
    for file in validation_filenames:

        # Extract file name without extension
        file_name = os.path.splitext(file)[0]

        # Set analysis directory path
        current_analysis_dir = os.path.join(analysis_dir, file_name)

        # Initialize NewsAnalysis object and load analysis data
        na = NewsAnalysis(None, current_analysis_dir, analysis_data_name=analysis_data_name, load_previous=True)

        # Create VideoStatus object and add to Validation object
        video_status = valid.VideoStatus(file_name, analysis_time=na.time_taken_s)
        valid.add(video_status)

        # Define helper function to extract row data
        def getRowData(r):
            [time_start, time_end, name, name_shown, presenter] = r

            # Ensure empty string is None
            if name == "":
                name = None

            # Parse time stamps
            time_start_s = parseTimestamp3(time_start)
            time_end_s = parseTimestamp3(time_end)

            return time_start_s, time_end_s, name, name_shown, presenter

        # Read validation file into pandas dataframe
        v_path = os.path.join(val_dir, file)
        df = pd.read_excel(v_path, header=0)

        # Convert dataframe rows to tuples and store in rows list
        rows = [tuple(row) for row in df.values]

        def getTimeRow(r):
            time_start_s, time_end_s, name, name_shown, presenter = getRowData(r)
            return time_start_s, time_end_s

        intersections, non_intersections1, non_intersections2 = intersection_and_iou(rows, getTimeRow,
                                                                                     na.scenes, getTimeScene,
                                                                                     iou_threshold=iou_threshold)
        valid.num_annotated_scenes += len(rows)
        valid.num_predicted_scenes += len(na.scenes)

        # Intersection handling
        for row, scene, iou in intersections:
            time_start_s, time_end_s, name, name_shown, presenter = getRowData(row)

            # scene_time_start = scene.scene_timestamps[0].get_seconds()
            # scene_time_end = scene.scene_timestamps[1].get_seconds()

            scene_time_start, scene_time_end = getTimeScene(scene)

            if scene.face_info is None:  # Intersection with scene, but no face found
                scene_status = video_status.SceneStatus(
                    actual_name=name,
                    actual_timestamp=[scene_time_start, scene_time_end],
                    detection_missing=True,
                    # iou=iou
                )
                video_status.add(scene_status)
            else:  # Intersection with scene with face found
                scene_status = video_status.SceneStatus(
                    actual_name=name,
                    actual_timestamp=[time_start_s, time_end_s],
                    predicted_name=scene.face_info.name,
                    predicted_time_stamp=[scene_time_start, scene_time_end],
                    iou=iou
                )
                video_status.add(scene_status)

        # Non matches for actual
        for row in non_intersections1:  # Missing intersections with actual data
            time_start_s, time_end_s, name, name_shown, presenter = getRowData(row)

            scene_status = video_status.SceneStatus(
                actual_name=name,
                actual_timestamp=[time_start_s, time_end_s],
                detection_missing=True
            )
            video_status.add(scene_status)

        # Non matches for prediction
        for scene in non_intersections2:  # Missing intersections with predicted data
            # matched_item_time_stamp = [scene.scene_timestamps[0].get_seconds(),
            #                            scene.scene_timestamps[1].get_seconds()]
            matched_item_time_stamp = list(getTimeScene(scene))

            if scene.face_info is not None:  # If face data found, mark as extra
                scene_status = video_status.SceneStatus(
                    predicted_name=scene.face_info.name,
                    predicted_time_stamp=matched_item_time_stamp,
                    detection_extra=True
                )
                video_status.add(scene_status)

    valid.save()
    valid.getPlot(show=show)


def mainValidate():
    results_dir = "data/comparing_segments_results"
    video_dir = "data/NewsSegments"
    output_analysis_dir = "data/comparing_segments/skip_seconds_1_2"
    validation_dir = "data/Validation2"

    def getTimeScene(s: NewsAnalysis.Scene):
        return s.scene_timestamps[0].get_seconds(), s.scene_timestamps[1].get_seconds()

    validateVideos(results_dir, output_analysis_dir, validation_dir, getTimeScene=getTimeScene, show=False)


# def test():
#     results_dir = "data/comparing_segments_results"
#     video_dir = "data/NewsSegments"
#
#     validation_dir = "data/Validation2"
#     analysis_dir = "data/comparing_segments/skip_seconds_1_2"
#
#     news_segments = iterNewsSegments(validation_dir, analysis_dir)
#     cm_ocr_names(news_segments)


if __name__ == "__main__":
    mainValidate()
    # test()
