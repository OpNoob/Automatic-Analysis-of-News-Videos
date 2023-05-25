import copy
import csv
import os
import pickle
import json
from enum import Enum

from Levenshtein import distance
import pandas as pd
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, mean_absolute_error
from tqdm import tqdm

import utils
from utils import parseTimestamp3, getRowData
from video_analysis_2 import NewsAnalysis, FaceInfoDatabase
import plots

"""
Validation from annotated videos.  Version 2.0
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


def getTimeRow(r):
    time_start_s, time_end_s, name, name_shown, presenter = getRowData(r)
    return time_start_s, time_end_s


def getTimeScene_1(s: NewsAnalysis.Scene):
    return s.scene_timestamps[0].get_seconds(), s.scene_timestamps[1].get_seconds()


def getTimeScene_2(s: NewsAnalysis.Scene):
    return s.time_start_s, s.time_end_s


def iterNewsSegments(val_dir, analysis_dir, analysis_data_name="analysis_data", return_filename=True,
                     show_updates=True):
    # Get filenames of validation files
    validation_filenames = [f for f in os.listdir(val_dir) if f.endswith(".xlsx")]

    if show_updates:
        pbar = tqdm(total=len(validation_filenames), desc=f"Getting results for '{analysis_dir}'")

    # Loop through each validation file
    for file in validation_filenames:
        # Extract file name without extension
        filename = os.path.splitext(file)[0]

        # Set analysis directory path
        current_analysis_dir = os.path.join(analysis_dir, filename)

        # Initialize NewsAnalysis object and load analysis data
        na = NewsAnalysis(None, current_analysis_dir, analysis_data_name=analysis_data_name, load_previous=True)

        # Read validation file into pandas dataframe
        v_path = os.path.join(val_dir, file)
        df = pd.read_excel(v_path, header=0)

        # Convert dataframe rows to tuples and store in rows list
        rows = [tuple(row) for row in df.values]

        if return_filename:
            yield rows, na, filename
        else:
            yield rows, na

        if show_updates:
            pbar.update()
    if show_updates:
        pbar.close()


def get_video_data(rows: list[tuple], na: NewsAnalysis):
    # Check if name is empty
    def nameEmpty(name: str):
        if name is None or isinstance(name, float) and pd.isna(name):
            return True
        return False

    # Clean nones
    def cleanNames(names: list):
        names = [name if not nameEmpty(name) else None for name in names]
        return names

    # row -> time_start_s, time_end_s, name, name_shown, presenter

    valid_ocr_names = list()
    analysis_ocr_names = list()

    valid_time_stamps = dict()
    analysis_time_stamps = dict()

    valid_durations = dict()
    analysis_durations = dict()

    for row in rows:
        time_start_s, time_end_s, name, name_shown, presenter = getRowData(row)

        # TIMESTAMP
        if not nameEmpty(name):
            if name in valid_time_stamps:
                valid_time_stamps[name].append((time_start_s, time_end_s))
            else:
                valid_time_stamps[name] = [(time_start_s, time_end_s)]

            duration = time_end_s - time_start_s
            if name in valid_durations:
                valid_durations[name] += duration
            else:
                valid_durations[name] = duration

        # OCR
        if name_shown:
            valid_ocr_names.append(name)

    for scene in na.scenes:

        # TIMESTAMP
        if scene.face_info is not None:
            name = scene.face_info.name
            if not nameEmpty(name):
                if hasattr(scene, "time_start_s") and hasattr(scene, "time_end_s"):
                    time_start_s, time_end_s = scene.time_start_s, scene.time_end_s
                elif hasattr(scene, "scene_timestamps") and hasattr(scene, "scene_timestamps"):
                    time_start_s, time_end_s = scene.scene_timestamps[0].get_seconds(), scene.scene_timestamps[
                        1].get_seconds()
                else:
                    print("ERROR IN GETTING TIME", scene.__dict__.keys())
                    time_start_s = 0
                    time_end_s = 0

                if name in analysis_time_stamps:
                    analysis_time_stamps[name].append((time_start_s, time_end_s))
                else:
                    analysis_time_stamps[name] = [(time_start_s, time_end_s)]

                duration = time_end_s - time_start_s
                if name in analysis_durations:
                    analysis_durations[name] += duration
                else:
                    analysis_durations[name] = duration

        # OCR
        analysis_ocr_names.append(scene.getName())

    # valid_names = [row[2] for row in rows]
    # analysis_names = [scene.getName() for scene in na.scenes]

    valid_ocr_names = cleanNames(valid_ocr_names)
    analysis_ocr_names = cleanNames(analysis_ocr_names)

    return valid_ocr_names, analysis_ocr_names, valid_durations, analysis_durations, valid_time_stamps, analysis_time_stamps


class Results:
    def __init__(self, analysis_folder, directory="data/comparing_segments_results_2",
                 get_time_row=None, get_time_scene=None,
                 ):
        os.makedirs(directory, exist_ok=True)

        self.name = analysis_folder
        self.directory = directory

        self.videos = list()

        if get_time_row is None:
            self.get_time_row = getTimeRow
        if get_time_scene is None:
            self.get_time_scene = getTimeScene_2

        # Confusion matrix
        # true positives (TP): correctly identified persons
        # false positives (FP): wrongly identified persons
        # false negatives (FN): missed persons
        self.cm_ocr = Results.ConfusionMatrixData()
        self.cm_ana = Results.ConfusionMatrixData()

        # mae for duration
        self.mae_duration = 0

        self.analysis_time = 0
        self.frame_count = 0
        self.analysis_fps = 0

    def save(self):
        filename = "results" if self.name is None else f"{self.name} Results"

        save_pickle_path = os.path.join(self.directory, f"{filename}.pkl")
        with open(save_pickle_path, "wb") as f:
            pickle.dump(self, f)

        save_json_path = os.path.join(self.directory, f"{filename}.json")
        with open(save_json_path, "w") as f:
            json.dump(self.toDict(), f, indent=4)

        # Graphs
        self.ConfusionMatrix()

    def load(self, path):
        with open(path, "rb") as f:
            self.__dict__ = pickle.load(f).__dict__
        return self

    def toDict(self):
        return {
            "name": self.name,
            "videos": [x.toDict() for x in self.videos],
        }

    def getResults(self, validation_dir="data/Validation2", analysis_dir="data/comparing_segments", ):

        analysis_dir_path = os.path.join(analysis_dir, self.name)

        # Iterate over videos
        news_segments = iterNewsSegments(validation_dir, analysis_dir_path, show_updates=True)
        for rows, na, filename in news_segments:
            # Add video data
            video = self.Video(filename)
            video.getResults(rows, na)

            # Add to list
            self.videos.append(video)

        # Calculate ocr values
        self.cm_ocr = self.getTotalCMData('cm_ocr')
        self.cm_ocr.calc()

        self.cm_ana = self.getTotalCMData('cm_ana')
        self.cm_ana.calc()

        # Calculate mae
        valid_durations = self.getTotalCounter("valid_durations")
        analysis_durations = self.getTotalCounter("analysis_durations")

        self.mae_duration = self.getMAE(valid_durations, analysis_durations)

        self.analysis_time = self.getTotal("analysis_time")
        self.frame_count = self.getTotal("frame_count")
        self.analysis_fps = self.frame_count / self.analysis_time

        return self

    def getValues(self, var_name):
        values = [getattr(video, var_name) for video in self.videos if hasattr(video, var_name)]

        if not values:
            print(f"Variable '{var_name}' not found in any video stats")
            return None

        return values

    def getAverage(self, var_name):
        values = self.getValues(var_name)

        avg = sum(values) / len(values)
        return avg

    def getTotal(self, var_name):
        values = self.getValues(var_name)

        s = sum(values)
        return s

    def getTotalCMData(self, var_name):
        values = self.getValues(var_name)

        return self.getTotalCM(values)

    @staticmethod
    def getTotalCM(cm_list: list):
        cm_return = Results.ConfusionMatrixData()

        for cm in cm_list:
            cm_return.tp += cm.tp
            cm_return.tn += cm.tn
            cm_return.fp += cm.fp
            cm_return.fn += cm.fn

        return cm_return

    def getTotalCounter(self, var_name):
        values = self.getValues(var_name)

        return self.getTotalCounterList(values)

    @staticmethod
    def getTotalCounterList(counter_list):

        return_dict = dict()

        for d in counter_list:
            for key, item in d.items():
                if key in return_dict:
                    return_dict[key] += item
                else:
                    return_dict[key] = item

        return return_dict

    @staticmethod
    def getMAE(valid_durations: dict, analysis_durations: dict):
        valid_names = set(valid_durations.keys())
        analysis_names = set(analysis_durations.keys())
        common_names = valid_names.intersection(analysis_names)

        v_durations = list()
        a_durations = list()
        for name in common_names:
            v_duration = valid_durations[name]
            a_duration = analysis_durations[name]

            v_durations.append(v_duration)
            a_durations.append(a_duration)

        if v_durations and a_durations:
            mae_duration = mean_absolute_error(v_durations, a_durations)
        else:
            mae_duration = 0
        return mae_duration

    def ConfusionMatrix(self, name="Output Results", show=False):
        graph_name = f"{self.name} {name}"
        save_path = os.path.join(self.directory, f"{graph_name}.png")

        # plot
        width_ratios = [10, 10, 4]
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3,
                                            figsize=(20, 6),
                                            gridspec_kw={'width_ratios': width_ratios},
                                            )

        def create_cm(ax, sub_name, cm_data: Results.ConfusionMatrixData):
            # create the confusion matrix
            cm = np.array([[cm_data.tp, cm_data.fp], [cm_data.fn, cm_data.tn]])

            im = ax.imshow(cm, cmap=plt.cm.Blues)

            # add labels, title, and color bar
            ax.set_xticks([0, 1])
            ax.set_yticks([0, 1])
            ax.set_xticklabels(['Positive', 'Negative'])
            ax.set_yticklabels(['Positive', 'Negative'])
            ax.set_xlabel('Predicted Label')
            ax.set_ylabel('True Label')
            ax.set_title(
                f"{sub_name}\n Precision: {cm_data.precision:.2f}, Recall: {cm_data.recall:.2f}, F1-Score: {cm_data.f1_score:.2f}, Accuracy: {cm_data.accuracy:.2f}")
            fig.colorbar(im)

            # add text annotations to each cell
            for i in range(2):
                for j in range(2):
                    ax.text(j, i, str(cm[i][j]), ha='center', va='center')

        create_cm(ax1, "OCR Names", self.cm_ocr)
        create_cm(ax2, "Analysis", self.cm_ana)

        # Add a general information subplot
        text = f"Mean Absolute Error (MAE): {self.mae_duration:.2f}\n" \
               f"Time Taken (mins) {self.analysis_time / (60 * 60):.2f}\n" \
               f"Analysis FPS {self.analysis_fps:.2f} hrs"
        ax3.set_title("More Metrics")
        ax3.text(0.05, 0.95, text, fontsize=7, ha='left', va='top')

        fig.suptitle(graph_name)

        # save the plot as an image file
        fig.savefig(save_path)

        if show:
            # show the plot
            plt.show()

    @staticmethod
    def getConfusionMatrixValues(valid_names, analysis_names):
        # Filtering Nones
        if None in valid_names:
            valid_names.remove(None)
        if None in analysis_names:
            analysis_names.remove(None)

        cm = Results.ConfusionMatrixData()

        # Calculations
        cm.tp = len(valid_names.intersection(analysis_names))
        cm.fp = len(analysis_names) - cm.tp
        cm.fn = len(valid_names) - cm.tp
        cm.tn = 0

        cm.calc()

        return cm

    @staticmethod
    def getPRFA(tp, fp, fn, tn):
        if tp + fp == 0:
            precision = 0
        else:
            precision = tp / (tp + fp)

        if tp + fn == 0:
            recall = 0
        else:
            recall = tp / (tp + fn)

        if precision + recall == 0:
            f1_score = 0
        else:
            f1_score = 2 * (precision * recall) / (precision + recall)

        if tp + fn == 0:
            accuracy = 0
        else:
            accuracy = (tp + tn) / (tp + fp + fn + tn)

        return precision, recall, f1_score, accuracy

    def getDatabaseChart(self, name="Database Output Results", show=False, title=True):
        graph_name = f"{self.name} {name}"
        save_path = os.path.join(self.directory, f"{graph_name}.png")

        # Get values for database
        database_len_list = self.getValues('len_database')

        # Get values
        cm_ocr_list = self.getValues('cm_ocr')
        cm_ana_list = self.getValues('cm_ana')
        time_list = self.getValues('analysis_time')
        # mae_list = self.getValues('mae_duration')

        # mae
        valid_durations_list = self.getValues("valid_durations")
        analysis_durations_list = self.getValues("analysis_durations")

        # Zip the lists together
        zipped = zip(database_len_list, cm_ocr_list, cm_ana_list, time_list, valid_durations_list,
                     analysis_durations_list)

        # Sort the zipped list based on the first element of each tuple (which is list1)
        sorted_zipped = sorted(zipped, key=lambda x: x[0])

        # Unzip the sorted list back into the five separate lists
        database_len_list, cm_ocr_list, cm_ana_list, time_list, valid_durations_list, analysis_durations_list = zip(
            *sorted_zipped)

        # num_database = list()
        num_database = database_len_list
        precision_ocr = list()
        recall_ocr = list()
        f1_score_ocr = list()
        accuracy_ocr = list()
        precision_ana = list()
        recall_ana = list()
        f1_score_ana = list()
        accuracy_ana = list()
        time = list()
        mae = list()

        # Calculate cumulative values
        for i in range(len(database_len_list)):
            cm_ocr_current = self.getTotalCM(cm_ocr_list[:i])
            cm_ocr_current.calc()

            cm_ana_current = self.getTotalCM(cm_ana_list[:i])
            cm_ana_current.calc()

            time_current = sum(time_list[:i]) / (60 * 60)

            # mae
            valid_durations_current = Results.getTotalCounterList(valid_durations_list[:i])
            analysis_durations_current = Results.getTotalCounterList(analysis_durations_list[:i])
            mae_current = Results.getMAE(valid_durations_current, analysis_durations_current)

            precision_ocr.append(cm_ocr_current.precision)
            recall_ocr.append(cm_ocr_current.recall)
            f1_score_ocr.append(cm_ocr_current.f1_score)
            accuracy_ocr.append(cm_ocr_current.accuracy)

            precision_ana.append(cm_ana_current.precision)
            recall_ana.append(cm_ana_current.recall)
            f1_score_ana.append(cm_ana_current.f1_score)
            accuracy_ana.append(cm_ana_current.accuracy)

            time.append(time_current)
            mae.append(mae_current)

        # create the first plot for precision, recall, f1 score, and accuracy
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))  #
        ax = ax.flatten()

        ax[0].plot(num_database, precision_ocr, label='Precision')
        ax[0].plot(num_database, recall_ocr, label='Recall')
        ax[0].plot(num_database, f1_score_ocr, label='F1 Score')
        ax[0].plot(num_database, accuracy_ocr, label='Accuracy')
        ax[0].set_xlabel('Number of People in database')
        ax[0].set_ylabel('Score')
        ax[0].set_title('Name Extraction')
        ax[0].legend()

        ax[1].plot(num_database, precision_ana, label='Precision')
        ax[1].plot(num_database, recall_ana, label='Recall')
        ax[1].plot(num_database, f1_score_ana, label='F1 Score')
        ax[1].plot(num_database, accuracy_ana, label='Accuracy')
        ax[1].set_xlabel('Number of People in database')
        ax[1].set_ylabel('Score')
        ax[1].set_title('Face Recognition')
        ax[1].legend()

        ax[2].plot(num_database, time)
        ax[2].set_xlabel('Number of People in database')
        ax[2].set_ylabel('Time (hrs)')
        ax[2].set_title('Time')

        ax[3].plot(num_database, mae)
        ax[3].set_xlabel('Number of People in database')
        ax[3].set_ylabel('Time (secs)')
        ax[3].set_title('Average MAE for duration of individuals')

        plt.subplots_adjust(left=0.1, right=0.9, bottom=0.1, top=0.9, wspace=0.3, hspace=0.5)

        if title:
            fig.suptitle(graph_name)

        # save the plot as an image file
        fig.savefig(save_path)

        if show:
            # show the plot
            plt.show()

    class Video:
        def __init__(self, name=""):
            self.name = name

            """
            Video results after full analysis
            """
            # Analysis
            self.valid_names = set()
            self.analysis_names = set()

            # OCR
            self.valid_ocr_names = set()
            self.analysis_ocr_names = set()

            self.valid_durations = dict()
            self.analysis_durations = dict()

            # Timestamps
            self.valid_time_stamps = dict()
            self.analysis_time_stamps = dict()

            # Confusion matrix
            # true positives (TP): correctly identified persons
            # false positives (FP): wrongly identified persons
            # false negatives (FN): missed persons
            self.cm_ocr = Results.ConfusionMatrixData()
            self.cm_ana = Results.ConfusionMatrixData()

            self.mae_duration = 0

            self.analysis_time = 0
            self.frame_count = 0
            self.analysis_fps = 0

            self.len_database = 0

        def getResults(self, rows: list[tuple], na: NewsAnalysis):
            valid_ocr_names, analysis_ocr_names, valid_durations, analysis_durations, valid_time_stamps, analysis_time_stamps = get_video_data(
                rows, na)

            self.valid_ocr_names, self.analysis_ocr_names = set(valid_ocr_names), set(analysis_ocr_names)
            self.valid_names, self.analysis_names = set(valid_time_stamps.keys()), set(analysis_time_stamps.keys())

            self.valid_durations = valid_durations
            self.analysis_durations = analysis_durations

            self.valid_time_stamps = valid_time_stamps
            self.analysis_time_stamps = analysis_time_stamps

            self.cm_ocr = Results.getConfusionMatrixValues(self.valid_ocr_names, self.analysis_ocr_names)
            self.cm_ana = Results.getConfusionMatrixValues(self.valid_names, self.analysis_names)

            self.mae_duration = Results.getMAE(valid_durations, analysis_durations)

            self.analysis_time = na.time_taken_s
            self.frame_count = na.frame_count
            self.analysis_fps = self.frame_count / self.analysis_time

            self.len_database = len(na.face_info_db.data)  # len(na.face_info_db)

        def toDict(self):
            ret_dict = copy.deepcopy(self.__dict__)

            ret_dict["valid_names"] = list(self.valid_names)
            ret_dict["analysis_names"] = list(self.analysis_names)

            ret_dict["valid_ocr_names"] = list(self.valid_ocr_names)
            ret_dict["analysis_ocr_names"] = list(self.analysis_ocr_names)

            ret_dict["cm_ocr"] = self.cm_ocr.toDict()
            ret_dict["cm_ana"] = self.cm_ana.toDict()

            return ret_dict

    class ConfusionMatrixData:
        def __init__(self):
            self.tp = 0
            self.fp = 0
            self.fn = 0
            self.tn = 0

            self.precision = None
            self.recall = None
            self.f1_score = None
            self.accuracy = None

        def calc(self):
            if self.tp + self.fp == 0:
                self.precision = 0
            else:
                self.precision = self.tp / (self.tp + self.fp)

            if self.tp + self.fn == 0:
                self.recall = 0
            else:
                self.recall = self.tp / (self.tp + self.fn)

            if self.precision + self.recall == 0:
                self.f1_score = 0
            else:
                self.f1_score = 2 * (self.precision * self.recall) / (self.precision + self.recall)

            if self.tp + self.fn == 0:
                self.accuracy = 0
            else:
                self.accuracy = (self.tp + self.tn) / (self.tp + self.fp + self.fn + self.tn)

        def toDict(self):
            return copy.deepcopy(self.__dict__)


def test():
    validation_dir = "data/Validation2"
    analysis_dir = "data/comparing_segments"
    results_dir = "data/comparing_segments_results_2"

    results = Results("tracking_skip_0_5", directory=results_dir).getResults(validation_dir=validation_dir,
                                                                             analysis_dir=analysis_dir)
    print(results.toDict())
    results.save()


def compareResults():
    validation_dir = "data/Validation2"
    analysis_dir = "data/comparing_segments"
    results_dir = "data/comparing_segments_results_2"

    for analysis_folder in os.listdir(analysis_dir):
        results = Results(analysis_folder, directory=results_dir).getResults(validation_dir=validation_dir,
                                                                             analysis_dir=analysis_dir)
        # print(results.toDict())
        results.save()


def databaseChart(analysis_folder):
    validation_dir = "data/Validation2"
    analysis_dir = "data/comparing_segments"
    results_dir = "data/comparing_segments_results_2"

    results = Results(analysis_folder, directory=results_dir).getResults(validation_dir=validation_dir,
                                                                         analysis_dir=analysis_dir)
    results.save()
    # results = Results(analysis_folder, directory=results_dir).load(os.path.join(results_dir, "tracking_skip_2 Results.pkl"))
    results.getDatabaseChart(title=False)


if __name__ == "__main__":
    # main()
    compareResults() # normal statistics for all videos analysed
    # databaseChart("tracking_database_2")  # incremental database
