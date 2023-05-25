import csv
import os
import pickle
import json
from enum import Enum
from Levenshtein import distance

from utils import parseTimestamp
from video_analysis_1 import NewsAnalysis
import plots

"""
Sample for validation 2
"""


def analyseVideos(directory, output_directory, filter_names: list[str] = None, validation_dir_filter: str = None,
                  **kwargs):
    os.makedirs(output_directory, exist_ok=True)

    directory_files = [f for f in os.listdir(directory) if f.endswith(".mp4")]
    if filter_names is None:
        if validation_dir_filter is None:
            files_to_analyse = list(directory_files)
        else:
            names = [os.path.splitext(f)[0] for f in os.listdir(validation_dir_filter)]
            files_to_analyse = [f for f in directory_files if os.path.splitext(f)[0] in names]
    else:
        files_to_analyse = [f for f in directory_files if os.path.splitext(f)[0] in filter_names]

    print("FILES: ", files_to_analyse)

    for video_name in files_to_analyse:
        video_path = os.path.join(directory, video_name)
        output_folder_analysis = os.path.join(output_directory, os.path.splitext(video_name)[0])

        if os.path.exists(output_folder_analysis):
            print(f"'{output_folder_analysis}' already exists, skipping.")
            continue

        na = NewsAnalysis(video_path, output_folder_analysis, overwrite_base_folder=True, **kwargs)
        na.analyseVideo()


def calculate_iou(start1, end1, start2, end2):
    intersection = max(0, min(end1, end2) - max(start1, start2))
    union = max(end1, end2) - min(start1, start2)
    return intersection / union


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
                self.name_distance = 0  # The name distance if both names are not None
                self.iou = iou

                self.actual_timestamp = actual_timestamp
                self.predicted_time_stamp = predicted_time_stamp

                # Time IoU
                # intersection = min(actual_timestamp[1], predicted_time_stamp[1]) - max(actual_timestamp[0], predicted_time_stamp[0])
                # union = max(actual_timestamp[1], predicted_time_stamp[1]) - min(actual_timestamp[0], predicted_time_stamp[0])
                # self.iou = intersection / union

                if detection_missing:
                    self.flag = self.flag.DetectionMissing
                elif detection_extra:
                    self.flag = self.flag.DetectionMissing
                else:
                    if actual_name is None:
                        if predicted_name is None:
                            self.flag = self.flag.MatchNone
                        else:
                            self.flag = self.flag.Extra
                    else:
                        if predicted_name is None:
                            self.flag = self.flag.Missing
                        else:
                            if actual_name.upper() == predicted_name.upper():
                                self.flag = self.flag.Match
                            else:
                                self.flag = self.flag.Mismatch

                            self.name_distance = distance(actual_name.upper(), predicted_name.upper())

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

        def __init__(self, name):
            self.name = name

            self.scene_statuses = list()

        def add(self, scene_status: SceneStatus):
            self.scene_statuses.append(scene_status)

        def toDict(self):
            return {
                "name": self.name,
                "scene_statuses": [s.toDict() for s in self.scene_statuses],
            }

    """
    Metrics: IOU (time), Confusion matrix, OCR accuracy
    """

    def __init__(self, directory):
        # os.makedirs(directory, exist_ok=True)

        self.directory = directory

        self.video_statuses = list()

    def save(self):
        save_pickle_path = os.path.join(self.directory, "results.pkl")
        with open(save_pickle_path, "wb") as f:
            pickle.dump(self, f)

        save_json_path = os.path.join(self.directory, "results.json")
        with open(save_json_path, "w") as f:
            json.dump(self.toDict(), f, indent=4)

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

    def genStats(self, path=None):
        if path is None:
            path = os.path.join(self.directory, "General_results.json")

        self.getCM()

        results_dict = {
            "Average IoU": self.getAverageIOU(),
            "Average name distance": self.getAverageNameDistance(),
        }

        with open(path, "w") as f:
            json.dump(results_dict, f, indent=4)

    def getAverageIOU(self):
        iou_sum = 0
        iou_count = 0

        for x in self:
            iou_sum += x.iou
            iou_count += 1

        iou_avg = iou_sum / iou_count
        return iou_avg

    def getAverageNameDistance(self):
        s = 0
        c = 0

        for x in self:
            s += x.name_distance
            c += 1

        avg = s / c
        return avg

    def getCM(self, graph_path=None):
        if graph_path is None:
            graph_path = os.path.join(self.directory, "Confusion Matrix.png")
        cm = plots.ConfusionMatrix("Confusion Matrix", save_location=graph_path)

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

        cm.genPlot()


def validateVideos(res_dir, analysis_dir, val_dir, analysis_data_name="analysis_data", iou_threshold=0.0, keep_detection_missing=False):
    valid = Validation(res_dir)

    # # KEYS
    # TIME_START_KEY = "Time start"
    # TIME_END_KEY = "Time end"
    # NAME_KEY = "Name"
    # NAME_SHOWN_KEY = "Is name shown?"
    # PRESENTER_KEY = "Is presenter?"

    validation_csv_filenames = [f for f in os.listdir(val_dir) if f.endswith(".csv")]
    for csv_file in validation_csv_filenames:
        # Load analysis
        file_name = os.path.splitext(csv_file)[0]
        current_analysis_dir = os.path.join(analysis_dir, file_name)
        na = NewsAnalysis(None, current_analysis_dir, analysis_data_name=analysis_data_name,
                          load_previous=True)  # Load analysis data

        video_status = valid.VideoStatus(file_name)
        valid.add(video_status)

        def getRowData(r):
            [time_start, time_end, name, name_shown, presenter] = r

            if name == "":  # Ensuring empty string is None
                name = None

            # Parsing time stamps
            time_start_s = parseTimestamp(time_start)
            time_end_s = parseTimestamp(time_end)

            return time_start_s, time_end_s, name, name_shown, presenter

        csv_path = os.path.join(val_dir, csv_file)
        with open(csv_path, newline='') as csvfile:
            reader = csv.reader(csvfile)
            rows = [tuple(row) for row in reader][1:]
            # [time_start, time_end, name, name_shown, presenter] = row

        unmatched_intervals_a = set(rows)
        unmatched_intervals_b = set(na.scenes)
        for interval_b in na.scenes:
            scene_time_start = interval_b.scene_timestamps[0].get_seconds()
            scene_time_end = interval_b.scene_timestamps[1].get_seconds()

            overlaps = []
            for interval_a in rows:
                time_start_s, time_end_s, name, name_shown, presenter = getRowData(interval_a)

                iou = calculate_iou(time_start_s, time_end_s, scene_time_start, scene_time_end)
                overlaps.append((interval_a, iou))
            overlaps = sorted(overlaps, key=lambda x: x[1], reverse=True)

            if overlaps:
                closest_overlap, iou = overlaps[0]
                if iou > iou_threshold:
                    time_start_s, time_end_s, name, name_shown, presenter = getRowData(closest_overlap)

                    if interval_b.face_info is None:
                        if keep_detection_missing:
                            scene_status = video_status.SceneStatus(
                                predicted_time_stamp=[scene_time_start, scene_time_end],
                                detection_missing=True
                            )
                            video_status.add(scene_status)
                    else:
                        scene_status = video_status.SceneStatus(
                            actual_name=name,
                            actual_timestamp=[time_start_s, time_end_s],
                            predicted_name=interval_b.face_info.name,
                            predicted_time_stamp=[scene_time_start, scene_time_end],
                            iou=iou
                        )
                        video_status.add(scene_status)

                    # Remove for sets
                    unmatched_intervals_a.discard(closest_overlap)
                    unmatched_intervals_b.discard(interval_b)

        for interval_a in unmatched_intervals_a:
            print(interval_a)
            time_start_s, time_end_s, name, name_shown, presenter = getRowData(interval_a)

            scene_status = video_status.SceneStatus(
                actual_name=name,
                actual_timestamp=[time_start_s, time_end_s],
                detection_missing=True
            )
            video_status.add(scene_status)

        if keep_detection_missing:
            for interval_b in unmatched_intervals_b:
                matched_item_time_stamp = [interval_b.scene_timestamps[0].get_seconds(),
                                           interval_b.scene_timestamps[1].get_seconds()]

                if interval_b.face_info is None:
                    scene_status = video_status.SceneStatus(
                        predicted_time_stamp=matched_item_time_stamp,
                        detection_missing=True
                    )
                    video_status.add(scene_status)
                else:
                    scene_status = video_status.SceneStatus(
                        predicted_name=interval_b.face_info.name,
                        predicted_time_stamp=matched_item_time_stamp,
                        detection_missing=True
                    )
                    video_status.add(scene_status)

    valid.save()
    valid.genStats()


if __name__ == "__main__":
    results_dir = "data/comparing/NewsSegments Sample/results"
    video_dir = "data/NewsSegments"
    output_analysis_dir = "data/comparing/NewsSegments Sample/analysis"
    validation_dir = "data/Validation2"

    analyseVideos(video_dir, output_analysis_dir, validation_dir_filter=validation_dir, skip_seconds_check=2,
                  resolution=(640, 360))
    validateVideos(results_dir, output_analysis_dir, validation_dir)
