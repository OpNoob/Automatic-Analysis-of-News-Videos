import copy
import json
import os
from datetime import datetime, date

import pandas as pd
import utils


class Validation:
    def __init__(self, directory="data/Validation2"):
        self.directory = directory

        self.files = list()
        self.video_annotations = list()

        # Calculated
        self.video_dates = list()
        self.num_annotations = 0
        self.named_people = list()
        self.num_named_named_annotations = 0
        self.num_unknown_named_annotations = 0
        self.num_presenters = 0
        self.num_name_shown = 0

        self.read()

    def toDict(self):
        ret_dict = {
            "directory": self.directory,

            "Num Video Segments": len(self),
            "Num Annotations": self.num_annotations,
            "Num Video Dates": len(self.video_dates),
            "Num Unique People Annotations": len(self.named_people),
            "Num Named People Annotations": self.num_named_named_annotations,
            "Num Unknown People Annotations": self.num_unknown_named_annotations,
            "Num Presenters": self.num_presenters,
            "Num Name Shown": self.num_name_shown,

            "Named People": list(sorted(self.named_people)),

            "Video Dates": [x.strftime('%d.%m.%Y') for x in self.video_dates],
            "video_annotations": [va.toDict() for va in self.video_annotations],

            "files": self.files,
        }

        return ret_dict

    def save(self, save_path):
        with open(save_path, "w") as f:
            json.dump(self.toDict(), f, indent=4)

    def read(self):
        self.files = [file for file in os.listdir(self.directory) if file.endswith(".xlsx")]

        for file in self.files:
            file_path = os.path.join(self.directory, file)
            video_name = os.path.splitext(file)[0]
            video = self.Video(video_name, file_path=file_path)
            self.video_annotations.append(video)

        # Calculations
        self.video_dates = self.getUniqueDates()

        annotations = self.getAllAnnotations()
        self.num_annotations = len(annotations)

        self.named_people = self.getNamedPeople(annotations)
        self.num_named_named_annotations = self.getNumNamedAnnotations(annotations)
        self.num_unknown_named_annotations = self.getNumNamedAnnotations(annotations, nones=True)
        self.num_presenters = self.getNumPresenters(annotations)
        self.num_name_shown = self.getNumNameShown(annotations)

    def __len__(self):
        return len(self.video_annotations)

    def __iter__(self):
        for x in self.video_annotations:
            yield x

    def __getitem__(self, item):
        return self.video_annotations[item]

    @staticmethod
    def parseTimestamps(time_stamp_str):
        """
        Returns the timestamp in seconds format ()

        :param time_stamp_str: Timestamp to parse
        :return:
        """

        hour, minute, second, frames = time_stamp_str.split(":")

        hour = int(hour)
        minute = int(minute)
        second = int(second)
        time_stamp_s = hour * 60 * 60 + minute * 60 + second
        return time_stamp_s

    def getDates(self):
        return [x.getDate() for x in self]

    def getUniqueDates(self, sort=True):
        dates_set = {x.getDate() for x in self}
        if sort:
            return sorted(dates_set)
        else:
            return sort

    def getAllAnnotations(self):
        return [annotation for video in self for annotation in video]

    @staticmethod
    def getNamedPeople(annotations):
        return {a.name for a in annotations if a.name is not None}

    @staticmethod
    def getNumNamedAnnotations(annotations, nones=False):
        c = 0
        for a in annotations:
            if (nones and a.name is None) or (not nones and a.name is not None):
                c += 1
        return c

    @staticmethod
    def getNumPresenters(annotations):
        c = 0
        for a in annotations:
            if a.is_a_presenter:
                c += 1
        return c

    @staticmethod
    def getNumNameShown(annotations):
        c = 0
        for a in annotations:
            if a.is_name_shown:
                c += 1
        return c

    def getAnnotationsFromDate(self, d: datetime.date):
        return [a for a in self.getAllAnnotations() if a.date == d]  # not tested

    class Video:
        def __init__(self, name, file_path=None):
            self.name = name
            self.annotations = list()

            if file_path is not None:
                if os.path.exists(file_path):
                    self.read(file_path)
                else:
                    print(f"File path '{file_path}' does not exist")

        def toDict(self):
            ret_dict = {
                "name": self.name,
                "num_annotations": len(self),
                "annotations": [a.toDict() for a in self.annotations],
            }
            return ret_dict

        def read(self, file_path):
            df = pd.read_excel(file_path, header=0)
            for row in df.values:
                annotation = self.Annotation(*row)
                self.annotations.append(annotation)

        def __len__(self):
            return len(self.annotations)

        def __iter__(self):
            for x in self.annotations:
                yield x

        def __getitem__(self, item):
            return self.annotations[item]

        def getDate(self, raw=False):
            date_str = self.name.split(" segment ")[0]
            if raw:
                return date_str
            else:
                return datetime.strptime(date_str, '%d.%m.%Y').date()

        class Annotation:
            def __init__(self, time_start, time_end, name, is_name_shown, is_a_presenter):
                if isinstance(name, float) and pd.isna(name) or name.isspace():
                    name = None

                # Parse time stamps
                time_start = Validation.parseTimestamps(time_start)
                time_end = Validation.parseTimestamps(time_end)

                self.time_start = time_start
                self.time_end = time_end
                self.name = name
                self.is_name_shown = is_name_shown
                self.is_a_presenter = is_a_presenter

            def toDict(self):
                ret_dict = copy.deepcopy(self.__dict__)
                ret_dict["time_start"] = utils.secondsStrMini(self.time_start)
                ret_dict["time_end"] = utils.secondsStrMini(self.time_end)
                return ret_dict

            def duration(self):
                return self.time_end - self.time_start


def main():
    v = Validation()
    v.save("data/comparing_segments_results_2/validation_statistics.json")
    # print(v)


if __name__ == "__main__":
    main()
