import os
import pickle
from collections import Counter
import json

import cv2
from utils import NEWS_VIDEOS_DIR, NEWS_VIDEO_SEGMENTS_DIR, getVideoNames, secondsStr


class VideoStatsDB:
    def __init__(self, save_dir="data/NewsStatistics", pickle_name="video_stats.pkl",
                 json_name="video_stats.json"):
        self._save_dir = save_dir
        self._pickle_name = pickle_name
        self._json_name = json_name

        self.videos = list()

        self.avg_fps = None
        self.avg_duration = None
        self.avg_frame_count = None
        self.counter_resolution = None
        self.counter_video_dates = None
        self.total_duration = None
        self.total_frame_count = None

    def toDict(self):
        return {
            "Number of videos": len(self.videos),
            "Number of News Video Dates": len(self.counter_video_dates),
            "Total Video Duration": secondsStr(self.total_duration),
            "Total Frame Count": self.total_frame_count,
            "Average Video Duration": secondsStr(self.avg_duration),
            "Average Frame Count": self.avg_frame_count,
            "Frequencies": {
                "Frames per Second (fps)": self.sortDictJSON(self.avg_fps),
                "Resolutions": self.sortDictJSON(self.counter_resolution),
                "Video Dates": self.sortDictJSON(self.counter_video_dates),
            },

            "videos": [x.toDict() for x in self.videos],
        }

    def calc(self):
        self.avg_fps = self.getCounter("fps")
        self.avg_duration = self.getAverage("duration")
        self.avg_frame_count = self.getAverage("frame_count")
        self.counter_resolution = self.getCounter("resolution")
        self.counter_video_dates = self.getCounter("video_date")
        self.total_duration = self.getTotal("duration")
        self.total_frame_count = self.getTotal("frame_count")

    def save(self):
        os.makedirs(self._save_dir, exist_ok=True)

        pickle_path = os.path.join(self._save_dir, self._pickle_name)
        json_path = os.path.join(self._save_dir, self._json_name)

        with open(pickle_path, "wb") as f:
            pickle.dump(self, f)

        with open(json_path, "w") as f:
            json.dump(self.toDict(), f, indent=4)

    def load(self, path):
        with open(path, "rb") as f:
            self.__dict__ = pickle.load(f)
        return self

    def add(self, video_paths: list or str, video_dir: str = None):
        if isinstance(video_paths, str):
            video_paths = [video_paths]
        elif isinstance(video_paths, list):
            video_paths = video_paths
        else:
            print(f"video_paths in incorrect format of type '{type(video_paths)}', expected list or str")

        if video_dir is not None:
            video_paths = [os.path.join(video_dir, video_path) for video_path in video_paths]

        for video_path in video_paths:
            vs = self.VideoStats(video_path)
            self.videos.append(vs)

        # Calculate averages and results at the end
        self.calc()

        # Saves
        # self.save()

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

    def getCounter(self, var_name):
        values = self.getValues(var_name)

        counter = Counter(values)
        return counter

    @staticmethod
    def sortDictJSON(counter: Counter):
        return {str(k): v for k, v in sorted(counter.items(), key=lambda item: item[1], reverse=True)}

    @staticmethod
    def parseDate(filename):
        return os.path.splitext(filename)[0].split(" segment")[0]

    @staticmethod
    def getVideoStats(video_path):
        video = cv2.VideoCapture(video_path)

        # Get video properties
        fps = video.get(cv2.CAP_PROP_FPS)
        resolution = (int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps

        return fps, resolution, frame_count, duration

    class VideoStats:
        def __init__(self, video_path):
            self.name = os.path.basename(video_path)
            self.video_date = VideoStatsDB.parseDate(self.name)
            self.fps, self.resolution, self.frame_count, self.duration = VideoStatsDB.getVideoStats(video_path)

        def toDict(self):
            return {
                "name": self.name,
                "fps": self.fps,
                "resolution": list(self.resolution),
                "frame_count": self.frame_count,
                "duration": self.duration,
            }


def main():
    output_dir = "data/VideoStats/"
    news_videos = getVideoNames(NEWS_VIDEOS_DIR)
    news_video_segments = getVideoNames(NEWS_VIDEO_SEGMENTS_DIR)

    video_stats = VideoStatsDB(save_dir=os.path.join(output_dir, f"{os.path.basename(NEWS_VIDEOS_DIR)}_stats"))
    video_stats.add(video_paths=news_videos, video_dir=NEWS_VIDEOS_DIR)
    print(video_stats.toDict())
    video_stats.save()

    video_stats_seg = VideoStatsDB(save_dir=os.path.join(output_dir, f"{os.path.basename(NEWS_VIDEO_SEGMENTS_DIR)}_stats"))
    video_stats_seg.add(video_paths=news_video_segments, video_dir=NEWS_VIDEO_SEGMENTS_DIR)
    print(video_stats.toDict())
    video_stats_seg.save()


if __name__ == "__main__":
    main()
