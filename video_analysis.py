import copy
import math
import os.path
import random
import shutil
import time
from datetime import datetime, timedelta
from enum import Enum
import cv2 as cv
import numpy as np
from tqdm import tqdm
import multiprocessing

import face_recognition_models
import face_recognition

import utils
from utils import *
import readText

"""

- Metric face encodings with face_recognition package
- Scenes with object tracking
- Comparing all scores of each scene to label a person (Using averages)
- Preprocessing image as well (gray scale and equalize histogram)
- Should check when matching, considering also the name found and give it priority
"""


def convert_bbox_to_cv2(bbox):
    top, right, bottom, left = bbox
    x = left
    y = top
    w = right - left
    h = bottom - top
    return x, y, w, h


def convert_bbox_to_face_recognition(bbox):
    x, y, w, h = bbox
    top = int(y)
    bottom = int(y + h)
    left = int(x)
    right = int(x + w)
    return top, right, bottom, left


def face_encoding(frame, bbox):
    found_face_encoding = face_recognition.face_encodings(frame, [bbox])[0]
    return found_face_encoding


class FaceInfo:
    """
    Facial information
    """

    def __init__(self, name: str, profile_image_data=None, profile_encoding=None, profile_face_image_data=None):
        """

        :param name: Name of person
        :param profile_image_data: Profile image
        :param profile_encoding: Profile face encoding
        :param profile_face_image_data: Profile image, but only keeping the face
        """
        self.name = name  # Name of person
        self.profile_image_data = profile_image_data  # The image showing this person
        self.profile_encoding = profile_encoding
        self.profile_face_image_data = profile_face_image_data

    def setProfileImage(self, image_data):
        self.profile_image_data = image_data

    def setProfileEncoding(self, encoding):
        self.profile_encoding = encoding

    def setProfile(self, name, encoding=None, image_data=None, face_image_data=None):
        self.name = name
        if encoding is not None:
            self.profile_encoding = encoding
        if image_data is not None:
            self.profile_image_data = image_data
        if face_image_data is not None:
            self.profile_face_image_data = face_image_data

    def toDict(self):
        return {
            "name": self.name,
        }


class FaceInfoDatabase:
    def __init__(self, database_path, face_info_list: list[FaceInfo] = None, load_available=True):
        self._database_path = database_path
        if load_available and self.load(show_warning=True):
            return

        if face_info_list is None:
            face_info_list = list()

        self.data = face_info_list

    def load(self, show_warning=True):
        if self._database_path is not None and os.path.exists(self._database_path):
            with open(self._database_path, "rb") as f:
                self.data = pickle.load(f).data
                return True
        if show_warning:
            print(f"Warning, database path {self._database_path} could not be found")
        return False

    def toDict(self):
        return {
            "amount": len(self.data),
            "data": {index: fi.toDict() for index, fi in enumerate(self.data)}
        }

    def save(self, save_json=True, save_images=True):
        # Pickle
        if self._database_path is not None:
            with open(self._database_path, "wb") as f:
                pickle.dump(self, f)

        # Json
        if save_json:
            json_path = os.path.splitext(self._database_path)[0] + ".json"
            with open(json_path, "w") as f:
                json.dump(self.toDict(), f, indent=4)

        # Save Images
        if save_images:
            self.saveImages()

    def saveImages(self, directory=None, duplicate_new=False, full_img=True):
        """
        Saving images of personal database

        :param directory:
        :param duplicate_new:
        :param full_img:  Keep the full frame.  If false only keep face image
        :return:
        """
        if directory is None:  # Setting directory if not set
            parent_directory = os.path.dirname(self._database_path)
            directory = os.path.join(parent_directory, "Images")

        if duplicate_new:  # Getting new folder is already exists
            directory_og = copy.deepcopy(directory)
            count = 0
            while os.path.exists(directory):
                directory = directory_og + f"({count}"
                count += 1

        os.makedirs(directory, exist_ok=True)  # Creating directory

        name_counter = dict()  # For handling duplicate names

        for fi in self.data:
            if full_img:
                image_data = fi.profile_image_data
            else:
                image_data = fi.profile_face_image_data
            name = fi.name
            if name is None:
                name = "Unknown"

            if name in name_counter:
                name_counter[name] += 1
                name = f"{name}({str(name_counter[name] - 1)})"
            else:
                name_counter[name] = 0

            image_path = os.path.join(directory, name) + ".png"

            cv.imwrite(image_path, image_data)
            # cv.imwrite(image_path, image_data)

    def __getitem__(self, index: int):
        return self.data[index]

    def __iter__(self):
        for fi in self.data:
            yield fi

    def __len__(self):
        return len(self.data)

    def add(self, face_info: FaceInfo):
        self.data.append(face_info)

    def getNames(self, face_info_list: list[FaceInfo] = None):
        if face_info_list is None:
            face_info_list = self.data

        return [x.name for x in face_info_list]

    def getProfileEncodings(self, face_info_list: list[FaceInfo] = None):
        if face_info_list is None:
            face_info_list = self.data

        return [x.profile_encoding for x in face_info_list]

    def inputFaceInfo(self, face_info_list: list[FaceInfo], duplicate_names=False):
        """
        Add new face_info data to this database

        :param face_info_list: The face info data to input to this database
        :param duplicate_names: If True, will allow people with duplicate names
        :return:
        """

        if duplicate_names:
            reserved_names = set()
        else:
            reserved_names = set(self.getNames())

        face_info_problems = list()

        for fi in face_info_list:
            # fi.name = fi.name.upper()  # Changing name to upper
            if fi.name in reserved_names:  # Problem
                face_info_problems.append(fi)
            else:  # Input into database
                self.data.append(fi)
        return face_info_problems

    def exportToDatabasePath(self, new_database_path, save=False):
        """
        Allows the export to another database

        :param new_database_path:
        :param save:
        :return:
        """
        export_database = FaceInfoDatabase(new_database_path,
                                           load_available=True)  # Create new instance with loaded data
        problems = export_database.inputFaceInfo(self.data)  # inputs data to imported database
        if save:
            export_database.save()
        return problems


def getMatchIndex(face_distances: list[np.ndarray], include_value: bool = True, maximum: bool = False):
    """
    Returns the closes match index of the face distances

    :param face_distances: the face distances
    :param include_value: Also returns the closes value as a tuple: (index, value)
    :param maximum: If True will match with the highest value instead the lowest
    :return: index or (index, value)
    """

    if len(face_distances) == 0:
        return

    avg = face_distances[0].copy()
    for x in face_distances[1:]:
        avg += x
    for i, num in enumerate(avg):
        avg[i] = num / len(face_distances)

    if len(avg) == 0:
        return

    if maximum:
        index = np.argmax(avg)
    else:
        index = np.argmin(avg)
    if include_value:
        return index, avg[index]
    return index


def detect_faces(image_bgr):
    rgb = cv.cvtColor(image_bgr, cv.COLOR_BGR2RGB)
    boxes = face_recognition.face_locations(rgb, model='hog')  # top, right, bottom, left

    return boxes


class NewsAnalysis:
    class Scene:
        """
            Class containing any information on the current Scene
        """

        class SceneInstance:
            """
            Holds the data of a single frame within a scene
            """

            def __init__(self, face_distances: list, face_encoding, name_found=None, image_data=None):
                """

                :param face_distances: The face distances
                :param face_encoding: The current face encoding
                :param name_found: The found name
                :param image_data: The current frame image
                """

                self.face_distances = face_distances
                self.face_encoding = face_encoding
                self.name_found = name_found
                self.image_data = image_data

            def destroyImageData(self):
                self.image_data = None

            def toDict(self):
                return {
                    "name_found": self.name_found,
                    # "predictions": list(self.predictions),
                    "face_distances": list(self.face_distances),
                }

        class Type(Enum):
            """
            The type of the scene
            """
            NoFace = 0  # No face encodings were found
            UnknownFace = 1  # Any face without a name
            Interview = 2  # Face and a name
            InterviewMatch = 3  # Any case that was matched with 'interview'

        def __init__(self, time_start_s, time_end_s=None):
            self.time_start_s = time_start_s
            self.time_end_s = time_end_s
            self.instances = dict()  # {timestamp: SceneInstance}

            # Calculated data
            self.face_info = None  # Holds final facial data of the scene
            self.match_confidence = None  # The match confidence with the given template
            self.scene_type = self.Type.NoFace  # The scene type
            self.new_person = None  # Marking if this scene holds an unidentified (new) person

        def getDuration(self):
            return self.time_end_s - self.time_start_s

        def add(self, time_stamp, scene_instance: SceneInstance, time_stamp_s=None):
            """
            Adds a scene instance

            :param time_stamp: The current timestamp of frame
            :param scene_instance: Scene instance of frame
            :return:
            """
            self.instances[time_stamp] = scene_instance
            if time_stamp_s is not None:
                self.time_end_s = time_stamp_s

        def calculated(self, face_info: FaceInfo, match_confidence, new_person: bool = None):
            self.face_info = face_info
            self.match_confidence = match_confidence
            if new_person is not None:
                self.new_person = new_person

        def evaluate(self):  # Final evaluation
            """
            Calculates scene type of scene

            :return:
            """
            if self.face_info is not None:  # If face found
                if self.getName() is not None:  # If name also found
                    self.scene_type = self.Type.Interview
                elif self.face_info.name is not None:  # If name was not found but name was matched
                    self.scene_type = self.Type.InterviewMatch
                else:  # If name was not found and no name matched
                    self.scene_type = self.Type.UnknownFace

        def getFaceDistances(self):
            return [x.face_distances for x in self.instances.values()]

        def getFaceEncodings(self):
            return [x.face_encoding for x in self.instances.values()]

        def getNames(self, with_count=False, exclude_none=True):
            names = [x.name_found for x in self.instances.values()]
            if with_count:
                return {
                    name: names.count(name) for name in set(names) if exclude_none and name is not None
                }
            else:
                return names

        def getName(self):
            names_dict = self.getNames(with_count=True, exclude_none=True)
            if len(names_dict) == 0:
                return
            return max(names_dict, key=names_dict.get)

        def getEncoding(self, first=False, middle=False, rand=False, average=False):
            """
            Returns encoding from the scene instances.  Only one of the parameter should be toggled as True.

            :param first: Gets the first encoding
            :param middle: Gets the middle encoding
            :param rand: Gets a random encoding
            :return:
            """
            instances = list(self.instances.values())

            if first:
                index = 0
            elif middle:
                index = len(instances) // 2
            elif rand:
                index = random.randint(0, len(instances) - 1)
            elif average:
                face_encodings_list = [x.face_encoding for x in instances]
                avg_face_encoding = sum(face_encodings_list) / len(face_encodings_list)
                return avg_face_encoding
            else:
                return

            return instances[index].face_encoding

        def getImageData(self):
            return [x.image_data for x in self.instances]

        def destroyImageData(self):
            for x in self.instances.values():
                x.destroyImageData()

        def toDict(self):
            """
            Converts class to dictionary
            :return: Dictionary
            """

            ret_dict = {
                # Time in mins..
                "start_scene": secondsStr(self.time_start_s),
                "end_scene": secondsStr(self.time_end_s),
                "match_value": self.match_confidence,
                "new_person": self.new_person,
                "face_info": self.face_info.toDict() if self.face_info is not None else None,
                "scene_type": self.scene_type.name,
                "instances": {timestamp: scene_instance.toDict() for timestamp, scene_instance in
                              self.instances.items()},
            }

            return ret_dict

    def __init__(self, video_path, base_output_folder,
                 unknown_default_name="Unknown", image_extension=".png",
                 analysis_data_name="analysis_data", output_json=True, face_images_folder="Images",
                 face_info_db_path=None,
                 skip_seconds_check: float = 5, min_scene_duration=0,
                 tolerance_match=0.6,
                 resolution: tuple[int, int] = None, images_with_info=False, keep_frames=False,
                 tracker_type="KCF", overwrite_tracker=False,
                 encodings_selection="average",
                 overwrite_base_folder=False,
                 load_previous=False,
                 show_warnings=True
                 ):
        """
        The main news analysis class

        :param video_path: The video path to analyse
        :param base_output_folder: The base output folder to output the analysis.
        :param unknown_default_name: The unknown person default name
        :param image_extension: The image extension
        :param analysis_data_name: Analysis pickle and json names
        :param output_json: Boolean to output a json file
        :param face_images_folder: The name of the folder for the facial images.  Note that a new folder will be created.
        :param face_info_db_path: A database path that can be inputted for the analysis to match with
        :param skip_seconds_check: Seconds to skip during analysis to make processing faster by reducing accuracy
        :param min_scene_duration: A minimum threshold for scene duration.  This is done to filter very short scenes
        :param tolerance_match: A tolerance match for facial matching
        :param resolution: The desired resolution of the image (Width, Height).  If none passed, the original will be used.
        :param images_with_info: The images will contain extra information overwritten.  This includes the scene number, name and confidence.
        :param keep_frames: Boolean to store frames as part of the information
        :param tracker_type: The tracker type to be used for the video analysis facial tracking
        :param overwrite_tracker: Boolean to overwrite the tracker bbox with the detection.  Note this will not change the scene timeframes.
        :param encodings_selection: 'first', 'middle' or 'average'. The face encoding selection method.
        :param overwrite_base_folder: If this is True, analysis will not be done if the base_output_folder already exists
        :param load_previous: If True, will load the previous news analysis in the same base_output_folder
        """

        # Save files
        self._pickle_path = os.path.join(base_output_folder, f"{analysis_data_name}.pkl")
        self._json_path = os.path.join(base_output_folder, f"{analysis_data_name}.json")
        self._face_images_folder = os.path.join(base_output_folder, face_images_folder)

        if load_previous:
            if os.path.exists(self._pickle_path):
                self.__dict__ = loadPickle(self._pickle_path).__dict__
                return
            else:
                if show_warnings:
                    print(f"Pickle path '{self._pickle_path}' not found")

        # Private variables
        self._video_path = video_path
        self._base_output_folder = base_output_folder
        self._image_extension = image_extension
        self._output_json = output_json
        self._time_start = time.time()
        self._time_end = None

        # Public variables
        self.unknown_default_name = unknown_default_name
        self.video_name = os.path.basename(video_path)
        self.skip_s = skip_seconds_check
        self.min_scene_duration = min_scene_duration
        self.tolerance_match = tolerance_match
        self.tracker_type = tracker_type
        self.overwrite_tracker = overwrite_tracker
        self.encodings_selection = encodings_selection

        # Extra info
        self.images_with_info = images_with_info
        self.keep_frames = keep_frames

        # Calculated variables (analysis method)
        self.time_taken_s = None
        self.fps = None
        self.frame_count = None
        self.analysis_date = None
        self.resolution = resolution  # Can be set.  If None, this will be replaced by the original resolution.
        self.original_resolution = None  # The original resolution of the video
        self.duration = None
        self.size = None

        # Internal variables (for functions)
        self._done = False  # Flag marks if done

        # Important data noted by '!'
        # ! Import database of faces if any
        if face_info_db_path is not None:
            self.face_info_db = FaceInfoDatabase(face_info_db_path, load_available=True)
        else:
            self.face_info_db = FaceInfoDatabase(None, load_available=False)

        # ! List of scenes
        self.scenes = list()  # Type: Scene()

        # Handles base output folder
        if not overwrite_base_folder and os.path.exists(base_output_folder):
            print(f"output_folder: '{base_output_folder}' already exists")
            self._done = True
            return

        # Create directory on analysis
        # os.makedirs(self._base_output_folder, exist_ok=True)  # Creating directory if not exists
        # os.makedirs(self._face_images_folder, exist_ok=True)  # Creating directory if not exists

    def createDirs(self):
        os.makedirs(self._base_output_folder, exist_ok=True)  # Creating directory if not exists
        os.makedirs(self._face_images_folder, exist_ok=True)  # Creating directory if not exists

    def load(self, pickle_path):  # Do not know if this works
        self.__dict__ = loadPickle(pickle_path).__dict__
        return self

    @staticmethod
    def staticLoad(analysis_dir, analysis_data_name: str = "analysis_data"):
        na = NewsAnalysis(None, analysis_dir, analysis_data_name=analysis_data_name, load_previous=True)
        return na

    def save(self):
        self.createDirs()  # Make sure directories exist

        # Calculating variables
        self.getSize(self._base_output_folder)
        self._time_end = time.time()
        self.time_taken_s = self.getTimeTaken()

        self.analysis_date = datetime.now()
        for scene in self.scenes:
            scene.evaluate()

        s_p, s_j = False, False

        # Pickle
        s_p = savePickle(self, self._pickle_path)

        # JSON
        if self._output_json:
            s_j = saveJson(self.toDict(), self._json_path)

        return s_p, s_j

    def getTimeTaken(self) -> int:
        """
        :return: Time in seconds
        """
        return self._time_end - self._time_start

    def getSize(self, folder):  # Get size in bytes
        total = 0  # https://note.nkmk.me/en/python-os-path-getsize/#:~:text=size%20with%20os.-,scandir()%20(Python%203.5%20or%20later),in%20a%20directory%20(folder).
        with os.scandir(folder) as it:
            for entry in it:
                if entry.is_file():
                    total += entry.stat().st_size
        self.size = total
        return total

    class DurationsDB:
        def __init__(self, scenes: list):
            self.durations = list()

            for scene in scenes:
                face_info = scene.face_info
                time_start, time_end = getTimeStamps(scene)
                # time_start, time_end = scene.time_start_s, scene.time_end_s

                duration_instance = self[face_info]
                if duration_instance is None:
                    duration_instance = self.Duration(face_info)
                    self.add(duration_instance)
                duration_instance.add_timestamp(time_start, time_end)

        def __len__(self):
            return len(self.durations)

        def toDict(self):
            return {
                "durations": [duration.toDict() for duration in self.durations]
            }

        def __getitem__(self, face_info):
            for x in self.durations:
                if x.face_info == face_info:
                    return x

        def index(self, face_info):
            for i, x in enumerate(self.durations):
                if x.face_info == face_info:
                    return i

        def __iter__(self):
            for x in self.durations:
                yield x

        def add(self, duration):
            self.durations.append(duration)

        def sort(self):
            self.durations = sorted(self.durations, key=lambda x: x.duration, reverse=True)

        class Duration:
            def __init__(self, face_info: FaceInfo):
                self.face_info = face_info

                self.timestamps = list()
                self.duration = 0

            def toDict(self):
                return {
                    "face_info": self.face_info.toDict(),
                    "duration": secondsStr(self.duration),
                    "timestamps": [{
                        "time_start": secondsStr(time_start),
                        "time_end": secondsStr(time_end),
                    }
                        for time_start, time_end in self.timestamps],
                }

            def add_timestamp(self, time_start, time_end):
                self.timestamps.append((time_start, time_end))
                self.duration += time_end - time_start

            def getTimeStampsFormatted(self):
                def conversion(total_seconds):
                    # Convert total seconds to a timedelta object
                    time_delta = timedelta(seconds=total_seconds)
                    # Format the timedelta object as a time string with a colon separator
                    time_string = str(time_delta).split('.')[0]
                    return time_string

                return [(conversion(time_start), conversion(time_end))
                        for time_start, time_end in self.timestamps]

            def getDuration(self):
                duration = 0
                for (time_start, time_end) in self.timestamps:
                    duration += time_end - time_start

                self.duration = duration
                return duration

    def getDurations(self, sort=True):
        durations = self.DurationsDB(self.scenes)
        if sort:
            durations.sort()

        return durations

    def toDict(self, skip_empty_scenes=True):
        return {
            # Additional info
            "video_name": self.video_name,
            "resolution": self.resolution,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "duration": secondsStr(self.duration),
            "size": self.size,
            "unknown_default_name": self.unknown_default_name,
            "skip_s": self.skip_s,
            "analysis_date": str(self.analysis_date),
            "time_taken_s": secondsStr(self.time_taken_s),

            # Durations
            "durations": self.getDurations().toDict()["durations"],

            # Important variables
            "face_info_db": self.face_info_db.toDict(),
            "scenes": {index: s.toDict() for index, s in enumerate(self.scenes) if
                       skip_empty_scenes and len(s.instances) > 0},
        }

    def analyseVideo(self, show_frame: bool = True, show_progress=True, callback_image=None, callback_update=None,
                     processes: int = 7,
                     write_all_images_dir=None, show_time=False,
                     ):
        """

        :param processes:
        :param show_frame:
        :param show_progress:
        :param callback_image: callback_image(frame, time_stamp_s)
        :return:
        """
        if self._done:
            print("Warning analysis already completed (start a new class instance to compute)")
            return
        if show_progress:
            print(f"Starting '{self._video_path}'")

        if write_all_images_dir is not None:
            os.makedirs(write_all_images_dir, exist_ok=True)

        self.createDirs()

        self._time_start = time.time()  # Start timer

        cap = cv.VideoCapture(self._video_path)  # Open video

        # Reading video information
        self.fps = cap.get(cv.CAP_PROP_FPS)
        self.frame_count = int(cap.get(cv.CAP_PROP_FRAME_COUNT))
        frame_skips_check = numFrames(self.skip_s, self.fps)

        # Set resolution
        width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
        height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
        self.original_resolution = (width, height)
        if self.resolution is None:
            set_resolution = False
        else:
            set_resolution = True
            width = self.resolution[0]
            height = self.resolution[1]
        self.resolution = (width, height)

        # Duration
        self.duration = getDuration(self.frame_count, self.fps)

        # Face encodings
        encodings_local_flat = self.face_info_db.getProfileEncodings()  # Run with each new scene

        # Face encoding multiprocess pool
        multiprocess = False if processes is None or processes == 0 else True
        if multiprocess:
            face_encoding_pool = multiprocessing.Pool(processes=processes)
            face_encoding_results = list()

        # Face tracker
        tracker = None

        def setTracker():
            nonlocal tracker

            if self.tracker_type == 'BOOSTING':
                tracker = cv.TrackerBoosting_create()
            elif self.tracker_type == 'MIL':
                tracker = cv.TrackerMIL_create()
            elif self.tracker_type == 'KCF':
                tracker = cv.TrackerKCF_create()
            elif self.tracker_type == 'TLD':
                tracker = cv.TrackerTLD_create()
            elif self.tracker_type == 'MEDIANFLOW':
                tracker = cv.TrackerMedianFlow_create()
            elif self.tracker_type == 'CSRT':
                tracker = cv.TrackerCSRT_create()
            elif self.tracker_type == 'MOSSE':
                tracker = cv.TrackerMOSSE_create()
            else:
                tracker = cv.TrackerKCF_create()

        scene = None
        bbox = None
        temp_image_map = dict()  # {name_found: frame}

        def callEndSceneDetected():
            nonlocal scene
            nonlocal bbox
            nonlocal temp_image_map
            nonlocal encodings_local_flat
            nonlocal multiprocess
            nonlocal face_encoding_results

            # Check if scene exceeds limit
            if scene is not None and scene.getDuration() >= self.min_scene_duration:
                self.scenes.append(scene)  # Add scene to scenes

                # Get results of face encodings
                if multiprocess:
                    for i, scene_instance in enumerate(scene.instances.values()):
                        found_face_encoding = face_encoding_results[i].get()
                        face_distances = face_recognition.face_distance(encodings_local_flat,
                                                                        found_face_encoding)
                        scene_instance.face_distances = face_distances
                        scene_instance.face_encoding = found_face_encoding
                    face_encoding_results.clear()

                # Process scene.
                self.processScene(scene, temp_image_map=temp_image_map)

                # Get new encodings
                encodings_local_flat = self.face_info_db.getProfileEncodings()

            # Reset variables
            scene = None
            bbox = None
            temp_image_map.clear()  # {name_found: frame}

        # Analysis
        if show_progress:
            pbar = tqdm(total=self.frame_count, desc=f"Analysing '{self._video_path}'")
        while cap.isOpened():
            ret, frame = cap.read()

            if ret:
                time_stamp = cap.get(cv.CAP_PROP_POS_FRAMES) - 1  # Timestamp
                time_stamp_s = getDuration(time_stamp, self.fps)  # Timestamp in seconds
                if callback_update is not None:
                    callback_update(time_stamp, self.frame_count)  # Percentage complete

                if set_resolution:  # Sets resolution if needed
                    frame = cv.resize(frame, self.resolution, interpolation=cv.INTER_AREA)

                # Keep image frame  if self.images_with_info:
                frame_copy = copy.deepcopy(frame)
                keep_frame_temp = frame if self.images_with_info else frame_copy

                # First try to get bbox with tracker (Ends Scene)
                if bbox is not None:

                    success, bbox_cv = tracker.update(frame)

                    if success:
                        bbox = convert_bbox_to_face_recognition(bbox_cv)
                    else:
                        # Tracker ended.

                        callEndSceneDetected()

                # If fails, try to get bbox with detect_faces (Starts Scene)
                skip_encoding = False
                if bbox is None:
                    boxes = detect_faces(frame)  # Get boxes of faces

                    if len(boxes) == 1:
                        bbox = boxes[0]

                        # bbox found, start a new scene
                        scene = self.Scene(time_stamp_s)
                        # self.scenes.append(scene)

                        bbox_cv = convert_bbox_to_cv2(bbox)
                        setTracker()
                        tracker.init(frame, bbox_cv)  # Start tracking
                    else:
                        bbox = None
                elif self.overwrite_tracker:  # If there is a bbox, just adjust the bbox regardless, if no face is found, assume there is still one
                    boxes = detect_faces(frame)  # Get boxes of faces
                    if len(boxes) == 1:
                        bbox = boxes[0]

                    else:
                        skip_encoding = True

                        # # Update tracker bbox
                        # bbox_cv = convert_bbox_to_cv2(bbox)
                        # setTracker()
                        # tracker.init(frame, bbox_cv)

                if bbox is not None and not skip_encoding:
                    name_found = readText.getTextFilter(frame)  # Gets name of person

                    if multiprocess:
                        # found_face_encoding = face_recognition.face_encodings(frame, [bbox])[0]  # Taking longest time
                        #
                        # face_distances = face_recognition.face_distance(encodings_local_flat, found_face_encoding)

                        async_result = face_encoding_pool.apply_async(face_encoding, args=(frame_copy, bbox))
                        face_encoding_results.append(async_result)

                        scene_inst = self.Scene.SceneInstance([], None,
                                                              name_found=name_found,
                                                              image_data=keep_frame_temp if self.keep_frames else None)
                    else:
                        found_face_encoding = face_recognition.face_encodings(frame, [bbox])[0]  # Taking longest time

                        face_distances = face_recognition.face_distance(encodings_local_flat, found_face_encoding)

                        scene_inst = self.Scene.SceneInstance(face_distances, found_face_encoding,
                                                              name_found=name_found,
                                                              image_data=keep_frame_temp if self.keep_frames else None)

                    scene.add(time_stamp, scene_inst, time_stamp_s=time_stamp_s)

                    # # Call back image
                    # if callback_image is not None:
                    #     callback_image(keep_frame_temp, time_stamp_s)

                    if name_found not in temp_image_map:
                        temp_image_map[name_found] = frame_copy
                    if self.unknown_default_name not in temp_image_map:
                        temp_image_map[self.unknown_default_name] = frame_copy

                    if show_frame:
                        fontScale = 0.5
                        org1 = (5, 20)
                        org2 = (5, 35)
                        org3 = (5, 50)
                        org4 = (5, 75)
                        thickness = 1
                        color = (0, 255, 0)
                        fontFace = cv.FONT_HERSHEY_TRIPLEX

                        cv.putText(img=frame, text=f"Name Found: '{name_found}'", org=org1,
                                   fontFace=cv.FONT_HERSHEY_TRIPLEX,
                                   fontScale=fontScale,
                                   color=color, thickness=thickness)

                        cv.putText(img=frame, text=f"Current Scene Name: '{scene.getName()}'", org=org2,
                                   fontFace=cv.FONT_HERSHEY_TRIPLEX,
                                   fontScale=fontScale,
                                   color=color, thickness=thickness)

                        cv.putText(img=frame,
                                   text=f"Scene #{len(self.scenes)}",
                                   org=org3,
                                   fontFace=fontFace,
                                   fontScale=fontScale,
                                   color=color, thickness=thickness)
                        if show_time:
                            cv.putText(img=frame,
                                       text=f"{secondsStr(time_stamp_s)}",
                                       org=org4,
                                       fontFace=fontFace,
                                       fontScale=fontScale,
                                       color=color, thickness=thickness)

                        if bbox is not None:
                            (top, right, bottom, left) = bbox
                            cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                        if write_all_images_dir is not None:
                            image_path = getDuplicateName(write_all_images_dir, f"Frame #{time_stamp}",
                                                          extension=".png")
                            cv.imwrite(image_path, frame,
                                       [int(cv.IMWRITE_PNG_COMPRESSION), 0])

                if show_frame:
                    cv.imshow(self.video_name, frame)

                    key = cv.waitKey(1)
                    # if key == ord('s'):
                    #     os.makedirs("screenshots")
                    #     screenshot_path = getDuplicateName(folder="screenshots", file_name="screenshot",
                    #                                        extension="png")
                    #     cv.imwrite(screenshot_path, frame)
                    if key == ord('q'):
                        break

                # Call back image
                if callback_image is not None:
                    callback_image(keep_frame_temp, time_stamp_s)

                # Skipping frames
                if frame_skips_check != 0:
                    cap.set(cv.CAP_PROP_POS_FRAMES, time_stamp + frame_skips_check)
                    if show_progress:
                        pbar.update(frame_skips_check)
                else:
                    if show_progress:
                        pbar.update()

            else:
                if scene is not None:
                    callEndSceneDetected()
                break

        if show_progress:
            pbar.close()
        if multiprocess:
            face_encoding_pool.close()
            face_encoding_pool.join()
        if show_frame:
            cv.destroyWindow(self.video_name)

        self._done = True
        self.save()

    def processScene(self, scene: Scene, temp_image_map=None):
        if temp_image_map is None:
            temp_image_map = dict()

        if len(scene.instances) > 0:
            value = None
            new_person = True
            face_info_match = None

            face_distances = scene.getFaceDistances()
            if len(face_distances) > 0:
                res = getMatchIndex(face_distances, include_value=True, maximum=False)
                if res is not None:
                    index, value = res
                    if value <= self.tolerance_match:  # Check if match with database
                        new_person = False
                        face_info_match = self.face_info_db[index]

                        # Update name of local face_info
                        if face_info_match.name is None:
                            new_name = scene.getName()
                            if new_name is not None:
                                # Handling new met name and profile image
                                face_info_match.setProfile(new_name, image_data=temp_image_map[new_name])

                                # Removing and Writing new image
                                os.remove(os.path.join(self._face_images_folder, face_info_match.image_basename))
                                image_path = getDuplicateName(self._face_images_folder, new_name,
                                                              extension=".png")
                                face_info_match.image_basename = os.path.basename(image_path)
                                cv.imwrite(image_path, face_info_match.profile_image_data)

            if new_person:  # Create new person
                new_name = scene.getName()

                first, middle, average = False, False, False
                if self.encodings_selection == "first":
                    first = True
                elif self.encodings_selection == "middle":
                    middle = True
                elif self.encodings_selection == "average":
                    average = True

                new_face_info = FaceInfo(new_name, profile_encoding=scene.getEncoding(
                    first=first, middle=middle, average=average),
                                         profile_image_data=temp_image_map[new_name])

                self.face_info_db.add(new_face_info)
                face_info_match = new_face_info

                # Handles images
                if new_name is None:
                    image_name = self.unknown_default_name
                else:
                    image_name = new_name
                image_path = getDuplicateName(self._face_images_folder, image_name, extension=".png")
                face_info_match.image_basename = os.path.basename(image_path)
                cv.imwrite(image_path, temp_image_map[image_name])

                # face_info_match.setProfileImage(temp_image_map[image_name])
                # # Set encoding
                # face_info_match.face_data[0].encoding = found_face_encoding

            # face_info_match.addFaceData(scene.getFaceDataList())
            # face_info_match.face_data.extend(
            #     scene.getFaceDataList())  # Add matched encodings (maybe add some threshold on what to add)
            scene.face_info = face_info_match
            # print("value", value)
            scene.calculated(face_info_match, value, new_person=new_person)

        self.save()


def getDuration(frame_count, fps,
                frame_skips=1):  # Returns duration in seconds.  Also accounting for any frame_skips.
    if frame_count is None or fps is None or frame_skips is None:
        return
    return (frame_count * (frame_skips)) / fps


def numFrames(seconds: int, fps: float) -> int:
    """
    :param seconds: Number of seconds
    :param fps: The frames per second of the video
    :return: The number of frames
    """
    frames = seconds * fps
    return round(frames)


def convert_size(size_bytes):  # https://stackoverflow.com/questions/5194057/better-way-to-convert-file-sizes-in-python
    if size_bytes == 0:
        return "0B"
    size_name = ("B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB")
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    return "%s %s" % (s, size_name[i])


def analyseDir(extract_folder, output_folder,
               limit_video: int = None,
               show_updates=False,
               face_info_db_path=None,
               export_db=None, database_name="face_database.dat", remove_unknown=True, log_name="log.json",
               processes=7, show_frame=True,
               **kwargs):
    """
        Analyze a directory of videos for faces and names.

        Parameters:
        extract_folder (str): Path to folder containing videos.
        output_folder (str): Path to folder where analysis output will be saved.
        limit_video (int): Optional limit on number of videos to analyze.
        show_updates (bool): Flag to print progress updates.
        face_info_db_path (str): Path to an existing face database.
        export_db (str): Path to export face data to a new database.
        database_name (str): Name of the new face database file.
        remove_unknown (bool): Flag to remove entries with unknown names from the new database.
        log_name (str): Name of the log file.
        processes (int): Number of processes for face encodings in a single video scene
        show_frame (bool): Flag to show frames
        kwargs: Additional keyword arguments for NewsAnalysis class.
    """
    if export_db is not None:
        face_info_db_path = os.path.join(export_db, database_name)
        log_path = os.path.join(export_db, log_name)

        os.makedirs(export_db, exist_ok=True)

        # Loading log_path
        if os.path.exists(log_path):
            with open(log_path, "r") as f:
                log = json.load(f)
        else:
            log = dict()

    time_start = time.time()

    returns = []

    videos = [f for f in os.listdir(extract_folder) if
              os.path.isfile(os.path.join(extract_folder, f)) and f.endswith(".mp4")]

    if limit_video is not None:
        videos = videos[:limit_video]

    pbar = tqdm(total=len(videos), desc=f"Analysing videos from '{extract_folder}' to '{output_folder}'")
    for video in videos:
        # print(video)
        name = video.replace(".mp4", "")
        video_path = os.path.join(extract_folder, video)
        output_folder_analysis = os.path.join(output_folder, name)

        if os.path.exists(output_folder_analysis):
            if show_updates:
                print(f"Analysis for date {video} already exists")
        else:
            na = NewsAnalysis(video_path, output_folder_analysis, overwrite_base_folder=True,
                              face_info_db_path=face_info_db_path, **kwargs)
            na.analyseVideo(processes=processes, show_frame=show_frame, show_progress=False)
            # With each video, update database if enabled
            if export_db is not None:
                export_database = FaceInfoDatabase(face_info_db_path, load_available=True)
                import_database = na.face_info_db
                # Keeping non-unknowns
                if remove_unknown:
                    face_info_data = [fi for fi in import_database.data if fi.name is not None]
                else:
                    face_info_data = import_database.data
                problems = export_database.inputFaceInfo(face_info_data, duplicate_names=False)
                export_database.save()

                log[output_folder_analysis] = {"Exports": [face_info.name for face_info in face_info_data],
                                               "Problems": [p.name for p in problems]}
                with open(log_path, "w") as f:
                    json.dump(log, f, indent=4)
        pbar.update()

    pbar.close()

    time_end = time.time()
    total_time = time_end - time_start
    if show_updates:
        print("Minutes taken: ", total_time / 60)

    return returns


# For multiprocessing
def doAnalysis(news_analysis: NewsAnalysis, index, show_frame=False):
    news_analysis.analyseVideo(show_frame=show_frame, show_progress=False, processes=0)
    return 1


def update(result, tqdm_obj):
    # Do something with the result, if needed
    # Update the tqdm progress bar
    tqdm_obj.update()


def analyseDirMulti(extract_folder, output_folder, show_updates=False, limit_video: int = None, processes=3,
                    show_frame=False, **kwargs):
    """
        folder: Path to folder containing videos
        frame_skips: Number of frames to skip each time
    """

    def custom_error_callback(error):
        print(f'Got error: {error}')

    time_start = time.time()

    pool = multiprocessing.Pool(processes=processes)

    videos = [f for f in os.listdir(extract_folder) if
              os.path.isfile(os.path.join(extract_folder, f)) and f.endswith(".mp4")]

    if limit_video is not None:
        videos = videos[:limit_video]

    pbar = tqdm(total=len(videos), desc=f"Analysing videos from '{extract_folder}' to '{output_folder}'")

    pool_results = list()

    for i, video in enumerate(videos):
        # print(video)
        name = video.replace(".mp4", "")
        video_path = os.path.join(extract_folder, video)
        output_folder_analysis = os.path.join(output_folder, name)

        if os.path.exists(output_folder_analysis):
            if show_updates:
                print(f"Analysis for date {video} already exists")
            pbar.update()
        else:
            na = NewsAnalysis(video_path, output_folder_analysis, overwrite_base_folder=True, **kwargs)
            result = pool.apply_async(doAnalysis,
                                      args=(na, i, show_frame),
                                      callback=lambda x: update(x, pbar),
                                      error_callback=custom_error_callback)
            pool_results.append(result)

    # Wait for all tasks to complete
    for result in pool_results:
        result.wait()

    # Close the multiprocessing pool
    pool.close()
    pool.join()

    # Close the tqdm progress bar
    pbar.close()

    time_end = time.time()
    total_time = time_end - time_start
    if show_updates:
        print(f"Time Taken: {utils.secondsStr(total_time)}")


def InputDatabaseAll(search_folder, export_folder,
                     database_name="face_database.dat",
                     log_name="log.json",
                     remove_unknown=True,
                     **kwargs):
    # Assigning paths
    export_database_path = os.path.join(export_folder, database_name)
    log_path = os.path.join(export_folder, log_name)

    os.makedirs(export_folder, exist_ok=True)

    # Loading log_path
    if os.path.exists(log_path):
        with open(log_path, "r") as f:
            log = json.load(f)
    else:
        log = dict()

    # Loading database
    export_database = FaceInfoDatabase(export_database_path, load_available=True)

    subfolders = [f for f in os.scandir(search_folder) if f.is_dir()]
    pbar = tqdm(total=len(subfolders), desc="Inputting in database")
    for subfolder in subfolders:
        import_database = NewsAnalysis(None, subfolder, load_previous=True, **kwargs).face_info_db

        # Keeping non-unknowns
        if remove_unknown:
            face_info_data = [fi for fi in import_database.data if fi.name is not None]
        else:
            face_info_data = import_database.data

        problems = export_database.inputFaceInfo(face_info_data, duplicate_names=False)
        export_database.save()

        log[subfolder.path] = {"Exports": [face_info.name for face_info in face_info_data],
                               "Problems": [p.name for p in problems]}

        with open(log_path, "w") as f:
            json.dump(log, f, indent=4)
        pbar.update()
    return export_database_path


def test():
    NewsAnalysis("data/NewsSegments/30.08.2022 segment #3.mp4", "data/test", overwrite_base_folder=True,
                 # skip_seconds_check=2,
                 skip_seconds_check=1,

                 # min_scene_duration=0,
                 min_scene_duration=0,

                 # tolerance_match=0.6,
                 tolerance_match=0.5,

                 # resolution=(640, 360),
                 overwrite_tracker=True,
                 tracker_type="KCF",

                 face_info_db_path="data/comparing_segments/tracking_database_2/database/face_database.dat"
                 ).analyseVideo(processes=7, show_frame=True, write_all_images_dir="screenshots")


if __name__ == "__main__":
    print("=== MAIN ===\n\n")
    test()

    NewsAnalysis("data/extract_full_news/01.08.2022.mp4", "data/test", overwrite_base_folder=True,
                 skip_seconds_check=0.5, resolution=(640, 360), tracker_type="KCF").analyseVideo(processes=7,
                                                                                                 show_frame=True)

    # analyseDir("data/extract_full_news", "data/comparing/encoding database 1", resolution=(640, 360))
    # analyseDir("data/extract_full_news", "data/comparing/encoding database 1", resolution=(640, 360), export_db="data/comparing/encoding database 1/database")

    # analyseDirMulti("data/extract_full_news", "data/comparing/encoding no database 1", resolution=(640, 360), processes=3)
    # analyseDir("data/extract_full_news", "data/comparing/encoding with database 1", resolution=(640, 360),
    #            show_updates=True, export_db="data/comparing/encoding with database 1/database")

    # InputDatabaseAll("dat/comparing/encoding averaging/no database", "dat/test/export")

    # NewsAnalysis("dat/extract_full_news/01.08.2022.mp4", "dat/test", overwrite_base_folder=True,
    #              skip_seconds_check=2, face_info_db_path="dat/test/export/face_database.dat").analyseVideo(resolution=(640, 360))
