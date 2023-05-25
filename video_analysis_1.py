import copy
import math
import os.path
import random
import shutil
import time
from datetime import datetime
from enum import Enum
import cv2 as cv
import numpy as np
from tqdm import tqdm
from scenedetect import detect, ContentDetector, SceneManager, open_video
import multiprocessing

import face_recognition

from utils import *
import readText

"""

- Metric face encodings with face_recognition package
- Middle image as template
- Scenes with scenedetect package
- Comparing all scores of each scene to label a person
"""


class FaceInfo:
    """
    Facial information
    """

    def __init__(self, name: str, profile_image_data=None, profile_encoding=None):
        """

        :param name: Name of person
        :param profile_image_data: Profile image
        :param profile_encoding: Profile face encoding
        """
        self.name = name  # Name of person
        self.profile_image_data = profile_image_data  # The image showing this person
        self.profile_encoding = profile_encoding

    def setProfileImage(self, image_data):
        self.profile_image_data = image_data

    def setProfileEncoding(self, encoding):
        self.profile_encoding = encoding

    def setProfile(self, name, encoding=None, image_data=None):
        self.name = name
        if encoding is not None:
            self.profile_encoding = encoding
        if image_data is not None:
            self.profile_image_data = image_data

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

    def saveImages(self, directory=None, duplicate_new=False):
        """
        Saving images of personal database

        :param directory:
        :param duplicate_new:
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
            image_data = fi.profile_image_data
            name = fi.name

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

        def __init__(self, scene_timestamps):
            """
            :param scene_timestamps: Scene start and end timestamps
            """

            self.scene_timestamps = scene_timestamps
            self.instances = dict()  # {timestamp: SceneInstance}

            # Calculated data
            self.face_info = None  # Holds final facial data of the scene
            self.match_confidence = None  # The match confidence with the given template
            self.scene_type = self.Type.NoFace  # The scene type
            self.new_person = None  # Marking if this scene holds an unidentified (new) person

        def add(self, time_stamp, scene_instance: SceneInstance):
            """
            Adds a scene instance

            :param time_stamp: The current timestamp of frame
            :param scene_instance: Scene instance of frame
            :return:
            """
            self.instances[time_stamp] = scene_instance

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

        def getEncoding(self, first=False, middle=False, rand=False):
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
            else:
                return

            return instances[index].face_encoding

        def toDict(self):
            """
            Converts class to dictionary
            :return: Dictionary
            """

            ret_dict = {
                # Time in mins..
                "start_scene": self.scene_timestamps[0].get_seconds(),
                "end_scene": self.scene_timestamps[1].get_seconds(),
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
                 skip_seconds_check=5, skip_scene_seconds_check=5, min_scene_duration=0,
                 tolerance_match=0.6,
                 resolution: tuple[int, int] = None, images_with_info=False, keep_frames=False,
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
        :param skip_scene_seconds_check: Seconds to skip during analysis for scene detection
        :param min_scene_duration: A minimum threshold for scene duration.  This is done to filter very short scenes
        :param tolerance_match: A tolerance match for facial matching
        :param resolution: The desired resolution of the image (Width, Height).  If none passed, the original will be used.
        :param images_with_info: The images will contain extra information overwritten.  This includes the scene number, name and confidence.
        :param keep_frames: Boolean to store frames as part of the information
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
        self.skip_s_scene = skip_scene_seconds_check
        self.min_scene_duration = min_scene_duration
        self.tolerance_match = tolerance_match

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

        os.makedirs(base_output_folder, exist_ok=True)  # Creating directory if not exists
        os.makedirs(self._face_images_folder, exist_ok=True)  # Creating directory if not exists

    def load(self, pickle_path):  # Do not know if this works
        self.__dict__ = loadPickle(pickle_path).__dict__

    def save(self):
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

    def toDict(self, skip_empty_scenes=True):
        return {
            # Additional info
            "video_name": self.video_name,
            "resolution": self.resolution,
            "frame_count": self.frame_count,
            "fps": self.fps,
            "duration": self.duration,
            "size": self.size,
            "unknown_default_name": self.unknown_default_name,
            "skip_s": self.skip_s,
            "skip_s_scene": self.skip_s_scene,
            "analysis_date": str(self.analysis_date),
            "time_taken_s": secondsStr(self.time_taken_s),

            # Important variables
            "face_info_db": self.face_info_db.toDict(),
            "scenes": {index: s.toDict() for index, s in enumerate(self.scenes) if
                       skip_empty_scenes and len(s.instances) > 0},
        }

    def analyseVideo(self, show_frame: bool = True, show_progress=True, callback_image=None):
        """

        :param show_frame:
        :param show_progress:
        :param callback_image: callback_image(frame)
        :return:
        """
        if self._done:
            print("Warning analysis already completed (start a new class instance to compute)")
            return
        print(f"Starting '{self._video_path}'")

        self._time_start = time.time()  # Start timer

        # Start retrieving scenes
        video = open_video(self._video_path)
        fps = video.frame_rate
        frame_skips_check_detector = numFrames(self.skip_s_scene, fps)
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=27.0))
        scene_manager.detect_scenes(video, show_progress=show_progress, frame_skip=frame_skips_check_detector)

        self.scenes = [self.Scene(s) for s in scene_manager.get_scene_list()]
        self.save()

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

        # Analysis
        # Looping over scenes
        if show_progress:
            pbar = tqdm(total=len(self.scenes), desc="Analysing scenes of video {}".format(self._video_path))
        for index, scene in enumerate(self.scenes):
            if show_progress:
                pbar.update(1)

            start_frame = scene.scene_timestamps[0].get_frames()
            end_frame = scene.scene_timestamps[1].get_frames()
            scene_duration = scene.scene_timestamps[1].get_seconds() - scene.scene_timestamps[0].get_seconds()

            # Scene must not be shorter than provided duration
            if scene_duration < self.min_scene_duration:
                continue

            # Scene encodings
            encodings_local_flat = self.face_info_db.getProfileEncodings()

            # Scene image map
            temp_image_map = dict()  # {name_found: frame}

            # Analyse Scene
            # Start reading frames from current scene
            cap.set(cv.CAP_PROP_POS_FRAMES, start_frame - 1)
            while cap.isOpened() and cap.get(cv.CAP_PROP_POS_FRAMES) - 1 < end_frame:
                ret, frame = cap.read()

                if ret:
                    time_stamp = cap.get(cv.CAP_PROP_POS_FRAMES) - 1  # Timestamp

                    if set_resolution:  # Sets resolution if needed
                        frame = cv.resize(frame, self.resolution, interpolation=cv.INTER_AREA)

                    boxes = detect_faces(frame)  # Gets boxes of faces

                    name_found = None
                    if len(boxes) == 1:  # Only allows one face
                        # Keep image frame
                        if self.images_with_info:
                            frame_copy = frame
                        else:
                            frame_copy = copy.deepcopy(frame)

                        name_found = readText.getTextFilter(frame)  # Gets name of person

                        frame_rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)

                        time_stamp = cap.get(cv.CAP_PROP_POS_FRAMES) - 1

                        found_face_encoding = face_recognition.face_encodings(frame_rgb, boxes)[0]

                        face_distances = face_recognition.face_distance(encodings_local_flat, found_face_encoding)

                        scene_inst = self.Scene.SceneInstance(face_distances, found_face_encoding,
                                                              name_found=name_found,
                                                              image_data=frame_copy if self.keep_frames else None)
                        scene.add(time_stamp, scene_inst)

                        # Call back image
                        if callback_image is not None:
                            callback_image(frame_copy)

                        if name_found not in temp_image_map:
                            temp_image_map[name_found] = frame_copy
                        if self.unknown_default_name not in temp_image_map:
                            temp_image_map[self.unknown_default_name] = frame_copy

                    if show_frame:
                        fontScale = 0.5
                        org1 = (5, 20)
                        org2 = (5, 35)
                        org3 = (5, 50)
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
                                   text=f"Scene #{index}",
                                   org=org3,
                                   fontFace=fontFace,
                                   fontScale=fontScale,
                                   color=color, thickness=thickness)

                        if len(boxes) > 0:
                            (top, right, bottom, left) = boxes[0]
                            cv.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

                        # image_path = getDuplicateName("dat/test/frames", f"Scene #{index}", extension=".png")
                        # cv.imwrite(image_path, frame)

                    if show_frame:
                        cv.imshow('frame', frame)
                        if cv.waitKey(1) & 0xFF == ord('q'):
                            break
                    # Skipping frames
                    if frame_skips_check != 0:
                        cap.set(cv.CAP_PROP_POS_FRAMES, time_stamp + frame_skips_check)
                else:
                    break

            # Either match of create new person
            if len(scene.instances) > 0:
                value = None
                new_person = True
                face_info_match = None

                face_distances = scene.getFaceDistances()
                if len(face_distances) > 0:
                    res = getMatchIndex(face_distances, include_value=True, maximum=False)
                    if res is not None:
                        index, value = res
                        if value < 0.6:  # Check if match with database
                            new_person = False
                            face_info_match = self.face_info_db[index]

                            # Update name of local face_info
                            if face_info_match.name is None:
                                new_name = scene.getName()
                                if new_name is not None:
                                    # Handling new met name and profile image
                                    face_info_match.setProfile(new_name,
                                                               image_data=temp_image_map[new_name])

                                    # Removing and Writing new image
                                    os.remove(os.path.join(self._face_images_folder, face_info_match.image_basename))
                                    image_path = getDuplicateName(self._face_images_folder, new_name,
                                                                  extension=".png")
                                    face_info_match.image_basename = os.path.basename(image_path)
                                    cv.imwrite(image_path, face_info_match.profile_image_data)

                if new_person:  # Create new person
                    new_name = scene.getName()
                    new_face_info = FaceInfo(new_name, profile_encoding=scene.getEncoding(middle=True),
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

        if show_progress:
            pbar.close()

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
               **kwargs):
    """
        folder: Path to folder containing videos
        frame_skips: Number of frames to skip each time
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
                              skip_seconds_check=2, face_info_db_path=face_info_db_path, **kwargs)
            na.analyseVideo()
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
        print(f"Time Taken: {secondsStr(total_time)}")


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


if __name__ == "__main__":
    print("=== MAIN ===\n\n")
    # NewsAnalysis("dat/extract_full_news/01.08.2022.mp4", "dat/test", overwrite_base_folder=True,
    #              skip_seconds_check=2).analyseVideo(resolution=(640, 360))

    # analyseDir("data/extract_full_news", "data/comparing/encoding database 1", resolution=(640, 360))
    # analyseDir("data/extract_full_news", "data/comparing/encoding database 1", resolution=(640, 360), export_db="data/comparing/encoding database 1/database")

    # analyseDirMulti("data/extract_full_news", "data/comparing/encoding no database 1", resolution=(640, 360), processes=3)
    analyseDir("data/extract_full_news", "data/comparing/encoding with database 1", resolution=(640, 360),
               show_updates=True, export_db="data/comparing/encoding with database 1/database")

    # InputDatabaseAll("dat/comparing/encoding averaging/no database", "dat/test/export")

    # NewsAnalysis("dat/extract_full_news/01.08.2022.mp4", "dat/test", overwrite_base_folder=True,
    #              skip_seconds_check=2, face_info_db_path="dat/test/export/face_database.dat").analyseVideo(resolution=(640, 360))
