from Katna.video import Video
from Katna.writer import KeyFrameDiskWriter
import os
from scenedetect import detect, ContentDetector, SceneManager, open_video, video_splitter
from video_analysis_1 import numFrames


def splitVideo():
    VIDEO_PATH = "tests/01.08.2022.mp4"

    video = open_video(VIDEO_PATH)
    fps = video.frame_rate
    frame_skips_check_detector = numFrames(2, 30)
    scene_manager = SceneManager()
    scene_manager.add_detector(ContentDetector(threshold=27.0))
    scene_manager.detect_scenes(video, show_progress=True, frame_skip=frame_skips_check_detector)
    video_splitter.split_video_ffmpeg(VIDEO_PATH, scene_manager.get_scene_list(),
                                      video_name="tests/scenes/$VIDEO_NAME-Scene-$SCENE_NUMBER.mp4", show_progress=True)


def getKeyFrame():
    # initialize video module
    vd = Video()

    directory = "tests/scenes/"
    for file in os.listdir(directory):
        name = os.path.splitext(file)[0]
        file_path = os.path.join(directory, file)

        diskwriter = KeyFrameDiskWriter(location=f"tests/selectedframes/{name}")

        vd.extract_video_keyframes(
            no_of_frames=1,
            file_path=file_path,
            writer=diskwriter,
        )


if __name__ == "__main__":
    # splitVideo()
    getKeyFrame()
