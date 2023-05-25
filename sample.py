import os

from video_analysis_1 import *

"""
For showing analysis video in action

Create a sample video for the analysis for viewing rather than actual analysis
"""


def extractSampleAnalysis(video_path, extract_dir="Examples", output_video_name="output.mp4"):
    base_name = os.path.basename(video_path)
    name = os.path.splitext(base_name)[0]
    output_dir = os.path.join(extract_dir, name)

    os.makedirs(output_dir, exist_ok=True)

    na = NewsAnalysis(video_path, output_dir, overwrite_base_folder=True, skip_seconds_check=0,
                      face_info_db_path=None,
                      keep_frames=True, images_with_info=True)
    na.analyseVideo(show_frame=True, show_progress=True)


def extractSampleVideo(dir_path, output_video_name="output #?.mp4", analysis_data_name="analysis_data"):
    output_video_path = os.path.join(dir_path, output_video_name)

    na = NewsAnalysis(None, dir_path, analysis_data_name=analysis_data_name, load_previous=True)

    size = na.resolution
    size = (int(size[0]), int(size[1]))
    fps = na.fps

    for index, scene in enumerate(na.scenes):
        scene_length = len(scene.instances)
        if scene_length > 0:
            result = cv.VideoWriter(output_video_path.replace("?", str(index)),
                                    cv.VideoWriter_fourcc(*'mp4v'),
                                    fps, size)

            pbar = tqdm(total=scene_length, desc=f"Extracting video from scene #{index}")
            for instance in scene.instances.values():
                frame = instance.image_data

                result.write(frame)
                cv.imshow('Frame', frame)

                if cv.waitKey(1) & 0xFF == ord('s'):
                    break

                pbar.update()
            pbar.close()

            result.release()

            # Closes all the frames
            cv.destroyAllWindows()


if __name__ == "__main__":
    # extractSampleVideo("data/NewsSegments/27.08.2022 segment #3.mp4")
    # extractSampleVideo("data/NewsSegments/03.09.2022 segment #1.mp4")

    # extractSampleAnalysis("data/NewsSegments/13.08.2022 segment #1.mp4")
    extractSampleVideo("Examples/13.08.2022 segment #1")
