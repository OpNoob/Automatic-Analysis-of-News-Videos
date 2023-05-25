import os

import cv2
import numpy as np

import video_analysis_2
import results


def main():
    video_name = "15.08.2022 segment #3"
    video_path = f"data/NewsSegments/{video_name}.mp4"
    base_dir = "Graphics/FrameTimeline"

    frame_folder = "frames"
    frame_folder_path = os.path.join(base_dir, frame_folder)
    os.makedirs(frame_folder_path, exist_ok=True)

    def save_images(img_frame, time_stamp_s):
        cv2.imwrite(os.path.join(frame_folder_path, f"{time_stamp_s}.png"), img_frame)

    na = video_analysis_2.NewsAnalysis(video_path,
                                       os.path.join(base_dir, video_name),
                                       overwrite_base_folder=False,
                                       load_previous=True,
                                       skip_seconds_check=1,
                                       resolution=(640, 360),
                                       tracker_type="KCF",
                                       face_info_db_path=None,
                                       keep_frames=True,
                                       images_with_info=True,
                                       )
    na.analyseVideo(show_frame=True, processes=7, callback_image=save_images)

    images = list()
    time_stamps = list()

    # Using the video dir
    image_files = os.listdir(frame_folder_path)
    for image_file in image_files:
        image_path = os.path.join(frame_folder_path, image_file)
        image = cv2.imread(image_path)

        images.append(image)

        time_stamp_s = os.path.splitext(image_file)[0]

        time_stamps.append(int(time_stamp_s.split(".0")[0]))

    # Using na
    # for scene in na.scenes:
    #     for frame_no, inst in scene.instances.items():
    #         time_stamp_s = video_analysis_2.getDuration(frame_no, na.fps)
    #         img_frame = inst.image_data
    #
    #         images.append(img_frame)
    #         time_stamps.append(time_stamp_s)
    #
    #         # cv2.imwrite(os.path.join(frame_folder_path, f"{time_stamp_s}.png"), img_frame)

    image_width, image_height = na.resolution

    num_columns = 3

    # Set the parameter to display every nth image
    n = 16

    # Filter the images based on the parameter
    filtered_images = images[::n]
    filtered_time_stamps = time_stamps[::n]

    # Calculate the number of rows and columns needed
    num_images = len(filtered_images)
    num_rows = int(np.ceil(num_images / num_columns))

    # Create a blank canvas
    canvas_width = num_columns * image_width
    canvas_height = num_rows * image_height + 30 * num_rows  # Additional space for image numbers
    canvas = np.zeros((canvas_height, canvas_width, 3), dtype=np.uint8)

    # Place the images and numbers on the canvas
    for i, image in enumerate(filtered_images):
        row = i // num_columns
        col = i % num_columns
        x_offset = col * image_width
        y_offset = row * image_height + 30 * row
        canvas[y_offset:y_offset + image_height, x_offset:x_offset + image_width] = image

        # Add the image number below each image
        image_number = i + 1
        cv2.putText(canvas, f"{str(int(filtered_time_stamps[i]))} seconds", (x_offset, y_offset + image_height + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)

    cv2.imwrite(os.path.join(base_dir, "frame_timeline.png"), canvas)

    results.generateTimelineAnalysis(na,
                                     save_path=os.path.join(base_dir, "analysis_timeline.png"),
                                     # validation_path="data/Validation2/01.08.2022 segment #0.xlsx",
                                     show=False)


if __name__ == "__main__":
    main()
