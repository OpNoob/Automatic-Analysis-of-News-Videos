import cv2
import readText


def get_frame_from_video(video_path, frame_number, save_path=None, show=False, resolution=(640, 360)):
    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not video.isOpened():
        print("Error opening video file")
        return None

    # Set the frame position to the desired frame number
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    # Read the frame at the desired position
    ret, frame = video.read()

    if resolution is not None:
        frame = cv2.resize(frame, resolution, interpolation=cv2.INTER_AREA)

    # Check if the frame was read successfully
    if not ret:
        print(f"Error reading frame {frame_number}")
        return None

    if show:
        # Display the frame
        cv2.imshow("Frame", frame)
        cv2.waitKey(0)

    if save_path is not None:
        cv2.imwrite(save_path, frame)

    # Release the video file and close the OpenCV windows
    video.release()
    cv2.destroyAllWindows()

    return frame


def get_frames_from_video(video_path, frame_number, thresh=100, show=False, resolution=(640, 360)):
    start_frame = frame_number - thresh

    # Open the video file
    video = cv2.VideoCapture(video_path)

    # Check if the video file is opened successfully
    if not video.isOpened():
        print("Error opening video file")
        return None

    # Set the frame position to the desired frame number
    video.set(cv2.CAP_PROP_POS_FRAMES, frame_number - thresh)

    c = 0
    while True:
        c += 1
        # Read the frame at the desired position
        ret, frame = video.read()

        if ret:
            if c > thresh * 2:
                break

            if resolution is not None:
                frame = cv2.resize(frame, resolution, interpolation=cv2.INTER_AREA)

            if show:
                cv2.imshow("Frame", frame)
                if cv2.waitKey(1) == ord("q"):
                    break

            yield frame
        else:
            break

    # Release the video file and close the OpenCV windows
    video.release()
    cv2.destroyAllWindows()


def main():



    image = get_frame_from_video("data/NewsSegments/05.09.2022 segment #2.mp4", 2565, save_path="screenshots/liz_problem.png")
    text = readText.getTextFilter(image)
    print(text)

    image = get_frame_from_video("data/NewsSegments/06.09.2022 segment #0.mp4", 1665,
                                 save_path="screenshots/clyde_problem.png")
    text = readText.getTextFilter(image)
    print(text)

    # frame_no = 2565
    # threshold = 100
    # for i, image in enumerate(get_frames_from_video("data/NewsSegments/05.09.2022 segment #2.mp4", frame_no, show=True, thresh=threshold)):
    #     text = readText.getTextFilter(image)
    #     print(frame_no - threshold + i, text)


if __name__ == "__main__":
    main()
