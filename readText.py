import math

import numpy as np
from pytesseract import pytesseract
import cv2 as cv

# pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
pytesseract.tesseract_cmd = r"Tesseract-OCR\tesseract.exe"

# https://muthu.co/all-tesseract-ocr-options/

"""

"""


def iterateBoxesFromMask(mask, image_line, horizontal=True, yield_roi=True):
    """
    :param mask: Binary mask
    :param image_line: Opencv image
    :param horizontal: Marks if line should either be horizontal or vertical
    :param yield_roi: Yields region of interest.  If this was False, then this would yield the x, y, w, h locations
    :return: The image for the region of interest
    """

    contours, hierarchy = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
        approx = cv.approxPolyDP(cnt, 0.01 * cv.arcLength(cnt, True), True)
        if len(approx) == 4:
            x, y, w, h = cv.boundingRect(cnt)
            # Width must be 4* greater than height
            if (horizontal and w > h * 4) or (not horizontal and h > w * 4):
                ROI = image_line[y:y + h, x:x + w]
                if yield_roi:
                    yield ROI
                else:
                    yield x, y, w, h


def getPercWhite(mask_image):
    n_white_pix = np.sum(mask_image == 255)
    size = mask_image.shape[0] * mask_image.shape[1]
    perc = n_white_pix / size
    return perc


def isImageColor(image_hsv, colors: list[tuple[tuple[int, int, int], tuple[int, int, int]]],
                 kernel_size: tuple[int, int] = (5, 5), dilate_iterations: int = 2, threshold: float = 0.6) -> bool:
    mask = cv.inRange(image_hsv, colors[0][0], colors[0][1])  # Start with first
    for next_mask in colors[1:]:  # Add all other masks
        mask |= cv.inRange(image_hsv, next_mask[0], next_mask[1])
    # cv.imshow("mask", mask)

    # Dilution
    kernel = np.ones(kernel_size, np.uint8)
    mask_diluted = cv.dilate(mask, kernel, iterations=dilate_iterations)  # iterations=2
    # cv.imshow("mask_diluted", mask_diluted)
    # cv.waitKey()

    if getPercWhite(mask_diluted) > threshold:
        return True
    return False


def getTextFilter(image_bgr):
    # Ensure image is in the same ratio
    assert image_bgr.shape[0] / image_bgr.shape[1] == 360/640  # Image must be in the same ratio of 640 x 360 (width x height)

    # range of white [HSV]
    lower_white_color = (0, 0, 230)
    upper_white_color = (255, 18, 255)

    # range of red [HSV]
    lower_red_color_1 = (0, 200, 70)
    upper_red_color_1 = (10, 255, 255)
    lower_red_color_2 = (170, 200, 70)
    upper_red_color_2 = (180, 255, 255)

    # Range of red line at end of text box [HSV]
    lower_red_line_color_1 = (0, 0, 170)
    upper_red_line_color_1 = (30, 10, 255)
    # lower_red_line_color_2 = (255, 0, 0)
    # upper_red_line_color_2 = (255, 255, 255)

    custom_config = r'-c tessedit_char_whitelist="ABCDEFGHIJKLMNOPQRSTUVWXYZ " --psm 7'

    text_final = None

    # print(image_bgr.shape)
    video_height, video_width, _ = image_bgr.shape

    # Applying ratios
    up_down_limit_first = (round(580/720 * video_height), round(610/720 * video_height))
    up_down_limit_second = (round(615/720 * video_height), round(645/720 * video_height))
    left_right_limit_check = (round(100/1280 * video_width), round(200/1280 * video_width))  # 300
    left_right_limit = (round(100/1280 * video_width), round(600/1280 * video_width))

    # For video (720, 1280, 3):
    # up_down_limit_first = (-140, -110)
    # up_down_limit_second = (-105, -75)
    # left_right_limit_check = (100, 300)
    # left_right_limit = (100, 600)

    # Check first line if name would be found
    first_line_check = image_bgr[up_down_limit_first[0]:up_down_limit_first[1], left_right_limit_check[0]:left_right_limit_check[1]]
    first_line_check_hsv = cv.cvtColor(first_line_check, cv.COLOR_BGR2HSV)  # change to hsv
    first_line_isRed = isImageColor(first_line_check_hsv, [(lower_red_color_1, upper_red_color_1), (lower_red_color_2, upper_red_color_2)])

    # cv.imshow("first_line_check", first_line_check)
    # cv.waitKey()

    if first_line_isRed:
        return
    else:  # Check second line as well
        second_line_check = image_bgr[up_down_limit_second[0]:up_down_limit_second[1],
                            left_right_limit_check[0]:left_right_limit_check[1]]
        second_line_check_hsv = cv.cvtColor(second_line_check, cv.COLOR_BGR2HSV)  # change to hsv
        second_line_isRed = isImageColor(second_line_check_hsv, [(lower_red_color_1, upper_red_color_1),
                                                               (lower_red_color_2, upper_red_color_2)])
        if second_line_isRed:
            return

    # Read first line if it is white, else read second line
    first_line_isWhite = isImageColor(first_line_check_hsv,
                                    [(lower_white_color, upper_white_color)])
    if first_line_isWhite:
        first_line = image_bgr[up_down_limit_first[0]:up_down_limit_first[1], left_right_limit[0]:left_right_limit[1]]
        image_lines = [first_line]
    else:
        second_line = image_bgr[up_down_limit_second[0]:up_down_limit_second[1],
                      left_right_limit[0]:left_right_limit[1]]
        image_lines = [second_line]

    for image_line in image_lines:
        # image_line = image_bgr[-140:-110]  # Selecting line where name might appear
        # image_line = image_bgr[-105:-78]  # Selecting line where name might appear
        # cv.imshow("lint", image_line)
        # cv.waitKey()

        # Closing image
        kernel = np.ones((7, 7), np.uint8)
        img_filtered = cv.dilate(image_line, kernel, iterations=4)  # iterations=2
        img_filtered = cv.erode(img_filtered, kernel, iterations=4)
        # cv.imshow("img_filtered", img_filtered)

        # Thresholding white pixels
        img_hsv = cv.cvtColor(img_filtered, cv.COLOR_BGR2HSV)
        mask = cv.inRange(img_hsv, lower_white_color, upper_white_color)
        # cv.imshow("mask", mask)
        # cv.waitKey()

        # Remove any small masks
        kernel = np.ones((5, 5), np.uint8)
        mask = cv.erode(mask, kernel, iterations=7)
        mask = cv.dilate(mask, kernel, iterations=7)

        # Cut off at first red line
        mask_red = cv.inRange(image_line, lower_red_line_color_1, upper_red_line_color_1)
        for x, y, w, h in iterateBoxesFromMask(mask_red, image_line, horizontal=False, yield_roi=False):
            if h == mask_red.shape[0]:  # Checks if red line goes throughout the image
                left_most = x - math.ceil(w/2)
                mask[:, left_most:] = 0


        # Applying mask
        # img_masked = cv.bitwise_and(image_line, image_line, mask=mask)
        # cv.imshow("img_masked", img_masked)
        # cv.waitKey()

        # cv.imshow("mask", mask)
        # cv.waitKey()

        # Contours (Get text from a box mask)
        texts = []
        for roi in iterateBoxesFromMask(mask, image_line):
            # cv.imshow("roi", roi)
            # cv.waitKey()
            text = pytesseract.image_to_string(roi, config=custom_config)
            if text is not None:
                text = text.replace("\n", "")
                texts.append(text)

        for text in texts:
            text_len = len(text)
            if 10 < text_len < 30:  # Must have at least 10 characters
                if text.count(" ") < 5:  # Must have at least less than 5 spaces
                    text_final = text
                    break

        # cv.waitKey()
        if not (text_final is None or text_final == ""):  # If text is found, stop search
            # print("Break")
            break

    # print(text_final)
    return text_final


def test_video():
    cap = cv.VideoCapture('dat/extract_full_news/video_1.mp4')
    cap.set(cv.CAP_PROP_POS_FRAMES, 5400)
    while cap.isOpened():
        ret, frame = cap.read()

        texts = getTextFilter(frame)
        print(texts)

        cv.imshow('frame', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break


def test_image():
    # image = cv.imread("dat/known_faces/video_2/CLIFTON GRIMA.png")
    # image = cv.imread("screenshots/Frame #2575.0.png")
    image = cv.imread("screenshots/liz_problem.png")
    # cv.imshow("test", image)
    # cv.waitKey(0)

    text = getTextFilter(image)
    print(text)


# test_image()
# test_video()

if __name__ == "__main__":
    test_image()
