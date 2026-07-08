import cv2
import numpy as np

def detect_duckweed(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_green = np.array([25, 40, 40])

    upper_green = np.array([95, 255, 255])

    mask = cv2.inRange(
        hsv,
        lower_green,
        upper_green
    )

    return mask