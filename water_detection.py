import cv2
import numpy as np

def detect_water(image):

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    lower_water = np.array([0, 0, 0])

    upper_water = np.array([180, 120, 120])

    mask = cv2.inRange(
        hsv,
        lower_water,
        upper_water
    )

    return mask