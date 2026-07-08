import cv2
import numpy as np

def detect_rings(image):

    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY
    )

    gray = cv2.GaussianBlur(
        gray,
        (9, 9),
        2
    )

    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=60,
        param1=120,
        param2=40,
        minRadius=25,
        maxRadius=55
    )

    ring_list = []

    if circles is not None:

        circles = np.round(
            circles[0, :]
        ).astype("int")

        for circle in circles:

            x = circle[0]
            y = circle[1]
            r = circle[2]

            # Ignore timestamp area
            if y > image.shape[0] * 0.75:
                continue

            # Ignore left edge
            if x < 50:
                continue

            # Ignore right edge
            if x > image.shape[1] - 50:
                continue

            # Ignore tiny circles
            if r < 25:
                continue

            # Ignore huge circles
            if r > 55:
                continue

            ring_list.append(
                (x, y, r)
            )

    return ring_list