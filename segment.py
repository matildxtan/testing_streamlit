import cv2
import numpy as np
import os
import csv
import subprocess
from datetime import datetime

from ring_detection import detect_rings
from upload_drive import upload_file

SEGMENTED_FOLDER_ID = "1LARIqUb-xXrA-vxVuNwZP-OGixt18Tu8"

RESULTS_FOLDER_ID = "1sFEFZv6rifUC0vsxul0mox7V3pRPURhA"

GRAPH_FOLDER_ID = "1KCOoTZGGg5O7rSEOOMsr0Br3NDl6rAkd"

INPUT_FOLDER = "duckweed_images"
OUTPUT_FOLDER = "segmented_images"

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

csv_file = open(
    "results.csv",
    "w",
    newline=""
)

writer = csv.writer(csv_file)

writer.writerow([
    "Date",
    "Time",
    "Image",
    "Duckweed %",
    "Water %"
])

for filename in os.listdir(INPUT_FOLDER):

    if filename.endswith(".jpg") or filename.endswith(".png"):

        image_path = os.path.join(
            INPUT_FOLDER,
            filename
        )

        image = cv2.imread(
            image_path
        )

        width = 1000

        height = int(
            image.shape[0] *
            width /
            image.shape[1]
        )

        image = cv2.resize(
            image,
            (width, height)
        )

        hsv = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2HSV
        )

        # Duckweed detection
        lower_green = np.array(
            [25, 40, 40]
        )

        upper_green = np.array(
            [95, 255, 255]
        )

        duckweed_mask = cv2.inRange(
            hsv,
            lower_green,
            upper_green
        )

        # Water detection
        lower_water = np.array(
            [0, 0, 0]
        )

        upper_water = np.array(
            [180, 255, 80]
        )

        water_mask = cv2.inRange(
            hsv,
            lower_water,
            upper_water
        )

        kernel = np.ones(
            (3, 3),
            np.uint8
        )

        duckweed_mask = cv2.morphologyEx(
            duckweed_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        water_mask = cv2.morphologyEx(
            water_mask,
            cv2.MORPH_OPEN,
            kernel
        )

        duckweed_pixels = cv2.countNonZero(
            duckweed_mask
        )

        water_pixels = cv2.countNonZero(
            water_mask
        )

        total_pixels = (
            duckweed_pixels +
            water_pixels
        )

        if total_pixels > 0:

            duckweed_percent = round(
                (
                    duckweed_pixels /
                    total_pixels
                ) * 100,
                2
            )

            water_percent = round(
                (
                    water_pixels /
                    total_pixels
                ) * 100,
                2
            )

        else:

            duckweed_percent = 0
            water_percent = 0

        result = image.copy()

        # Green overlay = duckweed
        result[
            duckweed_mask > 0
        ] = [0, 255, 0]

        # Blue overlay = water
        result[
            water_mask > 0
        ] = [255, 0, 0]

        # Detect rings
        rings = detect_rings(
            image
        )

        rings = sorted(
            rings,
            key=lambda ring: (
                ring[1],
                ring[0]
            )
        )

        ring_number = 1

        for ring in rings:

            x = ring[0]
            y = ring[1]
            r = ring[2]

            cv2.circle(
                result,
                (x, y),
                r,
                (0, 0, 255),
                2
            )

            cv2.putText(
                result,
                "R" + str(ring_number),
                (x - 15, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            ring_number += 1

        # Top-left information box
        cv2.rectangle(
            result,
            (10, 10),
            (250, 100),
            (255, 255, 255),
            -1
        )

        cv2.rectangle(
            result,
            (10, 10),
            (250, 100),
            (0, 0, 0),
            2
        )

        cv2.putText(
            result,
            "Duckweed: " +
            str(duckweed_percent) +
            "%",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

        cv2.putText(
            result,
            "Water: " +
            str(water_percent) +
            "%",
            (20, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2
        )

        output_path = os.path.join(
            OUTPUT_FOLDER,
            filename
        )

        cv2.imwrite(
            output_path,
            result
        )

        upload_file(
            output_path,
            SEGMENTED_FOLDER_ID
        )

        now = datetime.now()

        current_date = now.strftime(
            "%Y-%m-%d"
        )

        current_time = now.strftime(
            "%H:%M:%S"
        )

        writer.writerow([
            current_date,
            current_time,
            filename,
            duckweed_percent,
            water_percent
        ])

        print(filename)
        print(
            "Duckweed:",
            duckweed_percent,
            "%"
        )
        print(
            "Water:",
            water_percent,
            "%"
        )
        print(
            "Rings Detected:",
            len(rings)
        )
        print("---------------------")

csv_file.close()

upload_file(
    "results.csv",
    RESULTS_FOLDER_ID
)

print(
    "Generating growth graph..."
)

subprocess.run(
    ["python", "plot_graph.py"],
    check=True
)

print(
    "Growth graph generated successfully."
)

upload_file(
    "duckweed_growth_curve.png",
    GRAPH_FOLDER_ID
)

print(
    "Growth graph uploaded to Google Drive."
)

print("Done")