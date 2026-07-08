import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from PIL import Image, ExifTags

# =====================================
# Folder Setup
# =====================================
INPUT_DIR = Path("images")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = Path("output_segmented") / f"run_{timestamp}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =====================================
# Get Date and Time Captured
# =====================================
def get_capture_datetime(image_path):
    try:
        img = Image.open(image_path)
        exif_data = img.getexif()

        if exif_data:
            for tag_id, value in exif_data.items():
                tag_name = ExifTags.TAGS.get(tag_id, tag_id)

                if tag_name == "DateTimeOriginal":
                    return datetime.strptime(
                        value,
                        "%Y:%m:%d %H:%M:%S"
                    ).strftime("%Y-%m-%d %H:%M:%S")

                if tag_name == "DateTime":
                    return datetime.strptime(
                        value,
                        "%Y:%m:%d %H:%M:%S"
                    ).strftime("%Y-%m-%d %H:%M:%S")

    except Exception:
        pass

    modified_time = datetime.fromtimestamp(image_path.stat().st_mtime)
    return modified_time.strftime("%Y-%m-%d %H:%M:%S")


# =====================================
# Thresholds
# =====================================
LOWER_GREEN = np.array([22, 25, 25])
UPPER_GREEN = np.array([95, 255, 255])

LOWER_NEON = np.array([35, 150, 150])
UPPER_NEON = np.array([65, 255, 255])


# =====================================
# Find Images
# =====================================
INPUT_DIR = Path("duckweed_images")
image_files = (
    list(INPUT_DIR.glob("*.[jJ][pP][gG]")) +
    list(INPUT_DIR.glob("*.[pP][nN][gG]")) +
    list(INPUT_DIR.glob("*.[jJ][pP][eE][gG]"))
)

image_files = sorted(image_files)

if not image_files:
    print("No images found. Please place images in the 'images' folder.")
    exit()

all_results = []


# =====================================
# Processing Loop
# =====================================
for image_path in image_files:
    print(f"Processing: {image_path.name}")

    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Failed to read {image_path.name}")
        continue

    capture_datetime = get_capture_datetime(image_path)

    # Resize for consistent mask geometry
    original = cv2.resize(image, (1000, 750))
    height, width = original.shape[:2]

    # -------------------------------------
    # STEP 1: Tub 1 and Tub 2 Masks
    # -------------------------------------
    tub1_mask = np.zeros((height, width), dtype=np.uint8)
    tub2_mask = np.zeros((height, width), dtype=np.uint8)

    # Tub 1 = left tub
    cv2.ellipse(tub1_mask, (285, 260), (255, 215), 0, 0, 360, 255, -1)

    # Tub 2 = right tub
    cv2.ellipse(tub2_mask, (710, 260), (255, 210), 0, 0, 360, 255, -1)

    # Split image so Tub 1 and Tub 2 do not overlap
    split_x = 500
    tub1_mask[:, split_x:] = 0
    tub2_mask[:, :split_x] = 0

    # Remove bottom timestamp strip
    tub1_mask[700:750, :] = 0
    tub2_mask[700:750, :] = 0

    pond_mask = cv2.bitwise_or(tub1_mask, tub2_mask)

    # -------------------------------------
    # STEP 2: Wide HSV and ExG Segmentation
    # -------------------------------------
    hsv = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    hsv_mask = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)

    rgb = cv2.cvtColor(original, cv2.COLOR_BGR2RGB).astype(np.float32)

    R = rgb[:, :, 0]
    G = rgb[:, :, 1]
    B = rgb[:, :, 2]

    excess_green = (2 * G) - R - B

    exg_mask = np.zeros((height, width), dtype=np.uint8)
    exg_mask[
        (excess_green > 5) &
        (G > R * 1.01) &
        (G > B * 1.01) &
        (G > 20)
    ] = 255

    # -------------------------------------
    # STEP 3: Anti-mask for Green Plastic Rings
    # -------------------------------------
    spectral_plastic_mask = np.zeros((height, width), dtype=np.uint8)
    spectral_plastic_mask[
        (G > R * 1.6) &
        (G > 100)
    ] = 255

    neon_hsv_mask = cv2.inRange(hsv, LOWER_NEON, UPPER_NEON)

    plastic_exclusion = cv2.bitwise_or(
        spectral_plastic_mask,
        neon_hsv_mask
    )

    # -------------------------------------
    # STEP 4: Combine, Clean, and Restrict
    # -------------------------------------
    duckweed_mask = cv2.bitwise_and(hsv_mask, exg_mask)

    duckweed_mask = cv2.bitwise_and(
        duckweed_mask,
        cv2.bitwise_not(plastic_exclusion)
    )

    duckweed_mask = cv2.bitwise_and(
        duckweed_mask,
        pond_mask
    )

    kernel = np.ones((2, 2), np.uint8)

    duckweed_mask = cv2.morphologyEx(
        duckweed_mask,
        cv2.MORPH_OPEN,
        kernel,
        iterations=1
    )

    duckweed_mask = cv2.morphologyEx(
        duckweed_mask,
        cv2.MORPH_CLOSE,
        kernel,
        iterations=1
    )

    # -------------------------------------
    # STEP 5: Tub 1 and Tub 2 Calculations
    # -------------------------------------
    tub1_duckweed_mask = cv2.bitwise_and(
        duckweed_mask,
        duckweed_mask,
        mask=tub1_mask
    )

    tub2_duckweed_mask = cv2.bitwise_and(
        duckweed_mask,
        duckweed_mask,
        mask=tub2_mask
    )

    # Tub 1 values
    tub1_pond_pixels = cv2.countNonZero(tub1_mask)
    tub1_duckweed_pixels = cv2.countNonZero(tub1_duckweed_mask)
    tub1_water_pixels = tub1_pond_pixels - tub1_duckweed_pixels

    tub1_duckweed_percent = (
        (tub1_duckweed_pixels / tub1_pond_pixels) * 100
        if tub1_pond_pixels > 0 else 0
    )

    tub1_water_percent = (
        (tub1_water_pixels / tub1_pond_pixels) * 100
        if tub1_pond_pixels > 0 else 0
    )

    # Tub 2 values
    tub2_pond_pixels = cv2.countNonZero(tub2_mask)
    tub2_duckweed_pixels = cv2.countNonZero(tub2_duckweed_mask)
    tub2_water_pixels = tub2_pond_pixels - tub2_duckweed_pixels

    tub2_duckweed_percent = (
        (tub2_duckweed_pixels / tub2_pond_pixels) * 100
        if tub2_pond_pixels > 0 else 0
    )

    tub2_water_percent = (
        (tub2_water_pixels / tub2_pond_pixels) * 100
        if tub2_pond_pixels > 0 else 0
    )

    # Overall calculation = Tub 1 percentage + Tub 2 percentage
    overall_pond_pixels = tub1_pond_pixels + tub2_pond_pixels
    overall_duckweed_pixels = tub1_duckweed_pixels + tub2_duckweed_pixels
    overall_duckweed_percent = tub1_duckweed_percent + tub2_duckweed_percent
   

    # -------------------------------------
    # STEP 6: Save CSV Results
    # -------------------------------------
    all_results.append({
        "Image_Name": image_path.name,
        "Capture_DateTime": capture_datetime,

        "Tub_1_Pond_Pixels": tub1_pond_pixels,
        "Tub_1_Duckweed_Pixels": tub1_duckweed_pixels,
        "Tub_1_Water_Pixels": tub1_water_pixels,
        "Tub_1_Duckweed_Coverage_Percentage": round(tub1_duckweed_percent, 2),
        "Tub_1_Water_Percentage": round(tub1_water_percent, 2),

        "Tub_2_Pond_Pixels": tub2_pond_pixels,
        "Tub_2_Duckweed_Pixels": tub2_duckweed_pixels,
        "Tub_2_Water_Pixels": tub2_water_pixels,
        "Tub_2_Duckweed_Coverage_Percentage": round(tub2_duckweed_percent, 2),
        "Tub_2_Water_Percentage": round(tub2_water_percent, 2),

        "Total_Pond_Pixels": overall_pond_pixels,
        "Duckweed_Pixels": overall_duckweed_pixels,
        "Duckweed_Coverage_Percentage": round(overall_duckweed_percent, 2),
    })

    # -------------------------------------
    # STEP 7: Overlay Graphics and Text
    # -------------------------------------
    highlighted = original.copy()
    highlighted[duckweed_mask > 0] = [0, 255, 0]

    final_output = cv2.addWeighted(
        original,
        0.5,
        highlighted,
        0.5,
        0
    )

    # LEFT WHITE BOX = TUB 1
    cv2.rectangle(final_output, (10, 10), (390, 150), (255, 255, 255), -1)
    cv2.rectangle(final_output, (10, 10), (390, 150), (0, 0, 0), 2)

    cv2.putText(final_output, "Tub 1", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2)
    cv2.putText(final_output, f"Duckweed: {tub1_duckweed_percent:.2f}% ({tub1_duckweed_pixels:,} px)", (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2)
    cv2.putText(final_output, f"Water: {tub1_water_percent:.2f}% ({tub1_water_pixels:,} px)", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2)
    cv2.putText(final_output, f"Pond Pixels: {tub1_pond_pixels:,}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2)

    # RIGHT WHITE BOX = TUB 2
    cv2.rectangle(final_output, (600, 10), (990, 150), (255, 255, 255), -1)
    cv2.rectangle(final_output, (600, 10), (990, 150), (0, 0, 0), 2)

    cv2.putText(final_output, "Tub 2", (610, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (0, 0, 0), 2)
    cv2.putText(final_output, f"Duckweed: {tub2_duckweed_percent:.2f}% ({tub2_duckweed_pixels:,} px)", (610, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2)
    cv2.putText(final_output, f"Water: {tub2_water_percent:.2f}% ({tub2_water_pixels:,} px)", (610, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2)
    cv2.putText(final_output, f"Pond Pixels: {tub2_pond_pixels:,}", (610, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.58, (0, 0, 0), 2)

    # SEPARATE WHITE BOX = CAPTURE TIME + OVERALL
    cv2.rectangle(final_output, (300, 615), (720, 690), (255, 255, 255), -1)
    cv2.rectangle(final_output, (300, 615), (720, 690), (0, 0, 0), 2)

    cv2.putText(final_output, f"Captured: {capture_datetime}", (315, 640), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 2)
    cv2.putText(final_output, f"Overall Duckweed: {overall_duckweed_percent:.2f}% ({overall_duckweed_pixels:,} px)", (315, 665), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (0, 0, 0), 2)
    
    # Label tubs only, no circles
    cv2.putText(final_output, "Tub 1", (220, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
    cv2.putText(final_output, "Tub 1", (220, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    cv2.putText(final_output, "Tub 2", (650, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 3)
    cv2.putText(final_output, "Tub 2", (650, 520), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2)

    # Save image
    out_path = OUTPUT_DIR / f"{image_path.stem}_segmented.jpg"
    cv2.imwrite(str(out_path), final_output)

# =====================================
# STEP 8: Save CSV Data
# =====================================
if all_results:
    df = pd.DataFrame(all_results)
    csv_path = OUTPUT_DIR / "FYP_Duckweed_Results.csv"
    df.to_csv(csv_path, index=False)

    print("\n--- FYP Segmentation Complete ---")
    print(f"Processed {len(all_results)} images.")
    print(f"Results successfully saved to Excel/CSV format at: {csv_path}")
    print(f"Segmented images saved in: {OUTPUT_DIR}")