import streamlit as st
import pandas as pd
import numpy as np
import os
import cv2
import time
from datetime import datetime

# 1. PAGE SETUP & DIRECTORY CONFIGURATION
st.set_page_config(page_title="Duckweed Growth Monitor", layout="wide")
st.title("🌿 Real-Time Duckweed Cultivation Dashboard")
st.markdown("Automated image tracking, segmentation parsing, and local database logging.")

IMAGE_DIR = "duckweed_images"
OUTPUT_DIR = "segmented_duckweed"
LIVE_DIR = "live_feed"
DB_FILE = "growth_log.csv"

# Auto-create necessary project folders
for folder in [IMAGE_DIR, OUTPUT_DIR, LIVE_DIR]:
    if not os.path.exists(folder):
        os.makedirs(folder)

# Initialize local CSV database if missing
if not os.path.exists(DB_FILE):
    df_init = pd.DataFrame(columns=['Timestamp', 'Filename', 'Tub 1 Duckweed %', 'Tub 1 Water %', 'Tub 2 Duckweed %', 'Tub 2 Water %'])
    df_init.to_csv(DB_FILE, index=False)


# 2. AUTOMATION & SEGMENTATION LOGIC ENGINE
def process_and_log_images():
    """Scans for files, generates separate segmented copies, logs data, and isolates live stream."""
    if not os.path.exists(IMAGE_DIR):
        return None
        
    all_files = os.listdir(IMAGE_DIR)
    image_paths = [os.path.join(IMAGE_DIR, f) for f in all_files if f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
        
    if not image_paths:
        return None

    # Read the current database log state
    log_df = pd.read_csv(DB_FILE)
    already_processed = set(log_df['Filename'].tolist())
    
    # -------------------------------------------------------------
    # HARDCODED TUNED COORDINATES (From Solution 1 Tuning script)
    left_tub_coords = (742, 611, 528)   # (X, Y, Radius)
    right_tub_coords = (1846, 638, 516)  # (X, Y, Radius)
    # -------------------------------------------------------------
    
    LOWER_GREEN = np.array([25, 35, 35])    
    UPPER_GREEN = np.array([90, 255, 255])
    kernel = np.ones((3, 3), np.uint8)

    # Sort images by modification time (oldest to newest) for true historical continuity in processing and logging
    image_paths.sort(key=os.path.getmtime)
    new_data_logged = False
    
    for img_path in image_paths:
        filename = os.path.basename(img_path)
        
        img = cv2.imread(img_path)
        if img is None:
            continue
            
        height, width, _ = img.shape
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # Segment Duckweed
        global_duckweed = cv2.inRange(hsv, LOWER_GREEN, UPPER_GREEN)
        global_duckweed = cv2.morphologyEx(global_duckweed, cv2.MORPH_OPEN, kernel)
        
        # Tub 1 (Left Mesocosm) Calculations
        mask_tub1 = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask_tub1, (left_tub_coords[0], left_tub_coords[1]), left_tub_coords[2], 255, -1)
        total_pixels_t1 = cv2.countNonZero(mask_tub1)
        dw_mask_t1 = cv2.bitwise_and(global_duckweed, mask_tub1)
        
        # Zero-Division Safety Wrapper for Tub 1
        if total_pixels_t1 > 0:
            dw_pct_t1 = (cv2.countNonZero(dw_mask_t1) / total_pixels_t1) * 100
        else:
            dw_pct_t1 = 0.0
        water_pct_t1 = 100.0 - dw_pct_t1
        
        # Tub 2 (Right Mesocosm) Calculations
        mask_tub2 = np.zeros((height, width), dtype=np.uint8)
        cv2.circle(mask_tub2, (right_tub_coords[0], right_tub_coords[1]), right_tub_coords[2], 255, -1)
        total_pixels_t2 = cv2.countNonZero(mask_tub2)
        dw_mask_t2 = cv2.bitwise_and(global_duckweed, mask_tub2)
        
        # Zero-Division Safety Wrapper for Tub 2
        if total_pixels_t2 > 0:
            dw_pct_t2 = (cv2.countNonZero(dw_mask_t2) / total_pixels_t2) * 100
        else:
            dw_pct_t2 = 0.0
        water_pct_t2 = 100.0 - dw_pct_t2
        
        # --- ENFORCED INDIVIDUAL ARCHIVE GENERATION WITH TEXT OVERLAY ---
        archive_img = img.copy()
        water_mask_archive = cv2.bitwise_and(cv2.bitwise_not(global_duckweed), cv2.bitwise_or(mask_tub1, mask_tub2))
        archive_img[global_duckweed > 0] = [0, 255, 0]        # Green overlay for plants
        archive_img[water_mask_archive > 0] = [255, 50, 50]    # Blue overlay for water
        archive_img[cv2.bitwise_not(cv2.bitwise_or(mask_tub1, mask_tub2)) > 0] = tuple(c // 4 for c in img[cv2.bitwise_not(cv2.bitwise_or(mask_tub1, mask_tub2)) > 0]) # Dim background
        
        # Burn percentage text info directly into the image assets
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Draw text for Tub 1 (Top Left corner of image)
        cv2.rectangle(archive_img, (15, 15), (320, 95), (0, 0, 0), -1)
        cv2.putText(archive_img, "TUB 1 (Left)", (25, 40), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(archive_img, f"Duckweed: {dw_pct_t1:.2f}%", (25, 65), font, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(archive_img, f"Water: {water_pct_t1:.2f}%", (25, 85), font, 0.55, (255, 200, 0), 1, cv2.LINE_AA)
        
        # Draw text for Tub 2 (Top Right corner of image)
        cv2.rectangle(archive_img, (width - 320, 15), (width - 15, 95), (0, 0, 0), -1)
        cv2.putText(archive_img, "TUB 2 (Right)", (width - 310, 40), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(archive_img, f"Duckweed: {dw_pct_t2:.2f}%", (width - 310, 65), font, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(archive_img, f"Water: {water_pct_t2:.2f}%", (width - 310, 85), font, 0.55, (255, 200, 0), 1, cv2.LINE_AA)
        
        archive_path = os.path.join(OUTPUT_DIR, f"split_analysis_{filename}")
        cv2.imwrite(archive_path, archive_img)
        
        # Write to local CSV log only if the frame is completely new
        if filename not in already_processed:
            timestamp_str = datetime.fromtimestamp(os.path.getmtime(img_path)).strftime('%Y-%m-%d %H:%M:%S')
            new_row = pd.DataFrame([{
                'Timestamp': timestamp_str,
                'Filename': filename,
                'Tub 1 Duckweed %': round(dw_pct_t1, 2),
                'Tub 1 Water %': round(water_pct_t1, 2),
                'Tub 2 Duckweed %': round(dw_pct_t2, 2),
                'Tub 2 Water %': round(water_pct_t2, 2)
            }])
            
            log_df = pd.concat([log_df, new_row], ignore_index=True)
            already_processed.add(filename)
            new_data_logged = True

    if new_data_logged:
        log_df.to_csv(DB_FILE, index=False)
        
    # 3. LIVE STREAM VISUAL WRITER (Processes newest image using background atomic swap)
    latest_img_path = max(image_paths, key=os.path.getmtime)
    latest_filename = os.path.basename(latest_img_path)
    
    img_latest = cv2.imread(latest_img_path)
    height, width, _ = img_latest.shape
    hsv_latest = cv2.cvtColor(img_latest, cv2.COLOR_BGR2HSV)
    global_dw_latest = cv2.inRange(hsv_latest, LOWER_GREEN, UPPER_GREEN)
    global_dw_latest = cv2.morphologyEx(global_dw_latest, cv2.MORPH_OPEN, kernel)
    
    mask_t1 = np.zeros((height, width), dtype=np.uint8)
    mask_t2 = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(mask_t1, (left_tub_coords[0], left_tub_coords[1]), left_tub_coords[2], 255, -1)
    cv2.circle(mask_t2, (right_tub_coords[0], right_tub_coords[1]), right_tub_coords[2], 255, -1)
    
    dw_mask_t1 = cv2.bitwise_and(global_dw_latest, mask_t1)
    water_mask_t1 = cv2.bitwise_and(cv2.bitwise_not(dw_mask_t1), mask_t1)
    dw_mask_t2 = cv2.bitwise_and(global_dw_latest, mask_t2)
    water_mask_t2 = cv2.bitwise_and(cv2.bitwise_not(dw_mask_t2), mask_t2)
    
    output_img = img_latest.copy()
    output_img[cv2.bitwise_or(dw_mask_t1, dw_mask_t2) > 0] = [0, 255, 0]
    output_img[cv2.bitwise_or(water_mask_t1, water_mask_t2) > 0] = [255, 50, 50]
    output_img[cv2.bitwise_not(cv2.bitwise_or(mask_t1, mask_t2)) > 0] = tuple(c // 4 for c in img_latest[cv2.bitwise_not(cv2.bitwise_or(mask_t1, mask_t2)) > 0])
    
    # Fetch accurate rows
    latest_row = log_df[log_df['Filename'] == latest_filename].iloc[-1]
    t1_curr_dw = latest_row['Tub 1 Duckweed %']
    t1_curr_wa = latest_row['Tub 1 Water %']
    t2_curr_dw = latest_row['Tub 2 Duckweed %']
    t2_curr_wa = latest_row['Tub 2 Water %']
    
    cv2.rectangle(output_img, (15, 15), (320, 95), (0, 0, 0), -1)
    cv2.putText(output_img, "TUB 1 (Left)", (25, 40), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output_img, f"Duckweed: {t1_curr_dw:.2f}%", (25, 65), font, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(output_img, f"Water: {t1_curr_wa:.2f}%", (25, 85), font, 0.55, (255, 200, 0), 1, cv2.LINE_AA)
    
    cv2.rectangle(output_img, (width - 320, 15), (width - 15, 95), (0, 0, 0), -1)
    cv2.putText(output_img, "TUB 2 (Right)", (width - 310, 40), font, 0.6, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(output_img, f"Duckweed: {t2_curr_dw:.2f}%", (width - 310, 65), font, 0.55, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(output_img, f"Water: {t2_curr_wa:.2f}%", (width - 310, 85), font, 0.55, (255, 200, 0), 1, cv2.LINE_AA)
    
    live_visual_path = os.path.join(LIVE_DIR, "live_stream_feed.jpg")
    temp_visual_path = os.path.join(LIVE_DIR, "temp_live_stream_feed.jpg")
    
    cv2.imwrite(temp_visual_path, output_img)
    try:
        if os.path.exists(temp_visual_path) and os.path.getsize(temp_visual_path) > 0:
            if os.path.exists(live_visual_path):
                os.remove(live_visual_path)
            os.rename(temp_visual_path, live_visual_path)
    except Exception:
        pass
    
    # Return log_df sorted chronologically by actual modification time
    log_df['FileTime'] = log_df['Filename'].apply(lambda f: os.path.getmtime(os.path.join(IMAGE_DIR, f)) if os.path.exists(os.path.join(IMAGE_DIR, f)) else 0)
    log_df = log_df.sort_values(by='FileTime').drop(columns=['FileTime'])
    
    # Explicitly pull metrics from the single newest processed file entry
    true_latest_entry = log_df.iloc[-1]
    return true_latest_entry, log_df


# 3. INTERFACE BUILDER AND DISPLAY LOOP
engine_result = process_and_log_images()

if engine_result is not None:
    current_metrics, complete_history = engine_result
    
    t1_pct = current_metrics['Tub 1 Duckweed %']
    t2_pct = current_metrics['Tub 2 Duckweed %']
    
    # --- AUTOMATED ALERTS GATEWAY ---
    st.subheader("⚠️ System Threshold Status")
    alert_triggered = False
    
    if t1_pct >= 80.0:
        st.error(f"🚨 **HARVEST ALERT:** Tub 1 (Left Mesocosm) has hit {t1_pct}%. Biomass maximum density reached.")
        alert_triggered = True
    if t2_pct >= 80.0:
        st.error(f"🚨 **HARVEST ALERT:** Tub 2 (Right Mesocosm) has hit {t2_pct}%. Biomass maximum density reached.")
        alert_triggered = True
        
    if not alert_triggered:
        st.success("✅ Growth volume nominal. All cultivation zones are operating below the 80% harvest trigger.")
        
    st.write("---")
    
    # --- METRIC SCORECARDS ---
    col1, col2 = st.columns(2)
    with col1:
        st.metric(label="Tub 1 (Left) Duckweed Density", value=f"{t1_pct}%")
    with col2:
        st.metric(label="Tub 2 (Right) Duckweed Density", value=f"{t2_pct}%")
        
    st.write("---")
    
    # --- SPLIT LAYOUT VISUAL ENGINE ---
    col_view, col_timeline = st.columns([1, 1])
    
    with col_view:
        st.subheader("📸 Current Active Mask Feed")
        st.image(os.path.join(LIVE_DIR, "live_stream_feed.jpg"), caption=f"Source Frame: {current_metrics['Filename']}", use_container_width=True)
        
    with col_timeline:
        st.subheader("📈 Chronological Biomass Growth Curve")
        
        # --- FIXED LINE CHART LAYOUT ---
        # Converts filename order into sequential X-axis labels to spread lines horizontally
        chart_df = complete_history.copy()
        chart_df['Sample Reference'] = "Img " + (chart_df.index + 1).astype(str) + " (" + chart_df['Filename'] + ")"
        chart_data = chart_df[['Sample Reference', 'Tub 1 Duckweed %', 'Tub 2 Duckweed %']].set_index('Sample Reference')
        
        st.line_chart(chart_data)
        
        with st.expander("📂 View Local Spreadsheet Database Logs"):
            st.dataframe(complete_history.sort_values(by='Timestamp', ascending=False), use_container_width=True)
else:
    st.warning("Awaiting target data frames. Drop ALL valid images into 'duckweed_images/' to spin up analysis pipelines.")

# 4. FIVE-SECOND INTERVAL POLLING AUTO-REFRESH
time.sleep(5)
st.rerun()