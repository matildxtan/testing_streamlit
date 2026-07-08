import streamlit as st
import pandas as pd
from pathlib import Path
from PIL import Image
import plotly.express as px

# ==================================================
# PAGE SETUP
# ==================================================
st.set_page_config(
    page_title="Total Duckweed Growth Monitoring Dashboard",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Total Duckweed Growth Monitoring Dashboard")
st.write("This dashboard monitors the total duckweed growth from tub 1 and tub 2 over time.")

# ==================================================
# FOLDER SETUP
# ==================================================
BASE_DIR = Path(__file__).resolve().parent

SEARCH_FOLDERS = [
    BASE_DIR / "output_segmented",
    BASE_DIR / "output"
]

# ==================================================
# FIND CSV FILES
# ================================================++
csv_files = []
for folder in SEARCH_FOLDERS:
    if folder.exists():
        csv_files.extend(list(folder.rglob("*.csv")))

if len(csv_files) == 0:
    st.error("No CSV file found. Please run duckweed2.py first.")
    st.stop()

csv_files = sorted(csv_files, key=lambda x: x.stat().st_mtime, reverse=True)

selected_csv = st.selectbox(
    "Select CSV result file:",
    csv_files,
    format_func=lambda x: str(x.relative_to(BASE_DIR))
)

df = pd.read_csv(selected_csv)
df.columns = [col.strip() for col in df.columns]

# ==================================================
# REQUIRED COLUMNS
# ==================================================
required_columns = [
    "Image_Name",
    "Capture_DateTime",

    "Tub_1_Pond_Pixels",
    "Tub_1_Duckweed_Pixels",
    "Tub_1_Water_Pixels",
    "Tub_1_Duckweed_Coverage_Percentage",
    "Tub_1_Water_Percentage",

    "Tub_2_Pond_Pixels",
    "Tub_2_Duckweed_Pixels",
    "Tub_2_Water_Pixels",
    "Tub_2_Duckweed_Coverage_Percentage",
    "Tub_2_Water_Percentage",

    "Total_Pond_Pixels",
    "Duckweed_Pixels",
    "Duckweed_Coverage_Percentage"
]

missing_columns = [col for col in required_columns if col not in df.columns]

if missing_columns:
    st.error("Some required columns are missing from your CSV:")
    st.write(missing_columns)

    st.write("Your current CSV columns are:")
    st.write(list(df.columns))

    st.info("Run your latest duckweed2.py again, then select the newest CSV.")
    st.stop()

# ==================================================
# CONVERT DATA TYPES
# ==================================================
df["Capture_DateTime"] = pd.to_datetime(
    df["Capture_DateTime"],
    errors="coerce"
)

number_columns = [
    "Tub_1_Pond_Pixels",
    "Tub_1_Duckweed_Pixels",
    "Tub_1_Water_Pixels",
    "Tub_1_Duckweed_Coverage_Percentage",
    "Tub_1_Water_Percentage",

    "Tub_2_Pond_Pixels",
    "Tub_2_Duckweed_Pixels",
    "Tub_2_Water_Pixels",
    "Tub_2_Duckweed_Coverage_Percentage",
    "Tub_2_Water_Percentage",

    "Total_Pond_Pixels",
    "Duckweed_Pixels",
    "Duckweed_Coverage_Percentage"
]

for col in number_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df = df.dropna(subset=["Capture_DateTime"])
df = df.sort_values("Capture_DateTime").reset_index(drop=True)
df["Image_Number"] = range(1, len(df) + 1)

if len(df) == 0:
    st.error("No valid capture date/time found in the CSV.")
    st.stop()

# ==================================================
# DASHBOARD CALCULATED COLUMNS
# ==================================================
df["Combined_Duckweed_Percentage"] = (
    df["Tub_1_Duckweed_Coverage_Percentage"] +
    df["Tub_2_Duckweed_Coverage_Percentage"]
)

df["Combined_Water_Pixels"] = (
    df["Tub_1_Water_Pixels"] +
    df["Tub_2_Water_Pixels"]
)

latest = df.iloc[-1]

# ==================================================
# LATEST KEY RESULTS
# ==================================================
st.subheader("📌 Latest Key Results")

col1, col2, col3, col4 = st.columns(4)

col1.metric(
    "Tub 1 Duckweed",
    f"{latest['Tub_1_Duckweed_Coverage_Percentage']:.2f}%"
)

col2.metric(
    "Tub 2 Duckweed",
    f"{latest['Tub_2_Duckweed_Coverage_Percentage']:.2f}%"
)

col3.metric(
    "Combined Duckweed",
    f"{latest['Combined_Duckweed_Percentage']:.2f}%"
)

col4.metric(
    "Combined Duckweed Pixels",
    f"{int(latest['Duckweed_Pixels']):,}"
)

st.write(f"**Latest image analysed:** {latest['Image_Name']}")
st.write(f"**Captured date/time:** {latest['Capture_DateTime']}")

# ==================================================
# TUB 1, TUB 2, COMBINED BREAKDOWN
# ==================================================
st.subheader("🪴 Tub 1 and Tub 2 Breakdown")

tub_col1, tub_col2, overall_col = st.columns(3)

with tub_col1:
    st.write("### Tub 1")

    st.metric(
        "Duckweed Coverage",
        f"{latest['Tub_1_Duckweed_Coverage_Percentage']:.2f}%"
    )

    st.metric(
        "Water Coverage",
        f"{latest['Tub_1_Water_Percentage']:.2f}%"
    )

    st.metric(
        "Duckweed Pixels",
        f"{int(latest['Tub_1_Duckweed_Pixels']):,}"
    )

    st.metric(
        "Water Pixels",
        f"{int(latest['Tub_1_Water_Pixels']):,}"
    )

    st.metric(
        "Pond Pixels",
        f"{int(latest['Tub_1_Pond_Pixels']):,}"
    )

with tub_col2:
    st.write("### Tub 2")

    st.metric(
        "Duckweed Coverage",
        f"{latest['Tub_2_Duckweed_Coverage_Percentage']:.2f}%"
    )

    st.metric(
        "Water Coverage",
        f"{latest['Tub_2_Water_Percentage']:.2f}%"
    )

    st.metric(
        "Duckweed Pixels",
        f"{int(latest['Tub_2_Duckweed_Pixels']):,}"
    )

    st.metric(
        "Water Pixels",
        f"{int(latest['Tub_2_Water_Pixels']):,}"
    )

    st.metric(
        "Pond Pixels",
        f"{int(latest['Tub_2_Pond_Pixels']):,}"
    )

with overall_col:
    st.write("### Combined Tub 1 + Tub 2")

    st.metric(
        "Combined Duckweed %",
        f"{latest['Combined_Duckweed_Percentage']:.2f}%"
    )

    st.metric(
        "Combined Duckweed Pixels",
        f"{int(latest['Duckweed_Pixels']):,}"
    )

    st.metric(
        "Combined Water Pixels",
        f"{int(latest['Combined_Water_Pixels']):,}"
    )

    st.metric(
        "Total Pond Pixels",
        f"{int(latest['Total_Pond_Pixels']):,}"
    )

# ==================================================
# DUCKWEED LINE CHART + LATEST IMAGE
# ==================================================
st.subheader("📈 Duckweed Growth Over Time")
chart_df = df[
    [
        "Image_Number",
        "Tub_1_Duckweed_Coverage_Percentage",
        "Tub_2_Duckweed_Coverage_Percentage",
        "Combined_Duckweed_Percentage"
    ]
].copy()

chart_df = chart_df.rename(columns={
    "Image_Number": "Image Number",
    "Tub_1_Duckweed_Coverage_Percentage": "Tub 1 Duckweed (%)",
    "Tub_2_Duckweed_Coverage_Percentage": "Tub 2 Duckweed (%)",
    "Combined_Duckweed_Percentage": "Total Duckweed (%)"
})

left_col, right_col = st.columns([1.4, 1])

with left_col:
    fig = px.line(
        chart_df,
        x="Image Number",
        y=[
            "Tub 1 Duckweed (%)",
            "Tub 2 Duckweed (%)",
            "Total Duckweed (%)"
        ],
        markers=True,
        title="Duckweed Growth Over Time"
    )

    fig.update_layout(
        xaxis_title="Image Sequence Number",
        yaxis_title="Duckweed Coverage (%)",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

# ==================================================
# FIND LATEST SEGMENTED IMAGE
# ==================================================
run_folder = selected_csv.parent

image_files = (
    list(run_folder.rglob("*.jpg")) +
    list(run_folder.rglob("*.jpeg")) +
    list(run_folder.rglob("*.png")) +
    list(run_folder.rglob("*.JPG")) +
    list(run_folder.rglob("*.JPEG")) +
    list(run_folder.rglob("*.PNG"))
)

main_images = [
    img for img in image_files
    if "mask" not in img.name.lower()
]

latest_image_stem = Path(str(latest["Image_Name"])).stem.lower()

matched_images = [
    img for img in main_images
    if latest_image_stem in img.stem.lower()
]

if len(matched_images) > 0:
    latest_image_path = sorted(
        matched_images,
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )[0]
elif len(main_images) > 0:
    latest_image_path = sorted(
        main_images,
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )[0]
else:
    latest_image_path = None

with right_col:
    st.write("### 🖼️ Latest Segmented Image")

    if latest_image_path:
        st.image(
            Image.open(latest_image_path),
            caption=latest_image_path.name,
            use_container_width=True
        )
    else:
        st.warning("No segmented image found in this output folder.")
        st.write("Searched folder:")
        st.code(str(run_folder))

# ==================================================
# WATER LINE CHART
# ==================================================
st.subheader("💧 Water Percentage Over Time")

water_chart_df = df[
    [
        "Image_Number",
        "Tub_1_Water_Percentage",
        "Tub_2_Water_Percentage"
    ]
].copy()

water_chart_df = water_chart_df.rename(columns={
    "Image_Number": "Image_Number",
    "Tub_1_Water_Percentage": "Tub 1 Water (%)",
    "Tub_2_Water_Percentage": "Tub 2 Water (%)"
})
left_col, right_col = st.columns([1.4, 1])
water_fig = px.line(
    water_chart_df,
    x="Image_Number",
    y=[
        "Tub 1 Water (%)",
        "Tub 2 Water (%)"
    ],
    markers=True,
    title="Water Percentage Over Time"
)

water_fig.update_layout(
    xaxis_title="Image Sequence Number",
    yaxis_title="Water Percentage (%)",
    hovermode="x unified"
)

st.plotly_chart(water_fig, use_container_width=True)

# ==================================================
# PIXEL COUNT LINE CHART
# ==================================================
st.subheader("📊 Duckweed Pixel Count Over Time")

pixel_chart_df = df[
    [
        "Image_Number",
        "Tub_1_Duckweed_Pixels",
        "Tub_2_Duckweed_Pixels",
        "Duckweed_Pixels"
    ]
].copy()
left_col, right_col = st.columns([1.4, 1])
pixel_chart_df = pixel_chart_df.rename(columns={
    "Image_Number": "Image_Number",
    "Tub_1_Duckweed_Pixels": "Tub 1 Duckweed Pixels",
    "Tub_2_Duckweed_Pixels": "Tub 2 Duckweed Pixels",
    "Duckweed_Pixels": "Combined Duckweed Pixels"
})

pixel_fig = px.line(
    pixel_chart_df,
    x="Image_Number",
    y=[
        "Tub 1 Duckweed Pixels",
        "Tub 2 Duckweed Pixels",
        "Combined Duckweed Pixels"
    ],
    markers=True,
    title="Duckweed Pixel Count Over Time"
)

pixel_fig.update_layout(
    xaxis_title="Image Sequence Number",
    yaxis_title="Duckweed Pixel Count",
    hovermode="x unified"
)

st.plotly_chart(pixel_fig, use_container_width=True)

# ==================================================
# FULL CSV TABLE
# ==================================================
st.subheader("📄 Full CSV Results")

display_df = df.copy()
display_df["Capture_DateTime"] = display_df["Capture_DateTime"].astype(str)

st.dataframe(display_df, use_container_width=True)

# ==================================================
# REFRESH BUTTON
# ==================================================
if st.button("🔄 Refresh Dashboard"):
    st.rerun()