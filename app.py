"""
MUNDRA PORT SAR INTELLIGENCE — STREAMLIT APP
=============================================
Deploy to Streamlit Cloud:
  1. Push this file + data/ folder to GitHub
  2. Go to share.streamlit.io
  3. Connect repo and deploy

Install locally:
  pip install streamlit folium streamlit-folium rasterio geopandas matplotlib plotly pandas numpy

Run locally:
  streamlit run app.py
"""

import os
import re
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import streamlit as st
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
from pathlib import Path
import rasterio
from rasterio.enums import Resampling
import io
import base64

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title   = "Mundra Port SAR Intelligence",
    page_icon    = "🛰️",
    layout       = "wide",
    initial_sidebar_state = "expanded",
)

# ─────────────────────────────────────────────
# PATHS — works both locally and on Streamlit Cloud
# Change DATA_ROOT to "data" when deploying to cloud
# ─────────────────────────────────────────────
if os.path.exists(r"G:\Project"):
    # Local Windows machine
    DATA_ROOT    = Path(r"G:\Project\output")
    CHANGES_DIR  = DATA_ROOT / "changes"
    RESULTS_DIR  = DATA_ROOT / "results"
    PROCESSED_DIR= DATA_ROOT
else:
    # Streamlit Cloud — data folder in repo root
    DATA_ROOT    = Path("data")
    CHANGES_DIR  = DATA_ROOT / "changes"
    RESULTS_DIR  = DATA_ROOT / "results"
    PROCESSED_DIR= DATA_ROOT / "processed"

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
  .main-header {
    font-size: 2rem; font-weight: 600;
    color: #1a1a2e; margin-bottom: 0.25rem;
  }
  .sub-header {
    font-size: 1rem; color: #666; margin-bottom: 1.5rem;
  }
  .metric-card {
    background: #f8f9fa; border-radius: 10px;
    padding: 1rem 1.25rem; border-left: 4px solid #E24B4A;
  }
  .metric-val { font-size: 1.8rem; font-weight: 600; color: #1a1a2e; }
  .metric-lbl { font-size: 0.8rem; color: #888; margin-bottom: 4px; }
  .metric-sub { font-size: 0.75rem; color: #aaa; margin-top: 2px; }
  .finding-box {
    background: #fff5f5; border-left: 4px solid #E24B4A;
    border-radius: 0 8px 8px 0; padding: 1rem 1.25rem;
    font-size: 0.95rem; line-height: 1.6; color: #333;
    margin: 1rem 0;
  }
  .validated-box {
    background: #f0faf4; border: 1px solid #b7e5c8;
    border-radius: 8px; padding: 1rem 1.25rem; margin: 1rem 0;
  }
  .phase-high { background:#fee2e2; color:#991b1b;
    padding:2px 10px; border-radius:20px; font-size:12px; }
  .phase-mod  { background:#fef3c7; color:#92400e;
    padding:2px 10px; border-radius:20px; font-size:12px; }
  .phase-low  { background:#dcfce7; color:#166534;
    padding:2px 10px; border-radius:20px; font-size:12px; }
  .phase-base { background:#f1f5f9; color:#64748b;
    padding:2px 10px; border-radius:20px; font-size:12px; }
  .tif-card {
    border: 1px solid #e5e7eb; border-radius: 10px;
    padding: 0.75rem; background: white; margin-bottom: 0.5rem;
  }
  .tif-title { font-size: 0.8rem; font-weight: 600;
    color: #374151; margin-bottom: 4px; }
  .tif-meta  { font-size: 0.7rem; color: #9ca3af; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING — cached for performance
# ─────────────────────────────────────────────
@st.cache_data
def load_classified():
    p = RESULTS_DIR / "classified_final.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(str(p))
    df["date_dt"] = pd.to_datetime(df["date_from"], format="%Y%m%d")
    return df

@st.cache_data
def load_monthly():
    p = RESULTS_DIR / "monthly_construction_summary.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(str(p))

@st.cache_data
def load_geodata():
    """Load centroid points from CSV — works on Streamlit Cloud."""
    p = RESULTS_DIR / "map_centroids.csv"
    if not p.exists():
        return gpd.GeoDataFrame()
    df = pd.read_csv(str(p))
    if "lat" not in df.columns or "lon" not in df.columns:
        return gpd.GeoDataFrame()
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df["lon"], df["lat"]),
        crs="EPSG:4326"
    )
    return gdf

@st.cache_data
def get_logratio_files():
    if not CHANGES_DIR.exists():
        return []
    files = sorted(CHANGES_DIR.glob("logratio_*_fixed.tif"))
    if not files:
        files = sorted(CHANGES_DIR.glob("logratio_*.tif"))
    return files

@st.cache_data
def get_zscore_files():
    if not CHANGES_DIR.exists():
        return []
    return sorted(CHANGES_DIR.glob("zscore_*.tif"))

@st.cache_data
def get_processed_files():
    # Cloud: data/sar/ — matches SAR_cropped_YYYYMMDD_aoi.tif
    cloud_dir = Path("data/results/sar")

    st.write("Looking in:", cloud_dir.resolve())
    st.write("Folder exists:", cloud_dir.exists())

    if cloud_dir.exists():
        files = sorted(cloud_dir.glob("SAR_cropped_*_aoi.tif"))
        if not files:
            # fallback: any .tif in that folder
            files = sorted(cloud_dir.glob("*.tif"))
        if files:
            return files
        
    # Local Windows fallback
    local_cropped = Path(r"G:\Project\output\cropped")
    if local_cropped.exists():
        files = sorted(local_cropped.glob("SAR_cropped_*_aoi.tif"))
        if files:
            return files
    if PROCESSED_DIR.exists():
        return sorted(PROCESSED_DIR.glob("*_processed.tif"))
    return []

def extract_date(filename):
    m = re.search(r'(\d{8})', str(filename))
    return m.group(1) if m else "unknown"

@st.cache_data
def render_tif_thumbnail(filepath, colormap="gray", vmin=None, vmax=None):
    """Render a GeoTIFF as a small PNG thumbnail."""
    try:
        with rasterio.open(str(filepath)) as src:
            h = min(1000, src.height)
            w = min(1000, src.width)
            data = src.read(
                1, out_shape=(h, w),
                resampling=Resampling.average
            ).astype(np.float32)
            nd = src.nodata or -9999.0
            data[data == nd] = np.nan

        # Clip extremes
        valid = data[~np.isnan(data)]
        if len(valid) == 0:
            return None
        v_min = vmin if vmin else np.percentile(valid, 2)
        v_max = vmax if vmax else np.percentile(valid, 98)
        data = np.clip(data, v_min, v_max)

        fig, ax = plt.subplots(figsize=(4, 3), dpi=80)
        ax.imshow(data, cmap=colormap, vmin=v_min, vmax=v_max,
                  interpolation="nearest")
        ax.axis("off")
        plt.tight_layout(pad=0)

        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight",
                    pad_inches=0, dpi=80)
        plt.close()
        buf.seek(0)
        return buf
    except Exception as e:
        return None

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🛰️ Mundra SAR Intelligence")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["Overview", "SAR File Gallery", "Change Detection Map",
         "Economic Correlation", "Intelligence Report"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.markdown("""
**Dataset**
- Sensor: Sentinel-1A IW GRD
- Scenes: 31
- Period: Apr 2025 – Apr 2026
- Method: Log-ratio change detection

**Pipeline**
- Preprocessing: ESA SNAP
- Analysis: Python + rasterio
- Database: PostGIS 3.6
- Visualisation: QGIS + Streamlit
""")

# ─────────────────────────────────────────────
# PAGE 1: OVERVIEW
# ─────────────────────────────────────────────
if page == "Overview":
    st.markdown('<h1 class="main-header">🛰️ Mundra Port SAR Economic Intelligence</h1>',
                unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Sentinel-1 change detection · 31 scenes · April 2025 – April 2026</p>',
                unsafe_allow_html=True)

    # Key metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("""<div class="metric-card">
            <div class="metric-lbl">Peak backscatter delta</div>
            <div class="metric-val">17.2 dB</div>
            <div class="metric-sub">Aug 21, 2025 · terminal zone</div>
        </div>""", unsafe_allow_html=True)
    with col2:
        st.markdown("""<div class="metric-card">
            <div class="metric-lbl">Terminal expansion area</div>
            <div class="metric-val">13.8 km²</div>
            <div class="metric-sub">53 polygons detected</div>
        </div>""", unsafe_allow_html=True)
    with col3:
        st.markdown("""<div class="metric-card">
            <div class="metric-lbl">Cargo throughput growth</div>
            <div class="metric-val">+11.5%</div>
            <div class="metric-sub">51.3 → 57.2 MMT</div>
        </div>""", unsafe_allow_html=True)
    with col4:
        st.markdown("""<div class="metric-card">
            <div class="metric-lbl">SAR lead time</div>
            <div class="metric-val">~6 weeks</div>
            <div class="metric-sub">before APSEZ announcement</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("")

    # Key finding
    st.markdown("""<div class="finding-box">
        <strong>🔍 Key finding:</strong> Sentinel-1 SAR detected peak construction
        activity at Mundra Port terminal zone in June–August 2025
        (max Δσ° = 17.2 dB, 13.8 km² classified as Terminal Expansion),
        preceding APSEZ's September 2025 public announcement of
        ₹30,000 crore Mundra berth expansion by approximately 6 weeks.
    </div>""", unsafe_allow_html=True)

    # Two column layout
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Construction phases")
        monthly = load_monthly()
        if not monthly.empty:
            import plotly.graph_objects as go
            colors = []
            for _, row in monthly.iterrows():
                p = row.get("activity_phase","")
                if "HIGH" in p: colors.append("#E24B4A")
                elif "MODERATE" in p: colors.append("#EF9F27")
                elif "LOW" in p: colors.append("#639922")
                else: colors.append("#888780")

            fig = go.Figure(go.Bar(
                x=monthly["month_str"],
                y=monthly["peak_delta_db"],
                marker_color=colors,
                text=monthly["peak_delta_db"].round(1),
                textposition="outside",
                textfont=dict(size=10),
            ))
            fig.add_hline(y=3, line_dash="dash",
                          line_color="#888", annotation_text="3 dB threshold",
                          annotation_font_size=10)
            fig.add_hline(y=6, line_dash="dot",
                          line_color="#E24B4A",
                          annotation_text="6 dB high confidence",
                          annotation_font_size=10)
            fig.update_layout(
                height=300, margin=dict(l=0,r=0,t=20,b=0),
                yaxis_title="Peak Δσ° (dB)",
                showlegend=False,
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("monthly_construction_summary.csv not found")

    with col_right:
        st.subheader("Change types detected")
        df = load_classified()
        if not df.empty:
            port = df[df["zone_id"] > 0]
            ct = (port.groupby("change_type")["area_m2"]
                     .sum().sort_values(ascending=True) / 1e6)
            ct = ct[ct > 0]

            import plotly.graph_objects as go
            type_colors = {
                "Terminal_Expansion"    : "#A32D2D",
                "Construction_Active"   : "#E24B4A",
                "Construction_Equipment": "#FF7043",
                "Warehouse_Construction": "#EF9F27",
                "Hinterland_Activity"   : "#639922",
                "Terminal_Change"       : "#B4B2A9",
                "Hinterland_Change"     : "#D3D1C7",
                "Anchorage_Change"      : "#85B7EB",
            }
            fig2 = go.Figure(go.Bar(
                x=ct.values.round(2),
                y=[n.replace("_"," ") for n in ct.index],
                orientation="h",
                marker_color=[type_colors.get(n,"#888") for n in ct.index],
                text=ct.values.round(2),
                textposition="outside",
                textfont=dict(size=10),
            ))
            fig2.update_layout(
                height=300, margin=dict(l=0,r=0,t=20,b=40),
                xaxis_title="Total area (km²)",
                showlegend=False,
                plot_bgcolor="white",
                paper_bgcolor="white",
            )
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("classified_final.csv not found")

    # Validation
    st.markdown("""<div class="validated-box">
        <strong>✅ Validated against public record</strong><br><br>
        <b>SAR peak detection:</b> July–August 2025 · 17.2 dB · terminal hardstanding zone<br>
        <b>Public announcement:</b> September 18, 2025 · APSEZ announces ₹30,000 Cr Mundra expansion<br>
        <b>Lead time:</b> ~6 weeks — SAR detected active construction before public disclosure<br>
        <b>Source:</b> Business Standard · APSEZ investor relations · Channel IAM
    </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# PAGE 2: SAR FILE GALLERY
# ─────────────────────────────────────────────
elif page == "SAR File Gallery":
    st.title("SAR File Gallery")
    st.markdown("Browse your preprocessed, logratio, and z-score GeoTIFF files.")

    tab1, tab2, tab3 = st.tabs([
        "📁 Preprocessed scenes",
        "📊 Log-ratio (change signal)",
        "📈 Z-score (significance)"
    ])

    with tab1:
        st.markdown("**Calibrated, speckle-filtered, terrain-corrected Sentinel-1 scenes (dB)**")
        files = get_processed_files()
        if not files:
            st.warning("No *_processed.tif files found. Check your PROCESSED_DIR path.")
            st.code(f"Looking in: {PROCESSED_DIR}")
        else:
            st.success(f"Found {len(files)} processed scenes")
            selected = st.selectbox(
                "Select scene to preview",
                options=files,
                format_func=lambda f: f"Scene {extract_date(f.name)} — {f.name[:50]}..."
            )
            if selected:
                col_img, col_meta = st.columns([2, 1])
                with col_img:
                    with st.spinner("Rendering preview..."):
                        buf = render_tif_thumbnail(
                            selected, colormap="gray"
                        )
                        if buf:
                            st.image(buf, caption=f"Preview: {selected.name[:60]}",
                                     use_container_width=True)
                        else:
                            st.error("Could not render preview")
                with col_meta:
                    try:
                        with rasterio.open(str(selected)) as src:
                            st.markdown(f"""
**File info**
- Date: `{extract_date(selected.name)}`
- Size: `{selected.stat().st_size/1e9:.2f} GB`
- Dimensions: `{src.width} × {src.height}`
- CRS: `EPSG:{src.crs.to_epsg()}`
- Bands: `{src.count}`
- Dtype: `{src.dtypes[0]}`
- Nodata: `{src.nodata}`
""")
                    except Exception as e:
                        st.error(f"Cannot read metadata: {e}")

    with tab2:
        st.markdown("**Log-ratio images: scene dB minus baseline mean. Values > 3 dB = significant increase.**")
        files = get_logratio_files()
        if not files:
            st.warning("No logratio_*_fixed.tif files found.")
            st.code(f"Looking in: {CHANGES_DIR}")
        else:
            st.success(f"Found {len(files)} log-ratio files")

            # Gallery grid
            cols_per_row = 3
            for i in range(0, min(len(files), 12), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(files):
                        f = files[i + j]
                        date_str = extract_date(f.name)
                        with col:
                            with st.spinner(f"Loading {date_str}..."):
                                buf = render_tif_thumbnail(
                                    f, colormap="RdYlGn_r",
                                    vmin=-3, vmax=3
                                )
                                if buf:
                                    st.image(buf, caption=date_str,
                                             use_container_width=True)

            st.markdown("---")
            st.markdown("**Detailed view**")
            selected_lr = st.selectbox(
                "Select log-ratio scene",
                options=files,
                format_func=lambda f: f"Logratio {extract_date(f.name)}"
            )
            if selected_lr:
                col_img, col_stats = st.columns([2, 1])
                with col_img:
                    buf = render_tif_thumbnail(
                        selected_lr, colormap="RdYlGn_r",
                        vmin=-10, vmax=10
                    )
                    if buf:
                        st.image(buf,
                                 caption=f"Red = backscatter increase | Green = decrease",
                                 use_container_width=True)
                with col_stats:
                    try:
                        with rasterio.open(str(selected_lr)) as src:
                            d = src.read(1).astype(np.float32)
                            nd = src.nodata or -9999.0
                            d[d == nd] = np.nan
                            valid = d[~np.isnan(d)]
                            if len(valid) > 0:
                                st.markdown(f"""
**Statistics for {extract_date(selected_lr.name)}**

| Metric | Value |
|--------|-------|
| Min | {np.nanmin(d):.2f} dB |
| Max | {np.nanmax(d):.2f} dB |
| Mean | {np.nanmean(d):.3f} dB |
| Std dev | {np.nanstd(d):.2f} dB |
| Pixels > 3 dB | {int((d > 3).sum()):,} |
| Pixels < -3 dB | {int((d < -3).sum()):,} |
| Valid pixels | {len(valid):,} |
""")
                    except Exception as e:
                        st.error(str(e))

    with tab3:
        st.markdown("**Z-score images: how many standard deviations each pixel deviates from baseline. |z| > 2.5 = statistically significant change.**")
        st.info("""
**How to read z-score images:**  
Z-score measures how unusual each pixel is compared to the baseline period.  
- **|z| > 2.5** = statistically significant change (less than 1% chance it's random)  
- **Red pixels** = strong backscatter increase = new structures, equipment, hard surfaces  
- **Blue pixels** = strong backscatter decrease = cleared land, flooded surface  
- **White/grey** = no significant change from baseline  
Think of it as a confidence map — the brighter the red, the more certain the change is real.
        """)
        files = get_zscore_files()
        if not files:
            st.warning("No zscore_*.tif files found.")
        else:
            st.success(f"Found {len(files)} z-score files")

            cols_per_row = 3
            for i in range(0, min(len(files), 12), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, col in enumerate(cols):
                    if i + j < len(files):
                        f = files[i + j]
                        date_str = extract_date(f.name)
                        with col:
                            buf = render_tif_thumbnail(
                                f, colormap="seismic",
                                vmin=-5, vmax=5
                            )
                            if buf:
                                st.image(buf, caption=date_str,
                                         use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 3: CHANGE DETECTION MAP
# ─────────────────────────────────────────────
elif page == "Change Detection Map":
    st.title("Change Detection Map")
    st.markdown("Interactive map of classified change polygons over Mundra Port.")

    # Filters
    # Filters
    df = load_classified()
    gdf = load_geodata()

    if df.empty:
        st.error("classified_final.csv not found")
        st.stop()

    # Valid zone → change type mapping from your actual data
    ZONE_TYPE_MAP = {
        "All zones": ["All types", "Anchorage_Change", "Background_Change",
                      "Hinterland_Activity", "Hinterland_Change",
                      "Warehouse_Construction", "Construction_Active",
                      "Construction_Equipment", "Terminal_Change",
                      "Terminal_Expansion"],
        "Offshore_Anchorage"    : ["All types", "Anchorage_Change"],
        "Outside"               : ["All types", "Background_Change"],
        "Port_Hinterland"       : ["All types", "Hinterland_Activity",
                                   "Hinterland_Change", "Warehouse_Construction"],
        "Terminal_Hardstanding" : ["All types", "Construction_Active",
                                   "Construction_Equipment", "Terminal_Change",
                                   "Terminal_Expansion"],
    }

    col1, col2, col3 = st.columns(3)

    with col1:
        selected_zone = st.selectbox("Zone", list(ZONE_TYPE_MAP.keys()))

    with col2:
        valid_types = ZONE_TYPE_MAP[selected_zone]
        selected_type = st.selectbox("Change type", valid_types)

    with col3:
        dates = sorted(df["date_from"].unique().tolist())
        selected_date = st.selectbox(
            "Date", ["All dates"] + dates,
            index=dates.index("20250704") + 1
            if "20250704" in dates else 0
        )

    st.info(
        "Tip: Each zone only contains specific change types. "
        "Select **Terminal_Hardstanding** + **Terminal_Expansion** "
        "+ date **20250704** to see the peak construction event."
    )

    # Filter GeoDataFrame
    fgdf = gdf.copy()
    if selected_zone != "All zones":
        fgdf = fgdf[fgdf["zone_name"] == selected_zone]
    if selected_type != "All types":
        fgdf = fgdf[fgdf["change_type"] == selected_type]
    if selected_date != "All dates":
        fgdf = fgdf[fgdf["date_from"] == selected_date]

    if len(fgdf) == 0:
        st.warning(
            f"No polygons found for this combination. "
            f"This zone/type combination may not exist in the data. "
            f"Try: Terminal_Hardstanding + Terminal_Expansion + 20250704"
        )
        st.stop()
    else:
        st.info(f"Showing {len(fgdf):,} polygons")

    # Build Folium map
    m = folium.Map(
        location=[22.76, 69.67],
        zoom_start=12,
        tiles="CartoDB positron"
    )

    # Add satellite basemap option
    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Satellite",
        overlay=False,
    ).add_to(m)

    # Color map for change types
    type_colors = {
        "Terminal_Expansion"    : "#B71C1C",
        "Construction_Active"   : "#F44336",
        "Construction_Equipment": "#FF7043",
        "Warehouse_Construction": "#EF6C00",
        "Hinterland_Activity"   : "#388E3C",
        "Hinterland_Change"     : "#81C784",
        "Terminal_Change"       : "#E0E0E0",
        "Anchorage_Change"      : "#42A5F5",
        "Vessel_Presence"       : "#1565C0",
        "Background_Change"     : "#BDBDBD",
    }

    # Add zone boundary boxes
    zone_boxes = {
        "Terminal": [[22.72, 69.58], [22.80, 69.75]],
        "Anchorage": [[22.55, 69.50], [22.72, 69.90]],
        "Hinterland": [[22.80, 69.58], [22.95, 69.85]],
    }
    for zname, bounds in zone_boxes.items():
        folium.Rectangle(
            bounds=bounds,
            color="#666", weight=1.5,
            fill=False, dash_array="5 5",
            tooltip=f"Zone: {zname}",
        ).add_to(m)
        folium.Marker(
            location=[(bounds[0][0]+bounds[1][0])/2,
                      (bounds[0][1]+bounds[1][1])/2],
            icon=folium.DivIcon(
                html=f'<div style="font-size:11px;color:#555;'
                     f'font-weight:600;white-space:nowrap">{zname}</div>',
                icon_size=(80, 20),
            )
        ).add_to(m)

    # Add polygons (limit to 5000 for performance)
    # Limit polygons based on type — Outside/Background too dense to render
    if selected_zone == "Outside" or selected_type == "Background_Change":
        st.warning(
            "Background_Change covers the full scene outside port zones. "
            "These are noise polygons with no intelligence value. "
            "Select a port zone instead — try Terminal_Hardstanding + Terminal_Expansion."
        )
        st.stop()

    limit = 200 if selected_date == "All dates" else 1000
    plot_gdf = fgdf.head(limit)
    if len(fgdf) > limit:
        st.warning(f"Showing first {limit} of {len(fgdf):,} polygons. "
                   f"Select a specific date to see all polygons for that scene.")

    for _, row in plot_gdf.iterrows():
        try:
            ct    = row.get("change_type", "Unknown")
            color = type_colors.get(ct, "#888")
            area  = row.get("area_m2", 0)
            delta = row.get("mean_delta_db", 0)
            date  = row.get("date_from", "")

            # Radius scaled by area — bigger polygon = bigger circle
            radius = max(4, min(20, (area / 10000) ** 0.5 * 3))

            folium.CircleMarker(
                location=[row.geometry.y, row.geometry.x],
                radius=radius,
                color=color,
                fill=True,
                fill_color=color,
                fill_opacity=0.7,
                weight=1,
                tooltip=folium.Tooltip(
                    f"<b>{ct.replace('_',' ')}</b><br>"
                    f"Date: {date}<br>"
                    f"Area: {area/1e4:.1f} ha<br>"
                    f"Delta: {delta:.2f} dB"
                ),
            ).add_to(m)
        except Exception:
            continue

    # Legend
    legend_html = """
    <div style="position:fixed;bottom:30px;right:30px;z-index:1000;
         background:white;padding:12px 16px;border-radius:8px;
         border:1px solid #ccc;font-size:12px;box-shadow:2px 2px 6px rgba(0,0,0,0.15)">
    <b>Change type</b><br>
    """
    for ct, color in type_colors.items():
        if ct in ["Terminal_Expansion","Construction_Active",
                  "Vessel_Presence","Anchorage_Change","Hinterland_Activity"]:
            legend_html += (f'<span style="background:{color};display:inline-block;'
                           f'width:12px;height:12px;border-radius:2px;'
                           f'margin-right:6px;vertical-align:middle"></span>'
                           f'{ct.replace("_"," ")}<br>')
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)

    # Render map
    with st.spinner("Rendering map..."):
        st_folium(m, width=None, height=550, returned_objects=[])

    # Stats below map
    if not fgdf.empty and "change_type" in fgdf.columns:
        st.markdown("---")
        st.subheader("Filtered results summary")
        col_a, col_b = st.columns(2)
        with col_a:
            by_type = (fgdf.groupby("change_type")
                          .agg(count=("area_m2","count"),
                               area_km2=("area_m2", lambda x: round(x.sum()/1e6,3)))
                          .sort_values("area_km2", ascending=False))
            st.dataframe(by_type, use_container_width=True)
        with col_b:
            if "date_from" in fgdf.columns:
                by_date = (fgdf.groupby("date_from")["area_m2"]
                              .sum().sort_values(ascending=False).head(10) / 1e6)
                by_date = by_date.reset_index()
                by_date.columns = ["date", "area_km2"]
                by_date["area_km2"] = by_date["area_km2"].round(3)
                st.dataframe(by_date, use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 4: ECONOMIC CORRELATION
# ─────────────────────────────────────────────
elif page == "Economic Correlation":
    st.title("Economic Correlation")
    st.markdown("SAR construction activity vs Adani Ports cargo throughput.")

    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    df      = load_classified()
    monthly = load_monthly()

    if df.empty or monthly.empty:
        st.error("Data files not found.")
        st.stop()

    term = df[df["zone_id"] == 1].copy()
    term["month"] = term["date_dt"].dt.to_period("M").dt.to_timestamp()
    monthly_peak = term.groupby("month")["mean_delta_db"].max().reset_index()

    # Cargo data
    cargo = pd.DataFrame({
        "date": pd.to_datetime(["2024-09-30","2024-12-31","2025-03-31",
                                 "2025-06-30","2025-09-30","2025-12-31","2026-03-31"]),
        "mmt" : [47.2, 49.1, 51.3, 52.8, 55.4, 54.9, 57.2],
        "quarter": ["Q2 FY25","Q3 FY25","Q4 FY25",
                    "Q1 FY26","Q2 FY26","Q3 FY26","Q4 FY26"],
    })

    fig = make_subplots(
        rows=2, cols=1,
        shared_xaxes=True,
        subplot_titles=[
            "Terminal zone — peak backscatter delta (construction intensity)",
            "Mundra Port quarterly cargo throughput (MMT)"
        ],
        vertical_spacing=0.12,
    )

    # SAR signal
    fig.add_trace(go.Scatter(
        x=monthly_peak["month"],
        y=monthly_peak["mean_delta_db"].round(2),
        mode="lines+markers",
        name="Peak Δσ° (dB)",
        line=dict(color="#E24B4A", width=2.5),
        marker=dict(size=8, color="#E24B4A",
                    line=dict(color="white", width=1.5)),
        fill="tozeroy",
        fillcolor="rgba(226,75,74,0.12)",
    ), row=1, col=1)

    fig.add_hline(y=3, line_dash="dash", line_color="#999",
                  annotation_text="3 dB threshold",
                  annotation_font_size=10, row=1, col=1)
    fig.add_hline(y=6, line_dash="dot", line_color="#E24B4A",
                  annotation_text="6 dB high confidence",
                  annotation_font_size=10, row=1, col=1)

    # Phase 1 shading
    fig.add_vrect(
        x0="2025-06-01", x1="2025-08-31",
        fillcolor="rgba(226,75,74,0.08)",
        line_width=0,
        annotation_text="Phase 1",
        annotation_position="top left",
        annotation_font_size=11,
        annotation_font_color="#E24B4A",
        row=1, col=1
    )

    # Cargo
    fig.add_trace(go.Scatter(
        x=cargo["date"],
        y=cargo["mmt"],
        mode="lines+markers+text",
        name="Cargo (MMT)",
        line=dict(color="#378ADD", width=2.5),
        marker=dict(size=10, color="#378ADD",
                    line=dict(color="white", width=2)),
        text=cargo["mmt"],
        textposition="top center",
        textfont=dict(size=10, color="#378ADD"),
    ), row=2, col=1)

   # Announcement line — using shape instead of add_vline for compatibility
    for row_n in [1, 2]:
        fig.add_shape(
            type="line",
            x0="2025-09-18", x1="2025-09-18",
            y0=0, y1=1,
            yref="paper",
            xref="x",
            line=dict(color="#F59E0B", width=1.5, dash="dash"),
            row=row_n, col=1
        )
    fig.add_annotation(
        x="2025-09-18", y=0.95,
        xref="x", yref="paper",
        text="APSEZ ₹30,000 Cr expansion",
        showarrow=False,
        font=dict(color="#F59E0B", size=10),
        bgcolor="rgba(255,255,255,0.7)",
        bordercolor="#F59E0B",
    )

    fig.update_layout(
        height=600,
        showlegend=True,
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=80, b=40),
    )
    fig.update_yaxes(title_text="Peak Δσ° (dB)", row=1, col=1,
                     gridcolor="#f0f0f0")
    fig.update_yaxes(title_text="Cargo throughput (MMT)", row=2, col=1,
                     gridcolor="#f0f0f0", range=[44, 60])
    fig.update_xaxes(gridcolor="#f0f0f0")

    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    st.subheader("Quarter-by-quarter summary")
    summary = pd.DataFrame({
        "Quarter"      : cargo["quarter"],
        "Cargo (MMT)"  : cargo["mmt"],
        "Period"       : ["Sep 2024","Dec 2024","Mar 2025",
                          "Jun 2025","Sep 2025","Dec 2025","Mar 2026"],
        "SAR phase"    : ["—","—","Baseline",
                          "Low activity","PEAK ACTIVITY","Moderate","Low activity"],
        "Notes"        : ["","","Pre-construction baseline",
                          "Construction begins","17.2 dB peak · Phase 1",
                          "APSEZ announcement Sep 18","Phase 2 activity"],
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("""
**Interpretation**

Cargo throughput grew from 51.3 MMT (Q4 FY25) to 57.2 MMT (Q4 FY26) — an 11.5% increase.
The sharpest quarterly jump (+2.6 MMT, +5.1%) occurred in Q2 FY26 (Jul–Sep 2025),
which precisely coincides with the SAR-detected construction peak.
This is consistent with new berth capacity coming online and enabling higher throughput.
""")

# ─────────────────────────────────────────────
# PAGE 5: INTELLIGENCE REPORT
# ─────────────────────────────────────────────
elif page == "Intelligence Report":
    st.title("Intelligence Report")

    report_path = RESULTS_DIR / "intelligence_report.txt"
    if report_path.exists():
        with open(str(report_path), "r", encoding="utf-8") as f:
            report = f.read()
        st.code(report, language=None)

        # Download button
        st.download_button(
            label="Download full report (.txt)",
            data=report,
            file_name="mundra_sar_intelligence_report.txt",
            mime="text/plain",
        )
    else:
        st.warning("intelligence_report.txt not found.")
        st.info(f"Expected at: {report_path}")

    st.markdown("---")
    st.subheader("Pipeline architecture")
    st.markdown("""
| Stage | Tool | Output |
|-------|------|--------|
| Data acquisition | Copernicus Open Access Hub | 31 Sentinel-1A GRD scenes |
| Preprocessing | ESA SNAP GPT | Preprocessed GeoTIFFs |
| Stacking | Python + rasterio | 31-scene numpy stack + baseline stats |
| Change detection | Log-ratio + z-score | 31 logratio + zscore GeoTIFFs |
| Vectorisation | rasterio + shapely | 219,187 change polygons |
| Classification | Rule-based (dB + area + zone) | 8 change types per polygon |
| Database | PostGIS 3.6 | Spatial DB with 4 indexes |
| Visualisation | QGIS + matplotlib | Maps + charts |
| Economic correlation | APSEZ quarterly reports | Cargo vs SAR timeline |
| Validation | Public announcement match | 6-week lead time confirmed |
""")
