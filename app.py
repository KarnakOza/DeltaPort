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
# PATHS
# ─────────────────────────────────────────────
if os.path.exists(r"G:\Project"):
    DATA_ROOT    = Path(r"G:\Project\output")
    CHANGES_DIR  = DATA_ROOT / "changes"
    RESULTS_DIR  = DATA_ROOT / "results"
    PROCESSED_DIR= DATA_ROOT
else:
    DATA_ROOT    = Path("data")
    CHANGES_DIR  = DATA_ROOT / "changes"
    RESULTS_DIR  = DATA_ROOT / "results"
    PROCESSED_DIR= DATA_ROOT / "processed"

# ─────────────────────────────────────────────
# DARK THEME PLOTLY HELPER — call this on every fig
# ─────────────────────────────────────────────
DARK_BG   = "#0a0f0a"
GRID_COL  = "rgba(0,255,70,0.12)"
GREEN     = "#00ff46"
GREEN_DIM = "rgba(0,255,70,0.5)"
FONT_FAM  = "Share Tech Mono, Courier New, monospace"

def apply_dark_theme(fig, rows=1):
    """Apply consistent dark green terminal theme to any plotly figure."""
    fig.update_layout(
        plot_bgcolor  = DARK_BG,
        paper_bgcolor = DARK_BG,
        font          = dict(color=GREEN, family=FONT_FAM, size=11),
        legend        = dict(
            bgcolor     = "rgba(0,0,0,0.5)",
            bordercolor = GREEN_DIM,
            borderwidth = 1,
            font        = dict(color=GREEN, size=10),
        ),
        title_font    = dict(color=GREEN, family=FONT_FAM),
    )
    # Update ALL axes (works for subplots too)
    fig.update_xaxes(
        gridcolor    = GRID_COL,
        linecolor    = GREEN_DIM,
        tickfont     = dict(color=GREEN, size=10),
        title_font   = dict(color=GREEN_DIM, size=10),
        zerolinecolor= GREEN_DIM,
    )
    fig.update_yaxes(
        gridcolor    = GRID_COL,
        linecolor    = GREEN_DIM,
        tickfont     = dict(color=GREEN, size=10),
        title_font   = dict(color=GREEN_DIM, size=10),
        zerolinecolor= GREEN_DIM,
    )
    # Fix subplot titles (they're annotations internally)
    for ann in fig.layout.annotations:
        ann.font = dict(color=GREEN, size=11, family=FONT_FAM)
    return fig

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Share+Tech+Mono&display=swap');

.stApp {
    background-color: #0a0f0a !important;
}

section[data-testid="stSidebar"] {
    background-color: #060c06 !important;
    border-right: 1px solid rgba(0,255,70,0.15) !important;
}

section[data-testid="stSidebar"] * {
    color: rgba(0,255,70,0.8) !important;
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
    font-size: 12px !important;
    letter-spacing: 0.05em !important;
}

h1, h2, h3 {
    font-family: 'Share Tech Mono', 'Courier New', monospace !important;
    color: #00ff46 !important;
    letter-spacing: 0.02em !important;
}

p, li, td, th, label {
    font-family: -apple-system, 'Segoe UI', sans-serif !important;
    color: rgba(210,255,220,0.85) !important;
}

.stButton button {
    background: rgba(0,255,70,0.08) !important;
    border: 1px solid #00ff46 !important;
    color: #00ff46 !important;
    font-family: 'Share Tech Mono', monospace !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 0 !important;
}

.stButton button:hover {
    background: rgba(0,255,70,0.2) !important;
}

.stSelectbox > div > div {
    background: #060c06 !important;
    border: 1px solid rgba(0,255,70,0.3) !important;
    color: #00ff46 !important;
    border-radius: 0 !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: rgba(0,255,70,0.5) !important;
    font-family: 'Share Tech Mono', monospace !important;
    border-radius: 0 !important;
    border-bottom: 2px solid transparent !important;
    letter-spacing: 0.08em !important;
    font-size: 11px !important;
    text-transform: uppercase !important;
}

.stTabs [aria-selected="true"] {
    color: #00ff46 !important;
    border-bottom: 2px solid #00ff46 !important;
    background: rgba(0,255,70,0.05) !important;
}

.stDataFrame, .stTable {
    border: 1px solid rgba(0,255,70,0.2) !important;
}

.metric-card {
    background: rgba(0,255,70,0.04);
    border: 1px solid rgba(0,255,70,0.25);
    padding: 1rem;
}
.metric-val {
    font-size: 1.8rem;
    font-weight: 700;
    color: #00ff46;
    font-family: 'Share Tech Mono', monospace;
}
.metric-lbl {
    font-size: 9px;
    color: rgba(0,255,70,0.45);
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 4px;
}
.metric-sub {
    font-size: 9px;
    color: rgba(0,255,70,0.35);
    margin-top: 3px;
}
.finding-box {
    border: 1px solid rgba(0,255,70,0.35);
    border-left: 3px solid #00ff46;
    background: rgba(0,255,70,0.04);
    padding: 1rem;
    color: rgba(0,255,70,0.9);
    font-size: 12px;
    line-height: 1.7;
    margin: 1rem 0;
}
.validated-box {
    border: 1px solid rgba(0,255,70,0.25);
    background: rgba(0,255,70,0.03);
    padding: 1rem;
    color: rgba(0,255,70,0.7);
    font-size: 11px;
    line-height: 1.8;
}
.phase-high {
    background: rgba(255,68,68,0.12);
    color: #ff4444;
    border: 1px solid #ff4444;
    padding: 1px 8px;
    font-size: 10px;
    font-family: monospace;
}
.phase-mod {
    background: rgba(255,170,0,0.1);
    color: #ffaa00;
    border: 1px solid #ffaa00;
    padding: 1px 8px;
    font-size: 10px;
    font-family: monospace;
}
.phase-low {
    background: rgba(0,204,51,0.1);
    color: #00cc33;
    border: 1px solid #00cc33;
    padding: 1px 8px;
    font-size: 10px;
    font-family: monospace;
}
.phase-base {
    background: rgba(0,255,70,0.05);
    color: rgba(0,255,70,0.35);
    border: 1px solid rgba(0,255,70,0.2);
    padding: 1px 8px;
    font-size: 10px;
    font-family: monospace;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# DATA LOADING
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
    cloud_dir = Path("data/results/sar")
    if cloud_dir.exists():
        files = sorted(cloud_dir.glob("SAR_cropped_*_aoi.tif"))
        if not files:
            files = sorted(cloud_dir.glob("*.tif"))
        if files:
            return files
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
    try:
        with rasterio.open(str(filepath)) as src:
            h = min(300, src.height)
            w = min(300, src.width)
            data = src.read(
                1, out_shape=(h, w),
                resampling=Resampling.average
            ).astype(np.float32)
            nd = src.nodata or -9999.0
            data[data == nd] = np.nan

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
    except Exception:
        return None

# ─────────────────────────────────────────────
# TOP NAVIGATION (replaces the old sidebar radio nav)
# Change Detection Map is now the #2 stop — right after Overview —
# since that's the page with the actual intelligence payoff.
# ─────────────────────────────────────────────
NAV_ORDER = ["Overview", "Change Detection Map", "Economic Correlation",
             "SAR File Gallery", "Intelligence Report"]

if "page" not in st.session_state:
    st.session_state.page = "Overview"

def goto(p):
    st.session_state.page = p

st.markdown("""
<div style="display:flex;align-items:baseline;justify-content:space-between;
            border-bottom:1px solid rgba(0,255,70,0.18);padding-bottom:0.6rem;
            margin-bottom:0.8rem">
    <div style="font-family:'Share Tech Mono',monospace;font-size:15px;
                color:#00ff46;letter-spacing:0.04em">🛰️ DELTAPORT</div>
    <div style="font-size:10px;color:rgba(0,255,70,0.4)">Mundra Port · Sentinel-1 SAR</div>
</div>
""", unsafe_allow_html=True)

nav_cols = st.columns(len(NAV_ORDER))
for i, label in enumerate(NAV_ORDER):
    with nav_cols[i]:
        is_active = st.session_state.page == label
        st.button(
            label,
            key=f"nav_{label}",
            use_container_width=True,
            type="primary" if is_active else "secondary",
            on_click=goto,
            args=(label,),
        )

page = st.session_state.page

# ─────────────────────────────────────────────
# PAGE 1: OVERVIEW
# ─────────────────────────────────────────────
if page == "Overview":
    st.markdown("""
    <div style="border-bottom:1px solid rgba(0,255,70,0.2);
                padding-bottom:1rem;margin-bottom:1.5rem">
        <div style="font-size:9px;color:rgba(0,255,70,0.4);
                    letter-spacing:0.2em;margin-bottom:6px">
            // DELTAPORT INTELLIGENCE SYSTEM · ACTIVE
        </div>
        <h1 style="font-size:2.5rem;margin:0;line-height:1">DELTAPORT</h1>
        <div style="font-size:10px;color:rgba(0,255,70,0.45);
                    letter-spacing:0.15em;margin-top:6px">
            SATELLITE RADAR ANALYSIS · MUNDRA PORT · 31 SCENES · APR 2025–APR 2026
        </div>
    </div>
    """, unsafe_allow_html=True)

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

    finding_col, cta_col = st.columns([3, 1])
    with finding_col:
        st.markdown("""<div class="finding-box">
            <strong>🔍 Key finding:</strong> Sentinel-1 SAR detected peak construction
            activity at Mundra Port terminal zone in June–August 2025
            (max Δσ° = 17.2 dB, 13.8 km² classified as Terminal Expansion),
            preceding APSEZ's September 2025 public announcement of
            ₹30,000 crore Mundra berth expansion by approximately 6 weeks.
        </div>""", unsafe_allow_html=True)
    with cta_col:
        st.markdown("<div style='height:1.1rem'></div>", unsafe_allow_html=True)
        st.button("See it on the map →", use_container_width=True,
                   type="primary", on_click=goto, args=("Change Detection Map",))
        st.caption("219,187 change polygons, filterable by zone and date.")

    # ── AOI MAP ──────────────────────────────────
    st.markdown("""
    <div style="font-size:9px;color:rgba(0,255,70,0.4);letter-spacing:0.2em;
                margin:1.2rem 0 0.4rem 0">
        // AREA OF INTEREST · MUNDRA PORT · 22.76°N 69.67°E · UTM 43N
    </div>""", unsafe_allow_html=True)

    expand_map = st.toggle("Expand map", value=False, key="expand_aoi_map")

    if expand_map:
        map_col = st.container()
        info_col = None
        map_height = 640
    else:
        map_col, info_col = st.columns([3, 1])
        map_height = 460

    with map_col:
        aoi_map = folium.Map(
            location=[22.76, 69.67],
            zoom_start=12,
            tiles=None,
        )

        # Dark basemap
        folium.TileLayer(
            tiles="CartoDB dark_matter",
            name="Dark",
            attr="CartoDB",
        ).add_to(aoi_map)

        # Satellite layer toggle
        folium.TileLayer(
            tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
            attr="Google Satellite",
            name="Satellite",
            overlay=False,
        ).add_to(aoi_map)

        # AOI bounding box
        aoi_bounds = [[22.60465319, 69.46046562], [22.93724805, 69.81863230]]
        folium.Rectangle(
            bounds=aoi_bounds,
            color="#00ff46",
            weight=2,
            fill=True,
            fill_color="#00ff46",
            fill_opacity=0.04,
            dash_array="6 4",
            tooltip="AOI: Mundra Port region",
        ).add_to(aoi_map)

        # Zone polygons with labels
        zones = {
            "TERMINAL\nHARDSTANDING": {
                "bounds": [[22.72, 69.58], [22.80, 69.75]],
                "color" : "#E24B4A",
                "fill"  : 0.10,
                "tip"   : "Terminal Hardstanding — peak activity Aug 2025 · 17.2 dB",
            },
            "PORT\nHINTERLAND": {
                "bounds": [[22.80, 69.58], [22.95, 69.75]],
                "color" : "#EF9F27",
                "fill"  : 0.07,
                "tip"   : "Port Hinterland — warehouse & logistics activity",
            },
            "OFFSHORE\nANCHORAGE": {
                "bounds": [[22.55, 69.46], [22.72, 69.82]],
                "color" : "#42A5F5",
                "fill"  : 0.05,
                "tip"   : "Offshore Anchorage — vessel presence zone",
            },
        }

        for zname, z in zones.items():
            folium.Rectangle(
                bounds=z["bounds"],
                color=z["color"],
                weight=1.5,
                fill=True,
                fill_color=z["color"],
                fill_opacity=z["fill"],
                tooltip=folium.Tooltip(z["tip"]),
            ).add_to(aoi_map)

            cy = (z["bounds"][0][0] + z["bounds"][1][0]) / 2
            cx = (z["bounds"][0][1] + z["bounds"][1][1]) / 2
            label = zname.replace("\n", "<br>")
            folium.Marker(
                location=[cy, cx],
                icon=folium.DivIcon(
                    html=f'<div style="font-size:9px;color:{z["color"]};'
                         f'font-family:monospace;font-weight:700;'
                         f'text-align:center;white-space:nowrap;'
                         f'text-shadow:0 0 6px rgba(0,0,0,0.9)">'
                         f'{label}</div>',
                    icon_size=(120, 30),
                    icon_anchor=(60, 15),
                )
            ).add_to(aoi_map)

        # Peak event marker
        folium.CircleMarker(
            location=[22.762, 69.665],
            radius=14,
            color="#E24B4A",
            fill=True,
            fill_color="#E24B4A",
            fill_opacity=0.25,
            weight=2,
            tooltip=folium.Tooltip(
                "<b style='color:#E24B4A'>⚠ PEAK ACTIVITY</b><br>"
                "Aug 21, 2025<br>17.2 dB · Terminal Expansion<br>13.8 km² affected"
            ),
        ).add_to(aoi_map)
        folium.CircleMarker(
            location=[22.762, 69.665],
            radius=5,
            color="#E24B4A",
            fill=True,
            fill_color="#ff4444",
            fill_opacity=1.0,
            weight=0,
        ).add_to(aoi_map)

        # Scale + coord overlay
        scale_html = """
        <div style="position:absolute;bottom:10px;left:10px;z-index:1000;
             background:rgba(10,15,10,0.85);border:1px solid rgba(0,255,70,0.3);
             padding:6px 10px;font-family:monospace;font-size:10px;color:#00ff46">
            22.76°N &nbsp;69.67°E &nbsp;·&nbsp; UTM 43N<br>
            AOI: 39 × 37 km &nbsp;·&nbsp; EPSG:32643
        </div>"""
        aoi_map.get_root().html.add_child(folium.Element(scale_html))

        folium.LayerControl(position="topright").add_to(aoi_map)
        st_folium(aoi_map, width=None, height=map_height, returned_objects=[])

    if info_col is not None:
     with info_col:
        st.markdown("""
<div style="font-family:monospace;font-size:10px;color:rgba(0,255,70,0.85);
            line-height:2;padding-top:0.5rem">

<div style="color:#00ff46;font-size:11px;margin-bottom:0.5rem">
▸ ZONE LEGEND
</div>

<span style="color:#E24B4A">■</span> Terminal Hardstanding<br>
<span style="color:rgba(226,75,74,0.5)">  └ PEAK ZONE</span><br>
<span style="color:rgba(226,75,74,0.5)">  └ 17.2 dB · Aug 2025</span><br>
<br>
<span style="color:#EF9F27">■</span> Port Hinterland<br>
<span style="color:rgba(239,159,39,0.5)">  └ Warehouse / logistics</span><br>
<br>
<span style="color:#42A5F5">■</span> Offshore Anchorage<br>
<span style="color:rgba(66,165,245,0.5)">  └ Vessel presence</span><br>
<br>
<span style="color:#00ff46">□</span> AOI boundary<br>
<span style="color:rgba(0,255,70,0.4)">  └ 39 × 37 km</span><br>
<br>
<span style="color:#E24B4A">⬤</span> Peak event<br>
<span style="color:rgba(226,75,74,0.5)">  └ Aug 21 2025</span><br>

<div style="margin-top:1rem;border-top:1px solid rgba(0,255,70,0.15);
            padding-top:0.8rem;color:rgba(0,255,70,0.4);font-size:9px">
SENSOR &nbsp;&nbsp;&nbsp; S1A IW GRD<br>
SCENES &nbsp;&nbsp;&nbsp; 31<br>
PERIOD &nbsp;&nbsp;&nbsp; Apr 25–Apr 26<br>
PASS &nbsp;&nbsp;&nbsp;&nbsp; Descending<br>
POL &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; VV + VH<br>
RES &nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 20 m GSD
</div>
</div>
""", unsafe_allow_html=True)

    st.markdown("<div style='margin-top:1rem'></div>", unsafe_allow_html=True)

    # ── CHARTS ───────────────────────────────────
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("Construction phases")
        monthly = load_monthly()
        if not monthly.empty:
            import plotly.graph_objects as go

            colors = []
            for _, row in monthly.iterrows():
                p = row.get("activity_phase", "")
                if "HIGH" in p:     colors.append("#E24B4A")
                elif "MODERATE" in p: colors.append("#EF9F27")
                elif "LOW" in p:    colors.append("#639922")
                else:               colors.append("#444d44")

            fig = go.Figure(go.Bar(
                x=monthly["month_str"],
                y=monthly["peak_delta_db"],
                marker_color=colors,
                marker_line_color="rgba(0,255,70,0.3)",
                marker_line_width=1,
                text=monthly["peak_delta_db"].round(1),
                textposition="outside",
                textfont=dict(size=10, color=GREEN),
            ))
            fig.add_hline(
                y=3, line_dash="dash", line_color="rgba(0,255,70,0.4)",
                annotation_text="3 dB threshold",
                annotation_font=dict(color=GREEN_DIM, size=9),
                annotation_position="bottom right",
            )
            fig.add_hline(
                y=6, line_dash="dot", line_color="#E24B4A",
                annotation_text="6 dB high confidence",
                annotation_font=dict(color="#E24B4A", size=9),
                annotation_position="bottom right",
            )
            fig.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=30, b=0),
                yaxis_title="Peak Δσ° (dB)",
                showlegend=False,
            )
            apply_dark_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("monthly_construction_summary.csv not found")

    with col_right:
        st.subheader("Change types detected")
        df = load_classified()
        if not df.empty:
            import plotly.graph_objects as go

            port = df[df["zone_id"] > 0]
            ct = (port.groupby("change_type")["area_m2"]
                      .sum().sort_values(ascending=True) / 1e6)
            ct = ct[ct > 0]

            type_colors = {
                "Terminal_Expansion"    : "#A32D2D",
                "Construction_Active"   : "#E24B4A",
                "Construction_Equipment": "#FF7043",
                "Warehouse_Construction": "#EF9F27",
                "Hinterland_Activity"   : "#639922",
                "Terminal_Change"       : "#5a5a5a",
                "Hinterland_Change"     : "#3a3a3a",
                "Anchorage_Change"      : "#1a4a7a",
            }

            fig2 = go.Figure(go.Bar(
                x=ct.values.round(2),
                y=[n.replace("_", " ") for n in ct.index],
                orientation="h",
                marker_color=[type_colors.get(n, "#444") for n in ct.index],
                marker_line_color="rgba(0,255,70,0.2)",
                marker_line_width=1,
                text=ct.values.round(2),
                textposition="outside",
                textfont=dict(size=10, color=GREEN),
            ))
            fig2.update_layout(
                height=300,
                margin=dict(l=0, r=0, t=30, b=40),
                xaxis_title="Total area (km²)",
                showlegend=False,
            )
            apply_dark_theme(fig2)
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.info("classified_final.csv not found")

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
                        buf = render_tif_thumbnail(selected, colormap="gray")
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
                                    f, colormap="RdYlGn_r", vmin=-3, vmax=3)
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
                        selected_lr, colormap="RdYlGn_r", vmin=-10, vmax=10)
                    if buf:
                        st.image(buf,
                                 caption="Red = backscatter increase | Green = decrease",
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
                                f, colormap="seismic", vmin=-5, vmax=5)
                            if buf:
                                st.image(buf, caption=date_str,
                                         use_container_width=True)

# ─────────────────────────────────────────────
# PAGE 3: CHANGE DETECTION MAP
# ─────────────────────────────────────────────
elif page == "Change Detection Map":
    st.title("Change Detection Map")
    st.markdown("Interactive map of classified change polygons over Mundra Port.")

    df  = load_classified()
    gdf = load_geodata()

    if df.empty:
        st.error("classified_final.csv not found")
        st.stop()

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
        valid_types   = ZONE_TYPE_MAP[selected_zone]
        selected_type = st.selectbox("Change type", valid_types)
    with col3:
        dates         = sorted(df["date_from"].unique().tolist())
        selected_date = st.selectbox(
            "Date", ["All dates"] + dates,
            index=dates.index("20250704") + 1 if "20250704" in dates else 0
        )

    st.info("Tip: Select Terminal_Hardstanding + Terminal_Expansion + date 20250704 to see the peak construction event.")

    fgdf = gdf.copy()
    if selected_zone != "All zones":
        fgdf = fgdf[fgdf["zone_name"] == selected_zone]
    if selected_type != "All types":
        fgdf = fgdf[fgdf["change_type"] == selected_type]
    if selected_date != "All dates":
        fgdf = fgdf[fgdf["date_from"] == selected_date]

    if len(fgdf) == 0:
        st.warning("No polygons found for this combination. Try: Terminal_Hardstanding + Terminal_Expansion + 20250704")
        st.stop()
    else:
        st.info(f"Showing {len(fgdf):,} polygons")

    m = folium.Map(location=[22.76, 69.67], zoom_start=12, tiles="CartoDB dark_matter")

    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Satellite",
        overlay=False,
    ).add_to(m)

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

    zone_boxes = {
        "Terminal"  : [[22.72, 69.58], [22.80, 69.75]],
        "Anchorage" : [[22.55, 69.50], [22.72, 69.90]],
        "Hinterland": [[22.80, 69.58], [22.95, 69.85]],
    }
    for zname, bounds in zone_boxes.items():
        folium.Rectangle(
            bounds=bounds, color="#00ff46", weight=1.5,
            fill=False, dash_array="5 5",
            tooltip=f"Zone: {zname}",
        ).add_to(m)
        folium.Marker(
            location=[(bounds[0][0]+bounds[1][0])/2,
                      (bounds[0][1]+bounds[1][1])/2],
            icon=folium.DivIcon(
                html=f'<div style="font-size:11px;color:#00ff46;'
                     f'font-weight:600;white-space:nowrap;'
                     f'font-family:monospace">{zname}</div>',
                icon_size=(80, 20),
            )
        ).add_to(m)

    if selected_zone == "Outside" or selected_type == "Background_Change":
        st.warning("Background_Change covers the full scene — noise polygons with no intelligence value. Select a port zone instead.")
        st.stop()

    limit    = 200 if selected_date == "All dates" else 1000
    plot_gdf = fgdf.head(limit)
    if len(fgdf) > limit:
        st.warning(f"Showing first {limit} of {len(fgdf):,} polygons. Select a specific date to see all.")

    for _, row in plot_gdf.iterrows():
        try:
            ct     = row.get("change_type", "Unknown")
            color  = type_colors.get(ct, "#888")
            area   = row.get("area_m2", 0)
            delta  = row.get("mean_delta_db", 0)
            date   = row.get("date_from", "")
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

    legend_html = """
    <div style="position:fixed;bottom:30px;right:30px;z-index:1000;
         background:#0a0f0a;padding:12px 16px;border-radius:0;
         border:1px solid rgba(0,255,70,0.3);font-size:12px;
         font-family:monospace;color:#00ff46">
    <b style="color:#00ff46">CHANGE TYPE</b><br>
    """
    for ct, color in type_colors.items():
        if ct in ["Terminal_Expansion","Construction_Active",
                  "Vessel_Presence","Anchorage_Change","Hinterland_Activity"]:
            legend_html += (f'<span style="background:{color};display:inline-block;'
                           f'width:12px;height:12px;margin-right:6px;'
                           f'vertical-align:middle"></span>'
                           f'<span style="color:#00ff46">{ct.replace("_"," ")}</span><br>')
    legend_html += "</div>"
    m.get_root().html.add_child(folium.Element(legend_html))

    folium.LayerControl().add_to(m)

    cdm_expand = st.toggle("Expand map", value=False, key="expand_cdm_map")
    with st.spinner("Rendering map..."):
        st_folium(m, width=None, height=800 if cdm_expand else 550,
                  returned_objects=[])

    if not fgdf.empty and "change_type" in fgdf.columns:
        st.markdown("---")
        st.subheader("Filtered results summary")
        col_a, col_b = st.columns(2)
        with col_a:
            by_type = (fgdf.groupby("change_type")
                          .agg(count=("area_m2","count"),
                               area_km2=("area_m2", lambda x: round(x.sum()/1e6, 3)))
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
    monthly_peak  = term.groupby("month")["mean_delta_db"].max().reset_index()

    cargo = pd.DataFrame({
        "date"   : pd.to_datetime(["2024-09-30","2024-12-31","2025-03-31",
                                   "2025-06-30","2025-09-30","2025-12-31","2026-03-31"]),
        "mmt"    : [47.2, 49.1, 51.3, 52.8, 55.4, 54.9, 57.2],
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
        vertical_spacing=0.14,
    )

    # ── SAR signal ──
    fig.add_trace(go.Scatter(
        x=monthly_peak["month"],
        y=monthly_peak["mean_delta_db"].round(2),
        mode="lines+markers",
        name="Peak Δσ° (dB)",
        line=dict(color="#E24B4A", width=2.5),
        marker=dict(size=8, color="#E24B4A",
                    line=dict(color=DARK_BG, width=1.5)),
        fill="tozeroy",
        fillcolor="rgba(226,75,74,0.12)",
    ), row=1, col=1)

    fig.add_hline(
        y=3, line_dash="dash", line_color="rgba(0,255,70,0.4)",
        annotation_text="3 dB threshold",
        annotation_font=dict(color=GREEN_DIM, size=9),
        annotation_position="bottom right",
        row=1, col=1,
    )
    fig.add_hline(
        y=6, line_dash="dot", line_color="#E24B4A",
        annotation_text="6 dB high confidence",
        annotation_font=dict(color="#E24B4A", size=9),
        annotation_position="bottom right",
        row=1, col=1,
    )
    fig.add_vrect(
        x0="2025-06-01", x1="2025-08-31",
        fillcolor="rgba(226,75,74,0.08)", line_width=0,
        annotation_text="Phase 1",
        annotation_position="top left",
        annotation_font=dict(size=11, color="#E24B4A"),
        row=1, col=1,
    )

    # ── Cargo ──
    fig.add_trace(go.Scatter(
        x=cargo["date"],
        y=cargo["mmt"],
        mode="lines+markers+text",
        name="Cargo (MMT)",
        line=dict(color="#378ADD", width=2.5),
        marker=dict(size=10, color="#378ADD",
                    line=dict(color=DARK_BG, width=2)),
        text=cargo["mmt"],
        textposition="top center",
        textfont=dict(size=10, color="#378ADD"),
    ), row=2, col=1)

    # ── APSEZ announcement line ──
    for row_n in [1, 2]:
        fig.add_shape(
            type="line",
            x0="2025-09-18", x1="2025-09-18",
            y0=0, y1=1,
            yref="paper", xref="x",
            line=dict(color="#F59E0B", width=1.5, dash="dash"),
            row=row_n, col=1,
        )
    fig.add_annotation(
        x="2025-09-18", y=0.97,
        xref="x", yref="paper",
        text="APSEZ ₹30,000 Cr expansion",
        showarrow=False,
        font=dict(color="#F59E0B", size=10, family=FONT_FAM),
        bgcolor="rgba(10,15,10,0.8)",
        bordercolor="#F59E0B",
        borderwidth=1,
    )

    fig.update_layout(
        height=620,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom",
                    y=1.02, xanchor="right", x=1),
        margin=dict(l=60, r=20, t=80, b=40),
    )

    # Apply dark theme AFTER all traces/annotations are added
    apply_dark_theme(fig, rows=2)

    fig.update_yaxes(title_text="Peak Δσ° (dB)",        row=1, col=1)
    fig.update_yaxes(title_text="Cargo throughput (MMT)", row=2, col=1, range=[44, 60])

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Quarter-by-quarter summary")
    summary = pd.DataFrame({
        "Quarter"     : cargo["quarter"],
        "Cargo (MMT)" : cargo["mmt"],
        "Period"      : ["Sep 2024","Dec 2024","Mar 2025",
                         "Jun 2025","Sep 2025","Dec 2025","Mar 2026"],
        "SAR phase"   : ["—","—","Baseline",
                         "Low activity","PEAK ACTIVITY","Moderate","Low activity"],
        "Notes"       : ["","","Pre-construction baseline",
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

# ─────────────────────────────────────────────
# FOOTER — dataset facts that used to live in the permanent sidebar.
# Available everywhere, but out of the way.
# ─────────────────────────────────────────────
st.markdown("<div style='margin-top:2rem'></div>", unsafe_allow_html=True)
with st.expander("Dataset & pipeline details"):
    fcol1, fcol2 = st.columns(2)
    with fcol1:
        st.markdown("""
**Dataset**
- Sensor: Sentinel-1A IW GRD
- Scenes: 31
- Period: Apr 2025 – Apr 2026
- Method: Log-ratio change detection
""")
    with fcol2:
        st.markdown("""
**Pipeline**
- Preprocessing: ESA SNAP
- Analysis: Python + rasterio
- Database: PostGIS 3.6
- Visualisation: QGIS + Streamlit
""")
