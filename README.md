# DeltaPort

# Mundra Port SAR Economic Intelligence

Sentinel-1 SAR time series analysis detecting construction activity 
at Mundra Port, India's largest commercial port.

## Key Finding
SAR detected peak construction activity (17.2 dB) in June–August 2025, 
preceding APSEZ's ₹30,000 Cr expansion announcement by ~6 weeks.

## Pipeline
- Sensor: Sentinel-1A IW GRD (31 scenes, Apr 2025 – Apr 2026)
- Method: Log-ratio change detection + z-score thresholding
- Database: PostgreSQL + PostGIS 3.6
- Stack: Python, rasterio, geopandas, Streamlit

## Run locally
pip install -r requirements.txt
streamlit run app.py
