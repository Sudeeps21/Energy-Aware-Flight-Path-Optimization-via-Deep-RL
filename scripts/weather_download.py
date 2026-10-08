"""
weather_download.py — Download historical ERA5 weather statistics for drone delivery scenarios.

Supports both:
  1. Open-Meteo Historical Reanalysis API (fast, no key required, default)
  2. Copernicus Climate Data Store (CDS) ERA5 API (via cdsapi)

Regions defined in TEAM_PLAN_TASKS_TIMELINE_RESOURCES.md:
  - monsoon_coastal : Warm, strong winds, high gust factor (e.g., Coastal India/Mumbai)
  - cold_winter     : Sub-zero temperatures, high altitude (e.g., Leh/Ladakh)
  - calm_temperate  : Moderate temperature, low wind reference (e.g., Bangalore plateau)
"""

import os
import json
import argparse
import requests
import numpy as np

RAW_DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")

REGIONS = {
    "monsoon_coastal": {
        "name": "Monsoon Coastal Region (Mumbai/Goa Coast)",
        "lat": 18.96,
        "lon": 72.82,
        "start_date": "2023-06-01",
        "end_date": "2023-08-31",
    },
    "cold_winter": {
        "name": "Cold Altitude Winter Region (Leh/Ladakh)",
        "lat": 34.15,
        "lon": 77.57,
        "start_date": "2023-12-01",
        "end_date": "2024-02-29",
    },
    "calm_temperate": {
        "name": "Calm Temperate Plateau (Bangalore Reference)",
        "lat": 12.97,
        "lon": 77.59,
        "start_date": "2023-10-01",
        "end_date": "2023-11-30",
    },
}


def download_openmeteo(region_key: str, config: dict) -> dict:
    """Fetch hourly historical ERA5 variables from Open-Meteo Archive API."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": config["lat"],
        "longitude": config["lon"],
        "start_date": config["start_date"],
        "end_date": config["end_date"],
        "hourly": [
            "temperature_2m",
            "surface_pressure",
            "wind_speed_10m",
            "wind_direction_10m",
            "wind_gusts_10m",
        ],
        "timezone": "UTC",
    }
    
    print(f"[Open-Meteo] Fetching data for {config['name']} ({config['start_date']} to {config['end_date']})...")
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    
    hourly = data.get("hourly", {})
    records = {
        "region_key": region_key,
        "name": config["name"],
        "lat": config["lat"],
        "lon": config["lon"],
        "period": f"{config['start_date']} to {config['end_date']}",
        "temperature_2m": hourly.get("temperature_2m", []),
        "surface_pressure": hourly.get("surface_pressure", []),
        "wind_speed_10m": hourly.get("wind_speed_10m", []),
        "wind_direction_10m": hourly.get("wind_direction_10m", []),
        "wind_gusts_10m": hourly.get("wind_gusts_10m", []),
        "source": "Open-Meteo Historical ERA5 Reanalysis API (CC BY 4.0)",
    }
    return records


def download_cds(region_key: str, config: dict) -> str:
    """Download ERA5 NetCDF slice via Copernicus CDS API."""
    import cdsapi
    client = cdsapi.Client()
    
    out_file = os.path.join(RAW_DATA_DIR, f"{region_key}_era5.nc")
    lat, lon = config["lat"], config["lon"]
    # 1-degree bounding box
    bbox = [lat + 0.5, lon - 0.5, lat - 0.5, lon + 0.5]
    
    print(f"[Copernicus CDS] Submitting request for {region_key}...")
    client.retrieve(
        "reanalysis-era5-single-levels",
        {
            "product_type": "reanalysis",
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_temperature",
                "surface_pressure",
                "10m_wind_gust_since_previous_post_processing",
            ],
            "year": "2023",
            "month": ["01", "06", "12"],
            "day": [f"{d:02d}" for d in range(1, 29)],
            "time": [f"{h:02d}:00" for h in range(0, 24)],
            "area": bbox,
            "format": "netcdf",
        },
        out_file,
    )
    print(f"[Copernicus CDS] Downloaded {out_file}")
    return out_file


def main():
    parser = argparse.ArgumentParser(description="Download weather data for scenario cards")
    parser.add_argument("--source", choices=["openmeteo", "cds"], default="openmeteo",
                        help="Weather data source (default: openmeteo)")
    args = parser.parse_args()

    os.makedirs(RAW_DATA_DIR, exist_ok=True)

    for key, config in REGIONS.items():
        if args.source == "openmeteo":
            data = download_openmeteo(key, config)
            out_path = os.path.join(RAW_DATA_DIR, f"{key}_raw.json")
            with open(out_path, "w") as f:
                json.dump(data, f, indent=2)
            print(f"-> Saved raw weather data to {out_path}\n")
        else:
            download_cds(key, config)


if __name__ == "__main__":
    main()
