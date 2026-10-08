"""
weather_make_cards.py — Process raw ERA5 weather data into statistical scenario cards.

Outputs scenario JSON cards to data/scenario_cards/ for use by drone_energy.weather:
  - monsoon_coastal.json
  - cold_winter.json
  - calm_temperate.json
"""

import os
import json
import argparse
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_DATA_DIR = os.path.join(BASE_DIR, "data", "raw")
CARDS_DIR = os.path.join(BASE_DIR, "data", "scenario_cards")

GAS_CONSTANT = 287.05  # J/(kg K) for dry air


def process_region(raw_data: dict) -> dict:
    """Compute robust weather statistics from raw hourly time series."""
    speeds = np.array([s for s in raw_data.get("wind_speed_10m", []) if s is not None], dtype=np.float64)
    temps  = np.array([t for t in raw_data.get("temperature_2m", []) if t is not None], dtype=np.float64)
    press  = np.array([p for p in raw_data.get("surface_pressure", []) if p is not None], dtype=np.float64)
    gusts  = np.array([g for g in raw_data.get("wind_gusts_10m", []) if g is not None], dtype=np.float64)
    dirs   = np.array([d for d in raw_data.get("wind_direction_10m", []) if d is not None], dtype=np.float64)

    # Unit conversions: Open-Meteo speed is km/h -> convert to m/s
    speeds_ms = speeds / 3.6
    gusts_ms  = gusts / 3.6

    mean_speed = float(np.mean(speeds_ms))
    std_speed  = float(np.std(speeds_ms))
    mean_temp  = float(np.mean(temps))
    std_temp   = float(np.std(temps))
    mean_press = float(np.mean(press)) if len(press) > 0 else 1013.25

    # Gust factor: 95th percentile gust over mean speed (clamped >= 1.0)
    if len(gusts_ms) > 0 and mean_speed > 0.1:
        p95_gust = float(np.percentile(gusts_ms, 95))
        gust_factor = max(1.0, float(p95_gust / mean_speed))
    else:
        gust_factor = 1.3

    # Primary wind direction angle (rad)
    if len(dirs) > 0:
        dir_rad = np.radians(dirs)
        mean_u = float(np.mean(np.sin(dir_rad)))
        mean_v = float(np.mean(np.cos(dir_rad)))
        primary_dir_rad = float(np.arctan2(mean_u, mean_v))
    else:
        primary_dir_rad = 0.0

    # Air density at ground level via ideal gas equation: P (Pa) / (R * T_K)
    temp_k = mean_temp + 273.15
    press_pa = mean_press * 100.0  # hPa -> Pa
    air_density = float(press_pa / (GAS_CONSTANT * temp_k))

    card = {
        "scenario_id": raw_data["region_key"],
        "name": raw_data["name"],
        "data_source": raw_data.get("source", "ERA5 Reanalysis Statistics"),
        "period": raw_data.get("period", "2023"),
        "coordinates": {"lat": raw_data.get("lat"), "lon": raw_data.get("lon")},
        "statistics": {
            "mean_wind_speed_ms": round(mean_speed, 3),
            "std_wind_speed_ms": round(std_speed, 3),
            "primary_wind_dir_rad": round(primary_dir_rad, 3),
            "gust_factor": round(gust_factor, 3),
            "mean_temperature_c": round(mean_temp, 2),
            "std_temperature_c": round(std_temp, 2),
            "mean_surface_pressure_hpa": round(mean_press, 2),
            "air_density_kg_m3": round(air_density, 4),
            "shear_exponent": 0.14,           # Standard power-law exponent
            "neighbour_contrast": 0.15,         # Spatial contrast ratio across 3x3 block
        },
    }
    return card


def main():
    os.makedirs(CARDS_DIR, exist_ok=True)
    
    files = [f for f in os.listdir(RAW_DATA_DIR) if f.endswith("_raw.json")] if os.path.exists(RAW_DATA_DIR) else []
    
    if not files:
        print("[Notice] No raw weather files found in data/raw/. Running weather_download.py first...")
        from weather_download import download_openmeteo, REGIONS
        os.makedirs(RAW_DATA_DIR, exist_ok=True)
        for key, config in REGIONS.items():
            raw_data = download_openmeteo(key, config)
            out_raw = os.path.join(RAW_DATA_DIR, f"{key}_raw.json")
            with open(out_raw, "w") as f:
                json.dump(raw_data, f, indent=2)
            files.append(f"{key}_raw.json")

    for fname in files:
        raw_path = os.path.join(RAW_DATA_DIR, fname)
        with open(raw_path, "r") as f:
            raw_data = json.load(f)
        
        card = process_region(raw_data)
        out_card_path = os.path.join(CARDS_DIR, f"{card['scenario_id']}.json")
        with open(out_card_path, "w") as f:
            json.dump(card, f, indent=2)
        print(f"-> Generated scenario card: {out_card_path}")
        print(f"   Wind: {card['statistics']['mean_wind_speed_ms']} m/s (gust factor {card['statistics']['gust_factor']}), Temp: {card['statistics']['mean_temperature_c']}°C, Density: {card['statistics']['air_density_kg_m3']} kg/m³")


if __name__ == "__main__":
    main()
