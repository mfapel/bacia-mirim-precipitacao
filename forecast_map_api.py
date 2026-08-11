"""
Campos espaciais de previsão meteorológica — Região da Lagoa Mirim.

Grade de pontos sobre a bacia, consultados em uma única chamada à Open-Meteo.
Variáveis: precipitação diária, vento máximo 10m, rajada máxima.
Horizonte: 15 dias (best_match por coordenada).
"""

import requests
import numpy as np
import pandas as pd
import streamlit as st

OPENMETEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

# Grade: lat -30.0 a -34.5 (step 0.5°) × lon -49.5 a -55.5 (step 0.5°)
_LATS = np.arange(-30.0, -35.0, -0.5)   # 10 pontos
_LONS = np.arange(-49.5, -56.0, -0.5)   # 13 pontos

GRID_LATS = _LATS
GRID_LONS = _LONS

# Pares (lat, lon) para todas as células da grade
_LAT_GRID, _LON_GRID = np.meshgrid(_LATS, _LONS, indexing="ij")
_LAT_FLAT = _LAT_GRID.flatten().tolist()
_LON_FLAT = _LON_GRID.flatten().tolist()


@st.cache_data(ttl=3600, show_spinner=False)
def get_forecast_grid() -> dict:
    """
    Retorna dict com arrays 3D (n_lats, n_lons, n_days) para cada variável,
    mais o vetor de datas.

    Chaves: 'dates', 'precip', 'wind', 'gusts'
    Shape dos arrays: (n_lats, n_lons, n_days)
    """
    try:
        r = requests.get(
            OPENMETEO_FORECAST,
            params={
                "latitude":  ",".join(f"{v:.1f}" for v in _LAT_FLAT),
                "longitude": ",".join(f"{v:.1f}" for v in _LON_FLAT),
                "daily": "precipitation_sum,wind_speed_10m_max,wind_gusts_10m_max",
                "models": "best_match",
                "timezone": "America/Sao_Paulo",
                "forecast_days": 15,
            },
            timeout=30,
        )
        if r.status_code != 200:
            return {}

        items = r.json()
        if not items or not isinstance(items, list):
            return {}

        n_lats = len(_LATS)
        n_lons = len(_LONS)
        n_days = len(items[0]["daily"]["time"])
        dates = pd.to_datetime(items[0]["daily"]["time"])

        def _build_array(key: str) -> np.ndarray:
            flat = []
            for item in items:
                vals = item["daily"].get(key, [None] * n_days)
                flat.append([v if v is not None else np.nan for v in vals])
            arr_flat = np.array(flat)              # (n_lats*n_lons, n_days)
            return arr_flat.reshape(n_lats, n_lons, n_days)

        return {
            "dates":  dates,
            "precip": _build_array("precipitation_sum"),
            "wind":   _build_array("wind_speed_10m_max"),
            "gusts":  _build_array("wind_gusts_10m_max"),
        }

    except Exception:
        return {}
