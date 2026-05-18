"""
Camada de acesso a dados de precipitação — Bacia Mirim.

Estações: API pública INMET (lista e coordenadas).
Precipitação: Open-Meteo Archive API (ERA5 + modelos NWP, gratuita, sem autenticação).
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st

INMET_BASE = "https://apitempo.inmet.gov.br"
OPENMETEO_ARCHIVE = "https://archive-api.open-meteo.com/v1/archive"
OPENMETEO_FORECAST = "https://api.open-meteo.com/v1/forecast"

# Bounding box da Bacia Mirim e entorno
# Cobre: Jaguarão, Pelotas, Rio Grande, Santa Vitória do Palmar, Bagé, Canguçu
BBOX = {
    "lat_min": -34.0,
    "lat_max": -30.5,
    "lon_min": -54.5,
    "lon_max": -49.5,
}


@st.cache_data(ttl=3600)
def get_stations() -> pd.DataFrame:
    """Retorna estações automáticas INMET dentro do bounding box da Bacia Mirim."""
    url = f"{INMET_BASE}/estacoes/T"
    try:
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
    except requests.RequestException as e:
        st.error(f"Erro ao buscar estações: {e}")
        return pd.DataFrame()

    df = pd.DataFrame(resp.json())
    if df.empty:
        return df

    df["VL_LATITUDE"] = pd.to_numeric(df["VL_LATITUDE"], errors="coerce")
    df["VL_LONGITUDE"] = pd.to_numeric(df["VL_LONGITUDE"], errors="coerce")
    df = df.dropna(subset=["VL_LATITUDE", "VL_LONGITUDE"])

    mask = (
        (df["VL_LATITUDE"] >= BBOX["lat_min"])
        & (df["VL_LATITUDE"] <= BBOX["lat_max"])
        & (df["VL_LONGITUDE"] >= BBOX["lon_min"])
        & (df["VL_LONGITUDE"] <= BBOX["lon_max"])
    )
    return df[mask].reset_index(drop=True)


@st.cache_data(ttl=1800, show_spinner=False)
def get_daily_series(lat: float, lon: float, days: int) -> pd.DataFrame:
    """
    Retorna série diária de precipitação acumulada para uma coordenada.
    Usa Open-Meteo Archive API (ERA5 reanalysis + NWP).
    """
    end = datetime.now().date()
    start = end - timedelta(days=days)

    # Open-Meteo: dados históricos (archive) até ontem
    # Para hoje usa o endpoint de forecast
    yesterday = end - timedelta(days=1)

    params_archive = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start.strftime("%Y-%m-%d"),
        "end_date": yesterday.strftime("%Y-%m-%d"),
        "daily": "precipitation_sum",
        "timezone": "America/Sao_Paulo",
    }
    params_forecast = {
        "latitude": lat,
        "longitude": lon,
        "daily": "precipitation_sum",
        "timezone": "America/Sao_Paulo",
        "forecast_days": 1,
    }

    frames = []
    try:
        r = requests.get(OPENMETEO_ARCHIVE, params=params_archive, timeout=15)
        if r.status_code == 200:
            d = r.json()["daily"]
            frames.append(pd.DataFrame({"Data": d["time"], "Precipitação (mm)": d["precipitation_sum"]}))
    except requests.RequestException:
        pass

    try:
        r = requests.get(OPENMETEO_FORECAST, params=params_forecast, timeout=15)
        if r.status_code == 200:
            d = r.json()["daily"]
            frames.append(pd.DataFrame({"Data": d["time"], "Precipitação (mm)": d["precipitation_sum"]}))
    except requests.RequestException:
        pass

    if not frames:
        return pd.DataFrame(columns=["Data", "Precipitação (mm)"])

    df = pd.concat(frames).drop_duplicates("Data")
    df["Data"] = pd.to_datetime(df["Data"])
    df["Precipitação (mm)"] = pd.to_numeric(df["Precipitação (mm)"], errors="coerce").fillna(0)
    return df.sort_values("Data").reset_index(drop=True)


def get_accumulated(lat: float, lon: float, days: int = 7) -> float:
    """Retorna precipitação total acumulada nos últimos N dias (mm)."""
    df = get_daily_series(lat, lon, days)
    if df.empty:
        return 0.0
    return float(df["Precipitação (mm)"].sum())
