"""
Camada de acesso à API pública do INMET.
Estações automáticas (tipo T) filtradas pela Bacia Mirim.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
import streamlit as st

INMET_BASE = "https://apitempo.inmet.gov.br"

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
def get_station_data(station_code: str, date_start: str, date_end: str) -> pd.DataFrame:
    """
    Busca dados horários de uma estação entre duas datas (formato YYYY-MM-DD).
    Retorna DataFrame com coluna CHUVA (mm) e DT_MEDICAO (datetime).
    """
    url = f"{INMET_BASE}/estacao/{date_start}/{date_end}/{station_code}"
    try:
        resp = requests.get(url, timeout=30)
        if resp.status_code != 200:
            return pd.DataFrame()
        data = resp.json()
    except requests.RequestException:
        return pd.DataFrame()

    if not data:
        return pd.DataFrame()

    df = pd.DataFrame(data)
    df["CHUVA"] = pd.to_numeric(df.get("CHUVA", 0), errors="coerce").fillna(0)
    df["DT_MEDICAO"] = pd.to_datetime(df["DT_MEDICAO"], errors="coerce")
    return df


def get_accumulated(station_code: str, days: int = 7) -> float:
    """Retorna precipitação total acumulada nos últimos N dias (mm)."""
    end = datetime.now()
    start = end - timedelta(days=days)
    df = get_station_data(
        station_code,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    )
    if df.empty or "CHUVA" not in df.columns:
        return 0.0
    return float(df["CHUVA"].sum())


def get_daily_series(station_code: str, days: int) -> pd.DataFrame:
    """Retorna série diária de precipitação acumulada para o período."""
    end = datetime.now()
    start = end - timedelta(days=days)
    df = get_station_data(
        station_code,
        start.strftime("%Y-%m-%d"),
        end.strftime("%Y-%m-%d"),
    )
    if df.empty:
        return pd.DataFrame(columns=["Data", "Precipitação (mm)"])

    daily = (
        df.dropna(subset=["DT_MEDICAO"])
        .groupby(df["DT_MEDICAO"].dt.date)["CHUVA"]
        .sum()
        .reset_index()
    )
    daily.columns = ["Data", "Precipitação (mm)"]
    daily["Data"] = pd.to_datetime(daily["Data"])
    return daily
