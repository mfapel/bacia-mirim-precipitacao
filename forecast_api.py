"""
Previsão de precipitação — Bacia Mirim.

Fontes:
  - ECMWF IFS 0.25° : 15 dias determinístico  (Open-Meteo /v1/forecast)
  - GFS Ensemble 0.5°: até 35 dias, 31 membros (Open-Meteo /v1/ensemble)

Referências científicas:
  - Badagian et al. (2024): skill sub-sazonal positivo até semana 2 na bacia
    do Rio Uruguai; ECMWF melhor em todos os horizontes.
  - ERA5 validado para Bacia Mirim-São Gonçalo (RBGF, 2023): correlação
    forte, RMSE e MAPE baixos.
  - ECMWF IFS supera GFS em precipitação calibrada (AMS 2020), mas GFS
    ensemble estende o horizonte até 35 dias com skill marginal além da
    semana 2.
"""

import requests
import pandas as pd
import numpy as np
from datetime import date
import streamlit as st

OPENMETEO_FORECAST = "https://api.open-meteo.com/v1/forecast"
OPENMETEO_ENSEMBLE = "https://ensemble-api.open-meteo.com/v1/ensemble"

# Centro da Lagoa Mirim — ponto de referência para a previsão regional
BASIN_LAT = -32.50
BASIN_LON = -52.80


@st.cache_data(ttl=3600, show_spinner=False)
def get_ecmwf_15d(lat: float = BASIN_LAT, lon: float = BASIN_LON, _v: int = 2) -> pd.DataFrame:
    """
    Previsão ECMWF IFS 0.25° — precipitação diária nos próximos 15 dias.
    Skill positivo confirmado até ~14 dias (Badagian et al., 2024).
    """
    try:
        r = requests.get(
            OPENMETEO_FORECAST,
            params={
                "latitude": lat,
                "longitude": lon,
                "daily": "precipitation_sum",
                "models": "best_match",   # selecao automatica por coordenada
                "timezone": "America/Sao_Paulo",
                "forecast_days": 15,
            },
            timeout=15,
        )
        if r.status_code != 200:
            return pd.DataFrame()
        d = r.json()["daily"]
        df = pd.DataFrame({
            "Data":       pd.to_datetime(d["time"]),
            "Precip_mm":  pd.to_numeric(pd.Series(d["precipitation_sum"]), errors="coerce").fillna(0),
        })
        df["Acumulado_mm"] = df["Precip_mm"].cumsum()
        return df
    except Exception as e:
        import traceback
        st.session_state["_ecmwf_error"] = traceback.format_exc()
        return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def get_gfs_ensemble(lat: float = BASIN_LAT, lon: float = BASIN_LON) -> pd.DataFrame:
    """
    GFS Ensemble 0.5° — precipitação diária com faixa P10/P50/P90.
    Horizonte: até 35 dias; 31 membros.
    Skill marginal após semana 2 — apresentar apenas como faixa probabilística.
    """
    try:
        r = requests.get(
            OPENMETEO_ENSEMBLE,
            params={
                "latitude":  lat,
                "longitude": lon,
                "daily":     "precipitation_sum",
                "models":    "gfs025",
                "timezone":  "America/Sao_Paulo",
                "forecast_days": 35,
            },
            timeout=20,
        )
        if r.status_code != 200:
            return pd.DataFrame()

        d = r.json().get("daily", {})
        if not d:
            return pd.DataFrame()
        dates = pd.to_datetime(d["time"])

        arrays = []
        for key in d:
            if key.startswith("precipitation_sum_member"):
                arr = pd.to_numeric(pd.Series(d[key]), errors="coerce").values
                arrays.append(arr)

        if not arrays:
            return pd.DataFrame()

        mat = np.array(arrays)  # (n_members, n_days)

        # Remove dias onde todos os membros são NaN (além do horizonte real do modelo)
        valid_days = ~np.all(np.isnan(mat), axis=0)
        mat = mat[:, valid_days]
        dates = dates[valid_days]

        df = pd.DataFrame({
            "Data":    dates,
            "P10_mm":  np.nanpercentile(mat, 10, axis=0).clip(min=0),
            "P50_mm":  np.nanpercentile(mat, 50, axis=0).clip(min=0),
            "P90_mm":  np.nanpercentile(mat, 90, axis=0).clip(min=0),
        })

        # Acumulados a partir de hoje
        hoje = pd.Timestamp(date.today())
        df = df[df["Data"] >= hoje].reset_index(drop=True)
        df["P10_acum"] = df["P10_mm"].cumsum()
        df["P50_acum"] = df["P50_mm"].cumsum()
        df["P90_acum"] = df["P90_mm"].cumsum()

        return df
    except Exception:
        return pd.DataFrame()
