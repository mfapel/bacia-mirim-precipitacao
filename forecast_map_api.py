"""
Campos espaciais de previsão meteorológica — Região da Lagoa Mirim.

Grade de pontos sobre a bacia, consultados em uma única chamada à Open-Meteo.
Variáveis: precipitação diária, vento máximo 10m (+ direção), rajada máxima.
Horizonte: 15 dias (best_match por coordenada).
Limites municipais: IBGE API (malhas/municipios).
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

_LAT_GRID, _LON_GRID = np.meshgrid(_LATS, _LONS, indexing="ij")
GRID_LAT_FLAT = _LAT_GRID.flatten().tolist()
GRID_LON_FLAT = _LON_GRID.flatten().tolist()

# Polígono aproximado da Lagoa Mirim (sentido horário, fechado)
LAGOA_MIRIM_LAT = [
    -32.32, -32.50, -32.75, -33.00, -33.25, -33.50, -33.75, -34.00, -34.15, -34.35,
    -34.30, -34.10, -33.90, -33.65, -33.40, -33.15, -32.90, -32.65, -32.45, -32.32,
]
LAGOA_MIRIM_LON = [
    -52.62, -52.38, -52.23, -52.25, -52.35, -52.48, -52.62, -52.80, -53.05, -53.30,
    -53.58, -53.65, -53.55, -53.48, -53.38, -53.25, -53.10, -52.95, -52.80, -52.62,
]

# Municípios RS relevantes para a Bacia Mirim (códigos IBGE)
MUNICIPIOS_IBGE = [
    4311205,  # Jaguarão
    4314407,  # Pelotas
    4315602,  # Rio Grande
    4318002,  # São José do Norte
    4312757,  # Mostardas
    4321402,  # Tavares
    4317301,  # Santa Vitória do Palmar
    4304614,  # Chuí
    4300909,  # Arroio Grande
    4314209,  # Pedro Osório
    4309506,  # Herval
    4314902,  # Pinheiro Machado
    4304002,  # Capão do Leão
    4322152,  # Turuçu
    4305835,  # Cristal
    4304895,  # Cerrito
    4303905,  # Canguçu
    4301602,  # Bagé
]


@st.cache_data(ttl=86400 * 7, show_spinner=False)
def get_municipios_geojson() -> dict:
    """Busca GeoJSON de municípios da Bacia Mirim via IBGE API."""
    features = []
    for cod in MUNICIPIOS_IBGE:
        try:
            r = requests.get(
                f"https://servicodados.ibge.gov.br/api/v3/malhas/municipios/{cod}",
                params={"formato": "application/vnd.geo+json"},
                timeout=15,
            )
            if r.status_code != 200:
                continue
            gj = r.json()
            for feat in gj.get("features", []):
                if feat.get("properties") is None:
                    feat["properties"] = {}
                feat["properties"]["cod"] = str(cod)
                features.append(feat)
        except Exception:
            continue
    return {"type": "FeatureCollection", "features": features}


def geojson_to_latlon(gj: dict):
    """Extrai coordenadas de polígonos de um FeatureCollection como listas lat/lon com None como separador."""
    lats, lons = [], []
    for feat in gj.get("features", []):
        geom = feat.get("geometry", {})
        rings = []
        if geom.get("type") == "Polygon":
            rings = geom["coordinates"]
        elif geom.get("type") == "MultiPolygon":
            for poly in geom["coordinates"]:
                rings.extend(poly)
        for ring in rings:
            for lon, lat in ring:
                lons.append(lon)
                lats.append(lat)
            lons.append(None)
            lats.append(None)
    return lats, lons


@st.cache_data(ttl=3600, show_spinner=False)
def get_forecast_grid() -> dict:
    """
    Retorna dict com arrays 3D (n_lats, n_lons, n_days) para cada variável.

    Chaves: 'dates', 'precip', 'wind', 'gusts', 'u_wind', 'v_wind'
    Shape: (n_lats, n_lons, n_days)
    Componentes u/v em km/h (convenção meteorológica: FROM direction).
    """
    try:
        r = requests.get(
            OPENMETEO_FORECAST,
            params={
                "latitude":  ",".join(f"{v:.1f}" for v in GRID_LAT_FLAT),
                "longitude": ",".join(f"{v:.1f}" for v in GRID_LON_FLAT),
                "daily": (
                    "precipitation_sum,"
                    "wind_speed_10m_max,"
                    "wind_gusts_10m_max,"
                    "wind_direction_10m_dominant"
                ),
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

        def _arr(key: str) -> np.ndarray:
            flat = []
            for item in items:
                vals = item["daily"].get(key, [None] * n_days)
                flat.append([v if v is not None else np.nan for v in vals])
            return np.array(flat).reshape(n_lats, n_lons, n_days)

        wind_arr = _arr("wind_speed_10m_max")
        dir_arr  = _arr("wind_direction_10m_dominant")
        dir_rad  = np.deg2rad(dir_arr)

        # Componentes: convenção FROM → velocidade apontada em sentido contrário
        u_wind = -wind_arr * np.sin(dir_rad)   # eastward (km/h)
        v_wind = -wind_arr * np.cos(dir_rad)   # northward (km/h)

        return {
            "dates":  dates,
            "precip": _arr("precipitation_sum"),
            "wind":   wind_arr,
            "gusts":  _arr("wind_gusts_10m_max"),
            "u_wind": u_wind,
            "v_wind": v_wind,
        }

    except Exception:
        return {}


def build_wind_vectors(u_2d: np.ndarray, v_2d: np.ndarray,
                       scale: float = 0.009) -> tuple:
    """
    Constrói listas lat/lon para plotar vetores de vento como segmentos de linha
    com seta no extremo. Retorna (arrow_lats, arrow_lons).
    scale: graus por km/h (0.009 → 80 km/h ≈ 0.72°)
    """
    arrow_lats, arrow_lons = [], []
    for i, lat0 in enumerate(GRID_LATS):
        for j, lon0 in enumerate(GRID_LONS):
            ui = float(u_2d[i, j])
            vi = float(v_2d[i, j])
            if np.isnan(ui) or np.isnan(vi):
                continue
            dlat = vi * scale
            dlon = ui * scale
            lat1 = lat0 + dlat
            lon1 = lon0 + dlon

            # Haste do vetor
            arrow_lats += [lat0, lat1, None]
            arrow_lons += [lon0, lon1, None]

            # Seta: dois segmentos a ±150° da direção do vetor
            vec_len = np.sqrt(dlat ** 2 + dlon ** 2)
            if vec_len > 0.02:
                ah = 0.35 * vec_len
                angle = np.arctan2(dlat, dlon)  # ângulo em espaço lat/lon
                for da in [np.deg2rad(150), np.deg2rad(-150)]:
                    a = angle + da
                    arrow_lats += [lat1, lat1 + ah * np.sin(a), None]
                    arrow_lons += [lon1, lon1 + ah * np.cos(a), None]

    return arrow_lats, arrow_lons
