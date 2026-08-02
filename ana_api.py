"""
Acesso à API de telemetria da ANA (Agência Nacional de Águas).
Dados de nível (cm) e vazão (m³/s) em tempo quase real (15 min).
"""

import requests
import pandas as pd
import numpy as np
import xml.etree.ElementTree as ET
from datetime import datetime, timedelta
import streamlit as st

ANA_URL = "https://telemetriaws1.ana.gov.br/ServiceANA.asmx/DadosHidrometeorologicos"
HEADERS = {"User-Agent": "Mozilla/5.0"}

# Estações de nível relevantes para a Lagoa Mirim
ESTACOES_NIVEL = {
    "88690050": {
        "nome": "Eclusa São Gonçalo",
        "descricao": "Saída da Lagoa Mirim → Lagoa dos Patos",
        "rio": "Canal São Gonçalo",
        "lat": -31.81110,
        "lon": -52.38920,
    },
    "88260000": {
        "nome": "Passo das Pedras",
        "descricao": "Principal afluente da Lagoa Mirim",
        "rio": "Rio Jaguarão",
        "lat": -32.51940,
        "lon": -53.45580,
    },
}


@st.cache_data(ttl=900, show_spinner=False)  # cache 15 min
def get_nivel_serie(cod: str, days: int = 30) -> pd.DataFrame:
    """
    Retorna série temporal de nível (cm) e vazão (m³/s) de uma estação ANA.
    Atualização a cada 15 minutos.
    """
    end = datetime.now()
    start = end - timedelta(days=days)

    try:
        r = requests.get(
            ANA_URL,
            params={
                "codEstacao": cod,
                "dataInicio": start.strftime("%d/%m/%Y"),
                "dataFim": end.strftime("%d/%m/%Y"),
            },
            headers=HEADERS,
            timeout=20,
        )
        root = ET.fromstring(r.text)
        rows = root.findall(".//DadosHidrometereologicos")
    except Exception:
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    records = []
    for row in rows:
        d = {c.tag: c.text for c in row}
        records.append({
            "DataHora": d.get("DataHora"),
            "Nivel_cm": d.get("Nivel"),
            "Vazao_m3s": d.get("Vazao"),
        })

    df = pd.DataFrame(records)
    df["DataHora"] = pd.to_datetime(df["DataHora"], errors="coerce")
    df["Nivel_cm"] = pd.to_numeric(df["Nivel_cm"], errors="coerce")
    df["Vazao_m3s"] = pd.to_numeric(df["Vazao_m3s"], errors="coerce")
    df = df.dropna(subset=["DataHora"]).sort_values("DataHora").reset_index(drop=True)
    return df


def get_nivel_atual(cod: str) -> dict:
    """Retorna último registro disponível de uma estação."""
    df = get_nivel_serie(cod, days=2)
    if df.empty:
        return {}
    last = df.iloc[-1]
    return {
        "nivel_cm": last["Nivel_cm"],
        "vazao_m3s": last["Vazao_m3s"],
        "data_hora": last["DataHora"],
    }


# ── Modelo de estimativa — Sangradouro Santa Isabel ───────────────────────────
# Medições de campo (dois pontos reais):
#   1ª  24/02/2026 11:20 → nivel_SG=13 cm,  prof=100 cm (1,00 m)
#   2ª  30/07/2026 12:30 → nivel_SG=131 cm, prof=180 cm (1,80 m)
#
# Modelo A (k=1, offset fixo):  prof = nivel_SG + 87
# Modelo B (2 medições campo):   prof = k × nivel_SG + b
#   k = (180−100)/(131−13) = 80/118 ≈ 0.678
#   b = 100 − 0.678×13 ≈ 91.2 cm

SANGRADOURO_LAT    = -32.121142
SANGRADOURO_LON    = -52.599856
SANGRADOURO_NOME   = "Sangradouro — Santa Isabel"

# 1ª medição de campo — 24/02/2026 11:20
CALIB_NIVEL_REF    = 13.0    # cm (Eclusa São Gonçalo 11:15)
CALIB_PROF_REF     = 100.0   # cm

# 2ª medição de campo — 30/07/2026 12:30
CALIB2_NIVEL_REF   = 131.0   # cm (Eclusa São Gonçalo 12:30)
CALIB2_PROF_REF    = 180.0   # cm

OFFSET             = CALIB_PROF_REF - CALIB_NIVEL_REF   # = 87 cm (Modelo A)
CALIB_INCERTEZA_CM = 15.0    # cm — incerteza estimada por medição de campo (±15 cm)


# ── Modelo A (k=1, mantido para comparação / fallback) ────────────────────────

def estimar_sangradouro(nivel_sg_cm: float) -> dict:
    """Modelo A: offset fixo, k=1."""
    prof_est = nivel_sg_cm + OFFSET
    return {
        "profundidade_cm": max(0.0, prof_est),
        "ativo": prof_est > 0,
        "nivel_sg_ref": nivel_sg_cm,
        "offset_cm": OFFSET,
    }


@st.cache_data(ttl=900, show_spinner=False)
def get_sangradouro_serie(days: int = 30) -> pd.DataFrame:
    """Série temporal — Modelo A (k=1)."""
    df = get_nivel_serie("88690050", days=days)
    if df.empty:
        return pd.DataFrame()
    df = df[df["Nivel_cm"] > 0].copy()
    df["Prof_est_cm"] = (df["Nivel_cm"] + OFFSET).clip(lower=0)
    df["Prof_est_m"]  = df["Prof_est_cm"] / 100
    return df[["DataHora", "Nivel_cm", "Prof_est_cm", "Prof_est_m"]]


# ── Modelo B (2 medições de campo reais) ─────────────────────────────────────

# Slope e intercepto calculados diretamente dos dois pontos reais:
_K_CAMPO = (CALIB2_PROF_REF - CALIB_PROF_REF) / (CALIB2_NIVEL_REF - CALIB_NIVEL_REF)
_B_CAMPO = CALIB_PROF_REF - _K_CAMPO * CALIB_NIVEL_REF


@st.cache_data(ttl=86400, show_spinner=False)   # recalcula 1×/dia
def calibrar_modelo_b(dias_historico: int = 180) -> dict:
    """
    Modelo B calibrado com 2 medições de campo reais.
    nivel_seco (p05 histórico) usado apenas como threshold de ativação.
    Retorna dict com: k, b, nivel_seco, n_leituras, metodo
    """
    df = get_nivel_serie("88690050", days=dias_historico)
    if not df.empty:
        df = df[df["Nivel_cm"] > 0]

    n = int(len(df)) if not df.empty else 0
    nivel_seco = float(df["Nivel_cm"].quantile(0.05)) if n >= 20 else round(_B_CAMPO / -_K_CAMPO, 1)

    return {
        "k": round(_K_CAMPO, 3),
        "b": round(_B_CAMPO, 1),
        "nivel_seco": round(nivel_seco, 1),
        "n_leituras": n,
        "metodo": "opcao_b_2medicoes_campo",
    }


def estimar_sangradouro_b(nivel_sg_cm: float, modelo: dict) -> dict:
    """Modelo B: slope calibrado com mínimo histórico."""
    prof_est = modelo["k"] * nivel_sg_cm + modelo["b"]
    return {
        "profundidade_cm": max(0.0, prof_est),
        "ativo": prof_est > 0,
        "nivel_sg_ref": nivel_sg_cm,
        "k": modelo["k"],
        "b": modelo["b"],
        "nivel_seco": modelo["nivel_seco"],
        "metodo": modelo["metodo"],
    }


@st.cache_data(ttl=900, show_spinner=False)
def get_sangradouro_serie_b(days: int = 30) -> pd.DataFrame:
    """
    Série temporal — Modelo B com faixa de incerteza ±CALIB_INCERTEZA_CM.

    Faixa: propagação direta da incerteza da medição in loco (±15 cm).
    prof_low  = prof_B − 15 cm  (se medição fosse 85 cm)
    prof_high = prof_B + 15 cm  (se medição fosse 115 cm)
    Resulta em banda constante e visível, interpretável fisicamente.
    """
    df = get_nivel_serie("88690050", days=days)
    if df.empty:
        return pd.DataFrame()
    modelo = calibrar_modelo_b()
    df = df[df["Nivel_cm"] > 0].copy()
    nivel = df["Nivel_cm"]

    df["Prof_est_cm_A"] = (nivel + OFFSET).clip(lower=0)
    df["Prof_est_cm_B"] = (modelo["k"] * nivel + modelo["b"]).clip(lower=0)
    df["Prof_est_m_A"]  = df["Prof_est_cm_A"] / 100
    df["Prof_est_m_B"]  = df["Prof_est_cm_B"] / 100

    # Faixa ±CALIB_INCERTEZA_CM — propagação da incerteza da medição de campo
    delta_m = CALIB_INCERTEZA_CM / 100
    df["Prof_ci_low_m"]  = (df["Prof_est_m_B"] - delta_m).clip(lower=0)
    df["Prof_ci_high_m"] = df["Prof_est_m_B"] + delta_m

    return df[["DataHora", "Nivel_cm",
               "Prof_est_cm_A", "Prof_est_cm_B",
               "Prof_est_m_A",  "Prof_est_m_B",
               "Prof_ci_low_m", "Prof_ci_high_m"]]
