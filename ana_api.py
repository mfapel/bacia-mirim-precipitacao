"""
Acesso à API de telemetria da ANA (Agência Nacional de Águas).
Dados de nível (cm) e vazão (m³/s) em tempo quase real (15 min).
"""

import requests
import pandas as pd
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
# Calibração única: 24/02/2026 11:20
#   Profundidade in loco      = 100 cm (1 m)
#   Nível Eclusa São Gonçalo  =  13 cm
#   Offset                    =  87 cm   (depth = nivel_SG + 87)
# Hipótese k=1: variações de nível propagam-se uniformemente na lagoa.

SANGRADOURO_LAT   = -32.121142
SANGRADOURO_LON   = -52.599856
SANGRADOURO_NOME  = "Sangradouro — Santa Isabel"
CALIB_NIVEL_REF   = 13.0   # cm (Eclusa São Gonçalo em 24/02/2026 11:15)
CALIB_PROF_REF    = 100.0  # cm (medição in loco em 24/02/2026 11:20)
OFFSET            = CALIB_PROF_REF - CALIB_NIVEL_REF   # = 87 cm


def estimar_sangradouro(nivel_sg_cm: float) -> dict:
    """
    Estima profundidade (cm) no Sangradouro de Santa Isabel
    a partir do nível atual da Eclusa São Gonçalo.
    Retorna profundidade estimada e status (ativo / seco).
    """
    prof_est = nivel_sg_cm + OFFSET
    return {
        "profundidade_cm": max(0.0, prof_est),
        "ativo": prof_est > 0,
        "nivel_sg_ref": nivel_sg_cm,
        "offset_cm": OFFSET,
    }


@st.cache_data(ttl=900, show_spinner=False)
def get_sangradouro_serie(days: int = 30) -> pd.DataFrame:
    """Série temporal estimada de profundidade no Sangradouro."""
    df = get_nivel_serie("88690050", days=days)
    if df.empty:
        return pd.DataFrame()
    df = df.copy()
    # Remove zeros espúrios (sensor offline) — mantém apenas leituras > 0
    df = df[df["Nivel_cm"] > 0].copy()
    df["Prof_est_cm"] = (df["Nivel_cm"] + OFFSET).clip(lower=0)
    df["Prof_est_m"]  = df["Prof_est_cm"] / 100
    return df[["DataHora", "Nivel_cm", "Prof_est_cm", "Prof_est_m"]]
