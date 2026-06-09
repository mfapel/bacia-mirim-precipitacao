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
# Ponto de calibração in loco: 24/02/2026 11:20
#   Profundidade medida       = 100 cm (1 m)
#   Nível Eclusa São Gonçalo  =  13 cm (leitura 11:15)
#
# Modelo A (k=1, offset fixo):  prof = nivel_SG + 87
# Modelo B (2 pontos):          prof = k × nivel_SG + b
#   Segundo ponto: percentil 5 histórico de São Gonçalo → prof ≈ 0 (seco)

SANGRADOURO_LAT    = -32.121142
SANGRADOURO_LON    = -52.599856
SANGRADOURO_NOME   = "Sangradouro — Santa Isabel"
CALIB_NIVEL_REF    = 13.0    # cm (Eclusa São Gonçalo em 24/02/2026 11:15)
CALIB_PROF_REF     = 100.0   # cm (medição in loco em 24/02/2026 11:20)
OFFSET             = CALIB_PROF_REF - CALIB_NIVEL_REF   # = 87 cm (Modelo A)
CALIB_DEPTH_MIN_CM = 0.0     # cm — assumido: Sangradouro seco no mínimo histórico


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


# ── Modelo B (2 pontos: calibração + mínimo histórico) ───────────────────────

@st.cache_data(ttl=86400, show_spinner=False)   # recalcula 1×/dia
def calibrar_modelo_b(dias_historico: int = 180) -> dict:
    """
    Estima slope k a partir de dois pontos de calibração:
      P1: (CALIB_NIVEL_REF, CALIB_PROF_REF)  — medição in loco
      P2: (percentil 5 histórico, 0 cm)       — mínimo histórico ≈ seco

    Retorna dict com: k, b, nivel_seco, n_leituras, metodo
    """
    df = get_nivel_serie("88690050", days=dias_historico)
    if not df.empty:
        df = df[df["Nivel_cm"] > 0]

    if df.empty or len(df) < 20:
        return {
            "k": 1.0, "b": float(OFFSET),
            "nivel_seco": float(-OFFSET),
            "n_leituras": 0, "metodo": "fallback_k1",
        }

    nivel_seco = float(df["Nivel_cm"].quantile(0.05))

    if nivel_seco >= CALIB_NIVEL_REF:
        # Série sem variação abaixo do ponto de calibração → k=1
        return {
            "k": 1.0, "b": float(OFFSET),
            "nivel_seco": round(nivel_seco, 1),
            "n_leituras": int(len(df)), "metodo": "fallback_k1_sem_variacao",
        }

    k = (CALIB_PROF_REF - CALIB_DEPTH_MIN_CM) / (CALIB_NIVEL_REF - nivel_seco)
    b = CALIB_PROF_REF - k * CALIB_NIVEL_REF   # = -k × nivel_seco

    return {
        "k": round(k, 3),
        "b": round(b, 1),
        "nivel_seco": round(nivel_seco, 1),
        "n_leituras": int(len(df)),
        "metodo": "opcao_b_2pontos",
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


@st.cache_data(ttl=86400, show_spinner=False)
def _bootstrap_sigma_k(n_boot: int = 2000) -> float:
    """
    Bootstrap de σ(k): incerteza do slope estimado via variabilidade do p05 histórico.
    Usa semente fixa para reprodutibilidade. Cache 24h.

    Base analítica:
        k = CALIB_PROF_REF / (CALIB_NIVEL_REF − p05)
        σ_prof(nivel_SG) = σ_k × |nivel_SG − CALIB_NIVEL_REF|
    """
    df = get_nivel_serie("88690050", days=180)
    if df.empty:
        return 0.0
    hist = df[df["Nivel_cm"] > 0]["Nivel_cm"].values
    if len(hist) < 20:
        return 0.0

    rng = np.random.default_rng(42)
    k_samples = []
    for _ in range(n_boot):
        sample = rng.choice(hist, size=len(hist), replace=True)
        p05 = float(np.percentile(sample, 5))
        if p05 < CALIB_NIVEL_REF:
            k_samples.append(CALIB_PROF_REF / (CALIB_NIVEL_REF - p05))

    return float(np.std(k_samples)) if len(k_samples) > 10 else 0.0


@st.cache_data(ttl=900, show_spinner=False)
def get_sangradouro_serie_b(days: int = 30) -> pd.DataFrame:
    """
    Série temporal — Modelo B com IC 95% bootstrap.

    Intervalo de confiança analítico:
        σ_prof = σ_k × |nivel_SG − CALIB_NIVEL_REF|
        IC 95% = prof_B ± 1.96 × σ_prof
    O intervalo é zero no ponto de calibração e cresce com a distância dele.
    """
    df = get_nivel_serie("88690050", days=days)
    if df.empty:
        return pd.DataFrame()
    modelo = calibrar_modelo_b()
    df = df[df["Nivel_cm"] > 0].copy()

    df["Prof_est_cm_A"] = (df["Nivel_cm"] + OFFSET).clip(lower=0)
    df["Prof_est_cm_B"] = (modelo["k"] * df["Nivel_cm"] + modelo["b"]).clip(lower=0)
    df["Prof_est_m_A"]  = df["Prof_est_cm_A"] / 100
    df["Prof_est_m_B"]  = df["Prof_est_cm_B"] / 100

    # IC 95% — propagação analítica da incerteza de k
    sigma_k  = _bootstrap_sigma_k()
    half_ci  = 1.96 * sigma_k * (df["Nivel_cm"] - CALIB_NIVEL_REF).abs() / 100
    df["Prof_ci_low_m"]  = (df["Prof_est_m_B"] - half_ci).clip(lower=0)
    df["Prof_ci_high_m"] = df["Prof_est_m_B"] + half_ci

    return df[["DataHora", "Nivel_cm",
               "Prof_est_cm_A", "Prof_est_cm_B",
               "Prof_est_m_A",  "Prof_est_m_B",
               "Prof_ci_low_m", "Prof_ci_high_m"]]
