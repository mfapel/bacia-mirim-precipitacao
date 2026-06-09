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

def _kb_from_threshold(nivel_seco: float) -> tuple[float, float] | None:
    """Calcula (k, b) para um threshold de secagem. Retorna None se inválido."""
    if nivel_seco >= CALIB_NIVEL_REF:
        return None
    k = (CALIB_PROF_REF - CALIB_DEPTH_MIN_CM) / (CALIB_NIVEL_REF - nivel_seco)
    b = CALIB_PROF_REF - k * CALIB_NIVEL_REF
    return round(k, 3), round(b, 1)


@st.cache_data(ttl=86400, show_spinner=False)   # recalcula 1×/dia
def calibrar_modelo_b(dias_historico: int = 180) -> dict:
    """
    Estima slope k (modelo central, p05) e faixa de incerteza (p01–p10).

    Faixa de incerteza: reflete a incerteza epistêmica sobre qual nível
    corresponde ao Sangradouro seco — p01 (threshold mais baixo → k maior)
    e p10 (threshold mais alto → k menor) são os limites plausíveis.

    Retorna dict com: k, b, nivel_seco (p05),
                      k_p01, b_p01, k_p10, b_p10,
                      nivel_seco_p01, nivel_seco_p10,
                      n_leituras, metodo
    """
    df = get_nivel_serie("88690050", days=dias_historico)
    if not df.empty:
        df = df[df["Nivel_cm"] > 0]

    fallback = {
        "k": 1.0, "b": float(OFFSET), "nivel_seco": float(-OFFSET),
        "k_p01": 1.0, "b_p01": float(OFFSET),
        "k_p10": 1.0, "b_p10": float(OFFSET),
        "nivel_seco_p01": float(-OFFSET), "nivel_seco_p10": float(-OFFSET),
        "n_leituras": 0, "metodo": "fallback_k1",
    }

    if df.empty or len(df) < 20:
        return fallback

    p01  = float(df["Nivel_cm"].quantile(0.01))
    p05  = float(df["Nivel_cm"].quantile(0.05))
    p10  = float(df["Nivel_cm"].quantile(0.10))

    kb_central = _kb_from_threshold(p05)
    if kb_central is None:
        fallback.update({"nivel_seco": round(p05, 1), "n_leituras": int(len(df)),
                         "metodo": "fallback_k1_sem_variacao"})
        return fallback

    k_c, b_c = kb_central
    kb_p01 = _kb_from_threshold(p01) or (k_c, b_c)
    kb_p10 = _kb_from_threshold(p10) or (k_c, b_c)

    return {
        "k": k_c,      "b": b_c,      "nivel_seco": round(p05, 1),
        "k_p01": kb_p01[0], "b_p01": kb_p01[1], "nivel_seco_p01": round(p01, 1),
        "k_p10": kb_p10[0], "b_p10": kb_p10[1], "nivel_seco_p10": round(p10, 1),
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


@st.cache_data(ttl=900, show_spinner=False)
def get_sangradouro_serie_b(days: int = 30) -> pd.DataFrame:
    """
    Série temporal — Modelo B com faixa de incerteza p01–p10.

    Faixa de incerteza (não IC estatístico clássico):
      Limite inferior: modelo calibrado com p10 como threshold de secagem
      Limite superior: modelo calibrado com p01 como threshold de secagem
    Captura a incerteza epistêmica sobre quando exatamente o Sangradouro seca.
    """
    df = get_nivel_serie("88690050", days=days)
    if df.empty:
        return pd.DataFrame()
    modelo = calibrar_modelo_b()
    df = df[df["Nivel_cm"] > 0].copy()
    nivel = df["Nivel_cm"]

    df["Prof_est_cm_A"] = (nivel + OFFSET).clip(lower=0)
    df["Prof_est_cm_B"] = (modelo["k"]    * nivel + modelo["b"]).clip(lower=0)
    df["Prof_est_m_A"]  = df["Prof_est_cm_A"] / 100
    df["Prof_est_m_B"]  = df["Prof_est_cm_B"] / 100

    # Faixa de incerteza: limites p01 e p10
    prof_p01 = (modelo["k_p01"] * nivel + modelo["b_p01"]).clip(lower=0) / 100
    prof_p10 = (modelo["k_p10"] * nivel + modelo["b_p10"]).clip(lower=0) / 100
    df["Prof_ci_low_m"]  = np.minimum(prof_p01, prof_p10)
    df["Prof_ci_high_m"] = np.maximum(prof_p01, prof_p10)

    return df[["DataHora", "Nivel_cm",
               "Prof_est_cm_A", "Prof_est_cm_B",
               "Prof_est_m_A",  "Prof_est_m_B",
               "Prof_ci_low_m", "Prof_ci_high_m"]]
