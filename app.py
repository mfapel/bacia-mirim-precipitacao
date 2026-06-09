"""
Monitoramento de Precipitação — Bacia Mirim
Estações automáticas INMET · Streamlit MVP
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import pandas as pd
import json
from datetime import datetime, date

from inmet_api import get_stations, get_accumulated, get_daily_series
from ana_api import (get_nivel_serie, get_nivel_atual, ESTACOES_NIVEL,
                     estimar_sangradouro, get_sangradouro_serie,
                     SANGRADOURO_NOME, SANGRADOURO_LAT, SANGRADOURO_LON,
                     CALIB_NIVEL_REF, CALIB_PROF_REF, OFFSET)

# ── Carrega sub-bacias ────────────────────────────────────────────────────────
@st.cache_data
def load_bacias():
    with open("bacias.geojson", "r") as f:
        return json.load(f)

# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Precipitação · Bacia Mirim",
    page_icon="🌧️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .block-container { padding-top: 1.5rem; }
    .stMetric { background: #f0f4f8; border-radius: 8px; padding: 0.5rem 1rem; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("logo.png", width=180)
    st.title("Bacia Mirim")
    st.caption("Monitoramento de precipitação")

    PERIOD_LABELS = {
        1: "Últimas 24h",
        3: "Últimos 3 dias",
        7: "Últimos 7 dias",
        15: "Últimos 15 dias",
        30: "Últimos 30 dias",
        0: "Acumulado desde 24fev",
    }
    period = st.selectbox(
        "Período de análise",
        options=list(PERIOD_LABELS.keys()),
        index=2,
        format_func=lambda x: PERIOD_LABELS[x],
    )
    # Converte período especial (0) em número de dias desde 24/02/2026
    FEB24 = date(2026, 2, 24)
    period_days = (date.today() - FEB24).days if period == 0 else period

    st.markdown("---")
    st.markdown("**Dados:** Estações automáticas INMET")
    st.markdown("**Precipitação:** Open-Meteo (ERA5)")
    st.markdown(f"**Consultado em:** {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    st.markdown("---")

    # Legenda das sub-bacias
    st.markdown("**Sub-bacias da Lagoa Mirim**")
    bacias_sidebar = load_bacias()
    for feat in bacias_sidebar["features"]:
        p = feat["properties"]
        st.markdown(
            f'<span style="display:inline-block;width:12px;height:12px;'
            f'background:{p["cor"]};border-radius:2px;margin-right:6px;'
            f'opacity:0.8"></span>{p["nome"]}',
            unsafe_allow_html=True,
        )
    st.markdown("---")
    st.markdown(
        "⚠️ Dados não validados. Uso informativo.",
        help="Estações automáticas transmitem dados brutos sem controle de qualidade.",
    )

# ── Cabeçalho ─────────────────────────────────────────────────────────────────
st.title("🌧️ Precipitação — Bacia Mirim")
st.caption(
    "Estações automáticas INMET na região da Bacia Hidrográfica da Lagoa Mirim "
    "(RS/Brasil · fronteira com Uruguai)"
)

# ── Carregamento de estações ──────────────────────────────────────────────────
with st.spinner("Buscando estações INMET na região..."):
    stations_df = get_stations()

if stations_df.empty:
    st.error(
        "Não foi possível carregar as estações. "
        "Verifique a conectividade com a API do INMET e tente novamente."
    )
    st.stop()

st.success(f"{len(stations_df)} estações encontradas na região da Bacia Mirim.")

# ── Acumulado por estação ─────────────────────────────────────────────────────
@st.cache_data(ttl=1800, show_spinner=False)
def load_all_accumulated(coords: tuple, days: int) -> dict:
    # coords: tuple of (code, lat, lon)
    return {
        code: get_accumulated(lat, lon, days)
        for code, lat, lon in coords
    }


progress = st.progress(0, text=f"Calculando acumulado dos últimos {period_days} dias...")
coords = tuple(
    (row["CD_ESTACAO"], row["VL_LATITUDE"], row["VL_LONGITUDE"])
    for _, row in stations_df.iterrows()
)
accumulated = load_all_accumulated(coords, period_days)
progress.empty()

stations_df["ACUMULADO_MM"] = stations_df["CD_ESTACAO"].map(accumulated).fillna(0)


# ── Paleta de cores por intensidade ──────────────────────────────────────────
# Limiares: (limite_superior, cor) — último item é o "acima de tudo"
SCALE_SHORT = [
    (0,   "#CCCCCC", "Sem chuva"),
    (10,  "#A8D8EA", "0 – 10 mm"),
    (30,  "#4A90D9", "10 – 30 mm"),
    (60,  "#1A5276", "30 – 60 mm"),
    (100, "#E67E22", "60 – 100 mm"),
    (None,"#C0392B", "> 100 mm"),
]
SCALE_FEB24 = [
    (0,   "#CCCCCC", "Sem chuva"),
    (100, "#A8D8EA", "0 – 100 mm"),
    (250, "#4A90D9", "100 – 250 mm"),
    (400, "#1A5276", "250 – 400 mm"),
    (600, "#E67E22", "400 – 600 mm"),
    (None,"#C0392B", "> 600 mm"),
]

def precip_color(mm: float, scale: list) -> str:
    if mm == 0:
        return scale[0][1]
    for limit, color, _ in scale[1:]:
        if limit is None or mm < limit:
            return color
    return scale[-1][1]

def build_legend(scale: list) -> str:
    items = f'<span style="color:{scale[0][1]};font-size:18px">●</span> {scale[0][2]}<br>'
    for _, color, label in scale[1:]:
        items += f'<span style="color:{color};font-size:18px">●</span> {label}<br>'
    return (
        '<div style="position:fixed;bottom:30px;left:30px;z-index:1000;'
        'background:white;padding:12px 16px;border-radius:10px;'
        'box-shadow:0 2px 8px rgba(0,0,0,0.25);font-size:13px;line-height:1.8;">'
        f'<b>Acumulado (mm)</b><br>{items}</div>'
    )


# ── Nível da Lagoa Mirim — indicadores em tempo real ─────────────────────────
st.subheader("🌊 Nível da Lagoa Mirim")
st.caption("Fonte: ANA — Telemetria automática · Atualização a cada 15 min")

nivel_cols = st.columns(4)

# Eclusa São Gonçalo — proxy direto do nível da lagoa
with st.spinner("Buscando nível atual..."):
    sg = get_nivel_atual("88690050")
    pp = get_nivel_atual("88260000")

if sg:
    nivel_cols[0].metric(
        "Nível — Eclusa São Gonçalo",
        f"{sg['nivel_cm']:.0f} cm",
        help="Proxy do nível da Lagoa Mirim na saída para Lagoa dos Patos",
    )
    nivel_cols[1].metric(
        "Última leitura",
        sg['data_hora'].strftime("%d/%m %H:%M") if pd.notna(sg['data_hora']) else "—",
    )
if pp:
    nivel_cols[2].metric(
        "Nível — Rio Jaguarão (Passo Pedras)",
        f"{pp['nivel_cm']:.0f} cm",
        help="Nível do principal afluente da Lagoa Mirim",
    )
    nivel_cols[3].metric(
        "Vazão — Rio Jaguarão",
        f"{pp['vazao_m3s']:.1f} m³/s" if pp.get('vazao_m3s') and pd.notna(pp['vazao_m3s']) else "—",
    )

st.markdown("---")

# ── Gráficos de nível ─────────────────────────────────────────────────────────
st.subheader("📈 Série Temporal de Nível")
nivel_period = st.select_slider(
    "Período",
    options=[7, 15, 30, 60, 90],
    value=30,
    format_func=lambda x: f"{x} dias",
)

ncol1, ncol2 = st.columns(2)

with ncol1:
    st.markdown("**Eclusa São Gonçalo** · nível da Lagoa Mirim")
    with st.spinner("Carregando..."):
        df_sg = get_nivel_serie("88690050", days=nivel_period)
    if not df_sg.empty:
        fig_sg = go.Figure()
        fig_sg.add_trace(go.Scatter(
            x=df_sg["DataHora"], y=df_sg["Nivel_cm"],
            mode="lines", line=dict(color="#1A5276", width=1.5),
            name="Nível (cm)",
            hovertemplate="%{x|%d/%m %H:%M}<br>Nível: %{y:.0f} cm<extra></extra>",
        ))
        fig_sg.update_layout(
            yaxis_title="Nível (cm)", xaxis_title=None,
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(gridcolor="#EEEEEE"),
            xaxis=dict(gridcolor="#EEEEEE"),
            showlegend=False,
        )
        st.plotly_chart(fig_sg, use_container_width=True)
    else:
        st.warning("Sem dados disponíveis.")

with ncol2:
    st.markdown("**Passo das Pedras** · Rio Jaguarão — nível e vazão")
    with st.spinner("Carregando..."):
        df_pp = get_nivel_serie("88260000", days=nivel_period)
    if not df_pp.empty:
        fig_pp = go.Figure()
        fig_pp.add_trace(go.Scatter(
            x=df_pp["DataHora"], y=df_pp["Nivel_cm"],
            mode="lines", line=dict(color="#E15759", width=1.5),
            name="Nível (cm)", yaxis="y",
            hovertemplate="%{x|%d/%m %H:%M}<br>Nível: %{y:.0f} cm<extra></extra>",
        ))
        if df_pp["Vazao_m3s"].notna().any():
            fig_pp.add_trace(go.Scatter(
                x=df_pp["DataHora"], y=df_pp["Vazao_m3s"],
                mode="lines", line=dict(color="#4A90D9", width=1.5, dash="dot"),
                name="Vazão (m³/s)", yaxis="y2",
                hovertemplate="%{x|%d/%m %H:%M}<br>Vazão: %{y:.1f} m³/s<extra></extra>",
            ))
        fig_pp.update_layout(
            yaxis=dict(title="Nível (cm)", gridcolor="#EEEEEE", color="#E15759"),
            yaxis2=dict(title="Vazão (m³/s)", overlaying="y", side="right",
                        color="#4A90D9", showgrid=False),
            xaxis=dict(gridcolor="#EEEEEE"),
            height=300, margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.8)"),
        )
        st.plotly_chart(fig_pp, use_container_width=True)
    else:
        st.warning("Sem dados disponíveis.")

st.markdown("---")

# ── Sangradouro Santa Isabel — profundidade estimada ─────────────────────────
st.subheader("📍 Sangradouro — Santa Isabel (estimativa)")
st.caption(
    "Ponto não monitorado · Profundidade estimada a partir do nível da Eclusa São Gonçalo · "
    "Modelo calibrado com 1 medição in loco"
)

sc1, sc2, sc3 = st.columns(3)

if sg:
    est = estimar_sangradouro(sg['nivel_cm'])
    sc1.metric(
        "Profundidade estimada",
        f"{est['profundidade_cm'] / 100:.2f} m",
        help="Profundidade no Sangradouro de Santa Isabel (ponto não monitorado)",
    )
    sc2.metric(
        "Status",
        "🟢 Ativo" if est['ativo'] else "⚪ Seco",
        help="Ativo = sangradouro com lâmina d'água",
    )
    sc3.metric(
        "Nível ref. Eclusa São Gonçalo",
        f"{sg['nivel_cm']:.0f} cm",
        help=f"Nível usado como referência para estimativa (offset aplicado: {OFFSET:.0f} cm)",
    )
else:
    st.warning("Dados da Eclusa São Gonçalo indisponíveis — estimativa não calculada.")

with st.spinner("Carregando série do Sangradouro..."):
    df_sang = get_sangradouro_serie(days=nivel_period)

if not df_sang.empty:
    fig_sang = go.Figure()
    fig_sang.add_trace(go.Scatter(
        x=df_sang["DataHora"],
        y=df_sang["Prof_est_m"],
        mode="lines",
        fill="tozeroy",
        line=dict(color="#8E44AD", width=1.5),
        fillcolor="rgba(142,68,173,0.12)",
        name="Profundidade estimada (m)",
        hovertemplate="%{x|%d/%m %H:%M}<br>Prof. estimada: %{y:.2f} m<extra></extra>",
    ))
    fig_sang.update_layout(
        yaxis_title="Profundidade estimada (m)",
        xaxis_title=None,
        height=280,
        margin=dict(l=0, r=0, t=10, b=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis=dict(gridcolor="#EEEEEE", rangemode="tozero"),
        xaxis=dict(gridcolor="#EEEEEE"),
        showlegend=False,
    )
    st.plotly_chart(fig_sang, use_container_width=True)
else:
    st.warning("Sem dados disponíveis para o Sangradouro no período selecionado.")

st.caption(
    f"⚠️ Estimativa baseada em 1 medição in loco (24/02/2026 11:20 — 1,00 m de profundidade) e "
    f"leitura simultânea na Eclusa São Gonçalo ({CALIB_NIVEL_REF:.0f} cm às 11:15). "
    f"Offset: {OFFSET:.0f} cm · Hipótese k=1 (variações de nível propagam-se uniformemente)."
)

st.markdown("---")

# ── Métricas de precipitação ──────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
col1.metric("Estações monitoradas", len(stations_df))
col2.metric(
    "Maior acumulado",
    f"{stations_df['ACUMULADO_MM'].max():.1f} mm",
    stations_df.loc[stations_df["ACUMULADO_MM"].idxmax(), "DC_NOME"],
)
col3.metric(
    "Média regional",
    f"{stations_df['ACUMULADO_MM'].mean():.1f} mm",
)
col4.metric(
    "Estações com chuva",
    int((stations_df["ACUMULADO_MM"] > 0).sum()),
)

st.markdown("---")

# ── Mapa + Gráfico de detalhe ─────────────────────────────────────────────────
col_map, col_chart = st.columns([3, 2])

with col_map:
    st.subheader("Mapa de Estações")
    st.caption("Clique em uma estação para ver a série temporal")

    center_lat = stations_df["VL_LATITUDE"].mean()
    center_lon = stations_df["VL_LONGITUDE"].mean()

    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=8,
        tiles="CartoDB positron",
    )

    # ── Camada de sub-bacias (adicionada ANTES dos marcadores) ────────────────
    bacias_data = load_bacias()
    for feat in bacias_data["features"]:
        nome = feat["properties"]["nome"]
        cor  = feat["properties"]["cor"]
        area = feat["properties"]["area_km2"]
        folium.GeoJson(
            feat,
            name=nome,
            style_function=lambda f, c=cor: {
                "fillColor": c,
                "color": c,
                "weight": 1.2,
                "fillOpacity": 0.20,
                "dashArray": "4 4",
            },
            tooltip=folium.GeoJsonTooltip(
                fields=["nome", "area_km2"],
                aliases=["Sub-bacia:", "Área (km²):"],
                localize=True,
            ),
        ).add_to(m)

    scale = SCALE_FEB24 if period == 0 else SCALE_SHORT

    for _, row in stations_df.iterrows():
        mm = row["ACUMULADO_MM"]
        color = precip_color(mm, scale)
        folium.CircleMarker(
            location=[row["VL_LATITUDE"], row["VL_LONGITUDE"]],
            radius=11,
            color="white",
            weight=1.5,
            fill=True,
            fill_color=color,
            fill_opacity=0.9,
            tooltip=folium.Tooltip(
                f"<b>{row['DC_NOME']}</b><br>"
                f"Código: {row['CD_ESTACAO']}<br>"
                f"Acumulado: <b>{mm:.1f} mm</b>",
                sticky=True,
            ),
            popup=folium.Popup(
                f"<b>{row['DC_NOME']}</b><br>"
                f"Código: {row['CD_ESTACAO']}<br>"
                f"Lat: {row['VL_LATITUDE']:.4f} | Lon: {row['VL_LONGITUDE']:.4f}<br>"
                f"Acumulado {period_days}d: <b>{mm:.1f} mm</b>",
                max_width=220,
            ),
        ).add_to(m)

    # ── Marcador Sangradouro Santa Isabel ────────────────────────────────────
    if sg:
        est_map = estimar_sangradouro(sg['nivel_cm'])
        prof_m_map = est_map['profundidade_cm'] / 100
        status_map = "Ativo" if est_map['ativo'] else "Seco"
        folium.Marker(
            location=[SANGRADOURO_LAT, SANGRADOURO_LON],
            tooltip=folium.Tooltip(
                f"<b>{SANGRADOURO_NOME}</b><br>"
                f"Profundidade estimada: <b>{prof_m_map:.2f} m</b><br>"
                f"Status: {status_map}<br>"
                f"<i>Modelo calibrado em 24/02/2026</i>",
                sticky=True,
            ),
            popup=folium.Popup(
                f"<b>{SANGRADOURO_NOME}</b><br>"
                f"Lat: {SANGRADOURO_LAT:.4f} | Lon: {SANGRADOURO_LON:.4f}<br>"
                f"Profundidade estimada: <b>{prof_m_map:.2f} m</b><br>"
                f"Status: {status_map}<br>"
                f"<i>Ref.: Eclusa São Gonçalo ({sg['nivel_cm']:.0f} cm)<br>"
                f"Calibração: 1 medição in loco (24/02/2026)</i>",
                max_width=250,
            ),
            icon=folium.DivIcon(
                html=(
                    '<div style="font-size:22px;color:#8E44AD;'
                    'text-shadow:1px 1px 0 #fff,-1px -1px 0 #fff,'
                    '1px -1px 0 #fff,-1px 1px 0 #fff;line-height:1">★</div>'
                ),
                icon_size=(26, 26),
                icon_anchor=(13, 13),
            ),
        ).add_to(m)

    # Legenda dinâmica conforme escala do período
    legend_html = build_legend(scale)
    m.get_root().html.add_child(folium.Element(legend_html))

    map_result = st_folium(m, width=700, height=520, returned_objects=["last_object_clicked_popup"])

with col_chart:
    st.subheader("Série Temporal Diária")

    # Determina estação selecionada via clique ou default (maior acumulado)
    clicked_code = None
    clicked_name = None

    if map_result and map_result.get("last_object_clicked_popup"):
        popup_html = map_result["last_object_clicked_popup"]
        for _, row in stations_df.iterrows():
            if row["CD_ESTACAO"] in popup_html:
                clicked_code = row["CD_ESTACAO"]
                clicked_name = row["DC_NOME"]
                break

    if not clicked_code:
        idx = stations_df["ACUMULADO_MM"].idxmax()
        clicked_code = stations_df.loc[idx, "CD_ESTACAO"]
        clicked_name = stations_df.loc[idx, "DC_NOME"]
        st.info("Clique no mapa para selecionar uma estação.")

    sel = stations_df[stations_df["CD_ESTACAO"] == clicked_code].iloc[0]
    with st.spinner(f"Carregando dados de {clicked_name}..."):
        daily_df = get_daily_series(sel["VL_LATITUDE"], sel["VL_LONGITUDE"], period_days)

    total_mm = daily_df["Precipitação (mm)"].sum() if not daily_df.empty else 0
    st.metric(
        label=clicked_name,
        value=f"{total_mm:.1f} mm",
        delta=f"Código {clicked_code}",
    )

    if daily_df.empty:
        st.warning("Sem dados para esta estação no período selecionado.")
    else:
        fig = go.Figure()
        fig.add_trace(
            go.Bar(
                x=daily_df["Data"],
                y=daily_df["Precipitação (mm)"],
                marker_color="#4A90D9",
                marker_line_color="#1A5276",
                marker_line_width=0.5,
                name="Precipitação diária",
                hovertemplate="%{x|%d/%m/%Y}<br>%{y:.1f} mm<extra></extra>",
            )
        )
        fig.update_layout(
            xaxis_title=None,
            yaxis_title="mm",
            height=360,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="white",
            paper_bgcolor="white",
            yaxis=dict(gridcolor="#EEEEEE", zeroline=True, zerolinecolor="#DDDDDD"),
            xaxis=dict(gridcolor="#EEEEEE"),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)


# ── Rodapé ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Estações: INMET — Instituto Nacional de Meteorologia · "
    "Precipitação: Open-Meteo (ERA5 + NWP) · Uso informativo"
)
