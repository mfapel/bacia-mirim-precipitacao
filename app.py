"""
Monitoramento de Precipitação — Bacia Mirim
Estações automáticas INMET · Streamlit MVP
"""

import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime

from inmet_api import get_stations, get_accumulated, get_daily_series

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

    period = st.selectbox(
        "Período de análise",
        options=[1, 3, 7, 15, 30],
        index=2,
        format_func=lambda x: f"Últimos {x} dias",
    )

    st.markdown("---")
    st.markdown("**Dados:** Estações automáticas INMET")
    st.markdown("**Atualização:** a cada 30 min")
    st.markdown(f"**Consultado em:** {datetime.now().strftime('%d/%m/%Y %H:%M')}")
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


progress = st.progress(0, text=f"Calculando acumulado dos últimos {period} dias...")
coords = tuple(
    (row["CD_ESTACAO"], row["VL_LATITUDE"], row["VL_LONGITUDE"])
    for _, row in stations_df.iterrows()
)
accumulated = load_all_accumulated(coords, period)
progress.empty()

stations_df["ACUMULADO_MM"] = stations_df["CD_ESTACAO"].map(accumulated).fillna(0)


# ── Paleta de cores por intensidade ──────────────────────────────────────────
def precip_color(mm: float) -> str:
    if mm == 0:
        return "#CCCCCC"
    elif mm < 10:
        return "#A8D8EA"
    elif mm < 30:
        return "#4A90D9"
    elif mm < 60:
        return "#1A5276"
    elif mm < 100:
        return "#E67E22"
    else:
        return "#C0392B"


# ── Métricas resumidas ────────────────────────────────────────────────────────
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

    for _, row in stations_df.iterrows():
        mm = row["ACUMULADO_MM"]
        color = precip_color(mm)
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
                f"Acumulado {period}d: <b>{mm:.1f} mm</b>",
                max_width=220,
            ),
        ).add_to(m)

    # Legenda
    legend_html = """
    <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
         background:white;padding:12px 16px;border-radius:10px;
         box-shadow:0 2px 8px rgba(0,0,0,0.25);font-size:13px;line-height:1.8;">
      <b>Acumulado (mm)</b><br>
      <span style="color:#CCCCCC;font-size:18px">●</span> Sem chuva<br>
      <span style="color:#A8D8EA;font-size:18px">●</span> 0 – 10 mm<br>
      <span style="color:#4A90D9;font-size:18px">●</span> 10 – 30 mm<br>
      <span style="color:#1A5276;font-size:18px">●</span> 30 – 60 mm<br>
      <span style="color:#E67E22;font-size:18px">●</span> 60 – 100 mm<br>
      <span style="color:#C0392B;font-size:18px">●</span> &gt; 100 mm
    </div>
    """
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
        daily_df = get_daily_series(sel["VL_LATITUDE"], sel["VL_LONGITUDE"], period)

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

# ── Tabela comparativa ────────────────────────────────────────────────────────
st.markdown("---")
st.subheader("Comparativo — Todas as Estações")

summary = (
    stations_df[["DC_NOME", "CD_ESTACAO", "VL_LATITUDE", "VL_LONGITUDE", "ACUMULADO_MM"]]
    .copy()
    .rename(
        columns={
            "DC_NOME": "Estação",
            "CD_ESTACAO": "Código",
            "VL_LATITUDE": "Latitude",
            "VL_LONGITUDE": "Longitude",
            "ACUMULADO_MM": "Acumulado (mm)",
        }
    )
    .sort_values("Acumulado (mm)", ascending=False)
    .reset_index(drop=True)
)

col_tbl, col_bar = st.columns([2, 3])

with col_tbl:
    st.dataframe(
        summary.style.background_gradient(subset=["Acumulado (mm)"], cmap="Blues"),
        use_container_width=True,
        hide_index=True,
    )

with col_bar:
    fig_bar = px.bar(
        summary.head(15),
        x="Acumulado (mm)",
        y="Estação",
        orientation="h",
        color="Acumulado (mm)",
        color_continuous_scale="Blues",
        labels={"Acumulado (mm)": "mm"},
        title=f"Top 15 — Acumulado últimos {period} dias",
    )
    fig_bar.update_layout(
        height=420,
        margin=dict(l=0, r=0, t=40, b=0),
        plot_bgcolor="white",
        paper_bgcolor="white",
        coloraxis_showscale=False,
        yaxis=dict(autorange="reversed"),
    )
    st.plotly_chart(fig_bar, use_container_width=True)

# ── Rodapé ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Estações: INMET — Instituto Nacional de Meteorologia · "
    "Precipitação: Open-Meteo (ERA5 + NWP) · Uso informativo"
)
