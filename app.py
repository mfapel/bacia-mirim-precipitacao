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
from forecast_api import get_ecmwf_15d, get_gfs_ensemble, BASIN_LAT, BASIN_LON
from ana_api import (get_nivel_serie, get_nivel_atual, ESTACOES_NIVEL,
                     estimar_sangradouro, get_sangradouro_serie,
                     estimar_sangradouro_b, get_sangradouro_serie_b, calibrar_modelo_b,
                     SANGRADOURO_NOME, SANGRADOURO_LAT, SANGRADOURO_LON,
                     CALIB_NIVEL_REF, CALIB_PROF_REF,
                     CALIB2_NIVEL_REF, CALIB2_PROF_REF,
                     OFFSET, CALIB_INCERTEZA_CM)

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

# Modelo B carregado aqui para uso no mapa e na seção do Sangradouro
with st.spinner("Calibrando modelo Sangradouro..."):
    modelo_b = calibrar_modelo_b(dias_historico=180)

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
        est_map = estimar_sangradouro_b(sg['nivel_cm'], modelo_b)
        prof_m_map = est_map['profundidade_cm'] / 100
        status_map = "Ativo" if est_map['ativo'] else "Seco"
        folium.Marker(
            location=[SANGRADOURO_LAT, SANGRADOURO_LON],
            tooltip=folium.Tooltip(
                f"<b>{SANGRADOURO_NOME}</b><br>"
                f"Estimativa de profundidade: <b>{prof_m_map:.2f} m</b><br>"
                f"Status: {status_map}<br>"
                f"<i>Clique para ver a série histórica</i>",
                sticky=True,
            ),
            popup=folium.Popup(
                f"<b>{SANGRADOURO_NOME}</b><br>"
                f"Estimativa de profundidade: <b>{prof_m_map:.2f} m</b><br>"
                f"Status: {status_map} · Ref.: São Gonçalo {sg['nivel_cm']:.0f} cm<br>"
                f"<i>SANGRADOURO_ID</i>",
                max_width=250,
            ),
            icon=folium.DivIcon(
                html=(
                    '<div style="width:14px;height:14px;background:#117A65;'
                    'transform:rotate(45deg);border:2.5px solid white;'
                    'box-shadow:0 1px 5px rgba(0,0,0,0.5)"></div>'
                ),
                icon_size=(20, 20),
                icon_anchor=(10, 10),
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


# ── Sangradouro Santa Isabel — profundidade estimada (Modelo B) ───────────────
st.markdown("---")
st.subheader("📍 Sangradouro — Santa Isabel (estimativa)")
st.caption(
    "Ponto não monitorado · Profundidade estimada a partir do nível da Eclusa São Gonçalo · "
    "Modelo B: reta calibrada com 2 medições de campo (24/02/2026 e 30/07/2026)"
)

sc1, sc2, sc3 = st.columns(3)

if sg:
    est_b  = estimar_sangradouro_b(sg['nivel_cm'], modelo_b)
    prof_m = est_b['profundidade_cm'] / 100
    delta_m = CALIB_INCERTEZA_CM / 100
    ci_low  = max(0.0, prof_m - delta_m)
    ci_high = prof_m + delta_m

    sc1.metric(
        "Profundidade estimada",
        f"{prof_m:.2f} m",
        delta=f"faixa: {ci_low:.2f} – {ci_high:.2f} m",
        delta_color="off",
        help=f"Faixa ±{CALIB_INCERTEZA_CM:.0f} cm — propagação da incerteza da medição in loco (24/02/2026)",
    )
    sc2.metric(
        "Status",
        "🟢 Ativo" if est_b['ativo'] else "⚪ Seco",
        help="Ativo = sangradouro com lâmina d'água",
    )
    sc3.metric(
        "Nível ref. Eclusa São Gonçalo",
        f"{sg['nivel_cm']:.0f} cm",
        help=f"Nível usado como referência · slope k={modelo_b['k']:.2f}",
    )
else:
    st.warning("Dados da Eclusa São Gonçalo indisponíveis — estimativa não calculada.")

# Parâmetros do modelo (expansível)
metodo_label = {
    "opcao_b_2medicoes_campo": "✅ 2 medições de campo",
    "opcao_b_2pontos": "✅ Opção B — 2 pontos",
    "fallback_k1": "⚠️ Fallback — k=1 (dados insuficientes)",
    "fallback_k1_sem_variacao": "⚠️ Fallback — k=1 (sem variação histórica)",
}.get(modelo_b["metodo"], modelo_b["metodo"])

with st.expander("ℹ️ Parâmetros do modelo"):
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("Método", metodo_label)
    mc2.metric("Slope k", f"{modelo_b['k']:.3f}",
               help="k=1 → variação 1:1 com o lago; k>1 → Sangradouro mais sensível")
    mc3.metric("Nível de secagem (p05)", f"{modelo_b['nivel_seco']:.1f} cm",
               help="Percentil 5 histórico de São Gonçalo — proxy para sangradouro seco")
    mc4.metric("Leituras históricas", f"{modelo_b['n_leituras']:,}")
    st.caption(
        f"Equação: `prof (cm) = {modelo_b['k']:.3f} × nivel_SG + {modelo_b['b']:.1f}` "
        f"· Pt. 1: nivel_SG={CALIB_NIVEL_REF:.0f} cm → {CALIB_PROF_REF:.0f} cm (24/02/2026) "
        f"· Pt. 2: nivel_SG={CALIB2_NIVEL_REF:.0f} cm → {CALIB2_PROF_REF:.0f} cm (30/07/2026)"
    )

# Gráfico: aparece apenas quando o marcador do Sangradouro é clicado
sangradouro_clicked = bool(
    map_result
    and map_result.get("last_object_clicked_popup")
    and "SANGRADOURO_ID" in str(map_result["last_object_clicked_popup"])
)

if not sangradouro_clicked:
    st.info(
        "🟩 Clique no marcador **◆ Sangradouro — Santa Isabel** no mapa "
        "para ver a série histórica de profundidade estimada."
    )
else:
    FEB1_2026 = date(2026, 2, 1)
    sang_days = max(30, (date.today() - FEB1_2026).days)

    with st.spinner("Carregando série do Sangradouro..."):
        df_sang = get_sangradouro_serie_b(days=sang_days)

    if not df_sang.empty:
        CALIB_DT  = pd.Timestamp("2026-02-24 11:20")
        CALIB_M   = CALIB_PROF_REF / 100    # 1.00 m
        CALIB2_DT = pd.Timestamp("2026-07-30 12:30")
        CALIB2_M  = CALIB2_PROF_REF / 100   # 1.80 m

        fig_sang = go.Figure()

        # IC 95% — banda preenchida (renderiza antes das linhas)
        if "Prof_ci_low_m" in df_sang.columns:
            x_band = pd.concat([df_sang["DataHora"],
                                 df_sang["DataHora"].iloc[::-1]])
            y_band = pd.concat([df_sang["Prof_ci_high_m"],
                                 df_sang["Prof_ci_low_m"].iloc[::-1]])
            fig_sang.add_trace(go.Scatter(
                x=x_band, y=y_band,
                fill="toself",
                fillcolor="rgba(17,122,101,0.18)",
                line=dict(color="rgba(17,122,101,0.35)"),
                hoverinfo="skip",
                showlegend=True,
                name=f"Faixa ±{CALIB_INCERTEZA_CM:.0f} cm (incerteza da medição)",
            ))

        # Linha de controle — nível alvo para nova medição de campo
        CONTROLE_M = 2.5
        x_range = [pd.Timestamp("2026-02-01"), pd.Timestamp(date.today())]
        fig_sang.add_trace(go.Scatter(
            x=x_range, y=[CONTROLE_M, CONTROLE_M],
            mode="lines",
            line=dict(color="#E67E22", width=1.5, dash="dot"),
            name=f"Controle — nova medição ({CONTROLE_M:.1f} m)",
            hovertemplate=f"Nível de controle: {CONTROLE_M:.1f} m<extra></extra>",
        ))

        # Modelo B — linha principal
        fig_sang.add_trace(go.Scatter(
            x=df_sang["DataHora"], y=df_sang["Prof_est_m_B"],
            mode="lines",
            line=dict(color="#117A65", width=2),
            name=f"Modelo B (k={modelo_b['k']:.2f})",
            hovertemplate="%{x|%d/%m %H:%M}<br>Prof. estimada: <b>%{y:.2f} m</b><extra></extra>",
        ))
        # Modelo A (k=1) — tracejado para comparação
        fig_sang.add_trace(go.Scatter(
            x=df_sang["DataHora"], y=df_sang["Prof_est_m_A"],
            mode="lines",
            line=dict(color="#AAB7B8", width=1.2, dash="dot"),
            name="Modelo A (k=1)",
            hovertemplate="%{x|%d/%m %H:%M}<br>Modelo A: %{y:.2f} m<extra></extra>",
        ))
        # Ponto de calibração 1 — 24/02/2026
        fig_sang.add_trace(go.Scatter(
            x=[CALIB_DT], y=[CALIB_M],
            mode="markers+text",
            marker=dict(color="#E74C3C", size=14, symbol="diamond",
                        line=dict(color="white", width=2)),
            text=["1,00 m"],
            textposition="top center",
            textfont=dict(color="#E74C3C", size=11),
            name="Medição in loco (24/02/2026)",
            hovertemplate="<b>Medição in loco</b><br>24/02/2026 11:20<br>Profundidade: 1,00 m<extra></extra>",
        ))
        # Ponto de calibração 2 — 30/07/2026
        fig_sang.add_trace(go.Scatter(
            x=[CALIB2_DT], y=[CALIB2_M],
            mode="markers+text",
            marker=dict(color="#E74C3C", size=14, symbol="diamond",
                        line=dict(color="white", width=2)),
            text=["1,80 m"],
            textposition="top center",
            textfont=dict(color="#E74C3C", size=11),
            name="Medição in loco (30/07/2026)",
            hovertemplate="<b>Medição in loco</b><br>30/07/2026 12:30<br>Profundidade: 1,80 m<extra></extra>",
        ))

        fig_sang.update_layout(
            yaxis_title="Profundidade estimada (m)",
            xaxis_title=None,
            height=340,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(gridcolor="#EEEEEE", rangemode="tozero"),
            xaxis=dict(
                gridcolor="#EEEEEE",
                range=[pd.Timestamp("2026-02-01"), pd.Timestamp(date.today())],
            ),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)"),
        )
        st.plotly_chart(fig_sang, use_container_width=True)
        st.caption(
            f"Série desde 01/02/2026 · "
            f"Banda verde = faixa ±{CALIB_INCERTEZA_CM:.0f} cm (incerteza das medições de campo) · "
            f"◆ Pontos vermelhos = medições de campo (1,00 m em 24/02 e 1,80 m em 30/07) · "
            f"Linha tracejada = Modelo A (k=1, referência)."
        )
    else:
        st.warning("Sem dados disponíveis para o Sangradouro.")

# ── Previsão de Precipitação ──────────────────────────────────────────────────
st.markdown("---")
st.subheader("Previsão de Precipitação — Centro da Bacia Mirim")
st.caption(
    f"Ponto de referência: {BASIN_LAT:.2f}°S, {abs(BASIN_LON):.2f}°W · "
    "ECMWF IFS 0.4° (15 dias) e GFS Ensemble 0.5° (35 dias) · "
    "Skill positivo confirmado até ~14 dias para o sul do Brasil (Badagian et al., 2024)"
)

tab_15d, tab_35d = st.tabs(["📅 15 dias — ECMWF IFS (determinístico)", "📆 35 dias — GFS Ensemble (probabilístico)"])

with tab_15d:
    with st.spinner("Carregando previsão ECMWF IFS..."):
        df_15d = get_ecmwf_15d()

    if df_15d.empty:
        st.warning("Previsão ECMWF IFS indisponível no momento.")
    else:
        total_15d = df_15d["Precip_mm"].sum()
        fc1, fc2, fc3 = st.columns(3)
        fc1.metric("Acumulado previsto (15 dias)", f"{total_15d:.1f} mm")
        fc2.metric("Máximo diário previsto", f"{df_15d['Precip_mm'].max():.1f} mm",
                   df_15d.loc[df_15d['Precip_mm'].idxmax(), 'Data'].strftime('%d/%m'))
        fc3.metric("Dias com chuva previstos", int((df_15d['Precip_mm'] > 0.5).sum()))

        fig_15d = go.Figure()
        fig_15d.add_trace(go.Bar(
            x=df_15d["Data"], y=df_15d["Precip_mm"],
            marker_color="#4A90D9",
            marker_line_color="#1A5276",
            marker_line_width=0.5,
            name="Precipitação diária (mm)",
            hovertemplate="%{x|%d/%m/%Y}<br>%{y:.1f} mm<extra></extra>",
        ))
        fig_15d.add_trace(go.Scatter(
            x=df_15d["Data"], y=df_15d["Acumulado_mm"],
            mode="lines", yaxis="y2",
            line=dict(color="#E67E22", width=2),
            name="Acumulado (mm)",
            hovertemplate="%{x|%d/%m/%Y}<br>Acumulado: %{y:.1f} mm<extra></extra>",
        ))
        fig_15d.update_layout(
            xaxis_title=None, height=320,
            margin=dict(l=0, r=0, t=10, b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(title="mm/dia", gridcolor="#EEEEEE", zeroline=True),
            yaxis2=dict(title="Acumulado (mm)", overlaying="y", side="right",
                        color="#E67E22", showgrid=False),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)"),
            xaxis=dict(gridcolor="#EEEEEE"),
        )
        st.plotly_chart(fig_15d, use_container_width=True)
        st.caption(
            "⚠️ Skill da previsão diminui progressivamente após o dia 7. "
            "Valores além de 10 dias devem ser interpretados com cautela."
        )

with tab_35d:
    with st.spinner("Carregando ensemble GFS..."):
        df_35d = get_gfs_ensemble()

    if df_35d.empty:
        st.warning("Previsão ensemble GFS indisponível no momento.")
    else:
        ec1, ec2, ec3 = st.columns(3)
        ec1.metric("Mediana acumulada (35 dias)", f"{df_35d['P50_acum'].iloc[-1]:.0f} mm")
        ec2.metric("Cenário chuvoso (P90)",        f"{df_35d['P90_acum'].iloc[-1]:.0f} mm")
        ec3.metric("Cenário seco (P10)",            f"{df_35d['P10_acum'].iloc[-1]:.0f} mm")

        fig_35d = go.Figure()

        x_band = pd.concat([df_35d["Data"], df_35d["Data"].iloc[::-1]])
        y_band = pd.concat([df_35d["P90_acum"], df_35d["P10_acum"].iloc[::-1]])
        fig_35d.add_trace(go.Scatter(
            x=x_band, y=y_band,
            fill="toself",
            fillcolor="rgba(74,144,217,0.18)",
            line=dict(color="rgba(74,144,217,0.35)"),
            hoverinfo="skip", showlegend=True,
            name="Faixa P10–P90 (31 membros GFS)",
        ))
        fig_35d.add_trace(go.Scatter(
            x=df_35d["Data"], y=df_35d["P50_acum"],
            mode="lines",
            line=dict(color="#1A5276", width=2),
            name="Mediana acumulada (P50)",
            hovertemplate="%{x|%d/%m/%Y}<br>Mediana acumulada: %{y:.1f} mm<extra></extra>",
        ))
        fig_35d.add_trace(go.Bar(
            x=df_35d["Data"], y=df_35d["P50_mm"],
            marker_color="rgba(74,144,217,0.5)",
            name="Precipitação diária mediana",
            yaxis="y2",
            hovertemplate="%{x|%d/%m/%Y}<br>Mediana: %{y:.1f} mm<extra></extra>",
        ))
        semana2 = pd.Timestamp(date.today()) + pd.Timedelta(days=14)
        fig_35d.add_vline(
            x=semana2, line_dash="dot",
            line_color="#E74C3C", line_width=1.5,
            annotation_text="Semana 2 →<br>incerteza aumenta",
            annotation_position="top right",
            annotation_font_size=10,
            annotation_font_color="#E74C3C",
        )
        fig_35d.update_layout(
            xaxis_title=None, height=360,
            margin=dict(l=0, r=0, t=20, b=0),
            plot_bgcolor="white", paper_bgcolor="white",
            yaxis=dict(title="Acumulado (mm)", gridcolor="#EEEEEE"),
            yaxis2=dict(title="mm/dia", overlaying="y", side="right",
                        showgrid=False, color="#4A90D9"),
            legend=dict(x=0.01, y=0.99, bgcolor="rgba(255,255,255,0.85)"),
            xaxis=dict(gridcolor="#EEEEEE"),
            barmode="overlay",
        )
        st.plotly_chart(fig_35d, use_container_width=True)
        st.caption(
            "Banda azul = faixa de incerteza entre os 31 membros do GFS Ensemble 0.5° · "
            "Linha azul escura = mediana acumulada · "
            "Barras = precipitação diária mediana · "
            "Linha vermelha tracejada = limite do skill útil (~14 dias) · "
            "Fonte: Badagian et al. (2024), *Int. J. Climatology*."
        )

# ── Rodapé ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.caption(
    "Estações: INMET — Instituto Nacional de Meteorologia · "
    "Precipitação: Open-Meteo (ERA5 + NWP) · Uso informativo"
)
