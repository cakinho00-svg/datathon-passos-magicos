"""
Datathon Passos Mágicos — Dashboard Analítico + Predição de Risco
Pós-graduação Data Analytics — FIAP Postech — Fase 5
"""
import os, joblib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ── Configuração ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Passos Mágicos · Analytics",
    page_icon="⭐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Modelo ────────────────────────────────────────────────────────────────────
@st.cache_resource
def carregar_modelo():
    modelo   = joblib.load(os.path.join(BASE_DIR, "model", "modelo_risco.pkl"))
    scaler   = joblib.load(os.path.join(BASE_DIR, "model", "scaler.pkl"))
    features = joblib.load(os.path.join(BASE_DIR, "model", "features.pkl"))
    return modelo, scaler, features

@st.cache_data
def carregar_dados():
    path = os.path.join(BASE_DIR, "..", "data", "painel_pede_tratado.csv")
    return pd.read_csv(path)

modelo, scaler, FEATURES = carregar_modelo()

# ── Design System ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

/* Reset e base */
*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background-color: #0A0F1E !important;
    color: #F1F5F9 !important;
    font-family: 'Inter', sans-serif !important;
}

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0D1117 0%, #111827 100%) !important;
    border-right: 1px solid #1E293B !important;
}

[data-testid="stSidebar"] * { color: #CBD5E1 !important; }

/* Remove header padrão */
[data-testid="stHeader"] { background: transparent !important; }
.block-container { padding: 1.5rem 2rem 3rem 2rem !important; max-width: 1400px !important; }

/* Hero banner */
.hero {
    background: linear-gradient(135deg, #0D1B2A 0%, #111827 40%, #0D1B2A 100%);
    border: 1px solid #1E293B;
    border-radius: 16px;
    padding: 2.5rem 3rem;
    margin-bottom: 2rem;
    position: relative;
    overflow: hidden;
}
.hero::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(0,200,151,0.08) 0%, transparent 70%);
    pointer-events: none;
}
.hero::after {
    content: '';
    position: absolute;
    bottom: -80px; left: 40%;
    width: 400px; height: 400px;
    background: radial-gradient(circle, rgba(124,58,237,0.06) 0%, transparent 70%);
    pointer-events: none;
}
.hero-tag {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #00C897;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.hero-title {
    font-size: 2.2rem;
    font-weight: 800;
    color: #F1F5F9;
    line-height: 1.15;
    margin: 0 0 0.5rem 0;
}
.hero-title span { color: #00C897; }
.hero-sub {
    font-size: 0.95rem;
    color: #94A3B8;
    font-weight: 400;
    margin: 0;
}

/* KPI cards */
.kpi-row { display: flex; gap: 1rem; margin-bottom: 2rem; flex-wrap: wrap; }
.kpi-card {
    flex: 1; min-width: 140px;
    background: #111827;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.kpi-card:hover { border-color: #00C897; }
.kpi-card::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 3px;
    border-radius: 12px 12px 0 0;
}
.kpi-card.green::before  { background: #00C897; }
.kpi-card.purple::before { background: #7C3AED; }
.kpi-card.amber::before  { background: #F59E0B; }
.kpi-card.red::before    { background: #EF4444; }
.kpi-card.blue::before   { background: #3B82F6; }
.kpi-label {
    font-size: 0.72rem;
    font-weight: 500;
    color: #64748B;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 0.4rem;
}
.kpi-value {
    font-size: 1.9rem;
    font-weight: 800;
    color: #F1F5F9;
    line-height: 1;
    font-family: 'JetBrains Mono', monospace;
}
.kpi-value.green  { color: #00C897; }
.kpi-value.purple { color: #A78BFA; }
.kpi-value.amber  { color: #FBB03B; }
.kpi-value.red    { color: #F87171; }
.kpi-delta {
    font-size: 0.75rem;
    color: #64748B;
    margin-top: 0.3rem;
}

/* Section headers */
.section-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.68rem;
    color: #00C897;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
}
.section-title {
    font-size: 1.35rem;
    font-weight: 700;
    color: #F1F5F9;
    margin: 0 0 0.4rem 0;
}
.section-desc {
    font-size: 0.85rem;
    color: #64748B;
    margin-bottom: 1.5rem;
}

/* Chart cards */
.chart-card {
    background: #111827;
    border: 1px solid #1E293B;
    border-radius: 12px;
    padding: 1.4rem;
    margin-bottom: 1.2rem;
}
.chart-title {
    font-size: 0.9rem;
    font-weight: 600;
    color: #CBD5E1;
    margin-bottom: 0.3rem;
}
.chart-desc {
    font-size: 0.78rem;
    color: #475569;
    margin-bottom: 1rem;
    line-height: 1.5;
}

/* Tabs */
[data-testid="stTabs"] button {
    font-family: 'Inter', sans-serif !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
    color: #64748B !important;
    border-radius: 8px 8px 0 0 !important;
    padding: 0.6rem 1.2rem !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #00C897 !important;
    border-bottom: 2px solid #00C897 !important;
    background: transparent !important;
}

/* Risco cards */
.risk-card {
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-top: 1rem;
}
.risk-alto    { background: rgba(239,68,68,0.1);  border: 1px solid rgba(239,68,68,0.3); }
.risk-medio   { background: rgba(245,158,11,0.1); border: 1px solid rgba(245,158,11,0.3); }
.risk-baixo   { background: rgba(0,200,151,0.1);  border: 1px solid rgba(0,200,151,0.3); }
.risk-title   { font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem; }
.risk-alto  .risk-title  { color: #F87171; }
.risk-medio .risk-title  { color: #FBB03B; }
.risk-baixo .risk-title  { color: #00C897; }
.risk-body    { font-size: 0.85rem; color: #94A3B8; line-height: 1.6; }
.risk-rec     { font-size: 0.82rem; color: #CBD5E1; margin-top: 0.6rem; }

/* Sliders e inputs */
[data-testid="stSlider"] > div > div > div > div { background: #00C897 !important; }
[data-testid="stSlider"] label { color: #94A3B8 !important; font-size: 0.82rem !important; }
[data-testid="stNumberInput"] label { color: #94A3B8 !important; font-size: 0.82rem !important; }
[data-testid="stSelectbox"] label { color: #94A3B8 !important; font-size: 0.82rem !important; }

/* Botão principal */
[data-testid="stButton"] button[kind="primary"] {
    background: linear-gradient(135deg, #00C897, #00A87A) !important;
    color: #0A0F1E !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 0.7rem 1.5rem !important;
    letter-spacing: 0.03em !important;
    transition: opacity 0.2s !important;
}
[data-testid="stButton"] button[kind="primary"]:hover { opacity: 0.88 !important; }

/* Divisor */
hr { border-color: #1E293B !important; }

/* Sidebar navigation */
.nav-item {
    display: flex; align-items: center; gap: 0.7rem;
    padding: 0.6rem 0.8rem;
    border-radius: 8px;
    cursor: pointer;
    font-size: 0.88rem;
    font-weight: 500;
    color: #64748B;
    margin-bottom: 0.2rem;
    transition: all 0.15s;
}
.nav-item.active, .nav-item:hover {
    background: rgba(0,200,151,0.1);
    color: #00C897 !important;
}
.sidebar-logo {
    font-size: 1.3rem; font-weight: 800;
    color: #F1F5F9;
    margin-bottom: 0.2rem;
    letter-spacing: -0.02em;
}
.sidebar-logo span { color: #00C897; }
.sidebar-version {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem; color: #334155;
    margin-bottom: 2rem;
}

/* Metric override */
[data-testid="stMetric"] {
    background: #111827 !important;
    border: 1px solid #1E293B !important;
    border-radius: 12px !important;
    padding: 1rem 1.2rem !important;
}
[data-testid="stMetricLabel"] { color: #64748B !important; font-size: 0.75rem !important; }
[data-testid="stMetricValue"] { color: #F1F5F9 !important; font-size: 1.6rem !important; font-weight: 700 !important; }

/* Dataframe */
[data-testid="stDataFrame"] { border-radius: 10px !important; overflow: hidden !important; }

/* Upload area */
[data-testid="stFileUploader"] {
    background: #111827 !important;
    border: 1px dashed #1E293B !important;
    border-radius: 12px !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #0A0F1E; }
::-webkit-scrollbar-thumb { background: #1E293B; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)

# ── Helper: imagem com card ───────────────────────────────────────────────────
def chart_card(img_path, title, desc):
    st.markdown(f"""
    <div class="chart-card">
        <div class="chart-title">{title}</div>
        <div class="chart-desc">{desc}</div>
    </div>""", unsafe_allow_html=True)
    try:
        img = Image.open(os.path.join(BASE_DIR, "assets", img_path))
        st.image(img, use_container_width=True)
    except Exception:
        st.warning(f"Imagem não encontrada: {img_path}")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">Passos <span>Mágicos</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-version">// DATATHON · FIAP POSTECH · FASE 5</div>', unsafe_allow_html=True)

    pagina = st.radio(
        "Navegação",
        options=["🏠  Visão Geral", "📊  Análise por Indicador", "🤖  Modelo Preditivo", "🔍  Avaliar Aluno", "📋  Avaliação em Lote"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown('<p style="font-size:0.72rem;color:#334155;text-transform:uppercase;letter-spacing:0.1em;">Base de dados</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.82rem;color:#475569;">PEDE 2022 · 2023 · 2024</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.72rem;color:#334155;margin-top:0.8rem;text-transform:uppercase;letter-spacing:0.1em;">Modelo</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:0.82rem;color:#475569;">XGBoost · AUC 0.795</p>', unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 1 — VISÃO GERAL
# ══════════════════════════════════════════════════════════════════════════════
if pagina == "🏠  Visão Geral":

    st.markdown("""
    <div class="hero">
        <div class="hero-tag">// Datathon · Análise Educacional</div>
        <h1 class="hero-title">Transformando dados em <span>impacto real</span></h1>
        <p class="hero-sub">Dashboard analítico da Pesquisa Extensiva do Desenvolvimento Educacional (PEDE) —
        identificando padrões, riscos e oportunidades para crianças e jovens da Passos Mágicos.</p>
    </div>
    """, unsafe_allow_html=True)

    # KPIs
    st.markdown("""
    <div class="kpi-row">
        <div class="kpi-card green">
            <div class="kpi-label">Total de alunos</div>
            <div class="kpi-value green">3.030</div>
            <div class="kpi-delta">registros aluno-ano · 2022–2024</div>
        </div>
        <div class="kpi-card purple">
            <div class="kpi-label">Crescimento da base</div>
            <div class="kpi-value purple">+34%</div>
            <div class="kpi-delta">860 (2022) → 1.156 (2024)</div>
        </div>
        <div class="kpi-card green">
            <div class="kpi-label">INDE médio geral</div>
            <div class="kpi-value">7,27</div>
            <div class="kpi-delta">escala de 0 a 10</div>
        </div>
        <div class="kpi-card amber">
            <div class="kpi-label">Em risco (modelo)</div>
            <div class="kpi-value amber">19,8%</div>
            <div class="kpi-delta">dos alunos com histórico longitudinal</div>
        </div>
        <div class="kpi-card purple">
            <div class="kpi-label">Topázio em 2024</div>
            <div class="kpi-value purple">28%</div>
            <div class="kpi-delta">vs 15% em 2022 · evolução positiva</div>
        </div>
        <div class="kpi-card red">
            <div class="kpi-label">IAN severo</div>
            <div class="kpi-value red">~8%</div>
            <div class="kpi-delta">alunos com defasagem crítica</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    # Resumo dos achados
    st.markdown('<div class="section-label">// Principais achados</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">O que os dados revelam</h2>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    insights = [
        ("📈", "#00C897", "Programa em crescimento", "A base de alunos cresceu 34% entre 2022 e 2024, com aumento consistente de alunos na classificação Topázio (de 15% para 28%), evidenciando o impacto positivo do programa."),
        ("🔗", "#7C3AED", "Engajamento como driver", "IEG e IDA são os dois maiores preditores do INDE (r=0,75 e r=0,79). Alunos mais engajados performam melhor academicamente — o programa que ativa o engajamento ganha na frente."),
        ("⚠️", "#F59E0B", "Autopercepção superestimada", "IAA tem correlação fraca com IDA (r=0,12). Os alunos tendem a se autoavaliar bem independente do desempenho real — sinal de baixa calibração da autopercepção."),
    ]
    for col, (icon, cor, titulo, texto) in zip([col1, col2, col3], insights):
        with col:
            st.markdown(f"""
            <div class="chart-card" style="border-left: 3px solid {cor}; padding: 1.2rem 1.4rem;">
                <div style="font-size:1.6rem; margin-bottom:0.6rem;">{icon}</div>
                <div style="font-size:0.9rem; font-weight:600; color:#CBD5E1; margin-bottom:0.5rem;">{titulo}</div>
                <div style="font-size:0.8rem; color:#475569; line-height:1.6;">{texto}</div>
            </div>
            """, unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="section-label">// Efetividade do programa · Visão geral</div>', unsafe_allow_html=True)
    st.markdown('<h2 class="section-title">Distribuição de Pedras por Ano</h2>', unsafe_allow_html=True)
    chart_card("p10_efetividade.png",
               "Evolução da classificação PEDRA (2022–2024)",
               "Crescimento consistente de Topázio e Ametista confirma o impacto positivo do programa ao longo dos anos.")


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 2 — ANÁLISE POR INDICADOR
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "📊  Análise por Indicador":

    st.markdown('<div class="section-label">// Pesquisa PEDE · 2022–2024</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="section-title">Análise por Indicador</h1>', unsafe_allow_html=True)
    st.markdown('<p class="section-desc">Resposta às 11 perguntas de negócio do desafio, organizadas por indicador educacional.</p>', unsafe_allow_html=True)

    tabs = st.tabs(["📌 IAN", "📚 IDA", "💡 IEG", "🧠 IAA", "❤️ IPS", "🔬 IPP", "⭐ IPV", "🏆 INDE", "📈 Efetividade", "💡 Insights"])

    with tabs[0]:
        st.markdown('<div class="section-label">// Pergunta 1</div>', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">Adequação ao Nível (IAN)</h3>', unsafe_allow_html=True)
        st.markdown('<p class="section-desc">O IAN assume três valores: 2,5 (severo), 5,0 (moderado) ou 10,0 (adequado). Mede se o aluno está no nível escolar compatível com sua faixa etária.</p>', unsafe_allow_html=True)
        chart_card("p1_ian.png", "Distribuição do IAN e evolução anual",
                   "A proporção de alunos com IAN adequado (10,0) cresce de ~57% em 2022 para ~63% em 2024, indicando redução da defasagem ao longo do programa.")
        col1, col2, col3 = st.columns(3)
        col1.metric("IAN Severo (2,5)", "~8%", "alunos críticos")
        col2.metric("IAN Moderado (5,0)", "~30%", "atenção necessária")
        col3.metric("IAN Adequado (10,0)", "~62%", "dentro do esperado")

    with tabs[1]:
        st.markdown('<div class="section-label">// Pergunta 2</div>', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">Desempenho Acadêmico (IDA)</h3>', unsafe_allow_html=True)
        st.markdown('<p class="section-desc">O IDA representa a média das notas do indicador de aprendizagem. Analisamos sua evolução por ano e por fase.</p>', unsafe_allow_html=True)
        chart_card("p2_ida.png", "IDA médio por ano e por fase",
                   "O IDA médio se mantém acima de 7,0 nos três anos. Fases iniciais (ALFA e Fase 1) apresentam IDA menor, que cresce progressivamente nas fases avançadas.")

    with tabs[2]:
        st.markdown('<div class="section-label">// Pergunta 3</div>', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">Engajamento (IEG)</h3>', unsafe_allow_html=True)
        st.markdown('<p class="section-desc">O IEG mede o grau de participação e envolvimento do aluno nas atividades. É um dos principais preditores do INDE.</p>', unsafe_allow_html=True)
        chart_card("p3_ieg.png", "IEG vs IDA e IEG vs IPV",
                   "Correlação moderada-forte: IEG×IDA (r=0,54) e IEG×IPV (r=0,56). Alunos mais engajados têm maior desempenho acadêmico e maior chance de atingir o ponto de virada.")

    with tabs[3]:
        st.markdown('<div class="section-label">// Pergunta 4</div>', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">Autoavaliação (IAA)</h3>', unsafe_allow_html=True)
        st.markdown('<p class="section-desc">O IAA reflete a percepção do aluno sobre si mesmo. Analisamos se essa percepção é coerente com o desempenho real.</p>', unsafe_allow_html=True)
        chart_card("p4_iaa.png", "IAA vs IDA e IAA vs IEG",
                   "IAA tem correlação fraca com IDA (r=0,12) e IEG (r=0,13). Os alunos tendem a se autoavaliar positivamente independente do desempenho real — indicando baixa calibração da autopercepção.")

    with tabs[4]:
        st.markdown('<div class="section-label">// Pergunta 5</div>', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">Aspectos Psicossociais (IPS)</h3>', unsafe_allow_html=True)
        st.markdown('<p class="section-desc">O IPS mede o suporte socioemocional do aluno. Investigamos se há padrões de IPS que antecedem quedas de desempenho.</p>', unsafe_allow_html=True)
        chart_card("p5_ips.png", "IPS como preditor de queda no IDA",
                   "Alunos que caíram em IDA no ano seguinte apresentavam IPS ligeiramente inferior no ano anterior. O sinal é fraco mas consistente — IPS baixo pode ser um sinal precoce de risco.")

    with tabs[5]:
        st.markdown('<div class="section-label">// Pergunta 6</div>', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">Aspectos Psicopedagógicos (IPP)</h3>', unsafe_allow_html=True)
        st.markdown('<p class="section-desc">O IPP representa a avaliação psicopedagógica do aluno. Disponível apenas a partir de 2023.</p>', unsafe_allow_html=True)
        chart_card("p6_ipp.png", "IPP por categoria de defasagem (IAN)",
                   "IPP cai consistentemente entre alunos com IAN severo (r=0,12 com IAN). As avaliações psicopedagógicas confirmam a defasagem identificada pelo IAN.")

    with tabs[6]:
        st.markdown('<div class="section-label">// Pergunta 7</div>', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">Ponto de Virada (IPV)</h3>', unsafe_allow_html=True)
        st.markdown('<p class="section-desc">O IPV representa o momento de transformação do aluno. Analisamos quais indicadores mais influenciam esse ponto.</p>', unsafe_allow_html=True)
        chart_card("p7_ipv.png", "Correlações com IPV e comparativo por grupo",
                   "IDA e IEG têm a maior correlação com IPV. Em 2022, alunos que atingiram o Ponto de Virada apresentavam IDA e IEG médios significativamente superiores aos que não atingiram.")

    with tabs[7]:
        st.markdown('<div class="section-label">// Pergunta 8</div>', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">Multidimensionalidade — INDE</h3>', unsafe_allow_html=True)
        st.markdown('<p class="section-desc">Quais combinações de indicadores elevam mais a nota geral (INDE)?</p>', unsafe_allow_html=True)
        chart_card("p8_inde.png", "Correlações com INDE e média por Pedra",
                   "IDA (r=0,79) e IEG (r=0,75) são os maiores drivers do INDE. IPS é o menos correlacionado (r=0,20). A combinação IDA+IEG+IPV explica a maior parte da variação do INDE.")

    with tabs[8]:
        st.markdown('<div class="section-label">// Pergunta 10</div>', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">Efetividade do Programa</h3>', unsafe_allow_html=True)
        st.markdown('<p class="section-desc">Os indicadores mostram melhora consistente ao longo dos anos nas diferentes fases?</p>', unsafe_allow_html=True)
        chart_card("p10_efetividade.png", "Distribuição de Pedra por ano",
                   "Topázio cresce de 15% (2022) para 28% (2024). Quartzo diminui. O programa demonstra impacto real e consistente na evolução dos alunos ao longo dos ciclos.")

    with tabs[9]:
        st.markdown('<div class="section-label">// Pergunta 11</div>', unsafe_allow_html=True)
        st.markdown('<h3 class="section-title">Insights Adicionais</h3>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            chart_card("p11_insights.png", "Crescimento da base e INDE por gênero",
                       "Base cresceu 34% entre 2022 e 2024. Gênero não gera diferença significativa no INDE médio — o programa atende igualmente meninos e meninas.")
        with col2:
            chart_card("p11_defasagem.png", "Evolução da defasagem negativa",
                       "A proporção de alunos abaixo do nível ideal cai ao longo dos anos — sinal de que o programa reduz a defasagem escolar de forma consistente.")


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 3 — MODELO PREDITIVO (resultados)
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "🤖  Modelo Preditivo":

    st.markdown('<div class="section-label">// Pergunta 9 · Machine Learning</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="section-title">Modelo de Risco de Defasagem</h1>', unsafe_allow_html=True)
    st.markdown('<p class="section-desc">XGBoost treinado para identificar alunos com risco de queda no INDE ou aumento da defasagem no próximo ciclo.</p>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Algoritmo", "XGBoost")
    col2.metric("ROC-AUC", "0.795", "conjunto de teste")
    col3.metric("Acurácia", "80%", "geral")
    col4.metric("Features", "15", "variáveis preditoras")

    st.divider()

    c1, c2 = st.columns(2)
    with c1:
        chart_card("p9_modelo_avaliacao.png", "Avaliação do modelo",
                   "Comparativo de AUC entre Regressão Logística, Random Forest e XGBoost. Curvas ROC no conjunto de teste e Matriz de Confusão do modelo final.")
    with c2:
        chart_card("p9_feature_importance.png", "Importância das features",
                   "As 3 features mais relevantes (em vermelho) são os maiores preditores do risco. INDE, IDA e IEG lideram a importância no modelo XGBoost.")

    st.divider()
    st.markdown('<div class="section-label">// Definição do target</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="chart-card">
        <div class="chart-title">Como o risco é definido</div>
        <div class="chart-desc">
            Um aluno é classificado como <strong style="color:#F87171">em risco</strong> quando, no ano anterior, apresenta pelo menos uma das condições:<br><br>
            <span style="color:#00C897">▸</span> Queda de INDE superior a <strong style="color:#F1F5F9">0,3 ponto</strong> no ano seguinte<br>
            <span style="color:#00C897">▸</span> Piora no nível de defasagem (Defasagem mais negativa)<br><br>
            <span style="color:#475569">Isso permite usar os dados do ano atual para prever o que vai acontecer no próximo ciclo, 
            dando tempo para a equipe pedagógica intervir antes que a queda aconteça.</span>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 4 — AVALIAR ALUNO INDIVIDUAL
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "🔍  Avaliar Aluno":

    st.markdown('<div class="section-label">// Simulador de risco individual</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="section-title">Avaliar Aluno</h1>', unsafe_allow_html=True)
    st.markdown('<p class="section-desc">Insira os indicadores do aluno para calcular a probabilidade de risco de defasagem no próximo ciclo.</p>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown('<p style="font-size:0.78rem;color:#00C897;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.8rem;">📚 Indicadores Acadêmicos</p>', unsafe_allow_html=True)
        INDE = st.slider("INDE", 3.0, 9.5, 7.4, 0.1, help="Índice de Desenvolvimento Educacional")
        IDA  = st.slider("IDA — Desempenho Acadêmico", 0.0, 10.0, 6.7, 0.1)
        IAN  = st.selectbox("IAN — Adequação ao Nível",
                            [2.5, 5.0, 10.0],
                            format_func=lambda x: {2.5:"2.5 · Severo", 5.0:"5.0 · Moderado", 10.0:"10.0 · Adequado"}[x],
                            index=1)
        IPP  = st.slider("IPP — Psicopedagógico", 2.5, 10.0, 7.5, 0.1)
        Defasagem = st.select_slider("Defasagem (nível atual vs ideal)",
                                     options=list(range(-5, 4)), value=-1)

    with col2:
        st.markdown('<p style="font-size:0.78rem;color:#7C3AED;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.8rem;">💡 Indicadores Comportamentais</p>', unsafe_allow_html=True)
        IAA = st.slider("IAA — Autoavaliação", 0.0, 10.0, 8.8, 0.1)
        IEG = st.slider("IEG — Engajamento", 0.0, 10.0, 8.6, 0.1)
        IPS = st.slider("IPS — Psicossocial", 2.5, 10.0, 7.5, 0.1)
        IPV = st.slider("IPV — Ponto de Virada", 2.5, 10.0, 7.6, 0.1)

    with col3:
        st.markdown('<p style="font-size:0.78rem;color:#F59E0B;font-weight:600;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.8rem;">🎓 Perfil do Aluno</p>', unsafe_allow_html=True)
        Fase_Num     = st.selectbox("Fase atual", list(range(0, 9)),
                                    format_func=lambda x: "ALFA" if x==0 else f"Fase {x}", index=2)
        Ano_Ingresso = st.number_input("Ano de ingresso na PM", 2016, 2024, 2021)
        Ano_Ref      = st.selectbox("Ano de referência", [2022, 2023, 2024], index=1)
        Idade        = st.number_input("Idade", 7, 27, 12)

    # Feature engineering
    Anos_no_Programa     = Ano_Ref - Ano_Ingresso
    Media_Comportamental = np.mean([IAA, IEG, IPS])
    Media_Academica      = np.mean([IDA, IPV, IPP])
    Gap_Auto_Real        = IAA - IDA

    entrada = pd.DataFrame([{
        "INDE": INDE, "IAA": IAA, "IEG": IEG, "IPS": IPS,
        "IDA": IDA,  "IPV": IPV, "IAN": IAN, "IPP": IPP,
        "Defasagem": Defasagem, "Fase_Num": Fase_Num,
        "Anos_no_Programa": Anos_no_Programa, "Idade": Idade,
        "Media_Comportamental": Media_Comportamental,
        "Media_Academica": Media_Academica,
        "Gap_Auto_Real": Gap_Auto_Real,
    }])[FEATURES]

    st.divider()

    if st.button("▶  Calcular Probabilidade de Risco", type="primary", use_container_width=True):
        prob = modelo.predict_proba(entrada)[0][1]
        pct  = prob * 100

        c_gauge, c_result = st.columns([1, 1])

        with c_gauge:
            # Gauge
            fig, ax = plt.subplots(figsize=(5, 2.8), facecolor='#111827')
            ax.set_facecolor('#111827')
            ax.axis('off')

            # Arcos coloridos
            for cor, (t0, t1) in zip(['#00C897','#F59E0B','#EF4444'],
                                       [(0.67,1.0),(0.33,0.67),(0.0,0.33)]):
                t = np.linspace(np.pi*t0, np.pi*t1, 100)
                ax.plot(np.cos(t), np.sin(t), color=cor, linewidth=20, solid_capstyle='butt', alpha=0.85)

            # Ponteiro
            angulo = np.pi * (1 - prob)
            ax.annotate('', xy=(0.58*np.cos(angulo), 0.58*np.sin(angulo)),
                        xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='#F1F5F9', lw=3))
            ax.add_patch(plt.Circle((0,0), 0.06, color='#F1F5F9', zorder=5))

            cor_pct = '#EF4444' if prob >= 0.5 else '#F59E0B' if prob >= 0.3 else '#00C897'
            ax.text(0, -0.2, f"{pct:.1f}%", ha='center', va='center',
                    fontsize=26, fontweight='bold', color=cor_pct,
                    fontfamily='monospace')
            ax.text(0, -0.42, "probabilidade de risco", ha='center',
                    fontsize=9, color='#64748B')

            patches = [mpatches.Patch(color='#00C897', label='Baixo < 30%'),
                       mpatches.Patch(color='#F59E0B', label='Moderado 30–50%'),
                       mpatches.Patch(color='#EF4444', label='Alto > 50%')]
            ax.legend(handles=patches, loc='lower center', ncol=3,
                      fontsize=7, frameon=False,
                      labelcolor='#94A3B8')

            ax.set_xlim(-1.15, 1.15); ax.set_ylim(-0.55, 1.15)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with c_result:
            if prob >= 0.5:
                st.markdown(f"""
                <div class="risk-card risk-alto">
                    <div class="risk-title">🔴 Alto Risco — {pct:.1f}%</div>
                    <div class="risk-body">Alta probabilidade de queda no desempenho ou aumento da defasagem no próximo ciclo.</div>
                    <div class="risk-rec">
                        <strong style="color:#F87171">Ação recomendada:</strong><br>
                        Acionar acompanhamento psicopedagógico imediato. Revisar nível de defasagem com a equipe pedagógica. Monitorar engajamento semanalmente.
                    </div>
                </div>""", unsafe_allow_html=True)
            elif prob >= 0.3:
                st.markdown(f"""
                <div class="risk-card risk-medio">
                    <div class="risk-title">🟡 Risco Moderado — {pct:.1f}%</div>
                    <div class="risk-body">Sinais de alerta presentes. O aluno pode se estabilizar, mas monitoramento é recomendado.</div>
                    <div class="risk-rec">
                        <strong style="color:#FBB03B">Ação recomendada:</strong><br>
                        Monitorar IEG e IDA no próximo ciclo. Conversa com a equipe para entender fatores psicossociais recentes.
                    </div>
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="risk-card risk-baixo">
                    <div class="risk-title">🟢 Baixo Risco — {pct:.1f}%</div>
                    <div class="risk-body">Indicadores estáveis. Baixa probabilidade de queda no próximo período.</div>
                    <div class="risk-rec">
                        <strong style="color:#00C897">Ação recomendada:</strong><br>
                        Manter acompanhamento padrão. Considerar indicação para bolsa se INDE ≥ 8,23.
                    </div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            # Mini tabela de resumo
            st.markdown('<p style="font-size:0.75rem;color:#64748B;text-transform:uppercase;letter-spacing:0.08em;">Indicadores inseridos</p>', unsafe_allow_html=True)
            resumo = pd.DataFrame({
                "Indicador": ["INDE","IDA","IEG","IAA","IPS","IPV","IAN","IPP"],
                "Valor":     [f"{v:.1f}" for v in [INDE,IDA,IEG,IAA,IPS,IPV,IAN,IPP]],
            })
            st.dataframe(resumo, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# PÁGINA 5 — AVALIAÇÃO EM LOTE
# ══════════════════════════════════════════════════════════════════════════════
elif pagina == "📋  Avaliação em Lote":

    st.markdown('<div class="section-label">// Processamento em massa</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="section-title">Avaliação em Lote</h1>', unsafe_allow_html=True)
    st.markdown('<p class="section-desc">Faça upload de um CSV com múltiplos alunos para obter a probabilidade de risco de toda a turma de uma vez.</p>', unsafe_allow_html=True)

    st.markdown("""
    <div class="chart-card">
        <div class="chart-title">Colunas obrigatórias no CSV</div>
        <div class="chart-desc" style="font-family:'JetBrains Mono',monospace; font-size:0.78rem; color:#00C897;">
            RA · INDE · IAA · IEG · IPS · IDA · IPV · IAN · IPP · Defasagem · Fase_Num · Ano_Ingresso · Ano_Referencia · Idade
        </div>
    </div>
    """, unsafe_allow_html=True)

    arquivo = st.file_uploader("Selecione o arquivo CSV", type=["csv"])

    if arquivo:
        try:
            df_lote = pd.read_csv(arquivo)

            df_lote['Anos_no_Programa']     = df_lote['Ano_Referencia'] - df_lote['Ano_Ingresso']
            df_lote['Media_Comportamental'] = df_lote[['IAA','IEG','IPS']].mean(axis=1)
            df_lote['Media_Academica']      = df_lote[['IDA','IPV','IPP']].mean(axis=1)
            df_lote['Gap_Auto_Real']        = df_lote['IAA'] - df_lote['IDA']

            probs = modelo.predict_proba(df_lote[FEATURES])[:, 1]
            df_lote['Prob_Risco_%']  = (probs * 100).round(1)
            df_lote['Classificacao'] = pd.cut(
                probs, bins=[-0.01,0.30,0.50,1.01],
                labels=['🟢 Baixo','🟡 Moderado','🔴 Alto']
            )

            # KPIs do lote
            n_alto = (probs >= 0.5).sum()
            n_med  = ((probs >= 0.3) & (probs < 0.5)).sum()
            n_bx   = (probs < 0.3).sum()

            st.markdown(f"""
            <div class="kpi-row" style="margin-top:1.5rem;">
                <div class="kpi-card green">
                    <div class="kpi-label">Alunos processados</div>
                    <div class="kpi-value">{len(df_lote)}</div>
                </div>
                <div class="kpi-card green">
                    <div class="kpi-label">Baixo risco</div>
                    <div class="kpi-value green">{n_bx}</div>
                    <div class="kpi-delta">{n_bx/len(df_lote):.1%} da turma</div>
                </div>
                <div class="kpi-card amber">
                    <div class="kpi-label">Risco moderado</div>
                    <div class="kpi-value amber">{n_med}</div>
                    <div class="kpi-delta">{n_med/len(df_lote):.1%} da turma</div>
                </div>
                <div class="kpi-card red">
                    <div class="kpi-label">Alto risco</div>
                    <div class="kpi-value red">{n_alto}</div>
                    <div class="kpi-delta">{n_alto/len(df_lote):.1%} da turma</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

            col_exib = ['RA','Fase_Num','INDE','IDA','IEG','Prob_Risco_%','Classificacao'] \
                       if 'RA' in df_lote.columns else \
                       ['Fase_Num','INDE','IDA','IEG','Prob_Risco_%','Classificacao']

            st.markdown('<p style="font-size:0.78rem;color:#64748B;text-transform:uppercase;letter-spacing:0.08em;margin-top:1.5rem;">Resultado ordenado por risco</p>', unsafe_allow_html=True)
            st.dataframe(
                df_lote[col_exib].sort_values('Prob_Risco_%', ascending=False),
                hide_index=True,
                use_container_width=True,
            )

            csv_out = df_lote.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                "⬇  Baixar resultado completo (CSV)",
                data=csv_out,
                file_name="resultado_risco_passos_magicos.csv",
                mime="text/csv",
            )

        except Exception as e:
            st.error(f"Erro ao processar: {e}")
            st.info("Verifique se todas as colunas obrigatórias estão presentes e com os nomes corretos.")
