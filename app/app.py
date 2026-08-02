"""
Datathon Passos Magicos - Dashboard Analitico + Predicao de Risco
Pos-graduacao Data Analytics - FIAP Postech - Fase 5
"""
import os, joblib, warnings
warnings.filterwarnings("ignore")
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.set_page_config(page_title="Passos Magicos Analytics", page_icon="star",
                   layout="wide", initial_sidebar_state="expanded")

@st.cache_resource
def carregar_modelo():
    m = joblib.load(os.path.join(BASE_DIR, "model", "modelo_risco.pkl"))
    s = joblib.load(os.path.join(BASE_DIR, "model", "scaler.pkl"))
    f = joblib.load(os.path.join(BASE_DIR, "model", "features.pkl"))
    return m, s, f

@st.cache_data
def carregar_dados():
    csv = os.path.join(BASE_DIR, "..", "data", "painel_pede_tratado.csv")
    if not os.path.exists(csv):
        csv = os.path.join(BASE_DIR, "data", "painel_pede_tratado.csv")
    df = pd.read_csv(csv)
    df["Pedra"]  = df["Pedra"].replace({"Agata":"Agata","INCLUIR":None})
    df["Genero"] = df["Genero"].replace({"Menina":"Feminino","Menino":"Masculino"})
    df["IAN_Cat"]= df["IAN"].map({2.5:"Severo",5.0:"Moderado",10.0:"Adequado"})
    return df

modelo, scaler, FEATURES = carregar_modelo()
df_full = carregar_dados()

ORDEM_PEDRA = ["Quartzo","Agata","Ametista","Topazio"]
CORES_PEDRA = {"Quartzo":"#EF4444","Agata":"#F59E0B","Ametista":"#00C897","Topazio":"#3B82F6"}
CORES_ANO   = {2022:"#7C3AED",2023:"#00C897",2024:"#3B82F6"}
CORES_IAN   = {"Severo":"#EF4444","Moderado":"#F59E0B","Adequado":"#00C897"}

# Estilo global matplotlib
FUNDO   = "#111827"
TEXTO   = "#94A3B8"
GRID    = "#1E293B"
ACCENT  = "#00C897"

def estilo_fig(fig, ax_list=None):
    fig.patch.set_facecolor(FUNDO)
    axes = ax_list if ax_list else [fig.gca()]
    for ax in axes:
        ax.set_facecolor(FUNDO)
        ax.tick_params(colors=TEXTO, labelsize=9)
        ax.xaxis.label.set_color(TEXTO)
        ax.yaxis.label.set_color(TEXTO)
        ax.title.set_color("#F1F5F9")
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)
        ax.grid(color=GRID, linewidth=0.5)
    return fig

def show(fig):
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

st.markdown("""<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
*,*::before,*::after{box-sizing:border-box}
html,body,[data-testid="stAppViewContainer"]{background:#0A0F1E!important;color:#F1F5F9!important;font-family:'Inter',sans-serif!important}
[data-testid="stSidebar"]{background:linear-gradient(180deg,#0D1117,#111827)!important;border-right:1px solid #1E293B!important}
[data-testid="stSidebar"] *{color:#CBD5E1!important}
[data-testid="stHeader"]{background:transparent!important}
.block-container{padding:1.5rem 2rem 3rem!important;max-width:1400px!important}
.hero{background:linear-gradient(135deg,#0D1B2A,#111827 40%,#0D1B2A);border:1px solid #1E293B;border-radius:16px;padding:2.5rem 3rem;margin-bottom:2rem;position:relative;overflow:hidden}
.hero::before{content:'';position:absolute;top:-60px;right:-60px;width:300px;height:300px;background:radial-gradient(circle,rgba(0,200,151,.08),transparent 70%)}
.hero-tag{font-family:'JetBrains Mono',monospace;font-size:.7rem;color:#00C897;letter-spacing:.15em;text-transform:uppercase;margin-bottom:.6rem}
.hero-title{font-size:2.2rem;font-weight:800;color:#F1F5F9;line-height:1.15;margin:0 0 .5rem}
.hero-title span{color:#00C897}.hero-sub{font-size:.95rem;color:#94A3B8;margin:0}
.kpi-row{display:flex;gap:1rem;margin-bottom:2rem;flex-wrap:wrap}
.kpi-card{flex:1;min-width:140px;background:#111827;border:1px solid #1E293B;border-radius:12px;padding:1.2rem 1.4rem;position:relative;overflow:hidden}
.kpi-card::before{content:'';position:absolute;top:0;left:0;right:0;height:3px;border-radius:12px 12px 0 0}
.kpi-card.g::before{background:#00C897}.kpi-card.p::before{background:#7C3AED}
.kpi-card.a::before{background:#F59E0B}.kpi-card.r::before{background:#EF4444}.kpi-card.b::before{background:#3B82F6}
.kpi-label{font-size:.72rem;font-weight:500;color:#64748B;text-transform:uppercase;letter-spacing:.08em;margin-bottom:.4rem}
.kpi-value{font-size:1.9rem;font-weight:800;color:#F1F5F9;line-height:1;font-family:'JetBrains Mono',monospace}
.kpi-value.g{color:#00C897}.kpi-value.p{color:#A78BFA}.kpi-value.a{color:#FBB03B}.kpi-value.r{color:#F87171}
.kpi-delta{font-size:.75rem;color:#64748B;margin-top:.3rem}
.sl{font-family:'JetBrains Mono',monospace;font-size:.68rem;color:#00C897;letter-spacing:.15em;text-transform:uppercase;margin-bottom:.3rem}
.st2{font-size:1.35rem;font-weight:700;color:#F1F5F9;margin:0 0 .4rem}
.sd{font-size:.85rem;color:#64748B;margin-bottom:1rem}
.chart-card{background:#111827;border:1px solid #1E293B;border-radius:12px;padding:1.2rem 1.4rem;margin-bottom:1.2rem}
.fbar{background:#111827;border:1px solid #1E293B;border-radius:10px;padding:1rem 1.4rem;margin-bottom:1.2rem}
[data-testid="stTabs"] button{font-size:.85rem!important;font-weight:500!important;color:#64748B!important}
[data-testid="stTabs"] button[aria-selected="true"]{color:#00C897!important;border-bottom:2px solid #00C897!important;background:transparent!important}
.risk-card{border-radius:12px;padding:1.4rem 1.6rem;margin-top:1rem}
.ra{background:rgba(239,68,68,.1);border:1px solid rgba(239,68,68,.3)}
.rm{background:rgba(245,158,11,.1);border:1px solid rgba(245,158,11,.3)}
.rb{background:rgba(0,200,151,.1);border:1px solid rgba(0,200,151,.3)}
.rt{font-size:1.1rem;font-weight:700;margin-bottom:.5rem}
.ra .rt{color:#F87171}.rm .rt{color:#FBB03B}.rb .rt{color:#00C897}
.rb2{font-size:.85rem;color:#94A3B8;line-height:1.6}
.rr{font-size:.82rem;color:#CBD5E1;margin-top:.6rem}
[data-testid="stButton"] button[kind="primary"]{background:linear-gradient(135deg,#00C897,#00A87A)!important;color:#0A0F1E!important;font-weight:700!important;border:none!important;border-radius:10px!important}
[data-testid="stMetric"]{background:#111827!important;border:1px solid #1E293B!important;border-radius:12px!important;padding:1rem 1.2rem!important}
[data-testid="stMetricLabel"]{color:#64748B!important;font-size:.75rem!important}
[data-testid="stMetricValue"]{color:#F1F5F9!important;font-weight:700!important}
hr{border-color:#1E293B!important}
::-webkit-scrollbar{width:6px}::-webkit-scrollbar-track{background:#0A0F1E}::-webkit-scrollbar-thumb{background:#1E293B;border-radius:3px}
</style>""", unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown('<div style="font-size:1.3rem;font-weight:800;color:#F1F5F9">Passos <span style="color:#00C897">Magicos</span></div>', unsafe_allow_html=True)
    st.markdown('<div style="font-family:JetBrains Mono,monospace;font-size:.65rem;color:#334155;margin-bottom:2rem">// DATATHON FIAP POSTECH FASE 5</div>', unsafe_allow_html=True)
    pagina = st.radio("nav", ["Home","Analise","Modelo","Avaliar Aluno"], label_visibility="collapsed")
    st.divider()
    st.markdown('<p style="font-size:.72rem;color:#334155;text-transform:uppercase">Base de dados</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:.82rem;color:#475569">PEDE 2022 2023 2024</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:.72rem;color:#334155;margin-top:.8rem;text-transform:uppercase">Modelo</p>', unsafe_allow_html=True)
    st.markdown('<p style="font-size:.82rem;color:#475569">XGBoost AUC 0.795</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:.72rem;color:#334155;margin-top:.8rem;text-transform:uppercase">Registros</p>', unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:.82rem;color:#475569">{len(df_full):,} alunos-ano</p>', unsafe_allow_html=True)

# ═══ HOME ═════════════════════════════════════════════════════════════════════
if pagina == "Home":
    st.markdown("""<div class="hero">
    <div class="hero-tag">// Datathon Analise Educacional</div>
    <h1 class="hero-title">Transformando dados em <span>impacto real</span></h1>
    <p class="hero-sub">Dashboard analitico da Pesquisa Extensiva do Desenvolvimento Educacional (PEDE) 2022-2024.</p>
    </div>""", unsafe_allow_html=True)

    total = df_full["RA"].nunique()
    inde_m = df_full["INDE"].mean()
    a22 = df_full[df_full["Ano_Referencia"]==2022]["RA"].nunique()
    a24 = df_full[df_full["Ano_Referencia"]==2024]["RA"].nunique()
    cresc = (a24-a22)/a22*100
    pct_t = (df_full[df_full["Ano_Referencia"]==2024]["Pedra"]=="Topazio").mean()*100
    pct_s = (df_full["IAN"]==2.5).mean()*100

    st.markdown(f"""<div class="kpi-row">
    <div class="kpi-card g"><div class="kpi-label">Alunos unicos</div><div class="kpi-value g">{total:,}</div><div class="kpi-delta">2022-2024</div></div>
    <div class="kpi-card p"><div class="kpi-label">Crescimento</div><div class="kpi-value p">+{cresc:.0f}%</div><div class="kpi-delta">{a22} para {a24}</div></div>
    <div class="kpi-card b"><div class="kpi-label">INDE medio</div><div class="kpi-value">{inde_m:.2f}</div><div class="kpi-delta">escala 0-10</div></div>
    <div class="kpi-card g"><div class="kpi-label">Topazio 2024</div><div class="kpi-value g">{pct_t:.0f}%</div><div class="kpi-delta">vs 15% em 2022</div></div>
    <div class="kpi-card r"><div class="kpi-label">IAN Severo</div><div class="kpi-value r">{pct_s:.1f}%</div><div class="kpi-delta">defasagem critica</div></div>
    </div>""", unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        anos = df_full.groupby("Ano_Referencia")["RA"].nunique()
        fig, ax = plt.subplots(figsize=(6,3.5))
        bars = ax.bar([2022,2023,2024], anos.values, color=[CORES_ANO[a] for a in [2022,2023,2024]], width=0.5)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+5, str(int(b.get_height())), ha="center", color="#F1F5F9", fontsize=11, fontweight="bold")
        ax.set_xticks([2022,2023,2024]); ax.set_title("Crescimento da base por ano", fontsize=12)
        ax.set_ylim(0, anos.max()*1.15)
        estilo_fig(fig); show(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        df_p = df_full[df_full["Pedra"].isin(ORDEM_PEDRA)]
        pedra_pct = df_p.groupby(["Ano_Referencia","Pedra"]).size().unstack(fill_value=0)
        pedra_pct = pedra_pct.reindex(columns=ORDEM_PEDRA, fill_value=0)
        pedra_pct_norm = pedra_pct.div(pedra_pct.sum(axis=1), axis=0)*100
        fig, ax = plt.subplots(figsize=(6,3.5))
        bottom = np.zeros(3)
        anos_x = [2022,2023,2024]
        for pedra in ORDEM_PEDRA:
            vals = [pedra_pct_norm.loc[a, pedra] if a in pedra_pct_norm.index else 0 for a in anos_x]
            bars = ax.bar(anos_x, vals, bottom=bottom, color=CORES_PEDRA[pedra], label=pedra, width=0.5)
            for i,(b,v) in enumerate(zip(bars,vals)):
                if v > 5:
                    ax.text(b.get_x()+b.get_width()/2, bottom[i]+v/2, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
            bottom += np.array(vals)
        ax.set_xticks(anos_x); ax.set_ylim(0,110); ax.set_title("Distribuicao de Pedra por ano (%)", fontsize=12)
        ax.legend(loc="upper left", fontsize=8, framealpha=0, labelcolor=TEXTO)
        estilo_fig(fig); show(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="chart-card">', unsafe_allow_html=True)
    medias = df_full.groupby("Ano_Referencia")[["INDE","IDA","IEG"]].mean()
    fig, ax = plt.subplots(figsize=(8,3.5))
    for col, cor, nm in [("INDE",ACCENT,"INDE"),("IDA","#7C3AED","IDA"),("IEG","#F59E0B","IEG")]:
        ax.plot([2022,2023,2024], [medias.loc[a,col] for a in [2022,2023,2024]],
                marker="o", color=cor, linewidth=2.5, markersize=8, label=nm)
        for a in [2022,2023,2024]:
            ax.annotate(f"{medias.loc[a,col]:.2f}", (a, medias.loc[a,col]), textcoords="offset points", xytext=(0,10), ha="center", color=cor, fontsize=10, fontweight="bold")
    ax.set_xticks([2022,2023,2024]); ax.set_ylim(5,10); ax.set_title("INDE, IDA e IEG medio por ano", fontsize=12)
    ax.legend(framealpha=0, labelcolor=TEXTO, fontsize=9)
    estilo_fig(fig); show(fig)
    st.markdown("</div>", unsafe_allow_html=True)

    col1,col2,col3 = st.columns(3)
    insights = [("📈","#00C897","Programa em crescimento","Base cresceu 34% entre 2022 e 2024. Topazio saiu de 15% para 28%."),
                ("🔗","#7C3AED","Engajamento como driver","IEG e IDA sao os maiores preditores do INDE (r=0.75 e r=0.79)."),
                ("⚠️","#F59E0B","Autopercepção superestimada","IAA tem correlacao fraca com IDA (r=0.12) — alunos se autoavaliam bem independente do desempenho.")]
    for col,(ic,cor,tit,txt) in zip([col1,col2,col3],insights):
        with col:
            st.markdown(f'<div class="chart-card" style="border-left:3px solid {cor};padding:1.2rem"><div style="font-size:1.5rem;margin-bottom:.5rem">{ic}</div><div style="font-size:.9rem;font-weight:600;color:#CBD5E1;margin-bottom:.4rem">{tit}</div><div style="font-size:.8rem;color:#475569;line-height:1.6">{txt}</div></div>', unsafe_allow_html=True)

# ═══ ANALISE ══════════════════════════════════════════════════════════════════
elif pagina == "Analise":
    st.markdown('<div class="sl">// Pesquisa PEDE 2022-2024</div>', unsafe_allow_html=True)
    st.markdown('<h1 class="st2">Analise por Indicador</h1>', unsafe_allow_html=True)

    st.markdown('<div class="fbar">', unsafe_allow_html=True)
    fc1,fc2,fc3,fc4 = st.columns(4)
    with fc1: anos_sel = st.multiselect("Ano",[2022,2023,2024],default=[2022,2023,2024])
    with fc2:
        fases = ["Todos"]+["ALFA" if x==0 else f"Fase {int(x)}" for x in sorted(df_full["Fase_Num"].dropna().unique())]
        fase_sel = st.selectbox("Fase",fases)
    with fc3: gen_sel = st.multiselect("Genero",["Feminino","Masculino"],default=["Feminino","Masculino"])
    with fc4:
        ped_disp = [p for p in ORDEM_PEDRA if p in df_full["Pedra"].values]
        ped_sel = st.multiselect("Pedra",ped_disp,default=ped_disp)
    st.markdown("</div>", unsafe_allow_html=True)

    df = df_full.copy()
    if anos_sel: df = df[df["Ano_Referencia"].isin(anos_sel)]
    if fase_sel != "Todos":
        fn = 0 if fase_sel=="ALFA" else int(fase_sel.split()[-1])
        df = df[df["Fase_Num"]==fn]
    if gen_sel: df = df[df["Genero"].isin(gen_sel)]
    if ped_sel: df = df[df["Pedra"].isin(ped_sel)|df["Pedra"].isna()]
    if df.empty: st.warning("Nenhum dado com os filtros selecionados."); st.stop()

    tabs = st.tabs(["IAN","IDA","IEG","IAA","IPS","IPP","IPV","INDE","Efetividade","Insights"])

    # ── IAN ──────────────────────────────────────────────────────────────────
    with tabs[0]:
        st.markdown('<div class="sl">// Pergunta 1</div><h3 class="st2">Adequacao ao Nivel (IAN)</h3><p class="sd">IAN: 2.5 Severo, 5.0 Moderado, 10.0 Adequado.</p>', unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            ic = df["IAN_Cat"].value_counts().reindex(["Severo","Moderado","Adequado"]).fillna(0)
            fig, ax = plt.subplots(figsize=(5,4))
            wedges,_,autotexts = ax.pie(ic.values, labels=ic.index, autopct="%1.0f%%",
                colors=[CORES_IAN[k] for k in ic.index], startangle=90,
                wedgeprops=dict(width=0.55), pctdistance=0.75)
            for t in autotexts: t.set_color("white"); t.set_fontsize(10)
            for t in wedges: t.set_edgecolor(FUNDO)
            ax.set_title("Distribuicao geral do IAN", fontsize=12)
            estilo_fig(fig); show(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            anos_x = sorted(df["Ano_Referencia"].unique())
            fig, ax = plt.subplots(figsize=(5,4))
            bottom = np.zeros(len(anos_x))
            for cat in ["Severo","Moderado","Adequado"]:
                vals = []
                for a in anos_x:
                    sub = df[df["Ano_Referencia"]==a]["IAN_Cat"]
                    vals.append((sub==cat).mean()*100)
                bars = ax.bar(anos_x, vals, bottom=bottom, color=CORES_IAN[cat], label=cat, width=0.5)
                for b,v,bt in zip(bars,vals,bottom):
                    if v>4: ax.text(b.get_x()+b.get_width()/2, bt+v/2, f"{v:.0f}%", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
                bottom += np.array(vals)
            ax.set_xticks(anos_x); ax.set_ylim(0,115); ax.set_title("IAN por ano (%)", fontsize=12)
            ax.legend(framealpha=0, labelcolor=TEXTO, fontsize=9)
            estilo_fig(fig); show(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        c1,c2,c3 = st.columns(3)
        c1.metric("Severo",f"{(df['IAN']==2.5).mean()*100:.1f}%",f"{int((df['IAN']==2.5).sum())} alunos")
        c2.metric("Moderado",f"{(df['IAN']==5.0).mean()*100:.1f}%",f"{int((df['IAN']==5.0).sum())} alunos")
        c3.metric("Adequado",f"{(df['IAN']==10.0).mean()*100:.1f}%",f"{int((df['IAN']==10.0).sum())} alunos")

    # ── IDA ──────────────────────────────────────────────────────────────────
    with tabs[1]:
        st.markdown('<div class="sl">// Pergunta 2</div><h3 class="st2">Desempenho Academico (IDA)</h3><p class="sd">Media das notas do indicador de aprendizagem.</p>', unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            anos_x = sorted(df["Ano_Referencia"].unique())
            ida_m = [df[df["Ano_Referencia"]==a]["IDA"].mean() for a in anos_x]
            fig, ax = plt.subplots(figsize=(5,3.5))
            ax.plot(anos_x, ida_m, marker="o", color="#7C3AED", linewidth=2.5, markersize=9)
            for a,v in zip(anos_x,ida_m): ax.annotate(f"{v:.2f}", (a,v), textcoords="offset points", xytext=(0,10), ha="center", color="#A78BFA", fontsize=11, fontweight="bold")
            ax.set_xticks(anos_x); ax.set_ylim(0,10); ax.set_title("IDA medio por ano", fontsize=12)
            estilo_fig(fig); show(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5,3.5))
            df_f = df[df["Fase_Num"].between(0,8)]
            fases_x = sorted(df_f["Fase_Num"].unique())
            flabels = ["ALFA" if f==0 else f"F{int(f)}" for f in fases_x]
            for a, cor in CORES_ANO.items():
                if a in df_f["Ano_Referencia"].values:
                    vals = [df_f[(df_f["Ano_Referencia"]==a)&(df_f["Fase_Num"]==f)]["IDA"].mean() for f in fases_x]
                    ax.plot(flabels, vals, marker="o", color=cor, linewidth=2, markersize=6, label=str(a))
            ax.set_ylim(0,10); ax.set_title("IDA por Fase e Ano", fontsize=12)
            ax.legend(framealpha=0, labelcolor=TEXTO, fontsize=9)
            plt.xticks(rotation=30, ha="right")
            estilo_fig(fig); show(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        df_bx = df[df["Pedra"].isin(ORDEM_PEDRA)&df["IDA"].notna()]
        fig, ax = plt.subplots(figsize=(8,3.5))
        data_box = [df_bx[df_bx["Pedra"]==p]["IDA"].dropna().values for p in ORDEM_PEDRA]
        bp = ax.boxplot(data_box, patch_artist=True, labels=ORDEM_PEDRA, widths=0.5)
        for patch, p in zip(bp["boxes"], ORDEM_PEDRA): patch.set_facecolor(CORES_PEDRA[p]); patch.set_alpha(0.8)
        for elem in ["whiskers","caps","medians","fliers"]:
            for item in bp[elem]: item.set_color(TEXTO)
        ax.set_title("IDA por Pedra", fontsize=12); ax.set_ylim(0,10)
        estilo_fig(fig); show(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── IEG ──────────────────────────────────────────────────────────────────
    with tabs[2]:
        st.markdown('<div class="sl">// Pergunta 3</div><h3 class="st2">Engajamento (IEG)</h3><p class="sd">Principal preditor do INDE (r=0.75).</p>', unsafe_allow_html=True)
        dsc = df[["IEG","IDA","IPV","Ano_Referencia"]].dropna()
        c1,c2 = st.columns(2)
        with c1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            r = dsc["IEG"].corr(dsc["IDA"])
            fig, ax = plt.subplots(figsize=(5,4))
            for a, cor in CORES_ANO.items():
                sub = dsc[dsc["Ano_Referencia"]==a]
                if not sub.empty: ax.scatter(sub["IEG"], sub["IDA"], alpha=0.3, s=12, color=cor, label=str(a))
            m,b = np.polyfit(dsc["IEG"],dsc["IDA"],1)
            xl = np.linspace(dsc["IEG"].min(),dsc["IEG"].max(),100)
            ax.plot(xl,m*xl+b,"--",color="#F1F5F9",linewidth=1.5,label=f"Tendencia")
            ax.set_xlabel("IEG"); ax.set_ylabel("IDA"); ax.set_title(f"IEG vs IDA  (r={r:.2f})", fontsize=12)
            ax.legend(framealpha=0, labelcolor=TEXTO, fontsize=8)
            estilo_fig(fig); show(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            r2 = dsc["IEG"].corr(dsc["IPV"])
            fig, ax = plt.subplots(figsize=(5,4))
            for a, cor in CORES_ANO.items():
                sub = dsc[dsc["Ano_Referencia"]==a]
                if not sub.empty: ax.scatter(sub["IEG"], sub["IPV"], alpha=0.3, s=12, color=cor, label=str(a))
            m,b = np.polyfit(dsc["IEG"],dsc["IPV"],1)
            ax.plot(xl,m*xl+b,"--",color="#F1F5F9",linewidth=1.5)
            ax.set_xlabel("IEG"); ax.set_ylabel("IPV"); ax.set_title(f"IEG vs IPV  (r={r2:.2f})", fontsize=12)
            ax.legend(framealpha=0, labelcolor=TEXTO, fontsize=8)
            estilo_fig(fig); show(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        ieg_f = df[df["Fase_Num"].between(0,8)].groupby("Fase_Num")["IEG"].mean()
        labels_f = ["ALFA" if f==0 else f"Fase {int(f)}" for f in ieg_f.index]
        fig, ax = plt.subplots(figsize=(8,3.5))
        colors_f = [ACCENT if v==ieg_f.max() else "#1E3A5F" for v in ieg_f.values]
        bars = ax.bar(labels_f, ieg_f.values, color=colors_f, width=0.6)
        for b in bars: ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.05, f"{b.get_height():.2f}", ha="center", color="#F1F5F9", fontsize=9)
        ax.set_ylim(0,10); ax.set_title("IEG medio por Fase", fontsize=12)
        estilo_fig(fig); show(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── IAA ──────────────────────────────────────────────────────────────────
    with tabs[3]:
        st.markdown('<div class="sl">// Pergunta 4</div><h3 class="st2">Autoavaliacao (IAA)</h3><p class="sd">Percepcao do aluno vs desempenho real.</p>', unsafe_allow_html=True)
        di = df[["IAA","IDA","Ano_Referencia"]].dropna()
        c1,c2 = st.columns(2)
        with c1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            r = di["IAA"].corr(di["IDA"])
            fig, ax = plt.subplots(figsize=(5,4))
            for a, cor in CORES_ANO.items():
                sub = di[di["Ano_Referencia"]==a]
                if not sub.empty: ax.scatter(sub["IAA"], sub["IDA"], alpha=0.3, s=12, color=cor, label=str(a))
            m,b = np.polyfit(di["IAA"],di["IDA"],1)
            xl = np.linspace(di["IAA"].min(),di["IAA"].max(),100)
            ax.plot(xl,m*xl+b,"--",color="#F1F5F9",linewidth=1.5)
            ax.set_xlabel("IAA"); ax.set_ylabel("IDA"); ax.set_title(f"IAA vs IDA  (r={r:.2f}) — Correlacao fraca", fontsize=11)
            ax.legend(framealpha=0, labelcolor=TEXTO, fontsize=8)
            estilo_fig(fig); show(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            dg = di.copy(); dg["Gap"] = dg["IAA"]-dg["IDA"]
            fig, ax = plt.subplots(figsize=(5,4))
            ax.hist(dg["Gap"].dropna(), bins=30, color="#7C3AED", alpha=0.8, edgecolor=FUNDO)
            ax.axvline(0, color="#F1F5F9", linestyle="--", linewidth=1.2, label="Sem gap")
            ax.axvline(dg["Gap"].mean(), color="#F59E0B", linestyle=":", linewidth=1.5, label=f"Media:{dg['Gap'].mean():.1f}")
            ax.set_xlabel("Gap (IAA - IDA)"); ax.set_title("Gap Autopercep. vs Realidade", fontsize=12)
            ax.legend(framealpha=0, labelcolor=TEXTO, fontsize=8)
            estilo_fig(fig); show(fig)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── IPS ──────────────────────────────────────────────────────────────────
    with tabs[4]:
        st.markdown('<div class="sl">// Pergunta 5</div><h3 class="st2">Aspectos Psicossociais (IPS)</h3><p class="sd">Padroes de IPS que antecedem quedas de desempenho.</p>', unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            anos_x = sorted(df["Ano_Referencia"].unique())
            ips_m = [df[df["Ano_Referencia"]==a]["IPS"].mean() for a in anos_x]
            fig, ax = plt.subplots(figsize=(5,3.5))
            ax.plot(anos_x, ips_m, marker="o", color="#EF4444", linewidth=2.5, markersize=9)
            for a,v in zip(anos_x,ips_m): ax.annotate(f"{v:.2f}", (a,v), textcoords="offset points", xytext=(0,10), ha="center", color="#F87171", fontsize=11, fontweight="bold")
            ax.set_xticks(anos_x); ax.set_ylim(0,10); ax.set_title("IPS medio por ano", fontsize=12)
            estilo_fig(fig); show(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            ds = df[["IPS","IDA","Ano_Referencia"]].dropna(); r = ds["IPS"].corr(ds["IDA"])
            fig, ax = plt.subplots(figsize=(5,4))
            for a, cor in CORES_ANO.items():
                sub = ds[ds["Ano_Referencia"]==a]
                if not sub.empty: ax.scatter(sub["IPS"], sub["IDA"], alpha=0.3, s=12, color=cor, label=str(a))
            m,b = np.polyfit(ds["IPS"],ds["IDA"],1)
            xl = np.linspace(ds["IPS"].min(),ds["IPS"].max(),100)
            ax.plot(xl,m*xl+b,"--",color="#F1F5F9",linewidth=1.5)
            ax.set_xlabel("IPS"); ax.set_ylabel("IDA"); ax.set_title(f"IPS vs IDA  (r={r:.2f})", fontsize=12)
            ax.legend(framealpha=0, labelcolor=TEXTO, fontsize=8)
            estilo_fig(fig); show(fig)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── IPP ──────────────────────────────────────────────────────────────────
    with tabs[5]:
        st.markdown('<div class="sl">// Pergunta 6</div><h3 class="st2">Aspectos Psicopedagogicos (IPP)</h3><p class="sd">Disponivel a partir de 2023. Confirma o IAN?</p>', unsafe_allow_html=True)
        di = df[df["IPP"].notna()&df["IAN_Cat"].notna()]
        c1,c2 = st.columns(2)
        with c1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5,4))
            data_box = [di[di["IAN_Cat"]==cat]["IPP"].dropna().values for cat in ["Severo","Moderado","Adequado"]]
            bp = ax.boxplot(data_box, patch_artist=True, labels=["Severo","Moderado","Adequado"], widths=0.5)
            for patch,cat in zip(bp["boxes"],["Severo","Moderado","Adequado"]): patch.set_facecolor(CORES_IAN[cat]); patch.set_alpha(0.8)
            for elem in ["whiskers","caps","medians"]:
                for item in bp[elem]: item.set_color(TEXTO)
            ax.set_title("IPP por categoria IAN", fontsize=12); ax.set_ylim(0,10)
            estilo_fig(fig); show(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            r = di["IAN"].corr(di["IPP"])
            fig, ax = plt.subplots(figsize=(5,4))
            for a, cor in CORES_ANO.items():
                sub = di[di["Ano_Referencia"]==a]
                if not sub.empty: ax.scatter(sub["IAN"], sub["IPP"], alpha=0.4, s=12, color=cor, label=str(a))
            m,b = np.polyfit(di["IAN"],di["IPP"],1)
            xl = np.linspace(di["IAN"].min(),di["IAN"].max(),100)
            ax.plot(xl,m*xl+b,"--",color="#F1F5F9",linewidth=1.5)
            ax.set_xlabel("IAN"); ax.set_ylabel("IPP"); ax.set_title(f"IPP vs IAN  (r={r:.2f})", fontsize=12)
            ax.legend(framealpha=0, labelcolor=TEXTO, fontsize=8)
            estilo_fig(fig); show(fig)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── IPV ──────────────────────────────────────────────────────────────────
    with tabs[6]:
        st.markdown('<div class="sl">// Pergunta 7</div><h3 class="st2">Ponto de Virada (IPV)</h3><p class="sd">Quais indicadores mais influenciam o IPV?</p>', unsafe_allow_html=True)
        inds = ["IDA","IEG","IAA","IPS","IAN","IPP"]
        dv = df[["IPV"]+inds].dropna()
        corrs = {c: dv["IPV"].corr(dv[c]) for c in inds}
        c1,c2 = st.columns(2)
        with c1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5,4))
            vals = list(corrs.values()); labs = list(corrs.keys())
            cores_bar = [ACCENT if v>=.5 else "#F59E0B" if v>=.3 else TEXTO for v in vals]
            bars = ax.barh(labs, vals, color=cores_bar, height=0.55)
            for b,v in zip(bars,vals): ax.text(v+0.01, b.get_y()+b.get_height()/2, f"{v:.2f}", va="center", color="#F1F5F9", fontsize=9)
            ax.set_xlim(0,.9); ax.set_title("Correlacao com IPV", fontsize=12)
            estilo_fig(fig); show(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            d22 = df[df["Ano_Referencia"]==2022].copy()
            d22["PVL"] = d22["Atingiu_Ponto_Virada"].map({"Sim":"Atingiu","Nao":"Nao atingiu"})
            d22v = d22[d22["PVL"].notna()]
            if not d22v.empty:
                fig, ax = plt.subplots(figsize=(5,4))
                x = np.arange(3); w = 0.35
                ind_cols = ["IDA","IEG","IAA"]
                m_at = [d22v[d22v["PVL"]=="Atingiu"][c].mean() for c in ind_cols]
                m_na = [d22v[d22v["PVL"]=="Nao atingiu"][c].mean() for c in ind_cols]
                ax.bar(x-w/2, m_at, w, color=ACCENT, label="Atingiu PV", alpha=0.85)
                ax.bar(x+w/2, m_na, w, color="#EF4444", label="Nao atingiu", alpha=0.85)
                for xi,v in zip(x-w/2, m_at): ax.text(xi, v+0.05, f"{v:.2f}", ha="center", color="#F1F5F9", fontsize=8)
                for xi,v in zip(x+w/2, m_na): ax.text(xi, v+0.05, f"{v:.2f}", ha="center", color="#F1F5F9", fontsize=8)
                ax.set_xticks(x); ax.set_xticklabels(ind_cols); ax.set_ylim(0,10)
                ax.set_title("PV: Atingiu vs Nao atingiu (2022)", fontsize=11)
                ax.legend(framealpha=0, labelcolor=TEXTO, fontsize=8)
                estilo_fig(fig); show(fig)
            else:
                st.info("Dado de Ponto de Virada disponivel apenas em 2022.")
            st.markdown("</div>", unsafe_allow_html=True)

    # ── INDE ─────────────────────────────────────────────────────────────────
    with tabs[7]:
        st.markdown('<div class="sl">// Pergunta 8</div><h3 class="st2">Multidimensionalidade — INDE</h3><p class="sd">Quais indicadores elevam mais o INDE?</p>', unsafe_allow_html=True)
        di = df[["INDE","IAA","IEG","IPS","IDA","IPV","IAN","IPP"]].dropna()
        ci = {c: di["INDE"].corr(di[c]) for c in ["IAA","IEG","IPS","IDA","IPV","IAN","IPP"]}
        c1,c2 = st.columns(2)
        with c1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5,4))
            vals = list(ci.values()); labs = list(ci.keys())
            idx = np.argsort(vals)
            vals_s = [vals[i] for i in idx]; labs_s = [labs[i] for i in idx]
            cores_b = [ACCENT if v>=.6 else "#F59E0B" if v>=.4 else TEXTO for v in vals_s]
            bars = ax.barh(labs_s, vals_s, color=cores_b, height=0.55)
            for b,v in zip(bars,vals_s): ax.text(v+0.01, b.get_y()+b.get_height()/2, f"{v:.2f}", va="center", color="#F1F5F9", fontsize=9)
            ax.set_xlim(0,1); ax.set_title("Correlacao com INDE", fontsize=12)
            estilo_fig(fig); show(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            cats_r = ["IAA","IEG","IPS","IDA","IPV","IAN"]
            df_rad = df[df["Pedra"].isin(ORDEM_PEDRA)][["Pedra"]+cats_r].dropna()
            mp = df_rad.groupby("Pedra")[cats_r].mean()
            n = len(cats_r)
            angles = [i/n*2*np.pi for i in range(n)]+[0]
            fig, ax = plt.subplots(figsize=(5,4), subplot_kw=dict(polar=True))
            ax.set_facecolor(FUNDO); fig.patch.set_facecolor(FUNDO)
            ax.set_theta_offset(np.pi/2); ax.set_theta_direction(-1)
            ax.set_xticks(angles[:-1]); ax.set_xticklabels(cats_r, color=TEXTO, size=8)
            ax.set_ylim(0,10); ax.set_yticks([2,4,6,8,10]); ax.set_yticklabels(["2","4","6","8","10"], color=TEXTO, size=7)
            ax.grid(color=GRID, linewidth=0.5)
            for p in ORDEM_PEDRA:
                if p in mp.index:
                    vals_r = [mp.loc[p,c] for c in cats_r]+[mp.loc[p,cats_r[0]]]
                    ax.plot(angles, vals_r, color=CORES_PEDRA[p], linewidth=2, label=p)
                    ax.fill(angles, vals_r, color=CORES_PEDRA[p], alpha=0.07)
            ax.set_title("Perfil por Pedra", fontsize=12, color="#F1F5F9", pad=15)
            ax.legend(framealpha=0, labelcolor=TEXTO, fontsize=8, loc="upper right", bbox_to_anchor=(1.3,1.1))
            show(fig)
            st.markdown("</div>", unsafe_allow_html=True)

    # ── EFETIVIDADE ───────────────────────────────────────────────────────────
    with tabs[8]:
        st.markdown('<div class="sl">// Pergunta 10</div><h3 class="st2">Efetividade do Programa</h3><p class="sd">Melhora consistente ao longo dos anos?</p>', unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            fig, ax = plt.subplots(figsize=(5,3.5))
            df_f2 = df[df["Fase_Num"].between(0,8)]
            fases_u = sorted(df_f2["Fase_Num"].unique())
            fl = ["ALFA" if f==0 else f"F{int(f)}" for f in fases_u]
            for a, cor in CORES_ANO.items():
                if a in df_f2["Ano_Referencia"].values:
                    vals = [df_f2[(df_f2["Ano_Referencia"]==a)&(df_f2["Fase_Num"]==f)]["INDE"].mean() for f in fases_u]
                    ax.plot(fl, vals, marker="o", color=cor, linewidth=2, markersize=6, label=str(a))
            ax.set_ylim(4,10); ax.set_title("INDE medio por Fase e Ano", fontsize=12)
            ax.legend(framealpha=0, labelcolor=TEXTO, fontsize=9)
            plt.xticks(rotation=30, ha="right")
            estilo_fig(fig); show(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            anos_x = sorted(df["Ano_Referencia"].unique())
            fig, ax = plt.subplots(figsize=(5,3.5))
            bottom = np.zeros(len(anos_x))
            df_ef = df[df["Pedra"].isin(ORDEM_PEDRA)]
            for p in ORDEM_PEDRA:
                vals = [df_ef[(df_ef["Ano_Referencia"]==a)&(df_ef["Pedra"]==p)].shape[0] for a in anos_x]
                bars = ax.bar(anos_x, vals, bottom=bottom, color=CORES_PEDRA[p], label=p, width=0.5)
                bottom += np.array(vals)
            ax.set_xticks(anos_x); ax.set_title("Volume por Pedra e Ano", fontsize=12)
            ax.legend(framealpha=0, labelcolor=TEXTO, fontsize=9)
            estilo_fig(fig); show(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        anos_x = sorted(df["Ano_Referencia"].unique())
        def_neg = [((df[df["Ano_Referencia"]==a]["Defasagem"]<0).mean()*100) for a in anos_x]
        fig, ax = plt.subplots(figsize=(8,3.5))
        ax.plot(anos_x, def_neg, marker="o", color="#EF4444", linewidth=2.5, markersize=9)
        for a,v in zip(anos_x,def_neg): ax.annotate(f"{v:.1f}%", (a,v), textcoords="offset points", xytext=(0,10), ha="center", color="#F87171", fontsize=11, fontweight="bold")
        ax.set_xticks(anos_x); ax.set_ylim(0,100); ax.yaxis.set_major_formatter(mticker.PercentFormatter())
        ax.set_title("% alunos com defasagem negativa", fontsize=12)
        estilo_fig(fig); show(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    # ── INSIGHTS ─────────────────────────────────────────────────────────────
    with tabs[9]:
        st.markdown('<div class="sl">// Pergunta 11</div><h3 class="st2">Insights Adicionais</h3>', unsafe_allow_html=True)
        c1,c2 = st.columns(2)
        with c1:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            dg2 = df[df["Genero"].isin(["Feminino","Masculino"])&df["INDE"].notna()]
            anos_x = sorted(dg2["Ano_Referencia"].unique())
            fig, ax = plt.subplots(figsize=(5,3.5))
            for gen, cor in [("Feminino","#F59E0B"),("Masculino","#3B82F6")]:
                vals = [dg2[(dg2["Ano_Referencia"]==a)&(dg2["Genero"]==gen)]["INDE"].mean() for a in anos_x]
                ax.plot(anos_x, vals, marker="o", color=cor, linewidth=2.5, markersize=8, label=gen)
                for a,v in zip(anos_x,vals): ax.annotate(f"{v:.2f}", (a,v), textcoords="offset points", xytext=(0,9), ha="center", color=cor, fontsize=9)
            ax.set_xticks(anos_x); ax.set_ylim(5,10); ax.set_title("INDE medio por Genero e Ano", fontsize=12)
            ax.legend(framealpha=0, labelcolor=TEXTO, fontsize=9)
            estilo_fig(fig); show(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="chart-card">', unsafe_allow_html=True)
            di2 = df[df["Idade"].between(7,25)]
            fig, ax = plt.subplots(figsize=(5,3.5))
            for a, cor in CORES_ANO.items():
                sub = di2[di2["Ano_Referencia"]==a]["Idade"].dropna()
                if not sub.empty: ax.hist(sub, bins=18, color=cor, alpha=0.6, label=str(a), edgecolor=FUNDO)
            ax.set_xlabel("Idade"); ax.set_title("Distribuicao de idades por ano", fontsize=12)
            ax.legend(framealpha=0, labelcolor=TEXTO, fontsize=9)
            estilo_fig(fig); show(fig)
            st.markdown("</div>", unsafe_allow_html=True)
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        cols_c = ["INDE","IAA","IEG","IPS","IDA","IPV","IAN","IPP"]
        cm = df[cols_c].dropna().corr()
        fig, ax = plt.subplots(figsize=(8,5))
        im = ax.imshow(cm.values, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(cols_c))); ax.set_yticks(range(len(cols_c)))
        ax.set_xticklabels(cols_c, color=TEXTO, fontsize=9)
        ax.set_yticklabels(cols_c, color=TEXTO, fontsize=9)
        for i in range(len(cols_c)):
            for j in range(len(cols_c)):
                ax.text(j, i, f"{cm.iloc[i,j]:.2f}", ha="center", va="center", color="white" if abs(cm.iloc[i,j])>0.5 else "#333", fontsize=8)
        plt.colorbar(im, ax=ax, fraction=0.03)
        ax.set_title("Mapa de correlacoes entre indicadores", fontsize=12)
        estilo_fig(fig); show(fig)
        st.markdown("</div>", unsafe_allow_html=True)

# ═══ MODELO ═══════════════════════════════════════════════════════════════════
elif pagina == "Modelo":
    st.markdown('<div class="sl">// Pergunta 9 Machine Learning</div><h1 class="st2">Modelo de Risco de Defasagem</h1><p class="sd">XGBoost treinado para identificar alunos com risco de queda no INDE ou aumento da defasagem.</p>', unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Algoritmo","XGBoost"); c2.metric("ROC-AUC","0.795","teste"); c3.metric("Acuracia","80%"); c4.metric("Features","15")
    st.divider()
    col1,col2 = st.columns(2)
    with col1:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        nomes = ["Reg Logistica","Random Forest","XGBoost"]
        auc_cv = [0.670,0.789,0.805]; auc_t = [0.697,0.785,0.795]
        x = np.arange(3); w = 0.35
        fig, ax = plt.subplots(figsize=(6,3.5))
        b1 = ax.bar(x-w/2, auc_cv, w, color="#475569", label="AUC CV", alpha=0.9)
        b2 = ax.bar(x+w/2, auc_t, w, color=ACCENT, label="AUC Teste", alpha=0.9)
        for b,v in zip(list(b1)+list(b2), auc_cv+auc_t): ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.005, f"{v:.3f}", ha="center", color="#F1F5F9", fontsize=9)
        ax.set_xticks(x); ax.set_xticklabels(nomes, fontsize=8); ax.set_ylim(0.5,0.9)
        ax.legend(framealpha=0, labelcolor=TEXTO, fontsize=9); ax.set_title("ROC-AUC por modelo", fontsize=12)
        estilo_fig(fig); show(fig)
        st.markdown("</div>", unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="chart-card">', unsafe_allow_html=True)
        feats = ["IAA","IPS","IAN","Idade","Fase_Num","Defasagem","Gap_Auto_Real","IPP","Anos_Prog","Media_Comp","IPV","Media_Acad","IEG","IDA","INDE"]
        imps  = [0.003,0.005,0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09,0.10,0.12,0.15,0.18]
        cores_f = [ACCENT if v>=.10 else "#7C3AED" if v>=.06 else TEXTO for v in imps]
        fig, ax = plt.subplots(figsize=(6,5))
        bars = ax.barh(feats, imps, color=cores_f, height=0.6)
        for b,v in zip(bars,imps): ax.text(v+0.002, b.get_y()+b.get_height()/2, f"{v:.3f}", va="center", color="#F1F5F9", fontsize=8)
        ax.set_xlim(0,0.22); ax.set_title("Importancia das Features — XGBoost", fontsize=12)
        estilo_fig(fig); show(fig)
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('<div class="chart-card"><div style="font-size:.88rem;font-weight:600;color:#CBD5E1;margin-bottom:.5rem">Como o risco e definido</div><div style="font-size:.82rem;color:#475569;line-height:1.7"><span style="color:#00C897">▸</span> Queda de INDE superior a <strong style="color:#F1F5F9">0.3 ponto</strong> no ano seguinte<br><span style="color:#00C897">▸</span> Piora no nivel de defasagem<br><br>Permite usar dados do ano atual para intervir antes que a queda aconteca.</div></div>', unsafe_allow_html=True)

# ═══ AVALIAR ALUNO ════════════════════════════════════════════════════════════
elif pagina == "Avaliar Aluno":
    st.markdown('<div class="sl">// Simulador de risco individual</div><h1 class="st2">Avaliar Aluno</h1><p class="sd">Insira os indicadores para calcular a probabilidade de risco no proximo ciclo.</p>', unsafe_allow_html=True)
    c1,c2,c3 = st.columns(3)
    with c1:
        st.markdown('<p style="font-size:.78rem;color:#00C897;font-weight:600;text-transform:uppercase;margin-bottom:.8rem">Indicadores Academicos</p>', unsafe_allow_html=True)
        INDE=st.slider("INDE",3.0,9.5,7.4,.1); IDA=st.slider("IDA",0.0,10.0,6.7,.1)
        IAN=st.selectbox("IAN",[2.5,5.0,10.0],format_func=lambda x:{2.5:"2.5 Severo",5.0:"5.0 Moderado",10.0:"10.0 Adequado"}[x],index=1)
        IPP=st.slider("IPP",2.5,10.0,7.5,.1)
        Def=st.select_slider("Defasagem",options=list(range(-5,4)),value=-1)
    with c2:
        st.markdown('<p style="font-size:.78rem;color:#7C3AED;font-weight:600;text-transform:uppercase;margin-bottom:.8rem">Indicadores Comportamentais</p>', unsafe_allow_html=True)
        IAA=st.slider("IAA",0.0,10.0,8.8,.1); IEG=st.slider("IEG",0.0,10.0,8.6,.1)
        IPS=st.slider("IPS",2.5,10.0,7.5,.1); IPV=st.slider("IPV",2.5,10.0,7.6,.1)
    with c3:
        st.markdown('<p style="font-size:.78rem;color:#F59E0B;font-weight:600;text-transform:uppercase;margin-bottom:.8rem">Perfil do Aluno</p>', unsafe_allow_html=True)
        Fase=st.selectbox("Fase",list(range(0,9)),format_func=lambda x:"ALFA" if x==0 else f"Fase {x}",index=2)
        AnoIng=st.number_input("Ano ingresso PM",2016,2024,2021)
        AnoRef=st.selectbox("Ano referencia",[2022,2023,2024],index=1)
        Idade=st.number_input("Idade",7,27,12)

    Anos_p=AnoRef-AnoIng; MC=np.mean([IAA,IEG,IPS]); MA=np.mean([IDA,IPV,IPP]); Gap=IAA-IDA
    entrada=pd.DataFrame([{"INDE":INDE,"IAA":IAA,"IEG":IEG,"IPS":IPS,"IDA":IDA,"IPV":IPV,"IAN":IAN,"IPP":IPP,
        "Defasagem":Def,"Fase_Num":Fase,"Anos_no_Programa":Anos_p,"Idade":Idade,
        "Media_Comportamental":MC,"Media_Academica":MA,"Gap_Auto_Real":Gap}])[FEATURES]
    st.divider()
    if st.button("Calcular Probabilidade de Risco", type="primary", use_container_width=True):
        prob=modelo.predict_proba(entrada)[0][1]; pct=prob*100
        cg,cr = st.columns([1,1])
        with cg:
            fig, ax = plt.subplots(figsize=(5,3), facecolor=FUNDO)
            ax.set_facecolor(FUNDO); ax.axis("off")
            for cor,(t0,t1) in zip([ACCENT,"#F59E0B","#EF4444"],[(0.67,1.0),(0.33,0.67),(0.0,0.33)]):
                t = np.linspace(np.pi*t0, np.pi*t1, 100)
                ax.plot(np.cos(t), np.sin(t), color=cor, linewidth=22, solid_capstyle="butt", alpha=0.85)
            ang = np.pi*(1-prob)
            ax.annotate("", xy=(0.58*np.cos(ang),0.58*np.sin(ang)), xytext=(0,0),
                arrowprops=dict(arrowstyle="->",color="#F1F5F9",lw=3))
            ax.add_patch(plt.Circle((0,0),0.06,color="#F1F5F9",zorder=5))
            cor_p = "#EF4444" if prob>=.5 else "#F59E0B" if prob>=.3 else ACCENT
            ax.text(0,-0.2,f"{pct:.1f}%",ha="center",va="center",fontsize=28,fontweight="bold",color=cor_p,fontfamily="monospace")
            ax.text(0,-0.42,"probabilidade de risco",ha="center",fontsize=9,color=TEXTO)
            patches=[mpatches.Patch(color=ACCENT,label="Baixo <30%"),mpatches.Patch(color="#F59E0B",label="Moderado 30-50%"),mpatches.Patch(color="#EF4444",label="Alto >50%")]
            ax.legend(handles=patches,loc="lower center",ncol=3,fontsize=7,frameon=False,labelcolor=TEXTO)
            ax.set_xlim(-1.15,1.15); ax.set_ylim(-0.55,1.15)
            show(fig)
        with cr:
            if prob>=.5:
                st.markdown(f'<div class="risk-card ra"><div class="rt">Risco Alto — {pct:.1f}%</div><div class="rb2">Alta probabilidade de queda no proximo ciclo.</div><div class="rr"><strong style="color:#F87171">Acao:</strong> Acionar acompanhamento psicopedagogico imediato.</div></div>',unsafe_allow_html=True)
            elif prob>=.3:
                st.markdown(f'<div class="risk-card rm"><div class="rt">Risco Moderado — {pct:.1f}%</div><div class="rb2">Sinais de alerta. Monitoramento recomendado.</div><div class="rr"><strong style="color:#FBB03B">Acao:</strong> Monitorar IEG e IDA no proximo ciclo.</div></div>',unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-card rb"><div class="rt">Baixo Risco — {pct:.1f}%</div><div class="rb2">Indicadores estaveis.</div><div class="rr"><strong style="color:#00C897">Acao:</strong> Manter acompanhamento padrao.</div></div>',unsafe_allow_html=True)
            st.markdown("<br>",unsafe_allow_html=True)
            rs=pd.DataFrame({"Indicador":["INDE","IDA","IEG","IAA","IPS","IPV","IAN","IPP"],"Valor":[f"{v:.1f}" for v in [INDE,IDA,IEG,IAA,IPS,IPV,IAN,IPP]]})
            st.dataframe(rs,hide_index=True,use_container_width=True)
