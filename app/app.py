"""
Datathon Passos Magicos - Dashboard Analitico + Predicao de Risco
Pos-graduacao Data Analytics - FIAP Postech - Fase 5
"""
import os, joblib
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

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
    # CSV está em /data/ na raiz do repositório, um nível acima de /app/
    csv_path = os.path.join(BASE_DIR, "..", "data", "painel_pede_tratado.csv")
    if not os.path.exists(csv_path):
        # fallback: mesma pasta app/data/
        csv_path = os.path.join(BASE_DIR, "data", "painel_pede_tratado.csv")
    df = pd.read_csv(csv_path)
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

PL = dict(paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",
          font=dict(family="Inter,sans-serif",color="#94A3B8",size=12),
          title_font=dict(color="#F1F5F9",size=14),
          legend=dict(bgcolor="rgba(0,0,0,0)",font=dict(color="#94A3B8")),
          xaxis=dict(gridcolor="#1E293B",linecolor="#1E293B",tickfont=dict(color="#64748B")),
          yaxis=dict(gridcolor="#1E293B",linecolor="#1E293B",tickfont=dict(color="#64748B")),
          margin=dict(l=10,r=10,t=40,b=10))

def pl(fig,**kw):
    fig.update_layout(**{**PL,**kw}); return fig

def card(title, desc=""):
    st.markdown(f'<div class="chart-card"><div style="font-size:.88rem;font-weight:600;color:#CBD5E1;margin-bottom:.2rem">{title}</div><div style="font-size:.76rem;color:#475569;margin-bottom:.6rem">{desc}</div></div>',unsafe_allow_html=True)

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
[data-testid="stTabs"] button{font-size:.85rem!important;font-weight:500!important;color:#64748B!important;border-radius:8px 8px 0 0!important}
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
</style>""",unsafe_allow_html=True)

# SIDEBAR
with st.sidebar:
    st.markdown('<div style="font-size:1.3rem;font-weight:800;color:#F1F5F9">Passos <span style="color:#00C897">Magicos</span></div>',unsafe_allow_html=True)
    st.markdown('<div style="font-family:JetBrains Mono,monospace;font-size:.65rem;color:#334155;margin-bottom:2rem">// DATATHON FIAP POSTECH FASE 5</div>',unsafe_allow_html=True)
    pagina = st.radio("nav",["Home","Analise","Modelo","Avaliar Aluno"],label_visibility="collapsed")
    st.divider()
    st.markdown('<p style="font-size:.72rem;color:#334155;text-transform:uppercase">Base de dados</p>',unsafe_allow_html=True)
    st.markdown('<p style="font-size:.82rem;color:#475569">PEDE 2022 2023 2024</p>',unsafe_allow_html=True)
    st.markdown('<p style="font-size:.72rem;color:#334155;margin-top:.8rem;text-transform:uppercase">Modelo</p>',unsafe_allow_html=True)
    st.markdown('<p style="font-size:.82rem;color:#475569">XGBoost AUC 0.795</p>',unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:.72rem;color:#334155;margin-top:.8rem;text-transform:uppercase">Registros</p>',unsafe_allow_html=True)
    st.markdown(f'<p style="font-size:.82rem;color:#475569">{len(df_full):,} alunos-ano</p>',unsafe_allow_html=True)

# ═══ PAGINA 1: HOME ═══════════════════════════════════════════════════════════
if pagina == "Home":
    st.markdown("""<div class="hero">
    <div class="hero-tag">// Datathon Analise Educacional</div>
    <h1 class="hero-title">Transformando dados em <span>impacto real</span></h1>
    <p class="hero-sub">Dashboard analitico da Pesquisa Extensiva do Desenvolvimento Educacional (PEDE) 2022-2024.</p>
    </div>""",unsafe_allow_html=True)

    total = df_full["RA"].nunique()
    inde_m= df_full["INDE"].mean()
    pct_t = (df_full[df_full["Ano_Referencia"]==2024]["Pedra"]=="Topazio").mean()*100
    pct_s = (df_full["IAN"]==2.5).mean()*100
    a22   = df_full[df_full["Ano_Referencia"]==2022]["RA"].nunique()
    a24   = df_full[df_full["Ano_Referencia"]==2024]["RA"].nunique()
    cresc = (a24-a22)/a22*100

    st.markdown(f"""<div class="kpi-row">
    <div class="kpi-card g"><div class="kpi-label">Alunos unicos</div><div class="kpi-value g">{total:,}</div><div class="kpi-delta">2022-2024</div></div>
    <div class="kpi-card p"><div class="kpi-label">Crescimento</div><div class="kpi-value p">+{cresc:.0f}%</div><div class="kpi-delta">{a22} para {a24}</div></div>
    <div class="kpi-card b"><div class="kpi-label">INDE medio</div><div class="kpi-value">{inde_m:.2f}</div><div class="kpi-delta">escala 0-10</div></div>
    <div class="kpi-card g"><div class="kpi-label">Topazio 2024</div><div class="kpi-value g">{pct_t:.0f}%</div><div class="kpi-delta">vs 15% em 2022</div></div>
    <div class="kpi-card r"><div class="kpi-label">IAN Severo</div><div class="kpi-value r">{pct_s:.1f}%</div><div class="kpi-delta">defasagem critica</div></div>
    </div>""",unsafe_allow_html=True)

    st.divider()
    col1,col2 = st.columns(2)
    with col1:
        anos = df_full.groupby("Ano_Referencia")["RA"].nunique().reset_index()
        anos.columns=["Ano","Alunos"]
        fig=px.bar(anos,x="Ano",y="Alunos",text="Alunos",color="Alunos",
                   color_continuous_scale=[[0,"#1E3A5F"],[1,"#00C897"]],title="Crescimento da base por ano")
        fig.update_traces(textposition="outside")
        fig.update_coloraxes(showscale=False)
        pl(fig,showlegend=False)
        st.markdown('<div class="chart-card">',unsafe_allow_html=True)
        st.plotly_chart(fig,use_container_width=True)
        st.markdown("</div>",unsafe_allow_html=True)

    with col2:
        df_p=df_full[df_full["Pedra"].isin(ORDEM_PEDRA)]
        pp=df_p.groupby(["Ano_Referencia","Pedra"]).size().reset_index(name="n")
        tot=pp.groupby("Ano_Referencia")["n"].transform("sum")
        pp["pct"]=(pp["n"]/tot*100).round(1)
        pp["Pedra"]=pd.Categorical(pp["Pedra"],ORDEM_PEDRA); pp=pp.sort_values("Pedra")
        fig2=px.bar(pp,x="Ano_Referencia",y="pct",color="Pedra",barmode="stack",text="pct",
                    color_discrete_map=CORES_PEDRA,labels={"pct":"%","Ano_Referencia":"Ano"},
                    title="Distribuicao de Pedra por ano (%)")
        fig2.update_traces(texttemplate="%{text:.0f}%",textposition="inside")
        pl(fig2)
        st.markdown('<div class="chart-card">',unsafe_allow_html=True)
        st.plotly_chart(fig2,use_container_width=True)
        st.markdown("</div>",unsafe_allow_html=True)

    medias=df_full.groupby("Ano_Referencia")[["INDE","IDA","IEG"]].mean().reset_index()
    fig3=go.Figure()
    for col,cor,nm in [("INDE","#00C897","INDE"),("IDA","#7C3AED","IDA"),("IEG","#F59E0B","IEG")]:
        fig3.add_trace(go.Scatter(x=medias["Ano_Referencia"],y=medias[col],mode="lines+markers+text",
            name=nm,line=dict(color=cor,width=3),marker=dict(size=9),
            text=medias[col].round(2),textposition="top center",textfont=dict(color=cor,size=11)))
    pl(fig3,title="INDE, IDA e IEG medio por ano",yaxis=dict(range=[6,9],gridcolor="#1E293B",linecolor="#1E293B",tickfont=dict(color="#64748B")))
    st.markdown('<div class="chart-card">',unsafe_allow_html=True)
    st.plotly_chart(fig3,use_container_width=True)
    st.markdown("</div>",unsafe_allow_html=True)

    col1,col2,col3=st.columns(3)
    insights=[("📈","#00C897","Programa em crescimento","Base cresceu 34% entre 2022 e 2024 com Topazio saindo de 15% para 28%."),
              ("🔗","#7C3AED","Engajamento como driver","IEG e IDA sao os maiores preditores do INDE (r=0.75 e r=0.79)."),
              ("⚠️","#F59E0B","Autopercepção superestimada","IAA tem correlacao fraca com IDA (r=0.12) — alunos se avaliam bem independente do desempenho.")]
    for col,(ic,cor,tit,txt) in zip([col1,col2,col3],insights):
        with col:
            st.markdown(f'<div class="chart-card" style="border-left:3px solid {cor};padding:1.2rem 1.4rem"><div style="font-size:1.5rem;margin-bottom:.5rem">{ic}</div><div style="font-size:.9rem;font-weight:600;color:#CBD5E1;margin-bottom:.4rem">{tit}</div><div style="font-size:.8rem;color:#475569;line-height:1.6">{txt}</div></div>',unsafe_allow_html=True)

# ═══ PAGINA 2: ANALISE ════════════════════════════════════════════════════════
elif pagina == "Analise":
    st.markdown('<div class="sl">// Pesquisa PEDE 2022-2024</div>',unsafe_allow_html=True)
    st.markdown('<h1 class="st2">Analise por Indicador</h1>',unsafe_allow_html=True)

    st.markdown('<div class="fbar">',unsafe_allow_html=True)
    fc1,fc2,fc3,fc4=st.columns(4)
    with fc1: anos_sel=st.multiselect("Ano",[2022,2023,2024],default=[2022,2023,2024])
    with fc2:
        fases=["Todos"]+["ALFA" if x==0 else f"Fase {int(x)}" for x in sorted(df_full["Fase_Num"].dropna().unique())]
        fase_sel=st.selectbox("Fase",fases)
    with fc3: gen_sel=st.multiselect("Genero",["Feminino","Masculino"],default=["Feminino","Masculino"])
    with fc4:
        ped_disp=[p for p in ORDEM_PEDRA if p in df_full["Pedra"].values]
        ped_sel=st.multiselect("Pedra",ped_disp,default=ped_disp)
    st.markdown("</div>",unsafe_allow_html=True)

    df=df_full.copy()
    if anos_sel: df=df[df["Ano_Referencia"].isin(anos_sel)]
    if fase_sel!="Todos":
        fn=0 if fase_sel=="ALFA" else int(fase_sel.split()[-1])
        df=df[df["Fase_Num"]==fn]
    if gen_sel: df=df[df["Genero"].isin(gen_sel)]
    if ped_sel: df=df[df["Pedra"].isin(ped_sel)|df["Pedra"].isna()]
    if df.empty: st.warning("Nenhum dado com os filtros selecionados."); st.stop()

    tabs=st.tabs(["IAN","IDA","IEG","IAA","IPS","IPP","IPV","INDE","Efetividade","Insights"])

    with tabs[0]:
        st.markdown('<div class="sl">// Pergunta 1</div><h3 class="st2">Adequacao ao Nivel (IAN)</h3><p class="sd">IAN: 2.5 Severo, 5.0 Moderado, 10.0 Adequado. Mede se o aluno esta no nivel compativel com sua faixa etaria.</p>',unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            ic=df["IAN_Cat"].value_counts().reindex(["Severo","Moderado","Adequado"]).fillna(0).reset_index()
            ic.columns=["Cat","Qtd"]
            fig=px.pie(ic,names="Cat",values="Qtd",color="Cat",color_discrete_map=CORES_IAN,hole=.55,title="Distribuicao geral do IAN")
            fig.update_traces(textinfo="percent+label")
            pl(fig)
            st.markdown('<div class="chart-card">',unsafe_allow_html=True); st.plotly_chart(fig,use_container_width=True); st.markdown("</div>",unsafe_allow_html=True)
        with c2:
            ia=df.groupby(["Ano_Referencia","IAN_Cat"]).size().reset_index(name="n")
            ia["pct"]=(ia["n"]/ia.groupby("Ano_Referencia")["n"].transform("sum")*100).round(1)
            ia["IAN_Cat"]=pd.Categorical(ia["IAN_Cat"],["Severo","Moderado","Adequado"]); ia=ia.sort_values("IAN_Cat")
            fig2=px.bar(ia,x="Ano_Referencia",y="pct",color="IAN_Cat",barmode="stack",text="pct",
                        color_discrete_map=CORES_IAN,labels={"pct":"%","Ano_Referencia":"Ano","IAN_Cat":"IAN"},title="IAN por ano (%)")
            fig2.update_traces(texttemplate="%{text:.0f}%",textposition="inside"); pl(fig2)
            st.markdown('<div class="chart-card">',unsafe_allow_html=True); st.plotly_chart(fig2,use_container_width=True); st.markdown("</div>",unsafe_allow_html=True)
        c1,c2,c3=st.columns(3)
        c1.metric("Severo",f"{(df['IAN']==2.5).mean()*100:.1f}%",f"{int((df['IAN']==2.5).sum())} alunos")
        c2.metric("Moderado",f"{(df['IAN']==5.0).mean()*100:.1f}%",f"{int((df['IAN']==5.0).sum())} alunos")
        c3.metric("Adequado",f"{(df['IAN']==10.0).mean()*100:.1f}%",f"{int((df['IAN']==10.0).sum())} alunos")

    with tabs[1]:
        st.markdown('<div class="sl">// Pergunta 2</div><h3 class="st2">Desempenho Academico (IDA)</h3><p class="sd">Media das notas do indicador de aprendizagem por ano e por fase.</p>',unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            ida=df.groupby("Ano_Referencia")["IDA"].mean().reset_index()
            fig=go.Figure(go.Scatter(x=ida["Ano_Referencia"],y=ida["IDA"].round(2),mode="lines+markers+text",
                line=dict(color="#7C3AED",width=3),marker=dict(size=10),
                text=ida["IDA"].round(2),textposition="top center",textfont=dict(color="#A78BFA",size=13)))
            pl(fig,title="IDA medio por ano",xaxis=dict(tickvals=[2022,2023,2024],ticktext=["2022","2023","2024"],gridcolor="#1E293B",linecolor="#1E293B",tickfont=dict(color="#64748B")),yaxis=dict(range=[0,10],gridcolor="#1E293B",linecolor="#1E293B",tickfont=dict(color="#64748B")))
            st.markdown('<div class="chart-card">',unsafe_allow_html=True); st.plotly_chart(fig,use_container_width=True); st.markdown("</div>",unsafe_allow_html=True)
        with c2:
            df_f=df[df["Fase_Num"].between(0,8)&df["IDA"].notna()]
            if_ =df_f.groupby(["Fase_Num","Ano_Referencia"])["IDA"].mean().reset_index()
            if_["FL"]=if_["Fase_Num"].apply(lambda x:"ALFA" if x==0 else f"F{int(x)}")
            fig2=px.line(if_,x="FL",y="IDA",color="Ano_Referencia",markers=True,
                         color_discrete_map=CORES_ANO,labels={"IDA":"IDA medio","FL":"Fase","Ano_Referencia":"Ano"},title="IDA por Fase e Ano")
            fig2.update_traces(line=dict(width=2.5),marker=dict(size=7)); pl(fig2,yaxis=dict(range=[0,10],gridcolor="#1E293B",linecolor="#1E293B",tickfont=dict(color="#64748B")))
            st.markdown('<div class="chart-card">',unsafe_allow_html=True); st.plotly_chart(fig2,use_container_width=True); st.markdown("</div>",unsafe_allow_html=True)
        df_bx=df[df["Pedra"].isin(ORDEM_PEDRA)&df["IDA"].notna()]
        fig3=px.box(df_bx,x="Pedra",y="IDA",color="Pedra",category_orders={"Pedra":ORDEM_PEDRA},color_discrete_map=CORES_PEDRA,title="IDA por Pedra")
        pl(fig3,showlegend=False)
        st.markdown('<div class="chart-card">',unsafe_allow_html=True); st.plotly_chart(fig3,use_container_width=True); st.markdown("</div>",unsafe_allow_html=True)

    with tabs[2]:
        st.markdown('<div class="sl">// Pergunta 3</div><h3 class="st2">Engajamento (IEG)</h3><p class="sd">Grau de participacao do aluno. Principal preditor do INDE (r=0.75).</p>',unsafe_allow_html=True)
        dsc=df[["IEG","IDA","IPV","Ano_Referencia"]].dropna()
        c1,c2=st.columns(2)
        with c1:
            r=dsc["IEG"].corr(dsc["IDA"])
            fig=px.scatter(dsc,x="IEG",y="IDA",color="Ano_Referencia",color_discrete_map=CORES_ANO,opacity=.4,
                           trendline="ols",trendline_scope="overall",trendline_color_override="#F1F5F9",
                           title=f"IEG vs IDA (r={r:.2f})",labels={"Ano_Referencia":"Ano"})
            pl(fig); st.markdown('<div class="chart-card">',unsafe_allow_html=True); st.plotly_chart(fig,use_container_width=True); st.markdown("</div>",unsafe_allow_html=True)
        with c2:
            r2=dsc["IEG"].corr(dsc["IPV"])
            fig2=px.scatter(dsc,x="IEG",y="IPV",color="Ano_Referencia",color_discrete_map=CORES_ANO,opacity=.4,
                            trendline="ols",trendline_scope="overall",trendline_color_override="#F1F5F9",
                            title=f"IEG vs IPV (r={r2:.2f})",labels={"Ano_Referencia":"Ano"})
            pl(fig2); st.markdown('<div class="chart-card">',unsafe_allow_html=True); st.plotly_chart(fig2,use_container_width=True); st.markdown("</div>",unsafe_allow_html=True)
        ieg_f=df[df["Fase_Num"].between(0,8)].groupby("Fase_Num")["IEG"].mean().reset_index()
        ieg_f["FL"]=ieg_f["Fase_Num"].apply(lambda x:"ALFA" if x==0 else f"Fase {int(x)}")
        fig3=px.bar(ieg_f,x="FL",y="IEG",text=ieg_f["IEG"].round(2),title="IEG medio por Fase",color="IEG",color_continuous_scale=[[0,"#1E3A5F"],[1,"#00C897"]])
        fig3.update_traces(textposition="outside"); fig3.update_coloraxes(showscale=False)
        pl(fig3,showlegend=False,yaxis=dict(range=[0,10],gridcolor="#1E293B",linecolor="#1E293B",tickfont=dict(color="#64748B")))
        st.markdown('<div class="chart-card">',unsafe_allow_html=True); st.plotly_chart(fig3,use_container_width=True); st.markdown("</div>",unsafe_allow_html=True)

    with tabs[3]:
        st.markdown('<div class="sl">// Pergunta 4</div><h3 class="st2">Autoavaliacao (IAA)</h3><p class="sd">Percepcao do aluno sobre si mesmo vs desempenho real.</p>',unsafe_allow_html=True)
        di=df[["IAA","IDA","IEG","Ano_Referencia"]].dropna()
        c1,c2=st.columns(2)
        with c1:
            r=di["IAA"].corr(di["IDA"])
            fig=px.scatter(di,x="IAA",y="IDA",color="Ano_Referencia",color_discrete_map=CORES_ANO,opacity=.4,
                           trendline="ols",trendline_scope="overall",trendline_color_override="#F1F5F9",
                           title=f"IAA vs IDA (r={r:.2f}) - Correlacao fraca",labels={"Ano_Referencia":"Ano"})
            pl(fig); st.markdown('<div class="chart-card">',unsafe_allow_html=True); st.plotly_chart(fig,use_container_width=True); st.markdown("</div>",unsafe_allow_html=True)
        with c2:
            dg=df[["IAA","IDA"]].dropna().copy(); dg["Gap"]=dg["IAA"]-dg["IDA"]
            fig2=px.histogram(dg,x="Gap",nbins=30,title="Gap IAA - IDA (autopercep. vs realidade)",color_discrete_sequence=["#7C3AED"])
            fig2.add_vline(x=0,line_dash="dash",line_color="#F1F5F9",annotation_text="Sem gap",annotation_font_color="#94A3B8")
            fig2.add_vline(x=dg["Gap"].mean(),line_dash="dot",line_color="#F59E0B",
                           annotation_text=f"Media:{dg['Gap'].mean():.1f}",annotation_font_color="#F59E0B")
            pl(fig2,showlegend=False); st.markdown('<div class="chart-card">',unsafe_allow_html=True); st.plotly_chart(fig2,use_container_width=True); st.markdown("</div>",unsafe_allow_html=True)

    with tabs[4]:
        st.markdown('<div class="sl">// Pergunta 5</div><h3 class="st2">Aspectos Psicossociais (IPS)</h3><p class="sd">Padroes de IPS que antecedem quedas de desempenho.</p>',unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            ip=df.groupby("Ano_Referencia")["IPS"].mean().reset_index()
            fig=go.Figure(go.Scatter(x=ip["Ano_Referencia"],y=ip["IPS"].round(2),mode="lines+markers+text",
                line=dict(color="#EF4444",width=3),marker=dict(size=10),
                text=ip["IPS"].round(2),textposition="top center",textfont=dict(color="#F87171",size=13)))
            pl(fig,title="IPS medio por ano",yaxis=dict(range=[0,10],gridcolor="#1E293B",linecolor="#1E293B",tickfont=dict(color="#64748B")))
            st.markdown('<div class="chart-card">',unsafe_allow_html=True); st.plotly_chart(fig,use_container_width=True); st.markdown("</div>",unsafe_allow_html=True)
        with c2:
            ds=df[["IPS","IDA","Ano_Referencia"]].dropna(); r=ds["IPS"].corr(ds["IDA"])
            fig2=px.scatter(ds,x="IPS",y="IDA",color="Ano_Referencia",color_discrete_map=CORES_ANO,opacity=.4,
                            trendline="ols",trendline_scope="overall",trendline_color_override="#F1F5F9",
                            title=f"IPS vs IDA (r={r:.2f})",labels={"Ano_Referencia":"Ano"})
            pl(fig2); st.markdown('<div class="chart-card">',unsafe_allow_html=True); st.plotly_chart(fig2,use_container_width=True); st.markdown("</div>",unsafe_allow_html=True)

    with tabs[5]:
        st.markdown('<div class="sl">// Pergunta 6</div><h3 class="st2">Aspectos Psicopedagogicos (IPP)</h3><p class="sd">Disponivel a partir de 2023. Confirma ou contradiz o IAN?</p>',unsafe_allow_html=True)
        di=df[df["IPP"].notna()&df["IAN_Cat"].notna()]
        c1,c2=st.columns(2)
        with c1:
            fig=px.box(di,x="IAN_Cat",y="IPP",color="IAN_Cat",category_orders={"IAN_Cat":["Severo","Moderado","Adequado"]},
                       color_discrete_map=CORES_IAN,title="IPP por categoria IAN")
            pl(fig,showlegend=False); st.markdown('<div class="chart-card">',unsafe_allow_html=True); st.plotly_chart(fig,use_container_width=True); st.markdown("</div>",unsafe_allow_html=True)
        with c2:
            r=di["IAN"].corr(di["IPP"])
            fig2=px.scatter(di,x="IAN",y="IPP",color="Ano_Referencia",color_discrete_map=CORES_ANO,opacity=.5,
                            trendline="ols",trendline_scope="overall",trendline_color_override="#F1F5F9",
                            title=f"IPP vs IAN (r={r:.2f})",labels={"Ano_Referencia":"Ano"})
            pl(fig2); st.markdown('<div class="chart-card">',unsafe_allow_html=True); st.plotly_chart(fig2,use_container_width=True); st.markdown("</div>",unsafe_allow_html=True)

    with tabs[6]:
        st.markdown('<div class="sl">// Pergunta 7</div><h3 class="st2">Ponto de Virada (IPV)</h3><p class="sd">Quais indicadores mais influenciam o IPV?</p>',unsafe_allow_html=True)
        inds=["IDA","IEG","IAA","IPS","IAN","IPP"]
        dv=df[["IPV"]+inds].dropna()
        corrs={c:dv["IPV"].corr(dv[c]) for c in inds}
        dc=pd.DataFrame({"Ind":list(corrs.keys()),"r":list(corrs.values())}).sort_values("r")
        dc["Cor"]=dc["r"].apply(lambda x:"#00C897" if x>=.5 else "#F59E0B" if x>=.3 else "#94A3B8")
        c1,c2=st.columns(2)
        with c1:
            fig=go.Figure(go.Bar(x=dc["r"],y=dc["Ind"],orientation="h",marker_color=dc["Cor"].tolist(),
                text=dc["r"].round(2),textposition="outside",textfont=dict(color="#F1F5F9")))
            pl(fig,title="Correlacao com IPV",xaxis=dict(range=[-0.1,.9],gridcolor="#1E293B",linecolor="#1E293B",tickfont=dict(color="#64748B")))
            st.markdown('<div class="chart-card">',unsafe_allow_html=True); st.plotly_chart(fig,use_container_width=True); st.markdown("</div>",unsafe_allow_html=True)
        with c2:
            d22=df[df["Ano_Referencia"]==2022].copy()
            d22["PVL"]=d22["Atingiu_Ponto_Virada"].map({"Sim":"Atingiu PV","Nao":"Nao atingiu"})
            d22v=d22[d22["PVL"].notna()]
            if not d22v.empty:
                mp=d22v.groupby("PVL")[["IDA","IEG","IAA"]].mean().reset_index()
                mm=mp.melt(id_vars="PVL",var_name="Ind",value_name="Media")
                fig2=px.bar(mm,x="Ind",y="Media",color="PVL",barmode="group",
                            color_discrete_map={"Atingiu PV":"#00C897","Nao atingiu":"#EF4444"},
                            title="IDA IEG IAA - Atingiu vs Nao atingiu PV (2022)")
                fig2.update_traces(texttemplate="%{y:.2f}",textposition="outside")
                pl(fig2,yaxis=dict(range=[0,10],gridcolor="#1E293B",linecolor="#1E293B",tickfont=dict(color="#64748B")))
                st.markdown('<div class="chart-card">',unsafe_allow_html=True); st.plotly_chart(fig2,use_container_width=True); st.markdown("</div>",unsafe_allow_html=True)
            else:
                st.info("Dado de Ponto de Virada disponivel apenas em 2022.")

    with tabs[7]:
        st.markdown('<div class="sl">// Pergunta 8</div><h3 class="st2">Multidimensionalidade — INDE</h3><p class="sd">Quais combinacoes de indicadores elevam mais o INDE?</p>',unsafe_allow_html=True)
        di=df[["INDE","IAA","IEG","IPS","IDA","IPV","IAN","IPP"]].dropna()
        ci={c:di["INDE"].corr(di[c]) for c in ["IAA","IEG","IPS","IDA","IPV","IAN","IPP"]}
        dc=pd.DataFrame({"Ind":list(ci.keys()),"r":list(ci.values())}).sort_values("r")
        dc["Cor"]=dc["r"].apply(lambda x:"#00C897" if x>=.6 else "#F59E0B" if x>=.4 else "#94A3B8")
        c1,c2=st.columns(2)
        with c1:
            fig=go.Figure(go.Bar(x=dc["r"],y=dc["Ind"],orientation="h",marker_color=dc["Cor"].tolist(),
                text=dc["r"].round(2),textposition="outside",textfont=dict(color="#F1F5F9")))
            pl(fig,title="Correlacao com INDE",xaxis=dict(range=[0,1],gridcolor="#1E293B",linecolor="#1E293B",tickfont=dict(color="#64748B")))
            st.markdown('<div class="chart-card">',unsafe_allow_html=True); st.plotly_chart(fig,use_container_width=True); st.markdown("</div>",unsafe_allow_html=True)
        with c2:
            dr=df[df["Pedra"].isin(ORDEM_PEDRA)][["Pedra","IAA","IEG","IPS","IDA","IPV","IAN"]].dropna()
            mp=dr.groupby("Pedra").mean(); cats=["IAA","IEG","IPS","IDA","IPV","IAN"]
            fig2=go.Figure()
            for pedra,cor in CORES_PEDRA.items():
                if pedra in mp.index:
                    vals=[mp.loc[pedra,c] for c in cats]+[mp.loc[pedra,cats[0]]]
                    fig2.add_trace(go.Scatterpolar(r=vals,theta=cats+[cats[0]],name=pedra,
                        line=dict(color=cor,width=2),fill="toself"))
            pl(fig2,title="Perfil por Pedra",polar=dict(bgcolor="rgba(0,0,0,0)",
                radialaxis=dict(range=[0,10],gridcolor="#1E293B",color="#64748B"),
                angularaxis=dict(gridcolor="#1E293B",color="#94A3B8")))
            st.markdown('<div class="chart-card">',unsafe_allow_html=True); st.plotly_chart(fig2,use_container_width=True); st.markdown("</div>",unsafe_allow_html=True)

    with tabs[8]:
        st.markdown('<div class="sl">// Pergunta 10</div><h3 class="st2">Efetividade do Programa</h3><p class="sd">Melhora consistente ao longo dos anos?</p>',unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            if_=df[df["Fase_Num"].between(0,8)].groupby(["Fase_Num","Ano_Referencia"])["INDE"].mean().reset_index()
            if_["FL"]=if_["Fase_Num"].apply(lambda x:"ALFA" if x==0 else f"F{int(x)}")
            fig=px.line(if_,x="FL",y="INDE",color="Ano_Referencia",markers=True,
                        color_discrete_map=CORES_ANO,title="INDE medio por Fase e Ano",labels={"Ano_Referencia":"Ano","FL":"Fase"})
            fig.update_traces(line=dict(width=2.5),marker=dict(size=7))
            pl(fig,yaxis=dict(range=[4,10],gridcolor="#1E293B",linecolor="#1E293B",tickfont=dict(color="#64748B")))
            st.markdown('<div class="chart-card">',unsafe_allow_html=True); st.plotly_chart(fig,use_container_width=True); st.markdown("</div>",unsafe_allow_html=True)
        with c2:
            pn=df[df["Pedra"].isin(ORDEM_PEDRA)].groupby(["Ano_Referencia","Pedra"]).size().reset_index(name="n")
            pn["Pedra"]=pd.Categorical(pn["Pedra"],ORDEM_PEDRA); pn=pn.sort_values("Pedra")
            fig2=px.bar(pn,x="Ano_Referencia",y="n",color="Pedra",barmode="stack",
                        color_discrete_map=CORES_PEDRA,title="Volume por Pedra e Ano",labels={"n":"Alunos","Ano_Referencia":"Ano"})
            pl(fig2); st.markdown('<div class="chart-card">',unsafe_allow_html=True); st.plotly_chart(fig2,use_container_width=True); st.markdown("</div>",unsafe_allow_html=True)
        dn=df.groupby("Ano_Referencia").apply(lambda x:(x["Defasagem"]<0).mean()*100).reset_index()
        dn.columns=["Ano","Pct"]
        fig3=go.Figure(go.Scatter(x=dn["Ano"],y=dn["Pct"].round(1),mode="lines+markers+text",
            line=dict(color="#EF4444",width=3),marker=dict(size=10),
            text=dn["Pct"].round(1),textposition="top center",textfont=dict(color="#F87171",size=13),texttemplate="%{text:.1f}%"))
        pl(fig3,title="% alunos com defasagem negativa",yaxis=dict(range=[0,100],ticksuffix="%",gridcolor="#1E293B",linecolor="#1E293B",tickfont=dict(color="#64748B")))
        st.markdown('<div class="chart-card">',unsafe_allow_html=True); st.plotly_chart(fig3,use_container_width=True); st.markdown("</div>",unsafe_allow_html=True)

    with tabs[9]:
        st.markdown('<div class="sl">// Pergunta 11</div><h3 class="st2">Insights Adicionais</h3>',unsafe_allow_html=True)
        c1,c2=st.columns(2)
        with c1:
            dg=df[df["Genero"].isin(["Feminino","Masculino"])&df["INDE"].notna()]
            ig=dg.groupby(["Ano_Referencia","Genero"])["INDE"].mean().reset_index()
            fig=px.line(ig,x="Ano_Referencia",y="INDE",color="Genero",markers=True,
                        title="INDE medio por Genero e Ano",color_discrete_map={"Feminino":"#F59E0B","Masculino":"#3B82F6"})
            fig.update_traces(line=dict(width=2.5),marker=dict(size=9),
                text=ig["INDE"].round(2),textposition="top center",mode="lines+markers+text",textfont=dict(size=10))
            pl(fig,yaxis=dict(range=[5,10],gridcolor="#1E293B",linecolor="#1E293B",tickfont=dict(color="#64748B")))
            st.markdown('<div class="chart-card">',unsafe_allow_html=True); st.plotly_chart(fig,use_container_width=True); st.markdown("</div>",unsafe_allow_html=True)
        with c2:
            di=df[df["Idade"].between(7,25)]
            fig2=px.histogram(di,x="Idade",color="Ano_Referencia",barmode="overlay",opacity=.7,nbins=20,
                              title="Distribuicao de idades por ano",color_discrete_map=CORES_ANO,labels={"Ano_Referencia":"Ano"})
            pl(fig2); st.markdown('<div class="chart-card">',unsafe_allow_html=True); st.plotly_chart(fig2,use_container_width=True); st.markdown("</div>",unsafe_allow_html=True)
        cols_c=["INDE","IAA","IEG","IPS","IDA","IPV","IAN","IPP"]
        cm=df[cols_c].dropna().corr().round(2)
        fig3=px.imshow(cm,text_auto=True,color_continuous_scale=[[0,"#EF4444"],[.5,"#1E293B"],[1,"#00C897"]],
                       zmin=-1,zmax=1,title="Mapa de correlacoes entre indicadores")
        pl(fig3); fig3.update_xaxes(tickfont=dict(color="#94A3B8")); fig3.update_yaxes(tickfont=dict(color="#94A3B8"))
        st.markdown('<div class="chart-card">',unsafe_allow_html=True); st.plotly_chart(fig3,use_container_width=True); st.markdown("</div>",unsafe_allow_html=True)

# ═══ PAGINA 3: MODELO ═════════════════════════════════════════════════════════
elif pagina == "Modelo":
    st.markdown('<div class="sl">// Pergunta 9 Machine Learning</div><h1 class="st2">Modelo de Risco de Defasagem</h1><p class="sd">XGBoost treinado para identificar alunos com risco de queda no INDE ou aumento da defasagem.</p>',unsafe_allow_html=True)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Algoritmo","XGBoost"); c2.metric("ROC-AUC","0.795","teste"); c3.metric("Acuracia","80%"); c4.metric("Features","15")
    st.divider()
    c1,c2=st.columns(2)
    with c1:
        ma=pd.DataFrame({"Modelo":["Reg Logistica","Random Forest","XGBoost"],"CV":[0.670,0.789,0.805],"Teste":[0.697,0.785,0.795]})
        fig=go.Figure()
        fig.add_trace(go.Bar(name="AUC CV",x=ma["Modelo"],y=ma["CV"],marker_color="#475569",text=ma["CV"],textposition="outside"))
        fig.add_trace(go.Bar(name="AUC Teste",x=ma["Modelo"],y=ma["Teste"],marker_color="#00C897",text=ma["Teste"],textposition="outside"))
        fig.update_traces(texttemplate="%{text:.3f}")
        pl(fig,title="ROC-AUC por modelo",barmode="group",yaxis=dict(range=[.5,.9],gridcolor="#1E293B",linecolor="#1E293B",tickfont=dict(color="#64748B")))
        st.markdown('<div class="chart-card">',unsafe_allow_html=True); st.plotly_chart(fig,use_container_width=True); st.markdown("</div>",unsafe_allow_html=True)
    with c2:
        fi=pd.DataFrame({"Feature":["INDE","IDA","IEG","Media_Academica","IPV","Media_Comportamental","Anos_no_Programa","IPP","Gap_Auto_Real","Defasagem","Fase_Num","IAN","Idade","IPS","IAA"],
                          "Imp":[.18,.15,.12,.10,.09,.08,.07,.06,.05,.04,.03,.02,.01,.005,.003]}).sort_values("Imp")
        fi["Cor"]=["#00C897" if v>=.10 else "#7C3AED" if v>=.06 else "#475569" for v in fi["Imp"]]
        fig2=go.Figure(go.Bar(x=fi["Imp"],y=fi["Feature"],orientation="h",marker_color=fi["Cor"].tolist(),
            text=fi["Imp"].round(3),textposition="outside",textfont=dict(color="#F1F5F9",size=10)))
        pl(fig2,title="Importancia das Features - XGBoost",showlegend=False,xaxis=dict(gridcolor="#1E293B",linecolor="#1E293B",tickfont=dict(color="#64748B")))
        st.markdown('<div class="chart-card">',unsafe_allow_html=True); st.plotly_chart(fig2,use_container_width=True); st.markdown("</div>",unsafe_allow_html=True)
    st.markdown('<div class="chart-card"><div style="font-size:.88rem;font-weight:600;color:#CBD5E1;margin-bottom:.5rem">Como o risco e definido</div><div style="font-size:.82rem;color:#475569;line-height:1.7"><span style="color:#00C897">▸</span> Queda de INDE superior a <strong style="color:#F1F5F9">0.3 ponto</strong> no ano seguinte<br><span style="color:#00C897">▸</span> Piora no nivel de defasagem<br><br>Isso permite usar dados do ano atual para intervir antes que a queda aconteca.</div></div>',unsafe_allow_html=True)

# ═══ PAGINA 4: AVALIAR ALUNO ══════════════════════════════════════════════════
elif pagina == "Avaliar Aluno":
    st.markdown('<div class="sl">// Simulador de risco individual</div><h1 class="st2">Avaliar Aluno</h1><p class="sd">Insira os indicadores para calcular a probabilidade de risco no proximo ciclo.</p>',unsafe_allow_html=True)
    c1,c2,c3=st.columns(3)
    with c1:
        st.markdown('<p style="font-size:.78rem;color:#00C897;font-weight:600;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.8rem">Indicadores Academicos</p>',unsafe_allow_html=True)
        INDE=st.slider("INDE",3.0,9.5,7.4,.1)
        IDA=st.slider("IDA Desempenho Academico",0.0,10.0,6.7,.1)
        IAN=st.selectbox("IAN Adequacao ao Nivel",[2.5,5.0,10.0],format_func=lambda x:{2.5:"2.5 Severo",5.0:"5.0 Moderado",10.0:"10.0 Adequado"}[x],index=1)
        IPP=st.slider("IPP Psicopedagogico",2.5,10.0,7.5,.1)
        Def=st.select_slider("Defasagem nivel vs ideal",options=list(range(-5,4)),value=-1)
    with c2:
        st.markdown('<p style="font-size:.78rem;color:#7C3AED;font-weight:600;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.8rem">Indicadores Comportamentais</p>',unsafe_allow_html=True)
        IAA=st.slider("IAA Autoavaliacao",0.0,10.0,8.8,.1)
        IEG=st.slider("IEG Engajamento",0.0,10.0,8.6,.1)
        IPS=st.slider("IPS Psicossocial",2.5,10.0,7.5,.1)
        IPV=st.slider("IPV Ponto de Virada",2.5,10.0,7.6,.1)
    with c3:
        st.markdown('<p style="font-size:.78rem;color:#F59E0B;font-weight:600;text-transform:uppercase;letter-spacing:.1em;margin-bottom:.8rem">Perfil do Aluno</p>',unsafe_allow_html=True)
        Fase=st.selectbox("Fase atual",list(range(0,9)),format_func=lambda x:"ALFA" if x==0 else f"Fase {x}",index=2)
        AnoIng=st.number_input("Ano ingresso PM",2016,2024,2021)
        AnoRef=st.selectbox("Ano referencia",[2022,2023,2024],index=1)
        Idade=st.number_input("Idade",7,27,12)

    Anos_p=AnoRef-AnoIng; MC=np.mean([IAA,IEG,IPS]); MA=np.mean([IDA,IPV,IPP]); Gap=IAA-IDA
    entrada=pd.DataFrame([{"INDE":INDE,"IAA":IAA,"IEG":IEG,"IPS":IPS,"IDA":IDA,"IPV":IPV,"IAN":IAN,"IPP":IPP,
        "Defasagem":Def,"Fase_Num":Fase,"Anos_no_Programa":Anos_p,"Idade":Idade,
        "Media_Comportamental":MC,"Media_Academica":MA,"Gap_Auto_Real":Gap}])[FEATURES]
    st.divider()
    if st.button("Calcular Probabilidade de Risco",type="primary",use_container_width=True):
        prob=modelo.predict_proba(entrada)[0][1]; pct=prob*100
        cg,cr=st.columns([1,1])
        with cg:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
            fig_g, ax = plt.subplots(figsize=(5,2.8), facecolor="#111827")
            ax.set_facecolor("#111827"); ax.axis("off")
            for cor,(t0,t1) in zip(["#00C897","#F59E0B","#EF4444"],[(0.67,1.0),(0.33,0.67),(0.0,0.33)]):
                t = __import__("numpy").linspace(__import__("numpy").pi*t0, __import__("numpy").pi*t1, 100)
                ax.plot(__import__("numpy").cos(t), __import__("numpy").sin(t), color=cor, linewidth=20, solid_capstyle="butt", alpha=0.85)
            angulo = __import__("numpy").pi*(1-prob)
            ax.annotate("",xy=(0.58*__import__("numpy").cos(angulo),0.58*__import__("numpy").sin(angulo)),xytext=(0,0),
                arrowprops=dict(arrowstyle="->",color="#F1F5F9",lw=3))
            ax.add_patch(plt.Circle((0,0),0.06,color="#F1F5F9",zorder=5))
            cor_p="#EF4444" if prob>=.5 else "#F59E0B" if prob>=.3 else "#00C897"
            ax.text(0,-0.2,f"{pct:.1f}%",ha="center",va="center",fontsize=26,fontweight="bold",color=cor_p,fontfamily="monospace")
            ax.text(0,-0.42,"probabilidade de risco",ha="center",fontsize=9,color="#64748B")
            patches=[mpatches.Patch(color="#00C897",label="Baixo <30%"),mpatches.Patch(color="#F59E0B",label="Moderado 30-50%"),mpatches.Patch(color="#EF4444",label="Alto >50%")]
            ax.legend(handles=patches,loc="lower center",ncol=3,fontsize=7,frameon=False,labelcolor="#94A3B8")
            ax.set_xlim(-1.15,1.15); ax.set_ylim(-0.55,1.15)
            st.pyplot(fig_g,use_container_width=True); plt.close()
        with cr:
            if prob>=.5:
                st.markdown(f'<div class="risk-card ra"><div class="rt">Risco Alto {pct:.1f}%</div><div class="rb2">Alta probabilidade de queda no proximo ciclo.</div><div class="rr"><strong style="color:#F87171">Acao:</strong> Acionar acompanhamento psicopedagogico imediato.</div></div>',unsafe_allow_html=True)
            elif prob>=.3:
                st.markdown(f'<div class="risk-card rm"><div class="rt">Risco Moderado {pct:.1f}%</div><div class="rb2">Sinais de alerta. Monitoramento recomendado.</div><div class="rr"><strong style="color:#FBB03B">Acao:</strong> Monitorar IEG e IDA no proximo ciclo.</div></div>',unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-card rb"><div class="rt">Baixo Risco {pct:.1f}%</div><div class="rb2">Indicadores estaveis. Baixa probabilidade de queda.</div><div class="rr"><strong style="color:#00C897">Acao:</strong> Manter acompanhamento padrao.</div></div>',unsafe_allow_html=True)
            st.markdown("<br>",unsafe_allow_html=True)
            rs=pd.DataFrame({"Indicador":["INDE","IDA","IEG","IAA","IPS","IPV","IAN","IPP"],"Valor":[f"{v:.1f}" for v in [INDE,IDA,IEG,IAA,IPS,IPV,IAN,IPP]]})
            st.dataframe(rs,hide_index=True,use_container_width=True)
