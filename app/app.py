"""
Datathon Passos Mágicos — App de Predição de Risco de Defasagem
Pós-graduação Data Analytics — FIAP Postech — Fase 5
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="Passos Mágicos · Risco de Defasagem",
    page_icon="⭐",
    layout="wide",
)

# ── Carregamento do modelo ────────────────────────────────────────────────────
@st.cache_resource
def carregar_modelo():
    modelo   = joblib.load("model/modelo_risco.pkl")
    scaler   = joblib.load("model/scaler.pkl")
    features = joblib.load("model/features.pkl")
    return modelo, scaler, features

modelo, scaler, FEATURES = carregar_modelo()

# ── Estilos ───────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .titulo-app   { font-size:2rem; font-weight:700; color:#0d6e3c; }
    .subtitulo    { font-size:1rem; color:#555; margin-bottom:1.5rem; }
    .card-risco   { background:#fdecea; border-left:6px solid #d32f2f;
                    padding:1rem 1.2rem; border-radius:8px; margin-top:1rem; }
    .card-seguro  { background:#e8f5e9; border-left:6px solid #2e7d32;
                    padding:1rem 1.2rem; border-radius:8px; margin-top:1rem; }
    .card-alerta  { background:#fff8e1; border-left:6px solid #f9a825;
                    padding:1rem 1.2rem; border-radius:8px; margin-top:1rem; }
    .label-ind    { font-size:.8rem; color:#777; margin-bottom:2px; }
    .val-ind      { font-size:1.3rem; font-weight:700; color:#333; }
    hr-custom     { border:0; border-top:1px solid #e0e0e0; margin:1.5rem 0; }
</style>
""", unsafe_allow_html=True)

# ── Cabeçalho ─────────────────────────────────────────────────────────────────
col_logo, col_titulo = st.columns([1, 8])
with col_logo:
    st.markdown("## ⭐")
with col_titulo:
    st.markdown('<p class="titulo-app">Passos Mágicos · Predição de Risco de Defasagem</p>', unsafe_allow_html=True)
    st.markdown('<p class="subtitulo">Identifique antecipadamente alunos com risco de queda no desempenho educacional.</p>', unsafe_allow_html=True)

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
aba_pred, aba_lote, aba_info = st.tabs([
    "🔍 Avaliar Aluno Individual",
    "📋 Avaliar Lote (CSV)",
    "ℹ️ Sobre o Modelo",
])

# ══════════════════════════════════════════════════════════════════════════════
# ABA 1 — INDIVIDUAL
# ══════════════════════════════════════════════════════════════════════════════
with aba_pred:
    st.subheader("Dados do Aluno")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📚 Indicadores Acadêmicos**")
        INDE = st.slider("INDE (Índice Geral)",        min_value=3.0,  max_value=9.5,  value=7.4, step=0.1)
        IDA  = st.slider("IDA (Desempenho Acadêmico)", min_value=0.0,  max_value=10.0, value=6.7, step=0.1)
        IAN  = st.selectbox("IAN (Adequação ao Nível)",
                            options=[2.5, 5.0, 10.0],
                            format_func=lambda x: {2.5:"2.5 — Severo", 5.0:"5.0 — Moderado", 10.0:"10.0 — Adequado"}[x],
                            index=1)
        IPP  = st.slider("IPP (Psicopedagógico)",      min_value=2.5,  max_value=10.0, value=7.5, step=0.1)
        Defasagem = st.selectbox("Defasagem (nível vs ideal)",
                                 options=list(range(-5, 4)),
                                 index=4)  # default -1

    with col2:
        st.markdown("**💡 Indicadores Comportamentais**")
        IAA  = st.slider("IAA (Autoavaliação)",         min_value=0.0,  max_value=10.0, value=8.8, step=0.1)
        IEG  = st.slider("IEG (Engajamento)",           min_value=0.0,  max_value=10.0, value=8.6, step=0.1)
        IPS  = st.slider("IPS (Psicossocial)",          min_value=2.5,  max_value=10.0, value=7.5, step=0.1)
        IPV  = st.slider("IPV (Ponto de Virada)",       min_value=2.5,  max_value=10.0, value=7.6, step=0.1)

    with col3:
        st.markdown("**🎓 Perfil do Aluno**")
        Fase_Num = st.selectbox("Fase atual",
                                options=list(range(0, 9)),
                                format_func=lambda x: "ALFA" if x == 0 else f"Fase {x}",
                                index=2)
        Ano_Ingresso = st.number_input("Ano de ingresso na PM",
                                       min_value=2016, max_value=2024, value=2021)
        Ano_Ref = st.selectbox("Ano de referência dos dados", [2022, 2023, 2024], index=1)
        Idade   = st.number_input("Idade do aluno", min_value=7, max_value=27, value=12)

    # ── Feature engineering (espelha o notebook) ──────────────────────────────
    Anos_no_Programa    = Ano_Ref - Ano_Ingresso
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

    # ── Predição ──────────────────────────────────────────────────────────────
    if st.button("▶ Calcular Risco", type="primary", use_container_width=True):
        prob = modelo.predict_proba(entrada)[0][1]
        pct  = prob * 100

        st.divider()
        c1, c2 = st.columns([1, 1])

        with c1:
            # Gauge visual simples
            fig, ax = plt.subplots(figsize=(4, 2.2), subplot_kw=dict(aspect='equal'))
            ax.axis('off')
            theta = np.linspace(np.pi, 0, 300)
            for i, (cor, lim) in enumerate(zip(
                ['#2e7d32','#f9a825','#d32f2f'],
                [(0, 1/3), (1/3, 2/3), (2/3, 1)]
            )):
                t = np.linspace(np.pi*(1 - lim[1]), np.pi*(1 - lim[0]), 100)
                ax.plot(np.cos(t), np.sin(t), color=cor, linewidth=18, solid_capstyle='butt')

            angulo = np.pi * (1 - prob)
            ax.annotate('', xy=(0.55*np.cos(angulo), 0.55*np.sin(angulo)),
                        xytext=(0, 0),
                        arrowprops=dict(arrowstyle='->', color='#333', lw=2.5))
            ax.text(0, -0.15, f"{pct:.1f}%", ha='center', va='center',
                    fontsize=22, fontweight='bold',
                    color='#d32f2f' if prob >= 0.5 else '#f9a825' if prob >= 0.3 else '#2e7d32')
            ax.set_xlim(-1.1, 1.1); ax.set_ylim(-0.3, 1.1)
            patches = [mpatches.Patch(color='#2e7d32', label='Baixo (<30%)'),
                       mpatches.Patch(color='#f9a825', label='Moderado (30-50%)'),
                       mpatches.Patch(color='#d32f2f', label='Alto (>50%)')]
            ax.legend(handles=patches, loc='lower center', ncol=3,
                      fontsize=7, frameon=False)
            ax.set_title("Probabilidade de Risco", fontsize=11, pad=4)
            st.pyplot(fig, use_container_width=True)
            plt.close()

        with c2:
            if prob >= 0.5:
                st.markdown(f"""<div class="card-risco">
                    <b>🔴 ALTO RISCO ({pct:.1f}%)</b><br><br>
                    O modelo indica alta probabilidade de queda no desempenho
                    ou aumento da defasagem no próximo período.<br><br>
                    <b>Recomendação:</b> Acionar acompanhamento psicopedagógico
                    e revisar o nível de defasagem com a equipe pedagógica.
                </div>""", unsafe_allow_html=True)
            elif prob >= 0.3:
                st.markdown(f"""<div class="card-alerta">
                    <b>🟡 RISCO MODERADO ({pct:.1f}%)</b><br><br>
                    Há sinais de alerta que merecem atenção. O aluno pode
                    se estabilizar, mas o monitoramento é recomendado.<br><br>
                    <b>Recomendação:</b> Monitorar engajamento (IEG) e
                    desempenho acadêmico (IDA) no próximo ciclo.
                </div>""", unsafe_allow_html=True)
            else:
                st.markdown(f"""<div class="card-seguro">
                    <b>🟢 BAIXO RISCO ({pct:.1f}%)</b><br><br>
                    O aluno apresenta indicadores estáveis. Baixa
                    probabilidade de queda no próximo período.<br><br>
                    <b>Recomendação:</b> Manter o acompanhamento padrão.
                </div>""", unsafe_allow_html=True)

            st.markdown("**Resumo dos indicadores inseridos:**")
            resumo = pd.DataFrame({
                "Indicador": ["INDE","IDA","IEG","IAA","IPS","IPV","IAN","IPP"],
                "Valor":     [INDE,  IDA,  IEG,  IAA,  IPS,  IPV,  IAN,  IPP],
            })
            resumo["Valor"] = resumo["Valor"].round(1)
            st.dataframe(resumo, hide_index=True, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# ABA 2 — LOTE
# ══════════════════════════════════════════════════════════════════════════════
with aba_lote:
    st.subheader("Avaliação em lote via CSV")
    st.markdown("""
    Faça upload de um arquivo CSV com as colunas abaixo para obter a probabilidade
    de risco para múltiplos alunos de uma vez.

    **Colunas obrigatórias:**
    `INDE, IAA, IEG, IPS, IDA, IPV, IAN, IPP, Defasagem, Fase_Num, Ano_Ingresso, Ano_Referencia, Idade`
    """)

    arquivo = st.file_uploader("Selecione o CSV", type=["csv"])

    if arquivo:
        try:
            df_lote = pd.read_csv(arquivo)

            # Feature engineering
            df_lote['Anos_no_Programa']    = df_lote['Ano_Referencia'] - df_lote['Ano_Ingresso']
            df_lote['Media_Comportamental'] = df_lote[['IAA','IEG','IPS']].mean(axis=1)
            df_lote['Media_Academica']      = df_lote[['IDA','IPV','IPP']].mean(axis=1)
            df_lote['Gap_Auto_Real']        = df_lote['IAA'] - df_lote['IDA']

            X_lote = df_lote[FEATURES]
            probs  = modelo.predict_proba(X_lote)[:, 1]

            df_lote['Prob_Risco']    = (probs * 100).round(1)
            df_lote['Classificacao'] = pd.cut(
                probs,
                bins=[-0.01, 0.30, 0.50, 1.01],
                labels=['🟢 Baixo', '🟡 Moderado', '🔴 Alto']
            )

            col_res = ['RA','Nome','Fase_Num','INDE','Prob_Risco','Classificacao'] \
                      if 'RA' in df_lote.columns else ['Fase_Num','INDE','Prob_Risco','Classificacao']

            st.success(f"{len(df_lote)} alunos processados.")
            st.dataframe(
                df_lote[col_res].sort_values('Prob_Risco', ascending=False),
                hide_index=True,
                use_container_width=True,
            )

            csv_out = df_lote.to_csv(index=False).encode('utf-8-sig')
            st.download_button(
                "⬇ Baixar resultado completo (CSV)",
                data=csv_out,
                file_name="resultado_risco_defasagem.csv",
                mime="text/csv",
            )
        except Exception as e:
            st.error(f"Erro ao processar o arquivo: {e}")
            st.info("Verifique se todas as colunas obrigatórias estão presentes no CSV.")

# ══════════════════════════════════════════════════════════════════════════════
# ABA 3 — SOBRE O MODELO
# ══════════════════════════════════════════════════════════════════════════════
with aba_info:
    st.subheader("Sobre o Modelo Preditivo")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **Algoritmo:** XGBoost (Extreme Gradient Boosting)

        **Performance (conjunto de teste):**
        | Métrica | Valor |
        |---|---|
        | ROC-AUC | 0.795 |
        | Acurácia geral | 80% |
        | Precisão (Em Risco) | 50% |
        | Recall (Em Risco) | 18% |

        **Definição de risco:**
        O aluno é classificado como **em risco** quando no ano anterior apresenta:
        - Queda de INDE superior a **0,3 ponto** no ano seguinte, **ou**
        - Piora no nível de defasagem escolar
        """)

    with col_b:
        st.markdown("""
        **Features utilizadas (15):**
        | Feature | Descrição |
        |---|---|
        | INDE | Índice de Desenvolvimento Educacional |
        | IAA | Autoavaliação |
        | IEG | Engajamento |
        | IPS | Psicossocial |
        | IDA | Desempenho Acadêmico |
        | IPV | Ponto de Virada |
        | IAN | Adequação ao Nível |
        | IPP | Psicopedagógico |
        | Defasagem | Diferença nível atual vs ideal |
        | Fase_Num | Fase atual (0=ALFA a 8) |
        | Anos_no_Programa | Anos na Passos Mágicos |
        | Idade | Idade do aluno |
        | Media_Comportamental | Média IAA+IEG+IPS |
        | Media_Academica | Média IDA+IPV+IPP |
        | Gap_Auto_Real | IAA − IDA (superestimação) |
        """)

    st.divider()
    st.markdown("""
    **Limitações:**
    - O modelo foi treinado com dados de 2022–2024 da Pesquisa PEDE.
    - IPP não estava disponível em 2022 e foi imputado pela mediana do grupo fase+ano.
    - Alunos com apenas 1 ano de dados não compõem o conjunto de treino (sem rótulo futuro).
    - Recomenda-se revalidar o modelo a cada novo ciclo PEDE com dados atualizados.

    **Projeto:** Datathon Passos Mágicos — FIAP Postech Fase 5
    """)
