# Datathon — Passos Mágicos
**Pós-graduação em Data Analytics · FIAP Postech · Fase 5**

Análise de dados educacionais da Associação Passos Mágicos (2022–2024) com modelo preditivo de risco de defasagem escolar e aplicação interativa em Streamlit.

---

## Estrutura do repositório

```
├── notebooks/
│   ├── 01_tratamento_dados.ipynb   # Carga, diagnóstico e tratamento da base PEDE
│   ├── 02_eda_perguntas.ipynb      # EDA respondendo as 11 perguntas de negócio
│   └── 03_modelo_preditivo.ipynb   # Feature engineering, treino, avaliação e exportação do modelo
├── data/
│   └── painel_pede_tratado.csv     # Base consolidada (painel longo 2022-2024)
├── app/
│   ├── app.py                      # Aplicação Streamlit
│   ├── requirements.txt            # Dependências Python
│   └── model/
│       ├── modelo_risco.pkl        # Modelo XGBoost treinado
│       ├── scaler.pkl              # StandardScaler
│       └── features.pkl            # Lista de features
└── README.md
```

---

## Sobre o projeto

A **Associação Passos Mágicos** atua há 35 anos na transformação de vida de crianças e jovens em vulnerabilidade social por meio da educação. Este projeto utiliza dados da Pesquisa Extensiva do Desenvolvimento Educacional (PEDE) de 2022, 2023 e 2024 para:

1. **Analisar** os indicadores educacionais (INDE, IAN, IDA, IEG, IAA, IPS, IPP, IPV)
2. **Responder** às 11 perguntas de negócio do desafio
3. **Prever** o risco de defasagem de cada aluno antes que a queda aconteça

---

## Modelo Preditivo

**Algoritmo:** XGBoost  
**Target:** aluno que no ano seguinte apresenta queda de INDE > 0,3 **ou** piora no nível de defasagem  
**Performance:** ROC-AUC = 0.795 no conjunto de teste  
**Features:** 15 variáveis (indicadores brutos + features derivadas de trajetória e comportamento)

---

## Aplicação Streamlit

🔗 **[Acesse o app aqui](https://datathon-paapps-magicos-c83ufhmlkpkc6gawdhxbka.streamlit.app/)**

O app permite:
- Avaliar o risco de um aluno individualmente com sliders interativos
- Fazer upload de um CSV com múltiplos alunos e obter o risco em lote
- Consultar informações sobre o modelo e suas limitações

### Deploy local

```bash
cd app
pip install -r requirements.txt
streamlit run app.py
```

---

## Base de dados

A base original (`BASE_DE_DADOS_PEDE_2024_-_DATATHON.xlsx`) contém 3 abas anuais:
- **PEDE2022:** 860 alunos, 42 colunas
- **PEDE2023:** 1.014 alunos, 48 colunas  
- **PEDE2024:** 1.156 alunos, 50 colunas

Após tratamento, foi consolidada em um painel longo de **3.030 linhas × 38 colunas**.

---

## Tecnologias utilizadas

`Python` · `Pandas` · `Scikit-learn` · `XGBoost` · `Matplotlib` · `Seaborn` · `Streamlit` · `Joblib`
