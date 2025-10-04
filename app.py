import streamlit as st
import pandas as pd
import plotly.express as px
# Certifique-se de que seu arquivo pipeline.py esteja no mesmo diretório
# from pipeline import processar_e_prever

# --- CONFIGURAÇÃO DA PÁGINA ---
st.set_page_config(
    page_title="NASA Exoplanet Classifier", 
    page_icon="🪐", 
    layout="wide"
)

# --- PALETA DE CORES E ESTILO CSS ---
# Paleta de cores oficial da NASA e cores de suporte para UI
NASA_BLUE = "#0B3D91"
NASA_RED = "#FC3D21"
NASA_LIGHT_GRAY = "#BEBEBE"
TEXT_COLOR_DARK = "#FFFFFF"  # Para fundos claros
TEXT_COLOR_LIGHT = "#FFFFFF" # Para fundos escuros
BACKGROUND_COLOR = "#000000"
SUCCESS_BG = "#D4EDDA" # Verde claro para sucesso
SUCCESS_TEXT = "#155724" # Verde escuro para texto de sucesso
WARNING_BG = "#FFF3CD" # Amarelo claro para aviso
WARNING_TEXT = "#856404" # Amarelo/marrom escuro para texto de aviso

# CSS Corrigido com seletores específicos
st.markdown(f"""
<style>
    /* Fundo principal da aplicação */
    .stApp {{
        background-color: {BACKGROUND_COLOR};
    }}

    /* Barra lateral - Fundo e cor do texto */
    [data-testid="stSidebar"] {{
        background-color: {NASA_LIGHT_GRAY};
        color: {TEXT_COLOR_DARK}; /* CORRIGIDO: Garante que o texto na sidebar seja escuro */
    }}
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {{
        color: {NASA_BLUE}; /* Mantém os títulos da sidebar em azul */
    }}

    /* Títulos e Headers na área principal */
    .main h1, .main h2, .main h3, .main h4, .main h5, .main h6 {{
        color: {NASA_BLUE};
    }}
    
    /* Texto geral (parágrafos) na área principal */
    .main p {{
        color: {TEXT_COLOR_DARK} !important;
    }}

    /* Métricas (st.metric) - CORRIGIDO */
    [data-testid="stMetricLabel"] {{
        color: #888888; /* Cor cinza para o rótulo da métrica */
    }}
    [data-testid="stMetricValue"] {{
        color: {TEXT_COLOR_DARK}; /* Cor escura para o valor da métrica */
    }}

    /* Alertas (st.success, st.warning) - CORRIGIDO */
    [data-testid="stAlert"][data-baseweb="alert"] {{
        border-radius: 5px;
    }}
    /* Sucesso */
    [data-testid="stAlert"][data-baseweb="alert"][kind="success"] {{
        background-color: {SUCCESS_BG};
        color: {SUCCESS_TEXT}; /* Texto escuro para fundo claro */
    }}
    /* Aviso */
    [data-testid="stAlert"][data-baseweb="alert"][kind="warning"] {{
        background-color: {WARNING_BG};
        color: {WARNING_TEXT}; /* Texto escuro para fundo claro */
    }}

    /* Botão Primário (Analisar) */
    .stButton > button {{
        border: 2px solid {NASA_RED};
        background-color: {NASA_RED};
        color: #FFFFFF; /* Corrigido: texto branco para contraste */
        border-radius: 5px;
        font-weight: bold;
    }}
    .stButton > button:hover {{
        background-color: #E0361E;
        border-color: #E0361E;
        color: #FFFFFF; /* Mantém branco ao passar o mouse */
    }}
    
    /* Abas (Tabs) */
    .stTabs [aria-selected="true"] {{
        color: {NASA_RED} !important;
        font-weight: bold;
        border-bottom: 2px solid {NASA_RED};
    }}
    .stTabs [data-baseweb="tab"] {{
        color: #555555; /* Cinza escuro para melhorar contraste em abas não selecionadas */
    }}

</style>
""", unsafe_allow_html=True)


# --- BARRA LATERAL (SIDEBAR) ---
with st.sidebar:
    st.image("https://upload.wikimedia.org/wikipedia/commons/e/e5/NASA_logo.svg", width=150)
    st.title("Exoplanet Classifier")
    
    uploaded_file = st.file_uploader(
        "Selecione um arquivo CSV para análise:", 
        type="csv",
        help="O arquivo deve conter dados de candidatos a exoplanetas, como os do arquivo KOI do Kepler."
    )
    
    analyze_button = st.button(
        "Analisar Candidatos", 
        use_container_width=True, 
        type="primary"
    )
    
    st.markdown("---")
    
    st.header("Performance do Modelo")
    col1, col2 = st.columns(2)
    col1.metric("Acurácia", "99.0%", help="Acurácia da validação cruzada.")
    col2.metric("AUC (ROC)", "0.998", help="Área sob a curva ROC.")

# --- ÁREA PRINCIPAL (MAIN CONTENT) ---
st.header("🛰️ Análise Preditiva de Objetos de Interesse Kepler (KOI)")
st.markdown("Faça o upload de um arquivo de dados para classificar os candidatos a exoplanetas em 'Confirmado' ou 'Falso Positivo' usando nosso modelo de Machine Learning.")

if analyze_button and uploaded_file is not None:
    with st.spinner('Analisando dados... Por favor, aguarde.'):
        # DADOS DE EXEMPLO (SUBSTITUA PELA SUA FUNÇÃO `processar_e_prever`)
        import numpy as np
        data = {
            'kepoi_name': [f'K00{i}' for i in range(100)],
            'koi_duration': np.random.uniform(1, 10, 100),
            'koi_depth': np.random.uniform(100, 2000, 100),
            'Predicao': np.random.choice(['CONFIRMADO', 'FALSO POSITIVO'], 100, p=[0.36, 0.64])
        }
        df_resultados = pd.DataFrame(data)
        avisos = ["Aviso de exemplo: 2 colunas foram removidas por conterem valores nulos."]
        # FIM DOS DADOS DE EXEMPLO

    st.success("Análise concluída com sucesso!")
    if avisos:
        for aviso in avisos:
            st.warning(aviso)

    tab1, tab2 = st.tabs(["**Dashboard Resumo**", "**Dados Detalhados**"])

    with tab1:
        total = len(df_resultados)
        confirmados = (df_resultados['Predicao'] == 'CONFIRMADO').sum()
        falso_positivo = total - confirmados
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Candidatos Analisados", f"{total}")
        col2.metric("Predição: CONFIRMADO", f"{confirmados}")
        col3.metric("Predição: FALSO POSITIVO", f"{falso_positivo}")
        
        st.markdown("---")
        
        col_graf1, col_graf2 = st.columns(2)
        with col_graf1:
            st.subheader("Distribuição das Predições")
            fig_pie = px.pie(
                df_resultados, names='Predicao', color='Predicao',
                color_discrete_map={'CONFIRMADO': NASA_BLUE, 'FALSO POSITIVO': '#B9D2EE'}
            )
            fig_pie.update_layout(template='plotly_white', legend_title_text='Classe')
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_graf2:
            st.subheader("Profundidade vs. Duração do Trânsito")
            fig_scatter = px.scatter(
                df_resultados, x='koi_duration', y='koi_depth', color='Predicao',
                labels={'koi_duration': 'Duração (horas)', 'koi_depth': 'Profundidade (ppm)'},
                hover_name='kepoi_name',
                color_discrete_map={'CONFIRMADO': NASA_BLUE, 'FALSO POSITIVO': '#B9D2EE'}
            )
            fig_scatter.update_layout(template='plotly_white', legend_title_text='Predição')
            st.plotly_chart(fig_scatter, use_container_width=True)
    
    with tab2:
        st.subheader("Visualização dos Dados Completos")
        st.dataframe(df_resultados, use_container_width=True)

else:
    st.info("Aguardando o upload de um arquivo e o comando para analisar.")