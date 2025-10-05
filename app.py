import streamlit as st
import pandas as pd
import plotly.express as px
from pipeline import processar_e_prever
import os
import shap
import matplotlib.pyplot as plt
import numpy as np

st.set_page_config(
    page_title="Celestium: Exoplanet Classifier",
    page_icon="logo (2).png",
    layout="wide"
)

if 'analysis_complete' not in st.session_state:
    st.session_state.analysis_complete = False
    st.session_state.df_resultados = pd.DataFrame()
    st.session_state.avisos = []
    st.session_state.shap_values = None
    st.session_state.X_final = None
    st.session_state.explainer = None
    
st.markdown("""
<style>
    
/* --- FONTES NASA SPACE APPS --- */
@font-face {
    font-family: 'Fira Sans Black';
    src: url('FiraSans-Black.woff2') format('woff2');
}
@font-face {
    font-family: 'Fira Sans Bold';
    src: url('FiraSans-Bold.woff2') format('woff2');
}
@font-face {
    font-family: 'Viaduto Regular';
    src: url('Viaduto-Regular.woff2') format('woff2');
}
@font-face {
    font-family: 'Viaduto Bold';
    src: url('Viaduto-Bold.woff2') format('woff2');
}
@font-face {
    font-family: 'Viaduto Italic';
    src: url('Viaduto-Italic.woff2') format('woff2');
}

/* Aplicação das fontes */
h1, h2, h3, h4, h5, h6 {
    font-family: 'Fira Sans Black', 'Fira Sans Bold', 'Viaduto Bold', 'Viaduto Regular', sans-serif;
}
p, li, .stMarkdown, div, span {
    font-family: 'Viaduto Regular', 'Viaduto Bold', 'Viaduto Italic', sans-serif;
}

    /* ### INÍCIO DA SEÇÃO MODIFICADA ### */

    /* --- CONFIGURAÇÕES GERAIS --- */
    /* Define o fundo principal como branco e o texto como preto */
    .stApp {
        background-color: #FFFFFF;
    }
    h1, h2, h3, h4, h5, h6, p, li, .stMarkdown {
        color: #000000;
    }

    /* --- BARRA LATERAL (SIDEBAR) --- */
    /* Fundo azul escuro e texto branco */
    [data-testid="stSidebar"] {
        background-color: #07173F;
    }
    [data-testid="stSidebar"] * {
        color: #FFFFFF;
    }

    /* Adiciona uma borda branca sutil na imagem do logo */
    [data-testid="stSidebar"] img {
        border: 1px solid white;
        border-radius: 50%; /* Faz a borda ficar circular */
    }

    /* --- BOTÕES E WIDGETS --- */
    /* Botão primário (Analisar) laranja */
    .stButton > button {
        border: 2px solid #E43700;
        background-color: #E43700;
        color: #FFFFFF;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #E43700;
        border-color: #E43700;
        color: #FFFFFF;
    }

    /* Widget de Upload de Arquivo laranja */
    [data-testid="stFileUploader"] {
        background-color: #E43700;
        border-radius: 0.5rem;
        padding: 1rem;
    }
    [data-testid="stFileUploader"] label {
        color: #FFFFFF !important;
        font-weight: bold;
        margin-bottom: 8px; /* Adiciona espaço entre o rótulo e a área de upload */
        display: block; /* Garante que o rótulo ocupe sua própria linha */
    }
    [data-testid="stFileUploader"] section {
        background-color: transparent;
        border: 2px dashed #FFFFFF;
    }
    [data-testid="stFileUploader"] section [data-testid="stText"] {
        color: #FFFFFF;
    }
    [data-testid="stFileUploader"] section button {
        border: 2px solid #FFFFFF;
        background-color: #FFFFFF;
        color: #E43700; /* Texto do botão laranja para contraste */
    }
    /* Estiliza o nome do arquivo após o upload */
    [data-testid="stFileUploader"] [data-testid="stFileDeleteBtn"] {
        color: #FFFFFF; /* Cor do 'x' para apagar */
    }
     [data-testid="stFileUploader"] [data-testid="stFileName"] {
        color: #FFFFFF;
    }


    /* Abas (Tabs) com destaque laranja */
    .stTabs [aria-selected="true"] {
        color: #FC3D21 !important;
        font-weight: bold;
        border-bottom: 2px solid #E43700;
    }
    .stTabs [data-baseweb="tab"] {
        color: #555555; /* Cor para abas não selecionadas */
    }
    /* ### FIM DA SEÇÃO MODIFICADA ### */
</style>
""", unsafe_allow_html=True)
def run_analysis():
    if 'uploaded_file' in st.session_state and st.session_state.uploaded_file is not None:
        try:
            if 'objeto_selecionado' in st.session_state:
                del st.session_state['objeto_selecionado']
            df_bruto = pd.read_csv(st.session_state.uploaded_file)
            st.session_state.df_resultados, st.session_state.avisos, st.session_state.shap_values, st.session_state.X_final, st.session_state.explainer = processar_e_prever(df_bruto)
            st.session_state.analysis_complete = True
        except Exception as e:
            st.error(f"An error occurred during analysis: {e}")
            st.session_state.analysis_complete = False

with st.sidebar:
    st.title("Discovering New Worlds with AI")
    st.markdown("""
    <div style='font-size: small; color: #FFFFFF; opacity: 0.8;'>
    Required Format: The CSV file must follow the data scheme of NASA's Kepler Objects of Interest (KOI) database.
    </div>
    """, unsafe_allow_html=True)
    st.file_uploader("Upload a CSV file:", type="csv", key='uploaded_file')
    st.button("Analyze Candidates", use_container_width=True, type="primary", disabled=not st.session_state.get('uploaded_file'), on_click=run_analysis)
    st.markdown("---")

import base64
from PIL import Image
from io import BytesIO

def get_base64_image(image_path):
    img = Image.open(image_path)
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_base64 = base64.b64encode(buffered.getvalue()).decode()
    return img_base64

logo_path = "logo (2).png"
logo_base64 = get_base64_image(logo_path)

st.markdown(
    f"""
    <div style="
        display: flex;
        justify-content: flex-end;
        align-items: center;
        width: 100%;
        padding: 10px 20px;
        box-sizing: border-box;
    ">
        <img src="data:image/png;base64,{logo_base64}" 
             style="max-width: 180px; height: auto;">
    </div>
    """,
    unsafe_allow_html=True
)

st.title("Exoplanet Classification Tool")
st.markdown("Upload planetary transit data (Kepler Objects of Interest) to classify them using a Machine Learning model.")
if st.session_state.get('analysis_complete'):
    df_resultados = st.session_state.df_resultados
    avisos = st.session_state.avisos
    shap_values = st.session_state.shap_values
    X_final = st.session_state.X_final
    explainer = st.session_state.explainer

    if "Perfect analysis" in avisos[0]:
        st.success("Analysis successfully completed. All data was complete.")
    else:
        st.warning("Warning: Missing data was automatically filled, which may affect accuracy.")
        with st.expander("Click to view imputation details"):
            for aviso in avisos:
                st.info(aviso)
    
    tab1, tab2, tab3, tab4 = st.tabs([
        "**Summary Dashboard**",
        "**Results Table**",
        "**Individual Analysis**",
        "**About the Model**"
    ])    
    with tab1:
        df_display_viz = df_resultados.copy()
        mapeamento_viz = {'FALSO POSITIVO': 'FALSE POSITIVE', 'CONFIRMADO': 'CONFIRMED'}
        df_display_viz['Prediction_EN'] = df_display_viz['Predicao'].map(mapeamento_viz)
        
        total = len(df_display_viz)
        confirmados = (df_display_viz['Prediction_EN'] == 'CONFIRMED').sum()
        falso_positivo = total - confirmados
        
        col1, col2, col3 = st.columns(3)
        col1.metric("Candidates Analyzed", f"{total}")
        col2.metric("Prediction: CONFIRMED", f"{confirmados}")
        col3.metric("Prediction: FALSE POSITIVE", f"{falso_positivo}")
        
        st.markdown("---")

        col_graf1, col_graf2 = st.columns(2)
        with col_graf1:
            st.subheader("Prediction Distribution")
            
            fig_pie = px.pie(
                df_display_viz, 
                names='Prediction_EN', 
                color_discrete_map={'CONFIRMED':'#0B3D91', 'FALSE POSITIVE':'#B9D2EE'} 
            )
            fig_pie.update_layout(legend_title_text='')
            
            fig_pie.update_traces(hovertemplate='Prediction: %{label}<br>Count: %{value}<br>Proportion: %{percent}<extra></extra>')
            
            st.plotly_chart(fig_pie, use_container_width=True)
            st.markdown("<div style='text-align: center; font-size: small;'>Pie chart illustrating the proportion of classified candidates.</div>", unsafe_allow_html=True)
        
        with col_graf2:
            st.subheader("Depth vs. Transit Duration")
            
            fig_scatter = px.scatter(
                df_display_viz, 
                x='koi_duration', 
                y='koi_depth', 
                color='Prediction_EN', 
                labels={
                    'koi_duration': 'Duration (h)',
                    'koi_depth': 'Depth (ppm)',
                    'Prediction_EN': 'Prediction'  
                }, 
                hover_name='kepoi_name', 
                color_discrete_map={'CONFIRMED':'#0B3D91', 'FALSE POSITIVE':'#B9D2EE'}
            )
            fig_scatter.update_layout(legend_title_text='Prediction')

            st.plotly_chart(fig_scatter, use_container_width=True)
            st.markdown("<div style='text-align: center; font-size: small;'>Scatter plot correlating Transit Depth (`koi_depth`) and Duration (`koi_duration`).</div>", unsafe_allow_html=True)
 
    with tab2:
        st.subheader("Results per Candidate")
        rename_map = {
            'kepoi_name': 'Object of Interest',
            'Predicao': 'Prediction',
            'Score_Confianca': 'Model Confidence (%)',
            'koi_duration': 'Transit Duration (h)',
            'koi_depth': 'Transit Depth (ppm)',
            'koi_period': 'Orbital Period (days)',
            'koi_model_snr': 'Signal-to-Noise Ratio (SNR)',
            'koi_prad': 'Planetary Radius (R⨁)',
            'koi_teq': 'Equilibrium Temperature (K)',
            'koi_impact': 'Impact Parameter'
        }

        df_display = df_resultados.copy()
        
        if 'Status_Dados' in df_display.columns:
            df_display.drop(columns=['Status_Dados'], inplace=True)
        df_display.rename(columns=rename_map, inplace=True)

        st.dataframe(
            df_display.style.set_properties(**{'text-align': 'center'})
                           .set_table_styles([{'selector': 'th', 'props': [('text-align', 'center')]}]),
            use_container_width=True
        )

    with tab3:
        st.header("Detailed Analysis per Candidate")
        st.markdown("Select a candidate to view the technical justification for its classification.")

        if not df_resultados.empty:
            objeto_selecionado = st.selectbox('Select the Object of Interest (KOI):', df_resultados['kepoi_name'], key="objeto_selecionado")
            
            if objeto_selecionado:
                idx = df_resultados.index[df_resultados['kepoi_name'] == objeto_selecionado].tolist()[0]
                predicao = df_resultados.loc[idx, 'Predicao']
                confianca = df_resultados.loc[idx, 'Score_Confianca']
                
                st.subheader(f"Justification for the Classification of: {objeto_selecionado}")
                st.write(f"**Prediction:** {predicao} | **Confidence:** {confianca}")
                
                shap_values_classe_1 = shap_values[:, :, 1]
                
                plt.figure(figsize=(20, 5))
                fig = shap.force_plot(
                    explainer.expected_value[1], 
                    shap_values_classe_1[idx, :], 
                    X_final.iloc[idx, :].round(3), 
                    matplotlib=True,
                    show=False,
                    text_rotation=30
                )
                st.pyplot(fig, clear_figure=True)
                st.markdown("<div style='text-align: center; font-size: small;'>SHAP power plot. Features in red push the prediction to 'Confirmed', and those in blue to 'False Positive'.</div>", unsafe_allow_html=True)

                st.subheader("Detailed Technical Justification")
                
                mapeamento_explicacoes = {
                'koi_fpflag_nt': lambda v: f"the **presence of the 'not transit-like signal' flag** (value {v:.0f}), a strong indication against a valid candidate." if v == 1 else f"the **absence of the 'not transit-like signal' flag** (value {v:.0f}), indicating that the light curve is consistent with a transit.",
                'koi_fpflag_ss': lambda v: f"the **presence of the 'stellar variability' flag** (value {v:.0f}), suggesting that the signal likely arises from stellar activity." if v == 1 else f"the **absence of the 'stellar variability' flag** (value {v:.0f}), suggesting the signal is not caused by starspots.",
                'koi_fpflag_co': lambda v: f"the **presence of the 'centroid offset' flag** (value {v:.0f}), suggesting contamination from a background star." if v == 1 else f"the **absence of the 'centroid offset' flag** (value {v:.0f}), strengthening the hypothesis that the transit occurs in the target star.",
                'koi_fpflag_ec': lambda v: f"the **presence of the 'eclipsing binary' flag** (value {v:.0f}), a strong indicator of a false positive." if v == 1 else f"the **absence of the 'eclipsing binary' flag** (value {v:.0f}), reducing the likelihood of a false positive.",
                'koi_model_snr': lambda v: f"a **high signal-to-noise ratio (SNR) of {v:.2f}**, indicating a clear and strong transit signal." if v > 20 else f"a **low signal-to-noise ratio (SNR) of {v:.2f}**, suggesting a weak or noisy signal.",
                'koi_impact': lambda v: f"a **high impact parameter of {v:.3f}**, indicating a grazing transit, less likely to be planetary." if v > 0.8 else f"a **low impact parameter of {v:.3f}**, suggesting a central transit across the stellar disk.",
                'koi_prad': lambda v: f"an **estimated planetary radius of {v:.2f} Earth radii**, which is too large and more characteristic of a small star or brown dwarf." if v > 15 else f"an **estimated planetary radius of {v:.2f} Earth radii**, a plausible size for an exoplanet.",
                'koi_depth': lambda v: f"a **transit depth of {v:.1f} ppm**, too deep for a rocky planet, suggesting a gas giant or another type of object." if v > 100000 else f"a **transit depth of {v:.1f} ppm**, consistent with a planetary-sized object.",
                'koi_duration': lambda v: f"a **transit duration of {v:.2f} hours**, consistent with the orbit of a planetary candidate.",
                'koi_period': lambda v: f"an **orbital period of {v:.2f} days**.",
                'koi_ror': lambda v: f"a **radius ratio (planet/star) of {v:.4f}**, a key indicator for confirmation.",
                'koi_incl': lambda v: f"a **high orbital inclination of {v:.2f} degrees**, indicating an edge-on view as expected for transits.",
                'koi_teq': lambda v: f"an **equilibrium temperature of {v:.0f} K**.",
                'koi_insol': lambda v: f"an **insolation flux of {v:.2f} (relative to Earth)**.",
                'koi_dor': lambda v: f"a **transit-to-stellar radius distance of {v:.2f}**.",
                'koi_ldm_coeff1': lambda v: f"the **limb darkening coefficient (linear) of {v:.3f}**.",
                'koi_ldm_coeff2': lambda v: f"the **limb darkening coefficient (quadratic) of {v:.3f}**.",
                'koi_tce_plnt_num': lambda v: f"the **transit event threshold number ({v:.0f})**.",
                'koi_steff': lambda v: f"the **stellar effective temperature of {v:.0f} K**.",
                'koi_slogg': lambda v: f"the **stellar surface gravity of {v:.3f} (log10 cm/s²)**.",
                'koi_srad': lambda v: f"the **stellar radius of {v:.3f} solar radii**.",
                'koi_smass': lambda v: f"the **stellar mass of {v:.3f} solar masses**.",
                'ra': lambda v: f"the **right ascension of the star of {v:.3f} degrees**.",
                'dec': lambda v: f"the **declination of the star of {v:.3f} degrees**.",
                'koi_kepmag': lambda v: f"the **stellar magnitude in the Kepler filter of {v:.3f}**.",
                'koi_period_err1': lambda v: f"a **low uncertainty in the orbital period measurement** (+{v:.1e} days), indicating a stable, periodic signal.",
                'koi_period_err2': lambda v: f"a **low uncertainty in the orbital period measurement** ({v:.1e} days), indicating a stable, periodic signal.",
                'koi_time0_err1': lambda v: f"a **low uncertainty in the transit timing** (+{v:.1e} days).",
                'koi_time0_err2': lambda v: f"a **low uncertainty in the transit timing** ({v:.1e} days).",
                'koi_impact_err1': lambda v: f"a **low uncertainty in the impact parameter** (+{v:.2f}).",
                'koi_impact_err2': lambda v: f"a **low uncertainty in the impact parameter** ({v:.2f}).",
                'koi_duration_err1': lambda v: f"a **low uncertainty in the transit duration** (+{v:.1e} hours).",
                'koi_duration_err2': lambda v: f"a **low uncertainty in the transit duration** ({v:.1e} hours).",
                'koi_depth_err1': lambda v: f"a **low uncertainty in the transit depth** (+{v:.1f} ppm).",
                'koi_depth_err2': lambda v: f"a **low uncertainty in the transit depth** ({v:.1f} ppm).",
                'koi_prad_err1': lambda v: f"a **low uncertainty in the planetary radius** (+{v:.1e} Earth radii).",
                'koi_prad_err2': lambda v: f"a **low uncertainty in the planetary radius** ({v:.1e} Earth radii).",
                'koi_insol_err1': lambda v: f"a **low uncertainty in the insolation flux** (+{v:.1f}).",
                'koi_insol_err2': lambda v: f"a **low uncertainty in the insolation flux** ({v:.1f}).",
                'koi_dor_err1': lambda v: f"a **low uncertainty in the transit-to-stellar radius distance** (+{v:.1f}).",
                'koi_dor_err2': lambda v: f"a **low uncertainty in the transit-to-stellar radius distance** ({v:.1f}).",
                'koi_ror_err1': lambda v: f"a **low uncertainty in the radius ratio** (+{v:.1e}).",
                'koi_ror_err2': lambda v: f"a **low uncertainty in the radius ratio** ({v:.1e}).",
                'koi_steff_err1': lambda v: f"a **low uncertainty in the stellar temperature** (+{v:.1f} K).",
                'koi_steff_err2': lambda v: f"a **low uncertainty in the stellar temperature** ({v:.1f} K).",
                'koi_slogg_err1': lambda v: f"a **low uncertainty in the stellar surface gravity** (+{v:.2f}).",
                'koi_slogg_err2': lambda v: f"a **low uncertainty in the stellar surface gravity** ({v:.2f}).",
                'koi_srad_err1': lambda v: f"a **low uncertainty in the stellar radius** (+{v:.2f}).",
                'koi_srad_err2': lambda v: f"a **low uncertainty in the stellar radius** ({v:.2f}).",
                'koi_smass_err1': lambda v: f"a **low uncertainty in the stellar mass** (+{v:.2f}).",
                'koi_smass_err2': lambda v: f"a **low uncertainty in the stellar mass** ({v:.2f}).",
                }
                
                df_shap = pd.DataFrame({'feature': X_final.columns, 'valor': X_final.iloc[idx, :], 'shap_value': shap_values_classe_1[idx, :]})
                
                LIMIAR_DE_IMPACTO = 0.05
                
                if predicao == "CONFIRMADO":
                    st.markdown("The classification as **CONFIRMED** was mainly influenced by the following positive factors:")
                    fatores = df_shap[df_shap['shap_value'] > LIMIAR_DE_IMPACTO].sort_values(by='shap_value', ascending=False)
                else: 
                    st.markdown("The classification as **FALSE POSITIVE** was mainly influenced by the following warning signs:")
                    fatores = df_shap[df_shap['shap_value'] < -LIMIAR_DE_IMPACTO].sort_values(by='shap_value', ascending=True)

                if not fatores.empty:
                    for i, (index, row) in enumerate(fatores.iterrows()):
                        feature_name = row['feature']
                        valor = row['valor']
                        
                        if feature_name in mapeamento_explicacoes:
                            justificativa = mapeamento_explicacoes[feature_name](valor)
                            st.markdown(f"{i+1}. {justificativa.capitalize()} (`{feature_name}`)")
                        else: 
                            st.markdown(f"{i+1}. O parâmetro `{feature_name}` com valor de `{valor:.3f}` foi um fator relevante.")
                else:
                    st.markdown("The prediction for this object was close to the decision threshold, with no individual factors strongly influencing this classification.")
        else:
            st.info("No candidates to analyze.")

    with tab4:
        st.header("Model Performance and Validation")
        st.markdown("""
        The classifier was developed using the Random Forest algorithm, chosen for its robustness and generalization capability. The evaluation was performed through Stratified 5-Fold Cross-Validation, preserving the proportion between the CONFIRMED and FALSE POSITIVE classes. This combination resulted in a reliable and discriminative model, suitable for the binary classification problem.        """)
        mcol1, mcol2 = st.columns(2)
        mcol1.metric("Validation Accuracy", "99.0%")
        mcol2.metric("AUC (ROC)", "0.998")
        st.markdown("---")
        
        gcol1, gcol2 = st.columns(2)
        with gcol1:
            st.subheader("Confusion Matrix")
            st.image('matriz_confusao.png')
            st.markdown("<div style='text-align: center; font-size: small;'>The matrix quantifies the model’s correct and incorrect predictions for each class.</div>", unsafe_allow_html=True)
        with gcol2:
            st.subheader("ROC Curve")
            st.image('curva_roc.png')
            st.markdown("<div style='text-align: center; font-size: small;'>The ROC curve illustrates the classifier's ability to distinguish between classes.</div>", unsafe_allow_html=True)
else:
    st.info("Waiting for a CSV file upload to start analysis.")