# pipeline.py
import pandas as pd
import joblib
import json
import pickle

try:
    modelo = joblib.load('modelo_random_forest.pkl')
    colunas_modelo = json.load(open('colunas_modelo.json', 'r'))
    valores_imputacao = json.load(open('valores_imputacao.json', 'r'))
    explainer = pickle.load(open('shap_explainer.pkl', 'rb'))
except FileNotFoundError:
    raise RuntimeError("Arquivos de modelo/explicador não encontrados. Execute o script 'preparar_artefatos.py' primeiro.")

def processar_e_prever(df_bruto: pd.DataFrame):
    df_processado = df_bruto.copy()
    avisos = []
    colunas_interesse = colunas_modelo + ['kepoi_name']
    colunas_presentes = [col for col in colunas_interesse if col in df_processado.columns]
    df_processado = df_processado[colunas_presentes]
    colunas_ausentes = set(colunas_modelo) - set(df_processado.columns)
    if colunas_ausentes:
        aviso_cols = f"Colunas ausentes foram preenchidas com valores padrão: {', '.join(sorted(list(colunas_ausentes)))}"
        avisos.append(aviso_cols)
        for col in colunas_ausentes:
            df_processado[col] = valores_imputacao.get(col, 0)
    df_imputacao_check = df_processado[colunas_modelo]
    colunas_com_nan = df_imputacao_check.columns[df_imputacao_check.isnull().any()].tolist()
    if colunas_com_nan:
        aviso_nan = f"Células vazias foram preenchidas com valores padrão nas colunas: {', '.join(sorted(colunas_com_nan))}"
        avisos.append(aviso_nan)
        df_processado.fillna(valores_imputacao, inplace=True)
    if avisos:
        df_processado['Status_Dados'] = 'Imputado'
    else:
        df_processado['Status_Dados'] = 'Completo'
        avisos.append("Análise perfeita: todos os dados estavam completos e no formato esperado.")
    X_final = df_processado[colunas_modelo]
    predicoes_numericas = modelo.predict(X_final)
    scores_confianca = modelo.predict_proba(X_final)
    shap_values = explainer.shap_values(X_final)
    mapeamento_predicao = {0: 'FALSO POSITIVO', 1: 'CONFIRMADO'}
    df_processado['Predicao'] = [mapeamento_predicao[p] for p in predicoes_numericas]
    df_processado['Score_Confianca'] = (scores_confianca.max(axis=1) * 100).round(2).astype(str) + '%'
    colunas_resultado = ['kepoi_name', 'Predicao', 'Score_Confianca', 'Status_Dados', 'koi_depth', 'koi_duration', 'koi_prad', 'koi_teq', 'koi_period']
    colunas_finais_presentes = [col for col in colunas_resultado if col in df_processado.columns]
    df_resultado = df_processado[colunas_finais_presentes]
    return df_resultado, avisos, shap_values, X_final, explainer