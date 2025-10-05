# preparar_artefatos.py
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import roc_curve, confusion_matrix, auc
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import json
import shap
import pickle

print("Iniciando a preparação dos artefatos do modelo...")

# 1. Carregar os dados
print("Carregando o dataset original 'dados.csv'...")
df = pd.read_csv('dados.csv')

# 2. Pré-processamento
print("Aplicando pré-processamento dos dados...")
if 3008 in df.index:
    df = df.drop(3008, axis=0)

colunas_para_remover = [
    "kepid", "kepler_name", "koi_vet_stat", "koi_vet_date", "koi_pdisposition", "koi_score", "koi_disp_prov",
    "koi_comment", "koi_time0bk", "koi_time0bk_err1", "koi_time0bk_err2", "koi_srho", "koi_srho_err1", "koi_srho_err2",
    "koi_fittype", "koi_sma", "koi_limbdark_mod", "koi_parm_prov", "koi_max_sngle_ev", "koi_max_mult_ev", "koi_count",
    "koi_num_transits", "koi_tce_delivname", "koi_quarters", "koi_bin_oedp_sig", "koi_trans_mod", "koi_datalink_dvr",
    "koi_datalink_dvs", "koi_smet", "koi_smet_err1", "koi_smet_err2", "koi_sparprov", "koi_gmag", "koi_rmag",
    "koi_imag", "koi_zmag", "koi_jmag", "koi_hmag", "koi_kmag", "koi_fwm_stat_sig", "koi_fwm_sra", "koi_fwm_sra_err",
    "koi_fwm_sdec", "koi_fwm_sdec_err", "koi_fwm_srao", "koi_fwm_srao_err", "koi_fwm_sdeco", "koi_fwm_sdeco_err",
    "koi_fwm_prao", "koi_fwm_prao_err", "koi_fwm_pdeco", "koi_fwm_pdeco_err", "koi_dicco_mra", "koi_dicco_mra_err",
    "koi_dicco_mdec", "koi_dicco_mdec_err", "koi_dicco_msky", "koi_dicco_msky_err", "koi_dikco_mra", "koi_dikco_mra_err",
    "koi_dikco_mdec", "koi_dikco_mdec_err", "koi_dikco_msky", "koi_dikco_msky_err", "koi_eccen", "koi_ldm_coeff4", "koi_ldm_coeff3"
]
df = df.dropna(axis=1, how="all")
df = df.drop(columns=colunas_para_remover, errors='ignore')

# Lógica de filtragem robusta
df_treino = df[df['koi_disposition'].isin(['CONFIRMED', 'FALSE POSITIVE'])].copy()
mapeamento = {'FALSE POSITIVE': 0, 'CONFIRMED': 1}
df_treino['koi_disposition'] = df_treino['koi_disposition'].map(mapeamento)

X_treino = df_treino.drop(columns=["koi_disposition", "kepoi_name"])
y_treino = df_treino["koi_disposition"]

# 3. Salvar artefatos de imputação e colunas
print("Calculando e salvando valores de imputação e ordem das colunas...")
valores_imputacao = X_treino.mean().to_dict()
with open('valores_imputacao.json', 'w') as f:
    json.dump(valores_imputacao, f, indent=4)
colunas_modelo = X_treino.columns.tolist()
with open('colunas_modelo.json', 'w') as f:
    json.dump(colunas_modelo, f)
X_treino.fillna(valores_imputacao, inplace=True)

# 4. Treinar o modelo RandomForest final
print("Treinando o modelo Random Forest final...")
modelo_rf = RandomForestClassifier(random_state=42)
modelo_rf.fit(X_treino, y_treino)
print("Salvando o modelo em 'modelo_random_forest.pkl'...")
joblib.dump(modelo_rf, 'modelo_random_forest.pkl')

# 5. Criar e salvar o explicador SHAP compatível
print("Criando e salvando o explicador SHAP...")
explainer = shap.TreeExplainer(modelo_rf)
with open('shap_explainer.pkl', 'wb') as f:
    pickle.dump(explainer, f)

# 6. Gerar e salvar as métricas de performance
print("Calculando métricas de performance...")
y_pred_cv = cross_val_predict(modelo_rf, X_treino, y_treino, cv=5)
y_scores_cv = cross_val_predict(modelo_rf, X_treino, y_treino, cv=5, method="predict_proba")[:, 1]

# Matriz de Confusão e Curva ROC
cm = confusion_matrix(y_treino, y_pred_cv)
plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['False Positive', 'Confirmed'], yticklabels=['False Positive', 'Confirmed']); plt.title('Confusion Matrix'); plt.ylabel('True Class'); plt.xlabel('Predicted Class'); plt.savefig('matriz_confusao.png'); plt.close()
fpr, tpr, _ = roc_curve(y_treino, y_scores_cv); roc_auc = auc(fpr, tpr); plt.figure(figsize=(8, 6)); plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC Curve (Area = {roc_auc:0.3f})'); plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--'); plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate'); plt.title('ROC Curve'); plt.legend(loc="lower right"); plt.savefig('curva_roc.png'); plt.close()
# Acurácia
acuracia_cv = cross_val_score(modelo_rf, X_treino, y_treino, cv=5, scoring="accuracy").mean()
print(f"\nAcurácia (Validação Cruzada): {acuracia_cv*100:.1f}%")
print(f"AUC (ROC): {roc_auc:.3f}")

print("\nArtefatos preparados com sucesso!")