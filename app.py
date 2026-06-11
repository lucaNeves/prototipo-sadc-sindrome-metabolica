import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import lime.lime_tabular
import joblib
import google.generativeai as genai

from sklearn.inspection import permutation_importance

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="XAI - Síndrome Metabólica", layout="wide")
st.title("🩺 Diagnóstico de Síndrome Metabólica com XAI")
st.markdown("Sistema Híbrido de Suporte à Decisão Clínica (SHAP, LIME e Permutação).")

# 1.1 CONFIGURAÇÃO DA API E AUTO-DESCOBERTA DE MODELO
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    llm_disponivel = True
except Exception:
    llm_disponivel = False

@st.cache_resource
def obter_modelo_gemini():
    """Busca o melhor modelo disponível para a chave fornecida, priorizando estabilidade"""
    if not llm_disponivel:
        return None
    try:
        modelos_disponiveis = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # Priorizamos o 1.5-flash que possui um limite de 15 requisições/minuto no plano gratuito
        preferencias = ['models/gemini-1.5-flash', 'models/gemini-2.5-flash', 'models/gemini-1.5-pro', 'models/gemini-pro']
        
        for pref in preferencias:
            if pref in modelos_disponiveis:
                nome_limpo = pref.replace("models/", "")
                return genai.GenerativeModel(nome_limpo)
        
        if modelos_disponiveis:
            nome_limpo = modelos_disponiveis[0].replace("models/", "")
            return genai.GenerativeModel(nome_limpo)
            
        return None
    except Exception:
        return None

# DICIONÁRIO DE TRADUÇÃO PARA OS LAUDOS
DICIONARIO_PT = {
    'WaistCirc': 'Circunferência da Cintura',
    'BloodGlucose': 'Glicemia de Jejum',
    'Triglycerides': 'Triglicerídeos',
    'HDL': 'Colesterol HDL',
    'BMI': 'IMC',
    'Age': 'Idade',
    'Sex': 'Sexo Biológico',
    'Income': 'Renda Anual',
    'UricAcid': 'Ácido Úrico',
    'Albuminuria': 'Albuminúria',
    'UrAlbCr': 'Razão Albumina/Creatinina',
    'Marital_Married': 'Estado Civil (Casado)',
    'Marital_Single': 'Estado Civil (Solteiro)',
    'Marital_Separated': 'Estado Civil (Separado)',
    'Marital_Widowed': 'Estado Civil (Viúvo)',
    'Race_Black': 'Raça/Etnia (Negra)',
    'Race_Hispanic': 'Raça/Etnia (Hispânica)',
    'Race_MexAmerican': 'Raça/Etnia (Mexicano-Americana)',
    'Race_White': 'Raça/Etnia (Branca)',
    'Race_Other': 'Raça/Etnia (Outra)'
}

def traduzir_e_juntar(lista_variaveis):
    if not lista_variaveis:
        return "Nenhum fator relevante isolado."
    traduzidas = [DICIONARIO_PT.get(var, var) for var in lista_variaveis]
    return ", ".join([f"**{f}**" for f in traduzidas])

# 2. CARREGAMENTO DOS DADOS E CACHE DE IA
@st.cache_data
def carregar_dados():
    df = pd.read_csv('dataset_app.csv')
    X = df.drop(columns=['MetabolicSyndrome', 'seqn'], errors='ignore')
    y = df['MetabolicSyndrome']
    return df, X, y

# CACHE APLICADO: A IA só gera o texto de novo se as variáveis de entrada mudarem!
@st.cache_data(show_spinner=False, ttl=3600)
def chamar_gemini_global(top_features_str, nao_classicos_str):
    model = obter_modelo_gemini()
    if not model:
        return "⚠️ Erro: Falha de conexão com a API."
        
    prompt = f"""
    Você é um endocrinologista sênior especialista em IA.
    Analise o comportamento global deste modelo de Machine Learning para Síndrome Metabólica.

    1. **🩺 Validação Clínica (NCEP-ATP III)**: Avalie como os biomarcadores clássicos ({top_features_str}) validam a fisiopatologia. (Mencione a ausência da Pressão Arterial na base).
    2. **🧠 Análise Multidimensional**: Explique o papel das variáveis não-clássicas ({nao_classicos_str}) e como a IA avalia o risco holístico.

    Regras: Linguagem técnica, médica, sóbria, máximo de 2 parágrafos diretos. Não invente dados.
    """
    try:
        return model.generate_content(prompt).text
    except Exception as e:
        return f"Erro na chamada da API: {str(e)}"

# CACHE APLICADO: Memoriza laudos de pacientes já analisados
@st.cache_data(show_spinner=False, ttl=3600)
def chamar_gemini_local(prob_shap, status_risco, alertas_str, fatores_risco_str, fatores_protecao_str, fidelidade, tipo_modelo):
    model = obter_modelo_gemini()
    if not model:
        return {
            'laudo': "⚠️ Erro de conexão com o LLM.",
            'alto_risco': prob_shap >= 50.0,
            'tem_alertas': alertas_str != ""
        }

    prompt = f"""
    Como endocrinologista sênior, gere um laudo personalizado baseado nos dados do SADC (Modelo: {tipo_modelo}).

    DADOS:
    - Risco Algorítmico: {prob_shap:.1f}% ({status_risco})
    - Alertas NCEP-ATP III: {alertas_str if alertas_str else 'Nenhum limiar clássico estourou isoladamente.'}
    - Agravantes (Aumentaram o Risco): {fatores_risco_str}
    - Atenuantes (Reduziram o Risco): {fatores_protecao_str}
    - Fidelidade da Explicação LIME: {fidelidade:.1f}%

    Redija três seções curtas e diretas:
    1. **📋 Rastreio de Parâmetros Diretos (NCEP-ATP III)**: Foque nos alertas clássicos.
    2. **🤖 Auditoria Multidimensional**: Como a IA ponderou agravantes e atenuantes.
    3. **⚕️ Conduta Médica e Confiabilidade XAI**: Sugira conduta prudente (MEV, propedêutica complementar). Avalie a fidelidade (se < 85%, alerte para focar no SHAP Waterfall).
    """
    try:
        return {
            'laudo': model.generate_content(prompt).text,
            'alto_risco': prob_shap >= 50.0,
            'tem_alertas': alertas_str != ""
        }
    except Exception as e:
        return {
            'laudo': f"Erro de processamento da IA: {str(e)}",
            'alto_risco': prob_shap >= 50.0,
            'tem_alertas': alertas_str != ""
        }

# 4. PREPARAÇÃO DOS MODELOS E EXPLICADORES
@st.cache_resource
def preparar_modelos_e_xai(_X, _y):
    scaler = joblib.load('scaler.pkl')
    modelo_dt = joblib.load('modelo_dt.pkl')
    modelo_cat = joblib.load('modelo_cat.pkl')

    X_scaled = scaler.transform(_X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=_X.columns)

    explainer_dt = shap.TreeExplainer(modelo_dt)
    shap_values_dt = explainer_dt(X_scaled_df)
    shap_values_dt.data = _X.values 
    pfi_dt = permutation_importance(modelo_dt, X_scaled_df, _y, n_repeats=5, random_state=42)

    explainer_cat = shap.TreeExplainer(modelo_cat)
    shap_values_cat = explainer_cat(X_scaled_df)
    shap_values_cat.data = _X.values 
    pfi_cat = permutation_importance(modelo_cat, X_scaled_df, _y, n_repeats=5, random_state=42)

    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        X_scaled_df.values,
        feature_names=_X.columns.tolist(),
        class_names=['Baixo Risco', 'Alto Risco'],
        mode='classification',
        random_state=42
    )

    return scaler, modelo_dt, explainer_dt, shap_values_dt, pfi_dt, modelo_cat, explainer_cat, shap_values_cat, pfi_cat, explainer_lime

df, X, y = carregar_dados()
scaler, modelo_dt, explainer_dt, shap_values_dt, pfi_dt, modelo_cat, explainer_cat, shap_values_cat, pfi_cat, explainer_lime = preparar_modelos_e_xai(X, y)

st.divider()

# 5. INTERFACE DE SELEÇÃO DE MODELOS
st.subheader("⚙️ Modelo Ativo para Auditoria Explicável (XAI)")
tipo_modelo = st.selectbox(
    "Escolha qual motor matemático guiará as explicações abaixo:",
    ["Ensemble (CatBoost)", "Clássico (Árvore de Decisão)"]
)

if tipo_modelo == "Ensemble (CatBoost)":
    modelo_ativo = modelo_cat
    explainer_ativo = explainer_cat
    shap_values_ativos = shap_values_cat
    pfi_ativo = pfi_cat
else:
    modelo_ativo = modelo_dt
    explainer_ativo = explainer_dt
    shap_values_ativos = shap_values_dt[:, :, 1]
    pfi_ativo = pfi_dt

st.divider()

# --- SECÇÃO 1: GLOBAL ---
st.header(f"1. Interpretação Global ({tipo_modelo})")
st.write("Análise macroscópica de quais exames guiam o modelo e o impacto de cada variável.")

col_global_1, col_global_2 = st.columns(2)
with col_global_1:
    st.markdown("**SHAP (Impacto Marginal)**")
    fig_global, ax_global = plt.subplots(figsize=(6, 4))
    shap.summary_plot(shap_values_ativos, X, show=False)
    st.pyplot(fig_global)
with col_global_2:
    st.markdown("**Permutation Feature Importance (PFI)**")
    sorted_idx = pfi_ativo.importances_mean.argsort()
    fig_pfi, ax_pfi = plt.subplots(figsize=(6, 4))
    ax_pfi.barh(X.columns[sorted_idx], pfi_ativo.importances_mean[sorted_idx], color='#1f77b4')
    ax_pfi.set_xlabel("Queda Média na Performance (Importância)")
    fig_pfi.tight_layout()
    st.pyplot(fig_pfi)

# GERAÇÃO DO LAUDO GLOBAL (AGORA COM CACHE)
with st.spinner("O Sistema está gerando o parecer analítico global..."):
    importances = pfi_ativo.importances_mean
    indices = np.argsort(importances)[::-1]
    top_features = [X.columns[i] for i in indices[:6]]
    criterios_ncep = ['WaistCirc', 'BloodGlucose', 'Triglycerides', 'HDL']
    
    alinhados = [f for f in top_features if f in criterios_ncep]
    nao_classicos = [f for f in top_features if f not in criterios_ncep]
    
    str_alinhados = traduzir_e_juntar(alinhados)
    str_nao_classicos = traduzir_e_juntar(nao_classicos)
    
    laudo_global_texto = chamar_gemini_global(str_alinhados, str_nao_classicos)
    st.info(laudo_global_texto)

st.divider()

# --- SECÇÃO 2: LOCAL (PACIENTE ESPECÍFICO) ---
st.header("2. Interpretação Local (Pacientes do Dataset)")
st.write("Selecione um paciente histórico para entender as variáveis que orientaram o laudo algorítmico.")

def formatar_label_paciente(seqn_val):
    seqn_int = int(seqn_val)
    diag_val = df[df['seqn'] == seqn_val]['MetabolicSyndrome'].values[0]
    return f"Prontuário {seqn_int} - diagnóstico({int(diag_val)})"

paciente_selecionado = st.selectbox(
    "Selecione o paciente para auditoria:", 
    options=df['seqn'].unique(),
    format_func=formatar_label_paciente
)

idx_paciente = df[df['seqn'] == paciente_selecionado].index[0]
dados_paciente_brutos = X.iloc[[idx_paciente]]
dados_paciente_escalonados = scaler.transform(dados_paciente_brutos)

exp_lime = explainer_lime.explain_instance(dados_paciente_escalonados[0], modelo_ativo.predict_proba, num_features=8)
prob_shap = modelo_ativo.predict_proba(dados_paciente_escalonados)[0][1] * 100

try:
    prob_lime = max(0.0, min(100.0, exp_lime.local_pred[0] * 100))
except:
    prob_lime = prob_shap

fidelidade_xai = 100.0 - abs(prob_shap - prob_lime)

st.markdown("#### Quadro Clínico de Probabilidades")
col_prob1, col_prob2, col_prob3 = st.columns(3)
col_prob1.metric(label="Risco Exato (SHAP / Modelo)", value=f"{prob_shap:.1f}%")
col_prob2.metric(label="Aproximação LIME", value=f"{prob_lime:.1f}%")

if fidelidade_xai >= 90.0:
    col_prob3.success(f"Fidelidade XAI: {fidelidade_xai:.1f}% (Alta)")
elif fidelidade_xai >= 75.0:
    col_prob3.warning(f"Fidelidade XAI: {fidelidade_xai:.1f}% (Média)")
else:
    col_prob3.error(f"Fidelidade XAI: {fidelidade_xai:.1f}% (Baixa)")

col_local_1, col_local_2 = st.columns(2)
with col_local_1:
    st.markdown("**Justificativa Exata (SHAP Waterfall)**")
    fig_local, ax_local = plt.subplots(figsize=(5, 4))
    shap.plots.waterfall(shap_values_ativos[idx_paciente], show=False)
    st.pyplot(fig_local)
with col_local_2:
    st.markdown("**Justificativa Linear Local (LIME)**")
    fig_lime = exp_lime.as_pyplot_figure()
    fig_lime.set_size_inches(5, 4)
    fig_lime.tight_layout()
    st.pyplot(fig_lime)

# EXTRAÇÃO E FORMATAÇÃO DE STRINGS PARA O CACHE
if tipo_modelo == "Ensemble (CatBoost)":
    valores_shap = shap_values_ativos[idx_paciente].values if hasattr(shap_values_ativos[idx_paciente], "values") else shap_values_ativos[idx_paciente]
else:
    valores_shap = shap_values_dt[:, :, 1][idx_paciente]

colunas = X.columns.tolist()
idx_positivos = np.argsort(valores_shap)[::-1]
idx_negativos = np.argsort(valores_shap)

str_fr_todos = traduzir_e_juntar([colunas[i] for i in idx_positivos if valores_shap[i] > 0][:4])
str_fp_todos = traduzir_e_juntar([colunas[i] for i in idx_negativos if valores_shap[i] < 0][:3])

glicemia = dados_paciente_brutos['BloodGlucose'].values[0] if 'BloodGlucose' in dados_paciente_brutos else None
cintura = dados_paciente_brutos['WaistCirc'].values[0] if 'WaistCirc' in dados_paciente_brutos else None
trig = dados_paciente_brutos['Triglycerides'].values[0] if 'Triglycerides' in dados_paciente_brutos else None
hdl = dados_paciente_brutos['HDL'].values[0] if 'HDL' in dados_paciente_brutos else None

alertas_lista = []
if glicemia and glicemia >= 100: alertas_lista.append(f"Glicemia: {glicemia:.1f}")
if cintura and cintura >= 88: alertas_lista.append(f"Cintura: {cintura:.1f}")
if trig and trig >= 150: alertas_lista.append(f"Triglicerídeos: {trig:.1f}")
if hdl and hdl < 50: alertas_lista.append(f"HDL: {hdl:.1f}")
str_alertas = ", ".join(alertas_lista)
status_risco = "RISCO SIGNIFICATIVO" if prob_shap >= 50.0 else "RISCO CONTROLADO"

with st.spinner("A IA está gerando o parecer individualizado..."):
    # Graças ao cache, se o médico voltar para este paciente, a resposta será instantânea!
    resultado_llm_local = chamar_gemini_local(prob_shap, status_risco, str_alertas, str_fr_todos, str_fp_todos, fidelidade_xai, tipo_modelo)
    
    if resultado_llm_local['alto_risco']:
        st.error(resultado_llm_local['laudo'])
    elif resultado_llm_local['tem_alertas']:
        st.warning(resultado_llm_local['laudo'])
    else:
        st.success(resultado_llm_local['laudo'])

st.divider()

# --- SECÇÃO 3: SIMULADOR CLÍNICO ---
st.header("3. Entrada de Novo Paciente (Simulador Clínico)")
st.write("Insira os parâmetros abaixo para gerar um diagnóstico explicável dinâmico.")

with st.form("form_novo_paciente"):
    inp_col1, inp_col2, inp_col3, inp_col4 = st.columns(4)
    with inp_col1:
        age_in = st.number_input("Idade (Anos):", min_value=0, max_value=120, value=None)
        sex_in = st.selectbox("Sexo Biológico:", ["Masculino", "Feminino"], index=None)
        marital_in = st.selectbox("Estado Civil:", ["Married", "Single", "Separated", "Widowed", "Divorced/Other"], index=None)
    with inp_col2:
        income_in = st.number_input("Renda Anual (USD):", min_value=0, max_value=500000, value=None, step=1000)
        race_in = st.selectbox("Raça/Etnia:", ["Asian", "Black", "Hispanic", "MexAmerican", "White", "Other"], index=None)
        waist_in = st.number_input("Cintura (cm):", min_value=30.0, max_value=200.0, value=None, step=0.5)
    with inp_col3:
        bmi_in = st.number_input("IMC:", min_value=10.0, max_value=80.0, value=None, step=0.1)
        blood_in = st.number_input("Glicemia de Jejum (mg/dL):", min_value=30.0, max_value=500.0, value=None, step=1.0)
        hdl_in = st.number_input("Colesterol HDL (mg/dL):", min_value=5.0, max_value=150.0, value=None, step=1.0)
    with inp_col4:
        tri_in = st.number_input("Triglicerídeos (mg/dL):", min_value=10.0, max_value=1000.0, value=None, step=1.0)
        uric_in = st.number_input("Ácido Úrico (mg/dL):", min_value=1.0, max_value=20.0, value=None, step=0.1)
        alb_in = st.selectbox("Albuminúria (Grau):", [0, 1, 2], index=None)
        uralb_in = st.number_input("Razão Alb/Cr (mg/g):", min_value=0.0, max_value=5000.0, value=None, step=1.0)

    submitted = st.form_submit_button("Gerar Diagnóstico e XAI", type="primary")

if submitted:
    variaveis_entrada = [age_in, sex_in, marital_in, income_in, race_in, waist_in, bmi_in, blood_in, hdl_in, tri_in, uric_in, alb_in, uralb_in]
    if None in variaveis_entrada:
        st.warning("⚠️ Preencha todos os campos do formulário antes de gerar o laudo.")
    else:
        encoded_fields = {
            'Marital_Married': 0, 'Marital_Separated': 0, 'Marital_Single': 0, 'Marital_Widowed': 0,
            'Race_Black': 0, 'Race_Hispanic': 0, 'Race_MexAmerican': 0, 'Race_Other': 0, 'Race_White': 0
        }
        if f"Marital_{marital_in}" in encoded_fields: encoded_fields[f"Marital_{marital_in}"] = 1
        if f"Race_{race_in}" in encoded_fields: encoded_fields[f"Race_{race_in}"] = 1

        novo_paciente_dict = {
            'Age': age_in, 'Sex': 1 if sex_in == "Feminino" else 0, 'Income': income_in,
            'WaistCirc': waist_in, 'BMI': bmi_in, 'Albuminuria': alb_in, 'UrAlbCr': uralb_in,
            'UricAcid': uric_in, 'BloodGlucose': blood_in, 'HDL': hdl_in, 'Triglycerides': tri_in, **encoded_fields
        }

        df_novo_bruto = pd.DataFrame([novo_paciente_dict])[X.columns]
        df_novo_escalonado = scaler.transform(df_novo_bruto)

        st.markdown("### 📊 Laudo e Auditoria do Novo Paciente")
        p_new_shap = modelo_ativo.predict_proba(df_novo_escalonado)[0][1] * 100
        
        exp_lime_new = explainer_lime.explain_instance(df_novo_escalonado[0], modelo_ativo.predict_proba, num_features=8)
        try:
            p_new_lime = max(0.0, min(100.0, exp_lime_new.local_pred[0] * 100))
        except:
            p_new_lime = p_new_shap
            
        fidelidade_new_xai = 100.0 - abs(p_new_shap - p_new_lime)

        col_nprob1, col_nprob2, col_nprob3 = st.columns(3)
        col_nprob1.metric(label="Risco pelo Modelo (SHAP)", value=f"{p_new_shap:.1f}%")
        col_nprob2.metric(label="Aproximação Substituto (LIME)", value=f"{p_new_lime:.1f}%")
        
        if fidelidade_new_xai >= 90.0:
            st.success(f"Fidelidade XAI: {fidelidade_new_xai:.1f}% (Alta)")
        else:
            st.warning(f"Fidelidade XAI: {fidelidade_new_xai:.1f}%")

        col_nplots1, col_nplots2 = st.columns(2)
        with col_nplots1:
            st.markdown("**SHAP Waterfall Plot**")
            shap_values_new = explainer_ativo(df_novo_escalonado)[0] if tipo_modelo == "Ensemble (CatBoost)" else explainer_ativo(df_novo_escalonado)[:, :, 1][0]
            shap_values_new.data = df_novo_bruto.values[0]
            fig_new_shap, ax_new_shap = plt.subplots(figsize=(5, 4))
            shap.plots.waterfall(shap_values_new, show=False)
            st.pyplot(fig_new_shap)
            
        with col_nplots2:
            st.markdown("**LIME Explanation Plot**")
            fig_new_lime = exp_lime_new.as_pyplot_figure()
            fig_new_lime.set_size_inches(5, 4)
            fig_new_lime.tight_layout()
            st.pyplot(fig_new_lime)

        # PARSE DE DADOS DO SIMULADOR PARA O GEMINI
        valores_shap_new = shap_values_new.values if hasattr(shap_values_new, "values") else shap_values_new
        idx_pos_new = np.argsort(valores_shap_new)[::-1]
        idx_neg_new = np.argsort(valores_shap_new)

        str_fr_new = traduzir_e_juntar([colunas[i] for i in idx_pos_new if valores_shap_new[i] > 0][:4])
        str_fp_new = traduzir_e_juntar([colunas[i] for i in idx_neg_new if valores_shap_new[i] < 0][:3])

        alertas_new_list = []
        if blood_in and blood_in >= 100: alertas_new_list.append(f"Glicemia: {blood_in:.1f}")
        if waist_in and waist_in >= 88: alertas_new_list.append(f"Cintura: {waist_in:.1f}")
        if tri_in and tri_in >= 150: alertas_new_list.append(f"Triglicerídeos: {tri_in:.1f}")
        if hdl_in and hdl_in < 50: alertas_new_list.append(f"HDL: {hdl_in:.1f}")
        str_alertas_new = ", ".join(alertas_new_list)
        status_risco_new = "RISCO SIGNIFICATIVO" if p_new_shap >= 50.0 else "RISCO CONTROLADO"

        with st.spinner("A IA está consolidando o laudo clínico..."):
            laudo_simulador = chamar_gemini_local(p_new_shap, status_risco_new, str_alertas_new, str_fr_new, str_fp_new, fidelidade_new_xai, tipo_modelo)
            
            if laudo_simulador['alto_risco']:
                st.error(laudo_simulador['laudo'])
            elif laudo_simulador['tem_alertas']:
                st.warning(laudo_simulador['laudo'])
            else:
                st.success(laudo_simulador['laudo'])
