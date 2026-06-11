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
    """Motor de Auto-Descoberta: Busca o melhor modelo disponível para a chave fornecida"""
    if not llm_disponivel:
        return None
    try:
        # Pede ao Google a lista de todos os modelos que a sua chave tem permissão para usar
        modelos_disponiveis = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        # Ordem de preferência de modelos (do mais rápido/novo para o mais clássico)
        preferencias = ['models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-1.0-pro', 'models/gemini-pro']
        
        for pref in preferencias:
            if pref in modelos_disponiveis:
                nome_limpo = pref.replace("models/", "")
                return genai.GenerativeModel(nome_limpo)
        
        # Se nenhum da lista de preferência existir, pega o primeiro gerador de texto válido que retornar
        if modelos_disponiveis:
            nome_limpo = modelos_disponiveis[0].replace("models/", "")
            return genai.GenerativeModel(nome_limpo)
            
        return None
    except Exception:
        return None

# DICIONÁRIO DE TRADUÇÃO PARA OS LAUDOS
DICIONARIO_PT = {
    'WaistCirc': 'Circunferência da Cintura (WaistCirc)',
    'BloodGlucose': 'Glicemia de Jejum (BloodGlucose)',
    'Triglycerides': 'Triglicerídeos (Triglycerides)',
    'HDL': 'Lipoproteína de Alta Densidade (HDL)',
    'BMI': 'Índice de Massa Corporal (IMC)',
    'Age': 'Idade',
    'Sex': 'Sexo Biológico',
    'Income': 'Renda Anual',
    'UricAcid': 'Ácido Úrico (UricAcid)',
    'Albuminuria': 'Albuminúria',
    'UrAlbCr': 'Razão Albumina/Creatinina (UrAlbCr)',
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

def traduzir(lista_variaveis):
    return [DICIONARIO_PT.get(var, var) for var in lista_variaveis]

# 2. CARREGAMENTO DOS DADOS
@st.cache_data
def carregar_dados():
    df = pd.read_csv('dataset_app.csv')
    X = df.drop(columns=['MetabolicSyndrome', 'seqn'], errors='ignore')
    y = df['MetabolicSyndrome']
    return df, X, y

# 3. MOTORES DE INTEGRAÇÃO COM IA GENERATIVA (LLM)
def chamar_gemini_global(top_features_pt, nao_classicos_pt):
    model = obter_modelo_gemini()
    if not model:
        return "⚠️ Erro: Nenhum modelo Gemini compatível encontrado ou falha na API Key."
        
    prompt = f"""
    Você é um endocrinologista sênior e especialista em Inteligência Artificial em Saúde.
    Analise o comportamento global de um modelo de Machine Learning treinado para detectar Síndrome Metabólica.
    Com base nas variáveis de maior peso geral que o modelo aprendeu, redija dois pareceres médicos contidos em um único texto:

    1. **🩺 Validação Clínica do Algoritmo (Critérios NCEP-ATP III)**: Avalie se o fato de biomarcadores como {top_features_pt} estarem no topo confere robustez e sentido fisiopatológico à luz das diretrizes tradicionais. (Mencione de forma transparente que a Pressão Arterial foi omitida nesta base por limitações do dataset de origem).
    2. **🧠 Análise Multidimensional (Padrões Complementares)**: Explique o papel de variáveis não-clássicas observadas como influentes ({nao_classicos_pt}), justificando como o algoritmo captura o contínuo do risco metabólico além dos pontos de corte tradicionais de forma holística.

    Regras de Redação:
    - Linguagem formal, técnica, sóbria e consultiva para outros médicos lerem.
    - Evite termos alarmistas. Use negritos do markdown para destacar pontos vitais.
    - Resposta sucinta, estruturada em parágrafos limpos.
    """
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Erro na chamada da API: {str(e)}"

def chamar_gemini_local(prob_shap, alertas_clinicos, fatores_risco_todos, fatores_protecao_todos, fidelidade, tipo_modelo):
    model = obter_modelo_gemini()
    if not model:
        return {
            'laudo': "⚠️ Erro: Nenhum modelo Gemini compatível encontrado ou falha na API Key.",
            'alto_risco': prob_shap >= 50.0,
            'tem_alertas': len(alertas_clinicos) > 0
        }

    status_risco = "RISCO CLÍNICO SIGNIFICATIVO" if prob_shap >= 50.0 else "RISCO CLÍNICO CONTROLADO"
    
    prompt = f"""
    Você é um endocrinologista sênior atuando em um Sistema de Suporte à Decisão Clínica.
    Escreva uma análise médica personalizada para um paciente específico baseado nas métricas de risco e explicabilidade (XAI) do modelo `{tipo_modelo}`.

    DADOS DO PACIENTE:
    - Probabilidade Calculada pelo Modelo: {prob_shap:.1f}% de Risco ({status_risco})
    - Limiares Críticos Tradicionais Ultrapassados (NCEP-ATP III): {alertas_clinicos if alertas_clinicos else 'Nenhum limiar estrito de glicose, cintura, HDL ou triglicerídeos ultrapassado isoladamente.'}
    - Fatores Contribuintes Locais (Variáveis que o SHAP indicou que mais aumentaram o risco): {fatores_risco_todos}
    - Fatores Atenuantes Locais (Variáveis que o SHAP indicou que protegeram/puxaram o risco para baixo): {fatores_protecao_todos}
    - Grau de Concordância Estatística do Gráfico Substituto (Fidelidade LIME): {fidelidade:.1f}%

    Gere uma resposta estruturada contendo estritamente três seções com as seguintes diretrizes:
    1. **📋 Rastreio de Parâmetros Diretos (NCEP-ATP III)**: Descreva textualmente os achados biológicos objetivos do paciente frente aos limites rígidos da diretriz tradicional.
    2. **🤖 Auditoria Multidimensional (Explicabilidade Algorítmica)**: Analise de forma holística como a IA combinou as variáveis contribuintes e atenuantes para chegar no escore final de risco. 
    3. **⚕️ Conduta Médica Sugerida e Confiabilidade XAI**: Sugira uma conduta prudente baseada no nível de risco (MEV, propedêutica clínica como aferição de pressão arterial e exames lipidêmicos complementares). Avalie também a confiabilidade do LIME com base no grau de fidelidade fornecido (se for menor que 85%, alerte o clínico para focar no SHAP Waterfall).

    Regras de Redação:
    - Linguagem extremamente técnica, sóbria, sem termos alarmistas.
    - O texto deve demonstrar honestidade algorítmica, deixando claro que o software sugere risco estatístico e apoia o julgamento soberano do médico.
    - Retorne as seções de forma clara utilizando os títulos propostos em negrito.
    """
    try:
        response = model.generate_content(prompt)
        return {
            'laudo': response.text,
            'alto_risco': prob_shap >= 50.0,
            'tem_alertas': len(alertas_clinicos) > 0
        }
    except Exception as e:
        return {
            'laudo': f"Erro na chamada da API: {str(e)}",
            'alto_risco': prob_shap >= 50.0,
            'tem_alertas': len(alertas_clinicos) > 0
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
    "Escolha qual motor matemático (otimizado pelo GridSearch) guiará as explicações abaixo:",
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

# GERAÇÃO DO LAUDO GLOBAL VIA GEMINI
with st.spinner("O Sistema está gerando o parecer analítico global de diretrizes..."):
    importances = pfi_ativo.importances_mean
    indices = np.argsort(importances)[::-1]
    top_features = [X.columns[i] for i in indices[:6]]
    criterios_ncep = ['WaistCirc', 'BloodGlucose', 'Triglycerides', 'HDL']
    
    alinhados = [f for f in top_features if f in criterios_ncep]
    nao_classicos = [f for f in top_features if f not in criterios_ncep]
    
    laudo_global_texto = chamar_gemini_global(traduzir(alinhados), traduzir(nao_classicos))
    st.info(laudo_global_texto)

st.divider()

# --- SECÇÃO 2: LOCAL (PACIENTE ESPECÍFICO DO DATASET) ---
st.header("2. Interpretação Local (Pacientes do Dataset)")
st.write("Selecione um paciente histórico para entender as variáveis que orientaram o laudo algorítmico.")

def formatar_label_paciente(seqn_val):
    seqn_int = int(seqn_val)
    diag_val = df[df['seqn'] == seqn_val]['MetabolicSyndrome'].values[0]
    return f"Prontuário {seqn_int} - diagnóstico({int(diag_val)})"

pacientes_disponiveis = df['seqn'].unique()
paciente_selecionado = st.selectbox(
    "Selecione o paciente para auditoria:", 
    options=pacientes_disponiveis,
    format_func=formatar_label_paciente
)

idx_paciente = df[df['seqn'] == paciente_selecionado].index[0]
dados_paciente_brutos = X.iloc[[idx_paciente]]
dados_paciente_escalonados = scaler.transform(dados_paciente_brutos)

exp_lime = explainer_lime.explain_instance(
    dados_paciente_escalonados[0], 
    modelo_ativo.predict_proba, 
    num_features=8
)

st.markdown("#### Quadro Clínico de Probabilidades do Paciente Selecionado")
prob_shap = modelo_ativo.predict_proba(dados_paciente_escalonados)[0][1] * 100
try:
    prob_lime = exp_lime.local_pred[0] * 100
    prob_lime = max(0.0, min(100.0, prob_lime))
except:
    prob_lime = prob_shap

erro_lime = abs(prob_shap - prob_lime)
fidelidade_xai = 100.0 - erro_lime

col_prob1, col_prob2, col_prob3 = st.columns(3)
col_prob1.metric(label="Risco Exato (SHAP / Modelo)", value=f"{prob_shap:.1f}%")
col_prob2.metric(label="Aproximação LIME (Substituto)", value=f"{prob_lime:.1f}%")

if fidelidade_xai >= 90.0:
    col_prob3.success(f"Fidelidade da Explicação: {fidelidade_xai:.1f}% (Alta)")
elif fidelidade_xai >= 75.0:
    col_prob3.warning(f"Fidelidade da Explicação: {fidelidade_xai:.1f}% (Média)")
else:
    col_prob3.error(f"Fidelidade da Explicação: {fidelidade_xai:.1f}% (Baixa - Foque no SHAP)")

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

# EXTRAÇÃO DE VARIÁVEIS LOCAIS E ENVIO AO GEMINI (SEÇÃO 2)
if tipo_modelo == "Ensemble (CatBoost)":
    valores_shap = shap_values_ativos[idx_paciente].values if hasattr(shap_values_ativos[idx_paciente], "values") else shap_values_ativos[idx_paciente]
else:
    valores_shap = shap_values_dt[:, :, 1][idx_paciente]

colunas = X.columns.tolist()
idx_positivos = np.argsort(valores_shap)[::-1]
idx_negativos = np.argsort(valores_shap)

fr_todos = traduzir([colunas[i] for i in idx_positivos if valores_shap[i] > 0][:4])
fp_todos = traduzir([colunas[i] for i in idx_negativos if valores_shap[i] < 0][:3])

glicemia = dados_paciente_brutos['BloodGlucose'].values[0] if 'BloodGlucose' in dados_paciente_brutos else None
cintura = dados_paciente_brutos['WaistCirc'].values[0] if 'WaistCirc' in dados_paciente_brutos else None
trig = dados_paciente_brutos['Triglycerides'].values[0] if 'Triglycerides' in dados_paciente_brutos else None
hdl = dados_paciente_brutos['HDL'].values[0] if 'HDL' in dados_paciente_brutos else None

alertas_c = []
if glicemia and glicemia >= 100: alertas_c.append(f"Glicemia de Jejum: {glicemia:.1f} mg/dL")
if cintura and cintura >= 88: alertas_c.append(f"Circunferência da Cintura: {cintura:.1f} cm")
if trig and trig >= 150: alertas_c.append(f"Triglicerídeos: {trig:.1f} mg/dL")
if hdl and hdl < 50: alertas_c.append(f"HDL: {hdl:.1f} mg/dL")

with st.spinner("A IA está gerando o parecer clínico individualizado..."):
    resultado_llm_local = chamar_gemini_local(prob_shap, alertas_c, fr_todos, fp_todos, fidelidade_xai, tipo_modelo)
    
    if resultado_llm_local['alto_risco']:
        st.error(resultado_llm_local['laudo'])
    elif resultado_llm_local['tem_alertas']:
        st.warning(resultado_llm_local['laudo'])
    else:
        st.success(resultado_llm_local['laudo'])

st.divider()

# --- SECÇÃO 3: ENTRADA DE NOVO PACIENTE (FORMULÁRIO MANUAL SIMULADOR) ---
st.header("3. Entrada de Novo Paciente (Simulador Clínico)")
st.write("Insira os parâmetros clínicos e laboratoriais abaixo para gerar um diagnóstico explicável.")

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
        st.warning("⚠️ Atenção: Por favor, preencha todos os campos do formulário antes de gerar o laudo.")
    else:
        encoded_fields = {
            'Marital_Married': 0, 'Marital_Separated': 0, 'Marital_Single': 0, 'Marital_Widowed': 0,
            'Race_Black': 0, 'Race_Hispanic': 0, 'Race_MexAmerican': 0, 'Race_Other': 0, 'Race_White': 0
        }
        if f"Marital_{marital_in}" in encoded_fields: encoded_fields[f"Marital_{marital_in}"] = 1
        if f"Race_{race_in}" in encoded_fields: encoded_fields[f"Race_{race_in}"] = 1

        novo_paciente_dict = {
            'Age': age_in,
            'Sex': 1 if sex_in == "Feminino" else 0,
            'Income': income_in,
            'WaistCirc': waist_in,
            'BMI': bmi_in,
            'Albuminuria': alb_in,
            'UrAlbCr': uralb_in,
            'UricAcid': uric_in,
            'BloodGlucose': blood_in,
            'HDL': hdl_in,
            'Triglycerides': tri_in,
            **encoded_fields
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
            
        erro_new_lime = abs(p_new_shap - p_new_lime)
        fidelidade_new_xai = 100.0 - erro_new_lime

        col_nprob1, col_nprob2, col_nprob3 = st.columns(3)
        col_nprob1.metric(label="Risco pelo Modelo (SHAP)", value=f"{p_new_shap:.1f}%")
        col_nprob2.metric(label="Aproximação Substituto (LIME)", value=f"{p_new_lime:.1f}%")
        
        if fidelidade_new_xai >= 90.0:
            st.success(f"Fidelidade da Explicação: {fidelidade_new_xai:.1f}% (Alta)")
        elif fidelidade_new_xai >= 75.0:
            st.warning(f"Fidelidade da Explicação: {fidelidade_new_xai:.1f}% (Média)")
        else:
            st.error(f"Fidelidade da Explicação: {fidelidade_new_xai:.1f}% (Baixa - Foque no SHAP)")

        col_nplots1, col_nplots2 = st.columns(2)
        with col_nplots1:
            st.markdown("**SHAP Waterfall Plot**")
            if tipo_modelo == "Ensemble (CatBoost)":
                shap_values_new = explainer_ativo(df_novo_escalonado)[0]
            else:
                shap_values_new = explainer_ativo(df_novo_escalonado)[:, :, 1][0]
                
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

        # PARSE DE DADOS DO NOVO PACIENTE E INVOCAÇÃO DO GEMINI (SEÇÃO 3)
        valores_shap_new = shap_values_new.values if hasattr(shap_values_new, "values") else shap_values_new
        idx_pos_new = np.argsort(valores_shap_new)[::-1]
        idx_neg_new = np.argsort(valores_shap_new)

        fr_new = traduzir([colunas[i] for i in idx_pos_new if valores_shap_new[i] > 0][:4])
        fp_new = traduzir([colunas[i] for i in idx_neg_new if valores_shap_new[i] < 0][:3])

        alertas_new = []
        if blood_in and blood_in >= 100: alertas_new.append(f"Glicemia de Jejum: {blood_in:.1f} mg/dL")
        if waist_in and waist_in >= 88: alertas_new.append(f"Circunferência da Cintura: {waist_in:.1f} cm")
        if tri_in and tri_in >= 150: alertas_new.append(f"Triglicerídeos: {tri_in:.1f} mg/dL")
        if hdl_in and hdl_in < 50: alertas_new.append(f"HDL: {hdl_in:.1f} mg/dL")

        with st.spinner("A IA está consolidando o laudo clínico em tempo real..."):
            laudo_simulador = chamar_gemini_local(p_new_shap, alertas_new, fr_new, fp_new, fidelidade_new_xai, tipo_modelo)
            
            if laudo_simulador['alto_risco']:
                st.error(laudo_simulador['laudo'])
            elif laudo_simulador['tem_alertas']:
                st.warning(laudo_simulador['laudo'])
            else:
                st.success(laudo_simulador['laudo'])
