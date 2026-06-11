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
    if not llm_disponivel:
        return None
    try:
        modelos_disponiveis = [m.name for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        
        preferencias = ['models/gemini-2.5-flash', 'models/gemini-1.5-flash', 'models/gemini-1.5-pro', 'models/gemini-pro']
        
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

def traduzir(lista_variaveis):
    return [DICIONARIO_PT.get(var, var) for var in lista_variaveis]

def traduzir_e_juntar(lista_variaveis):
    if not lista_variaveis:
        return "Nenhum fator relevante isolado."
    traduzidas = [DICIONARIO_PT.get(var, var) for var in lista_variaveis]
    return ", ".join([f"**{f}**" for f in traduzidas])

# 2. CARREGAMENTO DOS DADOS
@st.cache_data
def carregar_dados():
    df = pd.read_csv('dataset_app.csv')
    X = df.drop(columns=['MetabolicSyndrome', 'seqn'], errors='ignore')
    y = df['MetabolicSyndrome']
    return df, X, y

# ========================================================
# MOTORES DE TEXTO CLÁSSICO (BASEADO EM REGRAS ESTATÍSTICAS)
# ========================================================
def gerar_laudo_global_classico(pfi_ativo, feature_names):
    importances = pfi_ativo.importances_mean
    indices = np.argsort(importances)[::-1]
    top_features = [feature_names[i] for i in indices[:6]] 

    criterios_ncep = ['WaistCirc', 'BloodGlucose', 'Triglycerides', 'HDL']
    alinhados = [f for f in top_features if f in criterios_ncep]
    nao_classicos = [f for f in top_features if f not in criterios_ncep]
    
    alinhados_pt = traduzir(alinhados)
    nao_classicos_pt = traduzir(nao_classicos)

    texto_classico = f"**🩺 Validação Clínica do Algoritmo (Critérios NCEP-ATP III)**\nA análise de importância global evidencia que o modelo prioriza preditores fortemente alinhados à fisiopatologia da Síndrome Metabólica. Biomarcadores como {', '.join([f'**{f}**' for f in alinhados_pt])} exercem a maior influência preditiva. Isto confere robustez à estratificação de risco.\n*(Nota de Transparência: A Pressão Arterial não compõe a matriz devido a limitações de recolha no dataset original).*"
    
    texto_holistico = f"**🧠 Análise Multidimensional (Padrões Complementares Identificados)**\nAlém dos critérios tradicionais, o algoritmo identifica fatores de forma integrada. Variáveis como {', '.join([f'**{f}**' for f in nao_classicos_pt])} apresentaram impacto significativo na estratificação. Isto indica que o sistema avalia o contínuo metabólico do indivíduo, rastreando vulnerabilidades de forma holística."
    
    return {'classico': texto_classico, 'holistico': texto_holistico}

def gerar_laudo_local_classico(dados_brutos, prob_shap, prob_lime, fidelidade, shap_values_paciente, tipo_modelo):
    colunas = dados_brutos.columns.tolist()
    valores_shap = shap_values_paciente.values if hasattr(shap_values_paciente, "values") else shap_values_paciente

    idx_positivos = np.argsort(valores_shap)[::-1]
    idx_negativos = np.argsort(valores_shap)

    fatores_risco_todos = traduzir([colunas[i] for i in idx_positivos if valores_shap[i] > 0][:4])
    fatores_protecao_todos = traduzir([colunas[i] for i in idx_negativos if valores_shap[i] < 0][:3])

    alto_risco = prob_shap >= 50.0
    status_diag = "RISCO CLÍNICO SIGNIFICATIVO" if alto_risco else "RISCO CLÍNICO CONTROLADO"

    glicemia = dados_brutos['BloodGlucose'].values[0] if 'BloodGlucose' in dados_brutos else None
    cintura = dados_brutos['WaistCirc'].values[0] if 'WaistCirc' in dados_brutos else None
    trig = dados_brutos['Triglycerides'].values[0] if 'Triglycerides' in dados_brutos else None
    hdl = dados_brutos['HDL'].values[0] if 'HDL' in dados_brutos else None

    alertas_clinicos = []
    if glicemia and glicemia >= 100: alertas_clinicos.append(f"Glicemia de Jejum ({glicemia:.1f} mg/dL)")
    if cintura and cintura >= 88: alertas_clinicos.append(f"Circunferência da Cintura ({cintura:.1f} cm)")
    if trig and trig >= 150: alertas_clinicos.append(f"Triglicerídeos ({trig:.1f} mg/dL)")
    if hdl and hdl < 50: alertas_clinicos.append(f"HDL ({hdl:.1f} mg/dL)")
    
    tem_alertas = len(alertas_clinicos) > 0

    texto_classico = f"**📋 Rastreio de Parâmetros Diretos (NCEP-ATP III)**\n**Alterações Identificadas:** {', '.join(alertas_clinicos) if tem_alertas else 'Nenhum dos limiares críticos monitorados (Glicose, Cintura, HDL ou Triglicerídeos) foi ultrapassado de forma isolada segundo a diretriz.'}"

    texto_ia = f"**🤖 Estratificação de Risco Algorítmica (Auditoria SHAP)**\nO modelo calculou uma probabilidade de **{prob_shap:.1f}%** para enquadramento do paciente na Síndrome Metabólica (**{status_diag}**).\n* **Fatores Contribuintes (Elevam o risco):** {', '.join([f'**{f}**' for f in fatores_risco_todos])}.\n* **Fatores Atenuantes (Reduzem o risco):** {', '.join([f'**{f}**' for f in fatores_protecao_todos])}."

    texto_conduta = f"**⚕️ Apoio à Decisão e Nível de Confiabilidade do Sistema**\n{'**Sugestão de Conduta:** Perfil de alto risco metabólico sistémico. Recomenda-se correlação clínica, aprofundamento da propedêutica e intervenções de MEV.' if alto_risco else '**Sugestão de Conduta:** Perfil sugere estabilidade metabólica. Recomenda-se seguimento clínico de rotina.'}\n*Auditoria (LIME): A concordância explicativa local é de {fidelidade:.1f}%.* {'Isto indica alta confiabilidade na leitura das variáveis.' if fidelidade >= 85.0 else 'Devido a interações não-lineares, a explicação simplificada deve ser lida com cautela; priorize o gráfico SHAP Waterfall.'}"

    return {'classico': texto_classico, 'ia': texto_ia, 'conduta': texto_conduta, 'tem_alertas': tem_alertas, 'alto_risco': alto_risco}

# ========================================================
# MOTORES DE IA GENERATIVA COM CACHE (LLM GEMINI)
# ========================================================
@st.cache_data(show_spinner=False, ttl=3600)
def chamar_gemini_global(top_features_str, nao_classicos_str):
    model = obter_modelo_gemini()
    if not model:
        return "⚠️ Erro: Falha de conexão com a API do Gemini. Verifique as configurações (Secrets)."
        
    prompt = f"""
    Você é um endocrinologista sênior especialista em IA. Analise o comportamento global deste modelo de Machine Learning para Síndrome Metabólica.
    1. **🩺 Validação Clínica (NCEP-ATP III)**: Avalie como os biomarcadores clássicos ({top_features_str}) validam a fisiopatologia. (Mencione a ausência da Pressão Arterial na base).
    2. **🧠 Análise Multidimensional**: Explique o papel das variáveis não-clássicas ({nao_classicos_str}) e como a IA avalia o risco holístico.
    
    DIRETRIZES DE FORMATAÇÃO ESTRITAS (LEIA COM ATENÇÃO):
    - Linguagem técnica, médica, sóbria, máximo de 2 parágrafos diretos.
    - Escreva 100% em Português (PT-BR).
    - RETORNE APENAS O TEXTO FINAL DO LAUDO. É ESTRITAMENTE PROIBIDO incluir rascunhos (Drafts), anotações do seu raciocínio ("Role", "Task", "Check"), ou repetir o prompt. Vá direto para o texto médico.
    """
    try:
        return model.generate_content(prompt).text
    except Exception as e:
        erro_str = str(e)
        if "429" in erro_str or "Quota" in erro_str:
            return "⏳ **Sistema em Resfriamento.** O limite de laudos simultâneos foi atingido. Aguarde cerca de 60 segundos e tente novamente."
        return f"⚠️ Erro na chamada da API: {erro_str}"

@st.cache_data(show_spinner=False, ttl=3600)
def chamar_gemini_local(prob_shap, status_risco, alertas_str, fatores_risco_str, fatores_protecao_str, fidelidade, tipo_modelo):
    model = obter_modelo_gemini()
    if not model:
        return {'laudo': "⚠️ Erro de conexão com o LLM.", 'alto_risco': prob_shap >= 50.0, 'tem_alertas': alertas_str != ""}

    prompt = f"""
    Como endocrinologista sênior, gere um laudo personalizado baseado nos dados do SADC (Modelo: {tipo_modelo}).
    DADOS:
    - Risco Algorítmico: {prob_shap:.1f}% ({status_risco})
    - Alertas NCEP-ATP III: {alertas_str if alertas_str else 'Nenhum limiar clássico isolado.'}
    - Agravantes: {fatores_risco_str}
    - Atenuantes: {fatores_protecao_str}
    - Fidelidade LIME: {fidelidade:.1f}%

    Redija três seções curtas e diretas:
    1. **📋 Rastreio de Parâmetros Diretos (NCEP-ATP III)**: Foque nos alertas clássicos.
    2. **🤖 Auditoria Multidimensional**: Como a IA ponderou agravantes e atenuantes.
    3. **⚕️ Conduta Médica e Confiabilidade XAI**: Sugira conduta prudente. Avalie a fidelidade LIME (se < 85%, avise para focar no SHAP Waterfall).

    DIRETRIZES DE FORMATAÇÃO ESTRITAS (LEIA COM ATENÇÃO):
    - Escreva 100% em Português (PT-BR).
    - RETORNE APENAS O TEXTO FINAL DAS 3 SEÇÕES. 
    - É ESTRITAMENTE PROIBIDO incluir blocos de raciocínio (Role, Task, Draft, Check) ou introduções antes do laudo. Inicie a sua resposta diretamente com o título da primeira seção.
    """
    try:
        return {'laudo': model.generate_content(prompt).text, 'alto_risco': prob_shap >= 50.0, 'tem_alertas': alertas_str != ""}
    except Exception as e:
        erro_str = str(e)
        if "429" in erro_str or "Quota" in erro_str:
            mensagem = "⏳ **Sistema em Resfriamento (Proteção Anti-Spam).** O limite de requisições do Gemini foi atingido. Aguarde e clique novamente."
        else:
            mensagem = f"⚠️ Erro de processamento da IA: {erro_str}"
        return {'laudo': mensagem, 'alto_risco': prob_shap >= 50.0, 'tem_alertas': alertas_str != ""}

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
        X_scaled_df.values, feature_names=_X.columns.tolist(), class_names=['Baixo Risco', 'Alto Risco'],
        mode='classification', random_state=42
    )

    return scaler, modelo_dt, explainer_dt, shap_values_dt, pfi_dt, modelo_cat, explainer_cat, shap_values_cat, pfi_cat, explainer_lime

df, X, y = carregar_dados()
scaler, modelo_dt, explainer_dt, shap_values_dt, pfi_dt, modelo_cat, explainer_cat, shap_values_cat, pfi_cat, explainer_lime = preparar_modelos_e_xai(X, y)

st.divider()

# 5. INTERFACE DE SELEÇÃO DE MODELOS
st.subheader("⚙️ Modelo Ativo para Auditoria Explicável (XAI)")
tipo_modelo = st.selectbox("Escolha qual motor matemático guiará as explicações abaixo:", ["Ensemble (CatBoost)", "Clássico (Árvore de Decisão)"])

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

aba1_trad, aba1_ia = st.tabs(["📝 Laudo Tradicional (Regras Estatísticas)", "✨ Laudo IA Generativa (Gemini)"])

with aba1_trad:
    laudos_globais_classicos = gerar_laudo_global_classico(pfi_ativo, X.columns.tolist())
    st.success(laudos_globais_classicos['classico'])  
    st.info(laudos_globais_classicos['holistico'])    

with aba1_ia:
    if st.button("✨ Gerar Parecer Analítico Global com IA", key="btn_ia_global"):
        with st.spinner("A IA está analisando os dados em cascata para encontrar um servidor livre..."):
            importances = pfi_ativo.importances_mean
            indices = np.argsort(importances)[::-1]
            top_features = [X.columns[i] for i in indices[:6]]
            criterios_ncep = ['WaistCirc', 'BloodGlucose', 'Triglycerides', 'HDL']
            
            alinhados = [f for f in top_features if f in criterios_ncep]
            nao_classicos = [f for f in top_features if f not in criterios_ncep]
            
            laudo_global_texto = chamar_gemini_global(traduzir_e_juntar(alinhados), traduzir_e_juntar(nao_classicos))
            
            if "⏳" in laudo_global_texto:
                st.warning(laudo_global_texto)
            else:
                st.info(laudo_global_texto)
    else:
        st.info("👆 Clique no botão acima para solicitar a análise global por IA Generativa.")

st.divider()

# --- SECÇÃO 2: LOCAL (PACIENTE ESPECÍFICO) ---
st.header("2. Interpretação Local (Pacientes do Dataset)")
st.write("Selecione um paciente histórico para entender as variáveis que orientaram o laudo algorítmico.")

def formatar_label_paciente(seqn_val):
    seqn_int = int(seqn_val)
    diag_val = df[df['seqn'] == seqn_val]['MetabolicSyndrome'].values[0]
    return f"Prontuário {seqn_int} - diagnóstico({int(diag_val)})"

paciente_selecionado = st.selectbox("Selecione o paciente para auditoria:", options=df['seqn'].unique(), format_func=formatar_label_paciente)

idx_paciente = df[df['seqn'] == paciente_selecionado].index[0]
dados_paciente_brutos = X.iloc[[idx_paciente]]
dados_paciente_escalonados = scaler.transform(dados_paciente_brutos)

exp_lime = explainer_lime.explain_instance(dados_paciente_escalonados[0], modelo_ativo.predict_proba, num_features=8)
prob_shap = modelo_ativo.predict_proba(dados_paciente_escalonados)[0][1] * 100

try: prob_lime = max(0.0, min(100.0, exp_lime.local_pred[0] * 100))
except: prob_lime = prob_shap

fidelidade_xai = 100.0 - abs(prob_shap - prob_lime)

st.markdown("#### Quadro Clínico de Probabilidades")
col_prob1, col_prob2, col_prob3 = st.columns(3)
col_prob1.metric(label="Risco Exato (SHAP / Modelo)", value=f"{prob_shap:.1f}%")
col_prob2.metric(label="Aproximação LIME", value=f"{prob_lime:.1f}%")

if fidelidade_xai >= 90.0: col_prob3.success(f"Fidelidade XAI: {fidelidade_xai:.1f}% (Alta)")
elif fidelidade_xai >= 75.0: col_prob3.warning(f"Fidelidade XAI: {fidelidade_xai:.1f}% (Média)")
else: col_prob3.error(f"Fidelidade XAI: {fidelidade_xai:.1f}% (Baixa)")

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

valores_shap = shap_values_ativos[idx_paciente].values if hasattr(shap_values_ativos[idx_paciente], "values") else shap_values_ativos[idx_paciente] if tipo_modelo == "Ensemble (CatBoost)" else shap_values_dt[:, :, 1][idx_paciente]

aba2_trad, aba2_ia = st.tabs(["📝 Laudo Tradicional (Regras Estatísticas)", "✨ Laudo IA Generativa (Gemini)"])

with aba2_trad:
    laudos_locais_classico = gerar_laudo_local_classico(dados_paciente_brutos, prob_shap, prob_lime, fidelidade_xai, valores_shap, tipo_modelo)
    if laudos_locais_classico['tem_alertas']: st.warning(laudos_locais_classico['classico'])
    else: st.success(laudos_locais_classico['classico'])
    st.info(laudos_locais_classico['ia'])
    if laudos_locais_classico['alto_risco']: st.error(laudos_locais_classico['conduta'])
    else: st.success(laudos_locais_classico['conduta'])

with aba2_ia:
    if st.button("✨ Gerar Parecer Clínico Individual com IA", key=f"btn_ia_local_{paciente_selecionado}"):
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

        with st.spinner("A IA está buscando um servidor com cota disponível para o laudo..."):
            resultado_llm_local = chamar_gemini_local(prob_shap, status_risco, str_alertas, str_fr_todos, str_fp_todos, fidelidade_xai, tipo_modelo)
            
            if "⏳" in resultado_llm_local['laudo']:
                st.warning(resultado_llm_local['laudo'])
            else:
                if resultado_llm_local['alto_risco']: st.error(resultado_llm_local['laudo'])
                elif resultado_llm_local['tem_alertas']: st.warning(resultado_llm_local['laudo'])
                else: st.success(resultado_llm_local['laudo'])
    else:
         st.info("👆 Clique no botão acima para submeter os exames deste paciente para o LLM.")

st.divider()

# --- SECÇÃO 3: SIMULADOR CLÍNICO ---
st.header("3. Entrada de Novo Paciente (Simulador Clínico)")
st.write("Insira os parâmetros abaixo para gerar um diagnóstico dinâmico.")

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

    st.markdown("---")
    st.markdown("**Opções de Geração:**")
    usar_ia_simulador = st.checkbox("🤖 Solicitar Auditoria Textual do Gemini (Sugerido para apresentações oficiais)")
    
    submitted = st.form_submit_button("Gerar Diagnóstico", type="primary")

if submitted:
    variaveis_entrada = [age_in, sex_in, marital_in, income_in, race_in, waist_in, bmi_in, blood_in, hdl_in, tri_in, uric_in, alb_in, uralb_in]
    if None in variaveis_entrada:
        st.warning("⚠️ Preencha todos os campos do formulário antes de processar.")
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
        try: p_new_lime = max(0.0, min(100.0, exp_lime_new.local_pred[0] * 100))
        except: p_new_lime = p_new_shap
            
        fidelidade_new_xai = 100.0 - abs(p_new_shap - p_new_lime)

        col_nprob1, col_nprob2, col_nprob3 = st.columns(3)
        col_nprob1.metric(label="Risco (SHAP)", value=f"{p_new_shap:.1f}%")
        col_nprob2.metric(label="Aproximação (LIME)", value=f"{p_new_lime:.1f}%")
        
        if fidelidade_new_xai >= 90.0: col_nprob3.success(f"Fidelidade: {fidelidade_new_xai:.1f}% (Alta)")
        else: col_nprob3.warning(f"Fidelidade: {fidelidade_new_xai:.1f}%")

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

        aba3_trad, aba3_ia = st.tabs(["📝 Laudo Tradicional (Regras Estatísticas)", "✨ Laudo IA Generativa (Gemini)"])
        valores_shap_new = shap_values_new.values if hasattr(shap_values_new, "values") else shap_values_new
        
        with aba3_trad:
            laudos_sim_classico = gerar_laudo_local_classico(df_novo_bruto, p_new_shap, p_new_lime, fidelidade_new_xai, valores_shap_new, tipo_modelo)
            if laudos_sim_classico['tem_alertas']: st.warning(laudos_sim_classico['classico'])
            else: st.success(laudos_sim_classico['classico'])
            st.info(laudos_sim_classico['ia'])
            if laudos_sim_classico['alto_risco']: st.error(laudos_sim_classico['conduta'])
            else: st.success(laudos_sim_classico['conduta'])

        with aba3_ia:
            if usar_ia_simulador:
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

                with st.spinner("A IA está buscando um servidor livre na nuvem para analisar..."):
                    laudo_simulador = chamar_gemini_local(p_new_shap, status_risco_new, str_alertas_new, str_fr_new, str_fp_new, fidelidade_new_xai, tipo_modelo)
                    if "⏳" in laudo_simulador['laudo']:
                        st.warning(laudo_simulador['laudo'])
                    else:
                        if laudo_simulador['alto_risco']: st.error(laudo_simulador['laudo'])
                        elif laudo_simulador['tem_alertas']: st.warning(laudo_simulador['laudo'])
                        else: st.success(laudo_simulador['laudo'])
            else:
                st.info("A IA não foi acionada para economizar recursos.\n\nPara visualizar a análise por IA, marque a caixa de seleção logo acima do botão e gere novamente o diagnóstico.")
