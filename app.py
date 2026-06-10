import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import lime.lime_tabular
import joblib

from sklearn.inspection import permutation_importance

# 1. CONFIGURAÇÃO DA PÁGINA
st.set_page_config(page_title="XAI - Síndrome Metabólica", layout="wide")
st.title("🩺 Diagnóstico de Síndrome Metabólica com XAI")
st.markdown("Sistema Híbrido de Suporte à Decisão Clínica (SHAP, LIME e Permutação).")

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

# MOTORES DE GERAÇÃO TEXTUAL CLÍNICA AUTOMATIZADA
def gerar_laudo_global(pfi_ativo, feature_names):
    importances = pfi_ativo.importances_mean
    indices = np.argsort(importances)[::-1]
    top_features = [feature_names[i] for i in indices[:6]] 

    # Embora a NCEP-ATP III tenha 5 critérios, usamos 4 pois a Pressão Arterial não está no dataset
    criterios_ncep = ['WaistCirc', 'BloodGlucose', 'Triglycerides', 'HDL']
    
    alinhados = [f for f in top_features if f in criterios_ncep]
    nao_classicos = [f for f in top_features if f not in criterios_ncep]
    
    alinhados_pt = traduzir(alinhados)
    nao_classicos_pt = traduzir(nao_classicos)

    texto_classico = f"""
    **🩺 Validação Clínica do Algoritmo (Critérios NCEP-ATP III)**
    A análise de importância global evidencia que o modelo prioriza preditores fortemente alinhados à fisiopatologia da Síndrome Metabólica. Biomarcadores estabelecidos na literatura, como {", ".join([f"**{f}**" for f in alinhados_pt])}, exercem a maior influência preditiva. Isto demonstra que a inteligência artificial reproduz o raciocínio clínico validado, garantindo robustez e confiabilidade na estratificação de risco.
    *(Nota de Transparência: O critério de Pressão Arterial preconizado pela diretriz não compõe a matriz preditiva deste modelo devido a limitações de coleta no coorte original).*
    """
    
    texto_holistico = f"""
    **🧠 Análise Multidimensional (Padrões Complementares Identificados)**
    Além dos critérios diagnósticos tradicionais, o algoritmo identifica fatores de risco de forma integrada. Variáveis como {", ".join([f"**{f}**" for f in nao_classicos_pt])} apresentaram impacto significativo na estratificação. Isto indica que o sistema avalia o contínuo metabólico do indivíduo, rastreando vulnerabilidades sociodemográficas e laboratoriais que atuam como cofatores no agravamento clínico, permitindo rastrear o risco de forma mais holística.
    """
    
    return {'classico': texto_classico, 'holistico': texto_holistico}

def gerar_laudo_local(dados_brutos, prob_shap, prob_lime, fidelidade, shap_values_paciente, tipo_modelo):
    colunas = dados_brutos.columns.tolist()

    if hasattr(shap_values_paciente, "values"):
        valores_shap = shap_values_paciente.values
    else:
        valores_shap = shap_values_paciente

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

    texto_classico = f"""
    **📋 Rastreio de Parâmetros Diretos (NCEP-ATP III)**
    Avaliação objetiva dos pontos de corte tradicionais para os marcadores monitorados pelo sistema.
    **Alterações Identificadas:** {", ".join(alertas_clinicos) if tem_alertas else "Nenhum dos limiares críticos monitorados (Glicose, Cintura, HDL ou Triglicerídeos) foi ultrapassado de forma isolada segundo os critérios estritos da diretriz."}
    """

    texto_ia = f"""
    **🤖 Estratificação de Risco Algorítmica (Auditoria SHAP)**
    O modelo calculou uma probabilidade de **{prob_shap:.1f}%** para o enquadramento do paciente na Síndrome Metabólica (**{status_diag}**), ponderando a interação contínua de todo o painel de variáveis.
    * **Fatores Contribuintes (Elevam o risco):** Os elementos que mais elevaram a estimativa de risco algorítmico neste paciente foram: {", ".join([f"**{f}**" for f in fatores_risco_todos])}.
    * **Fatores Atenuantes (Reduzem o risco):** As variáveis que reduziram a probabilidade predita, indicando preservação relativa, foram: {", ".join([f"**{f}**" for f in fatores_protecao_todos])}.
    """

    texto_conduta = f"""
    **⚕️ Apoio à Decisão e Nível de Confiabilidade do Sistema**
    {"**Sugestão de Conduta:** O rastreio algorítmico sugere um perfil de alto risco metabólico sistêmico. Recomenda-se a correlação com a prática clínica diária, aprofundamento da propedêutica (ex: aferição pressórica, perfil lipídico completo) e avaliação para início de intervenções de Mudança de Estilo de Vida (MEV)." if alto_risco else "**Sugestão de Conduta:** O perfil atual sugere estabilidade metabólica segundo o modelo preditivo. Recomenda-se o seguimento clínico de rotina e a manutenção das medidas preventivas primárias."}
    
    *Auditoria de Transparência (LIME): O grau de concordância explicativa local é de {fidelidade:.1f}%.* {"Isto indica alta confiabilidade na leitura do impacto individual das variáveis." if fidelidade >= 85.0 else "Devido a interações estatísticas não-lineares neste paciente específico, a explicação simplificada (LIME) deve ser interpretada com cautela; priorize a observação do gráfico exato (SHAP Waterfall)."}
    """

    return {
        'classico': texto_classico, 
        'ia': texto_ia, 
        'conduta': texto_conduta,
        'tem_alertas': tem_alertas,
        'alto_risco': alto_risco
    }

# 3. CARREGAMENTO DOS MODELOS OTIMIZADOS E EXPLICADORES
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

# 4. INTERFACE DE SELEÇÃO DE MODELOS
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

# INJEÇÃO DO LAUDO DA SEÇÃO 1 (COM CORES)
laudos_globais = gerar_laudo_global(pfi_ativo, X.columns.tolist())
st.success(laudos_globais['classico'])  
st.info(laudos_globais['holistico'])    

st.divider()

# --- SECÇÃO 2: LOCAL (PACIENTE ESPECÍFICO) ---
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

# INJEÇÃO DO LAUDO DA SEÇÃO 2 (COM CORES DINÂMICAS)
if tipo_modelo == "Ensemble (CatBoost)":
    sv_p = shap_values_ativos[idx_paciente]
else:
    sv_p = shap_values_dt[:, :, 1][idx_paciente]

laudos_locais = gerar_laudo_local(dados_paciente_brutos, prob_shap, prob_lime, fidelidade_xai, sv_p, tipo_modelo)

if laudos_locais['tem_alertas']:
    st.warning(laudos_locais['classico'])
else:
    st.success(laudos_locais['classico'])

st.info(laudos_locais['ia'])

if laudos_locais['alto_risco']:
    st.error(laudos_locais['conduta'])
else:
    st.success(laudos_locais['conduta'])

st.divider()

# --- SECÇÃO 3: ENTRADA DE NOVO PACIENTE (FORMULÁRIO MANUAL) ---
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

        # INJEÇÃO DO LAUDO DA SEÇÃO 3 (COM CORES DINÂMICAS)
        laudos_simulador = gerar_laudo_local(df_novo_bruto, p_new_shap, p_new_lime, fidelidade_new_xai, shap_values_new, tipo_modelo)

        if laudos_simulador['tem_alertas']:
            st.warning(laudos_simulador['classico'])
        else:
            st.success(laudos_simulador['classico'])

        st.info(laudos_simulador['ia'])

        if laudos_simulador['alto_risco']:
            st.error(laudos_simulador['conduta'])
        else:
            st.success(laudos_simulador['conduta'])
