import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import lime.lime_tabular
import joblib
import google.generativeai as genai

from sklearn.inspection import permutation_importance

# 1. CONFIGURAÇÃO DA PÁGINA E CSS CUSTOMIZADO
st.set_page_config(page_title="XAI - Síndrome Metabólica", layout="wide")

st.markdown("""
<style>
    /* Deixa as 3 abas principais grandes e centralizadas */
    button[data-baseweb="tab"] > div[data-testid="stMarkdownContainer"] > p {
        font-size: 20px !important;
        font-weight: 600 !important;
        padding: 10px 15px !important;
    }
    button[data-baseweb="tab"] { flex: 1; }
</style>
""", unsafe_allow_html=True)

st.title("🩺 Diagnóstico de Síndrome Metabólica com XAI")
st.markdown("Sistema Híbrido de Suporte à Decisão Clínica (CatBoost, SHAP, LIME e LLM).")
st.divider()

# 1.1 CONFIGURAÇÃO DA API GEMINI
try:
    genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
    llm_disponivel = True
except Exception:
    llm_disponivel = False

@st.cache_data(ttl=3600)
def listar_modelos_gemini():
    if not llm_disponivel: return []
    try:
        return [m.name.replace("models/", "") for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
    except:
        return []

def gerar_conteudo_com_fallback(prompt):
    if not llm_disponivel: return "ERRO_CHAVE"
    modelos = listar_modelos_gemini()
    preferencias = ['models/gemini-1.5-flash', 'models/gemini-1.0-pro', 'models/gemini-pro', 'models/gemini-1.5-pro']
    fila_modelos = [m for m in preferencias if m in modelos] + [m for m in modelos if m not in preferencias]
    if not fila_modelos: fila_modelos = preferencias

    teve_cota_excedida = False
    for nome_modelo in fila_modelos:
        try:
            modelo = genai.GenerativeModel(nome_modelo)
            return modelo.generate_content(prompt).text
        except Exception as e:
            if "429" in str(e) or "Quota" in str(e): teve_cota_excedida = True
            continue
    return "ESGOTOU_COTA" if teve_cota_excedida else "ERRO_GERAL"

# DICIONÁRIO DE TRADUÇÃO
DICIONARIO_PT = {
    'WaistCirc': 'Circunferência da Cintura', 'BloodGlucose': 'Glicemia de Jejum',
    'Triglycerides': 'Triglicerídeos', 'HDL': 'Colesterol HDL', 'BMI': 'IMC',
    'Age': 'Idade', 'Sex': 'Sexo Biológico', 'Income': 'Renda Anual',
    'UricAcid': 'Ácido Úrico', 'Albuminuria': 'Albuminúria',
    'UrAlbCr': 'Razão Albumina/Creatinina', 'Marital_Married': 'Casado(a)',
    'Marital_Single': 'Solteiro(a)', 'Marital_Separated': 'Separado(a)',
    'Marital_Widowed': 'Viúvo(a)', 'Race_Black': 'Etnia (Negra)',
    'Race_Hispanic': 'Etnia (Hispânica)', 'Race_MexAmerican': 'Etnia (Mexicana)',
    'Race_White': 'Etnia (Branca)', 'Race_Other': 'Etnia (Outra)'
}

def traduzir(lista): return [DICIONARIO_PT.get(v, v) for v in lista]
def traduzir_e_juntar(lista):
    if not lista: return "Nenhum fator relevante isolado."
    return ", ".join([f"**{f}**" for f in traduzir(lista)])

# 2. CARREGAMENTO DOS DADOS
@st.cache_data
def carregar_dados():
    df = pd.read_csv('dataset_app.csv')
    X = df.drop(columns=['MetabolicSyndrome', 'seqn'], errors='ignore')
    y = df['MetabolicSyndrome']
    return df, X, y

# ========================================================
# FUNÇÃO DO TERMÔMETRO DIDÁTICO
# ========================================================
def plotar_termometro_risco(probabilidade):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probabilidade,
        number = {'suffix': "%", 'font': {'size': 45}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risco Calculado pelo CatBoost", 'font': {'size': 22}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "rgba(0,0,0,0.3)"},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 30], 'color': "#2ecc71"},   # Verde
                {'range': [30, 50], 'color': "#f1c40f"},  # Amarelo
                {'range': [50, 100], 'color': "#e74c3c"}  # Vermelho
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': probabilidade}
        }
    ))
    fig.update_layout(height=350, margin=dict(l=20, r=20, t=50, b=20))
    return fig

# MOTORES DE TEXTO CLÁSSICO E IA
def gerar_laudo_global_classico(pfi_ativo, feature_names):
    importances = pfi_ativo.importances_mean
    indices = np.argsort(importances)[::-1]
    top_features = [feature_names[i] for i in indices[:6]]
    criterios_ncep = ['WaistCirc', 'BloodGlucose', 'Triglycerides', 'HDL']
    alinhados = traduzir([f for f in top_features if f in criterios_ncep])
    nao_classicos = traduzir([f for f in top_features if f not in criterios_ncep])

    return (
        f"**🩺 Validação Clínica (Critérios NCEP-ATP III)**\nA análise global evidencia que o modelo prioriza preditores alinhados à fisiopatologia da Síndrome Metabólica. Biomarcadores como {', '.join([f'**{f}**' for f in alinhados])} exercem a maior influência preditiva.\n*(Nota: A Pressão Arterial não compõe a matriz devido a limitações do dataset).* \n\n"
        f"**🧠 Análise Multidimensional (Padrões Complementares)**\nO algoritmo identifica preditores de forma integrada. Variáveis como {', '.join([f'**{f}**' for f in nao_classicos])} apresentaram impacto significativo na estratificação do risco sistêmico."
    )

def gerar_laudo_local_classico(dados_brutos, prob_shap, fidelidade, shap_values_paciente):
    colunas = dados_brutos.columns.tolist()
    valores_shap = shap_values_paciente.values if hasattr(shap_values_paciente, "values") else shap_values_paciente
    idx_positivos = np.argsort(valores_shap)[::-1]
    idx_negativos = np.argsort(valores_shap)
    fatores_risco = traduzir([colunas[i] for i in idx_positivos if valores_shap[i] > 0][:4])
    fatores_protecao = traduzir([colunas[i] for i in idx_negativos if valores_shap[i] < 0][:3])

    alto_risco = prob_shap >= 50.0
    status_diag = "RISCO CLÍNICO SIGNIFICATIVO" if alto_risco else "RISCO CLÍNICO CONTROLADO"
    
    glicemia = dados_brutos['BloodGlucose'].values[0] if 'BloodGlucose' in dados_brutos else None
    cintura = dados_brutos['WaistCirc'].values[0] if 'WaistCirc' in dados_brutos else None
    trig = dados_brutos['Triglycerides'].values[0] if 'Triglycerides' in dados_brutos else None
    hdl = dados_brutos['HDL'].values[0] if 'HDL' in dados_brutos else None

    alertas = []
    if glicemia and glicemia >= 100: alertas.append(f"Glicemia ({glicemia:.1f})")
    if cintura and cintura >= 88: alertas.append(f"Cintura ({cintura:.1f})")
    if trig and trig >= 150: alertas.append(f"Triglicerídeos ({trig:.1f})")
    if hdl and hdl < 50: alertas.append(f"HDL ({hdl:.1f})")

    t_classico = f"**📋 Rastreio de Parâmetros Diretos:** {', '.join(alertas) if alertas else 'Nenhum limiar crítico clássico foi ultrapassado isoladamente.'}"
    t_ia = f"**🤖 Estratificação (SHAP):** Risco de **{prob_shap:.1f}%** ({status_diag}).\n- **Agravantes:** {', '.join(fatores_risco)}.\n- **Atenuantes:** {', '.join(fatores_protecao)}."
    t_conduta = f"**⚕️ Conduta:** {'Perfil de alto risco. Sugere-se aprofundamento propedêutico.' if alto_risco else 'Perfil estável. Seguimento de rotina.'} (Fidelidade XAI: {fidelidade:.1f}%)"

    return {'texto_completo': f"{t_classico}\n\n{t_ia}\n\n{t_conduta}", 'tem_alertas': len(alertas) > 0, 'alto_risco': alto_risco}

@st.cache_data(show_spinner=False, ttl=3600)
def chamar_gemini_local(prob_shap, status_risco, alertas_str, fatores_risco_str, fatores_protecao_str, fidelidade):
    prompt = f"""[INSTRUÇÃO: RETORNE APENAS AS SEÇÕES DO LAUDO EM PORTUGUÊS. SEM COMENTÁRIOS EXTRAS].
    Gere um laudo médico curto baseado nestes dados do modelo:
    Risco: {prob_shap:.1f}% ({status_risco}). Alertas Clínicos: {alertas_str}. Agravantes Algorítmicos: {fatores_risco_str}. Atenuantes: {fatores_protecao_str}.
    Redija 3 seções: 1. Rastreio Direto, 2. Auditoria Multidimensional, 3. Conduta Sugerida."""
    texto = gerar_conteudo_com_fallback(prompt)
    if texto in ["ERRO_CHAVE", "ESGOTOU_COTA", "ERRO_GERAL"]: return "⚠️ Não foi possível gerar o Laudo de IA (Cota excedida ou Erro na API)."
    return texto

# 4. PREPARAÇÃO DOS MODELOS E EXPLICADORES
@st.cache_resource
def preparar_modelos_e_xai(_X, _y):
    scaler = joblib.load('scaler.pkl')
    modelo_cat = joblib.load('modelo_cat.pkl')
    X_scaled = scaler.transform(_X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=_X.columns)

    explainer_cat = shap.TreeExplainer(modelo_cat)
    shap_values_cat = explainer_cat(X_scaled_df)
    shap_values_cat.data = _X.values
    pfi_cat = permutation_importance(modelo_cat, X_scaled_df, _y, n_repeats=5, random_state=42)

    explainer_lime = lime.lime_tabular.LimeTabularExplainer(
        X_scaled_df.values, feature_names=_X.columns.tolist(), class_names=['Saudável', 'Doente'], mode='classification', random_state=42
    )
    return scaler, modelo_cat, explainer_cat, shap_values_cat, pfi_cat, explainer_lime

df, X, y = carregar_dados()
scaler, modelo_cat, explainer_cat, shap_values_cat, pfi_cat, explainer_lime = preparar_modelos_e_xai(X, y)
modelo_ativo, explainer_ativo, shap_values_ativos, pfi_ativo = modelo_cat, explainer_cat, shap_values_cat, pfi_cat

# ========================================================
# RENDERIZAÇÃO DAS ABAS PRINCIPAIS
# ========================================================
aba_simulador, aba_global, aba_local = st.tabs([
    "🩺 Simulador Clínico (Novo Paciente)", 
    "🌍 Interpretação Global (Como a IA pensa?)", 
    "👤 Auditoria de Históricos (Aprofundado)"
])

# --- ABA 1: SIMULADOR CLÍNICO ---
with aba_simulador:
    st.header("Entrada de Novo Paciente (Simulador Clínico)")

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
        usar_ia_simulador = st.checkbox("🤖 Solicitar Parecer Textual da Inteligência Artificial (Gemini)")
        submitted = st.form_submit_button("Gerar Diagnóstico e Laudos", type="primary")

    if submitted:
        variaveis_entrada = [age_in, sex_in, marital_in, income_in, race_in, waist_in, bmi_in, blood_in, hdl_in, tri_in, uric_in, alb_in, uralb_in]
        if None in variaveis_entrada:
            st.warning("⚠️ Preencha todos os campos do formulário antes de processar.")
        else:
            encoded_fields = {'Marital_Married': 0, 'Marital_Separated': 0, 'Marital_Single': 0, 'Marital_Widowed': 0, 'Race_Black': 0, 'Race_Hispanic': 0, 'Race_MexAmerican': 0, 'Race_Other': 0, 'Race_White': 0}
            if f"Marital_{marital_in}" in encoded_fields: encoded_fields[f"Marital_{marital_in}"] = 1
            if f"Race_{race_in}" in encoded_fields: encoded_fields[f"Race_{race_in}"] = 1

            df_novo_bruto = pd.DataFrame([{'Age': age_in, 'Sex': 1 if sex_in == "Feminino" else 0, 'Income': income_in, 'WaistCirc': waist_in, 'BMI': bmi_in, 'Albuminuria': alb_in, 'UrAlbCr': uralb_in, 'UricAcid': uric_in, 'BloodGlucose': blood_in, 'HDL': hdl_in, 'Triglycerides': tri_in, **encoded_fields}])[X.columns]
            df_novo_escalonado = scaler.transform(df_novo_bruto)
            
            p_new_shap = modelo_ativo.predict_proba(df_novo_escalonado)[0][1] * 100
            
            exp_lime_new = explainer_lime.explain_instance(df_novo_escalonado[0], modelo_ativo.predict_proba, num_features=8)
            p_new_lime = max(0.0, min(100.0, exp_lime_new.local_pred[0] * 100)) if hasattr(exp_lime_new, 'local_pred') else p_new_shap
            fidelidade_new_xai = 100.0 - abs(p_new_shap - p_new_lime)

            # --- RENDERIZANDO OS GRÁFICOS ---
            st.markdown("### 📊 Visão Clínica e Termômetro de Risco")
            st.plotly_chart(plotar_termometro_risco(p_new_shap), use_container_width=True)

            st.markdown("### 📈 Justificativas Algorítmicas (Matemática)")
            col_nplots1, col_nplots2 = st.columns(2)
            with col_nplots1:
                st.markdown("**SHAP Waterfall Plot**")
                shap_values_new = explainer_ativo(df_novo_escalonado)[0]
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

            # --- RENDERIZANDO OS LAUDOS TEXTUAIS (AGORA BEM VISÍVEIS) ---
            st.markdown("---")
            st.markdown("### 📝 Laudo Clínico (Regras Estatísticas)")
            laudos_sim = gerar_laudo_local_classico(df_novo_bruto, p_new_shap, fidelidade_new_xai, shap_values_new.values)
            
            if laudos_sim['alto_risco']: st.error(laudos_sim['texto_completo'])
            elif laudos_sim['tem_alertas']: st.warning(laudos_sim['texto_completo'])
            else: st.success(laudos_sim['texto_completo'])

            if usar_ia_simulador:
                st.markdown("### ✨ Parecer Multidimensional da Inteligência Artificial (Gemini)")
                with st.spinner("Conectando à IA..."):
                    valores = shap_values_new.values
                    idx_pos = np.argsort(valores)[::-1]
                    idx_neg = np.argsort(valores)
                    colunas = X.columns.tolist()
                    
                    alertas = [f"{var}: {val}" for var, val in zip(['Glicose', 'Cintura', 'Triglicerídeos'], [blood_in, waist_in, tri_in]) if val is not None and val >= 100] # Simplificação para prompt
                    
                    laudo_ia = chamar_gemini_local(
                        p_new_shap, "ALTO RISCO" if p_new_shap >= 50 else "BAIXO RISCO", 
                        ", ".join(alertas) if alertas else "Nenhum limiar clássico", 
                        traduzir_e_juntar([colunas[i] for i in idx_pos if valores[i] > 0][:4]), 
                        traduzir_e_juntar([colunas[i] for i in idx_neg if valores[i] < 0][:3]), 
                        fidelidade_new_xai
                    )
                    st.info(laudo_ia)


# --- ABA 2: GLOBAL ---
with aba_global:
    st.header("Interpretação Global (Como a IA pensa?)")
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("**1. Quais exames são mais importantes? (Impacto Médio SHAP)**")
        fig_bar, ax_bar = plt.subplots(figsize=(7, 5))
        shap.summary_plot(shap_values_ativos, X, plot_type="bar", show=False, color='#3498db')
        plt.xlabel("Peso Médio do Exame na Decisão")
        st.pyplot(fig_bar)

    with col_g2:
        st.markdown("**2. O que acontece se removermos um exame? (Queda PFI)**")
        sorted_idx = pfi_ativo.importances_mean.argsort()[-10:]
        fig_pfi, ax_pfi = plt.subplots(figsize=(7, 5))
        ax_pfi.barh([DICIONARIO_PT.get(X.columns[i], X.columns[i]) for i in sorted_idx], pfi_ativo.importances_mean[sorted_idx], color='#9b59b6')
        ax_pfi.set_xlabel("Queda de Acurácia (Importância)")
        fig_pfi.tight_layout()
        st.pyplot(fig_pfi)

    st.markdown("---")
    st.markdown("### 📝 Conclusões do Comportamento Global")
    st.success(gerar_laudo_global_classico(pfi_ativo, X.columns.tolist()))


# --- ABA 3: LOCAL (HISTÓRICOS) ---
with aba_local:
    st.header("Auditoria de Prontuários Históricos")
    paciente_selecionado = st.selectbox("Selecione o paciente histórico:", options=df['seqn'].unique(), format_func=lambda x: f"Prontuário {int(x)} - diagnóstico: {int(df[df['seqn'] == x]['MetabolicSyndrome'].values[0])}")

    idx_pac = df[df['seqn'] == paciente_selecionado].index[0]
    dados_pac_bruto = X.iloc[[idx_pac]]
    dados_pac_escalonado = scaler.transform(dados_pac_bruto)
    prob_shap_hist = modelo_ativo.predict_proba(dados_pac_escalonado)[0][1] * 100

    col_hist1, col_hist2 = st.columns(2)
    with col_hist1:
        st.markdown("**Cálculo Exato do Algoritmo (SHAP Waterfall)**")
        fig_water, ax_water = plt.subplots(figsize=(6, 5))
        shap.plots.waterfall(shap_values_ativos[idx_pac], show=False)
        st.pyplot(fig_water)
        
    with col_hist2:
        st.markdown("**Aproximação Linear (LIME)**")
        exp_lime_hist = explainer_lime.explain_instance(dados_pac_escalonado[0], modelo_ativo.predict_proba, num_features=6)
        fig_lime_hist = exp_lime_hist.as_pyplot_figure()
        fig_lime_hist.set_size_inches(6, 5)
        fig_lime_hist.tight_layout()
        st.pyplot(fig_lime_hist)

    st.markdown("---")
    st.markdown("### 📝 Laudo Técnico do Paciente Histórico")
    
    p_lime_hist = max(0.0, min(100.0, exp_lime_hist.local_pred[0] * 100)) if hasattr(exp_lime_hist, 'local_pred') else prob_shap_hist
    laudo_hist = gerar_laudo_local_classico(dados_pac_bruto, prob_shap_hist, 100.0 - abs(prob_shap_hist - p_lime_hist), shap_values_ativos[idx_pac].values)
    
    if laudo_hist['alto_risco']: st.error(laudo_hist['texto_completo'])
    else: st.success(laudo_hist['texto_completo'])
