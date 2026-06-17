import streamlit as st
import pandas as pd
import numpy as np
import shap
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import lime.lime_tabular
import joblib
import google.generativeai as genai

from sklearn.inspection import permutation_importance

# 1. CONFIGURAÇÃO DA PÁGINA E CSS CUSTOMIZADO
st.set_page_config(page_title="XAI - Síndrome Metabólica", layout="wide")

st.markdown("""
<style>
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

# 2. CARREGAMENTO DOS DADOS
@st.cache_data
def carregar_dados():
    df = pd.read_csv('dataset_app.csv')
    X = df.drop(columns=['MetabolicSyndrome', 'seqn'], errors='ignore')
    y = df['MetabolicSyndrome']
    return df, X, y

# ========================================================
# FUNÇÕES DE GRÁFICOS DIDÁTICOS (NOVAS)
# ========================================================
def plotar_termometro_risco(probabilidade):
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = probabilidade,
        number = {'suffix': "%", 'font': {'size': 40}},
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Risco de Síndrome Metabólica", 'font': {'size': 20}},
        gauge = {
            'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "rgba(0,0,0,0.3)"},
            'bgcolor': "white",
            'steps': [
                {'range': [0, 30], 'color': "#2ecc71"},   # Verde (Seguro)
                {'range': [30, 50], 'color': "#f1c40f"},  # Amarelo (Atenção)
                {'range': [50, 100], 'color': "#e74c3c"}  # Vermelho (Perigo)
            ],
            'threshold': {'line': {'color': "black", 'width': 4}, 'thickness': 0.75, 'value': probabilidade}
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    return fig

def plotar_boxplots_clinicos(dados_brutos_paciente, X_completo, y_completo, valores_shap_paciente):
    # Pega as 3 variáveis que mais impactaram a decisão deste paciente
    idx_top3 = np.argsort(np.abs(valores_shap_paciente))[::-1][:3]
    top3_cols = [X_completo.columns[i] for i in idx_top3]
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    for i, col in enumerate(top3_cols):
        # Cria um DataFrame temporário para o Seaborn
        df_plot = pd.DataFrame({'Valor': X_completo[col], 'Diagnóstico': y_completo.map({0: 'Saudáveis', 1: 'Doentes'})})
        
        sns.boxplot(x='Diagnóstico', y='Valor', data=df_plot, ax=axes[i], palette=['#2ecc71', '#e74c3c'])
        
        # Desenha a "Estrela" com o valor exato do paciente
        valor_pac = dados_brutos_paciente[col].values[0]
        axes[i].axhline(y=valor_pac, color='black', linestyle='--', alpha=0.5)
        axes[i].plot([0, 1], [valor_pac, valor_pac], marker='*', color='gold', markersize=20, linestyle='None', markeredgecolor='black', label="Este Paciente")
        
        axes[i].set_title(DICIONARIO_PT.get(col, col), fontweight='bold')
        axes[i].set_xlabel('')
        axes[i].set_ylabel('Valor do Exame')
        if i == 0: axes[i].legend()

    plt.tight_layout()
    return fig

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
    st.write("Insira os parâmetros para obter o diagnóstico. Os gráficos gerados são focados em facilitar o entendimento clínico.")

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

        submitted = st.form_submit_button("Gerar Diagnóstico Didático", type="primary")

    if submitted:
        variaveis_entrada = [age_in, sex_in, marital_in, income_in, race_in, waist_in, bmi_in, blood_in, hdl_in, tri_in, uric_in, alb_in, uralb_in]
        if None in variaveis_entrada:
            st.warning("⚠️ Preencha todos os campos do formulário antes de processar.")
        else:
            encoded_fields = {'Marital_Married': 0, 'Marital_Separated': 0, 'Marital_Single': 0, 'Marital_Widowed': 0, 'Race_Black': 0, 'Race_Hispanic': 0, 'Race_MexAmerican': 0, 'Race_Other': 0, 'Race_White': 0}
            if f"Marital_{marital_in}" in encoded_fields: encoded_fields[f"Marital_{marital_in}"] = 1
            if f"Race_{race_in}" in encoded_fields: encoded_fields[f"Race_{race_in}"] = 1

            novo_paciente_dict = {'Age': age_in, 'Sex': 1 if sex_in == "Feminino" else 0, 'Income': income_in, 'WaistCirc': waist_in, 'BMI': bmi_in, 'Albuminuria': alb_in, 'UrAlbCr': uralb_in, 'UricAcid': uric_in, 'BloodGlucose': blood_in, 'HDL': hdl_in, 'Triglycerides': tri_in, **encoded_fields}

            df_novo_bruto = pd.DataFrame([novo_paciente_dict])[X.columns]
            df_novo_escalonado = scaler.transform(df_novo_bruto)
            p_new_shap = modelo_ativo.predict_proba(df_novo_escalonado)[0][1] * 100

            st.markdown("### 📊 Visão Clínica: Termômetro de Risco")
            # Usa o novo Termômetro
            fig_gauge = plotar_termometro_risco(p_new_shap)
            st.plotly_chart(fig_gauge, use_container_width=True)

            st.markdown("### 📦 Onde este paciente se encaixa? (Comparação com a População)")
            shap_values_new = explainer_ativo(df_novo_escalonado)[0]
            valores_shap_new = shap_values_new.values
            
            # Chama a nossa nova função de Boxplot didático
            fig_boxes = plotar_boxplots_clinicos(df_novo_bruto, X, y, valores_shap_new)
            st.pyplot(fig_boxes)
            
            st.markdown("*(A **estrela dourada** mostra o exame do paciente atual. As caixas verdes representam os exames de pessoas saudáveis do banco de dados, e as vermelhas representam pessoas com a Síndrome).*")


# --- ABA 2: GLOBAL ---
with aba_global:
    st.header("Interpretação Global (Como a IA pensa?)")
    st.write("Gráficos simplificados para demonstrar as variáveis médicas que o algoritmo considera mais importantes no diagnóstico.")

    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.markdown("**1. Quais exames são mais importantes? (Impacto Médio)**")
        st.write("*(Gráfico de barras didático: Barras maiores indicam que a IA presta mais atenção neste exame)*")
        fig_bar, ax_bar = plt.subplots(figsize=(7, 5))
        # Mudamos o summary_plot para tipo BARRAS (mais didático)
        shap.summary_plot(shap_values_ativos, X, plot_type="bar", show=False, color='#3498db')
        plt.xlabel("Peso Médio do Exame na Decisão")
        st.pyplot(fig_bar)

    with col_g2:
        st.markdown("**2. O que acontece se removermos um exame? (Queda de Acerto)**")
        st.write("*(Se a barra é grande, significa que esconder este exame do médico faria ele errar muitos diagnósticos)*")
        sorted_idx = pfi_ativo.importances_mean.argsort()[-10:] # Mostra só os top 10 para ficar limpo
        fig_pfi, ax_pfi = plt.subplots(figsize=(7, 5))
        ax_pfi.barh([DICIONARIO_PT.get(X.columns[i], X.columns[i]) for i in sorted_idx], pfi_ativo.importances_mean[sorted_idx], color='#9b59b6')
        ax_pfi.set_xlabel("Queda de Acurácia (Importância)")
        fig_pfi.tight_layout()
        st.pyplot(fig_pfi)


# --- ABA 3: LOCAL (HISTÓRICOS) ---
with aba_local:
    st.header("Auditoria de Prontuários (Análise Matemática Aprofundada)")
    st.write("Área técnica para validação algorítmica. Recomendada para cientistas de dados e validação de engenharia clínica.")

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
