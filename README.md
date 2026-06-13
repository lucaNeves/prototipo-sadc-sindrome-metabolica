# 🩺 SADC - Sistema de Apoio à Decisão Clínica para Síndrome Metabólica com XAI

Este repositório contém o código-fonte, os modelos preditivos e o caderno de desenvolvimento de um **Sistema de Apoio à Decisão Clínica (SADC)** inovador, 
projetado para triagem, diagnóstico probabilístico e auditoria algorítmica da Síndrome Metabólica.

---

## 🚀 Acesse a Aplicação Online
O sistema está implantado na nuvem e pode ser acessado em tempo real através do seguinte link:
🔗 **[https://prototipo-sadc-sindrome-metabolica.streamlit.app/](https://prototipo-sadc-sindrome-metabolica.streamlit.app/)**

---

## ⚙️ Como Tudo Funciona: A Arquitetura do Sistema

O projeto foi estruturado seguindo um ciclo completo de Engenharia de Machine Learning, dividindo-se em duas grandes fases: **Offline** (desenvolvimento, treino e otimização) e **Online** (inferência reativa e interface do usuário).
1. **Abordagem Preditiva:** O sistema avalia o risco do paciente através de um modelo avançado de comitê baseado em gradiente chamado (**CatBoost**).
2. **Auditoria Algorítmica (XAI):** Para mitigar o problema da "caixa-preta", o sistema decompõe a decisão do modelo localmente utilizando **SHAP** (Teoria dos Jogos Cooperativos) e **LIME** (Modelos Substitutos Locais), calculando em tempo real a métrica de **Fidelidade Local (*Local Fidelity*)** para validar a confiabilidade da explicação. Globalmente, o sistema contrasta o SHAP com a **Importância por Permutação (PFI)**.
3. **Tradução Clínica Automatizada:** Cruzando os pesos matemáticos extraídos da XAI com os limiares fisiopatológicos da NCEP-ATP III (Circunferência da Cintura, Glicemia, Triglicerídeos, HDL), o sistema gera um parecer médico textual imediato com sugestões de conduta clínica e preventiva.

---

## 📓 O Caderno de Desenvolvimento (`.ipynb`)

O arquivo `modelos_com_XAI_2026_1.ipynb` representa o **núcleo de pesquisa científica e engenharia de dados** deste trabalho. Ele foi desenvolvido no ambiente Google Colab e documenta de forma transparente todo o pipeline experimental antes da implantação na nuvem:

* **Extração Automatizada:** Importação remota da base de dados epidemiológica **NHANES** (*National Health and Nutrition Examination Survey*) diretamente de planilhas eletrônicas na nuvem via APIs `gspread` e `gspread-dataframe`.
* **Engenharia de Atributos:** Saneamento de dados ausentes, binarização da variável de sexo e codificação de atributos nominais (*One-Hot Encoding*) para raça e estado civil, evitando colinearidade perfeita.
* **Otimização via GridSearchCV:** Pipeline estruturado que testa exaustivamente combinações de hiperparâmetros para os modelos clássicos e ensembles. O critério de seleção do campeão é o **F1-Score**, garantindo o equilíbrio métrico ideal entre Precisão e Sensibilidade (*Recall*) diante do desbalanceamento epidemiológico da base.
* **Persistência de Objetos (Serialização):** Para neutralizar o vazamento de dados (*data leakage*) e evitar recomputações redundantes em ambiente de produção, o notebook utiliza a biblioteca `joblib` para salvar os estados exatos do escalonador estatístico (`scaler.pkl`) e do cérebro do modelo campeão (`modelo_cat.pkl`).

---

## 📁 Estrutura do Repositório

* **`app.py`**: Código-fonte da aplicação web construído com o framework *Streamlit*. Gerencia a interface gráfica, a reatividade dos formulários, os gráficos de laudo do Matplotlib e o motor de geração de pareceres textuais.
* **`modelos_com_XAI_2026_1.ipynb`**: Notebook completo contendo o desenvolvimento científico, engenharia de dados, validação cruzada e exportação dos modelos.
* **`dataset_app.csv`**: Conjunto de dados tratado e formatado utilizado pelo painel para carregar pacientes históricos e extrair médias de SHAP global.
* **`scaler.pkl`**: Instância salva do *StandardScaler*, contendo as médias e desvios padrões originais do treino, garantindo a escala exata para novos inputs.
* **`modelo_cat.pkl`**: Modelo campeão (CatBoost Classifier) com hiperparâmetros otimizados.
* **`requirements.txt`**: Ficheiro de dependências exigido pelo servidor de nuvem para reconstruir o ecossistema Python (`pandas`, `scikit-learn`, `catboost`, `shap`, `lime`, `streamlit`, etc.).

---
