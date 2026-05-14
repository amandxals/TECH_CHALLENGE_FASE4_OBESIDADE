"""
Tech Challenge Fase 4 — Sistema Preditivo de Obesidade
Autora: Ana Raquel | POSTECH
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import json
import os

# ─── Configuração da página ────────────────────────────────────────────────────
st.set_page_config(
    page_title="ObesityPredict | POSTECH",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── CSS personalizado ─────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main {background-color: #f8f9fa;}
    .stTabs [data-baseweb="tab-list"] {gap: 8px;}
    .stTabs [data-baseweb="tab"] {
        background-color: #e9ecef;
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563eb !important;
        color: white !important;
    }
    .metric-card {
        background: white;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        text-align: center;
        margin-bottom: 12px;
    }
    .metric-card h2 {font-size: 2.2rem; color: #2563eb; margin: 0;}
    .metric-card p  {color: #6b7280; font-size: 0.9rem; margin: 4px 0 0;}
    .insight-box {
        background: #eff6ff;
        border-left: 5px solid #2563eb;
        border-radius: 0 10px 10px 0;
        padding: 14px 18px;
        margin-bottom: 12px;
    }
    .insight-box.warning {background:#fff7ed; border-color:#f97316;}
    .insight-box.success {background:#f0fdf4; border-color:#22c55e;}
    .insight-box.danger  {background:#fef2f2; border-color:#ef4444;}
    .pred-result {
        border-radius: 14px;
        padding: 28px;
        text-align: center;
        font-size: 1.6rem;
        font-weight: 700;
        margin-top: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Cores por classe ──────────────────────────────────────────────────────────
CLASS_COLORS = {
    "Insufficient_Weight": "#3b82f6",
    "Normal_Weight":        "#22c55e",
    "Overweight_Level_I":   "#facc15",
    "Overweight_Level_II":  "#f97316",
    "Obesity_Type_I":       "#ef4444",
    "Obesity_Type_II":      "#dc2626",
    "Obesity_Type_III":     "#7f1d1d",
}
CLASS_LABELS_PT = {
    "Insufficient_Weight": "Abaixo do Peso",
    "Normal_Weight":        "Peso Normal",
    "Overweight_Level_I":   "Sobrepeso I",
    "Overweight_Level_II":  "Sobrepeso II",
    "Obesity_Type_I":       "Obesidade Tipo I",
    "Obesity_Type_II":      "Obesidade Tipo II",
    "Obesity_Type_III":     "Obesidade Tipo III",
}

# ─── Carregar modelo e dados ───────────────────────────────────────────────────
_BASE = os.path.dirname(os.path.abspath(__file__))
_CANDIDATES = [_BASE, os.path.join(_BASE, "model")]
MODEL_DIR = next((d for d in _CANDIDATES if os.path.exists(os.path.join(d, "model.pkl"))), _BASE)

@st.cache_resource
def load_artifacts():
    model        = pickle.load(open(os.path.join(MODEL_DIR, "model.pkl"),        "rb"))
    le_map       = pickle.load(open(os.path.join(MODEL_DIR, "le_map.pkl"),       "rb"))
    target_le    = pickle.load(open(os.path.join(MODEL_DIR, "target_le.pkl"),    "rb"))
    feature_cols = pickle.load(open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "rb"))
    metrics      = json.load(  open(os.path.join(MODEL_DIR, "metrics.json"),     "r"))
    return model, le_map, target_le, feature_cols, metrics

@st.cache_data
def load_data():
    df = pd.read_csv(os.path.join(MODEL_DIR, "obesity_processed.csv"))
    return df

model, le_map, target_le, feature_cols, metrics = load_artifacts()
df = load_data()

# ─── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://via.placeholder.com/200x60/2563eb/ffffff?text=ObesityPredict", width=200)
    st.markdown("---")
    st.markdown("### 🏥 Tech Challenge — Fase 4")
    st.markdown("**Autora:** Ana Raquel")
    st.markdown("**Curso:** POSTECH — Data Analytics")
    st.markdown("---")
    acc = metrics["accuracy"]
    st.markdown(f"""
    <div class='metric-card'>
        <h2>{acc:.1%}</h2>
        <p>Acurácia do Modelo</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"""
    <div class='metric-card'>
        <h2>{metrics['cv_accuracy']:.1%}</h2>
        <p>Acurácia — Validação Cruzada</p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown(f"**Algoritmo:** {metrics['best_model'].replace('Forest','Forest 🌳')}")
    st.markdown(f"**Dataset:** {len(df):,} pacientes · 16 variáveis")

# ─── Abas principais ───────────────────────────────────────────────────────────
tab_dash, tab_pred, tab_pipeline = st.tabs([
    "📊 Dashboard Analítico",
    "🔮 Sistema Preditivo",
    "⚙️ Pipeline ML",
])

# ══════════════════════════════════════════════════════════════════════════════
# ABA 1 — DASHBOARD
# ══════════════════════════════════════════════════════════════════════════════
with tab_dash:
    st.markdown("## 📊 Painel Analítico — Obesidade")
    st.markdown("Visão geral dos dados de 2.111 pacientes para apoio à equipe médica.")

    # ── KPIs ──────────────────────────────────────────────────────────────────
    obese = df["Obesity"].str.startswith("Obesity").sum()
    overw = df["Obesity"].str.startswith("Overweight").sum()
    normal = (df["Obesity"] == "Normal_Weight").sum()
    insuf  = (df["Obesity"] == "Insufficient_Weight").sum()

    c1,c2,c3,c4 = st.columns(4)
    with c1:
        st.markdown(f"""
        <div class='metric-card'>
            <h2 style='color:#ef4444'>{obese/len(df):.0%}</h2>
            <p>Pacientes com Obesidade<br><small>({obese} de {len(df)})</small></p>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown(f"""
        <div class='metric-card'>
            <h2 style='color:#f97316'>{overw/len(df):.0%}</h2>
            <p>Pacientes com Sobrepeso<br><small>({overw} de {len(df)})</small></p>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown(f"""
        <div class='metric-card'>
            <h2 style='color:#22c55e'>{normal/len(df):.0%}</h2>
            <p>Peso Normal<br><small>({normal} de {len(df)})</small></p>
        </div>""", unsafe_allow_html=True)
    with c4:
        st.markdown(f"""
        <div class='metric-card'>
            <h2 style='color:#3b82f6'>{insuf/len(df):.0%}</h2>
            <p>Abaixo do Peso<br><small>({insuf} de {len(df)})</small></p>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # ── Distribuição das classes ───────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("### 🎯 Distribuição dos Níveis de Obesidade")
        counts = df["Obesity"].value_counts()
        labels = [CLASS_LABELS_PT.get(k, k) for k in counts.index]
        colors = [CLASS_COLORS.get(k, "#999") for k in counts.index]

        # Usar plotly se disponível, senão bar chart nativo do streamlit
        try:
            import plotly.graph_objects as go
            fig = go.Figure(go.Bar(
                x=counts.values,
                y=labels,
                orientation='h',
                marker_color=colors,
                text=counts.values,
                textposition='outside',
            ))
            fig.update_layout(
                height=320, margin=dict(l=10,r=30,t=10,b=10),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                xaxis_title="Nº de Pacientes", yaxis_title="",
                font=dict(size=12)
            )
            st.plotly_chart(fig, use_container_width=True)
        except ImportError:
            st.bar_chart(pd.Series(counts.values, index=labels))

    with col_b:
        st.markdown("### 🧬 Histórico Familiar vs Obesidade")
        cross = pd.crosstab(df["family_history"], df["Obesity"])
        # resumir em obeso/nao obeso
        cross["Obeso"] = cross.get("Obesity_Type_I",0) + cross.get("Obesity_Type_II",0) + cross.get("Obesity_Type_III",0)
        cross["Sobrepeso"] = cross.get("Overweight_Level_I",0) + cross.get("Overweight_Level_II",0)
        cross["Normal/Abaixo"] = cross.get("Normal_Weight",0) + cross.get("Insufficient_Weight",0)
        summary = cross[["Obeso","Sobrepeso","Normal/Abaixo"]]
        try:
            import plotly.graph_objects as go
            fig2 = go.Figure()
            palette = {"Obeso":"#ef4444","Sobrepeso":"#f97316","Normal/Abaixo":"#22c55e"}
            for col_name in ["Obeso","Sobrepeso","Normal/Abaixo"]:
                fig2.add_trace(go.Bar(
                    name=col_name,
                    x=["Sem histórico familiar","Com histórico familiar"],
                    y=[summary.loc["no",col_name], summary.loc["yes",col_name]],
                    marker_color=palette[col_name],
                ))
            fig2.update_layout(
                barmode='stack', height=320,
                margin=dict(l=10,r=10,t=10,b=10),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                legend=dict(orientation="h", y=-0.2),
                font=dict(size=12)
            )
            st.plotly_chart(fig2, use_container_width=True)
        except ImportError:
            st.dataframe(summary)

    # ── Insights ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 💡 Insights Clínicos para a Equipe Médica")

    # Calcular insights reais
    hist_yes_obese_pct = df[df["family_history"]=="yes"]["Obesity"].str.startswith("Obesity").mean()
    hist_no_obese_pct  = df[df["family_history"]=="no"]["Obesity"].str.startswith("Obesity").mean()
    favc_obese = df[df["FAVC"]=="yes"]["Obesity"].str.startswith("Obesity").mean()
    bmi_by_class = df.copy()
    bmi_by_class["BMI"] = bmi_by_class["Weight"] / (bmi_by_class["Height"]**2)
    low_faf = df[df["FAF"] < 1]["Obesity"].str.startswith("Obesity").mean()
    smoke_obese = df[df["SMOKE"]=="yes"]["Obesity"].str.startswith("Obesity").mean()
    smoke_no    = df[df["SMOKE"]=="no"]["Obesity"].str.startswith("Obesity").mean()

    i1, i2 = st.columns(2)
    with i1:
        st.markdown(f"""
        <div class='insight-box danger'>
            <strong>🧬 Histórico Familiar é Fator Crítico</strong><br>
            Pacientes <em>com</em> histórico familiar têm <strong>{hist_yes_obese_pct:.0%}</strong> de chance de obesidade,
            contra <strong>{hist_no_obese_pct:.0%}</strong> sem histórico — diferença de
            <strong>{(hist_yes_obese_pct - hist_no_obese_pct):.0%} pontos percentuais</strong>.
        </div>
        <div class='insight-box warning'>
            <strong>🍔 Alimentação Calórica Frequente</strong><br>
            Pacientes que consomem alimentos calóricos frequentemente têm <strong>{favc_obese:.0%}</strong>
            de probabilidade de obesidade. Intervenção nutricional é prioritária para esse grupo.
        </div>
        """, unsafe_allow_html=True)

    with i2:
        st.markdown(f"""
        <div class='insight-box warning'>
            <strong>🏃 Sedentarismo Aumenta Risco</strong><br>
            Pacientes com baixa frequência de atividade física (menos de 1x/semana) apresentam
            <strong>{low_faf:.0%}</strong> de obesidade. Prescrição de exercício é fundamental.
        </div>
        <div class='insight-box'>
            <strong>🚬 Tabagismo e Obesidade</strong><br>
            Fumantes: <strong>{smoke_obese:.0%}</strong> de obesidade.
            Não fumantes: <strong>{smoke_no:.0%}</strong>. O impacto do tabagismo
            isolado não é o mais relevante nesta amostra — fatores alimentares e
            genéticos se destacam mais.
        </div>
        """, unsafe_allow_html=True)

    # ── Análise por Gênero ────────────────────────────────────────────────────
    st.markdown("---")
    col_g1, col_g2 = st.columns(2)

    with col_g1:
        st.markdown("### 👥 Obesidade por Gênero")
        gender_obese = df.groupby("Gender")["Obesity"].apply(
            lambda x: (x.str.startswith("Obesity").sum() / len(x) * 100).round(1)
        )
        try:
            import plotly.graph_objects as go
            fig3 = go.Figure(go.Bar(
                x=gender_obese.index.map({"Female":"Feminino","Male":"Masculino"}),
                y=gender_obese.values,
                marker_color=["#ec4899","#3b82f6"],
                text=[f"{v:.1f}%" for v in gender_obese.values],
                textposition='outside',
            ))
            fig3.update_layout(
                height=280, margin=dict(l=10,r=10,t=10,b=10),
                yaxis_title="% com Obesidade", yaxis_range=[0,50],
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=13)
            )
            st.plotly_chart(fig3, use_container_width=True)
        except ImportError:
            st.dataframe(gender_obese)

    with col_g2:
        st.markdown("### 🚶 Transporte e Nível de Obesidade")
        trans_obese = df.groupby("MTRANS")["Obesity"].apply(
            lambda x: (x.str.startswith("Obesity") | x.str.startswith("Overweight")).mean()*100
        ).sort_values(ascending=False)
        trans_labels = {
            "Automobile":"Carro","Public_Transportation":"Transp. Público",
            "Walking":"A pé","Bike":"Bicicleta","Motorbike":"Moto"
        }
        try:
            import plotly.graph_objects as go
            fig4 = go.Figure(go.Bar(
                x=[trans_labels.get(k,k) for k in trans_obese.index],
                y=trans_obese.values,
                marker_color=["#ef4444","#f97316","#facc15","#22c55e","#3b82f6"][:len(trans_obese)],
                text=[f"{v:.0f}%" for v in trans_obese.values],
                textposition='outside',
            ))
            fig4.update_layout(
                height=280, margin=dict(l=10,r=10,t=10,b=10),
                yaxis_title="% Sobrepeso + Obesidade",
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12)
            )
            st.plotly_chart(fig4, use_container_width=True)
        except ImportError:
            st.dataframe(trans_obese)


# ══════════════════════════════════════════════════════════════════════════════
# ABA 2 — SISTEMA PREDITIVO
# ══════════════════════════════════════════════════════════════════════════════
with tab_pred:
    st.markdown("## 🔮 Sistema Preditivo de Obesidade")
    st.markdown(
        "Preencha os dados do paciente abaixo. O modelo calculará automaticamente "
        "o nível de obesidade mais provável com a probabilidade de cada classe."
    )

    with st.form("predict_form"):
        st.markdown("### 👤 Dados Pessoais")
        c1, c2, c3, c4 = st.columns(4)
        gender = c1.selectbox("Gênero", ["Female","Male"],
                               format_func=lambda x: "Feminino" if x=="Female" else "Masculino")
        age    = c2.slider("Idade (anos)", 14, 61, 25)
        height = c3.slider("Altura (m)", 1.45, 1.98, 1.70, step=0.01)
        weight = c4.slider("Peso (kg)",  39.0, 173.0, 70.0, step=0.5)

        st.markdown("### 🍽️ Hábitos Alimentares")
        c5, c6, c7, c8 = st.columns(4)
        family_history = c5.selectbox("Histórico familiar de excesso de peso?",
                                       ["yes","no"], format_func=lambda x: "Sim" if x=="yes" else "Não")
        favc = c6.selectbox("Come alimentos calóricos com frequência?",
                             ["yes","no"], format_func=lambda x: "Sim" if x=="yes" else "Não")
        fcvc = c7.select_slider("Frequência de vegetais (1=Raramente → 3=Sempre)", [1,2,3], value=2)
        ncp  = c8.select_slider("Refeições principais/dia", [1,2,3,4], value=3)

        c9, c10 = st.columns(2)
        caec = c9.selectbox("Come entre as refeições?",
                             ["no","Sometimes","Frequently","Always"],
                             format_func={"no":"Não","Sometimes":"Às vezes",
                                          "Frequently":"Frequentemente","Always":"Sempre"}.get)
        calc = c10.selectbox("Consome álcool?",
                              ["no","Sometimes","Frequently","Always"],
                              format_func={"no":"Não","Sometimes":"Às vezes",
                                           "Frequently":"Frequentemente","Always":"Sempre"}.get)

        st.markdown("### 🏃 Estilo de Vida")
        c11, c12, c13, c14, c15, c16 = st.columns(6)
        smoke  = c11.selectbox("Fuma?", ["no","yes"],
                                format_func=lambda x: "Não" if x=="no" else "Sim")
        ch2o   = c12.select_slider("Água/dia\n(1=<1L, 2=1-2L, 3=>2L)", [1,2,3], value=2)
        scc    = c13.selectbox("Monitora calorias?", ["no","yes"],
                                format_func=lambda x: "Não" if x=="no" else "Sim")
        faf    = c14.select_slider("Atividade física/sem\n(0=nenhuma → 3=diária)", [0,1,2,3], value=1)
        tue    = c15.select_slider("Tempo em telas/dia\n(0=<2h, 1=3-5h, 2=>5h)", [0,1,2], value=1)
        mtrans = c16.selectbox("Transporte habitual",
                                ["Automobile","Public_Transportation","Walking","Bike","Motorbike"],
                                format_func={"Automobile":"Carro","Public_Transportation":"Transporte Público",
                                             "Walking":"A pé","Bike":"Bicicleta","Motorbike":"Moto"}.get)

        submitted = st.form_submit_button("🔍 Realizar Predição", use_container_width=True)

    if submitted:
        # ── Montar dataframe com feature engineering igual ao treino ──────────
        bmi          = weight / (height ** 2)   # só para exibição clínica
        age_group    = ("adolescente" if age <= 18 else
                        "jovem_adulto" if age <= 30 else
                        "adulto" if age <= 45 else "idoso")
        sedentary    = (3 - faf) + tue
        nutrition    = fcvc - (1 if favc == "yes" else 0)

        row = {
            "Gender": gender, "Age": float(age), "Height": float(height),
            "Weight": float(weight), "family_history": family_history,
            "FAVC": favc, "FCVC": fcvc, "NCP": ncp, "CAEC": caec,
            "SMOKE": smoke, "CH2O": ch2o, "SCC": scc, "FAF": faf,
            "TUE": tue, "CALC": calc, "MTRANS": mtrans,
            "Age_group": age_group,  # BMI removido das features
            "sedentary_risk": sedentary, "nutrition_score": nutrition,
        }

        inp = pd.DataFrame([row])
        cat_cols = ["Gender","family_history","FAVC","CAEC","SMOKE","SCC","CALC","MTRANS","Age_group"]
        for c in cat_cols:
            le = le_map[c]
            val = inp[c].values[0]
            if val not in le.classes_:
                val = le.classes_[0]
            inp[c] = le.transform([val])

        inp = inp[feature_cols]
        pred_enc   = model.predict(inp)[0]
        pred_label = target_le.inverse_transform([pred_enc])[0]
        pred_pt    = CLASS_LABELS_PT.get(pred_label, pred_label)
        pred_color = CLASS_COLORS.get(pred_label, "#666")
        probas     = model.predict_proba(inp)[0]
        classes    = target_le.inverse_transform(range(len(probas)))

        # ── Resultado principal ────────────────────────────────────────────────
        st.markdown(f"""
        <div class='pred-result' style='background:{pred_color}22; border:3px solid {pred_color}; color:{pred_color}'>
            🏥 Diagnóstico Preditivo: <strong>{pred_pt}</strong>
        </div>
        """, unsafe_allow_html=True)

        # ── BMI calculado ──────────────────────────────────────────────────────
        bmi_cat = ("Abaixo do Peso" if bmi < 18.5 else
                   "Normal" if bmi < 25 else
                   "Sobrepeso" if bmi < 30 else "Obesidade")
        st.markdown(f"""
        <div class='insight-box' style='margin-top:16px'>
            <strong>📏 IMC Calculado:</strong> {bmi:.1f} kg/m² — <em>{bmi_cat}</em>
        </div>
        """, unsafe_allow_html=True)

        # ── Probabilidades ─────────────────────────────────────────────────────
        st.markdown("### 📊 Probabilidade por Classe")
        prob_df = pd.DataFrame({
            "Classe": [CLASS_LABELS_PT.get(c,c) for c in classes],
            "Probabilidade": probas,
            "Cor": [CLASS_COLORS.get(c,"#999") for c in classes],
        }).sort_values("Probabilidade", ascending=False)

        try:
            import plotly.graph_objects as go
            fig_p = go.Figure(go.Bar(
                x=prob_df["Probabilidade"]*100,
                y=prob_df["Classe"],
                orientation='h',
                marker_color=prob_df["Cor"].tolist(),
                text=[f"{v*100:.1f}%" for v in prob_df["Probabilidade"]],
                textposition='outside',
            ))
            fig_p.update_layout(
                height=300, margin=dict(l=10,r=60,t=10,b=10),
                xaxis_title="Probabilidade (%)", xaxis_range=[0,110],
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                font=dict(size=12)
            )
            st.plotly_chart(fig_p, use_container_width=True)
        except ImportError:
            st.dataframe(prob_df[["Classe","Probabilidade"]])

        # ── Recomendações clínicas ─────────────────────────────────────────────
        st.markdown("### 📋 Recomendações para a Equipe Médica")
        recs = []
        if pred_label in ["Obesity_Type_I","Obesity_Type_II","Obesity_Type_III"]:
            recs.append(("danger","🔴 Obesidade detectada. Avalie encaminhamento para nutricionista e endocrinologista."))
        if pred_label in ["Overweight_Level_I","Overweight_Level_II"]:
            recs.append(("warning","🟡 Sobrepeso. Orientação alimentar e incremento de atividade física são recomendados."))
        if pred_label == "Insufficient_Weight":
            recs.append(("","🔵 Abaixo do peso. Avaliar risco de desnutrição e distúrbios alimentares."))
        if faf <= 1:
            recs.append(("warning","🏃 Baixa atividade física. Estimule ao menos 150 min/semana de exercício moderado."))
        if favc == "yes":
            recs.append(("warning","🍔 Consumo frequente de alimentos calóricos. Intervenção nutricional prioritária."))
        if family_history == "yes":
            recs.append(("danger","🧬 Histórico familiar de sobrepeso. Monitoramento periódico recomendado."))
        if ch2o == 1:
            recs.append(("","💧 Ingestão hídrica baixa (<1L/dia). Oriente aumento do consumo de água."))
        if not recs:
            recs.append(("success","✅ Perfil dentro de parâmetros saudáveis. Manter hábitos atuais."))

        for tipo, msg in recs:
            st.markdown(f"<div class='insight-box {tipo}'>{msg}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# ABA 3 — PIPELINE ML
# ══════════════════════════════════════════════════════════════════════════════
with tab_pipeline:
    st.markdown("## ⚙️ Pipeline de Machine Learning — Documentação")

    st.markdown("""
    ### 1️⃣ O que é Machine Learning?
    > Imagine ensinar uma criança a reconhecer frutas mostrando centenas de exemplos.
    > Machine Learning faz o mesmo com computadores: mostramos muitos casos já resolvidos
    > para que ele aprenda a resolver casos novos.

    ---
    ### 2️⃣ Etapas do Pipeline
    """)

    steps = [
        ("📥 1. Carregamento dos Dados", f"2.111 pacientes com 16 variáveis cada. Sem valores faltantes."),
        ("🛠️ 2. Feature Engineering\n(Criação de Novas Variáveis)",
         """
         - **IMC** (Índice de Massa Corporal) = Peso ÷ Altura²
         - **Faixa etária** (adolescente / jovem adulto / adulto / idoso)
         - **Risco sedentário** = quanto menos exercício + mais tela, maior o valor
         - **Escore nutricional** = frequência de vegetais menos consumo de calóricos
         """),
        ("🔢 3. Encoding (Transformação de Texto em Número)",
         "Categorias como 'Feminino/Masculino' ou 'Sim/Não' são convertidas em números para o modelo processar."),
        ("✂️ 4. Divisão Treino/Teste", "80% dos dados para treinar o modelo, 20% para testá-lo em casos nunca vistos."),
        ("🌳 5. Treinamento dos Modelos",
         f"""
         Três algoritmos foram testados:

         | Algoritmo | Acurácia (Teste) | Acurácia (Validação Cruzada) |
         |---|---|---|
         | Random Forest 🏆 | {metrics['all_results']['RandomForest']['accuracy']:.2%} | {metrics['all_results']['RandomForest']['cv_accuracy']:.2%} |
         | Gradient Boosting | {metrics['all_results']['GradientBoosting']['accuracy']:.2%} | {metrics['all_results']['GradientBoosting']['cv_accuracy']:.2%} |
         | Decision Tree | {metrics['all_results']['DecisionTree']['accuracy']:.2%} | {metrics['all_results']['DecisionTree']['cv_accuracy']:.2%} |
         """),
        ("✅ 6. Modelo Escolhido",
         f"**Random Forest** com acurácia de **{metrics['accuracy']:.1%}** — supera amplamente o mínimo de 75% exigido."),
    ]

    for title, content in steps:
        with st.expander(title, expanded=True):
            st.markdown(content)

    st.markdown("---")
    st.markdown("### 🌳 Por que Random Forest?")
    st.markdown("""
    > Imagine uma floresta de árvores de decisão. Cada árvore "vota" em uma resposta,
    > e o modelo escolhe a resposta mais votada. Isso torna o modelo muito mais robusto
    > do que usar apenas uma árvore isolada.

    - ✅ Alta acurácia em dados tabulares
    - ✅ Resistente a overfitting (decoreba)
    - ✅ Lida bem com variáveis numéricas e categóricas
    - ✅ Interpretável via importância das features
    """)

    st.markdown("---")
    st.markdown("### 📌 Importância das Variáveis")

    feat_imp = pd.Series(model.feature_importances_, index=feature_cols).sort_values(ascending=False).head(12)
    feat_labels = {
        "BMI":"IMC","Weight":"Peso","Height":"Altura","Age":"Idade",
        "FCVC":"Frequência Vegetais","FAF":"Atividade Física","CH2O":"Consumo Água",
        "NCP":"Nº Refeições","TUE":"Tempo em Telas","sedentary_risk":"Risco Sedentário",
        "nutrition_score":"Escore Nutricional","FAVC":"Alimentos Calóricos",
        "family_history":"Histórico Familiar","CAEC":"Come Entre Refeições",
        "Gender":"Gênero","SMOKE":"Fuma","SCC":"Monitora Calorias",
        "CALC":"Álcool","MTRANS":"Transporte","Age_group":"Faixa Etária",
    }
    try:
        import plotly.graph_objects as go
        fig_fi = go.Figure(go.Bar(
            x=feat_imp.values * 100,
            y=[feat_labels.get(f,f) for f in feat_imp.index],
            orientation='h',
            marker_color='#2563eb',
            text=[f"{v*100:.1f}%" for v in feat_imp.values],
            textposition='outside',
        ))
        fig_fi.update_layout(
            height=400, margin=dict(l=10,r=60,t=10,b=10),
            xaxis_title="Importância (%)",
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            font=dict(size=12)
        )
        st.plotly_chart(fig_fi, use_container_width=True)
    except ImportError:
        st.dataframe(feat_imp)

    st.info("""
    📌 **Leitura:** O IMC (calculado a partir de Peso e Altura) é de longe a variável mais
    importante para o modelo — o que faz sentido clinicamente. Em seguida vêm Peso, Altura e
    Idade, que também determinam muito o nível de obesidade. Variáveis comportamentais como
    Frequência de Vegetais e Atividade Física também contribuem significativamente.
    """)
