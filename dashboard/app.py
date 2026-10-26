"""Dashboard interactivo del Proyecto Integrador METABRIC (Fase 4).

Ejecución local (desde la raíz del repositorio):
    streamlit run dashboard/app.py

Módulos exigidos por la guía:
  1. Filtros dinámicos (barra lateral): rangos de tiempo de seguimiento, edad y NPI; categorías.
  2. Tarjetas KPI: indicadores de la cohorte filtrada y del desempeño de los modelos.
  3. Predicción en vivo: NPI (Ridge) y subtipo PAM50 (SelectKBest + logística) para un perfil editable.
  4. Segmentación: clústeres K-Means en el espacio PCA (2D o 3D) con sus perfiles promedio.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src import config  # noqa: E402
from src.features import add_derived_features, PROLIFERATION_GENES  # noqa: E402

st.set_page_config(page_title="METABRIC: pronóstico, subtipos y segmentación", layout="wide")

CLASS_ORDER = ["LumA", "LumB", "Her2", "Basal", "claudin-low", "Normal"]
CLASS_NAMES = {"LumA": "Luminal A", "LumB": "Luminal B", "Her2": "HER2-enriquecido", "Basal": "Basal",
               "claudin-low": "Claudin-low", "Normal": "Normal-like"}
CLUSTER_NAMES = {0: "0 · Agresivo ER negativo", 1: "1 · Luminal de baja proliferación",
                 2: "2 · Intermedio (posible efecto de lote)", 3: "3 · Luminal clásico"}
PALETTE = {"0 · Agresivo ER negativo": "#C44E52", "1 · Luminal de baja proliferación": "#55A868",
           "2 · Intermedio (posible efecto de lote)": "#8C8C8C", "3 · Luminal clásico": "#4C72B0"}
TARGET_CLF = config.TARGET_CLASSIFICATION
CLUSTER_ORDER = list(CLUSTER_NAMES.values())


def es(x, d=2):
    """Formato numérico en español: punto para miles y coma para decimales."""
    return f"{x:,.{d}f}".replace(",", "X").replace(".", ",").replace("X", ".")


# --------------------------------------------------------------------------------------
# Carga de datos y modelos (en caché)
# --------------------------------------------------------------------------------------
@st.cache_data
def load_data():
    df = pd.read_csv(config.DATA_PROCESSED / "metabric_clean.csv", low_memory=False)
    split = pd.read_csv(config.DATA_PROCESSED / "splits.csv")
    clusters = pd.read_csv(config.DATA_PROCESSED / "clusters.csv")
    extra = pd.DataFrame({
        "split": split["split"].to_numpy(),
        "cluster": clusters["cluster"].to_numpy(),
        "subtipo": df[TARGET_CLF].map(CLASS_NAMES).fillna("No clasificado").to_numpy(),
        "muerte_por_cancer": (df["death_from_cancer"] == "Died of Disease").astype(int).to_numpy(),
        "estado_vital": np.where(df["overall_survival"] == 1, "Viva", "Fallecida"),
        "npi_ok": df["nottingham_prognostic_index"].where(df["npi_valid"] == 1).to_numpy(),
    }, index=df.index)
    extra["clúster"] = extra["cluster"].map(CLUSTER_NAMES)
    return pd.concat([df.drop(columns=["split"], errors="ignore"), extra], axis=1)


@st.cache_resource
def load_models():
    models = {
        "reg": joblib.load(config.MODELS_DIR / "final_regressor.joblib"),
        "clf": joblib.load(config.MODELS_DIR / "final_classifier.joblib"),
        "clu": joblib.load(config.MODELS_DIR / "clustering_kmeans.joblib"),
    }
    with open(config.MODELS_DIR / "final_models.json", encoding="utf-8") as f:
        models["meta"] = json.load(f)
    return models


@st.cache_data
def pca_coordinates(_clu, genes_frame):
    Z = _clu["pca"].transform(_clu["scaler"].transform(genes_frame))
    return pd.DataFrame(Z[:, :3], columns=["PC1", "PC2", "PC3"], index=genes_frame.index)


@st.cache_data
def load_metric_tables():
    out = {}
    for key, name in [("reg", "act6_regresion_prueba.csv"), ("clf", "act6_clasificacion_prueba.csv"),
                      ("smart", "act6_smart.csv"), ("shap_clf", "act6_shap_clasificacion_top30.csv"),
                      ("shap_reg", "act6_shap_regresion_top30.csv")]:
        path = config.METRICS_DIR / name
        if path.exists():
            out[key] = pd.read_csv(path)
    return out


def kaplan_meier(time, event):
    """Estimador de Kaplan-Meier: devuelve tiempos y supervivencia acumulada."""
    d = pd.DataFrame({"t": time, "e": event}).dropna().sort_values("t")
    times, surv, s = [0.0], [1.0], 1.0
    at_risk = len(d)
    for t, g in d.groupby("t"):
        deaths = g["e"].sum()
        if deaths > 0 and at_risk > 0:
            s *= 1 - deaths / at_risk
            times.append(t)
            surv.append(s)
        at_risk -= len(g)
    return times, surv


df = load_data()
M = load_models()
metrics = load_metric_tables()
genes = M["clu"]["genes"]
coords = pca_coordinates(M["clu"], df[genes])
df = pd.concat([df, coords], axis=1)

# --------------------------------------------------------------------------------------
# Barra lateral: filtros dinámicos
# --------------------------------------------------------------------------------------
st.sidebar.title("Filtros")
st.sidebar.caption("El conjunto de datos no tiene fechas de calendario; el filtro temporal usa el tiempo de "
                   "seguimiento desde el diagnóstico (meses).")
fu = st.sidebar.slider("Tiempo de seguimiento (meses)", 0, int(np.ceil(df["overall_survival_months"].max())),
                       (0, int(np.ceil(df["overall_survival_months"].max()))))
age = st.sidebar.slider("Edad al diagnóstico (años)", int(df["age_at_diagnosis"].min()),
                        int(np.ceil(df["age_at_diagnosis"].max())),
                        (int(df["age_at_diagnosis"].min()), int(np.ceil(df["age_at_diagnosis"].max()))))
npi_rng = st.sidebar.slider("Nottingham Prognostic Index", 1.0, 6.5, (1.0, 6.5), step=0.1)
SUB_OPTS = list(CLASS_NAMES.values()) + ["No clasificado"]
sel_sub = st.sidebar.multiselect("Subtipo PAM50", SUB_OPTS, default=SUB_OPTS)
sel_rec = st.sidebar.multiselect("Grupo por receptores", sorted(df["receptor_group"].unique()),
                                 default=sorted(df["receptor_group"].unique()))
sel_clu = st.sidebar.multiselect("Clúster", list(CLUSTER_NAMES.values()), default=list(CLUSTER_NAMES.values()))
sel_grade = st.sidebar.multiselect("Grado histológico", ["1", "2", "3", "Sin dato"], default=["1", "2", "3", "Sin dato"])
sel_split = st.sidebar.multiselect("Subconjunto", ["train", "val", "test"], default=["train", "val", "test"])

grade_lbl = df["neoplasm_histologic_grade"].map({1.0: "1", 2.0: "2", 3.0: "3"}).fillna("Sin dato")
mask = (df["overall_survival_months"].between(*fu) & df["age_at_diagnosis"].between(*age)
        & df["nottingham_prognostic_index"].between(*npi_rng)
        & df["subtipo"].isin(sel_sub) & df["receptor_group"].isin(sel_rec)
        & df["clúster"].isin(sel_clu) & grade_lbl.isin(sel_grade) & df["split"].isin(sel_split))
f = df[mask]
st.sidebar.markdown(f"**{es(len(f), 0)} de {es(len(df), 0)} pacientes** cumplen los filtros.")

# --------------------------------------------------------------------------------------
# Encabezado
# --------------------------------------------------------------------------------------
st.title("Cáncer de mama en METABRIC: pronóstico, subtipos moleculares y segmentación")
st.markdown("Proyecto Integrador de Ciencia de Datos, Universidad de La Salle. **Martín Gracia Socha**. "
            "Datos: METABRIC (Kaggle / cBioPortal), 1.903 pacientes con datos clínicos, expresión de 489 genes y "
            "mutaciones de 173 genes. *Herramienta académica: no debe usarse para decisiones clínicas.*")

tab_res, tab_exp, tab_pred, tab_seg, tab_mod = st.tabs(
    ["Resumen", "Exploración", "Predicción en vivo", "Segmentación", "Modelos y explicabilidad"])

# --------------------------------------------------------------------------------------
# 1. Resumen con KPIs y narrativa
# --------------------------------------------------------------------------------------
with tab_res:
    if f.empty:
        st.warning("No hay pacientes con los filtros seleccionados.")
    else:
        c = st.columns(5)
        c[0].metric("Pacientes", es(len(f), 0))
        c[1].metric("Edad mediana", f"{es(f['age_at_diagnosis'].median(), 0)} años")
        c[2].metric("NPI medio", es(f.loc[f.npi_valid == 1, "nottingham_prognostic_index"].mean()))
        c[3].metric("ER positivo", f"{100 * (f['er_status'] == 'Positive').mean():.0f} %")
        c[4].metric("Mortalidad por cáncer", f"{100 * f['muerte_por_cancer'].mean():.0f} %")
        meta = M["meta"]
        c = st.columns(5)
        c[0].metric("R² del NPI (prueba)", es(meta["regressor"]["test"]["R2"], 3),
                    help="Ridge; meta SMART: R² ≥ 0,20")
        c[1].metric("RMSE del NPI (prueba)", es(meta["regressor"]["test"]["RMSE"], 3))
        c[2].metric("F1 macro PAM50 (prueba)", es(meta["classifier"]["test"]["F1 macro"], 3),
                    help="SelectKBest + logística; meta SMART: F1 ≥ 0,75")
        c[3].metric("AUC-ROC PAM50 (prueba)", es(meta["classifier"]["test"]["AUC-ROC"], 3))
        c[4].metric("Silueta K-Means (prueba)", es(meta["clustering"]["silhouette_test"], 3))

        st.subheader("La historia en tres ideas")
        st.markdown(
            "1. **El cáncer de mama no es una sola enfermedad.** Los subtipos moleculares difieren en receptores, "
            "proliferación y pronóstico: los tumores Basal y HER2 tienen el NPI más alto y la mayor mortalidad.\n"
            "2. **La genómica aporta lo que la clínica no ve.** Con solo ER, PR y HER2, un modelo reconoce mal el subtipo "
            "(F1 = 0,38); con la expresión génica llega a F1 = 0,79 en pacientes nunca vistas.\n"
            "3. **Los datos se agrupan solos en perfiles de riesgo.** Sin usar etiquetas, K-Means separa un grupo agresivo "
            "ER negativo (41 % de mortalidad por cáncer) de grupos luminales de mejor pronóstico.")

        g1, g2 = st.columns(2)
        sub_counts = f["subtipo"].value_counts().reindex([CLASS_NAMES[k] for k in CLASS_ORDER]).dropna()
        g1.plotly_chart(px.bar(sub_counts, orientation="h", labels={"value": "Pacientes", "index": ""},
                               title="Pacientes por subtipo PAM50").update_layout(showlegend=False, yaxis_title="", xaxis_title="Pacientes"),
                        width="stretch")
        g2.plotly_chart(px.box(f[f.npi_valid == 1], x="subtipo", y="nottingham_prognostic_index", color="subtipo",
                               category_orders={"subtipo": [CLASS_NAMES[k] for k in CLASS_ORDER]},
                               labels={"nottingham_prognostic_index": "NPI", "subtipo": ""},
                               title="NPI por subtipo (mayor = peor pronóstico)").update_layout(showlegend=False),
                        width="stretch")

        group_by = st.radio("Curvas de supervivencia (muerte por cáncer) por:", ["subtipo", "clúster", "receptor_group"],
                            horizontal=True, format_func=lambda x: {"subtipo": "Subtipo PAM50", "clúster": "Clúster",
                                                                    "receptor_group": "Grupo por receptores"}[x])
        fig = go.Figure()
        for g, d in f.groupby(group_by):
            if len(d) < 10:
                continue
            t, s = kaplan_meier(d["overall_survival_months"], d["muerte_por_cancer"])
            fig.add_trace(go.Scatter(x=t, y=s, mode="lines", line_shape="hv", name=f"{g} (n = {len(d)})"))
        fig.update_layout(title="Supervivencia específica por cáncer (Kaplan-Meier)", xaxis_title="Meses desde el diagnóstico",
                          yaxis_title="Probabilidad de no morir por cáncer", yaxis_range=[0, 1.02])
        st.plotly_chart(fig, width="stretch")
        st.caption("Las muertes por otras causas se tratan como censura. Curva descriptiva, sin ajuste por otras variables.")

# --------------------------------------------------------------------------------------
# 2. Exploración libre
# --------------------------------------------------------------------------------------
with tab_exp:
    num_opts = {"Edad al diagnóstico": "age_at_diagnosis", "NPI": "nottingham_prognostic_index",
                "Tamaño tumoral (mm)": "tumor_size", "Ganglios positivos": "lymph_nodes_examined_positive",
                "Número de mutaciones": "mutation_count", "Puntaje de proliferación": "proliferation_score",
                "Supervivencia (meses)": "overall_survival_months"}
    gene_opts = {g.upper(): g for g in genes}
    col_opts = {"Subtipo PAM50": "subtipo", "Clúster": "clúster", "Grupo por receptores": "receptor_group",
                "Estado vital": "estado_vital", "Cohorte": "cohort"}
    a, b, cc = st.columns(3)
    xv = a.selectbox("Eje X", list(num_opts) + list(gene_opts), index=5)
    yv = b.selectbox("Eje Y", list(num_opts) + list(gene_opts), index=1)
    cv = cc.selectbox("Color", list(col_opts))
    xcol = num_opts.get(xv, gene_opts.get(xv))
    ycol = num_opts.get(yv, gene_opts.get(yv))
    st.plotly_chart(px.scatter(f, x=xcol, y=ycol, color=col_opts[cv], opacity=.65, labels={xcol: xv, ycol: yv},
                               hover_data=["patient_id"], title=f"{yv} frente a {xv}"), width="stretch")
    st.caption("Los genes se expresan como z-scores de ARNm (0 = promedio de la cohorte).")

# --------------------------------------------------------------------------------------
# 3. Predicción en vivo
# --------------------------------------------------------------------------------------
with tab_pred:
    st.markdown("Elija una paciente del **conjunto de prueba** como punto de partida y modifique sus variables "
                "clínicas o la expresión de los genes con más peso en los modelos. Las demás variables (cientos de genes "
                "y mutaciones) conservan los valores de la paciente elegida.")
    test_ids = df.loc[df.split == "test", "patient_id"].tolist()
    pid = st.selectbox("Paciente de referencia (patient_id)", test_ids, index=0)
    base = df[df.patient_id == pid].iloc[[0]].copy()

    key_genes = ["erbb2", "gata3", "egfr", "erbb3", "csf1r", "notch1", "aurka", "ccnb1", "ncoa3", "bcl2"]
    left, right = st.columns([1, 1])
    with left:
        st.markdown("**Variables clínicas**")
        e = {}
        c1, c2 = st.columns(2)
        e["age_at_diagnosis"] = c1.number_input("Edad", 20.0, 100.0, float(base["age_at_diagnosis"].iloc[0]), 1.0)
        e["inferred_menopausal_state"] = c2.selectbox("Estado menopáusico", ["Pre", "Post"],
                                                      index=["Pre", "Post"].index(base["inferred_menopausal_state"].iloc[0]))
        e["er_status"] = c1.selectbox("Receptor de estrógeno", ["Positive", "Negative"],
                                      index=["Positive", "Negative"].index(base["er_status"].iloc[0]))
        e["pr_status"] = c2.selectbox("Receptor de progesterona", ["Positive", "Negative"],
                                      index=["Positive", "Negative"].index(base["pr_status"].iloc[0]))
        e["her2_status"] = c1.selectbox("HER2", ["Positive", "Negative"],
                                        index=["Positive", "Negative"].index(base["her2_status"].iloc[0]))
        grade0 = base["neoplasm_histologic_grade"].iloc[0]
        e["neoplasm_histologic_grade"] = c2.selectbox("Grado histológico", [1.0, 2.0, 3.0],
                                                      index=[1.0, 2.0, 3.0].index(grade0) if grade0 in (1.0, 2.0, 3.0) else 1)
        size0 = base["tumor_size"].iloc[0]
        e["tumor_size"] = c1.number_input("Tamaño tumoral (mm)", 1.0, 200.0, float(size0) if pd.notna(size0) else 25.0, 1.0)
        e["lymph_nodes_examined_positive"] = c2.number_input("Ganglios positivos", 0, 50,
                                                             int(base["lymph_nodes_examined_positive"].iloc[0]))
    with right:
        st.markdown("**Expresión génica (z-score)**")
        gcols = st.columns(2)
        for i, g in enumerate(key_genes):
            e[g] = gcols[i % 2].slider(g.upper(), -4.0, 6.0, float(np.clip(base[g].iloc[0], -4, 6)), 0.1)

    edited = base.copy()
    for k, v in e.items():
        edited[k] = v
    edited = add_derived_features(edited)
    reg_feats = M["meta"]["regressor"]["features"]
    clf_feats = M["meta"]["classifier"]["features"]

    npi_pred = float(M["reg"].predict(edited[reg_feats])[0])
    proba = M["clf"].predict_proba(edited[clf_feats])[0]
    classes = list(M["clf"].classes_)
    pred_cls = classes[int(np.argmax(proba))]
    z = M["clu"]["pca"].transform(M["clu"]["scaler"].transform(edited[genes]))
    clu_pred = int(M["clu"]["kmeans"].predict(z)[0])
    st.session_state["edited_pca"] = z[0, :3].tolist()

    st.divider()
    k = st.columns(4)
    real_npi = base["nottingham_prognostic_index"].iloc[0]
    k[0].metric("NPI estimado (Ridge)", es(npi_pred),
                delta=f"{'+' if npi_pred >= real_npi else '-'}{es(abs(npi_pred - real_npi))} frente al NPI registrado ({es(real_npi)})", delta_color="off")
    k[1].metric("Subtipo predicho", CLASS_NAMES[pred_cls], f"probabilidad {proba.max():.0%}", delta_color="off")
    k[2].metric("Subtipo real (PAM50)", CLASS_NAMES.get(base[TARGET_CLF].iloc[0], "No clasificado"))
    k[3].metric("Clúster asignado", f"Clúster {clu_pred}", CLUSTER_NAMES[clu_pred].split(" · ")[1], delta_color="off")
    prob_df = pd.DataFrame({"Subtipo": [CLASS_NAMES[c] for c in classes], "Probabilidad": proba}).sort_values("Probabilidad")
    st.plotly_chart(px.bar(prob_df, x="Probabilidad", y="Subtipo", orientation="h", range_x=[0, 1],
                           title="Probabilidad de cada subtipo (modelo final de clasificación)"), width="stretch")
    st.caption(f"Puntaje de proliferación recalculado: {es(edited['proliferation_score'].iloc[0])} "
               f"(promedio de {', '.join(g.upper() for g in PROLIFERATION_GENES)}). El NPI no usa el tamaño, los ganglios "
               "ni el grado como predictores, porque forman parte de su fórmula; esos campos solo afectan al clasificador.")
    if clu_pred == 2:
        st.warning("El clúster 2 se concentra en una sola cohorte (83 % C3) y probablemente refleja un efecto de lote técnico.")

# --------------------------------------------------------------------------------------
# 4. Segmentación (clustering)
# --------------------------------------------------------------------------------------
with tab_seg:
    st.markdown("K-Means con **K = 4** sobre las 10 primeras componentes principales de la expresión de 489 genes "
                "(elegidas por el método del codo). Los grupos se formaron sin usar ninguna variable clínica.")
    a, b = st.columns([1, 3])
    dim = a.radio("Vista", ["2D", "3D"], horizontal=True)
    color_by = a.radio("Colorear por", ["Clúster", "Subtipo PAM50"])
    show_me = a.checkbox("Mostrar la paciente editada en «Predicción en vivo»", value=True)
    ccol = "clúster" if color_by == "Clúster" else "subtipo"
    kw = dict(color=ccol, opacity=.6, hover_data=["patient_id", "subtipo", "clúster"],
              color_discrete_map=PALETTE if ccol == "clúster" else None,
              category_orders={"clúster": CLUSTER_ORDER, "subtipo": [CLASS_NAMES[k] for k in CLASS_ORDER]})
    if dim == "2D":
        fig = px.scatter(f, x="PC1", y="PC2", **kw)
    else:
        fig = px.scatter_3d(f, x="PC1", y="PC2", z="PC3", **kw)
        fig.update_traces(marker_size=3)
    if show_me and "edited_pca" in st.session_state:
        p = st.session_state["edited_pca"]
        if dim == "2D":
            fig.add_trace(go.Scatter(x=[p[0]], y=[p[1]], mode="markers", name="Paciente editada",
                                     marker=dict(symbol="star", size=18, color="gold", line=dict(width=1, color="black"))))
        else:
            fig.add_trace(go.Scatter3d(x=[p[0]], y=[p[1]], z=[p[2]], mode="markers", name="Paciente editada",
                                       marker=dict(symbol="diamond", size=8, color="gold")))
    fig.update_layout(height=560, title="Pacientes en el espacio de componentes principales")
    b.plotly_chart(fig, width="stretch")

    prof = f.groupby("clúster").agg(
        pacientes=("patient_id", "size"), edad_media=("age_at_diagnosis", "mean"),
        NPI_medio=("npi_ok", "mean"),
        ER_pos_pct=("er_status", lambda s: 100 * (s == "Positive").mean()),
        triple_negativo_pct=("triple_negative", lambda s: 100 * s.mean()),
        proliferacion=("proliferation_score", "mean"),
        TP53_mutado_pct=("tp53_mut", lambda s: 100 * s.mean()),
        mortalidad_cancer_pct=("muerte_por_cancer", lambda s: 100 * s.mean()),
        supervivencia_mediana=("overall_survival_months", "median")).round(1)
    st.subheader("Perfil promedio de cada clúster (pacientes filtradas)")
    st.dataframe(prof, width="stretch")
    comp = pd.crosstab(f["clúster"], f["subtipo"], normalize="index").mul(100).round(0)
    st.plotly_chart(px.imshow(comp, text_auto=True, color_continuous_scale="Purples", aspect="auto",
                              labels=dict(color="% del clúster"), title="Composición por subtipo PAM50 (%)"),
                    width="stretch")
    st.info("**Lectura.** El clúster 0 reúne tumores ER negativos, muy proliferativos y con TP53 mutado: es el de mayor "
            "mortalidad. Los clústeres 1 y 3 son luminales; el 1 tiene la proliferación más baja y la menor mortalidad. "
            "El clúster 2 proviene en un 83 % de la cohorte C3 y no tiene una firma génica coherente: probablemente es un "
            "efecto de lote técnico y no un subtipo biológico.")

# --------------------------------------------------------------------------------------
# 5. Modelos y explicabilidad
# --------------------------------------------------------------------------------------
with tab_mod:
    st.markdown("Evaluación final en el **conjunto de prueba** (15 % de las pacientes, nunca usado para entrenar ni "
                "para elegir modelos), con intervalos de confianza del 95 % por *bootstrap*.")
    if "smart" in metrics:
        st.subheader("Cumplimiento de los objetivos SMART")
        st.dataframe(metrics["smart"], width="stretch", hide_index=True)
    c1, c2 = st.columns(2)
    if "reg" in metrics:
        c1.subheader("Regresión del NPI")
        c1.dataframe(metrics["reg"].round(3), width="stretch", hide_index=True)
    if "clf" in metrics:
        c2.subheader("Clasificación del subtipo PAM50")
        c2.dataframe(metrics["clf"].round(3), width="stretch", hide_index=True)
    c1, c2 = st.columns(2)
    if "shap_reg" in metrics:
        d = metrics["shap_reg"].head(15).iloc[::-1]
        c1.plotly_chart(px.bar(d, x="mean_abs_shap", y=d.columns[0], orientation="h",
                               labels={"mean_abs_shap": "|SHAP| medio", d.columns[0]: ""},
                               title="Variables con más peso en el NPI (Ridge, SHAP)"), width="stretch")
    if "shap_clf" in metrics:
        d = metrics["shap_clf"].head(15).iloc[::-1]
        c2.plotly_chart(px.bar(d, x="mean_abs_shap", y=d.columns[0], orientation="h",
                               labels={"mean_abs_shap": "|SHAP| medio", d.columns[0]: ""},
                               title="Variables con más peso en el subtipo (XGBoost, SHAP)"), width="stretch")
    st.markdown("Los modelos redescubren la biología conocida: **ERBB2** identifica HER2, **GATA3** y **ERBB3** los tumores "
                "luminales, el **puntaje de proliferación** separa Luminal A de Luminal B, **EGFR** y **NOTCH1** el subtipo "
                "Basal, y genes inmunes como **CSF1R** el subtipo claudin-low.")
    for fig_name, cap in [("30_clasificacion_prueba.png", "Clasificación en prueba: F1 con IC 95 % y matrices de confusión"),
                          ("31_interpretabilidad_regresion.png", "Interpretabilidad de la regresión (coeficientes, SHAP y permutación)")]:
        p = config.FIGURES_DIR / fig_name
        if p.exists():
            st.image(str(p), caption=cap, width="stretch")
