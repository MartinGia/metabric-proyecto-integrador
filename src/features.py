"""Ingeniería de variables basada en conocimiento del dominio y definición de los
conjuntos de predictores por tarea.

Las variables derivadas se calculan fila por fila con listas de genes y reglas fijadas
por la literatura (no por los datos), de modo que no introducen fuga de información.
"""
import numpy as np
import pandas as pd

# Genes de proliferación disponibles en el panel. AURKA y CCNB1 forman parte del grupo de
# proliferación de Oncotype DX (Paik et al., 2004); CDK1, CHEK1, CCNE1, CDC25A y E2F2 son
# reguladores del ciclo celular del Genomic Grade Index (Sotiriou et al., 2006).
PROLIFERATION_GENES = ["aurka", "ccnb1", "cdk1", "chek1", "ccne1", "cdc25a", "e2f2"]

# Genes conductores del cáncer de mama reportados en METABRIC (Pereira et al., 2016)
DRIVER_GENES = ["pik3ca", "tp53", "gata3", "map3k1", "cdh1", "kmt2c", "akt1", "pten",
                "cbfb", "runx1", "map2k4", "ncor1", "nf1", "rb1", "erbb2"]

AGE_BINS = [0, 40, 50, 60, 70, 120]
AGE_LABELS = ["<=40", "41-50", "51-60", "61-70", ">70"]


def tnm_stage_rule(tumor_size, lymph_nodes):
    """Estadio anatómico aproximado (AJCC) a partir de T (tamaño) y N (ganglios); asume M0."""
    T = np.select([tumor_size <= 20, tumor_size <= 50], [1, 2], 3).astype(float)
    T[np.asarray(tumor_size.isna())] = np.nan
    N = np.select([lymph_nodes == 0, lymph_nodes <= 3, lymph_nodes <= 9], [0, 1, 2], 3)
    stage = np.where(N >= 2, 3, np.where(T == 3, np.where(N == 0, 2, 3),
                                         np.where((T == 2) | (N == 1), 2, 1))).astype(float)
    stage[np.isnan(T)] = np.nan
    return pd.Series(stage, index=tumor_size.index)


def add_derived_features(df):
    """Agrega las variables derivadas. Devuelve una copia del DataFrame."""
    out = df.copy()

    # 1. Puntaje de proliferación: promedio de z-scores de genes del ciclo celular
    out["proliferation_score"] = out[PROLIFERATION_GENES].mean(axis=1)

    # 2. Grupo por receptores (subrogado clínico de St. Gallen) e indicador triple negativo
    hr_pos = (out["er_status"] == "Positive") | (out["pr_status"] == "Positive")
    her2_pos = out["her2_status"] == "Positive"
    out["receptor_group"] = np.select([her2_pos, hr_pos], ["HER2+", "HR+/HER2-"], "Triple negativo")
    out["triple_negative"] = (out["receptor_group"] == "Triple negativo").astype(int)

    # 3. Carga mutacional en genes conductores
    out["driver_mutation_count"] = out[[g + "_mut" for g in DRIVER_GENES]].sum(axis=1)

    # 4. Transformaciones logarítmicas de variables de conteo y tamaño (asimetría > 3)
    for c in ["tumor_size", "lymph_nodes_examined_positive", "mutation_count"]:
        out[f"log_{c}"] = np.log1p(out[c])

    # 5. Categoría ganglionar y grupos de edad (para perfiles y filtros del dashboard)
    ln = out["lymph_nodes_examined_positive"]
    out["node_category"] = np.select([ln == 0, ln <= 3], ["N0", "N1-3"], "N4+")
    out["age_group"] = pd.cut(out["age_at_diagnosis"], AGE_BINS, labels=AGE_LABELS).astype(str)

    # 6. Estadio por regla TNM, solo para describir y filtrar (no se usa como predictor)
    out["tumor_stage_rule"] = tnm_stage_rule(out["tumor_size"], ln)
    out["tumor_stage_filled"] = out["tumor_stage"].fillna(out["tumor_stage_rule"])
    return out


DERIVED = ["proliferation_score", "receptor_group", "triple_negative", "driver_mutation_count",
           "log_tumor_size", "log_lymph_nodes_examined_positive", "log_mutation_count",
           "node_category", "age_group", "tumor_stage_rule", "tumor_stage_filled"]

# Variables que nunca se usan como predictoras y motivo
NEVER_PREDICTORS = {
    "patient_id": "Identificador.",
    "cohort": "Lote de reclutamiento; se usa solo para analizar faltantes.",
    "overall_survival_months": "Desenlace posterior al diagnóstico.",
    "overall_survival": "Desenlace posterior al diagnóstico.",
    "death_from_cancer": "Desenlace posterior al diagnóstico.",
    "chemotherapy": "Tratamiento decidido a partir del pronóstico y el subtipo (causalidad inversa).",
    "hormone_therapy": "Tratamiento decidido a partir del pronóstico y el subtipo.",
    "radio_therapy": "Tratamiento decidido a partir del pronóstico y el subtipo.",
    "type_of_breast_surgery": "Tratamiento decidido a partir del pronóstico y el subtipo.",
    "integrative_cluster": "Etiqueta derivada de la misma expresión génica y número de copias.",
    "3-gene_classifier_subtype": "Etiqueta derivada de la misma expresión génica.",
    "primary_tumor_laterality": "Sin relación biológica esperada (independiente del subtipo, p = 0,35).",
    "tumor_stage": "26 % de faltantes MAR y redundante con tamaño y ganglios (regla TNM).",
    "tumor_stage_rule": "Derivada de tamaño y ganglios; solo descriptiva.",
    "tumor_stage_filled": "Solo descriptiva (filtros del dashboard).",
    "npi_valid": "Indicador de calidad del objetivo.",
    "cellularity": "Se usa su versión ordinal cellularity_ord.",
    "cancer_type_detailed": "Redundante con tumor_other_histologic_subtype.",
}

NPI_LEAKAGE = {
    "tumor_size": "Componente del NPI.", "log_tumor_size": "Componente del NPI.",
    "lymph_nodes_examined_positive": "Componente del NPI.",
    "log_lymph_nodes_examined_positive": "Componente del NPI.",
    "node_category": "Componente del NPI.", "neoplasm_histologic_grade": "Componente del NPI.",
}


def feature_sets(df, gene_cols, mut_cols, min_mut_prevalence=0.02):
    """Devuelve los bloques de predictores por tarea.

    Bloques: num (cuantitativas), ord (ordinales), nom (nominales), bin (binarias), genes.
    """
    frequent_muts = [c for c in mut_cols if df[c].mean() >= min_mut_prevalence]
    base = {
        "num": ["age_at_diagnosis", "log_mutation_count", "driver_mutation_count", "proliferation_score"],
        "ord": ["cellularity_ord"],
        "nom": ["inferred_menopausal_state", "er_status", "pr_status", "her2_status",
                "her2_status_measured_by_snp6", "tumor_other_histologic_subtype", "receptor_group"],
        "bin": ["triple_negative"] + frequent_muts,
        "genes": list(gene_cols),
    }
    regression = {k: list(v) for k, v in base.items()}

    classification = {k: list(v) for k, v in base.items()}
    classification["num"] += ["log_tumor_size", "log_lymph_nodes_examined_positive", "nottingham_prognostic_index"]
    classification["ord"] += ["neoplasm_histologic_grade"]

    clustering = {"genes": list(gene_cols)}
    return {"regression": regression, "classification": classification, "clustering": clustering}
