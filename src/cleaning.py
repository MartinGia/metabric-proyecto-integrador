"""Limpieza determinística del conjunto METABRIC.

Todas las reglas de este módulo se aplican fila por fila y NO aprenden parámetros
de los datos (no calculan medias, cuantiles ni modas). Por eso pueden ejecutarse
antes de la partición entrenamiento/validación/prueba sin generar fuga de información.
Las imputaciones que sí aprenden de los datos se hacen en src/preprocessing.py,
ajustadas solo con el conjunto de entrenamiento.
"""
import numpy as np
import pandas as pd

from . import config

ORDINAL_CELLULARITY = {"Low": 1, "Moderate": 2, "High": 3}

# Columnas que se eliminan y motivo
DROP_COLUMNS = {
    "oncotree_code": "Redundante: mapeo uno a uno con cancer_type_detailed.",
    "cancer_type": "Constante tras excluir el único sarcoma.",
    "er_status_measured_by_ihc": "Redundante con er_status (completa); 118 discrepancias entre métodos.",
}


def node_stage(lymph_nodes):
    """Estadio ganglionar del NPI: 1 = sin ganglios, 2 = 1 a 3, 3 = 4 o más."""
    ln = pd.Series(lymph_nodes)
    out = pd.Series(np.select([ln == 0, ln <= 3], [1, 2], 3), index=ln.index, dtype="float")
    return out.where(ln.notna())


def npi_residual(df):
    """NPI menos estadio ganglionar menos término de tamaño (0,002 x mm, escala de la fuente).

    Si el NPI se calculó con la fórmula, el residuo es el grado histológico (1, 2 o 3).
    """
    size_term = 0.002 * df["tumor_size"].fillna(0)
    return df["nottingham_prognostic_index"] - node_stage(df["lymph_nodes_examined_positive"]) - size_term


def clean_metabric(raw):
    """Aplica las reglas de limpieza y devuelve (datos_limpios, bitácora_de_acciones)."""
    df = raw.copy()
    log = []

    def note(action, n, reason):
        log.append({"accion": action, "registros_afectados": int(n), "justificacion": reason})

    _, _, mut_cols = config.split_blocks(raw)

    # 1. Error tipográfico
    n = (df["er_status_measured_by_ihc"] == "Positve").sum()
    df["er_status_measured_by_ihc"] = df["er_status_measured_by_ihc"].replace({"Positve": "Positive"})
    note("Corregir 'Positve' por 'Positive'", n, "Error tipográfico en la fuente.")

    # 2. Registro fuera de la población de estudio
    sarcoma = df["cancer_type"] == "Breast Sarcoma"
    df = df.loc[~sarcoma].copy()
    note("Excluir el sarcoma de mama", sarcoma.sum(), "No es un carcinoma; caso único fuera de la población.")

    # 3. Valores indeterminados codificados como categoría
    n = (df["her2_status_measured_by_snp6"] == "UNDEF").sum()
    df["her2_status_measured_by_snp6"] = df["her2_status_measured_by_snp6"].replace({"UNDEF": np.nan})
    note("HER2 por SNP6 'UNDEF' a faltante", n, "Es un valor indeterminado, no una categoría.")

    # 4. Estadio 0 (in situ) incompatible con tumores invasivos medidos
    n = (df["tumor_stage"] == 0).sum()
    df.loc[df["tumor_stage"] == 0, "tumor_stage"] = np.nan
    note("Estadio 0 a faltante", n, "Estadio in situ incoherente con carcinomas invasivos de 11 a 65 mm.")

    # 5. Faltante estructural (MNAR): mutation_count vacío = cero mutaciones detectadas
    panel_mut = (df[mut_cols].astype(str) != "0").sum(axis=1)
    mask = df["mutation_count"].isna() & (panel_mut == 0)
    df.loc[mask, "mutation_count"] = 0
    note("mutation_count faltante a 0", mask.sum(),
         "MNAR estructural: el mínimo observado es 1 y todos los faltantes tienen 0 mutaciones en el panel.")

    # 6. Imputación deductiva del grado a partir de la fórmula del NPI
    resid = npi_residual(df).round(2)
    deducible = df["neoplasm_histologic_grade"].isna() & resid.isin([1.0, 2.0, 3.0])
    df.loc[deducible, "neoplasm_histologic_grade"] = resid[deducible]
    note("Deducir el grado desde el NPI", deducible.sum(),
         "NPI = grado + estadio ganglionar + 0,002 x tamaño; el grado se despeja sin estimar nada.")

    # 7. Validez del NPI como variable objetivo
    resid = npi_residual(df).round(2)
    df["npi_valid"] = (resid == df["neoplasm_histologic_grade"]).astype(int)
    note("Marcar NPI no válido (npi_valid = 0)", (df["npi_valid"] == 0).sum(),
         "El NPI se calculó sin el grado (residuo 0) o no es coherente con sus componentes.")

    # 8. Categorías y tipos
    df["cancer_type_detailed"] = df["cancer_type_detailed"].replace({"Breast": "Breast (NOS)"})
    df["cohort"] = "C" + df["cohort"].astype(int).astype(str)
    df["cellularity_ord"] = df["cellularity"].map(ORDINAL_CELLULARITY)
    note("Codificar cellularity como ordinal (1 a 3)", df["cellularity"].notna().sum(),
         "Orden natural Low < Moderate < High; mapeo fijo, sin aprendizaje.")

    # 9. Mutaciones: texto del cambio de proteína a indicador binario
    for c in mut_cols:
        df[c] = (df[c].astype(str) != "0").astype(int)
    note("Mutaciones a indicadores 0/1", len(df), f"{len(mut_cols)} columnas; el cambio de proteína no se usa en los modelos.")

    # 10. Columnas redundantes o constantes
    df = df.drop(columns=list(DROP_COLUMNS))
    for c, why in DROP_COLUMNS.items():
        note(f"Eliminar {c}", 0, why)

    return df.reset_index(drop=True), pd.DataFrame(log)
