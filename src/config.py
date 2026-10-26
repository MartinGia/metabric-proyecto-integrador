"""Configuración central del proyecto: rutas, semilla y definición de bloques de variables.

Todas las rutas se resuelven a partir de la raíz del repositorio, de modo que los
notebooks y scripts funcionan sin importar desde qué carpeta se ejecuten.
"""
from pathlib import Path

# --- Rutas -----------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT_DIR / "data" / "raw"
DATA_PROCESSED = ROOT_DIR / "data" / "processed"
REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
MODELS_DIR = ROOT_DIR / "models"
DOCS_DIR = ROOT_DIR / "docs"

RAW_FILE = DATA_RAW / "METABRIC_RNA_Mutation.csv"

# --- Reproducibilidad ------------------------------------------------------
RANDOM_STATE = 42

# --- Bloques de columnas del CSV de Kaggle ---------------------------------
# El archivo tiene 693 columnas: 31 clínicas, 489 z-scores de ARNm y 173 de mutación.
N_CLINICAL = 31
N_GENES = 489

# --- Variables objetivo propuestas -----------------------------------------
TARGET_REGRESSION = "nottingham_prognostic_index"      # continua
TARGET_CLASSIFICATION = "pam50_+_claudin-low_subtype"  # nominal, 6 clases válidas

# El NPI se calcula como 0,2 x tamaño tumoral (cm) + estadio ganglionar + grado
# histológico. Estas variables NO deben usarse como predictoras del NPI
# (fuga de información / target leakage).
NPI_COMPONENTS = ["tumor_size", "lymph_nodes_examined_positive", "neoplasm_histologic_grade"]


def split_blocks(df):
    """Devuelve las listas de columnas clínicas, de expresión génica y de mutaciones."""
    cols = list(df.columns)
    clinical = cols[:N_CLINICAL]
    genes = cols[N_CLINICAL:N_CLINICAL + N_GENES]
    mutations = cols[N_CLINICAL + N_GENES:]
    assert all(c.endswith("_mut") for c in mutations), "Bloque de mutaciones inesperado"
    return clinical, genes, mutations


def ensure_dirs():
    """Crea las carpetas de salida si no existen."""
    for d in (DATA_PROCESSED, FIGURES_DIR, METRICS_DIR, MODELS_DIR, DOCS_DIR):
        d.mkdir(parents=True, exist_ok=True)
