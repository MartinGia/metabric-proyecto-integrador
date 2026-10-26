"""Funciones reutilizables para el análisis exploratorio de datos (EDA)."""
import numpy as np
import pandas as pd
from scipy import stats


def descriptive_table(df, cols):
    """Tendencia central, dispersión, asimetría y curtosis para variables cuantitativas."""
    rows = []
    for c in cols:
        s = df[c].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        mode = s.mode()
        rows.append({
            "variable": c,
            "n": int(s.size),
            "media": s.mean(),
            "mediana": s.median(),
            "moda": mode.iloc[0] if not mode.empty else np.nan,
            "desv_est": s.std(),
            "varianza": s.var(),
            "coef_var_%": 100 * s.std() / s.mean() if s.mean() != 0 else np.nan,
            "min": s.min(),
            "Q1": q1,
            "Q3": q3,
            "max": s.max(),
            "IQR": q3 - q1,
            "asimetria": stats.skew(s, bias=False),
            "curtosis_exceso": stats.kurtosis(s, fisher=True, bias=False),
        })
    return pd.DataFrame(rows).set_index("variable")


def frequency_table(series):
    """Tabla de frecuencias absolutas y relativas (incluye faltantes)."""
    counts = series.value_counts(dropna=False)
    out = pd.DataFrame({
        "frecuencia": counts,
        "porcentaje": (100 * counts / counts.sum()).round(2),
    })
    out["porcentaje_acumulado"] = out["porcentaje"].cumsum().round(2)
    out.index = out.index.map(lambda v: "Faltante (NaN)" if pd.isna(v) else v)
    return out


def missing_table(df):
    """Número y porcentaje exacto de valores faltantes por variable."""
    n_miss = df.isna().sum()
    out = pd.DataFrame({
        "faltantes": n_miss,
        "porcentaje": (100 * n_miss / len(df)).round(3),
    })
    return out.sort_values("porcentaje", ascending=False)


def outliers_iqr(df, cols, k=1.5):
    """Cuenta atípicos por la regla de Tukey (fuera de [Q1 - k*IQR, Q3 + k*IQR])."""
    rows = []
    for c in cols:
        s = df[c].dropna()
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        lo, hi = q1 - k * iqr, q3 + k * iqr
        n_out = int(((s < lo) | (s > hi)).sum())
        rows.append({"variable": c, "lim_inferior": lo, "lim_superior": hi,
                     "atipicos_IQR": n_out, "porcentaje_IQR": round(100 * n_out / s.size, 2)})
    return pd.DataFrame(rows).set_index("variable")


def outliers_zscore(df, cols, threshold=3.0):
    """Cuenta atípicos con |z| > umbral."""
    rows = []
    for c in cols:
        s = df[c].dropna()
        z = (s - s.mean()) / s.std()
        n_out = int((z.abs() > threshold).sum())
        rows.append({"variable": c, "atipicos_Z": n_out,
                     "porcentaje_Z": round(100 * n_out / s.size, 2)})
    return pd.DataFrame(rows).set_index("variable")


def cramers_v(contingency):
    """V de Cramér con corrección de sesgo (Bergsma, 2013)."""
    chi2 = stats.chi2_contingency(contingency, correction=False)[0]
    n = contingency.to_numpy().sum()
    r, k = contingency.shape
    phi2 = chi2 / n
    phi2corr = max(0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return np.sqrt(phi2corr / denom) if denom > 0 else np.nan


def chi2_tests(df, pairs):
    """Prueba de independencia chi-cuadrado y V de Cramér para pares de variables cualitativas."""
    rows = []
    for a, b in pairs:
        ct = pd.crosstab(df[a], df[b])
        chi2, p, dof, expected = stats.chi2_contingency(ct)
        rows.append({
            "variable_1": a, "variable_2": b, "chi2": chi2, "gl": dof, "p_valor": p,
            "V_Cramer": cramers_v(ct),
            "%_celdas_esperado<5": round(100 * (expected < 5).mean(), 1),
        })
    return pd.DataFrame(rows)


def missingness_tests(df, target_col, cat_refs, num_refs):
    """Compara el indicador de faltante de target_col con variables de referencia.

    Usa chi-cuadrado para las cualitativas y Mann-Whitney U para las cuantitativas.
    Si alguna asociación es significativa, se descarta MCAR (el faltante depende de
    variables observadas, compatible con MAR).
    """
    miss = df[target_col].isna()
    rows = []
    for c in cat_refs:
        if c == target_col:
            continue
        ct = pd.crosstab(df[c], miss)
        if ct.shape[1] < 2:
            continue
        p = stats.chi2_contingency(ct)[1]
        rows.append({"variable": target_col, "referencia": c, "prueba": "Chi-cuadrado", "p_valor": p})
    for c in num_refs:
        if c == target_col:
            continue
        a, b = df.loc[miss, c].dropna(), df.loc[~miss, c].dropna()
        if len(a) == 0 or len(b) == 0:
            continue
        p = stats.mannwhitneyu(a, b)[1]
        rows.append({"variable": target_col, "referencia": c, "prueba": "Mann-Whitney U", "p_valor": p})
    return pd.DataFrame(rows)
