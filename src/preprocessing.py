"""Partición de datos y pipelines de preprocesamiento sin fuga de información.

Todo lo que aprende parámetros de los datos (cuantiles de winsorización, medianas,
modelos del IterativeImputer, medias y desviaciones del escalado, categorías del One-Hot)
vive dentro de un ColumnTransformer que se ajusta (fit) únicamente con el conjunto de
entrenamiento y luego solo transforma (transform) validación y prueba.
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from . import config


class Winsorizer(BaseEstimator, TransformerMixin):
    """Recorta cada columna a los percentiles [lower, upper] aprendidos en entrenamiento."""

    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.lo_ = np.nanquantile(X, self.lower, axis=0)
        self.hi_ = np.nanquantile(X, self.upper, axis=0)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float)
        return np.clip(X, self.lo_, self.hi_)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(input_features, dtype=object)


def make_splits(df, stratify_col, train=0.70, val=0.15, test=0.15, seed=config.RANDOM_STATE):
    """Partición estratificada entrenamiento/validación/prueba. Devuelve una Serie con la etiqueta."""
    assert abs(train + val + test - 1) < 1e-9
    idx = df.index
    idx_train, idx_tmp = train_test_split(idx, train_size=train, stratify=df[stratify_col], random_state=seed)
    rel_val = val / (val + test)
    idx_val, idx_test = train_test_split(idx_tmp, train_size=rel_val, stratify=df.loc[idx_tmp, stratify_col],
                                         random_state=seed)
    split = pd.Series("train", index=idx, name="split")
    split.loc[idx_val] = "val"
    split.loc[idx_test] = "test"
    return split


def build_preprocessor(blocks, imputer="iterative", knn_neighbors=5, seed=config.RANDOM_STATE):
    """Construye el ColumnTransformer a partir de los bloques de src.features.feature_sets.

    - num + ord: winsorización (1-99 %), estandarización e imputación multivariada en conjunto
      (IterativeImputer por defecto, elegido en el experimento de enmascaramiento de la actividad 3;
      KNNImputer como alternativa), para que los faltantes MAR del grado y la celularidad se
      imputen con información de las demás variables clínicas.
    - nom: imputación con la categoría 'Desconocido' y One-Hot (categorías con < 10 casos se agrupan).
    - genes: estandarización (los z-scores originales se calcularon con toda la cohorte).
    - bin: sin transformación.
    """
    transformers = []
    clinical = blocks.get("num", []) + blocks.get("ord", [])
    if clinical:
        transformers.append(("clinical", Pipeline([
            ("winsor", Winsorizer(0.01, 0.99)),
            ("scale", StandardScaler()),
            ("impute", IterativeImputer(max_iter=20, random_state=seed) if imputer == "iterative"
                       else KNNImputer(n_neighbors=knn_neighbors)),
        ]), clinical))
    if blocks.get("nom"):
        transformers.append(("nominal", Pipeline([
            ("impute", SimpleImputer(strategy="constant", fill_value="Desconocido")),
            ("onehot", OneHotEncoder(handle_unknown="infrequent_if_exist", min_frequency=10,
                                     sparse_output=False)),
        ]), blocks["nom"]))
    if blocks.get("genes"):
        transformers.append(("genes", StandardScaler(), blocks["genes"]))
    if blocks.get("bin"):
        transformers.append(("binary", "passthrough", blocks["bin"]))
    return ColumnTransformer(transformers, remainder="drop", verbose_feature_names_out=True)


def all_columns(blocks):
    return [c for cols in blocks.values() for c in cols]
