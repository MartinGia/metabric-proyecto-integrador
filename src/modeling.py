"""Carga de datos procesados y funciones de evaluación reutilizables en las fases 2 y 3."""
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error,
                             mean_squared_error, precision_score, r2_score, recall_score, roc_auc_score)
from sklearn.model_selection import KFold, StratifiedKFold, cross_validate

from . import config

CLASS_ORDER = ["LumA", "LumB", "Her2", "Basal", "claudin-low", "Normal"]


def load_processed():
    """Carga los datos limpios, la partición y los conjuntos de predictores generados en la actividad 3."""
    df = pd.read_csv(config.DATA_PROCESSED / "metabric_clean.csv", low_memory=False)
    splits = pd.read_csv(config.DATA_PROCESSED / "splits.csv")
    assert (df["patient_id"].to_numpy() == splits["patient_id"].to_numpy()).all(), "La partición no coincide"
    df["split"] = splits["split"].to_numpy()
    with open(config.DATA_PROCESSED / "feature_sets.json", encoding="utf-8") as f:
        fs = json.load(f)
    return df, fs


def task_data(df, task):
    """Filtra los registros válidos para cada tarea y devuelve (datos, nombre_objetivo)."""
    if task == "regression":
        return df[df["npi_valid"] == 1].copy(), config.TARGET_REGRESSION
    if task == "classification":
        return df[df[config.TARGET_CLASSIFICATION] != "NC"].copy(), config.TARGET_CLASSIFICATION
    raise ValueError(task)


def regression_metrics(y_true, y_pred):
    return {"R2": r2_score(y_true, y_pred),
            "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
            "MAE": mean_absolute_error(y_true, y_pred)}


def classification_metrics(y_true, y_pred, proba=None, classes=None):
    out = {"Accuracy": accuracy_score(y_true, y_pred),
           "Balanced_Accuracy": balanced_accuracy_score(y_true, y_pred),
           "Precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
           "Recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
           "F1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0)}
    if proba is not None:
        out["AUC_ROC_ovr"] = roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=classes)
    return out


def cv_regression(model, X, y, folds=5, seed=config.RANDOM_STATE):
    """Validación cruzada K-Fold sobre entrenamiento: media y desviación de R2, RMSE y MAE."""
    cv = KFold(folds, shuffle=True, random_state=seed)
    res = cross_validate(model, X, y, cv=cv, return_train_score=True,
                         scoring={"R2": "r2", "RMSE": "neg_root_mean_squared_error", "MAE": "neg_mean_absolute_error"})
    return _summarize_cv(res, negate=["RMSE", "MAE"])


def cv_classification(model, X, y, folds=5, seed=config.RANDOM_STATE):
    """Validación cruzada estratificada sobre entrenamiento."""
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    res = cross_validate(model, X, y, cv=cv, return_train_score=True,
                         scoring={"Accuracy": "accuracy", "Balanced_Accuracy": "balanced_accuracy",
                                  "F1_macro": "f1_macro", "AUC_ROC_ovr": "roc_auc_ovr"})
    return _summarize_cv(res)


def _summarize_cv(res, negate=()):
    out = {}
    for k, v in res.items():
        if not (k.startswith("test_") or k.startswith("train_")):
            continue
        split, metric = k.split("_", 1)
        v = -v if metric in negate else v
        out[f"CV_{split}_{metric}_mean"] = float(np.mean(v))
        out[f"CV_{split}_{metric}_std"] = float(np.std(v))
    return out
