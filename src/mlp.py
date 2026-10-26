"""Reconstrucción del MLP de PyTorch entrenado en el cuaderno 05 y generación de predicciones.

El preprocesador se vuelve a ajustar de forma determinística con las mismas filas que en el
cuaderno 05 (entrenamiento sin el 15 % reservado para parada temprana), de modo que no hace
falta serializar objetos de scikit-learn ligados a una versión concreta.

Uso desde la terminal (genera las probabilidades del conjunto indicado):
    python -m src.mlp test
"""
import json
import sys

import numpy as np
import pandas as pd

from . import config
from .modeling import load_processed, task_data
from .preprocessing import build_preprocessor, all_columns


def _build_model(n_in, hidden, dropout, n_out):
    import torch.nn as nn
    layers, prev = [], n_in
    for h in hidden:
        layers += [nn.Linear(prev, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
        prev = h
    layers.append(nn.Linear(prev, n_out))

    class MLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x)

    return MLP()


def predict_split(split="test"):
    """Devuelve un DataFrame con patient_id y las probabilidades por clase del MLP."""
    import torch
    with open(config.MODELS_DIR / "mlp_pam50_config.json", encoding="utf-8") as f:
        meta = json.load(f)
    df, fs = load_processed()
    clf_df, _ = task_data(df, "classification")
    cols = all_columns(fs["classification"])
    tr = clf_df[clf_df.split == "train"]
    fit_rows = tr[~tr["patient_id"].isin(meta["early_stopping_patient_ids"])]
    pre = build_preprocessor(fs["classification"]).fit(fit_rows[cols])
    target = clf_df[clf_df.split == split]
    X = pre.transform(target[cols]).astype(np.float32)
    if X.shape[1] != meta["n_in"]:
        raise ValueError(f"El preprocesador produjo {X.shape[1]} columnas y el MLP espera {meta['n_in']} "
                         f"(entrenado con scikit-learn {meta['sklearn']}).")
    model = _build_model(meta["n_in"], meta["hidden"], meta["dropout"], len(meta["classes"]))
    state = torch.load(config.MODELS_DIR / "mlp_pam50.pt", map_location="cpu")
    model.load_state_dict(state)
    model.eval()
    with torch.no_grad():
        proba = torch.softmax(model(torch.tensor(X)), dim=1).numpy()
    out = pd.DataFrame(proba, columns=meta["classes"])
    out.insert(0, "patient_id", target["patient_id"].to_numpy())
    return out


if __name__ == "__main__":
    split = sys.argv[1] if len(sys.argv) > 1 else "test"
    res = predict_split(split)
    path = config.METRICS_DIR / f"act6_mlp_{split}_proba.csv"
    res.to_csv(path, index=False)
    print(f"{len(res)} pacientes -> {path.relative_to(config.ROOT_DIR)}")
