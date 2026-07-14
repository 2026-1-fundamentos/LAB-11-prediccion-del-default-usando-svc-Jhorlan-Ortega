# flake8: noqa: E501
import gzip
import json
import os
import pickle
import zipfile

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

# ──────────────────────────────────────────────────────────────────────────────
# Paso 1: Cargar y limpiar datos
# ──────────────────────────────────────────────────────────────────────────────

def load_and_clean(path: str) -> pd.DataFrame:
    """Carga un CSV desde un .zip, aplica limpieza según el enunciado."""
    with zipfile.ZipFile(path) as z:
        csv_name = z.namelist()[0]
        with z.open(csv_name) as f:
            df = pd.read_csv(f)

    # Renombrar columna objetivo
    df = df.rename(columns={"default payment next month": "default"})

    # Remover columna ID
    df = df.drop(columns=["ID"])

    # Eliminar registros con información no disponible (valores 0 en categoricas)
    df = df[df["EDUCATION"] != 0]
    df = df[df["MARRIAGE"] != 0]

    # EDUCATION > 4 → 4 (others)
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)

    return df


train_df = load_and_clean("files/input/train_data.csv.zip")
test_df  = load_and_clean("files/input/test_data.csv.zip")

print(f"Train shape: {train_df.shape} | Test shape: {test_df.shape}")
print(f"Distribución target (train):\n{train_df['default'].value_counts()}")

# ──────────────────────────────────────────────────────────────────────────────
# Paso 2: Separar features y target
# ──────────────────────────────────────────────────────────────────────────────

x_train = train_df.drop(columns=["default"])
y_train = train_df["default"]

x_test  = test_df.drop(columns=["default"])
y_test  = test_df["default"]

# ──────────────────────────────────────────────────────────────────────────────
# Paso 3: Construir el pipeline
# Orden: OneHotEncoder → PCA → StandardScaler → SelectKBest → SVC
# ──────────────────────────────────────────────────────────────────────────────

categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
numerical_features   = [c for c in x_train.columns if c not in categorical_features]

# Paso 3.1 — OneHotEncoder solo en las columnas categóricas,
#             passthrough en las numéricas
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_features),
        ("num", "passthrough", numerical_features),
    ]
)

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),          # 1 — OneHotEncoder
        ("pca",          PCA()),                  # 2 — PCA (todas las componentes)
        ("scaler",       StandardScaler()),       # 3 — Estandarización
        ("selector",     SelectKBest(f_classif)), # 4 — Selección de K mejores
        ("svc",          SVC(random_state=42)),   # 5 — SVM
    ]
)

# ──────────────────────────────────────────────────────────────────────────────
# Paso 4: Optimización de hiperparámetros con GridSearchCV (10-fold CV)
# ──────────────────────────────────────────────────────────────────────────────

param_grid = {
    "selector__k": [10, 15, 20],
    "svc__C":      [1, 10],
    "svc__kernel": ["rbf"],
    "svc__gamma":  ["scale"],
}

model = GridSearchCV(
    estimator=pipeline,
    param_grid=param_grid,
    cv=10,
    scoring="balanced_accuracy",
    n_jobs=-1,
    refit=True,
)

print("\nEntrenando modelo con GridSearchCV...")
model.fit(x_train, y_train)

print(f"Mejores parámetros: {model.best_params_}")
print(f"Mejor score CV: {model.best_score_:.4f}")

# ──────────────────────────────────────────────────────────────────────────────
# Paso 5: Guardar modelo comprimido con gzip
# ──────────────────────────────────────────────────────────────────────────────

os.makedirs("files/models", exist_ok=True)
with gzip.open("files/models/model.pkl.gz", "wb") as f:
    pickle.dump(model, f)

print("Modelo guardado en files/models/model.pkl.gz")

# ──────────────────────────────────────────────────────────────────────────────
# Pasos 6 y 7: Métricas y matrices de confusión
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(model, x, y, dataset_name: str):
    """Calcula métricas y matriz de confusión para un conjunto de datos."""
    y_pred = model.predict(x)

    metrics = {
        "type":              "metrics",
        "dataset":           dataset_name,
        "precision":         float(precision_score(y, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y, y_pred)),
        "recall":            float(recall_score(y, y_pred, zero_division=0)),
        "f1_score":          float(f1_score(y, y_pred, zero_division=0)),
    }

    cm = confusion_matrix(y, y_pred)
    cm_dict = {
        "type":    "cm_matrix",
        "dataset": dataset_name,
        "true_0":  {"predicted_0": int(cm[0, 0]), "predicted_1": int(cm[0, 1])},
        "true_1":  {"predicted_0": int(cm[1, 0]), "predicted_1": int(cm[1, 1])},
    }

    return metrics, cm_dict


train_metrics, train_cm = compute_metrics(model, x_train, y_train, "train")
test_metrics,  test_cm  = compute_metrics(model, x_test,  y_test,  "test")

# Mostrar resultados en consola
for m in [train_metrics, test_metrics]:
    print(f"\n[{m['dataset'].upper()}]")
    print(f"  Precision:         {m['precision']:.4f}")
    print(f"  Balanced accuracy: {m['balanced_accuracy']:.4f}")
    print(f"  Recall:            {m['recall']:.4f}")
    print(f"  F1-score:          {m['f1_score']:.4f}")

# Guardar en metrics.json (una línea JSON por registro)
os.makedirs("files/output", exist_ok=True)
with open("files/output/metrics.json", "w", encoding="utf-8") as f:
    for record in [train_metrics, test_metrics, train_cm, test_cm]:
        f.write(json.dumps(record) + "\n")

print("\nMétricas guardadas en files/output/metrics.json")
print("¡Listo!")