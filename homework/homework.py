# flake8: noqa: E501
"""
Modelo de clasificación para predicción de default de pagos.
"""

import gzip
import json
import os
import pickle
from pathlib import Path

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# ------------------------------------------------------------
# Ruta raíz del proyecto (homework.py está en carpeta "homework")
# ------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent

INPUT_DIR = ROOT_DIR / "files" / "input"
MODELS_DIR = ROOT_DIR / "files" / "models"
OUTPUT_DIR = ROOT_DIR / "files" / "output"


def load_dataset(path: str) -> pd.DataFrame:
    return pd.read_csv(path, compression="zip")


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    df = df.rename(columns={"default payment next month": "default"})
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    df = df.dropna()
    df.loc[df["EDUCATION"] > 4, "EDUCATION"] = 4
    return df


def create_pipeline() -> Pipeline:
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), categorical_features)
        ],
        remainder="passthrough",
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=42)),
        ]
    )


def create_estimator(pipeline: Pipeline) -> GridSearchCV:
    param_grid = {
        "classifier__n_estimators": [100, 200, 300],
        "classifier__max_depth": [10, 20, None],
        "classifier__min_samples_split": [2, 5, 10],
        "classifier__class_weight": ["balanced", None],
    }
    return GridSearchCV(
        pipeline,
        param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )


def calculate_precision_metrics(dataset_name: str, y_true, y_pred) -> dict:
    return {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1_score": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def calculate_confusion_metrics(dataset_name: str, y_true, y_pred) -> dict:
    cm = confusion_matrix(y_true, y_pred)
    return {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {
            "predicted_0": int(cm[0, 0]),
            "predicted_1": int(cm[0, 1]),
        },
        "true_1": {
            "predicted_0": int(cm[1, 0]),
            "predicted_1": int(cm[1, 1]),
        },
    }


def save_model(path: str, estimator: GridSearchCV):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with gzip.open(path, "wb") as f:
        pickle.dump(estimator, f)


def main():
    # Cargar y limpiar
    train_path = INPUT_DIR / "train_data.csv.zip"
    test_path = INPUT_DIR / "test_data.csv.zip"

    train_df = load_dataset(train_path)
    test_df = load_dataset(test_path)

    train_df = clean_dataset(train_df)
    test_df = clean_dataset(test_df)

    # Dividir
    x_train = train_df.drop(columns=["default"])
    y_train = train_df["default"]
    x_test = test_df.drop(columns=["default"])
    y_test = test_df["default"]

    # Pipeline y GridSearch
    pipeline = create_pipeline()
    estimator = create_estimator(pipeline)

    print("Iniciando búsqueda de hiperparámetros (10 folds)...")
    estimator.fit(x_train, y_train)

    best_model = estimator.best_estimator_

    # Guardar modelo
    model_path = MODELS_DIR / "model.pkl.gz"
    save_model(model_path, estimator)

    # Métricas
    train_pred = best_model.predict(x_train)
    test_pred = best_model.predict(x_test)

    train_metrics = calculate_precision_metrics("train", y_train, train_pred)
    test_metrics = calculate_precision_metrics("test", y_test, test_pred)
    train_cm = calculate_confusion_metrics("train", y_train, train_pred)
    test_cm = calculate_confusion_metrics("test", y_test, test_pred)

    # Guardar JSON
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_file = OUTPUT_DIR / "metrics.json"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(train_metrics) + "\n")
        f.write(json.dumps(test_metrics) + "\n")
        f.write(json.dumps(train_cm) + "\n")
        f.write(json.dumps(test_cm) + "\n")

    print("\n✅ Proceso completado con éxito")
    print(f"Mejores parámetros: {estimator.best_params_}")
    print(f"Balanced accuracy en entrenamiento: {train_metrics['balanced_accuracy']:.4f}")
    print(f"Balanced accuracy en prueba: {test_metrics['balanced_accuracy']:.4f}")


if __name__ == "__main__":
    main()