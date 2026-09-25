"""CivicFix - Logistic Regression variants.

Runs the same train/test split twice so the dashboard can show:
1. Logistic Regression without regularisation (penalty=None)
2. Logistic Regression with L2 regularisation (C=1.0)

Results are written to logistic_regression_results.json for Flask.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / 'dataset.csv'
RESULTS_PATH = BASE_DIR / 'logistic_regression_results.json'


def prepare_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f'Dataset not found: {DATA_PATH}')

    data = pd.read_csv(DATA_PATH)
    target = 'status'
    if target not in data.columns:
        raise ValueError("Target column 'status' not found.")

    X = data.drop(columns=[target], errors='ignore').copy()
    y = data[target].astype(str)

    X = X.drop(columns=[
        'created_date', 'closed_date', 'incident_address',
        'resolution_description'
    ], errors='ignore')

    X = pd.get_dummies(X, drop_first=True, dtype=float)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    classes = sorted(y.unique())
    class_mapping = {label: i for i, label in enumerate(classes)}
    y_encoded = y.map(class_mapping).astype(int)

    return X, y_encoded, classes


def evaluate(model, X_train, X_test, y_train, y_test):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        'accuracy': round(float(accuracy_score(y_test, y_pred)), 6),
        'precision': round(float(precision_score(y_test, y_pred, average='weighted', zero_division=0)), 6),
        'recall': round(float(recall_score(y_test, y_pred, average='weighted', zero_division=0)), 6),
        'f1_score': round(float(f1_score(y_test, y_pred, average='weighted', zero_division=0)), 6),
        'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
    }


def run():
    print('=' * 70)
    print('CIVICFIX - LOGISTIC REGRESSION VARIANTS')
    print('=' * 70)

    X, y, classes = prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f'Features: {X.shape[1]}')
    print(f'Training rows: {len(X_train)}')
    print(f'Testing rows: {len(X_test)}')

    print('\n1) Without regularisation')
    no_reg = evaluate(
        LogisticRegression(penalty=None, solver='lbfgs', max_iter=2000),
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    print('\n2) With L2 regularisation')
    with_reg = evaluate(
        LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=2000),
        X_train_scaled, X_test_scaled, y_train, y_test
    )

    results = {
        'model': 'Logistic Regression',
        'target': 'status',
        'classes': [str(x) for x in classes],
        'training_samples': int(len(X_train)),
        'testing_samples': int(len(X_test)),
        'features': int(X.shape[1]),
        'without_regularisation': no_reg,
        'with_regularisation': {**with_reg, 'penalty': 'L2', 'C': 1.0},
    }

    RESULTS_PATH.write_text(json.dumps(results, indent=4), encoding='utf-8')
    print(f'\nResults saved to: {RESULTS_PATH}')
    return results


if __name__ == '__main__':
    run()
