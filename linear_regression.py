import os
import json
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)

# ============================================================
# CIVICFIX - LOGISTIC REGRESSION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATASET_PATH = os.path.join(
    BASE_DIR,
    "datasetpreprocessed_v2.csv.csv"
)

RESULTS_PATH = os.path.join(
    BASE_DIR,
    "logistic_regression_results.json"
)


# ============================================================
# LOAD AND PREPARE DATA
# ============================================================

def prepare_data():

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(
            f"Dataset not found:\n{DATASET_PATH}"
        )

    df = pd.read_csv(DATASET_PATH)

    if "status" not in df.columns:
        raise ValueError(
            "Target column 'status' not found."
        )

    print("Original dataset shape:", df.shape)

    # --------------------------------------------------------
    # Target
    # --------------------------------------------------------

    y = df["status"].astype(str)

    # --------------------------------------------------------
    # Remove target and non-ML columns
    # --------------------------------------------------------

    columns_to_remove = [
        "status",
        "created_date",
        "closed_date",
        "incident_address",
        "resolution_description"
    ]

    X = df.drop(
        columns=columns_to_remove,
        errors="ignore"
    ).copy()

    # --------------------------------------------------------
    # Handle missing values
    # --------------------------------------------------------

    numerical_columns = X.select_dtypes(
        include=np.number
    ).columns

    categorical_columns = X.select_dtypes(
        exclude=np.number
    ).columns

    for column in numerical_columns:

        X[column] = X[column].fillna(
            X[column].median()
        )

    for column in categorical_columns:

        X[column] = X[column].fillna(
            "Not Available"
        )

    # --------------------------------------------------------
    # One-hot encode categorical features
    # --------------------------------------------------------

    X = pd.get_dummies(
        X,
        columns=categorical_columns,
        drop_first=False,
        dtype=float
    )

    # --------------------------------------------------------
    # Make sure everything is numeric
    # --------------------------------------------------------

    X = X.replace(
        [np.inf, -np.inf],
        np.nan
    )

    X = X.fillna(0)

    # --------------------------------------------------------
    # Encode target
    # --------------------------------------------------------

    label_encoder = LabelEncoder()

    y_encoded = label_encoder.fit_transform(y)

    print(
        "Number of classes:",
        len(label_encoder.classes_)
    )

    print(
        "Classes:",
        list(label_encoder.classes_)
    )

    print(
        "Final feature count:",
        X.shape[1]
    )

    return X, y_encoded, label_encoder


# ============================================================
# EVALUATE MODEL
# ============================================================

def evaluate_model(
    model,
    X_train,
    X_test,
    y_train,
    y_test
):

    model.fit(
        X_train,
        y_train
    )

    y_pred = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        y_pred
    )

    precision = precision_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        average="weighted",
        zero_division=0
    )

    cm = confusion_matrix(
        y_test,
        y_pred
    )

    return {
        "accuracy": round(
            float(accuracy),
            6
        ),
        "precision": round(
            float(precision),
            6
        ),
        "recall": round(
            float(recall),
            6
        ),
        "f1_score": round(
            float(f1),
            6
        ),
        "confusion_matrix": cm.tolist()
    }


# ============================================================
# MAIN
# ============================================================

def run_logistic_regression():

    print("=" * 70)
    print("CIVICFIX - LOGISTIC REGRESSION")
    print("=" * 70)

    # --------------------------------------------------------
    # Prepare data
    # --------------------------------------------------------

    X, y, label_encoder = prepare_data()

    print()

    # --------------------------------------------------------
    # Train / Test split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(
        "Training samples:",
        len(X_train)
    )

    print(
        "Testing samples:",
        len(X_test)
    )

    print()

    # ========================================================
    # 1. WITHOUT REGULARISATION
    # ========================================================

    print("-" * 70)
    print("1. LOGISTIC REGRESSION - WITHOUT REGULARISATION")
    print("-" * 70)

    # sklearn LogisticRegression always has regularisation
    # enabled by default. penalty=None disables it.

    model_without_regularisation = LogisticRegression(
        penalty=None,
        solver="lbfgs",
        max_iter=2000
    )

    no_reg_results = evaluate_model(
        model_without_regularisation,
        X_train,
        X_test,
        y_train,
        y_test
    )

    print(
        "Accuracy:",
        no_reg_results["accuracy"]
    )

    print(
        "Precision:",
        no_reg_results["precision"]
    )

    print(
        "Recall:",
        no_reg_results["recall"]
    )

    print(
        "F1 Score:",
        no_reg_results["f1_score"]
    )

    print()

    # ========================================================
    # 2. WITH L2 REGULARISATION
    # ========================================================

    print("-" * 70)
    print("2. LOGISTIC REGRESSION - WITH L2 REGULARISATION")
    print("-" * 70)

    model_with_regularisation = LogisticRegression(
        penalty="l2",
        C=1.0,
        solver="lbfgs",
        max_iter=2000
    )

    reg_results = evaluate_model(
        model_with_regularisation,
        X_train,
        X_test,
        y_train,
        y_test
    )

    print(
        "Penalty: L2"
    )

    print(
        "C:",
        1.0
    )

    print(
        "Accuracy:",
        reg_results["accuracy"]
    )

    print(
        "Precision:",
        reg_results["precision"]
    )

    print(
        "Recall:",
        reg_results["recall"]
    )

    print(
        "F1 Score:",
        reg_results["f1_score"]
    )

    print()

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    print("=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        f"Without Regularisation Accuracy: "
        f"{no_reg_results['accuracy']}"
    )

    print(
        f"With Regularisation Accuracy:    "
        f"{reg_results['accuracy']}"
    )

    # --------------------------------------------------------
    # Select best model
    # --------------------------------------------------------

    if (
        reg_results["accuracy"]
        >
        no_reg_results["accuracy"]
    ):

        best_model = "Logistic Regression with L2 Regularisation"
        best_accuracy = reg_results["accuracy"]

    else:

        best_model = "Logistic Regression without Regularisation"
        best_accuracy = no_reg_results["accuracy"]

    print()
    print(
        "Best Model:",
        best_model
    )

    print(
        "Best Accuracy:",
        best_accuracy
    )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results = {

        "model": "Logistic Regression",

        "target": "status",

        "classes": [
            str(value)
            for value in label_encoder.classes_
        ],

        "without_regularisation": no_reg_results,

        "with_regularisation": {
            **reg_results,
            "penalty": "L2",
            "C": 1.0
        },

        "best_model": best_model,

        "best_accuracy": best_accuracy

    }

    with open(
        RESULTS_PATH,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            results,
            file,
            indent=4
        )

    print()
    print(
        "Results saved to:"
    )

    print(
        RESULTS_PATH
    )

    print("=" * 70)
    print("LOGISTIC REGRESSION COMPLETED")
    print("=" * 70)

    return results


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    try:

        run_logistic_regression()

    except Exception as e:

        print()
        print("=" * 70)
        print("ERROR")
        print("=" * 70)
        print(str(e))
        print("=" * 70)