# ============================================================
# CIVICFIX - LINEAR REGRESSION
# WITH REGULARISATION (RIDGE)
# ============================================================

import os
import json
import warnings

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

warnings.filterwarnings("ignore")


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "datasetpreprocessed_v2.csv.csv"
)

RESULTS_PATH = os.path.join(
    BASE_DIR,
    "linear_regression_regularised_results.json"
)


# ============================================================
# LOAD DATASET
# ============================================================

def load_dataset():

    if not os.path.exists(DATA_PATH):

        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}"
        )

    return pd.read_csv(DATA_PATH)


# ============================================================
# FIND NUMERICAL TARGET
# ============================================================

def find_target_column(df):

    preferred_targets = [
        "target",
        "amount",
        "count",
        "delay",
        "duration",
        "latitude",
        "longitude"
    ]

    for col in preferred_targets:

        if (
            col in df.columns
            and pd.api.types.is_numeric_dtype(df[col])
        ):

            return col

    if "status" in df.columns:

        if pd.api.types.is_numeric_dtype(df["status"]):

            return "status"

    numeric_columns = df.select_dtypes(
        include=np.number
    ).columns.tolist()

    if not numeric_columns:

        raise ValueError(
            "No numerical target column available."
        )

    return numeric_columns[-1]


# ============================================================
# PREPARE DATA
# ============================================================

def prepare_data(df):

    target_col = find_target_column(df)

    print("=" * 70)
    print("CIVICFIX - REGULARISED LINEAR REGRESSION")
    print("=" * 70)

    print("Dataset shape:", df.shape)
    print("Target column:", target_col)

    # Remove rows with missing target
    df = df.dropna(
        subset=[target_col]
    ).copy()

    y = df[target_col]

    X = df.drop(
        columns=[target_col]
    )

    # --------------------------------------------------------
    # Remove unsuitable columns
    # --------------------------------------------------------

    columns_to_remove = []

    for col in X.columns:

        col_lower = col.lower()

        if col_lower in [
            "id",
            "incident_id",
            "unique_key"
        ]:

            columns_to_remove.append(col)

        elif col_lower in [
            "created_date",
            "closed_date"
        ]:

            columns_to_remove.append(col)

        elif col_lower in [
            "incident_address",
            "resolution_description"
        ]:

            columns_to_remove.append(col)

    X = X.drop(
        columns=columns_to_remove,
        errors="ignore"
    )

    # --------------------------------------------------------
    # Remove extremely high-cardinality categorical columns
    # --------------------------------------------------------

    high_cardinality = []

    for col in X.select_dtypes(
        include=["object", "string", "category"]
    ).columns:

        ratio = (
            X[col].nunique(dropna=True)
            / max(len(X), 1)
        )

        if ratio > 0.80:

            high_cardinality.append(col)

    X = X.drop(
        columns=high_cardinality,
        errors="ignore"
    )

    print(
        "Features used:",
        X.shape[1]
    )

    return X, y, target_col


# ============================================================
# BUILD RIDGE REGRESSION PIPELINE
# ============================================================

def build_pipeline(X):

    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "string", "category"]
    ).columns.tolist()

    # --------------------------------------------------------
    # Numerical pipeline
    # --------------------------------------------------------

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median"
                )
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )

    # --------------------------------------------------------
    # Categorical pipeline
    # --------------------------------------------------------

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    transformers = []

    if numeric_features:

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features
            )
        )

    if categorical_features:

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )

    # --------------------------------------------------------
    # Ridge Regression
    # --------------------------------------------------------

    model = Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor
            ),
            (
                "regressor",
                Ridge(
                    alpha=1.0
                )
            )
        ]
    )

    return model


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():

    df = load_dataset()

    X, y, target_col = prepare_data(df)

    # --------------------------------------------------------
    # Train / Test Split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )

    print(
        "Training rows:",
        len(X_train)
    )

    print(
        "Testing rows:",
        len(X_test)
    )

    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    model = build_pipeline(X)

    print("\nTraining Ridge Regression...")

    model.fit(
        X_train,
        y_train
    )

    print("Training completed.")

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )

    # ========================================================
    # METRICS
    # ========================================================

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    r2 = r2_score(
        y_test,
        y_pred
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n" + "=" * 70)
    print("REGULARISED LINEAR REGRESSION RESULTS")
    print("=" * 70)

    print("Model : Ridge Regression")
    print("Alpha : 1.0")
    print("Target:", target_col)

    print(
        "MAE :",
        round(mae, 4)
    )

    print(
        "MSE :",
        round(mse, 4)
    )

    print(
        "RMSE:",
        round(rmse, 4)
    )

    print(
        "R²  :",
        round(r2, 4)
    )

    # ========================================================
    # SAMPLE PREDICTIONS
    # ========================================================

    sample_size = min(
        10,
        len(y_test)
    )

    comparison = pd.DataFrame(
        {
            "Actual": y_test.iloc[:sample_size].values,
            "Predicted": y_pred[:sample_size]
        }
    )

    print("\nSample Predictions:")
    print(comparison)

    # ========================================================
    # RESULTS FOR FLASK
    # ========================================================

    results = {

        "model": "Ridge Regression",

        "regularisation": "L2",

        "alpha": 1.0,

        "target": target_col,

        "dataset_rows": int(
            len(df)
        ),

        "training_rows": int(
            len(X_train)
        ),

        "testing_rows": int(
            len(X_test)
        ),

        "features": int(
            X.shape[1]
        ),

        "mae": round(
            float(mae),
            4
        ),

        "mse": round(
            float(mse),
            4
        ),

        "rmse": round(
            float(rmse),
            4
        ),

        "r2": round(
            float(r2),
            4
        ),

        "predictions": [

            {
                "actual": round(
                    float(actual),
                    4
                ),

                "predicted": round(
                    float(predicted),
                    4
                )
            }

            for actual, predicted
            in zip(
                y_test.iloc[:sample_size],
                y_pred[:sample_size]
            )
        ]
    }

    # ========================================================
    # SAVE RESULTS
    # ========================================================

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

    print(
        "\nResults saved to:",
        RESULTS_PATH
    )

    print("=" * 70)

    return results


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    train_model()