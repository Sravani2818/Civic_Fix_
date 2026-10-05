# ============================================================
# CIVICFIX - LINEAR REGRESSION
# WITHOUT AND WITH REGULARISATION
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
from sklearn.linear_model import LinearRegression, Ridge, Lasso
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
    "dataset.csv"
)

RESULTS_PATH = os.path.join(
    BASE_DIR,
    "linear_regression_results.json"
)

TARGET = "latitude"


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
# PREPARE DATA
# ============================================================

def prepare_data(df):

    if TARGET not in df.columns:

        raise ValueError(
            f"Target column '{TARGET}' not found."
        )

    # --------------------------------------------------------
    # Remove rows where target is missing
    # --------------------------------------------------------

    df = df.dropna(
        subset=[TARGET]
    ).copy()

    y = pd.to_numeric(
        df[TARGET],
        errors="coerce"
    )

    valid = y.notna()

    df = df.loc[valid].copy()

    y = y.loc[valid]


    # --------------------------------------------------------
    # Remove target and unsuitable columns
    # --------------------------------------------------------

    columns_to_remove = [

        TARGET,

        "status",

        "created_date",

        "closed_date",

        "incident_address",

        "resolution_description"

    ]

    X = df.drop(
        columns=columns_to_remove,
        errors="ignore"
    )


    # --------------------------------------------------------
    # Remove extremely high-cardinality columns
    # --------------------------------------------------------

    high_cardinality = []

    categorical_columns = X.select_dtypes(
        include=[
            "object",
            "string",
            "category"
        ]
    ).columns

    for column in categorical_columns:

        ratio = (
            X[column].nunique(
                dropna=True
            )
            /
            max(len(X), 1)
        )

        if ratio > 0.80:

            high_cardinality.append(
                column
            )

    X = X.drop(
        columns=high_cardinality,
        errors="ignore"
    )

    return X, y


# ============================================================
# BUILD PREPROCESSOR
# ============================================================

def build_preprocessor(X):

    numeric_features = X.select_dtypes(
        include=np.number
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=[
            "object",
            "string",
            "category"
        ]
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


    return ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


# ============================================================
# EVALUATION
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

    predictions = model.predict(
        X_test
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    r2 = r2_score(
        y_test,
        predictions
    )

    return {

        "mse": round(
            float(mse),
            6
        ),

        "rmse": round(
            float(rmse),
            6
        ),

        "mae": round(
            float(mae),
            6
        ),

        "r2_score": round(
            float(r2),
            6
        )

    }


# ============================================================
# MAIN
# ============================================================

def run_linear_regression():

    print("=" * 70)
    print("CIVICFIX - LINEAR REGRESSION")
    print("=" * 70)


    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    df = load_dataset()

    X, y = prepare_data(
        df
    )


    print(
        "Original dataset shape:",
        df.shape
    )

    print(
        "Target column:",
        TARGET
    )

    print(
        "Rows used:",
        len(X)
    )

    print(
        "Features before encoding:",
        X.shape[1]
    )


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
    # Preprocessor
    # --------------------------------------------------------

    preprocessor = build_preprocessor(
        X
    )


    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    models = {

        "without_regularisation": (

            "LinearRegression",

            LinearRegression()

        ),

        "ridge_regularisation": (

            "Ridge",

            Ridge(
                alpha=1.0
            )

        ),

        "lasso_regularisation": (

            "Lasso",

            Lasso(
                alpha=0.001,
                max_iter=10000
            )

        )

    }


    results = {}


    # --------------------------------------------------------
    # Train each model
    # --------------------------------------------------------

    for key, (
        model_name,
        estimator
    ) in models.items():

        print()

        print(
            "Training:",
            model_name
        )


        pipeline = Pipeline(
            steps=[

                (
                    "preprocessor",
                    preprocessor
                ),

                (
                    "model",
                    estimator
                )

            ]
        )


        metrics = evaluate_model(

            pipeline,

            X_train,
            X_test,

            y_train,
            y_test

        )


        metrics["model"] = model_name

        results[key] = metrics


        print(
            "MSE :",
            metrics["mse"]
        )

        print(
            "RMSE:",
            metrics["rmse"]
        )

        print(
            "MAE :",
            metrics["mae"]
        )

        print(
            "R²  :",
            metrics["r2_score"]
        )


    # --------------------------------------------------------
    # Best model
    # --------------------------------------------------------

    best_key = max(

        results,

        key=lambda key:
        results[key]["r2_score"]

    )


    results["best_model"] = (
        results[best_key]["model"]
    )

    results["best_r2_score"] = (
        results[best_key]["r2_score"]
    )

    results["target"] = TARGET

    results["training_rows"] = (
        len(X_train)
    )

    results["testing_rows"] = (
        len(X_test)
    )


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

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
        "Results saved to:",
        RESULTS_PATH
    )

    print(
        "Best regression model:",
        results["best_model"]
    )

    print("=" * 70)


    return results


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    run_linear_regression()