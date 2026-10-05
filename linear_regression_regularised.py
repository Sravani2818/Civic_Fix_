# ============================================================
# CIVICFIX - RIDGE REGRESSION
# WITH L2 REGULARISATION
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

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "dataset.csv"
)

RESULTS_PATH = os.path.join(
    BASE_DIR,
    "linear_regression_regularised_results.json"
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


    # Remove missing target rows

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
    # Remove unsuitable columns
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
    # Remove very high-cardinality categorical columns
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
# BUILD RIDGE PIPELINE
# ============================================================

def build_pipeline(X):

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


    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder="drop"
    )


    # --------------------------------------------------------
    # Ridge
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

    print("=" * 70)
    print(
        "CIVICFIX - RIDGE REGRESSION "
        "WITH L2 REGULARISATION"
    )
    print("=" * 70)


    df = load_dataset()

    X, y = prepare_data(
        df
    )


    # --------------------------------------------------------
    # Train/Test Split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(

        X,
        y,

        test_size=0.20,

        random_state=42

    )


    print(
        "Dataset shape:",
        df.shape
    )

    print(
        "Target column:",
        TARGET
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
    # Build and train
    # --------------------------------------------------------

    model = build_pipeline(
        X
    )


    print(
        "\nTraining Ridge Regression..."
    )


    model.fit(
        X_train,
        y_train
    )


    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    y_pred = model.predict(
        X_test
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mse = mean_squared_error(
        y_test,
        y_pred
    )

    rmse = np.sqrt(
        mse
    )

    mae = mean_absolute_error(
        y_test,
        y_pred
    )

    r2 = r2_score(
        y_test,
        y_pred
    )


    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print("\n" + "=" * 70)

    print(
        "RIDGE REGRESSION RESULTS"
    )

    print("=" * 70)

    print(
        "Target:",
        TARGET
    )

    print(
        "L2 alpha:",
        1.0
    )

    print(
        f"MAE : {mae:.6f}"
    )

    print(
        f"MSE : {mse:.6f}"
    )

    print(
        f"RMSE: {rmse:.6f}"
    )

    print(
        f"R²  : {r2:.6f}"
    )


    # --------------------------------------------------------
    # Sample predictions
    # --------------------------------------------------------

    sample_size = min(
        10,
        len(y_test)
    )


    predictions = [

        {

            "actual":
            round(
                float(actual),
                4
            ),

            "predicted":
            round(
                float(predicted),
                4
            )

        }

        for actual, predicted

        in zip(

            y_test.iloc[
                :sample_size
            ],

            y_pred[
                :sample_size
            ]

        )

    ]


    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    results = {

        "model":
        "Ridge Regression",

        "regularisation":
        "L2",

        "alpha":
        1.0,

        "target":
        TARGET,

        "dataset_rows":
        int(len(df)),

        "training_rows":
        int(len(X_train)),

        "testing_rows":
        int(len(X_test)),

        "features":
        int(X.shape[1]),

        "mae":
        round(
            float(mae),
            6
        ),

        "mse":
        round(
            float(mse),
            6
        ),

        "rmse":
        round(
            float(rmse),
            6
        ),

        "r2":
        round(
            float(r2),
            6
        ),

        "predictions":
        predictions

    }


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