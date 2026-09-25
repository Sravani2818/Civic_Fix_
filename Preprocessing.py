import os
import pandas as pd
import numpy as np

from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


# ============================================================
# CONFIGURATION
# ============================================================



CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.csv")
OUTPUT_PATH = "datasetpreprocessed_v2.csv"


# ============================================================
# LOAD DATA
# ============================================================

def load_dataset(path=CSV_PATH):

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    return pd.read_csv(path)


# ============================================================
# PREPROCESS DATA
# ============================================================

def preprocess_data(path=CSV_PATH):

    data = load_dataset(path)

    # --------------------------------------------------------
    # ORIGINAL INFORMATION
    # --------------------------------------------------------

    original_rows = len(data)
    original_columns = len(data.columns)

    missing_before = int(
        data.isnull().sum().sum()
    )

    duplicates_before = int(
        data.duplicated().sum()
    )

    # --------------------------------------------------------
    # HANDLE MISSING VALUES
    # --------------------------------------------------------

    numeric_cols = data.select_dtypes(
        include=np.number
    ).columns

    for col in numeric_cols:

        if data[col].isnull().sum() > 0:

            data[col] = data[col].fillna(
                data[col].median()
            )

    categorical_cols = data.select_dtypes(
        include=["object", "string", "category"]
    ).columns

    for col in categorical_cols:

        if data[col].isnull().sum() > 0:

            if col in [
                "closed_date",
                "resolution_description"
            ]:

                data[col] = data[col].fillna(
                    "Not Available"
                )

            else:

                mode = data[col].mode()

                if not mode.empty:

                    data[col] = data[col].fillna(
                        mode.iloc[0]
                    )

                else:

                    data[col] = data[col].fillna(
                        "Not Available"
                    )

    missing_after = int(
        data.isnull().sum().sum()
    )

    # --------------------------------------------------------
    # REMOVE DUPLICATES
    # --------------------------------------------------------

    data = (
        data
        .drop_duplicates()
        .reset_index(drop=True)
    )

    duplicates_after = int(
        data.duplicated().sum()
    )

    rows_after_cleaning = len(data)

    # --------------------------------------------------------
    # REMOVE NON-FEATURE COLUMNS
    # --------------------------------------------------------

    columns_to_exclude = []

    for col in [
        "incident_address",
        "resolution_description",
        "created_date",
        "closed_date",
        "status"

    ]:

        if col in data.columns:

            columns_to_exclude.append(col)

    X = data.drop(
        columns=columns_to_exclude,
        errors="ignore"
    )

    # --------------------------------------------------------
    # FEATURE TYPES
    # --------------------------------------------------------

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
    # PREPROCESSING PIPELINE
    # --------------------------------------------------------

    transformers = []

    if numeric_features:

        numeric_transformer = Pipeline(
            steps=[
                (
                    "scaler",
                    StandardScaler()
                )
            ]
        )

        transformers.append(
            (
                "num",
                numeric_transformer,
                numeric_features
            )
        )

    if categorical_features:

        categorical_transformer = Pipeline(
            steps=[
                (
                    "onehot",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        sparse_output=False
                    )
                )
            ]
        )

        transformers.append(
            (
                "cat",
                categorical_transformer,
                categorical_features
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers
    )

    # --------------------------------------------------------
    # TRANSFORM DATA
    # --------------------------------------------------------

    X_preprocessed = preprocessor.fit_transform(X)

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    X_preprocessed = pd.DataFrame(
        X_preprocessed,
        columns=feature_names
    )

    # --------------------------------------------------------
    # FINAL CHECK
    # --------------------------------------------------------

    final_rows = len(X_preprocessed)
    final_features = len(
        X_preprocessed.columns
    )

    final_missing = int(
        X_preprocessed.isnull().sum().sum()
    )

    # --------------------------------------------------------
    # RE-ATTACH TARGET COLUMN
    # --------------------------------------------------------

    # We excluded 'status' from the features so it wouldn't get One-Hot Encoded.
    # Now we must glue the cleaned 'status' column back onto the end.

    if "status" in data.columns:
        X_preprocessed['status'] = data['status'].values
        print("Successfully re-attached 'status' column.")

    # --------------------------------------------------------
    # SAVE PREPROCESSED DATA
    # --------------------------------------------------------

    X_preprocessed.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # RETURN RESULTS
    # --------------------------------------------------------

    return {

        "original_rows": original_rows,

        "original_columns": original_columns,

        "missing_before": missing_before,

        "missing_after": missing_after,

        "duplicates_before": duplicates_before,

        "duplicates_after": duplicates_after,

        "rows_after_cleaning": rows_after_cleaning,

        "excluded_columns": columns_to_exclude,

        "numeric_features": numeric_features,

        "categorical_features": categorical_features,

        "original_feature_count": X.shape[1],

        "final_feature_count": final_features,

        "final_rows": final_rows,

        "final_missing": final_missing,

        "output_path": OUTPUT_PATH
    }


# ============================================================
# TEST ONLY WHEN FILE IS RUN DIRECTLY
# ============================================================

if __name__ == "__main__":

    result = preprocess_data()

    print("=" * 60)
    print("CIVICFIX PREPROCESSING")
    print("=" * 60)

    print(
        "Original rows:",
        result["original_rows"]
    )

    print(
        "Original columns:",
        result["original_columns"]
    )

    print(
        "Missing before:",
        result["missing_before"]
    )

    print(
        "Missing after:",
        result["missing_after"]
    )

    print(
        "Duplicates before:",
        result["duplicates_before"]
    )

    print(
        "Duplicates after:",
        result["duplicates_after"]
    )

    print(
        "Original features:",
        result["original_feature_count"]
    )

    print(
        "Final features:",
        result["final_feature_count"]
    )

    print(
        "Final missing values:",
        result["final_missing"]
    )

    print(
        "Saved to:",
        result["output_path"]
    )

    print("=" * 60)