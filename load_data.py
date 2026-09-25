import pandas as pd
import os

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = str(BASE_DIR / "dataset.csv")


def load_data(path: str = DATA_PATH) -> pd.DataFrame:

    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found: {path}"
        )

    return pd.read_csv(path)


def get_data_summary(path: str = DATA_PATH) -> dict:

    df = load_data(path)

    # ---------------------------------------------------------
    # DATA PREVIEW
    # ---------------------------------------------------------

    display_df = df.head(10).copy()

    display_df = display_df.where(
        display_df.notna(),
        "—"
    )

    # ---------------------------------------------------------
    # COLUMN INFORMATION
    # ---------------------------------------------------------

    dtype_info = []

    for column in df.columns:

        dtype_info.append({
            "column": column,
            "dtype": str(df[column].dtype),
            "missing": int(df[column].isna().sum()),
            "unique": int(df[column].nunique(dropna=True))
        })

    # ---------------------------------------------------------
    # NUMERICAL SUMMARY
    # ---------------------------------------------------------

    numerical_summary = []

    numeric_df = df.select_dtypes(include="number")

    if not numeric_df.empty:

        stats = numeric_df.describe().round(2)

        for column in stats.columns:

            row = {
                "Feature": column,
                "Count": stats.loc["count", column],
                "Mean": stats.loc["mean", column],
                "Std": stats.loc["std", column],
                "Min": stats.loc["min", column],
                "25%": stats.loc["25%", column],
                "50%": stats.loc["50%", column],
                "75%": stats.loc["75%", column],
                "Max": stats.loc["max", column]
            }

            numerical_summary.append(row)

    # ---------------------------------------------------------
    # SUMMARY
    # ---------------------------------------------------------

    summary = {

        # Dataset size
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),

        # Feature types
        "num_cols": df.select_dtypes(
            include="number"
        ).columns.tolist(),

        "cat_cols": df.select_dtypes(
            exclude="number"
        ).columns.tolist(),

        # Target
        "target_col": (
            "status"
            if "status" in df.columns
            else ""
        ),

        # Data quality
        "total_missing": int(
            df.isna().sum().sum()
        ),

        "duplicate_count": int(
            df.duplicated().sum()
        ),

        # Columns
        "columns": df.columns.tolist(),

        # Column information
        "dtype_info": dtype_info,

        # Preview
        "table_columns": display_df.columns.tolist(),

        "table_data": display_df.to_dict(
            orient="records"
        ),

        # Numerical statistics
        "numerical_summary": numerical_summary
    }

    return summary