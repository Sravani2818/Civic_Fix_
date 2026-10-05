import json
import os
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
STATIC_DIR = BASE_DIR / 'static'
MODEL_IMG_DIR = STATIC_DIR / 'images' / 'models'
MODEL_IMG_DIR.mkdir(parents=True, exist_ok=True)


def safe_json(value):
    if isinstance(value, dict):
        return {k: safe_json(v) for k, v in value.items()}
    if isinstance(value, list):
        return [safe_json(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return value


def read_json(filename):
    path = BASE_DIR / filename
    if not path.exists():
        return {}
    try:
        with path.open('r', encoding='utf-8') as f:
            return safe_json(json.load(f))
    except Exception as exc:
        print(f'Could not read {filename}: {exc}')
        return {}


def read_csv(filename):
    path = BASE_DIR / filename
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(filename).replace(
            [float('inf'), float('-inf')], pd.NA
        )
    except Exception as exc:
        print(f'Could not read {filename}: {exc}')
        return pd.DataFrame()


def result_row(filename):
    df = read_csv(filename)
    return safe_json(df.iloc[0].to_dict()) if not df.empty else {}


def preprocessing_summary():
    raw = read_csv('dataset.csv')
    processed = read_csv('datasetpreprocessed.csv')

    return {
        'status': 'Ready' if not processed.empty else 'Not available',
        'missing_before': int(raw.isna().sum().sum()) if not raw.empty else 0,
        'missing_after': int(processed.isna().sum().sum()) if not processed.empty else 0,
        'duplicates_before': int(raw.duplicated().sum()) if not raw.empty else 0,
        'duplicates_after': int(processed.duplicated().sum()) if not processed.empty else 0,
        'original_rows': int(len(raw)) if not raw.empty else 0,
        'processed_rows': int(len(processed)) if not processed.empty else 0,
        'original_features': (
            int(len(raw.columns) - 1)
            if not raw.empty and 'status' in raw
            else int(len(raw.columns))
            if not raw.empty else 0
        ),
        'final_features': (
            int(len(processed.columns) - 1)
            if not processed.empty and 'status' in processed
            else int(len(processed.columns))
            if not processed.empty else 0
        ),
        'scaling': 'StandardScaler used for Logistic Regression',
        'encoding': 'Categorical variables one-hot encoded',
    }


def ensure_regression_assets():
    data = read_json('linear_regression_regularised_results.json')
    preds = data.get('predictions', []) if isinstance(data, dict) else []

    if not preds:
        return []

    path = MODEL_IMG_DIR / 'ridge_actual_vs_predicted.png'

    if not path.exists():
        actual = [x.get('actual') for x in preds]
        predicted = [x.get('predicted') for x in preds]

        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(actual, predicted, alpha=0.75)

        lo, hi = min(actual + predicted), max(actual + predicted)
        ax.plot([lo, hi], [lo, hi], linestyle='--')

        ax.set_title('Ridge Regression: Actual vs Predicted')
        ax.set_xlabel('Actual latitude')
        ax.set_ylabel('Predicted latitude')
        ax.grid(alpha=.2)

        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches='tight')
        plt.close(fig)

    return ['images/models/ridge_actual_vs_predicted.png']


def ensure_feature_charts():
    created = []

    for model, filename, outname in [
        (
            'Random Forest',
            'random_forest_feature_importance.csv',
            'random_forest_feature_importance.png'
        ),
        (
            'AdaBoost',
            'adaboost_feature_importance.csv',
            'adaboost_feature_importance.png'
        ),
        (
            'XGBoost',
            'xgboost_feature_importance.csv',
            'xgboost_feature_importance.png'
        ),
    ]:

        df = read_csv(filename)

        if df.empty or len(df.columns) < 2:
            continue

        feature_col = next(
            (
                c for c in df.columns
                if c.lower() in {'feature', 'features', 'feature_name'}
            ),
            df.columns[0]
        )

        importance_col = next(
            (
                c for c in df.columns
                if 'importance' in c.lower()
            ),
            df.columns[-1]
        )

        temp = df[[feature_col, importance_col]].copy()

        temp[importance_col] = pd.to_numeric(
            temp[importance_col],
            errors='coerce'
        )

        temp = (
            temp.dropna()
            .sort_values(importance_col)
            .tail(12)
        )

        if temp.empty:
            continue

        path = MODEL_IMG_DIR / outname

        fig, ax = plt.subplots(figsize=(8, 5))

        ax.barh(
            temp[feature_col].astype(str),
            temp[importance_col]
        )

        ax.set_title(f'{model}: Top Feature Importance')
        ax.set_xlabel('Importance')
        ax.grid(axis='x', alpha=.2)

        fig.tight_layout()
        fig.savefig(path, dpi=160, bbox_inches='tight')
        plt.close(fig)

        created.append(f'images/models/{outname}')

    return created


def model_comparison():
    items = []

    sources = [
        (
            'Decision Tree',
            'Classification',
            'decision_tree_results.csv'
        ),
        (
            'Random Forest',
            'Classification',
            'random_forest_results.csv'
        ),
        (
            'AdaBoost',
            'Classification',
            'adaboost_results.csv'
        ),
        (
            'Gradient Boosting',
            'Classification',
            'gradient_boost_results.csv'
        ),
        (
            'XGBoost',
            'Classification',
            'xgboost_results.csv'
        ),
        (
            'LightGBM',
            'Classification',
            'lightgbm_results.csv'
        ),
    ]

    def classification_row(name, row):
        return {
            'model': name,
            'type': 'Classification',
            'accuracy': row.get('accuracy') if row else None,
            'precision': row.get('precision') if row else None,
            'recall': row.get('recall') if row else None,
            'f1': row.get('f1_score') if row else None,
            'mae': None,
            'mse': None,
            'rmse': None,
            'r2': None,
        }

    logistic = read_json('logistic_regression_results.json')

    if isinstance(logistic, dict):

        no_reg = logistic.get(
            'without_regularisation'
        ) or {}

        with_reg = logistic.get(
            'with_regularisation'
        ) or {}

        if no_reg:
            items.append(
                classification_row(
                    'Logistic Regression — Without Regularisation',
                    no_reg
                )
            )

        if with_reg:
            items.append(
                classification_row(
                    'Logistic Regression — With L2 Regularisation',
                    with_reg
                )
            )

    else:

        legacy = result_row(
            'logistic_regression_results.csv'
        )

        if legacy:
            items.append(
                classification_row(
                    'Logistic Regression',
                    legacy
                )
            )

    for name, kind, filename in sources:

        row = result_row(filename)

        items.append(
            classification_row(
                name,
                row
            )
        )

    lr = read_json(
        'linear_regression_results.json'
    )

    if isinstance(lr, dict):

        r = lr.get(
            'without_regularisation'
        )

        if r:
            items.append({
                'model':
                    'Linear Regression — Without Regularisation',

                'type':
                    'Regression',

                'accuracy':
                    None,

                'precision':
                    None,

                'recall':
                    None,

                'f1':
                    None,

                'r2':
                    r.get('r2_score'),

                'mae':
                    r.get('mae'),

                'mse':
                    r.get('mse'),

                'rmse':
                    r.get('rmse'),
            })

        for key, label in [
            (
                'ridge_regularisation',
                'Ridge (L2)'
            ),
            (
                'lasso_regularisation',
                'Lasso (L1)'
            )
        ]:

            r = lr.get(key)

            if r:
                items.append({
                    'model':
                        f'Linear Regression — With Regularisation ({label})',

                    'type':
                        'Regression',

                    'accuracy':
                        None,

                    'precision':
                        None,

                    'recall':
                        None,

                    'f1':
                        None,

                    'r2':
                        r.get('r2_score'),

                    'mae':
                        r.get('mae'),

                    'mse':
                        r.get('mse'),

                    'rmse':
                        r.get('rmse'),
                })

    return items


# ============================================================
# OVERALL INSIGHTS
# ============================================================

def insights(summary, preprocessing, comparison):

    lines = []

    lines.append(
        "The working dataset contains 50,000 rows and 15 columns, "
        "with 'status' used as the main classification target."
    )

    lines.append(
        "The main classification target is 'status', "
        "which represents the current state of each civic complaint."
    )

    lines.append(
        "The dataset contains 40,905 missing values, "
        "which are handled during preprocessing using suitable "
        "numerical and categorical imputation methods."
    )

    lines.append(
        "Categorical features are converted using One-Hot Encoding, "
        "while numerical values are cleaned and prepared for "
        "machine learning."
    )

    lines.append(
        "The dataset is highly imbalanced, with 'Closed' complaints "
        "forming the majority class and very few 'Pending' and "
        "'Unspecified' records."
    )

    lines.append(
        "Illegal Parking is one of the most frequent complaint types, "
        "while Brooklyn and Queens account for a large share "
        "of complaints."
    )

    lines.append(
        "XGBoost achieved the highest recorded classification accuracy "
        "among the stored models, at approximately 87.5%."
    )

    lines.append(
        "Ridge Regression achieved the highest recorded regression R² "
        "among the tested models, at approximately 0.882."
    )

    lines.append(
        "CivicFix demonstrates how machine learning, exploratory "
        "data analysis and visualization can help analyze civic "
        "complaints and support better civic management."
    )

    return lines


def ensure_comparison_chart():

    rows = model_comparison()

    cls = [
        r for r in rows
        if r.get('type') == 'Classification'
        and r.get('accuracy') is not None
    ]

    if not cls:
        return None

    path = MODEL_IMG_DIR / 'model_accuracy_comparison.png'

    names = [
        r['model']
        for r in cls
    ]

    vals = [
        float(r['accuracy'])
        for r in cls
    ]

    fig, ax = plt.subplots(
        figsize=(10, 5)
    )

    ax.bar(
        names,
        vals
    )

    ax.set_ylim(
        0,
        1
    )

    ax.set_ylabel(
        'Accuracy'
    )

    ax.set_title(
        'Classification Model Accuracy Comparison'
    )

    ax.tick_params(
        axis='x',
        rotation=28
    )

    ax.grid(
        axis='y',
        alpha=.2
    )

    fig.tight_layout()

    fig.savefig(
        path,
        dpi=160,
        bbox_inches='tight'
    )

    plt.close(fig)

    return 'images/models/model_accuracy_comparison.png'