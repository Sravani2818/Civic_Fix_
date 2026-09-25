from flask import Flask, render_template
import load_data
from dashboard_utils import (
    read_json, result_row, preprocessing_summary, model_comparison,
    ensure_regression_assets, ensure_feature_charts, insights
)

app = Flask(__name__)


def build_assets():
    assets = {}
    assets['ridge'] = ensure_regression_assets()
    feature_assets = ensure_feature_charts()
    assets['feature_importance'] = feature_assets
    from dashboard_utils import ensure_comparison_chart
    assets['comparison'] = ensure_comparison_chart()
    return assets


@app.route('/')
def index():
    try:
        summary = load_data.get_data_summary()
    except Exception as exc:
        print('Data summary error:', exc)
        summary = {'n_rows': 0, 'n_cols': 0, 'num_cols': [], 'cat_cols': [], 'target_col': '', 'table_columns': [], 'table_data': []}

    preprocessing = preprocessing_summary()
    linear_results = read_json('linear_regression_results.json')
    regularised_linear_results = read_json('linear_regression_regularised_results.json')
    logistic_variants = read_json('logistic_regression_results.json')
    if not logistic_variants:
        # Backward-compatible fallback for the older single-result CSV.
        legacy_logistic = result_row('logistic_regression_results.csv')
        logistic_variants = {
            'model': 'Logistic Regression',
            'target': 'status',
            'without_regularisation': {},
            'with_regularisation': legacy_logistic,
        }
    results = {
        'logistic': result_row('logistic_regression_results.csv'),
        'decision_tree': result_row('decision_tree_results.csv'),
        'random_forest': result_row('random_forest_results.csv'),
        'adaboost': result_row('adaboost_results.csv'),
        'gradient_boost': result_row('gradient_boost_results.csv'),
        'xgboost': result_row('xgboost_results.csv'),
        'lightgbm': result_row('lightgbm_results.csv'),
    }
    comparison = model_comparison()
    assets = build_assets()
    return render_template(
        'index.html',
        total_rows=summary.get('n_rows', 0), total_cols=summary.get('n_cols', 0),
        num_cols=summary.get('num_cols', []), cat_cols=summary.get('cat_cols', []),
        target_col=summary.get('target_col', ''), table_columns=summary.get('table_columns', []),
        table_data=summary.get('table_data', []), preprocessing=preprocessing,
        linear_results=linear_results, regularised_linear_results=regularised_linear_results,
        logistic_results=results['logistic'],
        logistic_variants=logistic_variants,
        logistic_without_results=logistic_variants.get('without_regularisation', {}),
        logistic_with_results=logistic_variants.get('with_regularisation', {}),
        decision_tree_results=results['decision_tree'],
        random_forest_results=results['random_forest'], adaboost_results=results['adaboost'],
        gradient_boost_results=results['gradient_boost'], xgboost_results=results['xgboost'],
        lightgbm_results=results['lightgbm'], model_comparison=comparison,
        insights=insights(summary, preprocessing, comparison), assets=assets,
        model_results={'linear': linear_results, 'regularised_linear': regularised_linear_results, **results}
    )


if __name__ == '__main__':
    print('CivicFix ML Analytics Dashboard')
    print('Open: http://127.0.0.1:5000')
    app.run(debug=True, host='127.0.0.1', port=5000)
