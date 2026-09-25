# CivicFix — ML Analytics Dashboard

**Bridging Citizens and Cities through AI-Driven Response Management**

This version keeps the existing Flask + Python ML project structure and upgrades the dashboard presentation and model-result data flow.

## Run

1. Create/activate a Python 3.10+ environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Start the dashboard:

```bash
python main.py
```

4. Open `http://127.0.0.1:5000`.

## Model scripts

The classification scripts now resolve project paths relative to their own file and write their result CSVs into the project directory. Decision Tree also generates its tree visualization automatically.

Run individual models when you want to refresh their stored outputs:

```bash
python decision_tree.py
python random_forest.py
python adaboost.py
python gradient_boost.py
python xgboost_model.py
python lightgbm_model.py
python logistic_regression.py
```

Gradient Boosting can take substantially longer on the full one-hot encoded dataset because it uses the existing `GradientBoostingClassifier` configuration. The dashboard intentionally does not fabricate a result while that model has no stored output.

## Dashboard assets

The Flask dashboard automatically prepares presentation assets from stored project results, including regression prediction and feature-importance charts. Model-generated assets are saved under `static/images/models/`.

## Important

No accuracy, precision, recall, F1, MAE, RMSE, R², training-row count, testing-row count, or feature-importance value is hard-coded into the dashboard. If a model has not produced a result file yet, its section reports that the result is not available rather than inventing a value.


### Logistic Regression variants
Run `python logistic_regression.py` to generate both the without-regularisation and L2-regularised results in `logistic_regression_results.json`.
