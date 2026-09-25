import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# ============================================================
# CIVICFIX - RANDOM FOREST
# ============================================================

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = str(BASE_DIR / "dataset.csv")
TARGET = "status"

print("=" * 60)
print("CIVICFIX RANDOM FOREST")
print("=" * 60)

# Load dataset
df = pd.read_csv(CSV_PATH)

print("Dataset shape:", df.shape)

if TARGET not in df.columns:
    raise ValueError("Target column 'status' not found.")

# ------------------------------------------------------------
# Features
# ------------------------------------------------------------

features = [
    "agency_name",
    "complaint_type",
    "descriptor",
    "location_type",
    "incident_zip",
    "borough",
    "city",
    "latitude",
    "longitude",
    "open_data_channel_type"
]

features = [col for col in features if col in df.columns]

X = df[features].copy()
y = df[TARGET].copy()

# ------------------------------------------------------------
# Missing values
# ------------------------------------------------------------

for col in X.columns:
    if X[col].dtype == "object":
        X[col] = X[col].fillna("Unknown")
    else:
        X[col] = X[col].fillna(X[col].median())

y = y.fillna("Unknown")

# ------------------------------------------------------------
# One-hot encoding
# ------------------------------------------------------------

categorical_cols = X.select_dtypes(
    include=["object"]
).columns

X = pd.get_dummies(
    X,
    columns=categorical_cols,
    drop_first=True
)

X = X.astype(float)

# ------------------------------------------------------------
# Encode target
# ------------------------------------------------------------

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

print("Features after encoding:", X.shape[1])
print("Target classes:", list(label_encoder.classes_))

# ------------------------------------------------------------
# Train / Test split
# ------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.20,
    random_state=42,
    stratify=y_encoded
)

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# ------------------------------------------------------------
# Random Forest
# ------------------------------------------------------------

model = RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    min_samples_split=10,
    min_samples_leaf=5,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")

model.fit(X_train, y_train)

# ------------------------------------------------------------
# Prediction
# ------------------------------------------------------------

y_pred = model.predict(X_test)

# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------

accuracy = accuracy_score(y_test, y_pred)

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

print("\n" + "=" * 60)
print("RANDOM FOREST RESULTS")
print("=" * 60)

print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

from pathlib import Path
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "random_forest_results.csv"
pd.DataFrame([{
    "model": "Random Forest",
    "accuracy": accuracy,
    "precision": precision,
    "recall": recall,
    "f1_score": f1,
    "training_samples": len(X_train),
    "testing_samples": len(X_test),
    "features": X.shape[1]
}]).to_csv(RESULTS_PATH, index=False)
MODEL_IMG_DIR = BASE_DIR / "static" / "images" / "models"
MODEL_IMG_DIR.mkdir(parents=True, exist_ok=True)
imp = pd.DataFrame({"feature": X.columns, "importance": model.feature_importances_}).sort_values("importance").tail(12)
fig, ax = plt.subplots(figsize=(8,5)); ax.barh(imp["feature"].astype(str), imp["importance"]); ax.set_title("Random Forest — Top Feature Importance"); ax.set_xlabel("Importance"); fig.tight_layout(); fig.savefig(MODEL_IMG_DIR / "random_forest_feature_importance.png", dpi=160); plt.close(fig)


print("\nRandom Forest completed successfully.")
print("=" * 60)