import pandas as pd
import re

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

from lightgbm import LGBMClassifier


# ============================================================
# CIVICFIX - LIGHTGBM
# ============================================================

from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent
CSV_PATH = str(BASE_DIR / "dataset.csv")
TARGET = "status"

print("=" * 60)
print("CIVICFIX LIGHTGBM")
print("=" * 60)


# ------------------------------------------------------------
# LOAD DATA
# ------------------------------------------------------------

df = pd.read_csv(CSV_PATH)

print("Dataset shape:", df.shape)

if TARGET not in df.columns:
    raise ValueError("Target column 'status' not found.")


# ------------------------------------------------------------
# FEATURES
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

features = [
    col for col in features
    if col in df.columns
]

X = df[features].copy()
y = df[TARGET].copy()


# ------------------------------------------------------------
# HANDLE MISSING VALUES
# ------------------------------------------------------------

for col in X.columns:

    if X[col].dtype == "object":
        X[col] = X[col].fillna("Unknown")

    else:
        X[col] = X[col].fillna(X[col].median())

y = y.fillna("Unknown")


# ------------------------------------------------------------
# ONE-HOT ENCODING
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
# FIX LIGHTGBM FEATURE NAMES
# ------------------------------------------------------------

def clean_feature_name(name):

    name = str(name)

    # Replace all characters except letters, numbers and _
    name = re.sub(r"[^A-Za-z0-9_]", "_", name)

    # Remove repeated underscores
    name = re.sub(r"_+", "_", name)

    # Remove leading/trailing underscores
    name = name.strip("_")

    # Make sure name is not empty
    if not name:
        name = "feature"

    return name


X.columns = [
    clean_feature_name(col)
    for col in X.columns
]


# Make column names unique
X.columns = pd.Index(X.columns).map(
    lambda x: x
)

if X.columns.duplicated().any():

    new_columns = []
    counts = {}

    for col in X.columns:

        if col not in counts:
            counts[col] = 0
            new_columns.append(col)

        else:
            counts[col] += 1
            new_columns.append(
                f"{col}_{counts[col]}"
            )

    X.columns = new_columns


# ------------------------------------------------------------
# ENCODE TARGET
# ------------------------------------------------------------

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)

print("Features after encoding:", X.shape[1])
print("Target classes:", list(label_encoder.classes_))


# ------------------------------------------------------------
# TRAIN / TEST SPLIT
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
# LIGHTGBM MODEL
# ------------------------------------------------------------

model = LGBMClassifier(
    n_estimators=100,
    learning_rate=0.1,
    max_depth=8,
    num_leaves=31,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=-1
)


# ------------------------------------------------------------
# TRAIN
# ------------------------------------------------------------

print("\nTraining LightGBM...")

model.fit(
    X_train,
    y_train
)


# ------------------------------------------------------------
# PREDICTION
# ------------------------------------------------------------

y_pred = model.predict(X_test)


# ------------------------------------------------------------
# EVALUATION
# ------------------------------------------------------------

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


# ------------------------------------------------------------
# RESULTS
# ------------------------------------------------------------

print("\n" + "=" * 60)
print("LIGHTGBM RESULTS")
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

print(
    confusion_matrix(
        y_test,
        y_pred
    )
)


from pathlib import Path
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

BASE_DIR = Path(__file__).resolve().parent
RESULTS_PATH = BASE_DIR / "lightgbm_results.csv"
pd.DataFrame([{
    "model": "LightGBM",
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
ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=label_encoder.classes_, xticks_rotation=45)
plt.title("LightGBM — Confusion Matrix")
plt.tight_layout(); plt.savefig(MODEL_IMG_DIR / "lightgbm_confusion_matrix.png", dpi=160); plt.close()


print("\nLightGBM completed successfully.")

print("=" * 60)