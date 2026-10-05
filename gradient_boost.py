import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)


# ============================================================
# CIVICFIX - GRADIENT BOOSTING
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CSV_PATH = BASE_DIR / "dataset.csv"

TARGET = "status"


print("=" * 60)
print("CIVICFIX GRADIENT BOOSTING")
print("=" * 60)


# ============================================================
# LOAD DATASET
# ============================================================

df = pd.read_csv(CSV_PATH)

print("Dataset shape:", df.shape)


if TARGET not in df.columns:

    raise ValueError(
        "Target column 'status' not found."
    )


# ============================================================
# FEATURES
# ============================================================

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


# Keep only columns that actually exist

features = [

    col
    for col in features
    if col in df.columns

]


X = df[features].copy()

y = df[TARGET].copy()


print("Features used:", features)


# ============================================================
# HANDLE MISSING VALUES
# ============================================================

print("\nHandling missing values...")


for col in X.columns:

    # --------------------------------------------------------
    # NUMERICAL COLUMN
    # --------------------------------------------------------

    if pd.api.types.is_numeric_dtype(X[col]):

        X[col] = X[col].fillna(
            X[col].median()
        )


    # --------------------------------------------------------
    # CATEGORICAL / STRING COLUMN
    # --------------------------------------------------------

    else:

        X[col] = X[col].fillna(
            "Unknown"
        )


# Target missing values

y = y.fillna("Unknown")


# ============================================================
# ONE-HOT ENCODING
# ============================================================

categorical_cols = X.select_dtypes(
    include=[
        "object",
        "string",
        "category"
    ]
).columns


X = pd.get_dummies(

    X,

    columns=categorical_cols,

    drop_first=True

)


# Convert everything to numeric

X = X.astype(float)


print(
    "Features after encoding:",
    X.shape[1]
)


# ============================================================
# ENCODE TARGET
# ============================================================

label_encoder = LabelEncoder()

y_encoded = label_encoder.fit_transform(y)


print(
    "Target classes:",
    list(label_encoder.classes_)
)


# ============================================================
# TRAIN / TEST SPLIT
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y_encoded,

    test_size=0.20,

    random_state=42,

    stratify=y_encoded

)


print(
    "Training records:",
    len(X_train)
)

print(
    "Testing records:",
    len(X_test)
)


# ============================================================
# GRADIENT BOOSTING MODEL
# ============================================================

model = GradientBoostingClassifier(

    n_estimators=100,

    learning_rate=0.1,

    max_depth=3,

    min_samples_split=10,

    min_samples_leaf=5,

    random_state=42

)


print(
    "\nTraining Gradient Boosting..."
)


# ============================================================
# TRAIN
# ============================================================

model.fit(

    X_train,

    y_train

)


print(
    "Training completed successfully."
)


# ============================================================
# PREDICTION
# ============================================================

y_pred = model.predict(

    X_test

)


# ============================================================
# EVALUATION METRICS
# ============================================================

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


# ============================================================
# DISPLAY RESULTS
# ============================================================

print("\n" + "=" * 60)

print(
    "GRADIENT BOOSTING RESULTS"
)

print("=" * 60)


print(
    f"Accuracy  : {accuracy:.4f}"
)

print(
    f"Precision : {precision:.4f}"
)

print(
    f"Recall    : {recall:.4f}"
)

print(
    f"F1 Score  : {f1:.4f}"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

print(
    "\nClassification Report:"
)


print(

    classification_report(

        y_test,

        y_pred,

        target_names=label_encoder.classes_,

        zero_division=0

    )

)


# ============================================================
# CONFUSION MATRIX
# ============================================================

print(
    "\nConfusion Matrix:"
)


cm = confusion_matrix(

    y_test,

    y_pred

)


print(cm)


# ============================================================
# SAVE RESULTS CSV
# ============================================================

RESULTS_PATH = (

    BASE_DIR
    / "gradient_boost_results.csv"

)


results_df = pd.DataFrame([

    {

        "model":
        "Gradient Boosting",

        "accuracy":
        accuracy,

        "precision":
        precision,

        "recall":
        recall,

        "f1_score":
        f1,

        "training_samples":
        len(X_train),

        "testing_samples":
        len(X_test),

        "features":
        X.shape[1]

    }

])


results_df.to_csv(

    RESULTS_PATH,

    index=False

)


print(
    "\nResults saved to:"
)

print(
    RESULTS_PATH
)


# ============================================================
# SAVE CONFUSION MATRIX IMAGE
# ============================================================

MODEL_IMG_DIR = (

    BASE_DIR
    / "static"
    / "images"
    / "models"

)


MODEL_IMG_DIR.mkdir(

    parents=True,

    exist_ok=True

)


CONFUSION_PATH = (

    MODEL_IMG_DIR
    / "gradient_boost_confusion_matrix.png"

)


disp = ConfusionMatrixDisplay(

    confusion_matrix=cm,

    display_labels=label_encoder.classes_

)


disp.plot(

    xticks_rotation=45

)


plt.title(
    "Gradient Boosting - Confusion Matrix"
)


plt.tight_layout()


plt.savefig(

    CONFUSION_PATH,

    dpi=160

)


plt.close()


print(
    "Confusion matrix saved to:"
)

print(
    CONFUSION_PATH
)


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

feature_importance = pd.DataFrame({

    "feature":
    X.columns,

    "importance":
    model.feature_importances_

})


feature_importance = feature_importance.sort_values(

    by="importance",

    ascending=False

)


FEATURE_IMPORTANCE_PATH = (

    BASE_DIR
    / "gradient_boost_feature_importance.csv"

)


feature_importance.to_csv(

    FEATURE_IMPORTANCE_PATH,

    index=False

)


print(
    "Feature importance saved to:"
)

print(
    FEATURE_IMPORTANCE_PATH
)


# ============================================================
# TOP 10 FEATURES
# ============================================================

print(
    "\nTop 10 Important Features:"
)


print(

    feature_importance.head(10).to_string(

        index=False

    )

)


# ============================================================
# FINAL
# ============================================================

print("\n" + "=" * 60)

print(
    "GRADIENT BOOSTING COMPLETED SUCCESSFULLY."
)

print("=" * 60)