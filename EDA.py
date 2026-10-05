import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# ============================================================
# CIVICFIX - EXPLORATORY DATA ANALYSIS
# Plotly based EDA
# ============================================================

CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "dataset.csv")

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(
        f"Dataset not found at: {CSV_PATH}"
    )

# ============================================================
# 1. LOAD DATA
# ============================================================

data = pd.read_csv(CSV_PATH)

print("=" * 80)
print("CIVICFIX - EDA")
print("=" * 80)

print("\n1. DATA LOADED")
print("-" * 80)
print("Dataset Shape:", data.shape)
print("\nFirst 5 Records:")
print(data.head())

# ============================================================
# 2. BASIC INFORMATION
# ============================================================

print("\n2. BASIC INFORMATION")
print("-" * 80)

print("\nColumns:")
print(data.columns.tolist())

print("\nData Types:")
print(data.dtypes)

print("\nDataset Information:")
data.info()

# ============================================================
# 3. MISSING VALUE ANALYSIS
# ============================================================

print("\n3. MISSING VALUE ANALYSIS")
print("-" * 80)

missing_count = data.isnull().sum()
missing_percentage = (missing_count / len(data)) * 100

missing_df = pd.DataFrame({
    "Column": missing_count.index,
    "Missing Count": missing_count.values,
    "Missing Percentage": missing_percentage.values
})

# Keep only columns having missing values
missing_df = missing_df[
    missing_df["Missing Count"] > 0
].sort_values(
    by="Missing Count",
    ascending=False
)

print("\nMissing Value Summary:")
print(missing_df)


# ============================================================
# MISSING VALUES GRAPH
# ============================================================

if not missing_df.empty:

    fig = px.bar(
        missing_df,
        x="Column",
        y="Missing Count",
        text="Missing Count",
        title="Missing Values by Column",
        labels={
            "Column": "Column Name",
            "Missing Count": "Number of Missing Values"
        }
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        template="plotly_white",
        title_x=0.5,
        xaxis_tickangle=-45
    )

    fig.show()

else:

    print("\nNo missing values found in the dataset.")
# ============================================================
# 4. DUPLICATE ANALYSIS
# ============================================================

print("\n4. DUPLICATE ANALYSIS")
print("-" * 80)

duplicate_count = data.duplicated().sum()

print("Duplicate Rows:", duplicate_count)

# ============================================================
# 5. TARGET VARIABLE - STATUS
# ============================================================

print("\n5. TARGET VARIABLE ANALYSIS")
print("-" * 80)

if "status" in data.columns:

    status_counts = data["status"].value_counts()

    print(status_counts)

    fig = px.bar(
        x=status_counts.index,
        y=status_counts.values,
        labels={
            "x": "Complaint Status",
            "y": "Number of Complaints"
        },
        title="Complaint Status Distribution"
    )

    fig.update_layout(
        template="plotly_white",
        title_x=0.5
    )

    fig.show()

# ============================================================
# 6. NUMERIC FEATURE DISTRIBUTIONS
# ============================================================

print("\n6. NUMERIC FEATURE DISTRIBUTIONS")
print("-" * 80)

numeric_columns = data.select_dtypes(
    include=np.number
).columns.tolist()

print("Numeric Columns:")
print(numeric_columns)

for column in numeric_columns:

    fig = px.histogram(
        data,
        x=column,
        marginal="box",
        title=f"Distribution of {column}"
    )

    fig.update_layout(
        template="plotly_white",
        title_x=0.5
    )

    fig.show()

# ============================================================
# 7. OUTLIER DETECTION
# ============================================================

print("\n7. OUTLIER DETECTION")
print("-" * 80)

for column in numeric_columns:

    fig = px.box(
        data,
        y=column,
        title=f"Outlier Analysis - {column}"
    )

    fig.update_layout(
        template="plotly_white",
        title_x=0.5
    )

    fig.show()

# ============================================================
# 8. CORRELATION ANALYSIS
# ============================================================

print("\n8. CORRELATION ANALYSIS")
print("-" * 80)

if len(numeric_columns) >= 2:

    correlation = data[
        numeric_columns
    ].corr()

    print(correlation)

    fig = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        title="Correlation Heatmap"
    )

    fig.update_layout(
        template="plotly_white",
        title_x=0.5
    )

    fig.show()

# ============================================================
# 9. GEOGRAPHICAL DISTRIBUTION
# ============================================================

print("\n9. GEOGRAPHICAL DISTRIBUTION")
print("-" * 80)

if (
    "latitude" in data.columns
    and
    "longitude" in data.columns
):

    fig = px.scatter(
        data,
        x="longitude",
        y="latitude",
        title="Geographical Distribution of Complaints",
        opacity=0.6,
        hover_data=[
            column
            for column in [
                "borough",
                "complaint_type",
                "status"
            ]
            if column in data.columns
        ]
    )

    fig.update_layout(
        template="plotly_white",
        title_x=0.5
    )

    fig.show()

# ============================================================
# 10. CATEGORICAL FEATURE ANALYSIS
# ============================================================

print("\n10. CATEGORICAL FEATURE ANALYSIS")
print("-" * 80)

categorical_columns = data.select_dtypes(
    exclude=np.number
).columns.tolist()

print("Categorical Columns:")
print(categorical_columns)

# Focus on important categorical variables
important_categories = [
    "borough",
    "agency_name",
    "complaint_type",
    "location_type",
    "open_data_channel_type"
]

for column in important_categories:

    if column in data.columns:

        top_values = (
            data[column]
            .value_counts()
            .head(10)
            .reset_index()
        )

        top_values.columns = [
            column,
            "Count"
        ]

        fig = px.bar(
            top_values,
            x=column,
            y="Count",
            title=f"Top 10 {column.replace('_', ' ').title()}"
        )

        fig.update_layout(
            template="plotly_white",
            title_x=0.5,
            xaxis_tickangle=-45
        )

        fig.show()

# ============================================================
# 11. BOROUGH VS STATUS
# ============================================================

print("\n11. BOROUGH VS STATUS")
print("-" * 80)

if (
    "borough" in data.columns
    and
    "status" in data.columns
):

    borough_status = pd.crosstab(
        data["borough"],
        data["status"]
    ).reset_index()

    fig = px.bar(
        borough_status,
        x="borough",
        y=[
            column
            for column in borough_status.columns
            if column != "borough"
        ],
        title="Complaint Status by Borough",
        barmode="group"
    )

    fig.update_layout(
        template="plotly_white",
        title_x=0.5,
        xaxis_tickangle=-45
    )

    fig.show()

# ============================================================
# 12. AGENCY VS STATUS
# ============================================================

print("\n12. AGENCY VS STATUS")
print("-" * 80)

if (
    "agency_name" in data.columns
    and
    "status" in data.columns
):

    top_agencies = (
        data["agency_name"]
        .value_counts()
        .head(10)
        .index
    )

    agency_data = data[
        data["agency_name"].isin(top_agencies)
    ]

    fig = px.histogram(
        agency_data,
        x="agency_name",
        color="status",
        title="Top Agencies vs Complaint Status",
        barmode="group"
    )

    fig.update_layout(
        template="plotly_white",
        title_x=0.5,
        xaxis_tickangle=-45
    )

    fig.show()

# ============================================================
# 13. TEMPORAL ANALYSIS
# ============================================================

print("\n13. TEMPORAL ANALYSIS")
print("-" * 80)

if "created_date" in data.columns:

    date_data = data.copy()

    date_data["created_date"] = pd.to_datetime(
        date_data["created_date"],
        errors="coerce"
    )

    date_data = date_data.dropna(
        subset=["created_date"]
    )

    if len(date_data) > 0:

        monthly_data = (
            date_data
            .set_index("created_date")
            .resample("ME")
            .size()
            .reset_index(name="Complaint_Count")
        )

        fig = px.line(
            monthly_data,
            x="created_date",
            y="Complaint_Count",
            markers=True,
            title="Complaint Volume Over Time"
        )

        fig.update_layout(
            template="plotly_white",
            title_x=0.5,
            xaxis_title="Date",
            yaxis_title="Number of Complaints"
        )

        fig.show()

# ============================================================
# 14. RESOLUTION TIME ANALYSIS
# ============================================================

print("\n14. RESOLUTION TIME ANALYSIS")
print("-" * 80)

if (
    "created_date" in data.columns
    and
    "closed_date" in data.columns
):

    data["created_date"] = pd.to_datetime(
        data["created_date"],
        errors="coerce"
    )

    data["closed_date"] = pd.to_datetime(
        data["closed_date"],
        errors="coerce"
    )

    data["Resolution_Time_Hours"] = (
        data["closed_date"]
        - data["created_date"]
    ).dt.total_seconds() / 3600

    resolved_data = data[
        data["Resolution_Time_Hours"].notna()
        &
        (data["Resolution_Time_Hours"] > 0)
    ]

    print(
        "Resolved Cases:",
        len(resolved_data)
    )

    if len(resolved_data) > 0:

        print(
            "Average Resolution Time:",
            round(
                resolved_data[
                    "Resolution_Time_Hours"
                ].mean(),
                2
            ),
            "hours"
        )

        fig = px.histogram(
            resolved_data,
            x="Resolution_Time_Hours",
            marginal="box",
            title="Resolution Time Distribution"
        )

        fig.update_layout(
            template="plotly_white",
            title_x=0.5,
            xaxis_title="Resolution Time (Hours)"
        )

        fig.show()

        if "borough" in resolved_data.columns:

            fig = px.box(
                resolved_data,
                x="borough",
                y="Resolution_Time_Hours",
                title="Resolution Time by Borough"
            )

            fig.update_layout(
                template="plotly_white",
                title_x=0.5,
                xaxis_tickangle=-45
            )

            fig.show()

# ============================================================
# 15. RELATIONSHIP ANALYSIS
# ============================================================

print("\n15. RELATIONSHIP ANALYSIS")
print("-" * 80)

pair_columns = [
    column
    for column in [
        "latitude",
        "longitude",
        "Resolution_Time_Hours"
    ]
    if column in data.columns
]

if len(pair_columns) >= 2:

    sample_size = min(
        1500,
        len(data)
    )

    sample_data = data.sample(
        n=sample_size,
        random_state=42
    )

    fig = px.scatter_matrix(
        sample_data,
        dimensions=pair_columns,
        title="Relationship Between Numeric Features"
    )

    fig.update_layout(
        template="plotly_white",
        title_x=0.5
    )

    fig.show()

# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("EDA COMPLETED SUCCESSFULLY")
print("=" * 80)

print("\nDataset Shape:", data.shape)
print("Numeric Features:", len(numeric_columns))
print("Categorical Features:", len(categorical_columns))
print("Duplicate Rows:", duplicate_count)

if "status" in data.columns:
    print(
        "Target Variable: status"
    )

print("\nAll EDA analysis completed.")