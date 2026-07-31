import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="ML Model Comparison Dashboard",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# TITLE
# ============================================================

st.title("🤖 Machine Learning Model Comparison Dashboard")

st.write(
    "Compare Multiple Machine Learning Algorithms "
    "with Interactive Visualization"
)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("⚙ Dashboard")

st.sidebar.info(
    """
    Steps:

    1. Upload CSV Dataset
    2. Select Target Column
    3. Click Train Models
    4. Compare Performance
    5. View Best Model
    """
)


# ============================================================
# FILE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📂 Upload CSV Dataset",
    type=["csv"]
)

if uploaded_file is None:
    st.info("📂 Please upload a CSV dataset to start.")
    st.stop()


# ============================================================
# READ CSV
# ============================================================

try:
    df = pd.read_csv(uploaded_file)

except Exception as e:
    st.error(f"❌ CSV file could not be read: {e}")
    st.stop()


# Remove empty rows and columns
df = df.dropna(axis=0, how="all")
df = df.dropna(axis=1, how="all")


if df.empty:
    st.error("❌ Dataset is empty.")
    st.stop()


# ============================================================
# DATASET PREVIEW
# ============================================================

col1, col2 = st.columns([3, 1])

with col1:
    st.subheader("📋 Dataset Preview")

    st.dataframe(
        df.head(),
        use_container_width=True
    )

with col2:
    st.subheader("📊 Dataset Info")

    st.metric("Rows", df.shape[0])
    st.metric("Columns", df.shape[1])


st.divider()


# ============================================================
# TARGET COLUMN
# ============================================================

# Automatically select Exited if available
if "Exited" in df.columns:
    default_target = list(df.columns).index("Exited")
else:
    default_target = 0

target = st.selectbox(
    "🎯 Select Target Column",
    df.columns,
    index=default_target
)


# ============================================================
# X AND Y
# ============================================================

X = df.drop(columns=[target]).copy()
y = df[target].copy()


# ============================================================
# REMOVE MISSING TARGET
# ============================================================

valid_target = y.notna()

X = X.loc[valid_target].copy()
y = y.loc[valid_target].copy()


if len(X) < 20:
    st.error("❌ Dataset has too few valid rows for training.")
    st.stop()


# ============================================================
# REMOVE ID-LIKE COLUMNS
# ============================================================

id_like_columns = []

for col in X.columns:

    col_lower = col.lower()

    unique_count = X[col].nunique(dropna=True)

    # Known ID column names
    known_id = any(
        word in col_lower
        for word in [
            "rownumber",
            "customerid",
            "customer_id",
            "userid",
            "user_id",
            "accountid",
            "account_id"
        ]
    )

    # Completely unique numeric/string column
    completely_unique = unique_count == len(X)

    if known_id or completely_unique:
        id_like_columns.append(col)


if id_like_columns:

    st.info(
        "ℹ️ ID-like columns removed: "
        + ", ".join(id_like_columns)
    )

    X = X.drop(
        columns=id_like_columns,
        errors="ignore"
    )


if X.shape[1] == 0:
    st.error("❌ No usable feature columns remain.")
    st.stop()


# ============================================================
# TARGET ENCODING
# ============================================================

if (
    y.dtype == "object"
    or str(y.dtype).startswith("category")
    or y.dtype == "bool"
):

    target_encoder = LabelEncoder()

    y = pd.Series(
        target_encoder.fit_transform(
            y.astype(str)
        ),
        index=y.index
    )

else:

    y = pd.to_numeric(
        y,
        errors="coerce"
    )

    valid_y = y.notna()

    X = X.loc[valid_y].copy()
    y = y.loc[valid_y].copy()


# ============================================================
# TARGET CHECK
# ============================================================

if y.nunique() < 2:
    st.error(
        "❌ Target column must contain at least 2 classes."
    )
    st.stop()


# ============================================================
# CONVERT BOOLEAN COLUMNS
# ============================================================

for col in X.columns:

    if X[col].dtype == "bool":

        X[col] = X[col].astype(int)


# ============================================================
# DETECT COLUMN TYPES
# ============================================================

numeric_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()


categorical_columns = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()


# Convert datetime columns to strings
datetime_columns = X.select_dtypes(
    include=["datetime", "datetimetz"]
).columns.tolist()


for col in datetime_columns:

    X[col] = X[col].astype(str)


categorical_columns.extend(
    datetime_columns
)


# ============================================================
# PREPROCESSOR
# ============================================================

transformers = []


# Numeric preprocessing
if numeric_columns:

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

    transformers.append(
        (
            "numeric",
            numeric_pipeline,
            numeric_columns
        )
    )


# Categorical preprocessing
if categorical_columns:

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    transformers.append(
        (
            "categorical",
            categorical_pipeline,
            categorical_columns
        )
    )


if not transformers:
    st.error("❌ No usable feature columns found.")
    st.stop()


preprocessor = ColumnTransformer(
    transformers=transformers,
    remainder="drop"
)


# ============================================================
# TRAIN TEST SPLIT
# ============================================================

try:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

except ValueError:

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )


# ============================================================
# TRAIN BUTTON
# ============================================================

st.divider()

train_button = st.button(
    "🚀 Train Models",
    type="primary",
    use_container_width=True
)


if train_button:

    st.subheader("🚀 Training Models...")


    # ========================================================
    # PREPROCESS DATA
    # ========================================================

    with st.spinner("⏳ Preparing dataset..."):

        try:

            X_train_processed = preprocessor.fit_transform(
                X_train
            )

            X_test_processed = preprocessor.transform(
                X_test
            )

        except Exception as e:

            st.error(
                "❌ Data preprocessing failed."
            )

            st.exception(e)

            st.stop()


    # ========================================================
    # CONVERT TO NUMPY
    # ========================================================

    try:

        X_train_processed = np.asarray(
            X_train_processed,
            dtype=np.float64
        )

        X_test_processed = np.asarray(
            X_test_processed,
            dtype=np.float64
        )

    except Exception as e:

        st.error(
            "❌ Could not convert processed data to numeric format."
        )

        st.exception(e)

        st.stop()


    # Remove NaN and infinity
    X_train_processed = np.nan_to_num(
        X_train_processed,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    X_test_processed = np.nan_to_num(
        X_test_processed,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )


    st.success(
        f"✅ Preprocessing completed. "
        f"Training features: {X_train_processed.shape[1]}"
    )


    # ========================================================
    # MODELS
    # ========================================================

    models = {

        "Logistic Regression":
            LogisticRegression(
                max_iter=300,
                random_state=42
            ),

        "Decision Tree":
            DecisionTreeClassifier(
                max_depth=10,
                random_state=42
            ),

        "Random Forest":
            RandomForestClassifier(
                n_estimators=50,
                max_depth=10,
                random_state=42,
                n_jobs=2
            ),

        "XGBoost":
            XGBClassifier(
                n_estimators=50,
                max_depth=4,
                learning_rate=0.1,
                eval_metric="logloss",
                random_state=42,
                n_jobs=2
            )
    }


    results = []
    predictions = {}


    progress = st.progress(0)
    status = st.empty()


    # ========================================================
    # TRAIN MODELS
    # ========================================================

    for i, (name, model) in enumerate(models.items()):

        status.info(
            f"⏳ Training {name}..."
        )

        try:

            # IMPORTANT:
            # Train on processed numeric data
            model.fit(
                X_train_processed,
                y_train
            )


            pred = model.predict(
                X_test_processed
            )


            predictions[name] = pred


            accuracy = accuracy_score(
                y_test,
                pred
            )

            precision = precision_score(
                y_test,
                pred,
                average="weighted",
                zero_division=0
            )

            recall = recall_score(
                y_test,
                pred,
                average="weighted",
                zero_division=0
            )

            f1 = f1_score(
                y_test,
                pred,
                average="weighted",
                zero_division=0
            )


            results.append(
                {
                    "Model": name,
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1 Score": f1
                }
            )


            st.success(
                f"✅ {name} completed"
            )


        except Exception as e:

            st.error(
                f"❌ {name} failed"
            )

            st.exception(e)


        progress.progress(
            (i + 1) / len(models)
        )


    status.success(
        "✅ Model training completed!"
    )


    # ========================================================
    # RESULTS
    # ========================================================

    results_df = pd.DataFrame(results)


    if results_df.empty:

        st.error(
            "❌ No model was successfully trained."
        )

        st.stop()


    st.divider()


    # ========================================================
    # PERFORMANCE TABLE
    # ========================================================

    st.subheader(
        "🏆 Model Performance Comparison"
    )


    display_df = results_df.copy()


    for col in [
        "Accuracy",
        "Precision",
        "Recall",
        "F1 Score"
    ]:

        display_df[col] = (
            display_df[col] * 100
        ).round(2).astype(str) + "%"


    st.dataframe(
        display_df,
        use_container_width=True,
        hide_index=True
    )


    # ========================================================
    # BEST MODEL
    # ========================================================

    best = results_df.loc[
        results_df["Accuracy"].idxmax()
    ]


    st.divider()

    st.subheader("🥇 Best Model")


    st.success(
        f"🏆 Best Model: {best['Model']} "
        f"| Accuracy: {best['Accuracy']:.2%}"
    )


    # ========================================================
    # METRIC CARDS
    # ========================================================

    st.subheader(
        "📊 Performance Metrics"
    )


    c1, c2, c3, c4 = st.columns(4)


    c1.metric(
        "🎯 Accuracy",
        f"{best['Accuracy']:.2%}"
    )

    c2.metric(
        "📌 Precision",
        f"{best['Precision']:.2%}"
    )

    c3.metric(
        "📈 Recall",
        f"{best['Recall']:.2%}"
    )

    c4.metric(
        "⭐ F1 Score",
        f"{best['F1 Score']:.2%}"
    )


    # ========================================================
    # ACCURACY CHART
    # ========================================================

    st.divider()

    st.subheader(
        "📊 Accuracy Comparison"
    )


    fig, ax = plt.subplots(
        figsize=(10, 5)
    )


    sns.barplot(
        data=results_df,
        x="Model",
        y="Accuracy",
        hue="Model",
        legend=False,
        ax=ax
    )


    ax.set_ylim(0, 1)

    ax.set_title(
        "Accuracy Comparison",
        fontsize=16,
        fontweight="bold"
    )

    ax.set_xlabel("Models")
    ax.set_ylabel("Accuracy")


    plt.xticks(
        rotation=15
    )


    for p in ax.patches:

        height = p.get_height()

        ax.annotate(
            f"{height:.2%}",
            (
                p.get_x() + p.get_width() / 2,
                height
            ),
            ha="center",
            va="bottom"
        )


    st.pyplot(
        fig,
        use_container_width=True
    )

    plt.close(fig)


    # ========================================================
    # METRICS LINE CHART
    # ========================================================

    st.subheader(
        "📈 Metrics Comparison"
    )


    metric_df = results_df.set_index(
        "Model"
    )


    st.line_chart(
        metric_df
    )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.subheader(
        "🔥 Confusion Matrix"
    )


    available_models = list(
        predictions.keys()
    )


    selected_model = st.selectbox(
        "Select Model",
        available_models
    )


    pred = predictions[
        selected_model
    ]


    cm = confusion_matrix(
        y_test,
        pred
    )


    fig2, ax2 = plt.subplots(
        figsize=(6, 5)
    )


    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="YlGnBu",
        linewidths=1,
        square=True,
        ax=ax2
    )


    ax2.set_title(
        f"{selected_model} Confusion Matrix"
    )

    ax2.set_xlabel(
        "Predicted Label"
    )

    ax2.set_ylabel(
        "Actual Label"
    )


    st.pyplot(
        fig2
    )

    plt.close(fig2)


    # ========================================================
    # FINAL SUMMARY
    # ========================================================

    st.divider()


    st.success(
        f"🏆 Best Performing Model: "
        f"{best['Model']} "
        f"with Accuracy "
        f"{best['Accuracy']:.2%}"
    )
