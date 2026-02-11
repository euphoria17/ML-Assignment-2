import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score
)

# -------------------------
# Streamlit Page Config
# -------------------------
st.set_page_config(
    page_title="ML Assignment 2",
    layout="wide"
)

st.title("Machine Learning Classification Demo")
st.write("Upload a CSV dataset, select a model, and view evaluation metrics.")

# -------------------------
# Model Selection
# -------------------------
model_dict = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(),
    "KNN": KNeighborsClassifier(),
    "Naive Bayes": GaussianNB(),
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    "XGBoost": XGBClassifier(
        eval_metric="logloss",
        use_label_encoder=False,
        random_state=42
    )
}

selected_model_name = st.selectbox(
    "Select a Classification Model",
    list(model_dict.keys())
)

# -------------------------
# Dataset Upload
# -------------------------
uploaded_file = st.file_uploader(
    "Upload CSV Dataset (must contain target column 'y')",
    type=["csv"]
)

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Dataset Preview")
    st.dataframe(df.head())

    if "y" not in df.columns:
        st.error("The uploaded CSV must contain a target column named 'y'")
    else:
        X = df.drop("y", axis=1)
        y = df["y"].map({"yes": 1, "no": 0})

        # Identify column types
        categorical_cols = X.select_dtypes(include=["object"]).columns
        numerical_cols = X.select_dtypes(exclude=["object"]).columns

        # Preprocessing
        preprocessor = ColumnTransformer(
            transformers=[
                ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
                ("num", "passthrough", numerical_cols),
            ]
        )

        model = model_dict[selected_model_name]

        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("classifier", model),
            ]
        )

        # Train-test split (demo purpose)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )

        pipeline.fit(X_train, y_train)

        # Predictions
        y_pred = pipeline.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = None

        # -------------------------
        # Evaluation Metrics
        # -------------------------
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        st.subheader("Evaluation Metrics")

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", round(acc, 3))
        col1.metric("AUC", round(auc, 3) if auc is not None else "N/A")

        col2.metric("Precision", round(prec, 3))
        col2.metric("Recall", round(rec, 3))

        col3.metric("F1 Score", round(f1, 3))
        col3.metric("MCC", round(mcc, 3))

        # -------------------------
        # Confusion Matrix
        # -------------------------
        st.subheader("Confusion Matrix")

        cm = confusion_matrix(y_test, y_pred)

        fig, ax = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")

        st.pyplot(fig)
