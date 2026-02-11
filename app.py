import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    roc_auc_score
)

st.set_page_config(page_title="ML Assignment 2", layout="wide")

st.title("Machine Learning Classification Demo")
st.write("Upload test data, select a model, and view evaluation results")

# -------------------------
# Model selection
# -------------------------
model_dict = {
    "Logistic Regression": "models/logistic_regression.pkl",
    "Decision Tree": "models/decision_tree.pkl",
    "KNN": "models/knn.pkl",
    "Naive Bayes": "models/naive_bayes.pkl",
    "Random Forest": "models/random_forest.pkl",
    "XGBoost": "models/xgboost.pkl"
}

selected_model_name = st.selectbox(
    "Select a Classification Model",
    list(model_dict.keys())
)

model_path = model_dict[selected_model_name]
model = joblib.load(model_path)

# -------------------------
# Dataset upload
# -------------------------
uploaded_file = st.file_uploader(
    "Upload Test Dataset (CSV)",
    type=["csv"]
)

if uploaded_file is not None:
    test_df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Test Data")
    st.write(test_df.head())

    if "y" not in test_df.columns:
        st.error("Uploaded CSV must contain target column 'y'")
    else:
        X_test = test_df.drop("y", axis=1)
        y_test = test_df["y"].map({"yes": 1, "no": 0})

        # -------------------------
        # Prediction
        # -------------------------
        y_pred = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
        else:
            auc = None

        # -------------------------
        # Metrics
        # -------------------------
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)

        st.subheader("Evaluation Metrics")
        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", round(acc, 3))
        col1.metric("AUC", round(auc, 3) if auc else "N/A")

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
