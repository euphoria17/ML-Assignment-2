import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef
)

import joblib
from xgboost import XGBClassifier


# Load dataset
data_path = "/home/cloud/Desktop/ML Assignment 2/data/bank_data.csv"
df = pd.read_csv(data_path, sep=';')

print("Dataset shape:", df.shape)

# Separate features and target
X = df.drop("y", axis=1)
y = df["y"].map({"yes": 1, "no": 0})

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

print("Total features after encoding:", X.shape[1])

#train_test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

#Standarization
scaler = StandardScaler()

X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


#model evaluation
def evaluate_model(model, X_test, y_test, use_proba=True):
    y_pred = model.predict(X_test)
    
    if use_proba:
        y_prob = model.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    else:
        auc = None

    return {
        "Accuracy": accuracy_score(y_test, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred)
    }

results = {}

# 1. Logistic Regression
log_reg = LogisticRegression(max_iter=1000)
log_reg.fit(X_train_scaled, y_train)

results["Logistic Regression"] = evaluate_model(
    log_reg, X_test_scaled, y_test
)

joblib.dump(log_reg, "/home/cloud/Desktop/ML Assignment 2/models/logistic_regression.pkl")


# 2. Decision Tree
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)

results["Decision Tree"] = evaluate_model(
    dt, X_test, y_test
)

joblib.dump(dt, "/home/cloud/Desktop/ML Assignment 2/models/decision_tree.pkl")

# 3. K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

results["KNN"] = evaluate_model(
    knn, X_test_scaled, y_test
)

joblib.dump(knn, "/home/cloud/Desktop/ML Assignment 2/models/knn.pkl")

# 4. Naive Bayes
nb = GaussianNB()
nb.fit(X_train, y_train)

results["Naive Bayes"] = evaluate_model(
    nb, X_test, y_test
)

joblib.dump(nb, "/home/cloud/Desktop/ML Assignment 2/models/naive_bayes.pkl")

# 5. Random Forest
rf = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)
rf.fit(X_train, y_train)

results["Random Forest"] = evaluate_model(
    rf, X_test, y_test
)

joblib.dump(rf, "/home/cloud/Desktop/ML Assignment 2/models/random_forest.pkl")

# 6. XGBoost
xgb = XGBClassifier(
    eval_metric="logloss",
    random_state=42
)
xgb.fit(X_train, y_train)

results["XGBoost"] = evaluate_model(
    xgb, X_test, y_test
)

joblib.dump(xgb, "/home/cloud/Desktop/ML Assignment 2/models/xgboost.pkl")

# Convert results to DataFrame
results_df = pd.DataFrame(results).T

print("\nModel Evaluation Results:")
print(results_df)

