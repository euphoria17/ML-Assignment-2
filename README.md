## Problem Statement

The objective of this project is to build and compare multiple machine learning
classification models to predict whether a customer will subscribe to a term
deposit based on marketing campaign data. The project demonstrates the complete
end-to-end machine learning workflow including data preprocessing, model
training, evaluation, and deployment using a Streamlit web application.

## Dataset Description

The dataset used for this project is the Bank Marketing Dataset obtained from the
UCI Machine Learning Repository.

- Total instances: 45,211
- Total features: 16 (before encoding)
- Target variable: `y`
  - yes: customer subscribed to term deposit
  - no: customer did not subscribe to term deposit
- The dataset contains both numerical and categorical attributes related to
  client information and previous marketing interactions.


## Models and Evaluation Metrics

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---------|----------|-----|-----------|--------|----------|-----|
| Logistic Regression | 0.901250 | 0.905443 | 0.643979 | 0.348771 | 0.452483 | 0.426446 |
| Decision Tree | 0.877585 | 0.713397 | 0.477828 | 0.499055 | 0.488211 | 0.418854 |
| KNN | 0.893619 | 0.808369 | 0.586022 | 0.309074 | 0.404703 | 0.374213 |
| Naive Bayes | 0.864315 | 0.823126 | 0.433096 | 0.517013 | 0.471349 | 0.396248 |
| Random Forest | 0.904899 | 0.927217 | 0.658147 | 0.389414 | 0.489311 | 0.459154 |
| XGBoost | 0.907995 | 0.929058 | 0.634845 | 0.502836 | 0.561181 | 0.514894 |

## Observations

| ML Model | Observation |
|--------|-------------|
| Logistic Regression | Provides a strong baseline with high accuracy and AUC, but recall is relatively low, indicating conservative positive predictions. |
| Decision Tree | Achieves higher recall compared to Logistic Regression but has lower AUC, suggesting reduced generalization performance. |
| KNN | Shows decent accuracy but lower recall, indicating difficulty in identifying minority class samples. |
| Naive Bayes | Achieves relatively higher recall but lower precision, leading to more false positive predictions. |
| Random Forest | Improves overall performance with better balance between precision and recall compared to single-tree models. |
| XGBoost | Delivers the best overall performance with the highest AUC, F1-score, and MCC, making it the most effective model for this dataset. |


## Deployment

The trained models were deployed using Streamlit Community Cloud. The application
allows users to upload test data, select a classification model, view evaluation
metrics, and visualize the confusion matrix interactively.
