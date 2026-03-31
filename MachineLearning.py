# IMPORTS =====================================================================
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import requests
from bs4 import BeautifulSoup
import json
import time
import seaborn as sns
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# TRAIN ON TRAINING SET, TEST ON 2025 SEASON ==================================
train_df = pd.read_csv('/Users/harounshah/Downloads/Senior Thesis/final_data_EDA.csv')
test_df = pd.read_csv('/Users/harounshah/Downloads/Senior Thesis/final_data_test.csv')

y_train = train_df['home_win']
X_train = train_df.filter(like="diff", axis=1)

y_test = test_df['home_win']
X_test_raw = test_df.filter(like="diff", axis=1)

# Align test columns to the exact training feature set.
X_test = X_test_raw.reindex(columns=X_train.columns)

print(f"\nX_train shape: {X_train.shape}")
print(f"X_test shape:  {X_test.shape}\n")

# LOGISTIC REGRESSION =========================================================

# Hyperparameter tuning (training only; 2025 remains held-out)
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=5000, solver="liblinear"))
])

param_grid = {
    "lr__C": [0.01, 0.1, 1, 10, 100],
    "lr__penalty": ["l1", "l2"],
    "lr__class_weight": [None, "balanced"]
}

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
lr_search = GridSearchCV(
    estimator=lr_pipe,
    param_grid=param_grid,
    scoring="roc_auc",
    cv=cv,
    n_jobs=-1
)
lr_search.fit(X_train, y_train)

print("Best Logistic Regression (CV AUC):", round(lr_search.best_score_, 3))
print("Best Logistic Regression Params:", lr_search.best_params_)

log_reg = lr_search.best_estimator_

y_pred_lr = log_reg.predict(X_test)
print("Logistic Regression Test Accuracy:", accuracy_score(y_test, y_pred_lr))

# RANDOM FOREST ===============================================================
rf_model = RandomForestClassifier(n_estimators = 200, random_state = 42)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_rf)
print("Random Forest Test Accuracy:", accuracy)

importances = pd.Series(rf_model.feature_importances_, index = X_train.columns)
importances.sort_values().plot(kind = "barh", figsize = (12,8))
plt.title("Feature Importances")
plt.tight_layout()
plt.savefig("Figures/Feature_Importance.png")

# GRADIENT BOOSTING ===========================================================
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)

y_pred_gb = gb_model.predict(X_test)
print("Gradient Boosting Test Accuracy:", accuracy_score(y_test, y_pred_gb))

# SUPPORT VECTOR MACHINE ======================================================
svm_model = SVC(probability = True)
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)
print("SVM Test Accuracy:", accuracy_score(y_test, y_pred_svm))

# NAIVE BAYES (GAUSSIAN) ======================================================

nb_model = GaussianNB()

# Fit the model
nb_model.fit(X_train, y_train)

# Predict on test data
y_pred_nb = nb_model.predict(X_test)

# Accuracy
nb_accuracy = accuracy_score(y_test, y_pred_nb)
print("Naive Bayes Test Accuracy:", nb_accuracy, "\n")


# EVALUATION METRICS ==========================================================
def plot_conf_mat(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(title)
    plt.savefig(f"Figures/Confusion Matrices/{title}")
    plt.close()


plot_conf_mat(y_test, y_pred_rf, "CM_Random Forest")
plot_conf_mat(y_test, y_pred_lr, "CM_Logistic Regression")
plot_conf_mat(y_test, y_pred_gb, "CM_Gradient Boosting")
plot_conf_mat(y_test, y_pred_svm, "CM_SVM")
plot_conf_mat(y_test, y_pred_nb, "CM_Naive Bayes")

# print("Random Forest Classification Report:")
# print(classification_report(y_test, y_pred_rf), "\n")

# scores = cross_val_score(rf_model, X, y, cv=5, scoring='accuracy')
# print("Random Forest CV Mean:", scores.mean())
# print("Random Forest CV Std:", scores.std(), "\n")

models = {
    "Logistic Regression": log_reg,
    "Random Forest": rf_model,
    "Gradient Boosting": gb_model,
    "SVM": svm_model,
    "Naive Bayes": nb_model
}

for name, model in models.items():
    probs = model.predict_proba(X_test)[:,1]
    auc = roc_auc_score(y_test, probs)
    print(f"{name} AUC: {auc:.3f}")
    
    RocCurveDisplay.from_predictions(y_test, probs)
    plt.title(f"{name} ROC Curve")
    plt.savefig(f"Figures/ROC Curves/ROC_{name}")
    plt.close()
