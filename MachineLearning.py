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
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB

# SPLITTING DATA INTO TRAIN AND TEST ==========================================
df = pd.read_csv('/Users/harounshah/Downloads/Senior Thesis/final_data.csv')

print(f"\nMatrix Dimensions w/ all Features: {df.shape}\n")

y = df['home_win']
X = df.filter(like="diff", axis=1) # Taking only the difference features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

# LOGISTIC REGRESSION =========================================================

log_reg = LogisticRegression(max_iter = 5000)
log_reg.fit(X_train, y_train)

y_pred_lr = log_reg.predict(X_test)
print("Logistic Regression Accuracy:", accuracy_score(y_test, y_pred_lr))

# RANDOM FOREST ===============================================================
rf_model = RandomForestClassifier(n_estimators = 200, random_state = 42)

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_rf)
print("Random Forest Test Accuracy:", accuracy)

importances = pd.Series(rf_model.feature_importances_, index = X.columns)
importances.sort_values().plot(kind = "barh", figsize = (12,8))
plt.title("Feature Importances")
plt.tight_layout()
plt.savefig("Figures/Feature_Importance.png")

# GRADIENT BOOSTING ===========================================================
gb_model = GradientBoostingClassifier()
gb_model.fit(X_train, y_train)

y_pred_gb = gb_model.predict(X_test)
print("Gradient Boosting Accuracy:", accuracy_score(y_test, y_pred_gb))

# SUPPORT VECTOR MACHINE ======================================================
svm_model = SVC(probability = True)
svm_model.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)
print("SVM Accuracy:", accuracy_score(y_test, y_pred_svm))

# ============================
# NAIVE BAYES (GAUSSIAN) MODEL
# ============================

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
