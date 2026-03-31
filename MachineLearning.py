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
import os
from sklearn.model_selection import cross_val_score, StratifiedKFold, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay, ConfusionMatrixDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# CONFIG ======================================================================
# Set to True only when you want to re-run tuning.
RUN_HYPERPARAM_TUNING = False
# If set, only that model's tuning runs (others load saved params / defaults).
# Options: None, "log_reg", "random_forest", "grad_boost", "svm", "naive_bayes"
TUNE_ONLY = None
TUNING_PARAMS_PATH = "/Users/harounshah/Downloads/Senior Thesis/best_hyperparams.json"


def load_best_params():
    try:
        with open(TUNING_PARAMS_PATH, "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


def save_best_params(params_dict):
    with open(TUNING_PARAMS_PATH, "w") as f:
        json.dump(params_dict, f, indent=2)

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

lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("lr", LogisticRegression(max_iter=5000, solver="liblinear"))
])

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
best_params = load_best_params()

if RUN_HYPERPARAM_TUNING and (TUNE_ONLY is None or TUNE_ONLY == "log_reg"):
    lr_param_grid = {
        "lr__C": [0.01, 0.1, 1, 10, 100],
        "lr__penalty": ["l1", "l2"],
        "lr__class_weight": [None, "balanced"]
    }

    lr_search = GridSearchCV(
        estimator=lr_pipe,
        param_grid=lr_param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1
    )
    lr_search.fit(X_train, y_train)

    best_params["log_reg"] = lr_search.best_params_
    save_best_params(best_params)

    print("Best Logistic Regression (Train CV AUC):", round(lr_search.best_score_, 3))
    print("Best Logistic Regression Params:", lr_search.best_params_)
    log_reg = lr_search.best_estimator_
else:
    if "log_reg" in best_params:
        lr_pipe.set_params(**best_params["log_reg"])
        print("Using saved Logistic Regression params:", best_params["log_reg"])
    else:
        print("No saved Logistic Regression params found; using defaults.")
    log_reg = lr_pipe.fit(X_train, y_train)

# If tuning ran, best_estimator_ is already fit. If not, we fit above.

y_pred_lr = log_reg.predict(X_test)
train_acc_lr = accuracy_score(y_train, log_reg.predict(X_train))
test_acc_lr = accuracy_score(y_test, y_pred_lr)
print("Logistic Regression Train Accuracy:", train_acc_lr)
print("Logistic Regression 2025 Test Accuracy:", test_acc_lr)

# RANDOM FOREST ===============================================================
rf_model = RandomForestClassifier(random_state=42)
if RUN_HYPERPARAM_TUNING and (TUNE_ONLY is None or TUNE_ONLY == "random_forest"):
    rf_param_dist = {
        # More regularized search space to reduce overfitting
        "n_estimators": [300, 600, 900],
        "max_depth": [4, 6, 8, 10, 12],
        "min_samples_split": [10, 20, 50],
        "min_samples_leaf": [2, 4, 8, 16],
        "max_features": ["sqrt", "log2"],
        "class_weight": [None, "balanced"],
    }

    rf_search = RandomizedSearchCV(
        estimator=rf_model,
        param_distributions=rf_param_dist,
        n_iter=15,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=-1,
    )
    rf_search.fit(X_train, y_train)

    best_params["random_forest"] = rf_search.best_params_
    save_best_params(best_params)

    print("\nBest Random Forest (Train CV AUC):", round(rf_search.best_score_, 3))
    print("Best Random Forest Params:", rf_search.best_params_)
    rf_model = rf_search.best_estimator_
else:
    if "random_forest" in best_params:
        rf_model.set_params(**best_params["random_forest"])
        print("\nUsing saved Random Forest params:", best_params["random_forest"])
    else:
        print("\nNo saved Random Forest params found; using defaults.")

rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred_rf)
train_acc_rf = accuracy_score(y_train, rf_model.predict(X_train))
test_acc_rf = accuracy
print("Random Forest Train Accuracy:", train_acc_rf)
print("Random Forest 2025 Test Accuracy:", test_acc_rf)

# GRADIENT BOOSTING ===========================================================
gb_model = GradientBoostingClassifier(random_state=42)
if RUN_HYPERPARAM_TUNING and (TUNE_ONLY is None or TUNE_ONLY == "grad_boost"):
    gb_param_dist = {
        "n_estimators": [100, 200, 400, 600],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "max_depth": [2, 3, 4],
        "min_samples_split": [2, 5, 10],
        "min_samples_leaf": [1, 2, 4],
        "subsample": [0.6, 0.8, 1.0],
        "max_features": [None, "sqrt", "log2"],
    }

    gb_search = RandomizedSearchCV(
        estimator=gb_model,
        param_distributions=gb_param_dist,
        n_iter=10,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=-1,
    )
    gb_search.fit(X_train, y_train)

    best_params["grad_boost"] = gb_search.best_params_
    save_best_params(best_params)

    print("\nBest Gradient Boosting (Train CV AUC):", round(gb_search.best_score_, 3))
    print("Best Gradient Boosting Params:", gb_search.best_params_)
    gb_model = gb_search.best_estimator_
else:
    if "grad_boost" in best_params:
        gb_model.set_params(**best_params["grad_boost"])
        print("\nUsing saved Gradient Boosting params:", best_params["grad_boost"])
    else:
        print("\nNo saved Gradient Boosting params found; using defaults.")

gb_model.fit(X_train, y_train)

y_pred_gb = gb_model.predict(X_test)
train_acc_gb = accuracy_score(y_train, gb_model.predict(X_train))
test_acc_gb = accuracy_score(y_test, y_pred_gb)
print("Gradient Boosting Train Accuracy:", train_acc_gb)
print("Gradient Boosting 2025 Test Accuracy:", test_acc_gb)

# SUPPORT VECTOR MACHINE ======================================================
svm_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("svm", SVC(probability=True))
])

if RUN_HYPERPARAM_TUNING and (TUNE_ONLY is None or TUNE_ONLY == "svm"):
    svm_param_dist = {
        "svm__kernel": ["rbf", "linear"],
        "svm__C": [0.01, 0.1, 1, 10, 100],
        "svm__gamma": ["scale", "auto", 0.001, 0.01, 0.1, 1],
        "svm__class_weight": [None, "balanced"],
    }

    svm_search = RandomizedSearchCV(
        estimator=svm_pipe,
        param_distributions=svm_param_dist,
        n_iter=15,
        scoring="roc_auc",
        cv=cv,
        random_state=42,
        n_jobs=-1,
    )
    svm_search.fit(X_train, y_train)

    best_params["svm"] = svm_search.best_params_
    save_best_params(best_params)

    print("\nBest SVM (Train CV AUC):", round(svm_search.best_score_, 3))
    print("Best SVM Params:", svm_search.best_params_)
    svm_model = svm_search.best_estimator_
else:
    if "svm" in best_params:
        svm_pipe.set_params(**best_params["svm"])
        print("\nUsing saved SVM params:", best_params["svm"])
    else:
        print("\nNo saved SVM params found; using defaults.")
    svm_model = svm_pipe.fit(X_train, y_train)

y_pred_svm = svm_model.predict(X_test)
train_acc_svm = accuracy_score(y_train, svm_model.predict(X_train))
test_acc_svm = accuracy_score(y_test, y_pred_svm)
print("SVM Train Accuracy:", train_acc_svm)
print("SVM 2025 Test Accuracy:", test_acc_svm)

# NAIVE BAYES (GAUSSIAN) ======================================================

nb_model = GaussianNB()
if RUN_HYPERPARAM_TUNING and (TUNE_ONLY is None or TUNE_ONLY == "naive_bayes"):
    # Only real hyperparameter for GaussianNB; mostly numerical stability.
    # Use a simple manual sweep to keep it fast and avoid parallel job overhead.
    candidates = np.logspace(-12, -6, 13)
    best_auc = -1
    best_vs = None
    for vs in candidates:
        m = GaussianNB(var_smoothing=float(vs))
        auc = cross_val_score(m, X_train, y_train, cv=cv, scoring="roc_auc").mean()
        if auc > best_auc:
            best_auc = auc
            best_vs = float(vs)

    best_params["naive_bayes"] = {"var_smoothing": best_vs}
    save_best_params(best_params)

    print("\nBest Naive Bayes (Train CV AUC):", round(best_auc, 3))
    print("Best Naive Bayes Params:", {"var_smoothing": best_vs})
    nb_model = GaussianNB(var_smoothing=best_vs)
else:
    if "naive_bayes" in best_params:
        nb_model.set_params(**best_params["naive_bayes"])
        print("\nUsing saved Naive Bayes params:", best_params["naive_bayes"])
    else:
        print("\nNo saved Naive Bayes params found; using defaults.")

# Fit the model
nb_model.fit(X_train, y_train)

# Predict on test data
y_pred_nb = nb_model.predict(X_test)

# Accuracy
nb_accuracy = accuracy_score(y_test, y_pred_nb)
train_acc_nb = accuracy_score(y_train, nb_model.predict(X_train))
test_acc_nb = nb_accuracy
print("Naive Bayes Train Accuracy:", train_acc_nb)
print("Naive Bayes 2025 Test Accuracy:", test_acc_nb, "\n")


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

# MODEL COMPARISON TABLE ======================================================
comparison_rows = [
    {
        "Model": "Logistic Regression",
        "Train Accuracy": train_acc_lr,
        "2025 Test Accuracy": test_acc_lr,
        # AUC filled below
    },
    {
        "Model": "Random Forest",
        "Train Accuracy": train_acc_rf,
        "2025 Test Accuracy": test_acc_rf,
        # AUC filled below
    },
    {
        "Model": "Gradient Boosting",
        "Train Accuracy": train_acc_gb,
        "2025 Test Accuracy": test_acc_gb,
        # AUC filled below
    },
    {
        "Model": "SVM",
        "Train Accuracy": train_acc_svm,
        "2025 Test Accuracy": test_acc_svm,
        # AUC filled below
    },
    {
        "Model": "Naive Bayes",
        "Train Accuracy": train_acc_nb,
        "2025 Test Accuracy": test_acc_nb,
        # AUC filled below
    },
]

# Compute AUC for table (using the same models used for ROC plots)
auc_by_model = {}
for name, model in models.items():
    probs = model.predict_proba(X_test)[:, 1]
    auc_by_model[name] = roc_auc_score(y_test, probs)

for row in comparison_rows:
    row["2025 Test AUC"] = auc_by_model.get(row["Model"], np.nan)

comparison_df = pd.DataFrame(comparison_rows).sort_values("2025 Test AUC", ascending=False)
comparison_df.to_csv("model_comparison_table.csv", index=False)
print("\nSaved model comparison table: model_comparison_table.csv")
print(comparison_df.to_string(index=False))

# FEATURE IMPORTANCE PLOTS =====================================================
features = X_train.columns.tolist()
FEATURE_IMPORTANCE_DIR = "Figures/Feature Importance"
os.makedirs(FEATURE_IMPORTANCE_DIR, exist_ok=True)

def plot_importance(values, title, out_filename, xlabel="Importance (higher = more important)", top_k=10):
    s = pd.Series(values, index=features).sort_values(ascending=False)
    s_top = s.head(min(top_k, len(s)))

    plt.figure(figsize=(8, 5))
    plt.barh(list(s_top.index)[::-1], list(s_top.values)[::-1])
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(os.path.join(FEATURE_IMPORTANCE_DIR, out_filename), bbox_inches="tight")
    plt.close()
    return s_top

# Logistic Regression: coefficients on standardized features (use absolute values for "importance")
lr_coefs = None
if hasattr(log_reg, "named_steps") and "lr" in log_reg.named_steps:
    lr_est = log_reg.named_steps["lr"]
    lr_coefs = np.abs(lr_est.coef_).ravel()
    plot_importance(
        lr_coefs,
        title="Feature Importance - Logistic Regression (|coef|)",
        out_filename="feature_importance_lr.png",
    )

# Random Forest
plot_importance(
    rf_model.feature_importances_,
    title="Feature Importance - Random Forest",
    out_filename="feature_importance_rf.png",
)

# Gradient Boosting
plot_importance(
    gb_model.feature_importances_,
    title="Feature Importance - Gradient Boosting",
    out_filename="feature_importance_gb.png",
)

# SVM (only meaningful if linear kernel; otherwise coefficients not available)
svm_coefs = None
try:
    svm_est = svm_model.named_steps["svm"] if hasattr(svm_model, "named_steps") else None
    if svm_est is not None and hasattr(svm_est, "coef_"):
        svm_coefs = np.abs(svm_est.coef_).ravel()
        plot_importance(
            svm_coefs,
            title="Feature Importance - SVM Linear (|coef|)",
            out_filename="feature_importance_svm_linear.png",
        )
except Exception:
    svm_coefs = None

# Consensus ranking (average rank across LR/RF/GB, plus linear-SVM when available)
rank_sources = {
    "rf": rf_model.feature_importances_,
    "gb": gb_model.feature_importances_,
}
if lr_coefs is not None:
    rank_sources["lr"] = lr_coefs
if svm_coefs is not None:
    rank_sources["svm"] = svm_coefs

rank_df = pd.DataFrame(rank_sources, index=features)
rank_mat = rank_df.rank(axis=0, ascending=False, method="average")
consensus_score = rank_mat.mean(axis=1)  # lower rank-score => more important

top_idx = np.argsort(consensus_score.values)[:10]
top_feats = [features[i] for i in top_idx]
top_scores = consensus_score.values[top_idx]

plt.figure(figsize=(8, 5))
plt.barh(top_feats[::-1], top_scores[::-1])
plt.xlabel("Consensus rank score (lower = more important)")
plt.title("Feature Importance Consensus (LR/RF/GB/SVM where available)")
plt.tight_layout()
plt.savefig(os.path.join(FEATURE_IMPORTANCE_DIR, "feature_importance_consensus.png"), bbox_inches="tight")
plt.close()

# Print top 5 to console for quick thesis copy-paste
top5_idx = np.argsort(consensus_score.values)[:5]
top5 = [(features[i], float(consensus_score.values[i])) for i in top5_idx]
print("\nTop 5 consensus features (lower score = more important):")
for feat, score in top5:
    print(f"  {feat}: {score:.2f}")
