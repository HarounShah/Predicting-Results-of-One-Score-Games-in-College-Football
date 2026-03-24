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
from sklearn.feature_selection import mutual_info_classif
from scipy.stats import ttest_ind

# SPLITTING DATA INTO X & Y ===================================================
df = pd.read_csv('/Users/harounshah/Downloads/Senior Thesis/final_data.csv')

print(f"\nMatrix Dimensions w/ all Features: {df.shape}\n")

y = df['home_win']
X = df.filter(like="diff", axis=1) # Taking only the difference features

print(f"Matrix Dimensions w/ only Difference Features: {X.shape}\n")
print(X.columns, '\n')

# CLASS BALANCE SUMMARY ========================================================
class_counts = y.value_counts().sort_index()
class_percents = y.value_counts(normalize=True).sort_index() * 100

balance_table = pd.DataFrame({
    "count": class_counts,
    "percent": class_percents.round(2)
})

print("Class Balance (home_win):")
print(balance_table)

plt.figure(figsize=(6, 4))
ax = sns.countplot(x=y, hue=y, palette="Set2", legend=False)
ax.set_title("Class Balance: home_win")
ax.set_xlabel("home_win (0 = home loss, 1 = home win)")
ax.set_ylabel("Count")
ax.set_ylim(0, 1600)

total = len(y)
for p in ax.patches:
    count = int(p.get_height())
    percent = 100 * count / total
    ax.annotate(
        f"{count}\n({percent:.1f}%)",
        (p.get_x() + p.get_width() / 2, p.get_height()),
        ha="center",
        va="bottom",
        fontsize=9
    )

plt.tight_layout()
plt.savefig("Figures/ClassBalance.png", bbox_inches='tight')


# FEATURE VISUALIZATIONS ======================================================
# Set up plotting style
sns.set(style="whitegrid", palette="muted", font_scale=0.9)

# Create a figure grid — adjust size to fit all variables
n_cols = 4
n_rows = int(len(X.columns) / n_cols) + 1
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 2))
axes = axes.flatten()

# HISTOGRAMS
for i, col in enumerate(X.columns):
    ax = axes[i]
    if col in ["interceptions_diff", "fumblesLost_diff", "turnovers_diff"]:
        sns.histplot(X[col], kde=True, kde_kws={"bw_adjust": 1.5}, ax=ax, bins=np.arange(X[col].min() - 0.5, X[col].max() + 1.5, 1), color='skyblue')
    elif col == "fourthDown%_diff":
        sns.histplot(X[col], kde=True, ax=ax, bins=15, color='skyblue')
    else:
        sns.histplot(X[col], kde=True, ax=ax, bins=21, color='skyblue')
    ax.set_title(col)
    ax.set_xlabel("")
    ax.set_ylabel("Count")

# Hide any extra subplots (if grid > number of columns)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle("Distribution of Difference Features", fontsize=16, y=1.02)
plt.savefig("Figures/Histograms.png", bbox_inches='tight')

# BOXPLOTS
n_cols = 4
n_rows = int(len(X.columns) / n_cols) + 1
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 2))
axes = axes.flatten()

for i, col in enumerate(X.columns):
    ax = axes[i]
    sns.boxplot(x=X[col], ax=ax, color='lightgreen')
    ax.set_title(col)
    ax.set_xlabel("")

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle("Boxplots of Difference Features", fontsize=16, y=1.02)
plt.savefig("Figures/Boxplots.png", bbox_inches='tight')

# CORRELATION MATRIX
plt.figure(figsize=(12, 10))
corr = X.corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation Matrix of Difference Features")
plt.savefig("Figures/CorrelationMatrix.png", bbox_inches='tight')

# HISTOGRAMS (GROUPED BY CLASS)
n_cols = 4
n_rows = int(len(X.columns) / n_cols) + 1
fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, n_rows * 2))
axes = axes.flatten()

for i, col in enumerate(X.columns):
    ax = axes[i]
    if col in ["interceptions_diff", "fumblesLost_diff", "turnovers_diff"]:
        sns.histplot(data=df, x=col, hue=y, kde=True, kde_kws={"bw_adjust": 1.5}, element="step", ax=ax, bins=np.arange(df[col].min() - 0.5, df[col].max() + 1.5, 1), alpha=0.4)
        ax.set_title(col, pad=10, loc='center')
    elif col == "fourthDown%_diff":
        sns.histplot(data=df, x=col, hue=y, kde=True, element="step", ax=ax, bins=15, alpha=0.4)
        ax.set_title(col, pad=10, loc='center')
    else:
        sns.histplot(data=df, x=col, hue=y, kde=True, element="step", ax=ax, bins=21, alpha=0.4)
        ax.set_title(col)
        ax.set_xlabel("")
        ax.set_ylabel("Count")

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle("Feature Histograms by Class", fontsize=16, y=1.02)
plt.savefig("Figures/HistogramsByClass.png", bbox_inches='tight')

# FEATURE NUMERICAL SUMMARIES

summaries = pd.DataFrame()

for col in X.columns:
    desc = X[col].describe().drop('count')
    summaries[col] = desc

summaries = summaries.T 
summaries.index.name = 'feature'
summaries.to_csv("5numsums.csv", index = True)

# 

corrs = X.apply(lambda col: col.corr(y))
corrs = corrs.sort_values(ascending = False)
# print("Feature–Target Pearson Correlations:")
# print(corrs)

mi = mutual_info_classif(X, y, discrete_features = False, random_state = 42)

mi_series = pd.Series(mi, index = X.columns)
mi_series = mi_series.sort_values(ascending = False)

# print("\nMutual Information Scores:")
# print(mi_series)

results = pd.DataFrame({
    'Pearson Correlation': corrs,
    'Mutual Information Score': mi_series
})

results = results.sort_values(by = 'Pearson Correlation', ascending = False)
results['Normalized Mutual Information Score'] = results['Mutual Information Score'] / results['Mutual Information Score'].max()
print("\nFeature Target Correlations:")
print(results)

# FEATURE-WISE STATISTICAL TESTS (SIMPLIFIED) ==================================
test_rows = []

for col in X.columns:
    win_vals = df.loc[df["home_win"] == 1, col].dropna().values
    loss_vals = df.loc[df["home_win"] == 0, col].dropna().values

    # Welch t-test (difference in means; no equal-variance assumption)
    t_stat, t_p = ttest_ind(win_vals, loss_vals, equal_var=False)

    # Cohen's d
    n1, n0 = len(win_vals), len(loss_vals)
    s1 = np.std(win_vals, ddof=1)
    s0 = np.std(loss_vals, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * s1**2 + (n0 - 1) * s0**2) / (n1 + n0 - 2))
    cohens_d = (np.mean(win_vals) - np.mean(loss_vals)) / pooled_sd if pooled_sd != 0 else np.nan

    test_rows.append({
        "feature": col,
        "mean_home_win1": np.mean(win_vals),
        "mean_home_win0": np.mean(loss_vals),
        "mean_diff_win_minus_loss": np.mean(win_vals) - np.mean(loss_vals),
        "welch_t_p": t_p,
        "cohens_d": cohens_d
    })

tests_df = pd.DataFrame(test_rows)

# Label Cohen's d magnitude
abs_d = tests_df["cohens_d"].abs()
tests_df["cohens_d_magnitude"] = np.where(
    abs_d < 0.2, "negligible",
    np.where(abs_d < 0.5, "small", np.where(abs_d < 0.8, "medium", "large"))
)

tests_df["significant_p_lt_0_05"] = tests_df["welch_t_p"] < 0.05
tests_df = tests_df.sort_values("welch_t_p", ascending=True)

print("\nPer-feature statistical tests (simple view):")
print(
    tests_df[
        [
            "feature",
            "mean_diff_win_minus_loss",
            "welch_t_p",
            "cohens_d",
            "cohens_d_magnitude",
            "significant_p_lt_0_05"
        ]
    ].to_string(index=False)
)

# WIN-RATE BY FEATURE BINS =====================================================
# Use top features by absolute Pearson correlation for interpretable win-rate plots.
top_bin_features = corrs.abs().sort_values(ascending=False).head(6).index.tolist()
bin_rows = []

n_cols = 3
n_rows = int(np.ceil(len(top_bin_features) / n_cols))
fig, axes = plt.subplots(n_rows, n_cols, figsize=(14, n_rows * 3.5))
axes = np.array(axes).flatten()

for i, col in enumerate(top_bin_features):
    ax = axes[i]

    # Quantile bins keep sample sizes relatively balanced per bin.
    binned = pd.qcut(df[col], q=6, duplicates="drop")
    grouped = df.groupby(binned, observed=False)["home_win"].agg(["mean", "count"]).reset_index()
    grouped.columns = ["bin", "win_rate", "count"]

    # Approximate 95% CI for proportion: p +/- 1.96 * sqrt(p*(1-p)/n)
    se = np.sqrt((grouped["win_rate"] * (1 - grouped["win_rate"])) / grouped["count"])
    grouped["ci_low"] = np.clip(grouped["win_rate"] - 1.96 * se, 0, 1)
    grouped["ci_high"] = np.clip(grouped["win_rate"] + 1.96 * se, 0, 1)

    x = np.arange(len(grouped))
    win_rate_vals = grouped["win_rate"].values
    yerr = np.vstack([
        win_rate_vals - grouped["ci_low"].values,
        grouped["ci_high"].values - win_rate_vals
    ])

    ax.errorbar(x, win_rate_vals, yerr=yerr, fmt="o-", capsize=4, color="tab:blue")
    ax.set_xticks(x)
    ax.set_xticklabels([str(v) for v in grouped["bin"]], rotation=30, ha="right")
    ax.set_ylim(0, 1)
    ax.set_title(f"{col} vs Home Win Rate")
    ax.set_xlabel("Quantile Bin")
    ax.set_ylabel("P(home_win = 1)")
    ax.grid(alpha=0.25)

    for _, row in grouped.iterrows():
        bin_rows.append({
            "feature": col,
            "bin": str(row["bin"]),
            "count": int(row["count"]),
            "win_rate": row["win_rate"],
            "ci_low": row["ci_low"],
            "ci_high": row["ci_high"]
        })

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle("Home Win Rate by Feature Quantile Bins", fontsize=16, y=1.02)
plt.savefig("Figures/WinRateByFeatureBins.png", bbox_inches="tight")

bin_summary_df = pd.DataFrame(bin_rows)
bin_summary_df.to_csv("win_rate_by_feature_bins.csv", index=False)

# POST-EDA FEATURE SET FOR MODELING ============================================
features_to_drop_for_modeling = ["tackles_diff", "interceptions_diff", "fumblesLost_diff"]
X_model = X.drop(columns=features_to_drop_for_modeling, errors="ignore")

print(f"\nOriginal EDA feature count: {X.shape[1]}")
print(f"Modeling feature count: {X_model.shape[1]}")
print("Dropped for modeling:", features_to_drop_for_modeling)
print("\nModeling features:")
print(X_model.columns)

modeling_df = X_model.copy()
modeling_df["home_win"] = y.values
modeling_df.to_csv("final_data_EDA.csv", index=False)
print("\nSaved post-EDA modeling dataset: final_data_EDA.csv")
