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

# SPLITTING DATA INTO X & Y ===================================================
df = pd.read_csv('/Users/harounshah/Downloads/Senior Thesis/final_data.csv')

print(f"\nMatrix Dimensions w/ all Features: {df.shape}\n")

y = df['home_win']
X = df.filter(like="diff", axis=1) # Taking only the difference features

print(f"Matrix Dimensions w/ only Difference Features: {X.shape}\n")
print(X.columns, '\n')


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
