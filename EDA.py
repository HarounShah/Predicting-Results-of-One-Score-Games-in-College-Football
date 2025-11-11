# IMPORTS =====================================================================
import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
import sklearn
import requests
from bs4 import BeautifulSoup
import json
import time
import seaborn as sns

# SPLITTING DATA INTO X & Y ===================================================
df = pd.read_csv('/Users/harounshah/Downloads/Senior Thesis/cfbd_games_2014_2024_combined.csv')

print(df.head())
print(df.shape)
print(df.columns)

y = df['home_win']
X = df.filter(like="diff", axis=1)

print(X.shape)
# print(X.columns)


# FEATURE VISUALIZATIONS ======================================================
# Set up plotting style
sns.set(style="whitegrid", palette="muted", font_scale=0.9)

# Create a figure grid — adjust size to fit all variables
n_cols = 5
n_rows = int(len(X.columns) / n_cols) + 1
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, n_rows * 1.8))
axes = axes.flatten()

# HISTOGRAMS
for i, col in enumerate(X.columns):
    ax = axes[i]
    sns.histplot(X[col], kde=True, ax=ax, bins=20, color='skyblue')
    ax.set_title(col)
    ax.set_xlabel("")
    ax.set_ylabel("Count")

# Hide any extra subplots (if grid > number of columns)
for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle("Distribution of '_diff' Features", fontsize=16, y=1.02)
plt.show()

# BOXPLOTS
n_cols = 5
n_rows = int(len(X.columns) / n_cols) + 1
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, n_rows * 1.8))
axes = axes.flatten()

for i, col in enumerate(X.columns):
    ax = axes[i]
    sns.boxplot(x=X[col], ax=ax, color='lightgreen')
    ax.set_title(col)
    ax.set_xlabel("")

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle("Boxplots of '_diff' Features", fontsize=16, y=1.02)
plt.show()

# CORRELATION MATRIX
plt.figure(figsize=(10, 8))
corr = X.corr()
sns.heatmap(corr, annot=False, cmap="coolwarm", center=0)
plt.title("Correlation Matrix of '_diff' Features")
plt.show()

# HISTOGRAMS (GROUPED BY CLASS)
n_cols = 5
n_rows = int(len(X.columns) / n_cols) + 1
fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, n_rows * 1.8))
axes = axes.flatten()

for i, col in enumerate(X.columns):
    ax = axes[i]
    sns.histplot(data=df, x=col, hue=y, kde=True, element="step", ax=ax)
    ax.set_title(col)
    ax.set_xlabel("")
    ax.set_ylabel("Count")
    ax.legend_.remove()

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.suptitle("Feature Histograms by Class", fontsize=16, y=1.02)
plt.show()
