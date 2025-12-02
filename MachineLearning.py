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
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# SPLITTING DATA INTO TRAIN AND TEST ==========================================
df = pd.read_csv('/Users/harounshah/Downloads/Senior Thesis/final_data.csv')

print(f"\nMatrix Dimensions w/ all Features: {df.shape}\n")

y = df['home_win']
X = df.filter(like="diff", axis=1) # Taking only the difference features

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# RANDOM FOREST ===============================================================
rf_model = RandomForestClassifier(n_estimators=200, random_state=42)

rf_model.fit(X_train, y_train)

y_pred = rf_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Random Forest Test Accuracy:", accuracy)

importances = pd.Series(rf_model.feature_importances_, index=X.columns)
importances.sort_values().plot(kind="barh", figsize=(12,8))
plt.title("Feature Importances")
plt.savefig("Figures/Feature_Importance.png")

# LOGISTIC REGRESSION =========================================================