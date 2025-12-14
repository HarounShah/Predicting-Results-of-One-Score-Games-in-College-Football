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
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, RocCurveDisplay
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC

# SPLITTING DATA INTO TRAIN AND TEST ==========================================
df = pd.read_csv('/Users/harounshah/Downloads/Senior Thesis/final_data.csv')

# df = df.filter(like="diff", axis=1) # Taking only the difference features]

print(df.columns)