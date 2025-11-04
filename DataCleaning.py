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

data = pd.read_csv("/Users/harounshah/Downloads/Senior Thesis/Data/Final.csv")

data['result'] = (data['home_score'] > data['away_score']).astype(int) # Creating label - *REMEMBER TO DELETE*
data = data.drop(['away_adjQBR', 'home_adjQBR'], axis = 1) # Removing QBR columns

pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

print(data.head())
print(data.shape)

null_counts = data.isnull().sum().sort_values(ascending=False)
print(null_counts)

# print(data[data['away_interceptions'].isnull()])
print(data[data['away_receptions'].isnull()])

# print(data[data['home_interceptions'].isnull()]['away_interceptions'])


