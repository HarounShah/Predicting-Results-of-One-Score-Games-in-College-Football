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
columns_to_drop = ['home_adjQBR', 'away_adjQBR', 
                   'away_grossAvgPuntYards', 'away_punts', 'away_puntYards', 'away_touchbacks', 'away_puntsInside20', 'away_longPunt',
                   'home_grossAvgPuntYards', 'home_punts', 'home_puntYards', 'home_touchbacks', 'home_puntsInside20', 'home_longPunt',
                   'home_kickReturnTouchdowns', 'home_yardsPerKickReturn', 'home_longKickReturn', 'home_kickReturns', 'home_kickReturnYards',
                   'away_kickReturnTouchdowns', 'away_yardsPerKickReturn', 'away_longKickReturn', 'away_kickReturns', 'away_kickReturnYards',
                   'home_yardsPerPuntReturn', 'home_longPuntReturn', 'home_puntReturnTouchdowns', 'home_puntReturnYards', 'home_puntReturns',
                   'away_yardsPerPuntReturn', 'away_longPuntReturn', 'away_puntReturnTouchdowns', 'away_puntReturnYards', 'away_puntReturns'
                   ]
rows_to_drop = ['401310704', '400787107', '400763639', '401286296', '400869842', '401015050', '401310704', '400934561', '401301030', '401310704', '401415661', '401643763']
data = data.drop(columns_to_drop, axis = 1) # Removing QBR, punting, punt returning, and kick returning data
# data = data.drop(rows_to_drop, axis = 0) # Removing 'odd' games with important missing data

pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

print(data.head())
print(data.shape)

null_counts = data.isnull().sum().sort_values(ascending=False)
print(null_counts)

# print(data[data['away_interceptions'].isnull()])
print(data[data['away_receptions'].isnull()])

# print(data[data['home_interceptions'].isnull()]['away_interceptions'])


