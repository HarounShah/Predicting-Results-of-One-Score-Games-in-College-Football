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

print(data.head())
print(data.shape)

null_counts = data.isnull().sum().sort_values(ascending=False)

print(null_counts)

data2 = data.dropna().reset_index()

print(data2.head())
print(data2.shape)