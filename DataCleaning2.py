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

# --- Load team-level data ---
df = pd.read_csv("cfbd_team_stats_2014_2024.csv")
print(f"Original shape: {df.shape}")

# --- Split into home and away teams ---
home = df[df["homeAway"] == "home"].copy()
away = df[df["homeAway"] == "away"].copy()

# --- Merge into one row per game ---
games = pd.merge(
    home,
    away,
    on="game_id",
    suffixes=("_home", "_away")
)

print(f"Merged shape: {games.shape}")
print(games[["game_id", "team_home", "team_away", "points_home", "points_away"]].head())

# --- Create target variables ---
games["home_win"] = (games["points_home"] > games["points_away"]).astype(int)
games["one_score"] = (abs(games["points_home"] - games["points_away"]) <= 8).astype(int)

# --- Identify numeric columns to create difference features ---
num_cols = [col for col in games.columns if any(stat in col for stat in ["Yards", "TDs", "turnovers", "points"])]

# --- Convert numeric columns safely ---
for col in num_cols:
    games[col] = pd.to_numeric(games[col], errors="coerce")  # non-numeric -> NaN

# # --- Create difference features: home - away ---
# for col in num_cols:
#     if col.endswith("_home") and col.replace("_home", "_away") in games.columns:
#         away_col = col.replace("_home", "_away")
#         diff_col = col.replace("_home", "_diff")
#         games[diff_col] = games[col] - games[away_col]

# print(f"Created {len(num_cols)} difference features.")

# --- Save the final combined dataset ---
games.to_csv("cfbd_games_2014_2024_combined.csv", index=False)
print(f"\n💾 Saved {games.shape[0]} rows × {games.shape[1]} columns → cfbd_games_2014_2024_combined.csv")

games = games.drop([
    'totalPenaltiesYards_away', 
    'totalPenaltiesYards_home', 
    'kickReturns_away', 
    'kickReturnYards_away', 
    'kickReturnTDs_away', 
    'kickReturnYards_home',
    'kickReturnTDs_home',
    'kickReturns_home',
    'interceptionYards_away',
    'interceptionTDs_away',
    'interceptionTDs_home',
    'interceptionYards_home',
    'puntReturnYards_away',
    'puntReturns_away',
    'puntReturnTDs_away',
    'puntReturnYards_home',
    'puntReturns_home',
    'puntReturnTDs_home',
    'defensiveTDs_away',
    'defensiveTDs_home'
    ], axis = 1)
pd.set_option('display.max_rows', None)

null_counts = games.isnull().sum().sort_values(ascending=False)
print(null_counts)

no_nulls = games.dropna()
print(no_nulls.shape)