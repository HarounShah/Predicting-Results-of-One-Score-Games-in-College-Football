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
from MachineLearning import log_reg, rf_model, gb_model, svm_model, nb_model
import copy

# SPLITTING DATA INTO TRAIN AND TEST ==========================================
df = pd.read_csv('/Users/harounshah/Downloads/Senior Thesis/final_data_test.csv')

# # print(f"\nMatrix Dimensions w/ all Features: {df.shape}\n")
# # df = df.filter(like="diff", axis=1) # Taking only the difference feature
# # print(df.columns)

# teams = ["James Madison", "Oregon", "Texas Tech", 
#          "Alabama", "Oklahoma", "Indiana",
#          "Tulane", "Ole Miss", "Georgia",
#          "Miami", "Texas A&M", "Ohio State"]

# # for team in teams:
# #     count = 0
# #     for row in df['team_home']:
# #         if row == team:
# #             count += 1
# #     for row in df['team_away']:
# #         if row == team:
# #             count += 1
# #     print(f"{team} appears {count} times")

# def get_team_avg_diffs(df, teams):

#     diff_cols = df.filter(like = "_diff").columns

#     team_season_stats = {}
#     for team in teams:
#         temp = df[(df['team_home'] == team) | (df['team_away'] == team)].sort_values(by='week').copy()
#         away_mask = temp['team_away'] == team
#         temp.loc[away_mask, diff_cols] *= -1
#         avg_diffs = temp[diff_cols].mean().to_dict()
#         team_season_stats[team] = avg_diffs

#     return team_season_stats

# def predict_matchup(df, teams, avg_diffs, model):

#     home_team = teams[0]
#     away_team = teams[1]
#     team_stats = {home_team: avg_diffs[home_team], away_team: avg_diffs[away_team]}

#     matchup = {}
#     diff_cols = df.filter(like = "_diff").columns
#     for col in diff_cols:
#         matchup[col] = team_stats[home_team][col] - team_stats[away_team][col]

#     deflections = matchup.pop('passesDeflected_diff')
#     third_down = matchup.pop('thirdDown%_diff')
#     matchup['passesDeflected_diff'] = deflections
#     matchup['thirdDown%_diff'] = third_down

#     test_df = pd.DataFrame([matchup])

#     return int(model.predict(test_df)[0])

# avg_diffs = get_team_avg_diffs(df, teams)

# first_round = [['Oregon', 'James Madison'], ['Oklahoma', 'Alabama'], ['Ole Miss', 'Tulane'], ['Texas A&M', 'Miami']]
# quarterfinals = [['Texas Tech'], ['Indiana'], ['Georgia'], ['Ohio State']]

# def create_bracket(avg_diffs, first_round, quarterfinals, model):
#     semifinals = [[], []]
#     final = []
#     quarterfinals = copy.deepcopy(quarterfinals)
#     # print("\nFirst Round Matchups:", first_round)
#     winners1 = []
#     for matchup in first_round:
#         pred = predict_matchup(df, matchup, avg_diffs, model)
#         if pred == 0:
#             winners1.append(matchup[1])
#         else:
#             winners1.append(matchup[0])

#     for i in range(4):
#         quarterfinals[i].append(winners1[i])
#     # print("\nQuarterfinal Matchups:", quarterfinals)

#     winners2 = []
#     for matchup in quarterfinals:
#         pred = predict_matchup(df, matchup, avg_diffs, model)
#         if pred == 0:
#             winners2.append(matchup[1])
#         else:
#             winners2.append(matchup[0])

#     semifinals[0].append(winners2[0])
#     semifinals[0].append(winners2[1])
#     semifinals[1].append(winners2[2])
#     semifinals[1].append(winners2[3])
#     # print("\nSemifinal Matchups:", semifinals)

#     final = []

#     for matchup in semifinals:
#         team1, team2 = matchup
#         pred = predict_matchup(df, matchup, avg_diffs, model)

#         winner = team1 if pred == 1 else team2
#         final.append(winner)
    
#     team1, team2 = final
#     pred = predict_matchup(df, final, avg_diffs, model)

#     champion = team1 if pred == 1 else team2

#     # ==================
#     # VISUALIZATION
#     # ==================
    
#     fig, ax = plt.subplots(figsize=(16, 8))
#     ax.axis("off")

#     x = {"first": 0.1, "quarter": 0.35, "semi": 0.6, "final": 0.85}

#     y_first = [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2]
#     y_quarter = [0.75, 0.55, 0.35, 0.15]
#     y_semi = [0.6, 0.3]
#     y_final = 0.45

#     LINE_COLOR = "#333333"

#     # ---- First Round ----
#     for i, matchup in enumerate(first_round):
#         ax.text(x["first"], y_first[2*i], matchup[0], ha="left", va="center")
#         ax.text(x["first"], y_first[2*i+1], matchup[1], ha="left", va="center")

#         ax.plot([x["first"], x["quarter"]],
#                 [y_first[2*i], y_quarter[i]], lw=1, color = LINE_COLOR)
#         ax.plot([x["first"], x["quarter"]],
#                 [y_first[2*i+1], y_quarter[i]], lw=1, color = LINE_COLOR)

#     # ---- Quarterfinals ----
#     for i, matchup in enumerate(quarterfinals):
#         bye_team = matchup[0]
#         winner = matchup[1]

#         ax.text(x["quarter"], y_quarter[i] + 0.035,
#                 bye_team, ha="center", va="center")

#         ax.text(x["quarter"], y_quarter[i] - 0.035,
#                 winner, ha="center", va="center", fontweight="bold")

#         ax.plot([x["quarter"], x["semi"]],
#                 [y_quarter[i], y_semi[i//2]], lw=1, color = LINE_COLOR)

#     # ---- Semifinals ----
#     for i, matchup in enumerate(semifinals):
#         team1, team2 = matchup

#         # Determine winner
#         winner = final[i]
#         y_center = y_semi[i] 

#         # Draw both team names
#         ax.text(x["semi"], y_semi[i] + 0.04, team1,
#                 ha="center", va="center",
#                 fontweight="bold" if team1 == winner else "normal")

#         ax.text(x["semi"], y_semi[i] - 0.04, team2,
#                 ha="center", va="center",
#                 fontweight="bold" if team2 == winner else "normal")

#         ax.plot([x["semi"], x["final"]],
#                 [y_center, y_final],
#                 color=LINE_COLOR,
#                 lw=1)

#     # ---- Final ----
#     team1, team2 = final

#     ax.text(x["final"], y_final + 0.04, team1,
#             ha="center", va="center",
#             fontweight="bold" if team1 == champion else "normal")

#     ax.text(x["final"], y_final - 0.04, team2,
#             ha="center", va="center",
#             fontweight="bold" if team2 == champion else "normal")

#     ax.text(x["final"] + 0.07, y_final,
#             f"{champion}",
#             ha="left", va="center",
#             fontsize=14, fontweight="bold")

#     models = {
#     log_reg: "Logistic Regression",
#     rf_model: "Random Forest",
#     gb_model: "Gradient Boosting",
#     svm_model: "SVM",
#     nb_model: "Naive Bayes"}
#     plt.title(f"College Football Playoff Bracket Prediction - {models[model]}", fontsize=16)
#     ax.set_xlim(0.15, 0.95)
#     ax.set_ylim(0.05, 0.95)
#     plt.savefig(f"Figures/Brackets/Bracket - {models[model]}")

#     # ==================
#     # RESULTS
#     # ==================
#     correct = 0
#     for winner in winners1:
#         if winner in ['Oregon', 'Alabama', 'Ole Miss', 'Miami']:
#             correct += 1
#     for winner in winners2:
#         if winner in ['Oregon', 'Indiana', 'Ole Miss', 'Miami']:
#             correct += 1
#     if team1 in ['Indiana', 'Miami']:
#         correct += 1
#     if team2 in ['Indiana', 'Miami']:
#         correct += 1
#     if champion == 'Indiana':
#         correct += 1
#     score = correct / 11
#     print(f"{models[model]} Score: {correct} / 11 = {round(score, 3)}")

# for model in [log_reg, rf_model, gb_model, svm_model, nb_model]:
#     create_bracket(avg_diffs, first_round, quarterfinals, model)

# =====================
# POST PLAYOFF ANALYSIS
# =====================
    
# Same 12 difference features as final_data_EDA.csv / MachineLearning.py.
# Dropped for modeling (redundant or weak): interceptions_diff, fumblesLost_diff,
# totalYards_diff, tackles_diff — keep turnovers_diff, rushing/passing splits, etc.
FEATURE_NAMES = [
    "possessionTime_diff",
    "turnovers_diff",
    "yardsPerRushAttempt_diff",
    "rushingYards_diff",
    "yardsPerPass_diff",
    "netPassingYards_diff",
    "firstDowns_diff",
    "tacklesForLoss_diff",
    "sacks_diff",
    "qbHurries_diff",
    "passesDeflected_diff",
    "thirdDown%_diff",
]

TexMia = [
    7.4667,   # possessionTime_diff (7:28)
    2,        # turnovers_diff
    -3.8,     # yardsPerRushAttempt_diff
    -86,      # rushingYards_diff
    1.0,      # yardsPerPass_diff
    134,      # netPassingYards_diff
    9,        # firstDowns_diff
    -3,       # tacklesForLoss_diff
    -5,       # sacks_diff
    0,        # qbHurries_diff
    0,        # passesDeflected_diff
    0.1944,   # thirdDown%_diff
]

GeoOM = [
    5.0667,    # possessionTime_diff
    0,         # turnovers_diff
    -0.7,      # yardsPerRushAttempt_diff
    13,        # rushingYards_diff
    -1.3,      # yardsPerPass_diff (Game ID 401769073)
    -143,      # netPassingYards_diff
    0,         # firstDowns_diff
    -2,        # tacklesForLoss_diff
    -1,        # sacks_diff
    0,         # qbHurries_diff
    4,         # passesDeflected_diff
    -0.1269,   # thirdDown%_diff
]

OMMia = [
    -22.7333,  # possessionTime_diff (18:38 − 41:22)
    -1,        # turnovers_diff
    2.1,       # yardsPerRushAttempt_diff
    -70,       # rushingYards_diff
    -0.1,      # yardsPerPass_diff
    9,         # netPassingYards_diff
    -5,        # firstDowns_diff
    4,         # tacklesForLoss_diff (5 − 1)
    3,         # sacks_diff (4 − 1)
    0,         # qbHurries_diff (CFBD 0 both)
    4,         # passesDeflected_diff (7 − 3)
    -0.109,    # thirdDown%_diff (≈0.462 − ≈0.571)
]

IndMia = [
    12.8,      # possessionTime_diff (36.4 − 23.6 minutes)
    -1,        # turnovers_diff (0 − 1)
    -2.3,      # yardsPerRushAttempt_diff (2.9 − 5.2)
    21,        # rushingYards_diff (131 − 110)
    -0.4,      # yardsPerPass_diff
    -46,       # netPassingYards_diff (186 − 232)
    5,         # firstDowns_diff (20 − 15)
    0,         # tacklesForLoss_diff
    -2,        # sacks_diff (IND 1 − MIA 3)
    0,         # qbHurries_diff
    1,         # passesDeflected_diff
    0.101,     # thirdDown%_diff (0.375 − 0.273)
]

GAMES = [
    ("Texas A&M @ Miami (TexMia)", TexMia),
    ("Georgia @ Ole Miss (GeoOM)", GeoOM),
    ("Ole Miss @ Miami (OMMia)", OMMia),
    ("Indiana @ Miami (IndMia)", IndMia),
]

MODELS = [
    ("Logistic Regression", log_reg),
    ("Random Forest", rf_model),
    ("Gradient Boosting", gb_model),
    ("SVM", svm_model),
    ("Naive Bayes", nb_model),
]

for game_label, game in GAMES:
    X = pd.DataFrame([game], columns=FEATURE_NAMES)
    print(f"\n{game_label}")
    print("  (class 0 = away win, class 1 = home win)")
    for model_name, model in MODELS:
        pred = int(model.predict(X)[0])
        proba = model.predict_proba(X)[0]
        proba_str = np.array2string(
            np.round(proba, 4), precision=4, separator=", "
        )
        print(f"  {model_name:20s}  pred={pred}  P(away), P(home) = {proba_str}")