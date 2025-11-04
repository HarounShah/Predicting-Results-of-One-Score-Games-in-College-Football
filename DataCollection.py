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


# READING IN AND UNDERSTANDING THE DATA =======================================

data_df = pd.DataFrame()
for i in range(2014, 2025):
    if i == 2020:
        pass
    else:
        temp_df = pd.read_csv(f'/Users/harounshah/Downloads/Senior Thesis/{i}.csv')
        data_df = pd.concat([data_df, temp_df], axis = 0)

# print('\nSize of Dataframe (Rows, Columns): ' + str(data_df.shape) + '\n')

# print(data_df.dtypes)
# print('\n========================================\n')

    

# SELECTING ONE-SCORE GAMES ===================================================

onescore_data_df = data_df[abs(data_df['HomePoints'] - data_df['AwayPoints']) <= 8]

print('\n# of One-Score FBS Games: ' + str(onescore_data_df.shape[0]) + '\n')


# EXTRACTING GAME IDs TO COMPILE ESPN URLS ====================================

game_ids = onescore_data_df['Id']
game_ids = game_ids.reset_index(drop = True)

game_urls = []

for id in game_ids:
    game_urls.append(
        f'https://www.espn.com/college-football/matchup/_/gameId/{id}')


# EXTRACTING GAME DATA FROM ESPN ==============================================

def get_game_summary(game_id):
    """Fetch summary stats for one game from ESPN API."""
    url = f"https://site.api.espn.com/apis/site/v2/sports/football/college-football/summary?event={game_id}"
    headers = {"User-Agent": "Mozilla/5.0"}

    # If url not valid, pass over it
    try:
        r = requests.get(url, headers=headers, timeout=10)
        if r.status_code == 404:
            print(f"Game {game_id} not found (404)")
            return None
        r.raise_for_status()
        data = r.json()
    except requests.exceptions.RequestException as e:
        print(f"Request failed for {game_id}: {e}")
        return None

    try: 
        # Parse competitor info
        competitors = data["header"]["competitions"][0]["competitors"]
        home = next(t for t in competitors if t["homeAway"] == "home")
        away = next(t for t in competitors if t["homeAway"] == "away")

        # Defensive parsing of boxscore data
        teams_stats = data.get("boxscore", {}).get("teams", [])
        if len(teams_stats) < 2:
            print(f"Missing boxscore data for {game_id}")
            return None

        # ESPN sometimes flips order; match by teamId
        def get_team_stats(team_id):
            for team in teams_stats:
                if str(team["team"]["id"]) == str(team_id):
                    return team["statistics"]
            return None
        
        # Defensive parsing of boxscore data
        all_stats = data.get("boxscore", {}).get("players", [])
        if len(teams_stats) < 2:
            print(f"Missing boxscore data for {game_id}")
            return None
        
        # ESPN sometimes flips order; match by teamId
        def get_all_stats(team_id):
            for team in all_stats:
                if str(team["team"]["id"]) == str(team_id):
                    return team["statistics"]
            return None

        home_stats = get_team_stats(home["team"]["id"])
        away_stats = get_team_stats(away["team"]["id"])
        all_home_stats = get_all_stats(home["team"]["id"])
        all_away_stats = get_all_stats(away["team"]["id"])

        def grab_all_stats(stats, stat, home_away):
            # ESPN uses camelCase keys for stat names in the JSON
            stat_map = {
                'passing': 'passing',
                'rushing': 'rushing',
                'receiving': 'receiving',
                'fumbles': 'fumbles',
                'defense': 'defensive',        # note: sometimes "defensive"
                'interceptions': 'interceptions',
                'kick_returns': 'kickReturns',
                'punt_returns': 'puntReturns',
                'kicking': 'kicking',
                'punting': 'punting'
            }

            if stat not in stat_map:
                print(f"Invalid stat name: {stat}")
                return {}

            stat_name = stat_map[stat]

            # Find the correct section dynamically
            section = next((s for s in stats if s.get("name") == stat_name), None)
            if section is None:
                # print(f"⚠️ Missing {stat_name} stats for {home_away} team.")
                return {}

            dictionary2 = dict(zip(section.get("keys", []), section.get("totals", [])))

            if home_away == 'home':
                dictionary2 = {f"home_{k}": v for k, v in dictionary2.items()}
            elif home_away == 'away':
                dictionary2 = {f"away_{k}": v for k, v in dictionary2.items()}

            return dictionary2

        if not home_stats or not away_stats or not all_home_stats or not all_away_stats:
            print(f"Stats missing for {game_id}")
            return None

        # Flatten into a dict
        def safe_value(stats, i):
            return stats[i]["value"] if i < len(stats) else None

        season = data["header"]["season"]["year"]

        final = {
            "game_id": game_id,
            "season": season,
            "home_team": home["team"]["displayName"],
            "away_team": away["team"]["displayName"],
            "home_score": home["score"],
            "away_score": away["score"],
            "home_first_downs": safe_value(home_stats, 0),
            "away_first_downs": safe_value(away_stats, 0),
            "home_3d_eff": safe_value(home_stats, 1),
            "away_3d_eff": safe_value(away_stats, 1),
            "home_4d_eff": safe_value(home_stats, 2),
            "away_4d_eff": safe_value(away_stats, 2),
            # "home_yards": safe_value(home_stats, 3),
            # "away_yards": safe_value(away_stats, 3),
            # "home_pass_yards": safe_value(home_stats, 4),
            # "away_pass_yards": safe_value(away_stats, 4),
            # "home_comp_atmpts": safe_value(home_stats, 5),
            # "away_comp_atmpts": safe_value(away_stats, 5),
            # "home_ypp": safe_value(home_stats, 6),
            # "away_ypp": safe_value(away_stats, 6),
            # "home_rush_yards": safe_value(home_stats, 7),
            # "away_rush_yards": safe_value(away_stats, 7),
            # "home_rush_atmpts": safe_value(home_stats, 8),
            # "away_rush_atmpts": safe_value(away_stats, 8),
            # "home_ypr": safe_value(home_stats, 9),
            # "away_ypr": safe_value(away_stats, 9),
            "home_pens_yards": home_stats[10]['displayValue'],
            "away_pens_yards": away_stats[10]['displayValue'],
            "home_turnovers": home_stats[11]['displayValue'],
            "away_turnovers": away_stats[11]['displayValue'],
            # "home_fumbles": safe_value(home_stats, 12),
            # "away_fumbles": safe_value(away_stats, 12),
            # "home_ints": safe_value(home_stats, 13),
            # "away_ints": safe_value(away_stats, 13),
            "home_possession": safe_value(home_stats, 14),
            "away_possession": safe_value(away_stats, 14)
        }

        stats = ['passing', 'rushing', 'receiving', 'fumbles', 'defense', 'interceptions', 'kick_returns', 'punt_returns', 'kicking', 'punting']
        for stat in stats:
            add_home = grab_all_stats(all_home_stats, stat, 'home')
            add_away = grab_all_stats(all_away_stats, stat, 'away')
            final.update(add_home)
            final.update(add_away)
        return final


    # If unable to parse, throw an error
    except Exception as e:
        print(f"Error parsing {game_id}: {e}")
        return None

game_data = []
failed_games = []

for i, game_id in enumerate(game_ids, start=1):
    info = get_game_summary(game_id)
    if info:
        game_data.append(info)
    else:
        failed_games.append(game_id)
    
    if i % 100 == 0:
        print(f"Processed {i} games...")
    time.sleep(0.2)

games_df = pd.DataFrame(game_data)
print("\n✅ Successfully parsed games:", len(games_df))
print("❌ Failed games:", len(failed_games), ':', failed_games)
print(games_df.head())

games_df.to_csv('/Users/harounshah/Downloads/Senior Thesis/Final.csv', index=False)


# MESSING W JSON EXPLORATION ETC ==============================================

url = f"https://site.api.espn.com/apis/site/v2/sports/football/college-football/summary?event={game_ids[0]}"
headers = {"User-Agent": "Mozilla/5.0"}
data = requests.get(url, headers=headers).json()

# print('SUBSECTIONS OF EACH JSON')
# print(data.keys(), '\n')

# print('AVAILABLE TEAM STATS:')
# for i in range(len(data['boxscore']['teams'][0]['statistics'])):
#     print(data['boxscore']['teams'][0]['statistics'][i]['name'])

# for key in data.keys():
#     print(f"\n=== {key} ===")
#     if isinstance(data[key], dict):
#         print(data[key].keys(), '\n')
#     elif isinstance(data[key], list):
#         # If the value is a list, print keys of the first item (if it’s a dict)
#         if len(data[key]) > 0 and isinstance(data[key][0], dict):
#             print(data[key][0].keys(), '\n')
#         else:
#             print(f"List with {len(data[key])} items (no dicts inside)\n")
#     else:
#         print(f"Value type: {type(data[key]).__name__}\n")

def explore_json(d, depth=0, max_depth=3):
    if depth > max_depth:
        return
    if isinstance(d, dict):
        for k, v in d.items():
            print("  " * depth + f"{k}: {type(v).__name__}")
            explore_json(v, depth + 1, max_depth)
    elif isinstance(d, list):
        print("  " * depth + f"[List of {len(d)} items]")
        if len(d) > 0:
            explore_json(d[0], depth + 1, max_depth)

# explore_json(data['boxscore']['teams'][0]['statistics'], max_depth=2)

# print(json.dumps(data['boxscore']['players'][0]['statistics'][1]['totals'], indent=2))

# away_team = data['boxscore']['players'][0]['statistics']
# home_team = data['boxscore']['players'][1]['statistics']

# passing = dict(zip(away_team[0]['keys'], away_team[0]['totals']))
# rushing = dict(zip(away_team[1]['keys'], away_team[1]['totals']))
# receiving = dict(zip(away_team[2]['keys'], away_team[2]['totals']))
# fumbles = dict(zip(away_team[3]['keys'], away_team[3]['totals']))
# defense = dict(zip(away_team[4]['keys'], away_team[4]['totals']))
# interceptions = dict(zip(away_team[5]['keys'], away_team[5]['totals']))
# kick_returns = dict(zip(away_team[6]['keys'], away_team[6]['totals']))
# punt_returns = dict(zip(away_team[7]['keys'], away_team[7]['totals']))
# kicking = dict(zip(away_team[8]['keys'], away_team[8]['totals']))
# punting = dict(zip(away_team[9]['keys'], away_team[9]['totals']))

# teams_stats = data.get("boxscore", {}).get("teams", [])

# print(teams_stats[0]['statistics'][10])
# print(teams_stats[0]['statistics'][11])

# print(get_game_summary(401012787)
# print(get_game_summary(401112128)