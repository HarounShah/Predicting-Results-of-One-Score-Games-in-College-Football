# Predicting Results of One Score Games in College Football

Data Collection:

Data Cleaning:

    Questions:
        If a team does not attempt any 3rd/4th down conversions, how do we replace the '-' in the dataframe?

    Missing Data: 
        There is a game (401310704) where all away team passing, rushing, and receiving is missing.
        There is a game (400787107) where both home and away posession time is missing.
        There are two games (400763639 & 401286296) where home passing is missing.
        There are three games (400869842, 401015050, 401310704) where home kicking is missing.
        There are five games (400934561, 401301030, 401310704, 401415661, 401643763) where home receiving is missing.
        There are six games () where away kicking is missing.
        There are seven games () where home receiving is missing.
        There are 15 games where away punting is missing.
        There are 17 games where home punting is missing.
        There are 339 games where home kick returning is missing.
        There are 403 games where away kick returning is missing.
        There are 791 games where home punt returning is missing.
        There are 902 games wehere away punt returning is missing.
        There are 1000 games where away fumbling is missing.
        There are 1020 games where home fumbling is missing.
        There are 1283 games where away intercepting is missing.
        There are 1322 games where home intercepting is missing.
        There are 2395 games where ALL DEFENSIVE DATA IS MISSING.

    Solutions to Missing Data:
        Remove games with rare missing data.
        Remove punting, kick returning, and punt returning data.
            Doesn't seem that important.

    To Do:
        Replace null interceptions with 0.

Exploratory Data Analysis: