import sys
import warnings

import numpy as np
import pandas as pd

import utils.c_logging as c_logging

LOG = c_logging.getLogger(__name__)

warnings.filterwarnings("ignore")


class Preprocessor(object):
    def __init__(self, conf):
        self.conf = conf


    def preprocess(self):
        # Load data at the game level only
        LOG.info("Load data at the game level only")
        try:
            games = pd.read_parquet(self.conf.games_file)
        except:
            LOG.error("Cannot find input data")
            sys.exit()

        # Sort and reindex
        LOG.info("Sort and reindex")
        games.sort_values(["season", "datetime"], inplace=True)
        games.reset_index(drop=True, inplace=True)

        # Correct comma sign in some numerical columns
        LOG.info("Correct comma sign in some numerical columns")
        cols = ["away_pace", "away_efg", "away_tov", "away_orb", "away_ftfga", "away_ortg", "home_pace",
                "home_efg", "home_tov", "home_orb", "home_ftfga", "home_ortg"]
        for col in cols:
            games[col] = games[col].str.replace(",", ".")
            games[col] = games[col].astype(float)

        # Retrieve all available seasons
        LOG.info("Retrieve all available seasons")
        seasons = games["season"].unique()
        LOG.info("Number of available seasons: " + str(len(seasons)))

        # Store number of regular season games for each season
        # (smaller number one year because of lockout)
        LOG.info("Store number of regular season games for each season")
        games["nb_reg"] = 82
        games.loc[games["season"] == 2011, "nb_reg"] = 66

        # Iterate on each season
        LOG.info("Iterate on each season")
        all_seasons = []
        for season in seasons:
            season_summaries = self.process_season(games, season)
            all_seasons.extend(season_summaries)

        LOG.info("Save summary dataframe with all seasons")
        df_all_seasons = pd.DataFrame(all_seasons)
        df_all_seasons.columns = df_all_seasons.columns.str.strip()
        df_all_seasons.to_csv("data/processed/all_seasons.csv", header=True, index=False)


    def process_season(self, games, season):
        LOG.info("--- SEASON: " + str(season) + " ---")

        # Select data for that season
        #LOG.info("Iterate on each season")
        current_season = games.loc[games["season"] == season, :]

        # Retrieve teams playing that season
        #LOG.info("Retrieve teams playing that season")
        teams = current_season["home_name"].unique()
        LOG.info("Number of teams playing in " + str(season) + " :" + str(len(teams)))

        # Compute league-wide season averages to account for trend changes over time (pace increasing for ex.)
        #LOG.info("Compute league-wide season averages")
        current_season["league_pace"] = round(((sum(current_season["away_pace"]) + sum(current_season["home_pace"])) / 2) / current_season.shape[0], 3)
        current_season["league_efg"] = round(((sum(current_season["away_efg"]) + sum(current_season["home_efg"])) / 2) / current_season.shape[0], 3)
        current_season["league_tov"] = round(((sum(current_season["away_tov"]) + sum(current_season["home_tov"])) / 2) / current_season.shape[0], 3)
        current_season["league_orb"] = round(((sum(current_season["away_orb"]) + sum(current_season["home_orb"])) / 2) / current_season.shape[0], 3)
        current_season["league_ftfga"] = round(((sum(current_season["away_ftfga"]) + sum(current_season["home_ftfga"])) / 2) / current_season.shape[0], 3)
        current_season["league_ortg"] = round(((sum(current_season["away_ortg"]) + sum(current_season["home_ortg"])) / 2) / current_season.shape[0], 3)

        # Iterate on each team
        #LOG.info("Iterate on each team")
        season_summaries = []
        for team in teams:
            summary_reg = self.process_team(season, current_season, team)
            season_summaries.append(summary_reg)

        return season_summaries


    def process_team(self, season, current_season, team):
        #LOG.info("------ Team: " + team + " ------")
        current_team_home = current_season.loc[current_season["home_name"] == team, :]
        current_team_away = current_season.loc[current_season["away_name"] == team, :]

        # Compute current games played, current wins, current losses
        #LOG.info("Compute current games played, current wins, current losses")
        current_team_home["games_played"] = current_team_home["home_wlratio"].str.split("-", expand=True)[0].astype(int) + \
            current_team_home["home_wlratio"].str.split("-", expand=True)[1].astype(int)
        current_team_away["games_played"] = current_team_away["away_wlratio"].str.split("-", expand=True)[0].astype(int) + \
            current_team_away["away_wlratio"].str.split("-", expand=True)[1].astype(int)
        current_team_home["wins"] = current_team_home["home_wlratio"].str.split("-", expand=True)[0].astype(int)
        current_team_away["wins"] = current_team_away["away_wlratio"].str.split("-", expand=True)[0].astype(int)
        current_team_home["losses"] = current_team_home["home_wlratio"].str.split("-", expand=True)[1].astype(int)
        current_team_away["losses"] = current_team_away["away_wlratio"].str.split("-", expand=True)[1].astype(int)

        # Detect regular season and playoff games
        #LOG.info("Detect regular season and playoff games")
        all_home_rows, all_away_rows = [], []
        previous_high = 0
        for _, row in current_team_home.iterrows():
            if row["games_played"] >= previous_high:
                previous_high = row["games_played"]
                row["is_playoffs"] = 0
            else:
                row["is_playoffs"] = 1
            all_home_rows.append(row)
        previous_high = 0
        for _, row in current_team_away.iterrows():
            if row["games_played"] >= previous_high:
                previous_high = row["games_played"]
                row["is_playoffs"] = 0
            else:
                row["is_playoffs"] = 1
            all_away_rows.append(row)
        current_team_home = pd.DataFrame(all_home_rows)
        current_team_away = pd.DataFrame(all_away_rows)

        current_team_home_reg = current_team_home.loc[current_team_home["is_playoffs"] == 0]
        current_team_away_reg = current_team_away.loc[current_team_away["is_playoffs"] == 0]
        current_team_home_reg.sort_values(["datetime"], inplace=True)
        current_team_away_reg.sort_values(["datetime"], inplace=True)
        current_team_home_reg.reset_index(drop=True, inplace=True)
        current_team_away_reg.reset_index(drop=True, inplace=True)

        current_team_home_playoffs = current_team_home.loc[current_team_home["is_playoffs"] == 1]
        current_team_away_playoffs = current_team_away.loc[current_team_away["is_playoffs"] == 1]
        current_team_playoffs = pd.concat([current_team_home_playoffs, current_team_away_playoffs])
        current_team_playoffs.sort_values(["datetime"], inplace=True)
        current_team_playoffs.reset_index(drop=True, inplace=True)

        # Get total playoff wins/losses (playoff wins == our target)
        #LOG.info("Get total playoff wins/losses")
        previous_wins, previous_losses, total_wins, total_losses = 0, 0, 0, 0
        for _, row in current_team_playoffs.iterrows():
            if (row["wins"] > previous_wins) & (row["losses"] == previous_losses):
                # New win in same series
                total_wins += 1
                previous_wins = row["wins"]
            elif (row["losses"] > previous_losses) & (row["wins"] == previous_wins):
                # New loss in same series
                total_losses += 1
                previous_losses = row["losses"]
            elif (row["wins"] > 0) & (row["losses"] == 0):
                # New win in new series
                total_wins += 1
                previous_wins = row["wins"]
                previous_losses = 0
            elif (row["losses"] > 0) & (row["wins"] == 0):
                # New loss in new series
                total_losses += 1
                previous_losses = row["losses"]
                previous_wins = 0

        # Create team season summaries
        #LOG.info("Create team season summaries")
        attendance = int(np.mean(current_team_home_reg["attendance"]))
        wins = max(max(current_team_home_reg["wins"]), max(current_team_away_reg["wins"]))
        losses = max(max(current_team_home_reg["losses"]), max(current_team_away_reg["losses"]))
        pts_for = np.round(
            (sum(current_team_home_reg["home_ftscore"]) + np.sum(current_team_away_reg["away_ftscore"])) / current_team_home_reg.iloc[0]["nb_reg"], 4)
        pts_against = np.round(
            (sum(current_team_home_reg["away_ftscore"]) + np.sum(current_team_away_reg["home_ftscore"])) / current_team_home_reg.iloc[0]["nb_reg"], 4)
        pts_diff = np.round(pts_for - pts_against, 4)
        pace = np.round(
            (sum(current_team_home_reg["home_pace"]) + sum(current_team_away_reg["away_pace"])) / current_team_home_reg.iloc[0]["nb_reg"], 4)
        pace_vs_avg = np.round(pace - current_team_home_reg.iloc[0]["league_pace"], 4)
        efg = np.round(
            (sum(current_team_home_reg["home_efg"]) + sum(current_team_away_reg["away_efg"])) / current_team_home_reg.iloc[0]["nb_reg"], 4)
        efg_vs_avg = np.round(efg - current_team_home_reg.iloc[0]["league_efg"], 4)
        opp_efg = np.round(
            (sum(current_team_home_reg["away_efg"]) + sum(current_team_away_reg["home_efg"])) / current_team_home_reg.iloc[0]["nb_reg"], 4)
        opp_efg_vs_avg = np.round(opp_efg - current_team_home_reg.iloc[0]["league_efg"], 4)
        tov = np.round(
            (sum(current_team_home_reg["home_tov"]) + sum(current_team_away_reg["away_tov"])) / current_team_home_reg.iloc[0]["nb_reg"], 4)
        tov_vs_avg = np.round(tov - current_team_home_reg.iloc[0]["league_tov"], 4)
        opp_tov = np.round(
            (sum(current_team_home_reg["away_tov"]) + sum(current_team_away_reg["home_tov"])) / current_team_home_reg.iloc[0]["nb_reg"], 4)
        opp_tov_vs_avg = np.round(opp_tov - current_team_home_reg.iloc[0]["league_tov"], 4)
        orb = np.round(
            (sum(current_team_home_reg["home_orb"]) + sum(current_team_away_reg["away_orb"])) / current_team_home_reg.iloc[0]["nb_reg"], 4)
        orb_vs_avg = np.round(orb - current_team_home_reg.iloc[0]["league_orb"], 4)
        opp_orb = np.round(
            (sum(current_team_home_reg["away_orb"]) + sum(current_team_away_reg["home_orb"])) / current_team_home_reg.iloc[0]["nb_reg"], 4)
        opp_orb_vs_avg = np.round(opp_orb - current_team_home_reg.iloc[0]["league_orb"], 4)
        ftfga = np.round(
            (sum(current_team_home_reg["home_ftfga"]) + sum(current_team_away_reg["away_ftfga"])) / current_team_home_reg.iloc[0]["nb_reg"], 4)
        ftfga_vs_avg = np.round(ftfga - current_team_home_reg.iloc[0]["league_ftfga"], 4)
        opp_ftfga = np.round(
            (sum(current_team_home_reg["away_ftfga"]) + sum(current_team_away_reg["home_ftfga"])) / current_team_home_reg.iloc[0]["nb_reg"], 4)
        opp_ftfga_vs_avg = np.round(opp_ftfga - current_team_home_reg.iloc[0]["league_ftfga"], 4)
        ortg = np.round(
            (sum(current_team_home_reg["home_ortg"]) + sum(current_team_away_reg["away_ortg"])) / current_team_home_reg.iloc[0]["nb_reg"], 4)
        ortg_vs_avg = np.round(ortg - current_team_home_reg.iloc[0]["league_ortg"], 4)
        opp_ortg = np.round(
            (sum(current_team_home_reg["away_ortg"]) + sum(current_team_away_reg["home_ortg"])) / current_team_home_reg.iloc[0]["nb_reg"], 4)
        opp_ortg_vs_avg = np.round(opp_ortg - current_team_home_reg.iloc[0]["league_ortg"], 4)

        summary_reg = {
            "season": season,
            "name": current_team_home_reg.iloc[0]["home_name"],
            "wins": wins,
            "losses": losses,
            #"pts_for": pts_for,
            #"pts_against": pts_against,
            "pts_diff": pts_diff,
            #"efg": efg,
            "efg_vs_avg": efg_vs_avg,
            "opp_efg_vs_avg": opp_efg_vs_avg,
            #"tov": tov,
            "tov_vs_avg": tov_vs_avg,
            "opp_tov_vs_avg": opp_tov_vs_avg,
            #"orb": orb,
            "orb_vs_avg": orb_vs_avg,
            "opp_orb_vs_avg": opp_orb_vs_avg,
            #"ftfga": ftfga,
            "ftfga_vs_avg": ftfga_vs_avg,
            "opp_ftfga_vs_avg": opp_ftfga_vs_avg,
            #"ortg": ortg,
            "ortg_vs_avg": ortg_vs_avg,
            "opp_ortg_vs_avg": opp_ortg_vs_avg,
            #"pace": pace,
            "pace_vs_avg": pace_vs_avg,
            "attendance": attendance,
            "playoff_wins": total_wins,
        }

        return summary_reg
