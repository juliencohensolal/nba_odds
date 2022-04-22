import sys
import time

import joblib
from lightgbm.sklearn import LGBMRegressor
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import utils.c_logging as c_logging

LOG = c_logging.getLogger(__name__)


def scoring(y_true, y_pred):
    if isinstance(y_true, pd.core.series.Series):
        mask = y_true.notnull()
        y_true = y_true[mask].tolist()
        y_pred = y_pred[mask].tolist()
    elif isinstance(y_pred, np.ndarray):
        negmask = np.isnan(y_true)
        y_true = y_true[~negmask]
        y_pred = y_pred[~negmask]

    return np.round(mean_squared_error(y_true, y_pred), 5)


class Oddsmaker(object):
    def __init__(self, conf, experiment_id):
        self.conf = conf
        self.experiment_id = experiment_id


    def train_model(self):
        # Load preprocessed data
        LOG.info("Load preprocessed data")
        try:
            all_seasons = pd.read_csv("data/processed/all_seasons.csv")
        except:
            LOG.error("Cannot find preprocessed data")
            sys.exit()

        # Use all seasons but 2018 (the one we try to predict) and 2017 (which will be used for prediction)
        LOG.info("Use all seasons but 2018 and 2017")
        all_seasons = all_seasons.loc[(all_seasons["season"] != 2018) & (all_seasons["season"] != 2017), :]
        
        # Define model
        # Using best model among several benchmarked (incl. RidgeRegressor, Extra-Trees, XGBoost, LightGBM,etc)
        # Hyper-parameter values found through several iterations of Bayesian Optimization
        LOG.info("Define model")
        model = LGBMRegressor(
            n_estimators=5000,
            learning_rate=0.025,
            max_depth=7,
            num_leaves=127,
            colsample_bytree=0.75,
            min_child_samples=20,
            min_split_gain=0.005,
            subsample=0.60,
            subsample_freq=7,
            seed=self.conf.seed)

        # Evaluate model with cross-validation score
        LOG.info("Evaluate model with cross-validation score")
        all_preds = []
        all_targets = []
        all_indices = []
        best_iters = []
        y = all_seasons["playoff_wins"]
        train = all_seasons.drop(["playoff_wins"], axis=1)
        kf = KFold(n_splits=self.conf.n_folds, shuffle=True, random_state=self.conf.seed)
        for j, (train_idx, valid_idx) in enumerate(kf.split(train, y)):
            start = time.time()

            X_train = train.iloc[train_idx]
            X_valid = train.iloc[valid_idx]
            y_train = y.iloc[X_train.index]
            y_valid = y.iloc[X_valid.index]

            # OOF predictions
            model.fit(
                X_train.drop(["season", "name"], axis=1),
                y_train,
                eval_set=[(X_valid.drop(["season", "name"], axis=1), y_valid)],
                early_stopping_rounds=self.conf.early_stopping,
                verbose=0)
            oof_preds = model.predict(X_valid.drop(["season", "name"], axis=1), ntree_limit=model.best_iteration_)
            best_iter = model.best_iteration_

            all_preds.extend(oof_preds)
            all_targets.extend(y_valid.values)
            all_indices.extend(X_valid.index.values)
            best_iters.append(best_iter)

            # Log fold results
            LOG.info(
                "FOLD " + str(j + 1) + " - " + str(np.round((time.time() - start), 2)) + 
                "sec - best iter : " + str(best_iter) + 
                " - CV : " + str(np.round(scoring(y_valid.values, oof_preds), 5)) + 
                " on " + str(oof_preds.shape[0]) + " rows")

        postpro_df = pd.DataFrame()
        postpro_df["indices"] = all_indices
        postpro_df["targets"] = all_targets
        postpro_df["preds"] = all_preds
        postpro_df.loc[postpro_df["preds"] < 0, "preds"] = 0
        cv_score = scoring(postpro_df["targets"], postpro_df["preds"])
        LOG.info("------------> CV score : " + str(np.round(cv_score, 5)))
        avg_best_iter = int(np.mean(best_iters))
        LOG.info("------------> Average best iter : " + str(avg_best_iter))

        # Train on whole dataset
        LOG.info("Train on whole dataset")
        n_iter = int(avg_best_iter * (1 + (1/self.conf.n_folds)))
        model = LGBMRegressor(
            n_estimators=n_iter,
            learning_rate=0.025,
            max_depth=7,
            num_leaves=127,
            colsample_bytree=0.75,
            min_child_samples=20,
            min_split_gain=0.005,
            subsample=0.60,
            subsample_freq=7,
            seed=self.conf.seed)
        model.fit(
            X_train.drop(["season", "name"], axis=1),
            y_train,
            verbose=0)

        # Save trained model
        LOG.info("Save trained model")
        joblib.dump(model, "models/" + self.conf.model_name)

        # Save predictions plot
        LOG.info("Save predictions plot")
        plt.grid()
        plt.scatter(
            postpro_df["targets"], 
            postpro_df["preds"], 
            s=8, alpha=0.7)
        plt.title("Targets vs Predictions")
        plt.xlabel("Targets")
        plt.ylabel("Predictions")
        plt.plot(
            [0, max(max(postpro_df["targets"]), max(postpro_df["preds"]))], 
            [0, max(max(postpro_df["targets"]), max(postpro_df["preds"]))], 
            c = "red")
        plt.savefig("plots/" + str(self.experiment_id) + "_preds.jpg")

        # Save feature importance plot
        coefs = pd.DataFrame(sorted(zip(
            model.feature_importances_, 
            train.drop(["season", "name"], axis=1).columns)), 
            columns=['Value', 'Feature'])
        plt.figure(figsize=(20, 25))
        sns.barplot(x="Value", y="Feature", data=coefs.sort_values(by="Value", ascending=False))
        plt.tight_layout()
        plt.savefig("plots/" + str(self.experiment_id) + "_feat_importance.jpg")


    def before_regular_predict(self):
        # Load preprocessed data
        LOG.info("Load preprocessed data")
        try:
            all_seasons = pd.read_csv("data/processed/all_seasons.csv")
        except:
            LOG.error("Cannot find preprocessed data")
            sys.exit()

        # Use 2017 regular season data to predict 2018 playoff results
        LOG.info("Use 2017 regular season data to predict 2018 playoff results")
        train = all_seasons.loc[all_seasons["season"] == 2017, :]
        train = train.drop(["playoff_wins"], axis=1)

        # Perform predictions
        self.predict(train, "before_regular")


    def before_playoffs_predict(self):
        # Load preprocessed data
        LOG.info("Load preprocessed data")
        try:
            all_seasons = pd.read_csv("data/processed/all_seasons.csv")
        except:
            LOG.error("Cannot find preprocessed data")
            sys.exit()

        # Use 2018 regular season data to predict 2018 playoff results
        LOG.info("Use 2018 regular season data to predict 2018 playoff results")
        train = all_seasons.loc[all_seasons["season"] == 2018, :]
        train = train.drop(["playoff_wins"], axis=1)

        # Perform predictions
        # In real life, we wouldn't try to compute odds for teams that didn't qualify for playoffs
        self.predict(train, "before_playoffs")


    def predict(self, train, title):
        # Load pre-saved model
        try:
            model = joblib.load("models/" + self.conf.model_name)
        except:
            LOG.error("Cannot find pre-saved model")
            sys.exit()

        # Perform inference
        LOG.info("Perform inference")
        test_preds = model.predict(train.drop(["season", "name"], axis=1), ntree_limit=model.best_iteration_)

        # Compute championship win percentage
        LOG.info("Compute championship win percentage")
        total_predicted_wins = sum(test_preds)
        expected_win_pcts = []
        for i in range(len(test_preds)):
            expected_win_pct = test_preds[i] * 100 / total_predicted_wins
            expected_win_pcts.append(expected_win_pct)
            #LOG.info(test.iloc[i]["name"] + " - expected playoff wins:" + str(np.round(test_preds[i], 2)) + " - expected final win %:" + str(np.round(expected_win_pct, 2)))

        # Translate projected wins into odds
        LOG.info("----- CHAMPIONSHIP BETTING ODDS -----")
        all_odds = []
        for i in range(len(expected_win_pcts)):
            odds = round(100 / expected_win_pcts[i])
            if odds > self.conf.highest_odds:
                odds = self.conf.highest_odds
            all_odds.append(odds)
            LOG.info(train.iloc[i]["name"] + " - " + str(odds))

        # Save odds
        df_odds = pd.DataFrame(columns=["Team", "Odds"])
        df_odds["Team"] = train["name"]
        df_odds["Odds"] = all_odds
        df_odds.sort_values(["Team"], ascending=True, inplace=True)
        df_odds.reset_index(drop=True, inplace=True)
        df_odds.to_csv("odds/" + str(self.experiment_id) + "_" + title + "_odds.csv")
