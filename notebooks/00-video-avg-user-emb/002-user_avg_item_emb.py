# %%
import os
import json
import pandas as pd
import asyncio
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import pathlib
from tqdm import tqdm
from sklearn.preprocessing import normalize

load_dotenv(".env")
DATA_ROOT = pathlib.Path(os.getenv("DATA_ROOT"))
# %%
df_user_item_emb = pd.read_parquet(DATA_ROOT / "emb_analysis" / "user_item_emb.parquet")
df_user_item_emb.head()

assert df_user_item_emb["embedding"].isna().sum() == 0

WATCHED_MIN_VIDEOS = 2
WATCHED_MAX_VIDEOS = 100

run_config = {
    "exp1": True,
    "exp2": True,
    "exp3": True,
}


# %%
class Experiment1:
    def __init__(self, tag="002-user_avg_item_emb-exp1"):
        self.df_user_item_emb = pd.read_parquet(
            DATA_ROOT / "emb_analysis" / "user_item_emb.parquet"
        )

        self.tag = tag

        # sort by last watched timestamp latest to oldest
        self.df_user_item_emb = self.df_user_item_emb.sort_values(
            "last_watched_timestamp", ascending=False
        )

        # get top 100 videos per user
        self.df_user_item_emb = (
            self.df_user_item_emb.groupby("user_id")
            .filter(lambda x: len(x) >= WATCHED_MIN_VIDEOS)
            .groupby("user_id")
            .head(WATCHED_MAX_VIDEOS)
        )

    def filter_logic(self):
        dft = self.df_user_item_emb.copy()

        # clean mean_percentage_watched
        dft["mean_percentage_watched"] = (
            dft["mean_percentage_watched"].astype(float).fillna(0)
        )

        dft["mean_percentage_watched"] = dft["mean_percentage_watched"] * 100

        # user should have watched at least 25% of the video
        dft = dft[dft["mean_percentage_watched"] > 25]

        return dft

    def mean_of_arrays(self, arrs):
        arr_stack = np.vstack(arrs)
        return arr_stack.mean(axis=0).tolist()

    def normalize_embeddings(self, df):
        """L2 normalize embeddings"""
        embeddings = np.vstack(df["embedding"])
        normalized = normalize(embeddings, norm="l2")
        df["embedding"] = [emb.tolist() for emb in normalized]
        return df

    def calculate_user_avg_item_emb(self):
        df_req = self.filter_logic()

        # Calculate average embedding per user
        df_user_avg_emb = (
            df_req.groupby("user_id")
            .agg(
                embedding=("embedding", self.mean_of_arrays),
            )
            .reset_index()
        )
        # L2 normalize the embeddings
        df_user_avg_emb = self.normalize_embeddings(df_user_avg_emb)
        return df_user_avg_emb

    def calc_emb_and_save_to_parquet(self):
        df_user_avg_emb = self.calculate_user_avg_item_emb()
        df_user_avg_emb.to_parquet(DATA_ROOT / "emb_analysis" / f"{self.tag}.parquet")
        return df_user_avg_emb


# %%
if run_config["exp1"]:
    exp1 = Experiment1()
    df_ue_exp1 = exp1.calc_emb_and_save_to_parquet()


# %%


class Experiment2:
    def __init__(self, tag="002-user_avg_item_emb-exp2"):
        self.df_user_item_emb = pd.read_parquet(
            DATA_ROOT / "emb_analysis" / "user_item_emb.parquet"
        )
        self.tag = tag

        # sort by last watched timestamp latest to oldest
        self.df_user_item_emb = self.df_user_item_emb.sort_values(
            "last_watched_timestamp", ascending=False
        )

        # get top 100 videos per user
        self.df_user_item_emb = (
            self.df_user_item_emb.groupby("user_id")
            .filter(lambda x: len(x) >= WATCHED_MIN_VIDEOS)
            .groupby("user_id")
            .head(WATCHED_MAX_VIDEOS)
        )

    def filter_logic(self):
        dft = self.df_user_item_emb.copy()
        dft["embedding"] = dft["embedding"] * dft["mean_percentage_watched"]

        # clean mean_percentage_watched
        dft["mean_percentage_watched"] = (
            dft["mean_percentage_watched"].astype(float).fillna(0)
        )
        dft["mean_percentage_watched"] = dft["mean_percentage_watched"] * 100

        # user should have watched at least 25% of the video
        dft = dft[dft["mean_percentage_watched"] > 25]

        return dft

    def mean_of_arrays(self, arrs):
        arr_stack = np.vstack(arrs)
        return arr_stack.mean(axis=0).tolist()

    def normalize_embeddings(self, df):
        """L2 normalize embeddings"""
        embeddings = np.vstack(df["embedding"])
        normalized = normalize(embeddings, norm="l2")
        df["embedding"] = [emb.tolist() for emb in normalized]
        return df

    def calculate_user_avg_item_emb(self):
        df_req = self.filter_logic()

        # Calculate average embedding per user
        df_user_avg_emb = (
            df_req.groupby("user_id")
            .agg(
                embedding=("embedding", self.mean_of_arrays),
            )
            .reset_index()
        )
        # L2 normalize the embeddings
        df_user_avg_emb = self.normalize_embeddings(df_user_avg_emb)
        return df_user_avg_emb

    def calc_emb_and_save_to_parquet(self):
        df_user_avg_emb = self.calculate_user_avg_item_emb()
        df_user_avg_emb.to_parquet(DATA_ROOT / "emb_analysis" / f"{self.tag}.parquet")
        return df_user_avg_emb


# %%
if run_config["exp2"]:
    exp2 = Experiment2()
    df_ue_exp2 = exp2.calc_emb_and_save_to_parquet()


# %%


class Experiment3:
    def __init__(self, tag="002-user_avg_item_emb-exp3"):
        self.df_user_item_emb = pd.read_parquet(
            DATA_ROOT / "emb_analysis" / "user_item_emb.parquet"
        )
        self.tag = tag
        self.dft = None

        # sort by last watched timestamp latest to oldest
        self.df_user_item_emb = self.df_user_item_emb.sort_values(
            "last_watched_timestamp", ascending=False
        )

        # get top 100 videos per user
        self.df_user_item_emb = (
            self.df_user_item_emb.groupby("user_id")
            .filter(lambda x: len(x) >= WATCHED_MIN_VIDEOS)
            .groupby("user_id")
            .head(WATCHED_MAX_VIDEOS)
        )

    def calculate_position_weight(self, positions, decay_factor=10):
        """Calculate weight based on position in user's watch history

        Args:
            positions: Array of positions (0 = most recent, higher = older)
            decay_factor: Controls how quickly weight decays with position

        Returns:
            Array of weights
        """
        return np.exp(-positions / decay_factor)

    def filter_logic(self):
        # Clean mean_percentage_watched without creating a copy
        mean_watched = (
            self.df_user_item_emb["mean_percentage_watched"]
            .astype(np.float32)
            .fillna(0)
            .clip(0, 1)
        )

        mean_watched = mean_watched * 100

        # user should have watched at least 25% of the video
        mask = mean_watched > 25

        # Apply filters first to reduce memory usage in subsequent operations
        filtered_df = self.df_user_item_emb[mask].copy()
        filtered_mean_watched = mean_watched[mask]

        # Create a new dataframe to store results
        result_dfs = []

        # Process each user group
        for user_id, user_group in filtered_df.groupby("user_id"):
            # Sort by timestamp descending (most recent first)
            user_group = user_group.sort_values(
                "last_watched_timestamp", ascending=False
            )

            # Calculate positions (0 = most recent)
            positions = np.arange(len(user_group))

            # Calculate weights based on position
            position_weight = self.calculate_position_weight(positions)

            # Add weights to user group
            user_group = user_group.copy()
            user_group["position_weight"] = position_weight
            user_group["mean_watched"] = filtered_mean_watched[user_group.index]
            user_group["combined_weight"] = np.sqrt(
                user_group["position_weight"] * user_group["mean_watched"]
            )

            # Apply weights to embeddings
            user_group["embedding"] = [
                (np.array(emb) * w).tolist()
                for emb, w in zip(
                    user_group["embedding"], user_group["combined_weight"]
                )
            ]

            # Add to results
            result_dfs.append(user_group)

        # Combine all user groups
        filtered_df = pd.concat(result_dfs)

        self.dft = filtered_df
        return filtered_df

    def mean_of_arrays(self, arrs):
        arr_stack = np.vstack(arrs)
        return arr_stack.mean(axis=0).tolist()

    def normalize_embeddings(self, df):
        """L2 normalize embeddings"""
        embeddings = np.vstack(df["embedding"])
        normalized = normalize(embeddings, norm="l2")
        df["embedding"] = [emb.tolist() for emb in normalized]
        return df

    def calculate_user_avg_item_emb(self):
        df_req = self.filter_logic()

        # Calculate average embedding per user
        df_user_avg_emb = (
            df_req.groupby("user_id")
            .agg(
                embedding=("embedding", self.mean_of_arrays),
            )
            .reset_index()
        )
        # L2 normalize the embeddings
        df_user_avg_emb = self.normalize_embeddings(df_user_avg_emb)
        return df_user_avg_emb

    def calc_emb_and_save_to_parquet(self):
        df_user_avg_emb = self.calculate_user_avg_item_emb()
        df_user_avg_emb.to_parquet(DATA_ROOT / "emb_analysis" / f"{self.tag}.parquet")
        return df_user_avg_emb


# Run Experiment 3 if configured
if run_config["exp3"]:
    exp3 = Experiment3()
    df_ue_exp3 = exp3.calc_emb_and_save_to_parquet()

# %%
print(exp3.dft["user_id"].value_counts())

df_test = exp3.dft.copy()
usr = "ltx3n-cck73-wgrjb-oz2kp-75rbp-4y6r3-rz7oq-axqnb-gbqml-qjboh-aae"

df_test[df_test["user_id"] == usr].sort_values(
    "last_watched_timestamp", ascending=False
)["position_weight"].reset_index(drop=True).plot()
# %%
df_test[df_test["user_id"] == usr]["last_watched_timestamp"].dt.date.unique()
# %%
