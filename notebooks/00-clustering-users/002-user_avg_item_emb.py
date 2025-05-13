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

load_dotenv(".env")
DATA_ROOT = pathlib.Path(os.getenv("DATA_ROOT"))
# %%
df_user_item_emb = pd.read_parquet(DATA_ROOT / "emb_analysis" / "user_item_emb.parquet")
df_user_item_emb.head()

assert df_user_item_emb["embedding"].isna().sum() == 0


# %%
def filter_logic(df):
    dft = df.copy()

    # clean mean_percentage_watched
    dft["mean_percentage_watched"] = (
        dft["mean_percentage_watched"].astype(float).fillna(0)
    )

    # user should have watched at least 25% of the video
    dft = dft[dft["mean_percentage_watched"] > 0.25]

    # sort by last watched timestamp latest to oldest
    dft = dft.sort_values("last_watched_timestamp", ascending=False)

    # get top 100 videos per user
    dft = dft.groupby("user_id").head(100)

    return dft


df_req = filter_logic(df_user_item_emb)
df_req.head()


# %%
def mean_of_arrays(arrs):

    arr_stack = np.vstack(arrs)
    return arr_stack.mean(axis=0).tolist()


# Calculate average embedding per user
df_user_avg_emb = (
    df_req.groupby("user_id")
    .agg(
        embedding=("embedding", mean_of_arrays),
    )
    .reset_index()
)
# %%

df_user_avg_emb.to_parquet(DATA_ROOT / "emb_analysis" / "002-user_avg_item_emb.parquet")
# %%
df_user_avg_emb.columns
