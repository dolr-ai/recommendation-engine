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

df_user_interaction = pd.read_parquet(
    DATA_ROOT / "user_interaction" / "user_interaction_2025-02-01_2025-04-30.parquet"
)
# %%
df_user_interaction
# %%
df_video_index = pd.read_parquet(DATA_ROOT / "video_index" / "video_index_all.parquet")
df_video_index = df_video_index.dropna(subset=["embedding"])

# %%
assert df_video_index["embedding"].isna().sum() == 0
# %%


# %%
def mean_of_arrays(arrs):
    arr_stack = np.vstack([np.array(arr) for arr in arrs if arr is not None])
    return arr_stack.mean(axis=0).tolist()


df_video_index_agg = (
    df_video_index.groupby("uri")
    .agg(
        embedding=("embedding", mean_of_arrays),
        emb_list=("embedding", list),
        is_nsfw=("is_nsfw", list),
        nsfw_ec=("nsfw_ec", list),
        post_id=("post_id", "first"),
        nsfw_gore=("nsfw_gore", list),
        timestamp=("timestamp", "first"),
        canister_id=("canister_id", "first"),
    )
    .reset_index()
)
# %%
df_video_index_agg.head()
df_video_index_agg["video_id"] = (
    df_video_index_agg["uri"].str.split("/").str[-1].str.split(".mp4").str[0]
)

# %%
df_req_usr_item_emb = df_user_interaction[
    ["user_id", "video_id", "mean_percentage_watched", "last_watched_timestamp"]
].merge(
    df_video_index_agg[["video_id", "embedding", "emb_list"]], on="video_id", how="left"
)

df_req_usr_item_emb = df_req_usr_item_emb.dropna(subset=["emb_list"])
df_req_usr_item_emb = df_req_usr_item_emb.drop(columns=["emb_list"])
# %%
save_path = DATA_ROOT / "emb_analysis" / "user_item_emb.parquet"
save_path.parent.mkdir(parents=True, exist_ok=True)
df_req_usr_item_emb.to_parquet(save_path)
# %%

print("number of unique users", df_req_usr_item_emb["user_id"].nunique())
print("number of unique videos", df_req_usr_item_emb["video_id"].nunique())
# %%

save_path = DATA_ROOT / "emb_analysis" / "user_item_emb.parquet"
df_user_item_emb = pd.read_parquet(save_path)
df_user_item_emb.head()
# %%
df_user_interaction["video_id"].value_counts()
