# %%
import os
import json
import pandas as pd
import asyncio
import random

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from dotenv import load_dotenv
import pathlib
from tqdm import tqdm
from utils.common_utils import path_exists
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score
import plotly.express as px
from kneed import KneeLocator

# utils
from utils.gcp_utils import GCPUtils

# setup configs
print(load_dotenv("/Users/sagar/work/yral/recommendation-engine/.env"))

print(os.getenv("DATA_ROOT"))

DATA_ROOT = pathlib.Path(os.getenv("DATA_ROOT"))
GCP_CREDENTIALS_PATH = os.environ.get("GCP_CREDENTIALS_PATH")

with open(GCP_CREDENTIALS_PATH, "r") as f:
    _ = json.load(f)
    gcp_credentials_str = json.dumps(_)

gcp = GCPUtils(gcp_credentials=gcp_credentials_str)
del gcp_credentials_str, _

# %%
# df_video_index = gcp.data.pull_video_index_data(
#     columns=[
#         "uri",
#         "post_id",
#         "timestamp",
#         "canister_id",
#         # "embedding",
#         "is_nsfw",
#         "nsfw_ec",
#         "nsfw_gore",
#     ]
# )
# df_video_index.head()
# %%
# df_video_index.to_parquet(
#     "/Users/sagar/work/yral/recommendation-engine/data-temp/master_dag_output/master_dag_video_index.parquet"
# )
df_video_index = pd.read_parquet(
    "/Users/sagar/work/yral/recommendation-engine/data-temp/master_dag_output/master_dag_video_index.parquet"
)
# %%

df_video_index["video_id"] = df_video_index["uri"].apply(
    lambda x: x.split("/")[-1].split(".")[0]
)
# video_ids_list = [
#     f"'gs://yral-videos/{video_id}.mp4'" for video_id in video_ids
# ]
# https://yral.com/hot-or-not/76qol-iiaaa-aaaak-qelkq-cai/451
df_video_index["url"] = df_video_index.apply(
    lambda x: f"https://yral.com/hot-or-not/{x['canister_id']}/{x['post_id']}", axis=1
)

video_id_to_url_map = df_video_index.set_index("video_id")["url"].to_dict()
# %%
video_id_to_url_map
# %%
df = pd.read_pickle(
    "/Users/sagar/work/yral/recommendation-engine/data-temp/master_dag_output/master_dag_user_cluster_output_results.pkl"
)

df_op = pd.read_parquet(
    "/Users/sagar/work/yral/recommendation-engine/data-temp/master_dag_output/master_dag_user_cluster_output.parquet"
)
# %%

etype_to_type_of_embedding = {
    "etype1": "<Avg Interaction> Embedding",
    "etype2": "<Avg Interaction, Cluster Distribution> Embedding",
    "etype3": "<Avg Interaction, Cluster Distribution, Temporal> Embedding",
    "etype4": "<User> Embedding",
    # exploration purposes
    "etype5": "<Cluster Distribution> Embedding",
    "etype6": "<Temporal> Embedding",
}

type_of_embedding = "etype4"


df_temp = df[type_of_embedding]["df_clustered"]
df_temp.merge(df_op[["user_id", "engagement_metadata_list"]], on="user_id")
df_temp["videos_watched"] = df_temp["engagement_metadata_list"].apply(
    lambda x: [i["video_id"] for i in x if i is not None]
)
df_cluster_grp = df_temp.groupby("cluster_id", as_index=False).agg(
    {"videos_watched": "sum"}
)
df_cluster_grp["videos_watched"] = df_cluster_grp["videos_watched"].apply(
    lambda x: list(set(x))
)
df_cluster_grp["videos_watched"] = df_cluster_grp["videos_watched"].apply(
    lambda x: [video_id_to_url_map.get(video_id, None) for video_id in x]
)
# %%
df_cluster_grp.columns, df_cluster_grp["cluster_id"].max()
# %%
all_videos_of_cluster = df_cluster_grp[df_cluster_grp["cluster_id"] == 1][
    "videos_watched"
].iloc[0]

print(
    'open -a "Google Chrome" '
    + " ".join(
        [
            i
            for i in random.sample(
                all_videos_of_cluster,
                10,
            )
        ]
    )
)
# %%

# %%
