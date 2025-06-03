# %%
import os
import json
from IPython.display import display
import pandas as pd
import asyncio
import random
import concurrent.futures

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timezone, timedelta
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
from concurrent.futures import ThreadPoolExecutor
import itertools


# utils
from utils.gcp_utils import GCPUtils


# %%
# setup configs
def setup_configs(env_path="./.env"):
    print(load_dotenv(env_path))

    DATA_ROOT = os.getenv("DATA_ROOT", "/home/dataproc/recommendation-engine/data_root")
    DATA_ROOT = pathlib.Path(DATA_ROOT)

    print(os.getenv("GCP_CREDENTIALS_PATH_PROD"))
    print(os.getenv("GCP_CREDENTIALS_PATH"))

    GCP_CREDENTIALS_PATH = os.getenv(
        "GCP_CREDENTIALS_PATH_PROD",
        "/home/dataproc/recommendation-engine/credentials_stage.json",
    )

    with open(GCP_CREDENTIALS_PATH, "r") as f:
        _ = json.load(f)
        gcp_credentials_str = json.dumps(_)

    gcp_utils = GCPUtils(gcp_credentials=gcp_credentials_str)
    del gcp_credentials_str, _
    print(f"DATA_ROOT: {DATA_ROOT}")
    return DATA_ROOT, gcp_utils


DATA_ROOT, gcp_utils = setup_configs(
    "/Users/sagar/work/yral/recommendation-engine/notebooks/04-candidate-generation/.env"
)
# %%

pd.options.mode.chained_assignment = None  # Disable warning


# Global variables for thresholds and cluster filtering
WATCH_PERCENTAGE_THRESHOLD_MIN = 0.5
WATCH_PERCENTAGE_THRESHOLD_SUCCESS = 0.75
USER_VIDEO_COUNT_THRESHOLD = 10  # Minimum videos watched threshold

# %%
# load data


# df_user_interaction = gcp_utils.data.pull_user_interaction_data(
#     start_date=datetime(2025, 3, 1), end_date=datetime(2025, 5, 29)
# )
# df_user_interaction.to_parquet(DATA_ROOT / "user_interaction.parquet")

df_user_interaction = pd.read_parquet(DATA_ROOT / "user_interaction.parquet")
# %%
# load engagement metadata
df_clusters = pd.read_parquet(DATA_ROOT / "engagement_metadata_6_months.parquet")
# %%


# %%
def get_modified_iou_score(
    df_clusters,
    target_cluster=None,
    min_req_users_for_video=2,
    # todo: add sample parameter for future where one of the df_base here will be sampled
    # for now this sample factor is set to 1, meaning all rows will be considered
    sample_factor=1,
):
    dfc = df_clusters.copy(deep=True)
    dfc_min = dfc[
        dfc["mean_percentage_watched"] > WATCH_PERCENTAGE_THRESHOLD_MIN
    ].reset_index(drop=True)

    dfc_success = dfc[
        dfc["mean_percentage_watched"] > WATCH_PERCENTAGE_THRESHOLD_SUCCESS
    ].reset_index(drop=True)

    # [minimum]
    # groupby cluster_id and video_id
    dfc_grp_min = dfc_min.groupby(["cluster_id", "video_id"], as_index=False).agg(
        user_id_list_min=("user_id", list),
        num_unique_users=("user_id", "nunique"),
    )
    # filter dfs on min required users
    dfc_grp_min = dfc_grp_min[
        dfc_grp_min["num_unique_users"] >= min_req_users_for_video
    ]

    # [success]
    # groupby cluster_id and video_id
    dfc_grp_success = dfc_success.groupby(
        ["cluster_id", "video_id"], as_index=False
    ).agg(
        user_id_list_success=("user_id", list),
        num_unique_users=("user_id", "nunique"),
    )
    # filter dfs on min required users
    dfc_grp_success = dfc_grp_success[
        dfc_grp_success["num_unique_users"] >= min_req_users_for_video
    ]

    if target_cluster is None:
        pass
    else:
        dfc_grp_min = dfc_grp_min[dfc_grp_min["cluster_id"] == target_cluster]
        dfc_grp_success = dfc_grp_success[
            dfc_grp_success["cluster_id"] == target_cluster
        ]

    # get base df for pairwise comparison
    df_base = dfc_grp_min[
        [
            "cluster_id",
            "video_id",
            "user_id_list_min",
        ]
    ].merge(
        dfc_grp_success[
            [
                "cluster_id",
                "video_id",
                "user_id_list_success",
            ]
        ],
        on=["cluster_id", "video_id"],
        how="inner",
    )

    df_base["d"] = 1

    # this will avoid large scale pair wise comparison
    df_req = df_base.sample(frac=sample_factor).merge(
        df_base,
        on=["d"],
        suffixes=["_x", "_y"],
    )

    # remove duplicate pairs
    # in this case A->B == B->A
    df_req["pkey"] = df_req["video_id_x"] + "_" + df_req["video_id_y"]
    df_req = df_req.drop_duplicates(subset=["pkey"])
    df_req = df_req.drop(columns=["pkey"]).reset_index(drop=True)

    # remove same video id comparison
    if target_cluster is not None:
        df_req = df_req[df_req["video_id_x"] != df_req["video_id_y"]].reset_index(
            drop=True
        )
    else:
        df_req = df_req[
            (df_req["video_id_x"] != df_req["video_id_y"])
            & (df_req["cluster_id_x"] != df_req["cluster_id_y"])
        ].reset_index(drop=True)

    # total number of users who have watched at least WATCH_PERCENTAGE_THRESHOLD_MIN of video_x and video_y
    df_req["den"] = df_req["user_id_list_min_x"].apply(lambda x: len(set(x))) + df_req[
        "user_id_list_min_y"
    ].apply(lambda x: len(set((x))))

    # total number of users who have watched more than WATCH_PERCENTAGE_THRESHOLD_SUCCESS of video_x and video_y
    df_req["num"] = df_req.apply(
        lambda x: len(
            set(x["user_id_list_success_x"]).intersection(
                set(x["user_id_list_success_y"])
            )
        ),
        axis=1,
    )

    # iou_modified = how many users have successfully completed both videos out of users who have completed bare minimum requirement of watching the video
    # added *2 to make it more interpretable
    # range now is 0-1 without *2 it was 0-0.5
    df_req["iou_modified"] = ((df_req["num"] / df_req["den"]).round(2)) * 2

    # print(df_req[df_req["iou_modified"] > 0]["iou_modified"].describe())
    df_req = df_req[df_req["iou_modified"] > 0].reset_index(drop=True)
    return df_req


res_dict = {}
for i in range(0, df_clusters["cluster_id"].max()):
    print(f"Cluster {i}")

    df_temp = get_modified_iou_score(
        df_clusters,
        target_cluster=i,
        min_req_users_for_video=2,
    )
    # display(df_temp)
    print(df_temp.shape)
    df_cand = df_temp[df_temp["iou_modified"] > df_temp["iou_modified"].quantile(0.95)]
    print(df_cand.shape)
    # print(
    #     df_cand[["den", "num", "iou_modified"]]
    #     .describe(np.arange(0.9, 1, 0.01))
    #     .to_string()
    # )
    # print(df_temp["iou_modified"].describe(np.arange(0.9, 1, 0.01)))
    print(f"number of candidates: {df_cand.shape[0]}")
    print("-" * 100)
    res_dict[i] = {
        "candidates": df_cand,
        "metadata": {
            "cluster_id": i,
            "number_of_candidates": df_cand.shape[0],
            "number_of_users": df_clusters[df_clusters["cluster_id"] == i][
                "user_id"
            ].nunique(),
            "number_of_videos": df_clusters[df_clusters["cluster_id"] == i][
                "video_id"
            ].nunique(),
            "number_of_interactions": df_clusters[df_clusters["cluster_id"] == i].shape[
                0
            ],
        },
    }

# %%
# df_clusters[df_clusters["cluster_id"] == 3]

pd.DataFrame([res_dict[i]["metadata"] for i in res_dict])[
    [
        "cluster_id",
        "number_of_candidates",
        "number_of_interactions",
        "number_of_videos",
        "number_of_users",
    ]
]
# %%
res_dict[1]["candidates"]["iou_modified"].describe(np.arange(0, 1, 0.1))
# %%
res_dict[1]["candidates"]
