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

# Set output directories for this notebook
VISUALIZATION_DIR = DATA_ROOT / "visualizations" / "temporal_merged_embeddings"
TRANSFORMED_DIR = DATA_ROOT / "transformed" / "temporal_merged_embeddings"

# Create directories if they don't exist
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
TRANSFORMED_DIR.mkdir(parents=True, exist_ok=True)

# %%
df_emb = pd.read_parquet(
    TRANSFORMED_DIR / "user_merged_embeddings_with_clusters.parquet"
)
# %%
df_emb["video_ids"] = df_emb["engagement_metadata_list"].apply(
    lambda x: [i.get("video_id") for i in x if i.get("video_id") is not None]
)

# %%
df_emb["video_ids"].iloc[0]
# %%

# %%
df_cluster_video_ids = (
    df_emb.groupby("cluster").agg(video_ids=("video_ids", list)).reset_index()
)
df_cluster_video_ids
# %%
df_cluster_video_ids["flat_video_ids"] = df_cluster_video_ids["video_ids"].apply(
    lambda x: np.concatenate(x).tolist()
)
# %%

print(
    str(
        random.choices(
            df_cluster_video_ids[df_cluster_video_ids["cluster"] == 1][
                "flat_video_ids"
            ].iloc[0],
            k=100,
        )
    )[1:-1]
)
