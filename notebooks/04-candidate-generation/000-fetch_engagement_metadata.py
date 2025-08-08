# %%
import os
import json
import pandas as pd
import asyncio
import random
import concurrent.futures

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
from concurrent.futures import ThreadPoolExecutor

# utils
from utils.gcp_utils import GCPUtils


# %%
def setup_configs(env_path="./.env"):
    print(load_dotenv(env_path))

    DATA_ROOT = os.getenv("DATA_ROOT", "/home/dataproc/recommendation-engine/data_root")
    DATA_ROOT = pathlib.Path(DATA_ROOT)

    print(os.getenv("GCP_CREDENTIALS_PATH_PROD"))
    print(os.getenv("GCP_CREDENTIALS_PATH_STAGE"))

    GCP_CREDENTIALS_PATH_STAGE = os.getenv(
        "GCP_CREDENTIALS_PATH_STAGE",
        "/home/dataproc/recommendation-engine/credentials_stage.json",
    )
    GCP_CREDENTIALS_PATH_PROD = os.getenv(
        "GCP_CREDENTIALS_PATH_PROD",
        "/home/dataproc/recommendation-engine/credentials_prod.json",
    )

    with open(GCP_CREDENTIALS_PATH_STAGE, "r") as f:
        _ = json.load(f)
        gcp_credentials_str_stage = json.dumps(_)

    with open(GCP_CREDENTIALS_PATH_PROD, "r") as f:
        _ = json.load(f)
        gcp_credentials_str_prod = json.dumps(_)

    gcp_utils_stage = GCPUtils(gcp_credentials=gcp_credentials_str_stage)
    gcp_utils_prod = GCPUtils(gcp_credentials=gcp_credentials_str_prod)

    del gcp_credentials_str_stage, gcp_credentials_str_prod, _
    print(f"DATA_ROOT: {DATA_ROOT}")
    return DATA_ROOT, gcp_utils_stage, gcp_utils_prod


DATA_ROOT, gcp_utils_stage, gcp_utils_prod = setup_configs(
    "/Users/sagar/work/yral/recommendation-engine/notebooks/04-candidate-generation/.env"
)

# %%
t = gcp_utils_stage.bigquery.execute_query(
    "SELECT MAX(cluster_id) as max_cluster_id FROM `stage_test_tables.test_user_clusters`",
    to_dataframe=True,
)
max_cluster_id = t.iloc[0]["max_cluster_id"]
print(max_cluster_id)

# %%


def fetch_cluster_data(cluster_id):
    query = f"SELECT * FROM `stage_test_tables.test_user_clusters` WHERE cluster_id = {cluster_id}"
    try:
        df = gcp_utils_stage.bigquery.execute_query(query, to_dataframe=True)
        print(f"Successfully fetched cluster {cluster_id}")
        return df
    except Exception as e:
        print(f"Error fetching cluster {cluster_id}: {e}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error


all_cluster_data = []
num_workers = max_cluster_id + 1
with ThreadPoolExecutor(max_workers=num_workers) as executor:
    # Use a list comprehension to create a list of futures
    futures = [
        executor.submit(fetch_cluster_data, i) for i in range(max_cluster_id + 1)
    ]

    # Iterate over the futures as they complete
    for future in tqdm(
        concurrent.futures.as_completed(futures),
        total=len(futures),
        desc="Fetching Clusters",
    ):
        result = future.result()
        all_cluster_data.append(result)

# Concatenate all the DataFrames into a single DataFrame
df = pd.concat(all_cluster_data, ignore_index=True)
print(f"Shape of the combined DataFrame: {df.shape}")


# %%
(DATA_ROOT / "master_dag_output").mkdir(parents=True, exist_ok=True)
df.to_parquet(DATA_ROOT / "engagement_metadata_6_months.parquet")
# %%
print(DATA_ROOT)
