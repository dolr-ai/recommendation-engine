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
# setup configs

DATA_ROOT = "/home/dataproc/recommendation-engine/data_root"
GCP_CREDENTIALS_PATH = "/home/dataproc/recommendation-engine/credentials_stage.json"
# %%
# load data

df = pd.read_parquet(f"{DATA_ROOT}/master_dag_output/engagement_metadata.parquet")

# %%
df["cluster_id"].value_counts()

# %%
df[df['cluster_id']==3]

# %%
df["cluster_id"].value_counts().plot(kind="barh")

# %%