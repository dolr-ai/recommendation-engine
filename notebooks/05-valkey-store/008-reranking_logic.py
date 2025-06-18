# %%
import os
import sys
import json
from dotenv import load_dotenv
import numpy as np
import pandas as pd
from tqdm import tqdm

from redis.commands.search.query import Query

from utils.gcp_utils import GCPUtils
from utils.valkey_utils import ValkeyVectorService, ValkeyService


# %%
# Setup configs
def setup_configs(env_path="./.env", if_enable_prod=False, if_enable_stage=True):
    print(load_dotenv(env_path))

    GCP_CREDENTIALS_PATH_STAGE = os.getenv(
        "GCP_CREDENTIALS_PATH_STAGE",
        "/home/dataproc/recommendation-engine/credentials_stage.json",
    )
    with open(GCP_CREDENTIALS_PATH_STAGE, "r") as f:
        _ = json.load(f)
        gcp_credentials_str_stage = json.dumps(_)
    gcp_utils_stage = GCPUtils(gcp_credentials=gcp_credentials_str_stage)
    del gcp_credentials_str_stage

    return gcp_utils_stage


gcp_utils_stage = setup_configs(
    "/root/recommendation-engine/notebooks/05-valkey-store/.env",
    if_enable_prod=False,
    if_enable_stage=True,
)

# %%
valkey_config = {
    "host": "10.128.15.206",  # Primary endpoint
    "port": 6379,
    "instance_id": "candidate-valkey-instance",
    "ssl_enabled": True,
    "socket_timeout": 15,
    "socket_connect_timeout": 15,
}

valkey_service = ValkeyService(core=gcp_utils_stage.core, **valkey_config)

# Test connection
connection_success = valkey_service.verify_connection()
print(f"Valkey connection successful: {connection_success}")
# %%
# Initialize the vector service using the improved implementation
vector_service = ValkeyVectorService(
    core=gcp_utils_stage.core,
    host="10.128.15.206",
    port=6379,
    instance_id="candidate-valkey-instance",
    ssl_enabled=True,
    socket_timeout=15,
    socket_connect_timeout=15,
    vector_dim=1408,  # Use the actual embedding dimension from data
    prefix="video_id:",  # Use custom prefix for this application
)
connection_success = vector_service.verify_connection()
print(f"Vector connection successful: {connection_success}")


# %%
query = """
select user_id, cluster_id, count(video_id) count_num_videos from `jay-dhanwant-experiments.stage_test_tables.test_user_clusters`
group by user_id, cluster_id
order by count_num_videos desc
limit 1000
"""
df_temp = gcp_utils_stage.bigquery.execute_query(query, to_dataframe=True)


# %%
df_temp.tail(1).to_dict(orient="records")
q2 = """
select user_id, cluster_id, video_id, last_watched_timestamp, mean_percentage_watched from `jay-dhanwant-experiments.stage_test_tables.test_user_clusters`
where user_id = "22e3h-u7fki-4eb6r-dqgy6-hfkam-27qfu-pgxww-l76hq-5uuk2-evhq7-pae"
"""
temp = gcp_utils_stage.bigquery.execute_query(q2, to_dataframe=True)
# %%
temp
