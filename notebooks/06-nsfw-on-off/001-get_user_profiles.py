# %%
# uncomment this to enable debug logging
# import os
# os.environ["LOG_LEVEL"] = "DEBUG"

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


from candidate_cache.get_candidates_meta import (
    UserClusterWatchTimeFetcher,
    UserWatchTimeQuantileBinsFetcher,
)

# %%
NUM_USERS_PER_CLUSTER_BIN = 10
HISTORY_LATEST_N_VIDEOS_PER_USER = 100
MIN_VIDEOS_PER_USER_FOR_SAMPLING = 50


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
os.environ["GCP_CREDENTIALS"] = gcp_utils_stage.core.gcp_credentials

# %%
valkey_config = {
    "host": "10.128.15.210",  # Primary endpoint
    "port": 6379,
    "instance_id": "candidate-cache",
    "ssl_enabled": True,
    "socket_timeout": 15,
    "socket_connect_timeout": 15,
    "cluster_enabled": True,
}

valkey_service = ValkeyService(core=gcp_utils_stage.core, **valkey_config)

# Test connection
# connection_success = valkey_service.verify_connection()
# print(f"Valkey connection successful: {connection_success}")
# # %%
# query = """
# select user_id, cluster_id, video_id, mean_percentage_watched, last_watched_timestamp, nsfw_label
# from `jay-dhanwant-experiments.stage_test_tables.test_clean_and_nsfw_split`
# order by user_id,last_watched_timestamp desc
# """
# df_history = gcp_utils_stage.bigquery.execute_query(query, to_dataframe=True)

# df_counts_and_watch_time = (
#     df_history.groupby(["user_id", "cluster_id"], as_index=False)
#     .agg(
#         count_num_videos=("video_id", "count"),
#         mean_percentage_watched=("mean_percentage_watched", "sum"),
#         nsfw_label=("nsfw_label", "unique"),
#     )
#     .reset_index(drop=True)
# )
# # %%
# df_counts_and_watch_time["cluster_id"].value_counts()

# # %%
# df_counts_and_watch_time[
#     [
#         "cluster_id",
#     ]
# ]

# %%
query_videos_with_candidates = [
    "71405feddfc147729944ff248f26e15b",
    "382bf5d2b5bb4d269540a42e124c5cc6",
    "6c1f93ec132543338b742a368ee5531d",
    "ee8b7e1927424faf8ff7f4e7ece88270",
    "d13f02dc9c804c96b3ad78ac91184130",
    "921cba3560d54f6494bbb84903bb39f1",
    "ae3feaf9b2454a3d9ee56f6a604a695c",
    "af33fa1ea3784642bea22cbef2c54ff7",
    "399014f96cc34882ad1b6807c523d803",
    "e8ee009da81042f3b1b325a566a35b0a",
    "302deb0f70fe4d1aabb0591c7de10a04",
    "307a65d65c2f476ea2fc4669f7a8048a",
    "0828ac3dacfc4b1582b79a00fdeb6c0d",
    "a50777e6096b43babb331b645252476e",
    "0921b1b630904b38925a7c38babffb85",
    "7e07231e20fb452ea583785c2caf21bd",
    "8ec0ecb0d24d4d1da8d281377b6e3980",
    "0978d443cc784dc68ae37c2aee7b6251",
    "e339832545394a13bc59a6b96b614265",
    "e912d7d1934248c3b312b928ac33bfba",
    "98b24c25c6564e01a383686d7658567e",
    "662b92a3ab0e4bc88dcfc261aa66800d",
    "6576f312ed954b648a86ef897922bd87",
    "f50a80cffce64f8eb085c13d81902685",
    "ac66e36db29348f18c1e0e3088f3667b",
    "6b069a83d44546398ca0d1e14e3ef6c4",
    "e069e32197614d668e81c5a6401652ae",
    "639b9af881e84069aaba204305305419",
    "0291a0ab42b440088aba0de8c35abe24",
    "1e607a71e5ea4f37a5b96042a42cd5ed",
    "a3257993d0b54585ab95ca7f7e3712eb",
    "a3b5b566532049dd980b8d09ca5d47c6",
    "d5d84a83a9d544d1b1caa0b4fc6d44df",
    "a4d5e1ccc0f14c0a81d38aa9d712277b",
    "c15b68e83bf94ad785a9d82b75b3ac15",
    "5677a787f9c74b26b1910bead360e046",
    "c8f3d5ba161442d996068e813624db13",
    "dd43d490236640ebb528ea936866198d",
    "54c451538a8147e1badd822248882dee",
    "cb25c1300fdf4a7cb74d348c07cacd23",
    "5d007d6151544c65bcce4ba2c315c9e1",
    "750375feb65a40f2b7ef6d6d9d257a2e",
    "968173934a2b4bcc9477f8eff1392bd8",
    "578f47c244b24becae0b9e7df9c4f2be",
    "0d594f5f8a994401b11d88597d376a0f",
    "9d52f506afc949a2b8af88447fb61a7f",
    "56afb2a840444bad909e7345d880f890",
    "53782f4d47b248ab8f1006e197e9f35d",
    "8d53b32e2a294854ad58fc1284b7b91e",
    "310292c19cb04bfeb78f75c0e1220980",
    "fd22817ebcd64639a935dade58b81280",
]
all_search_space = [
    "6c8fccc2af4e4c58a197668a8da2f441",
    "403fa3a1e6dc45bcbf6807d9f03cbbc6",
    "97695a1534f54eecbf2253ee11c6d83d",
    "5546ec7ab2da4c36a45eb8bca40f34b3",
    "454f590873b144c5811c805ef8ff0075",
    "648cfa48458a404fb978f54bd94415c1",
    "2472e3f1cbb742038f0e86a27c8ac98a",
    "114797ccbfd94d8e954d0409fde61e9f",
    "aeb4642b5ba045a4a52c2b53990f197f",
    "b1518e4408044fcebfccaa8e7ae4d194",
    "4d18e95bc34743e88b57e86b23c36eab",
    "51d9a26f209d42c18866a48d1f6b5ad7",
    "680ae2b5f3ee4548ac4d565b2314aab8",
    "9d5bb54e0e5e4dba96d3f80aab61550e",
    "92331f2f87d64c0eb60d946090cd22e0",
    "a77079b66ae3467cac1156297280eb1a",
    "4551777138f94320b2943658fd53db49",
    "ca636d4220814788ae1370ce308adcd4",
    "c3fa3a77e80a4257ba1e0b66a7674077",
    "7978b80b49ac43cfbeafcf50d7dbbac2",
    "ec47acdce9474b4986e475443c946d02",
    "7de00c6d560845dcbe73267e1c2a1c81",
    "9ef90dfccd9d4d5e8298f77e4f36d899",
    "982eb041bfef4584bd67811547cb1de4",
    "2b9d5cd531a34bb38b13acdbecd4151c",
    "788cfa0c493e412595c604fedb6f4517",
    "55cb3be4fed5437e9f5aed471c7149f5",
]

from recommendation.utils.similarity_bq import SimilarityManager

similarity_manager = SimilarityManager(
    gcp_utils=gcp_utils_stage,
)

similarity_manager.calculate_similarity(
    query_videos_with_candidates,
    all_search_space,
)
