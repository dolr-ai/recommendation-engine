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
from utils.common_utils import path_exists

# utils
from utils.gcp_utils import GCPUtils

# setup configs
print(load_dotenv(".env"))
print(os.getenv("GCP_CREDENTIALS_PATH"))
print(os.getenv("DATA_ROOT"))

with open(os.getenv("GCP_CREDENTIALS_PATH"), "r") as f:
    _ = json.load(f)
    gcp_credentials_str = json.dumps(_)

gcp = GCPUtils(gcp_credentials=gcp_credentials_str)
del gcp_credentials_str, _

DATA_ROOT = pathlib.Path(os.getenv("DATA_ROOT"))
# %%
"""
dir(gcp.bigquery)


pd.DataFrame(
    gcp.bigquery.list_tables(
        project_id="hot-or-not-feed-intelligence", dataset_id="yral_ds"
    )
)
gcp.bigquery.execute_query(
    "SELECT * FROM `hot-or-not-feed-intelligence.yral_ds.video_index` LIMIT 10"
)

schemas = gcp.bigquery.get_table_schemas(
    project_id="hot-or-not-feed-intelligence", dataset_id="yral_ds"
)
"""
0

# %%
start_date = datetime(2025, 4, 1)
end_date = datetime(2025, 4, 30)
user_interaction_path = (
    DATA_ROOT
    / "user_interaction"
    / f"user_interaction_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.parquet"
)
if not path_exists(user_interaction_path):
    user_interaction_path.parent.mkdir(parents=True, exist_ok=True)
    df_user_interaction = gcp.data.pull_user_interaction_data(
        start_date=start_date, end_date=end_date
    )
    df_user_interaction.to_parquet(user_interaction_path)
else:
    df_user_interaction = pd.read_parquet(user_interaction_path)

# %%
df_user_interaction["video_id"].nunique()
# %%


async def fetch_video_batch(batch_id, video_ids, gcp_utils, output_dir):
    """Fetch video index data for a batch of video IDs

    Args:
        batch_id: ID of the batch for naming the output file
        video_ids: List of video IDs to fetch
        gcp_utils: GCPUtils instance for data fetching
        output_dir: Directory to save the output
    """
    loop = asyncio.get_event_loop()
    df = await loop.run_in_executor(
        None, lambda: gcp_utils.data.pull_video_index_data(video_ids=video_ids)
    )
    output_path = output_dir / f"video_index_batch_{batch_id}.parquet"
    df.to_parquet(output_path)
    return df


async def fetch_all_video_data(video_ids, batch_size, gcp_utils, output_dir):
    """Fetch all video data in parallel batches

    Args:
        video_ids: List of all video IDs to fetch
        batch_size: Size of each batch
        gcp_utils: GCPUtils instance for data fetching
        output_dir: Directory to save the outputs
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create batches of the specified size
    batches = [
        video_ids[i : i + batch_size] for i in range(0, len(video_ids), batch_size)
    ]

    tasks = []
    for i, batch in enumerate(batches):
        tasks.append(fetch_video_batch(i, batch, gcp_utils, output_dir))

    # Show progress
    results = []
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        results.append(await f)

    # Combine all results
    all_video_data = pd.concat(results, ignore_index=True)
    # Save combined result
    all_video_data.to_parquet(output_dir / "video_index_all.parquet")
    return all_video_data


# Get unique video IDs
unique_video_ids = df_user_interaction["video_id"].unique().tolist()
print(f"running fetch video index on {len(unique_video_ids)} videos")
# # Run the async function with parameters
df_video_index = asyncio.run(
    fetch_all_video_data(
        video_ids=unique_video_ids,
        batch_size=100,
        gcp_utils=gcp,
        output_dir=DATA_ROOT / "video_index",
    )
)
# %%
# df_video_index
# df_video_index["uri"].value_counts()
# %%
