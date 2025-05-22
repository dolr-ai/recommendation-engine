# %%
import os
import json
import pandas as pd
import asyncio
import numpy as np
import argparse
import sys
from datetime import datetime
import pathlib
from tqdm import tqdm
from utils.common_utils import path_exists, get_logger

logger = get_logger()

# utils
from utils.gcp_utils import GCPUtils

DATA_ROOT = pathlib.Path("./data").absolute()

GCP_CREDENTIALS_PATH = "/home/dataproc/recommendation-engine/credentials.json"
with open(GCP_CREDENTIALS_PATH, "r") as f:
    _ = json.load(f)
    gcp_credentials_str = json.dumps(_)

# Initialize GCP utils
gcp = GCPUtils(gcp_credentials=gcp_credentials_str)
logger.info(f"DATA_ROOT: {DATA_ROOT}")


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


async def fetch_user_interaction_batch(batch_id, start_date, end_date):
    """Fetch user interaction data for a specific date range batch

    Args:
        batch_id: ID of the batch for naming the output file
        start_date: Start date for this batch
        end_date: End date for this batch

    Returns:
        DataFrame containing user interaction data for this batch
    """
    batch_path = (
        DATA_ROOT
        / "user_interaction"
        / f"user_interaction_batch_{batch_id}_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.parquet"
    )

    if not path_exists(batch_path):
        batch_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Pulling user interaction data batch {batch_id} from {start_date} to {end_date}"
        )

        df_batch = gcp.data.pull_user_interaction_data(
            start_date=start_date, end_date=end_date
        )

        df_batch.to_parquet(batch_path)
        logger.info(f"Saved user interaction batch {batch_id} to {batch_path}")
    else:
        logger.info(
            f"Loading existing user interaction batch {batch_id} from {batch_path}"
        )
        df_batch = pd.read_parquet(batch_path)

    return df_batch


async def fetch_user_interaction_data(start_date, end_date, batch_days=7):
    """Fetch and save user interaction data for the given date range in batches

    Args:
        start_date: Start date for the data pull
        end_date: End date for the data pull
        batch_days: Number of days per batch

    Returns:
        DataFrame containing combined user interaction data
    """
    # Create date ranges for each batch
    current_date = start_date
    date_ranges = []

    while current_date < end_date:
        batch_end = min(current_date + pd.Timedelta(days=batch_days - 1), end_date)
        date_ranges.append((current_date, batch_end))
        current_date = batch_end + pd.Timedelta(days=1)

    logger.info(
        f"Split date range into {len(date_ranges)} batches of {batch_days} days each"
    )

    # Fetch each batch in parallel
    tasks = []
    for i, (batch_start, batch_end) in enumerate(date_ranges):
        tasks.append(fetch_user_interaction_batch(i, batch_start, batch_end))

    # Show progress
    batch_results = []
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
        batch_results.append(await f)

    # Combine all batch results
    all_user_data = pd.concat(batch_results, ignore_index=True)

    # todo: when data increases stop concatenating all files in memory
    # Save combined file as well
    combined_path = (
        DATA_ROOT
        / "user_interaction"
        / f"user_interaction_all_{start_date.strftime('%Y-%m-%d')}_{end_date.strftime('%Y-%m-%d')}.parquet"
    )
    all_user_data.to_parquet(combined_path)
    logger.info(f"Saved combined user interaction data to {combined_path}")

    logger.info(f"Unique video ids: {all_user_data['video_id'].nunique()}")
    logger.info(f"Unique user ids: {all_user_data['user_id'].nunique()}")

    return all_user_data


async def fetch_video_index_data(video_ids, batch_size=100):
    """Fetch and save video index data for the given video IDs

    Args:
        video_ids: List of video IDs to fetch data for
        batch_size: Number of videos per batch

    Returns:
        DataFrame containing video index data
    """
    logger.info(
        f"Fetching video index data for {len(video_ids)} videos with batch size {batch_size}"
    )

    output_dir = DATA_ROOT / "video_index"

    # Run the async function with parameters
    video_data = await fetch_all_video_data(
        video_ids=video_ids,
        batch_size=batch_size,
        gcp_utils=gcp,
        output_dir=output_dir,
    )

    logger.info(f"Video index data saved to {output_dir / 'video_index_all.parquet'}")

    return video_data


def parse_args():
    parser = argparse.ArgumentParser(description="Pull user interaction and video data")
    parser.add_argument(
        "--start-date",
        type=str,
        default="2025-02-01",
        help="Start date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        default="2025-02-05",
        help="End date in YYYY-MM-DD format",
    )
    parser.add_argument(
        "--user-data-batch-days",
        type=int,
        default=7,
        help="Number of days per user interaction data batch",
    )
    parser.add_argument(
        "--video-batch-size",
        type=int,
        default=100,
        help="Number of videos per batch for video index data",
    )
    return parser.parse_args()


async def main():
    args = parse_args()
    logger.info(f"Arguments received: {sys.argv}")

    try:
        start_date = datetime.strptime(args.start_date, "%Y-%m-%d")
        end_date = datetime.strptime(args.end_date, "%Y-%m-%d")
    except ValueError:
        logger.error("Invalid date format. Please use YYYY-MM-DD")
        sys.exit(1)

    logger.info(
        f"Processing data from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}"
    )

    # Fetch user interaction data in batches
    user_data = await fetch_user_interaction_data(
        start_date=start_date, end_date=end_date, batch_days=args.user_data_batch_days
    )

    # Extract unique video IDs from user data
    unique_video_ids = user_data["video_id"].unique().tolist()

    # Fetch video index data
    video_data = await fetch_video_index_data(
        video_ids=unique_video_ids, batch_size=args.video_batch_size
    )

    logger.info("Data processing complete")
    logger.info(f"User interactions: {len(user_data)} rows")
    logger.info(f"Video data: {len(video_data)} rows")


if __name__ == "__main__":
    asyncio.run(main())

# %%
# df_video_index
# df_video_index["uri"].value_counts()
# %%
