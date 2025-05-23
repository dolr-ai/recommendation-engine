"""
Input:
- user_interaction_all.parquet
- video_clusters.parquet (output from get_video_clusters.py)

Output:
- user_video_clusters_distribution.parquet (user-level cluster distributions)

This script calculates the distribution of video clusters that each user has watched
and generates a distribution vector for each user.
"""

import os
import sys
import pathlib
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.sql.window import Window
import numpy as np

# Initialize Spark Session
spark = SparkSession.builder.appName("User Video Clusters Distribution").getOrCreate()

# Define constants
WATCHED_MIN_VIDEOS = 2
WATCHED_MAX_VIDEOS = 100
WATCHED_MIN_PERCENTAGE = 25
DATA_ROOT = "/home/dataproc/recommendation-engine/data_root"


def extract_video_id_from_uri(uri):
    """Extract video_id from URI field"""
    if uri is None:
        return None
    try:
        # Extract the part after the last '/' and before '.mp4'
        parts = uri.split("/")
        last_part = parts[-1]
        video_id = last_part.split(".mp4")[0]
        return video_id
    except Exception as e:
        print(f"Error extracting video_id from URI {uri}: {e}")
        return None


def create_cluster_distribution_array(cluster_counts, num_clusters):
    """
    Convert cluster counts dictionary to normalized distribution array

    Args:
        cluster_counts: Dictionary of {cluster_id: count}
        num_clusters: Total number of clusters (for array size)

    Returns:
        Normalized distribution array
    """
    total_videos = sum(cluster_counts.values())
    if total_videos == 0:
        return [0.0] * num_clusters

    # Create distribution array with counts normalized to sum to 1
    distribution = [0.0] * num_clusters
    for cluster_id, count in cluster_counts.items():
        if 0 <= cluster_id < num_clusters:  # Ensure valid cluster ID
            distribution[cluster_id] = count / total_videos

    return distribution


def calculate_user_cluster_distributions():
    """
    Calculates user cluster distributions based on video engagement.

    1. Load user interaction data and video clusters
    2. Join them to get cluster assignments for watched videos
    3. Filter based on watch criteria
    4. Calculate cluster distribution for each user
    5. Save results
    """
    print("STEP 1: Loading input data")

    # Load input data
    user_interaction_path = "/tmp/user_interaction/user_interaction_all.parquet"
    video_clusters_path = "/tmp/transformed/video_clusters/video_clusters.parquet"

    df_user_interaction = spark.read.parquet(user_interaction_path)
    df_video_clusters = spark.read.parquet(video_clusters_path)

    # Print schema and counts for debugging
    print("User interaction count:", df_user_interaction.count())
    df_user_interaction.printSchema()

    print("Video clusters count:", df_video_clusters.count())
    df_video_clusters.printSchema()

    # Get max cluster ID to determine array size
    max_cluster_id = df_video_clusters.agg(F.max("cluster")).collect()[0][0]
    num_clusters = max_cluster_id + 1
    print(f"Number of clusters: {num_clusters}")

    print("\nSTEP 2: Joining user interactions with video clusters")

    # Join user interactions with video clusters
    df_user_video_clusters = df_user_interaction.join(
        df_video_clusters, on="video_id", how="inner"
    )

    # Convert timestamp to sortable format
    df_user_video_clusters = df_user_video_clusters.withColumn(
        "last_watched_timestamp",
        F.to_timestamp(df_user_video_clusters["last_watched_timestamp"]),
    )

    # Sort by timestamp (newest first)
    df_user_video_clusters = df_user_video_clusters.sort(
        F.desc("last_watched_timestamp")
    )

    print("\nSTEP 3: Filtering users and videos based on criteria")

    # Filter for users with minimum video watches
    df_user_counts = (
        df_user_video_clusters.groupBy("user_id")
        .count()
        .filter(F.col("count") >= WATCHED_MIN_VIDEOS)
    )

    print(
        f"Users who have watched atleast {WATCHED_MIN_VIDEOS} videos: {df_user_counts.count()}"
    )

    # Join to keep only users with enough videos
    df_filtered = df_user_video_clusters.join(
        df_user_counts.select("user_id"), on="user_id", how="inner"
    )

    # Window function to get top N recent videos per user
    window_spec = Window.partitionBy("user_id").orderBy(
        F.desc("last_watched_timestamp")
    )
    df_filtered = df_filtered.withColumn("row_number", F.row_number().over(window_spec))
    df_filtered = df_filtered.filter(F.col("row_number") <= WATCHED_MAX_VIDEOS)

    print("After recency filter count:", df_filtered.count())

    # Clean percentage watched and filter
    df_filtered = df_filtered.withColumn(
        "mean_percentage_watched",
        F.when(F.col("mean_percentage_watched").isNull(), 0).otherwise(
            F.col("mean_percentage_watched")
        ),
    )

    df_filtered = df_filtered.withColumn(
        "mean_percentage_watched_pct", F.col("mean_percentage_watched") * 100
    )

    # Filter for videos watched at least 25%
    df_filtered = df_filtered.filter(
        F.col("mean_percentage_watched_pct") > WATCHED_MIN_PERCENTAGE
    )

    print("After percentage watched filter count:", df_filtered.count())

    print("\nSTEP 4: Calculating cluster distributions per user")

    # Include all interaction fields in metadata
    df_filtered = df_filtered.withColumn(
        "engagement_metadata",
        F.struct(
            F.col("video_id"),
            F.col("last_watched_timestamp"),
            F.col("mean_percentage_watched"),
            F.col("liked"),
            F.col("last_liked_timestamp"),
            F.col("shared"),
            F.col("last_shared_timestamp"),
            F.col("cluster").cast("int").alias("cluster_label"),
        ),
    )

    # Group by user to create list of engagement metadata
    df_user_metadata = df_filtered.groupBy("user_id").agg(
        F.collect_list("engagement_metadata").alias("engagement_metadata_list")
    )

    # Calculate cluster counts per user using PySpark aggregation
    df_user_cluster_counts = df_filtered.groupBy("user_id", "cluster").count()

    # Pivot to create cluster count columns
    cluster_ids = list(range(num_clusters))
    df_user_cluster_pivot = (
        df_user_cluster_counts.groupBy("user_id")
        .pivot("cluster", cluster_ids)
        .sum("count")
        .fillna(0)
    )

    # Create UDF to normalize distributions
    def normalize_distribution(*counts):
        # Convert counts to list and filter out None values
        counts_list = [c if c is not None else 0 for c in counts]
        total = sum(counts_list)
        if total == 0:
            return [0.0] * len(counts_list)
        return [float(c) / total for c in counts_list]

    # Register UDF with array return type
    normalize_distribution_udf = F.udf(normalize_distribution, ArrayType(FloatType()))

    # Create argument list for all cluster columns
    cluster_columns = [f"`{i}`" for i in cluster_ids]

    # Apply normalization to create distribution array
    df_user_distributions = df_user_cluster_pivot.withColumn(
        "cluster_distribution",
        normalize_distribution_udf(*[F.col(col) for col in cluster_columns]),
    )

    # Join with the user metadata to include both distributions and full metadata
    df_result = df_user_distributions.join(
        df_user_metadata, on="user_id", how="inner"
    ).select("user_id", "cluster_distribution", "engagement_metadata_list")

    print("Final user distributions count:", df_result.count())
    df_result.printSchema()

    return df_result


def main():
    """Main execution function"""
    # First, copy files to HDFS if needed
    import subprocess

    # Create local output directory if it doesn't exist
    local_output_dir = f"{DATA_ROOT}/emb_analysis"
    subprocess.call(["mkdir", "-p", local_output_dir])

    # Create hdfs directories, overwriting if they exist
    subprocess.call(["hdfs", "dfs", "-mkdir", "-p", "/tmp/user_interaction"])
    subprocess.call(["hdfs", "dfs", "-mkdir", "-p", "/tmp/emb_analysis"])

    # Copy input files to HDFS
    subprocess.call(
        [
            "hdfs",
            "dfs",
            "-put",
            "-f",
            f"{DATA_ROOT}/user_interaction/user_interaction_all.parquet",
            "/tmp/user_interaction/user_interaction_all.parquet",
        ]
    )

    subprocess.call(
        [
            "hdfs",
            "dfs",
            "-put",
            "-f",
            f"{DATA_ROOT}/transformed/video_clusters/video_clusters.parquet",
            "/tmp/transformed/video_clusters/video_clusters.parquet",
        ]
    )

    # Calculate user cluster distributions
    df_result = calculate_user_cluster_distributions()

    # Write results to HDFS
    output_path = "/tmp/emb_analysis/user_video_clusters_distribution.parquet"
    df_result.write.mode("overwrite").parquet(output_path)

    # Copy results back to local filesystem
    subprocess.call(["mkdir", "-p", local_output_dir])
    subprocess.call(["hdfs", "dfs", "-get", "-f", output_path, local_output_dir])

    # Show a sample of data that was written
    print("\nSample of data written:")
    df_result.show(5)
    df_result.printSchema()

    print(f"Successfully wrote user video clusters distribution to {output_path}")
    print(
        f"And copied back to {local_output_dir}/user_video_clusters_distribution.parquet"
    )


if __name__ == "__main__":
    main()
