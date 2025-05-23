"""
Input:
- user_video_clusters_distribution.parquet (output from get_user_video_cluster_distribution.py)

Output:
- user_temporal_cluster_embeddings.parquet (user-level temporal cluster distribution embeddings)

This script processes user interaction data to create temporal embeddings based on
when users interact with different video clusters over time.
"""

import os
import sys
import pathlib
import subprocess

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
import numpy as np
from datetime import datetime, timezone
import math

# Initialize Spark Session
spark = SparkSession.builder.appName("Temporal Interaction Embeddings").getOrCreate()

# Define constants
DATA_ROOT = "/home/dataproc/recommendation-engine/data_root"
NUM_TIME_BINS = 16
ROPE_DIM = 16


def rope_inspired_encoding(time_counts, dim=ROPE_DIM):
    """
    Create a position-aware temporal encoding inspired by RoPE (Rotary Position Embedding)

    Args:
        time_counts: Dictionary with cluster ids as keys and lists of time bin counts as values,
                   or a list of count values for each time bin
        dim: Dimension of the resulting embedding vector

    Returns:
        Array containing the temporal encoding
    """
    # Check if input is a dictionary or list
    if isinstance(time_counts, dict):
        # Get all values (time bin arrays) and flatten them
        all_bins = []
        for cluster_id, bins in time_counts.items():
            all_bins.extend(bins)
        time_counts = all_bins

    # Now process the flattened time counts
    n_time_periods = len(time_counts)
    features = np.zeros(dim)

    for t, count in enumerate(time_counts):
        if count == 0:
            continue

        # Create position encoding similar to RoPE
        position = t / n_time_periods  # Normalize position to [0,1]
        for i in range(dim // 2):
            theta = position / (10000 ** (2 * i / dim))
            features[2 * i] += count * np.sin(theta)
            features[2 * i + 1] += count * np.cos(theta)

    # Normalize
    magnitude = np.linalg.norm(features)
    if magnitude > 0:
        features = features / magnitude

    return features.tolist()


def create_temporal_cluster_distribution(
    engagement_metadata, min_timestamp, max_timestamp, num_bins=NUM_TIME_BINS
):
    """
    Create a temporal distribution of cluster accesses across equal time bins.

    Args:
        engagement_metadata: List of dictionaries containing cluster_label and last_watched_timestamp
        min_timestamp: The minimum timestamp in the dataset
        max_timestamp: The maximum timestamp in the dataset
        num_bins: Number of equal time bins to create

    Returns:
        A dictionary with cluster labels as keys and lists of counts per time bin as values
    """
    # Calculate time range and bin width in days
    total_time_range = (max_timestamp - min_timestamp).total_seconds() / (24 * 3600)
    bin_width_days = total_time_range / num_bins

    # Initialize the cluster distribution dictionary
    cluster_distribution = {}

    # Process each engagement record
    for record in engagement_metadata:
        cluster_label = record["cluster_label"]
        if cluster_label is None:
            continue

        timestamp = record["last_watched_timestamp"]

        # Calculate which bin this timestamp falls into (using days)
        days_from_start = (timestamp - min_timestamp).total_seconds() / (24 * 3600)
        bin_index = min(int(days_from_start / bin_width_days), num_bins - 1)

        # Initialize this cluster in the distribution dict if it doesn't exist
        if cluster_label not in cluster_distribution:
            cluster_distribution[cluster_label] = [0] * num_bins

        # Increment the count for this cluster in the appropriate time bin
        cluster_distribution[cluster_label][bin_index] += 1

    return cluster_distribution


def get_timestamp_boundaries():
    """
    Get the minimum and maximum timestamps from the user interactions data.

    Returns:
        tuple: (min_timestamp, max_timestamp)
    """
    try:
        # Try to load the user interaction data
        user_interaction_path = "/tmp/user_interaction/user_interaction_all.parquet"
        user_df = spark.read.parquet(user_interaction_path)

        # Get min and max timestamps
        timestamp_stats = user_df.select(
            F.min("last_watched_timestamp").alias("min_timestamp"),
            F.max("last_watched_timestamp").alias("max_timestamp"),
        ).collect()[0]

        min_date = timestamp_stats["min_timestamp"]
        max_date = timestamp_stats["max_timestamp"]

        print(
            f"Derived time boundaries from user interactions: {min_date} to {max_date}"
        )
        return min_date, max_date

    except Exception as e:
        print(f"Error getting timestamp boundaries from user interactions: {e}")
        raise e


def generate_temporal_embeddings(input_path, output_path):
    """
    Generate temporal embeddings from user video cluster distributions.

    1. Load user_video_clusters_distribution.parquet
    2. Process engagement metadata to extract temporal patterns
    3. Generate temporal embeddings using RoPE-inspired encoding
    4. Save the resulting embeddings
    """
    print("STEP 1: Loading input data")

    # Load input data
    df_user_dist = spark.read.parquet(input_path)

    # Print schema and record count
    print("User video clusters distribution count:", df_user_dist.count())
    df_user_dist.printSchema()

    print("\nSTEP 2: Creating temporal cluster distribution UDFs")

    # Get timestamp boundaries from user interactions
    min_timestamp, max_timestamp = get_timestamp_boundaries()
    print(f"Using time range: {min_timestamp} to {max_timestamp}")

    # Register UDFs for temporal processing
    temporal_dist_schema = MapType(IntegerType(), ArrayType(IntegerType()))

    # Create UDFs
    @F.udf(temporal_dist_schema)
    def create_temporal_dist_udf(engagement_list):
        if not engagement_list:
            return {}
        result = create_temporal_cluster_distribution(
            engagement_list, min_timestamp, max_timestamp, NUM_TIME_BINS
        )
        print(f"DEBUG - temporal_dist sample: {str(result)[:200]}...")
        return result

    @F.udf(ArrayType(FloatType()))
    def rope_encoding_udf(cluster_dist):
        if not cluster_dist:
            return [0.0] * ROPE_DIM

        # Directly pass the cluster distribution dictionary to rope_inspired_encoding
        # This exactly matches the notebook implementation
        result = rope_inspired_encoding(cluster_dist, ROPE_DIM)
        print(f"DEBUG - rope_encoding sample: {str(result)[:200]}...")
        return result

    @F.udf(MapType(IntegerType(), ArrayType(FloatType())))
    def cluster_wise_rope_encoding_udf(cluster_dist):
        if not cluster_dist:
            return {}

        # Create a dictionary mapping cluster IDs to their temporal embeddings
        cluster_embeddings = {}

        for cluster_id, time_bins in cluster_dist.items():
            encoding = rope_inspired_encoding(time_bins, ROPE_DIM)
            cluster_embeddings[cluster_id] = encoding

        print(
            f"DEBUG - cluster_temporal_embeddings sample: {str(cluster_embeddings)[:200]}..."
        )
        return cluster_embeddings

    print("\nSTEP 3: Generating temporal embeddings")

    # Apply UDFs to generate temporal embeddings
    print("Creating temporal_cluster_distribution column...")
    df_temporal = df_user_dist.withColumn(
        "temporal_cluster_distribution",
        create_temporal_dist_udf("engagement_metadata_list"),
    )
    print("Sample of temporal_cluster_distribution:")
    df_temporal.select("temporal_cluster_distribution").limit(2).show(truncate=False)

    # Create overall temporal embedding (RoPE encoded across all clusters)
    print("Creating temporal_embedding column...")
    print("NOTE: This step passes the dictionary directly to rope_inspired_encoding,")
    print(
        "      which then flattens the bins internally - matching the notebook behavior."
    )
    df_temporal = df_temporal.withColumn(
        "temporal_embedding", rope_encoding_udf("temporal_cluster_distribution")
    )
    print("Sample of temporal_embedding:")
    df_temporal.select("temporal_embedding").limit(2).show(truncate=False)

    # Create per-cluster temporal embeddings
    print("Creating cluster_temporal_embeddings column...")
    print("NOTE: This creates separate embeddings for each cluster's time bins.")
    df_temporal = df_temporal.withColumn(
        "cluster_temporal_embeddings",
        cluster_wise_rope_encoding_udf("temporal_cluster_distribution"),
    )
    print("Sample of cluster_temporal_embeddings:")
    df_temporal.select("cluster_temporal_embeddings").limit(2).show(truncate=False)

    # Add metadata about the time boundaries used
    df_temporal = df_temporal.withColumn(
        "time_boundaries",
        F.struct(
            F.lit(
                min_timestamp.isoformat()
                if hasattr(min_timestamp, "isoformat")
                else str(min_timestamp)
            ).alias("min_timestamp"),
            F.lit(
                max_timestamp.isoformat()
                if hasattr(max_timestamp, "isoformat")
                else str(max_timestamp)
            ).alias("max_timestamp"),
            F.lit(NUM_TIME_BINS).alias("num_bins"),
        ),
    )

    # Select important columns for output
    df_result = df_temporal.select(
        "user_id",
        "cluster_distribution",
        "temporal_cluster_distribution",
        "temporal_embedding",
        "cluster_temporal_embeddings",
        "time_boundaries",
    )

    print("\nSTEP 4: Writing output data")

    # Write results to HDFS
    df_result.write.mode("overwrite").parquet(output_path)

    print(f"Successfully generated temporal embeddings at {output_path}")
    print("\nOutput schema:")
    df_result.printSchema()

    print("\nSample of data written:")
    df_result.limit(2).show(truncate=False)

    return df_result


def main():
    """Main execution function"""
    # First, copy files to HDFS if needed

    # Create local output directory if it doesn't exist
    local_output_dir = f"{DATA_ROOT}/emb_analysis"
    subprocess.call(["mkdir", "-p", local_output_dir])

    # Create hdfs directories, overwriting if they exist
    subprocess.call(["hdfs", "dfs", "-mkdir", "-p", "/tmp/emb_analysis"])
    subprocess.call(["hdfs", "dfs", "-mkdir", "-p", "/tmp/user_interaction"])

    # Copy input files to HDFS if not already there
    # Copy user_video_clusters_distribution.parquet
    input_file = "user_video_clusters_distribution.parquet"
    subprocess.call(
        [
            "hdfs",
            "dfs",
            "-put",
            "-f",
            f"{DATA_ROOT}/emb_analysis/{input_file}",
            f"/tmp/emb_analysis/{input_file}",
        ]
    )

    # Copy user_interaction_all.parquet (needed for timestamp boundaries)
    interaction_file = "user_interaction_all.parquet"
    subprocess.call(
        [
            "hdfs",
            "dfs",
            "-put",
            "-f",
            f"{DATA_ROOT}/user_interaction/{interaction_file}",
            f"/tmp/user_interaction/{interaction_file}",
        ]
    )

    # Generate temporal embeddings
    input_path = "/tmp/emb_analysis/user_video_clusters_distribution.parquet"
    output_path = "/tmp/emb_analysis/user_temporal_cluster_embeddings.parquet"
    _ = generate_temporal_embeddings(input_path, output_path)

    # Copy results back to local filesystem
    local_path = f"{local_output_dir}/user_temporal_cluster_embeddings.parquet"

    subprocess.call(["hdfs", "dfs", "-get", "-f", output_path, local_output_dir])

    print(f"Successfully wrote temporal embeddings to {output_path}")
    print(f"And copied back to {local_path}")


if __name__ == "__main__":
    main()
