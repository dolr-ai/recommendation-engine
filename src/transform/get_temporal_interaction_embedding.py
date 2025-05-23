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


def rope_inspired_encoding(cluster_id, time_bins, dim=ROPE_DIM):
    """
    Create a position-aware temporal encoding inspired by RoPE (Rotary Position Embedding)

    Args:
        cluster_id: ID of the cluster
        time_bins: List of count values for each time bin
        dim: Dimension of the resulting embedding vector (must be even)

    Returns:
        Array containing the temporal encoding
    """
    if dim % 2 != 0:
        raise ValueError("Embedding dimension must be even for ROPE-style encodings")

    # Initialize the final embedding for this cluster
    cluster_embedding = np.zeros(dim)

    # Create a base embedding based on cluster ID
    base_embedding = np.zeros(dim)
    for i in range(dim):
        # Create a unique pattern based on cluster ID
        base_embedding[i] = np.sin(cluster_id * (i + 1) / dim) * 0.5 + 0.5

    # Normalize the base embedding
    base_embedding = base_embedding / np.linalg.norm(base_embedding)

    # For each time bin, apply a rotary position encoding if there were accesses
    for pos, access_count in enumerate(time_bins):
        if access_count > 0:
            # Create position-dependent frequencies
            freqs = 1.0 / (10000 ** (np.arange(0, dim, 2) / dim))

            # Calculate rotation angles based on position
            theta = pos * freqs

            # Create rotation matrix elements
            cos_values = np.cos(theta)
            sin_values = np.sin(theta)

            # Apply rotation to pairs of dimensions
            rotated_embedding = base_embedding.copy()
            for i in range(0, dim, 2):
                # Rotation in 2D subspace
                x, y = rotated_embedding[i], rotated_embedding[i + 1]
                cos_theta, sin_theta = cos_values[i // 2], sin_values[i // 2]

                rotated_embedding[i] = x * cos_theta - y * sin_theta
                rotated_embedding[i + 1] = x * sin_theta + y * cos_theta

            # Scale by access count and add to the final embedding
            cluster_embedding += rotated_embedding * access_count

    # Normalize the final embedding
    if np.linalg.norm(cluster_embedding) > 0:
        cluster_embedding = cluster_embedding / np.linalg.norm(cluster_embedding)

    return cluster_embedding.tolist()


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
        return create_temporal_cluster_distribution(
            engagement_list, min_timestamp, max_timestamp, NUM_TIME_BINS
        )

    @F.udf(MapType(IntegerType(), ArrayType(FloatType())))
    def cluster_wise_rope_encoding_udf(cluster_dist):
        if not cluster_dist:
            return {}

        # Create a dictionary mapping cluster IDs to their temporal embeddings
        cluster_embeddings = {}

        for cluster_id, time_bins in cluster_dist.items():
            if any(time_bins):
                encoding = rope_inspired_encoding(cluster_id, time_bins, ROPE_DIM)
                cluster_embeddings[cluster_id] = encoding

        return cluster_embeddings

    @F.udf(ArrayType(FloatType()))
    def average_cluster_embeddings_udf(cluster_embeddings_map):
        if not cluster_embeddings_map:
            return [0.0] * ROPE_DIM

        # Extract all embeddings from the map
        embeddings = [np.array(emb) for emb in cluster_embeddings_map.values()]

        # Take the average of all embeddings
        if embeddings:
            avg_embedding = np.mean(embeddings, axis=0)
            # Normalize the average embedding
            norm = np.linalg.norm(avg_embedding)
            if norm > 0:
                avg_embedding = avg_embedding / norm
            return avg_embedding.tolist()
        else:
            return [0.0] * ROPE_DIM

    print("\nSTEP 3: Generating temporal embeddings")

    # Apply UDFs to generate temporal embeddings
    df_temporal = df_user_dist.withColumn(
        "temporal_cluster_distribution",
        create_temporal_dist_udf("engagement_metadata_list"),
    )

    # Create per-cluster temporal embeddings
    df_temporal = df_temporal.withColumn(
        "cluster_temporal_embeddings",
        cluster_wise_rope_encoding_udf("temporal_cluster_distribution"),
    )

    # Create overall temporal embedding by averaging the per-cluster embeddings
    df_temporal = df_temporal.withColumn(
        "temporal_embedding",
        average_cluster_embeddings_udf("cluster_temporal_embeddings"),
    )

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
