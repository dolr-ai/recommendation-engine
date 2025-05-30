"""
Input:
- user_interaction_all.parquet
- video_index_all.parquet

Output:
- video_interaction_average.parquet

This script combines the functionality of:
1. 001-get_user_item_emb.py - Preparing user-video interaction data with embeddings
2. 002-user_avg_item_emb.py - Computing average embeddings per user (using Experiment 1)
"""

import os
import sys
import pathlib
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml.feature import Normalizer
import numpy as np
from pyspark.ml.linalg import Vectors, VectorUDT

# Initialize Spark Session
spark = SparkSession.builder.appName("Video Interaction Average").getOrCreate()

# Define constants
WATCHED_MIN_VIDEOS = 2
WATCHED_MAX_VIDEOS = 100
WATCHED_MIN_PERCENTAGE = 25
DATA_ROOT = "/home/dataproc/recommendation-engine/data_root"


# UDFs for embedding operations
def array_to_vector(array):
    return Vectors.dense(array)


array_to_vector_udf = F.udf(array_to_vector, VectorUDT())


def vector_to_array(vector):
    return vector.toArray().tolist()


vector_to_array_udf = F.udf(vector_to_array, ArrayType(FloatType()))


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


def prepare_user_item_embeddings():
    """
    First part of the process (equivalent to 001-get_user_item_emb.py)
    - Load user interactions and video index data
    - Extract video_id from URI
    - Join to create user-item embedding dataset
    """
    print("STEP 1: Preparing user-item embeddings")

    # Load input data from HDFS
    user_interaction_path = "/tmp/user_interaction/user_interaction_all.parquet"
    video_index_path = "/tmp/video_index/video_index_all.parquet"

    df_user_interaction = spark.read.parquet(user_interaction_path)
    df_video_index = spark.read.parquet(video_index_path)

    # Print schema and counts for debugging
    print("User interaction count:", df_user_interaction.count())
    df_user_interaction.printSchema()

    print("Video index count:", df_video_index.count())
    df_video_index.printSchema()

    # Register UDF for video_id extraction
    extract_video_id_udf = F.udf(extract_video_id_from_uri, StringType())

    # Clean video index data and extract video_id
    df_video_index = df_video_index.filter(F.col("embedding").isNotNull())
    df_video_index = df_video_index.withColumn(
        "video_id", extract_video_id_udf(F.col("uri"))
    )

    # Verify video_id extraction
    print(
        "Video index with video_id count:",
        df_video_index.filter(F.col("video_id").isNotNull()).count(),
    )
    df_video_index.select("uri", "video_id").show(5, truncate=False)

    # Group by URI and aggregate embeddings
    # (Simplified compared to notebook since we're using Spark)
    df_video_index_agg = df_video_index.groupBy("uri", "video_id").agg(
        F.first("embedding").alias("embedding"),
        F.first("post_id").alias("post_id"),
        F.first("timestamp").alias("timestamp"),
        F.first("canister_id").alias("canister_id"),
        F.first("is_nsfw").alias("is_nsfw"),
        F.first("nsfw_ec").alias("nsfw_ec"),
    )

    # Join user_interaction with video_index_agg
    df_user_item_emb = df_user_interaction.select(
        "user_id", "video_id", "mean_percentage_watched", "last_watched_timestamp"
    ).join(
        df_video_index_agg.select("video_id", "embedding"), on="video_id", how="inner"
    )

    # Filter out nulls and save intermediate result
    df_user_item_emb = df_user_item_emb.filter(F.col("embedding").isNotNull())

    print("User-item embeddings count:", df_user_item_emb.count())
    print("Unique users:", df_user_item_emb.select("user_id").distinct().count())
    print("Unique videos:", df_user_item_emb.select("video_id").distinct().count())

    # Save intermediate result to HDFS
    user_item_emb_path = "/tmp/emb_analysis/user_item_emb.parquet"
    df_user_item_emb.write.mode("overwrite").parquet(user_item_emb_path)

    return df_user_item_emb


def calculate_user_average_embeddings(df_user_item_emb):
    """
    Second part of the process (equivalent to 002-user_avg_item_emb.py, Experiment 1)
    - Sort by recency
    - Filter by watch percentage
    - Calculate average embeddings per user
    - Normalize embeddings
    """
    print("\nSTEP 2: Calculating user average embeddings (Experiment 1)")

    # Convert timestamp to sortable format
    df_user_item_emb = df_user_item_emb.withColumn(
        "last_watched_timestamp",
        F.to_timestamp(df_user_item_emb["last_watched_timestamp"]),
    )

    # Sort by timestamp (newest first) and get top N videos per user
    df_user_item_emb = df_user_item_emb.sort(F.desc("last_watched_timestamp"))

    # Filter for users with minimum video watches
    df_user_counts = (
        df_user_item_emb.groupBy("user_id")
        .count()
        .filter(F.col("count") >= WATCHED_MIN_VIDEOS)
    )

    print("Users with min videos:", df_user_counts.count())

    # Join to keep only users with enough videos
    df_filtered = df_user_item_emb.join(
        df_user_counts.select("user_id"), on="user_id", how="inner"
    )

    # Window function to get top N recent videos per user
    from pyspark.sql.window import Window

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

    # Filter for videos watched at least 25% (as in notebook)
    df_filtered = df_filtered.filter(
        F.col("mean_percentage_watched_pct") > WATCHED_MIN_PERCENTAGE
    )
    print("After percentage watched filter count:", df_filtered.count())

    # Define UDF to average arrays
    def average_arrays(arrays):
        if not arrays or len(arrays) == 0:
            return None
        try:
            return np.mean(arrays, axis=0).tolist()
        except Exception as e:
            print(f"Error averaging arrays: {e}")
            return None

    average_arrays_udf = F.udf(average_arrays, ArrayType(FloatType()))

    # Calculate average embedding per user
    df_user_avg = df_filtered.groupBy("user_id").agg(
        F.collect_list("embedding").alias("embedding_arrays")
    )

    df_user_avg = df_user_avg.withColumn(
        "embedding", average_arrays_udf(F.col("embedding_arrays"))
    ).select("user_id", "embedding")

    # Filter out nulls if any
    df_user_avg = df_user_avg.filter(F.col("embedding").isNotNull())
    print("Users with average embeddings:", df_user_avg.count())

    # L2 normalize the embeddings
    df_user_avg = df_user_avg.withColumn(
        "embedding_vector", array_to_vector_udf(F.col("embedding"))
    )

    normalizer = Normalizer(
        inputCol="embedding_vector", outputCol="normalized_embedding_vector", p=2.0
    )
    df_user_avg = normalizer.transform(df_user_avg)

    # Convert back to array format
    df_user_avg = df_user_avg.withColumn(
        "normalized_embedding",
        vector_to_array_udf(F.col("normalized_embedding_vector")),
    )

    # Select final columns
    df_result = df_user_avg.select(
        "user_id", F.col("normalized_embedding").alias("embedding")
    )

    print("Final result count:", df_result.count())
    return df_result


def main():
    """Main execution function"""
    # First, copy files to HDFS
    import subprocess

    # Create local output directory if it doesn't exist
    local_output_dir = f"{DATA_ROOT}/emb_analysis"
    subprocess.call(["mkdir", "-p", local_output_dir])

    # Create hdfs directories, overwriting if they exist
    subprocess.call(["hdfs", "dfs", "-mkdir", "-p", "/tmp/user_interaction"])
    subprocess.call(["hdfs", "dfs", "-mkdir", "-p", "/tmp/video_index"])
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
            f"{DATA_ROOT}/video_index/video_index_all.parquet",
            "/tmp/video_index/video_index_all.parquet",
        ]
    )

    # Step 1: Prepare user-item embeddings (001-get_user_item_emb.py)
    df_user_item_emb = prepare_user_item_embeddings()

    # Step 2: Calculate user average embeddings (002-user_avg_item_emb.py)
    df_result = calculate_user_average_embeddings(df_user_item_emb)

    # Write results to HDFS
    output_path = "/tmp/emb_analysis/video_interaction_average.parquet"
    df_result.write.mode("overwrite").parquet(output_path)

    # Create local output directory if not exists (double-check)
    subprocess.call(["mkdir", "-p", local_output_dir])

    # Copy results back to local filesystem
    subprocess.call(["hdfs", "dfs", "-get", "-f", output_path, local_output_dir])

    # Show a sample of data that was written
    print("\nSample of data written:")
    df_result.show(5)

    print(f"Successfully wrote video interaction averages to {output_path}")
    print(f"And copied back to {local_output_dir}/video_interaction_average.parquet")


if __name__ == "__main__":
    main()
