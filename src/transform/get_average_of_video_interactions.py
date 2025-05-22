"""
Input:
- user_interaction_all.parquet
- video_index_all.parquet

Output:
- video_interaction_average.parquet
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
DATA_ROOT = "/home/dataproc/recommendation-engine/data_root"


# UDFs for embedding operations
def array_to_vector(array):
    return Vectors.dense(array)


array_to_vector_udf = F.udf(array_to_vector, VectorUDT())


def vector_to_array(vector):
    return vector.toArray().tolist()


vector_to_array_udf = F.udf(vector_to_array, ArrayType(FloatType()))


def main():
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

    # Load input data from HDFS
    user_interaction_path = "/tmp/user_interaction/user_interaction_all.parquet"
    video_index_path = "/tmp/video_index/video_index_all.parquet"

    user_interaction_df = spark.read.parquet(user_interaction_path)
    video_index_df = spark.read.parquet(video_index_path)

    # Join user interactions with video embeddings
    # Rename post_id to video_id for the join
    video_index_df = video_index_df.withColumnRenamed("post_id", "video_id")
    joined_df = user_interaction_df.join(
        video_index_df.select("video_id", "embedding"), on="video_id", how="inner"
    )

    # Convert timestamp to a sortable format if needed
    joined_df = joined_df.withColumn(
        "last_watched_timestamp", F.to_timestamp(joined_df["last_watched_timestamp"])
    )

    # Filter for users with minimum video watches
    user_video_counts = (
        joined_df.groupBy("user_id")
        .count()
        .filter(F.col("count") >= WATCHED_MIN_VIDEOS)
    )
    filtered_df = joined_df.join(
        user_video_counts.select("user_id"), on="user_id", how="inner"
    )

    # Window function to get top N recent videos per user
    from pyspark.sql.window import Window

    window_spec = Window.partitionBy("user_id").orderBy(
        F.desc("last_watched_timestamp")
    )
    filtered_df = filtered_df.withColumn("row_number", F.row_number().over(window_spec))
    filtered_df = filtered_df.filter(F.col("row_number") <= WATCHED_MAX_VIDEOS)

    # Clean and normalize percentage watched
    filtered_df = filtered_df.withColumn(
        "mean_percentage_watched",
        F.when(F.col("mean_percentage_watched").isNull(), 0).otherwise(
            F.col("mean_percentage_watched")
        ),
    )

    filtered_df = filtered_df.withColumn(
        "mean_percentage_watched_pct", F.col("mean_percentage_watched") * 100
    )

    # Filter for videos watched at least 25%
    filtered_df = filtered_df.filter(F.col("mean_percentage_watched_pct") > 25)

    # Convert arrays to vectors for processing
    filtered_df = filtered_df.withColumn(
        "embedding_vector", array_to_vector_udf(F.col("embedding"))
    )

    # Aggregate embeddings per user - simple mean (Experiment 1)
    # Define UDF to average arrays
    def average_arrays(arrays):
        if not arrays:
            return None
        return np.mean(arrays, axis=0).tolist()

    average_arrays_udf = F.udf(average_arrays, ArrayType(FloatType()))

    # Calculate average embedding
    user_avg_df = filtered_df.groupBy("user_id").agg(
        F.collect_list("embedding").alias("embedding_arrays")
    )

    user_avg_df = user_avg_df.withColumn(
        "embedding", average_arrays_udf(F.col("embedding_arrays"))
    ).select("user_id", "embedding")

    # L2 normalize the embeddings
    user_avg_df = user_avg_df.withColumn(
        "embedding_vector", array_to_vector_udf(F.col("embedding"))
    )

    normalizer = Normalizer(
        inputCol="embedding_vector", outputCol="normalized_embedding_vector", p=2.0
    )
    user_avg_df = normalizer.transform(user_avg_df)

    # Convert back to array format
    user_avg_df = user_avg_df.withColumn(
        "normalized_embedding",
        vector_to_array_udf(F.col("normalized_embedding_vector")),
    )

    # Select final columns
    result_df = user_avg_df.select(
        "user_id", F.col("normalized_embedding").alias("embedding")
    )

    # Write results to HDFS
    output_path = "/tmp/emb_analysis/video_interaction_average.parquet"
    result_df.write.mode("overwrite").parquet(output_path)

    # Create local output directory if not exists (double-check)
    subprocess.call(["mkdir", "-p", local_output_dir])

    # Copy results back to local filesystem
    subprocess.call(["hdfs", "dfs", "-get", "-f", output_path, local_output_dir])

    print(f"Successfully wrote video interaction averages to {output_path}")
    print(f"And copied back to {local_output_dir}/video_interaction_average.parquet")


if __name__ == "__main__":
    main()
