"""
Input:
- video_index_all.parquet

Output:
- video_clusters.parquet
- video_cluster_label_map.json

This script implements video clustering using KMeans in PySpark, based on the notebook
000-video_embedding_clusters.py, but optimized for distributed processing.
"""

import os
import sys
import json
import pathlib
from datetime import datetime
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import *
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import StandardScaler
from pyspark.ml.linalg import Vectors, VectorUDT
import numpy as np
from kneed import KneeLocator

# Initialize Spark Session
spark = SparkSession.builder.appName("Video Clustering").getOrCreate()

# Define constants
MIN_K_VIDEO = 3
# todo: increase this to 15 later
MAX_K_VIDEO = 10

# todo: remove this later
K_VALUES = [6, 12]

DEFAULT_OPTIMAL_K_CLUSTERS = 8
DATA_ROOT = "/home/dataproc/recommendation-engine/data_root"


# UDFs for embedding operations
def array_to_vector(array):
    """Convert array to vector for ML processing"""
    if array is None:
        return None
    return Vectors.dense(array)


array_to_vector_udf = F.udf(array_to_vector, VectorUDT())


def vector_to_array(vector):
    """Convert vector back to array"""
    if vector is None:
        return None
    return vector.toArray().tolist()


vector_to_array_udf = F.udf(vector_to_array, ArrayType(FloatType()))


def average_arrays(arrays):
    """Average multiple arrays element-wise"""
    if not arrays or len(arrays) == 0:
        return None

    # Filter out None values
    valid_arrays = [arr for arr in arrays if arr is not None]
    if not valid_arrays:
        return None

    # Ensure all arrays have the same length
    length = len(valid_arrays[0])
    if not all(len(arr) == length for arr in valid_arrays):
        return None

    # Calculate the average for each position
    result = [0.0] * length
    for arr in valid_arrays:
        for i, val in enumerate(arr):
            result[i] += val

    return [val / len(valid_arrays) for val in result]


# Register UDF for averaging arrays
average_arrays_udf = F.udf(average_arrays, ArrayType(FloatType()))


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


def save_json(data, path):
    """Helper function to save data to a JSON file"""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def create_directories(paths):
    """Create directories if they don't exist"""
    for path in paths:
        os.makedirs(path, exist_ok=True)


def prepare_video_embeddings():
    """
    Prepare video embeddings for clustering
    - Load video index data
    - Extract video_id from URI
    - Group by video_id and get embeddings
    """
    print("STEP 1: Preparing video embeddings")

    # Load video index data from HDFS
    video_index_path = "/tmp/video_index/video_index_all.parquet"
    df_video_index = spark.read.parquet(video_index_path)

    # Print schema and counts for debugging
    print("Video index count:", df_video_index.count())
    df_video_index.printSchema()

    # Register UDF for video_id extraction
    extract_video_id_udf = F.udf(extract_video_id_from_uri, StringType())

    # Clean video index data and extract video_id
    df_video_index = df_video_index.filter(F.col("embedding").isNotNull())
    df_video_index = df_video_index.withColumn(
        "video_id", extract_video_id_udf(F.col("uri"))
    )
    df_video_index = df_video_index.filter(F.col("video_id").isNotNull())

    # Verify video_id extraction
    print(
        "Video index with valid video_id count:",
        df_video_index.filter(F.col("video_id").isNotNull()).count(),
    )
    df_video_index.select("uri", "video_id").show(5, truncate=False)

    # Group by video_id and average the embeddings using collect_list and our custom UDF
    df_video_embeddings = df_video_index.groupBy("video_id").agg(
        average_arrays_udf(F.collect_list("embedding")).alias("embedding")
    )

    # Convert array embeddings to vectors for ML processing
    df_video_embeddings = df_video_embeddings.withColumn(
        "embedding_vector", array_to_vector_udf(F.col("embedding"))
    )

    print("Prepared video embeddings count:", df_video_embeddings.count())
    return df_video_embeddings


def cluster_videos(
    df_video_embeddings,
    min_k=MIN_K_VIDEO,
    max_k=MAX_K_VIDEO,
    default_k=DEFAULT_OPTIMAL_K_CLUSTERS,
):
    """
    Perform KMeans clustering on video embeddings
    - Standardize features
    - Find optimal k using elbow method and silhouette score
    - Apply KMeans with the optimal k
    - Return the video-cluster mapping
    """
    print("STEP 2: Clustering video embeddings")

    # Standardize features
    standardScaler = StandardScaler(
        inputCol="embedding_vector",
        outputCol="scaled_embedding",
        withStd=True,
        withMean=True,
    )
    scaler_model = standardScaler.fit(df_video_embeddings)
    df_scaled = scaler_model.transform(df_video_embeddings)

    # Create output directory
    trans_dir = f"{DATA_ROOT}/transformed/video_clusters"
    create_directories([trans_dir])

    # Find optimal k
    print(f"Finding optimal k in range {min_k} to {max_k}...")

    # Create empty lists to store metrics
    silhouette_scores = []
    inertia_values = []
    # todo: remove this later
    # k_values = list(range(min_k, max_k + 1))
    k_values = K_VALUES

    # Evaluate different k values
    for k in k_values:
        print(f"Evaluating k={k}")
        kmeans = KMeans(k=k, featuresCol="scaled_embedding")
        model = kmeans.fit(df_scaled)
        predictions = model.transform(df_scaled)

        # Calculate silhouette score (higher is better)
        evaluator = ClusteringEvaluator(
            predictionCol="prediction",
            featuresCol="scaled_embedding",
            metricName="silhouette",
        )
        silhouette = evaluator.evaluate(predictions)
        silhouette_scores.append(silhouette)

        # Extract inertia (WSSSE - within-set sum of squared errors)
        inertia = model.summary.trainingCost
        inertia_values.append(inertia)

        print(f"  k={k}, Silhouette={silhouette:.4f}, WSSSE={inertia:.2f}")

    # Find optimal k using silhouette score (higher is better)
    optimal_k_silhouette = k_values[np.argmax(silhouette_scores)]
    print(f"Optimal k from silhouette score: {optimal_k_silhouette}")

    # Find optimal k using elbow method
    try:
        # Use kneed package to find the "elbow point"
        kneedle = KneeLocator(
            k_values, inertia_values, curve="convex", direction="decreasing"
        )
        optimal_k_elbow = kneedle.elbow
        print(f"Optimal k from elbow method: {optimal_k_elbow}")
    except Exception as e:
        print(f"Elbow method failed, using default k: {default_k}. Error: {e}")
        optimal_k_elbow = default_k

    # Choose final optimal k
    if optimal_k_elbow:
        optimal_k = max(optimal_k_silhouette, optimal_k_elbow, default_k)
    else:
        optimal_k = max(optimal_k_silhouette, default_k)

    print(f"Using optimal k: {optimal_k}")

    # Train final model with optimal k
    final_kmeans = KMeans(k=optimal_k, seed=42, featuresCol="scaled_embedding")
    final_model = final_kmeans.fit(df_scaled)
    final_predictions = final_model.transform(df_scaled)

    # Select only the columns we need
    df_video_clusters = final_predictions.select(
        "video_id", F.col("prediction").alias("cluster")
    ).join(df_video_embeddings.select("video_id", "embedding"), "video_id", "inner")

    # Calculate cluster sizes for summary
    cluster_counts = df_video_clusters.groupBy("cluster").count().collect()
    cluster_sizes = [
        {"cluster": int(row["cluster"]), "video_count": int(row["count"])}
        for row in cluster_counts
    ]

    # Create summary dictionary with current timestamp
    current_timestamp = datetime.now().isoformat()

    # Convert silhouette scores and inertia values to regular Python types for JSON serialization
    silhouette_dict = {
        int(k): float(score) for k, score in zip(k_values, silhouette_scores)
    }
    inertia_dict = {
        int(k): float(inertia) for k, inertia in zip(k_values, inertia_values)
    }

    summary = {
        "total_videos": int(df_video_embeddings.count()),
        "optimal_clusters_silhouette": int(optimal_k_silhouette),
        "optimal_clusters_elbow": int(optimal_k_elbow) if optimal_k_elbow else None,
        "optimal_clusters_used": int(optimal_k),
        "silhouette_scores": silhouette_dict,
        "inertia_values": inertia_dict,
        "cluster_sizes": cluster_sizes,
        "timestamp": current_timestamp,
    }

    # Save summary
    save_json(summary, f"{trans_dir}/video_cluster_analysis_summary.json")

    # Create and save cluster map as JSON
    df_map = df_video_clusters.collect()
    cluster_map = {row["video_id"]: int(row["cluster"]) for row in df_map}
    save_json(cluster_map, f"{trans_dir}/video_cluster_label_map.json")

    print(f"Video clustering complete. Found {optimal_k} clusters.")
    print(f"Results saved to {trans_dir}")

    return df_video_clusters, optimal_k


def main():
    """Main execution function"""
    # First, copy files to HDFS
    import subprocess

    # Create local output directory
    trans_dir = f"{DATA_ROOT}/transformed/video_clusters"
    create_directories([trans_dir])

    # Create hdfs directories
    subprocess.call(["hdfs", "dfs", "-mkdir", "-p", "/tmp/video_index"])
    subprocess.call(["hdfs", "dfs", "-mkdir", "-p", "/tmp/transformed/video_clusters"])

    # Copy input files to HDFS
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

    # Step 1: Prepare video embeddings
    df_video_embeddings = prepare_video_embeddings()

    # Step 2: Cluster videos
    df_video_clusters, optimal_k = cluster_videos(
        df_video_embeddings,
        min_k=MIN_K_VIDEO,
        max_k=MAX_K_VIDEO,
        default_k=DEFAULT_OPTIMAL_K_CLUSTERS,
    )

    # Write results to HDFS
    output_path = "/tmp/transformed/video_clusters/video_clusters.parquet"
    df_video_clusters.write.mode("overwrite").parquet(output_path)

    # Copy results back to local filesystem
    subprocess.call(["hdfs", "dfs", "-get", "-f", output_path, trans_dir])

    # Show a sample of clusters
    print("\nSample of video clusters:")
    df_video_clusters.show(10)

    # Show cluster sizes
    print("\nVideo counts per cluster:")
    df_video_clusters.groupBy("cluster").count().orderBy("cluster").show(optimal_k)

    print(f"Successfully wrote video clusters to {output_path}")
    print(f"And copied back to {trans_dir}/video_clusters.parquet")
    print(f"Cluster mapping saved to {trans_dir}/video_cluster_label_map.json")


if __name__ == "__main__":
    main()
