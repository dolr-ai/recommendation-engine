"""
Input:
- merged_user_embeddings.parquet (output from merge_part_embeddings.py)

Output:
- user_clusters.parquet (users with their assigned cluster IDs)
- user_cluster_label_map.json (mapping of user_id to cluster ID)

This script implements user clustering using KMeans in PySpark, similar to
get_video_clusters.py, but for user embeddings instead of video embeddings.
"""

import os
import json
import subprocess
import time
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
spark = SparkSession.builder.appName("User Clustering").getOrCreate()

# Define constants
MIN_K_USER = 3
MAX_K_USER = 10

# Using only 3 K values for faster execution
# Note: Elbow method requires at least 2 k values, using single value will disable elbow method
K_VALUES = [6]


DEFAULT_OPTIMAL_K_CLUSTERS = 6
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


def save_json(data, path):
    """Helper function to save data to a JSON file"""
    print(f"Saving JSON to: {path}")
    start_time = time.time()
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"JSON saved in {time.time() - start_time:.2f} seconds")


def create_directories(paths):
    """Create directories if they don't exist"""
    for path in paths:
        os.makedirs(path, exist_ok=True)


def prepare_user_embeddings():
    """
    Prepare user embeddings for clustering
    - Load merged user embeddings data
    - Filter for non-null embeddings
    - Convert to vector format for clustering
    """
    print("STEP 1: Preparing user embeddings")
    start_time = time.time()

    # Load merged user embeddings data from HDFS
    user_embeddings_path = "/tmp/emb_analysis/merged_user_embeddings.parquet"
    df_user_embeddings = spark.read.parquet(user_embeddings_path)

    # Print schema and counts for debugging
    print("User embeddings count:", df_user_embeddings.count())
    df_user_embeddings.printSchema()

    # Filter for non-null embeddings
    df_user_embeddings = df_user_embeddings.filter(F.col("user_embedding").isNotNull())

    # Convert array embeddings to vectors for ML processing
    df_user_embeddings = df_user_embeddings.withColumn(
        "embedding_vector", array_to_vector_udf(F.col("user_embedding"))
    )

    print("Prepared user embeddings count:", df_user_embeddings.count())
    print(f"STEP 1 completed in {time.time() - start_time:.2f} seconds")
    return df_user_embeddings


def cluster_users(
    df_user_embeddings,
    min_k=MIN_K_USER,
    max_k=MAX_K_USER,
    default_k=DEFAULT_OPTIMAL_K_CLUSTERS,
):
    """
    Perform KMeans clustering on user embeddings
    - Standardize features
    - Find optimal k using elbow method and silhouette score
    - Apply KMeans with the optimal k
    - Return the user-cluster mapping
    """
    print("STEP 2: Clustering user embeddings")
    clustering_start_time = time.time()

    # Standardize features
    standardScaler = StandardScaler(
        inputCol="embedding_vector",
        outputCol="scaled_embedding",
        withStd=True,
        withMean=True,
    )
    scaler_model = standardScaler.fit(df_user_embeddings)
    df_scaled = scaler_model.transform(df_user_embeddings)

    # Create output directory
    trans_dir = f"{DATA_ROOT}/transformed/user_clusters"
    create_directories([trans_dir])

    # Find optimal k
    print(f"Finding optimal k in range {min_k} to {max_k}...")

    # Create empty lists to store metrics
    silhouette_scores = []
    inertia_values = []
    k_values = K_VALUES  # For testing, use predefined values

    # Evaluate different k values
    for k in k_values:
        print(f"Evaluating k={k}")
        k_start_time = time.time()
        
        kmeans = KMeans(k=k, featuresCol="scaled_embedding")
        print(f"  Training KMeans model...")
        model = kmeans.fit(df_scaled)
        print(f"  KMeans trained in {time.time() - k_start_time:.2f} seconds")
        
        transform_start = time.time()
        predictions = model.transform(df_scaled)
        print(f"  Predictions generated in {time.time() - transform_start:.2f} seconds")

        # Calculate silhouette score (higher is better)
        eval_start = time.time()
        evaluator = ClusteringEvaluator(
            predictionCol="prediction",
            featuresCol="scaled_embedding",
            metricName="silhouette",
        )
        silhouette = evaluator.evaluate(predictions)
        silhouette_scores.append(silhouette)
        print(f"  Silhouette evaluation in {time.time() - eval_start:.2f} seconds")

        # Extract inertia (WSSSE - within-set sum of squared errors)
        inertia = model.summary.trainingCost
        inertia_values.append(inertia)

        total_k_time = time.time() - k_start_time
        print(f"  k={k}, Silhouette={silhouette:.4f}, WSSSE={inertia:.2f} (Total: {total_k_time:.2f}s)")

    # Find optimal k using silhouette score (higher is better)
    if len(silhouette_scores) == 0 or all(np.isnan(silhouette_scores)):
        print("No valid silhouette scores, using default k")
        optimal_k_silhouette = default_k
    else:
        # Filter out NaN values and find the best k
        valid_scores = [(k, score) for k, score in zip(k_values, silhouette_scores) if not np.isnan(score)]
        if valid_scores:
            optimal_k_silhouette = max(valid_scores, key=lambda x: x[1])[0]
        else:
            optimal_k_silhouette = default_k
    print(f"Optimal k from silhouette score: {optimal_k_silhouette}")

    # Find optimal k using elbow method
    optimal_k_elbow = None
    try:
        # Validate input arrays for elbow method
        if len(k_values) < 2 or len(inertia_values) < 2:
            print(f"Not enough data points for elbow method (need at least 2, got {len(k_values)})")
        elif len(set(inertia_values)) < 2:
            print("All inertia values are the same, cannot determine elbow")
        elif any(np.isnan(inertia_values)) or any(np.isinf(inertia_values)):
            print("Invalid inertia values detected (NaN or inf), cannot determine elbow")
        else:
            # Use kneed package to find the "elbow point"
            kneedle = KneeLocator(
                k_values, inertia_values, curve="convex", direction="decreasing"
            )
            optimal_k_elbow = kneedle.elbow
            print(f"Optimal k from elbow method: {optimal_k_elbow}")

        # Check if elbow is None (no clear elbow point found)
        if optimal_k_elbow is None:
            print(f"No clear elbow point found, will use silhouette score method")
    except Exception as e:
        print(f"Elbow method failed: {e}")
        optimal_k_elbow = None

    # Choose final optimal k - handle None case properly
    if len(k_values) == 1:
        # If only one k value was tested, use that value
        optimal_k = k_values[0]
        print(f"Only one k value tested ({k_values[0]}), using that value")
    elif optimal_k_elbow is not None:
        optimal_k = max(optimal_k_silhouette, optimal_k_elbow, default_k)
    else:
        optimal_k = max(optimal_k_silhouette, default_k)

    print(f"Using optimal k: {optimal_k}")

    # Train final model with optimal k
    final_model_start = time.time()
    print("Training final KMeans model...")
    final_kmeans = KMeans(k=optimal_k, featuresCol="scaled_embedding")
    final_model = final_kmeans.fit(df_scaled)
    print(f"Final model trained in {time.time() - final_model_start:.2f} seconds")
    
    final_transform_start = time.time()
    print("Generating final predictions...")
    final_predictions = final_model.transform(df_scaled)
    print(f"Final predictions generated in {time.time() - final_transform_start:.2f} seconds")

    # Select only the columns we need
    df_user_clusters = final_predictions.select(
        "user_id", F.col("prediction").alias("cluster")
    ).join(
        df_user_embeddings.select(
            "user_id",
            "user_embedding",
            "avg_interaction_embedding",
            "cluster_distribution_embedding",
            "temporal_embedding",
            "engagement_metadata_list",
        ),
        "user_id",
        "inner",
    )

    # Calculate cluster sizes for summary
    cluster_counts = df_user_clusters.groupBy("cluster").count().collect()
    cluster_sizes = [
        {"cluster": int(row["cluster"]), "user_count": int(row["count"])}
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
        "total_users": int(df_user_embeddings.count()),
        "optimal_clusters_silhouette": int(optimal_k_silhouette),
        "optimal_clusters_elbow": (
            int(optimal_k_elbow) if optimal_k_elbow is not None else None
        ),
        "optimal_clusters_used": int(optimal_k),
        "silhouette_scores": silhouette_dict,
        "inertia_values": inertia_dict,
        "cluster_sizes": cluster_sizes,
        "timestamp": current_timestamp,
    }

    # Save summary
    save_json(summary, f"{trans_dir}/user_cluster_analysis_summary.json")

    # Create and save cluster map as JSON - use pandas by default
    print("Creating cluster mapping using pandas...")
    map_creation_start = time.time()
    
    total_users = df_user_clusters.count()
    print(f"Creating cluster map for {total_users} users...")
    
    try:
        # Use pandas by default - fuck Spark collect()
        pandas_df = df_user_clusters.select("user_id", "cluster").toPandas()
        cluster_map = dict(zip(pandas_df['user_id'], pandas_df['cluster'].astype(int)))
        print(f"Cluster mapping created via pandas in {time.time() - map_creation_start:.2f} seconds")
    except Exception as e:
        print(f"Pandas conversion failed ({e}), falling back to collect...")
        # Only fallback to collect if pandas completely fails
        df_map = df_user_clusters.select("user_id", "cluster").collect()
        cluster_map = {row["user_id"]: int(row["cluster"]) for row in df_map}
        print(f"Cluster mapping created via collect fallback in {time.time() - map_creation_start:.2f} seconds")
    
    save_json(cluster_map, f"{trans_dir}/user_cluster_label_map.json")

    total_clustering_time = time.time() - clustering_start_time
    print(f"User clustering complete. Found {optimal_k} clusters.")
    print(f"Total clustering time: {total_clustering_time:.2f} seconds")
    print(f"Results saved to {trans_dir}")

    # Print dimensions of each embedding type
    print("\nEmbedding dimensions in get_user_clusters.py:")

    # Convert to pandas to easily access the first row
    sample_row = df_user_clusters.limit(1).toPandas()

    if not sample_row.empty:
        print(f"user_embedding dimensions: {len(sample_row['user_embedding'].iloc[0])}")
        print(
            f"avg_interaction_embedding dimensions: {len(sample_row['avg_interaction_embedding'].iloc[0])}"
        )
        print(
            f"temporal_embedding dimensions: {len(sample_row['temporal_embedding'].iloc[0])}"
        )
        print(
            f"cluster_distribution_embedding dimensions: {len(sample_row['cluster_distribution_embedding'].iloc[0])}"
        )
    else:
        print("No data available to print dimensions")

    return df_user_clusters, optimal_k


def main():
    """Main execution function"""

    # Create local output directory
    trans_dir = f"{DATA_ROOT}/transformed/user_clusters"
    create_directories([trans_dir])
    print(f"Created local directory: {trans_dir}")

    # Create hdfs directories
    subprocess.call(["hdfs", "dfs", "-mkdir", "-p", "/tmp/emb_analysis"])
    subprocess.call(["hdfs", "dfs", "-mkdir", "-p", "/tmp/transformed/user_clusters"])
    # Verify HDFS directories were created
    print("Verifying HDFS directories:")
    subprocess.call(["hdfs", "dfs", "-ls", "/tmp/transformed"])

    # Copy input files to HDFS
    subprocess.call(
        [
            "hdfs",
            "dfs",
            "-put",
            "-f",
            f"{DATA_ROOT}/emb_analysis/merged_user_embeddings.parquet",
            "/tmp/emb_analysis/merged_user_embeddings.parquet",
        ]
    )

    # Step 1: Prepare user embeddings
    df_user_embeddings = prepare_user_embeddings()

    # Step 2: Cluster users
    df_user_clusters, optimal_k = cluster_users(
        df_user_embeddings,
        min_k=MIN_K_USER,
        max_k=MAX_K_USER,
        default_k=DEFAULT_OPTIMAL_K_CLUSTERS,
    )

    # Write results to HDFS
    output_path = "/tmp/transformed/user_clusters/user_clusters.parquet"
    write_start_time = time.time()
    print(f"Writing results to HDFS: {output_path}")
    
    # Keep original partitioning for downstream compatibility
    df_user_clusters.write.mode("overwrite").parquet(output_path)
    print(f"HDFS write completed in {time.time() - write_start_time:.2f} seconds")

    # Copy results back to local filesystem
    local_output_path = f"{trans_dir}/user_clusters.parquet"
    copy_start_time = time.time()
    print(f"Copying results to local filesystem: {local_output_path}")
    
    # Create local directory if it doesn't exist
    subprocess.call(["mkdir", "-p", local_output_path])

    # Copy all partition files from HDFS to local
    copy_result = subprocess.call(
        ["hdfs", "dfs", "-get", "-f", f"{output_path}/*", local_output_path]
    )
    
    if copy_result == 0:
        print(f"Local copy completed in {time.time() - copy_start_time:.2f} seconds")
    else:
        print(f"Local copy failed with return code: {copy_result}")

    # Verify local file was created
    if os.path.exists(local_output_path):
        print(f"Successfully copied to local path: {local_output_path}")
        # List the files to verify
        subprocess.call(["ls", "-la", local_output_path])
        # Get directory size to verify it's not empty
        local_size = (
            subprocess.check_output(["du", "-sh", local_output_path])
            .decode()
            .split()[0]
        )
        print(f"Local directory size: {local_size}")
    else:
        print(f"WARNING: Failed to copy to local path: {local_output_path}")
        # Try to create the directory again and copy
        os.makedirs(trans_dir, exist_ok=True)
        subprocess.call(["mkdir", "-p", local_output_path])
        subprocess.call(
            ["hdfs", "dfs", "-get", "-f", f"{output_path}/*", local_output_path]
        )

    # Show a sample of clusters
    print("\nSample of user clusters:")
    df_user_clusters.show(10)

    # Show cluster sizes
    print("\nUser counts per cluster:")
    df_user_clusters.groupBy("cluster").count().orderBy("cluster").show(int(optimal_k))

    # Verify HDFS file exists
    hdfs_check = subprocess.run(
        ["hdfs", "dfs", "-test", "-e", output_path], capture_output=True
    )
    if hdfs_check.returncode == 0:
        print(f"Successfully wrote user clusters to {output_path}")
    else:
        print(f"WARNING: HDFS file {output_path} does not exist after writing!")

    print(f"And copied back to {local_output_path}")
    print(f"Cluster mapping saved to {trans_dir}/user_cluster_label_map.json")

    # List both locations to verify
    print("\nVerifying HDFS output directory:")
    subprocess.call(["hdfs", "dfs", "-ls", "/tmp/transformed/user_clusters"])
    print("\nVerifying local output directory:")
    subprocess.call(["ls", "-la", trans_dir])


if __name__ == "__main__":
    main()
