# %%
import os
import json
import pandas as pd
import asyncio
import random

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timezone
from dotenv import load_dotenv
import pathlib
from tqdm import tqdm
from utils.common_utils import path_exists
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, normalize
from sklearn.metrics import silhouette_score
import plotly.express as px
from kneed import KneeLocator

# utils
from utils.gcp_utils import GCPUtils

# setup configs
print(load_dotenv("/Users/sagar/work/yral/recommendation-engine/.env"))

print(os.getenv("DATA_ROOT"))

DATA_ROOT = pathlib.Path(os.getenv("DATA_ROOT"))
# %%
df = pd.read_json(
    "/Users/sagar/work/yral/recommendation-engine/data-temp/test_new_temporal.json"
)
# %%

df.columns

# %%
# Examine the shape and structure of temporal_embedding
print(f"Total users: {len(df)}")
print(f"Sample temporal_embedding dimension: {len(df['temporal_embedding'].iloc[0])}")


# %%
# Function to perform clustering on embeddings and visualize results
def cluster_and_visualize_embeddings(
    df,
    embedding_column,
    min_k=2,
    max_k=15,
    default_k=3,
    tsne_perplexity=30,
    tsne_n_iter=1000,
):
    # Extract embeddings as numpy array
    embeddings = np.array(df[embedding_column].tolist())
    print(f"Embedding array shape: {embeddings.shape}")

    # Perform dimensionality reduction with t-SNE
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(
        n_components=2,
        perplexity=tsne_perplexity,
        n_iter=tsne_n_iter,
        random_state=42,
        metric="cosine",
    )
    tsne_results = tsne.fit_transform(embeddings)

    # Create dataframe with t-SNE results
    df_tsne = pd.DataFrame(tsne_results, columns=["tsne_1", "tsne_2"])
    df_tsne["user_id"] = df["user_id"].values

    # Plot t-SNE projection
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5)
    ax.set_title("t-SNE Projection of Temporal Embeddings")
    plt.show()

    # Determine optimal number of clusters
    silhouette_scores = []
    inertia_values = []
    k_range = range(min_k, max_k)

    print("Determining optimal k for temporal embedding clusters...")
    for k in tqdm(k_range, desc="K-Means"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(embeddings)

        if (
            len(np.unique(cluster_labels)) > 1
        ):  # Silhouette score requires at least 2 labels
            score = silhouette_score(embeddings, cluster_labels, metric="cosine")
            silhouette_scores.append(score)
        else:
            silhouette_scores.append(
                -1
            )  # Assign a low score if only one cluster is formed

        inertia_values.append(kmeans.inertia_)
        print(
            f"K={k}, Silhouette Score: {silhouette_scores[-1]:.4f}, Inertia: {inertia_values[-1]:.2f}"
        )

    # Plot silhouette scores
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_range, silhouette_scores, "o-")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("Silhouette Score for Different Cluster Counts")
    ax.grid(True)
    plt.show()

    # Plot inertia values
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_range, inertia_values, "o-")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia (Sum of Squared Distances)")
    ax.set_title("Elbow Method for Optimal k")
    ax.grid(True)
    plt.show()

    # Identify optimal k
    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    print(
        f"Optimal number of clusters based on silhouette score: {optimal_k_silhouette}"
    )

    try:
        kneedle = KneeLocator(
            list(k_range), inertia_values, curve="convex", direction="decreasing"
        )
        optimal_k_elbow = kneedle.elbow
        print(f"Optimal number of clusters based on elbow method: {optimal_k_elbow}")
    except Exception as e:
        print(f"Elbow method failed: {e}, using default k")
        optimal_k_elbow = default_k

    optimal_k = max(optimal_k_silhouette, optimal_k_elbow, default_k)
    print(f"Using optimal k: {optimal_k}")

    # Perform KMeans clustering with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(embeddings)

    # Add cluster labels to dataframes
    df_result = df.copy()
    df_result["cluster"] = cluster_labels
    df_tsne["cluster"] = cluster_labels

    # Visualize clusters with t-SNE
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(
        tsne_results[:, 0],
        tsne_results[:, 1],
        c=cluster_labels,
        alpha=0.6,
        cmap="viridis",
        s=50,  # Slightly larger points for better visibility
    )

    # Add cluster number labels at centroids
    centroids = {}
    for i in range(optimal_k):
        mask = cluster_labels == i
        if np.any(mask):  # Ensure there are points in this cluster
            centroids[i] = (
                np.mean(tsne_results[mask, 0]),
                np.mean(tsne_results[mask, 1]),
            )

    for cluster_id, (x, y) in centroids.items():
        ax.annotate(
            f"{cluster_id}",
            (x, y),
            fontsize=14,
            fontweight="bold",
            ha="center",
            va="center",
            bbox=dict(boxstyle="circle,pad=0.3", fc="white", ec="black", alpha=0.8),
        )

    fig.colorbar(scatter, ax=ax, label="Cluster")
    ax.set_title(f"t-SNE Projection with {optimal_k} Clusters - Temporal Embeddings")
    plt.show()

    # Create interactive plotly visualization
    fig_plotly = px.scatter(
        df_tsne,
        x="tsne_1",
        y="tsne_2",
        color="cluster",
        hover_data=["user_id"],
        title=f"t-SNE Projection of Temporal Embeddings ({optimal_k} clusters)",
        color_continuous_scale=px.colors.qualitative.G10,
    )
    fig_plotly.update_layout(width=1000, height=800, legend_title_text="Cluster")

    # Add cluster number annotations to plotly figure
    for cluster_id, (x, y) in centroids.items():
        fig_plotly.add_annotation(
            x=x,
            y=y,
            text=f"{cluster_id}",
            showarrow=False,
            font=dict(size=16, color="black", family="Arial Black"),
            bgcolor="white",
            bordercolor="black",
            borderwidth=2,
            borderpad=4,
            opacity=0.8,
        )

    fig_plotly.show()

    # Analyze cluster characteristics
    cluster_stats = (
        df_result.groupby("cluster").agg(user_count=("user_id", "count")).reset_index()
    )

    print("\nCluster Statistics:")
    print(cluster_stats)

    # Visualize cluster sizes
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="cluster", y="user_count", data=cluster_stats, ax=ax)
    ax.set_title(f"User Count per Cluster (k={optimal_k}) - Temporal Embeddings")
    plt.show()

    return df_result, df_tsne, cluster_stats


# %%
# Perform clustering and visualization using temporal_embedding
df_clustered, df_tsne, cluster_stats = cluster_and_visualize_embeddings(
    df, embedding_column="temporal_embedding", min_k=2, max_k=10, default_k=3
)

# %%
# Display distribution of users across clusters
plt.figure(figsize=(10, 6))
cluster_counts = df_clustered["cluster"].value_counts().sort_index()
cluster_counts.plot(kind="bar")
plt.title("Number of Users in Each Cluster")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.show()
# %%


# %%
# Calculate cosine similarity between all embeddings
def calculate_cosine_similarity_matrix(df, embedding_column):
    """
    Calculate cosine similarity matrix between all embeddings

    Args:
        df: DataFrame containing the embeddings
        embedding_column: Column name containing the embeddings

    Returns:
        similarity_matrix: Matrix of cosine similarities
        user_ids: List of user IDs in the same order as the matrix
    """
    # Extract embeddings as numpy array
    embeddings = np.array(df[embedding_column].tolist())

    # Get user IDs in the same order
    user_ids = df["user_id"].values

    # Calculate cosine similarity
    print("Calculating cosine similarity matrix...")
    # Normalize embeddings for cosine similarity
    normalized_embeddings = normalize(embeddings, norm="l2")

    # Calculate dot product which gives cosine similarity for normalized vectors
    similarity_matrix = np.dot(normalized_embeddings, normalized_embeddings.T)

    return similarity_matrix, user_ids


# %%
# Calculate similarity matrix
similarity_matrix, user_ids = calculate_cosine_similarity_matrix(
    df, "temporal_embedding"
)

# %%
# Plot heatmap of the full similarity matrix
plt.figure(figsize=(12, 10))
sns.heatmap(
    similarity_matrix,
    cmap="viridis",
    xticklabels=False,
    yticklabels=False,
    vmin=0,
    vmax=1,
)
plt.title("Cosine Similarity Heatmap (All Users)")
plt.xlabel("Users")
plt.ylabel("Users")
plt.show()

# %%
# If we have too many users, sample a subset for a more detailed heatmap
max_users_for_labeled_heatmap = 50
if len(df) > max_users_for_labeled_heatmap:
    # Sample users
    sample_indices = np.random.choice(
        range(len(df)), size=max_users_for_labeled_heatmap, replace=False
    )

    # Get sampled similarity matrix
    sampled_similarity = similarity_matrix[sample_indices][:, sample_indices]
    sampled_user_ids = user_ids[sample_indices]

    plt.figure(figsize=(16, 14))
    sns.heatmap(
        sampled_similarity,
        cmap="viridis",
        xticklabels=False,
        yticklabels=False,
        vmin=0,
        vmax=1,
    )
    plt.title(
        f"Cosine Similarity Heatmap (Sample of {max_users_for_labeled_heatmap} Users)"
    )
    plt.xlabel("Users")
    plt.ylabel("Users")
    plt.show()

# %%
# Plot heatmap with cluster information
# This shows similarity patterns organized by cluster
if "cluster" in df.columns:
    # Sort by cluster
    df_sorted = df.sort_values("cluster")
    sorted_indices = df_sorted.index

    # Reorder similarity matrix by cluster
    cluster_ordered_similarity = similarity_matrix[sorted_indices, :][:, sorted_indices]

    # Get cluster boundaries for visualization
    cluster_labels = df_sorted["cluster"].values
    cluster_boundaries = np.where(np.diff(cluster_labels) != 0)[0] + 0.5

    # Plot
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cluster_ordered_similarity,
        cmap="viridis",
        xticklabels=False,
        yticklabels=False,
        vmin=0,
        vmax=1,
    )

    # Add lines to show cluster boundaries
    for boundary in cluster_boundaries:
        plt.axhline(y=boundary, color="r", linestyle="-", linewidth=1)
        plt.axvline(x=boundary, color="r", linestyle="-", linewidth=1)

    plt.title("Cosine Similarity Heatmap (Ordered by Cluster)")
    plt.xlabel("Users (ordered by cluster)")
    plt.ylabel("Users (ordered by cluster)")
    plt.show()

# %%
# Calculate and plot intra-cluster and inter-cluster similarities
if "cluster" in df.columns:
    # Get list of clusters
    clusters = sorted(df["cluster"].unique())
    n_clusters = len(clusters)

    # Calculate average similarity within and between clusters
    cluster_similarity_matrix = np.zeros((n_clusters, n_clusters))

    for i, cluster_i in enumerate(clusters):
        for j, cluster_j in enumerate(clusters):
            # Get indices for users in each cluster
            indices_i = df[df["cluster"] == cluster_i].index
            indices_j = df[df["cluster"] == cluster_j].index

            # Calculate average similarity between these clusters
            mean_sim = np.mean(similarity_matrix[np.ix_(indices_i, indices_j)])
            cluster_similarity_matrix[i, j] = mean_sim

    # Plot cluster similarity heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cluster_similarity_matrix,
        annot=True,
        fmt=".2f",
        cmap="viridis",
        xticklabels=clusters,
        yticklabels=clusters,
        vmin=0,
        vmax=1,
    )
    plt.title("Average Cosine Similarity Between Clusters")
    plt.xlabel("Cluster")
    plt.ylabel("Cluster")
    plt.show()

    # Bar plot of intra-cluster similarities
    intra_cluster_similarities = np.diag(cluster_similarity_matrix)

    plt.figure(figsize=(10, 6))
    plt.bar(range(len(clusters)), intra_cluster_similarities, tick_label=clusters)
    plt.ylim(0, 1)
    plt.axhline(y=np.mean(intra_cluster_similarities), color="r", linestyle="-")
    plt.title("Intra-cluster Similarity (Higher is Better)")
    plt.xlabel("Cluster")
    plt.ylabel("Average Cosine Similarity")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.show()

# %%
