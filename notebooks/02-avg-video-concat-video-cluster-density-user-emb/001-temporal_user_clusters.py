# %%
import os
import json
import pandas as pd
import asyncio
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

# Set output directories for this notebook
VISUALIZATION_DIR = DATA_ROOT / "visualizations" / "temporal_merged_embeddings"
TRANSFORMED_DIR = DATA_ROOT / "transformed" / "temporal_merged_embeddings"

# Create directories if they don't exist
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
TRANSFORMED_DIR.mkdir(parents=True, exist_ok=True)

# %%
algorithm_1_dir = DATA_ROOT / "emb_analysis"
df_alg1_emb = pd.read_parquet(algorithm_1_dir / "002-user_avg_item_emb-exp1.parquet")

# %%
algorithm_2_dir = (
    DATA_ROOT / "transformed" / "user_clusters_by_video_cluster_distribution"
)
df_alg2_emb = pd.read_parquet(
    algorithm_2_dir / "user_clusters_with_distributions.parquet"
)


# %%
def rope_inspired_encoding(time_counts, dim=16):
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

    return features


def create_temporal_cluster_distribution(
    engagement_data,
    min_timestamp,
    max_timestamp,
    num_bins=16,
    rope_dim=16,
):
    """
    Create a temporal distribution of cluster accesses across equal time bins.

    Args:
        engagement_data: List of dictionaries containing cluster_label and last_watched_timestamp
        min_timestamp: The minimum timestamp in the dataset
        max_timestamp: The maximum timestamp in the dataset
        num_bins: Number of equal time bins to create (default: 16)

    Returns:
        A dictionary with cluster labels as keys and lists of counts per time bin as values
    """
    # Calculate time range and bin width in days
    total_time_range = (max_timestamp - min_timestamp).total_seconds() / (24 * 3600)
    bin_width_days = total_time_range / num_bins

    # Initialize the cluster distribution dictionary
    # Keys are cluster labels, values are lists with counts for each time bin
    cluster_distribution = {}

    # Process each engagement record
    for record in engagement_data:
        cluster_label = record["cluster_label"]
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


start_date = datetime(2025, 2, 1, tzinfo=timezone.utc)
end_date = datetime(2025, 4, 30, tzinfo=timezone.utc)

res1 = create_temporal_cluster_distribution(
    df_alg2_emb["engagement_metadata_list"].iloc[0],
    min_timestamp=start_date,
    max_timestamp=end_date,
    num_bins=16,
)

res2 = rope_inspired_encoding(res1)

df_alg2_emb["temporal_cluster_distribution"] = df_alg2_emb[
    "engagement_metadata_list"
].apply(
    lambda x: create_temporal_cluster_distribution(
        x, min_timestamp=start_date, max_timestamp=end_date, num_bins=16
    )
)

df_alg2_emb["temporal_cluster_distribution_rope"] = df_alg2_emb[
    "temporal_cluster_distribution"
].apply(rope_inspired_encoding)
# %%
df_alg2_emb["engagement_metadata_list"].iloc[0]
# %%
res1
# output
# {1: [0, 0, 0, 0, 13, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  5: [0, 0, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
#  7: [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]}

# %%
res2
# output
# array([1.62711235e-01, 3.06241358e-01, 5.43108956e-02, 3.49650722e-01,
#        1.72668491e-02, 3.54135636e-01, 5.46317989e-03, 3.54585587e-01,
#        1.72770162e-03, 3.54630597e-01, 5.46350147e-04, 3.54635098e-01,
#        1.72771179e-04, 3.54635549e-01, 5.46350468e-05, 3.54635594e-01])
# %%
df_alg2_emb["temporal_cluster_distribution_rope"].iloc[0]


# %%
df_alg1_emb.head()

# %%
df_alg2_emb.head()
# %%
df_alg1_emb.shape, df_alg2_emb.shape
# %%
df_req = df_alg1_emb.merge(df_alg2_emb, on="user_id", how="inner")

# %%
df_req["temporal_merged_embedding"] = df_req.apply(
    lambda x: np.concatenate(
        [
            x["embedding"],
            x["cluster_distributions"],
            x["temporal_cluster_distribution_rope"],
        ]
    ),
    axis=1,
)

# Normalize the merged embeddings
embeddings_array = np.vstack(df_req["temporal_merged_embedding"])
normalized_embeddings = normalize(embeddings_array, norm="l2")
df_req["temporal_merged_embedding"] = [emb.tolist() for emb in normalized_embeddings]

# %%
df_req["temporal_merged_embedding"].apply(len).value_counts()
# %%

sns.histplot(df_req.iloc[0]["temporal_merged_embedding"])
# %%
sns.histplot(df_req.iloc[0]["embedding"])
# %%


def _create_directories(visualization_dir, transformed_dir):
    """Create directories if they don't exist."""
    visualization_dir.mkdir(parents=True, exist_ok=True)
    transformed_dir.mkdir(parents=True, exist_ok=True)


def _save_plot(fig, path, title=None):
    """Save a matplotlib figure to a file."""
    if title:
        plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_json(data, path):
    """Save data as JSON."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def cluster_users_with_merged_embeddings(
    df_with_merged_embeddings: pd.DataFrame,
    visualization_dir: pathlib.Path,
    transformed_dir: pathlib.Path,
    tsne_perplexity: int = 30,
    tsne_n_iter: int = 1000,
    min_k_user: int = 2,
    max_k_user: int = 15,
    default_optimal_k_user: int = 3,
):
    """
    Performs clustering on merged user embeddings.

    Args:
        df_with_merged_embeddings: DataFrame with user_id and merged_embedding.
        visualization_dir: Path to save visualizations.
        transformed_dir: Path to save transformed data.
        tsne_perplexity: Perplexity for t-SNE.
        tsne_n_iter: Number of iterations for t-SNE.
        min_k_user: Minimum number of user clusters.
        max_k_user: Maximum number of user clusters.
        default_optimal_k_user: Default optimal k for users.

    Returns:
        DataFrame with user_id and assigned user cluster labels.
    """
    print("Starting user clustering with merged embeddings...")
    _create_directories(visualization_dir, transformed_dir)

    # Extract embeddings as numpy array
    merged_embeddings = np.array(
        df_with_merged_embeddings["temporal_merged_embedding"].tolist()
    )

    print(f"Merged embedding shape: {merged_embeddings.shape}")

    # Create dictionary to store all analysis information
    summary = {
        "total_users": int(len(df_with_merged_embeddings)),
        "embedding_dimension": int(merged_embeddings.shape[1]),
        "timestamp": datetime.now().isoformat(),
    }

    # Perform dimensionality reduction with t-SNE
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(
        n_components=2,
        perplexity=tsne_perplexity,
        n_iter=tsne_n_iter,
        random_state=42,
        metric="cosine",
    )
    tsne_results = tsne.fit_transform(merged_embeddings)

    # Create dataframe with t-SNE results
    df_tsne = pd.DataFrame(tsne_results, columns=["tsne_1", "tsne_2"])
    df_tsne["user_id"] = df_with_merged_embeddings["user_id"].values

    # Save t-SNE results
    df_tsne.to_parquet(transformed_dir / "user_tsne_merged_embeddings.parquet")

    # Visualize t-SNE projection
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5)
    _save_plot(
        fig,
        visualization_dir / "tsne_projection_merged.png",
        "t-SNE Projection of Merged User Embeddings",
    )

    # Determine optimal number of clusters
    silhouette_scores = []
    inertia_values = []
    k_range = range(min_k_user, max_k_user)

    print("Determining optimal k for merged user clusters...")
    for k in tqdm(k_range, desc="K-Means"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto")
        cluster_labels = kmeans.fit_predict(merged_embeddings)

        if (
            len(np.unique(cluster_labels)) > 1
        ):  # Silhouette score requires at least 2 labels
            score = silhouette_score(merged_embeddings, cluster_labels, metric="cosine")
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
    ax.grid(True)
    _save_plot(
        fig,
        visualization_dir / "silhouette_scores_merged.png",
        "Silhouette Score for Different Cluster Counts - Merged Embeddings",
    )

    # Plot inertia values
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_range, inertia_values, "o-")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia (Sum of Squared Distances)")
    ax.grid(True)
    _save_plot(
        fig,
        visualization_dir / "elbow_method_merged.png",
        "Elbow Method for Optimal k - Merged Embeddings",
    )

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
        optimal_k_elbow = default_optimal_k_user

    optimal_k = max(optimal_k_silhouette, optimal_k_elbow, default_optimal_k_user)
    print(f"Using optimal k: {optimal_k}")

    # Perform KMeans clustering with optimal k
    kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(merged_embeddings)

    # Add cluster labels to dataframes
    df_with_merged_embeddings["cluster"] = cluster_labels
    df_tsne["cluster"] = cluster_labels

    # Save clustered data
    df_user_clusters = df_with_merged_embeddings[["user_id", "cluster"]]
    df_user_clusters.to_parquet(transformed_dir / "user_clusters_merged.parquet")
    df_tsne.to_parquet(transformed_dir / "user_tsne_with_clusters.parquet")

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
    _save_plot(
        fig,
        visualization_dir / "tsne_clusters_merged.png",
        f"t-SNE Projection with {optimal_k} Clusters - Merged Embeddings",
    )

    # Create interactive plotly visualization
    fig_plotly = px.scatter(
        df_tsne,
        x="tsne_1",
        y="tsne_2",
        color="cluster",
        hover_data=["user_id"],
        title=f"t-SNE Projection of Merged User Embeddings ({optimal_k} clusters)",
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

    fig_plotly.write_html(visualization_dir / "interactive_tsne_clusters_merged.html")

    # Analyze cluster characteristics
    cluster_stats = (
        df_with_merged_embeddings.groupby("cluster")
        .agg(user_count=("user_id", "count"))
        .reset_index()
    )

    print("\nCluster Statistics:")
    print(cluster_stats)

    # Visualize cluster sizes
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="cluster", y="user_count", data=cluster_stats, ax=ax)
    _save_plot(
        fig,
        visualization_dir / "cluster_sizes_merged.png",
        f"User Count per Cluster (k={optimal_k}) - Merged Embeddings",
    )

    # Save analysis summary
    summary.update(
        {
            "optimal_clusters_silhouette": int(optimal_k_silhouette),
            "optimal_clusters_elbow": int(optimal_k_elbow) if optimal_k_elbow else None,
            "optimal_clusters_used": int(optimal_k),
            "silhouette_scores": {
                int(k): float(s) for k, s in zip(k_range, silhouette_scores)
            },
            "inertia_values": {
                int(k): float(i) for k, i in zip(k_range, inertia_values)
            },
            "cluster_sizes": cluster_stats.to_dict(orient="records"),
        }
    )

    _save_json(summary, visualization_dir / "merged_cluster_analysis_summary.json")
    print(
        f"User cluster analysis complete. Results saved to {visualization_dir} and {transformed_dir}"
    )

    return df_with_merged_embeddings


def convert_int_keys_to_str(data_dict):
    """
    Convert integer keys in a dictionary to strings to make it serializable.

    Args:
        data_dict: Dictionary with potential integer keys

    Returns:
        Dictionary with string keys
    """
    if not isinstance(data_dict, dict):
        return data_dict

    return {str(k): v for k, v in data_dict.items()}


# Fix the temporal_cluster_distribution column for serialization
def prepare_dataframe_for_serialization(df):
    """
    Prepare a dataframe for serialization by converting integer keys in dictionaries to strings.

    Args:
        df: DataFrame to prepare

    Returns:
        DataFrame with serializable columns
    """
    df_copy = df.copy()

    # Convert integer keys to strings in temporal_cluster_distribution
    if "temporal_cluster_distribution" in df_copy.columns:
        df_copy["temporal_cluster_distribution"] = df_copy[
            "temporal_cluster_distribution"
        ].apply(convert_int_keys_to_str)

    return df_copy


# Function to run the entire analysis pipeline with serialization fix
def run_merged_embedding_analysis(df_merged=None):
    """
    Run the complete merged embedding analysis pipeline.

    Args:
        df_merged: DataFrame with merged embeddings. If None, uses the global df_req.

    Returns:
        DataFrame with user_id, merged_embeddings, and cluster assignments.
    """
    if df_merged is None:
        df_merged = df_req

    # Run clustering
    df_clustered = cluster_users_with_merged_embeddings(
        df_merged,
        VISUALIZATION_DIR,
        TRANSFORMED_DIR,
        tsne_perplexity=30,
        tsne_n_iter=1000,
        min_k_user=2,
        max_k_user=15,
        default_optimal_k_user=3,
    )

    # Prepare for serialization
    df_for_saving = prepare_dataframe_for_serialization(df_clustered)

    # Save final results
    df_for_saving.to_parquet(
        TRANSFORMED_DIR / "user_merged_embeddings_with_clusters.parquet"
    )

    print("Merged embedding analysis complete!")
    return df_clustered


# %%

# Run the merged embedding analysis
df_clustered = run_merged_embedding_analysis()
print(
    f"Analysis complete. Found {df_clustered['cluster'].nunique()} clusters among {len(df_clustered)} users."
)
print(f"Results saved to:")
print(f"- Visualizations: {VISUALIZATION_DIR}")
print(f"- Transformed data: {TRANSFORMED_DIR}")

# %%
# Display cluster distribution
cluster_counts = df_clustered["cluster"].value_counts().sort_index()
print("Cluster distribution:")
print(cluster_counts)

# Plot cluster distribution
plt.figure(figsize=(10, 6))
cluster_counts.plot(kind="bar")
plt.title("Number of Users in Each Cluster")
plt.xlabel("Cluster")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(
    VISUALIZATION_DIR / "cluster_distribution.png", dpi=300, bbox_inches="tight"
)
plt.close()

# %%

import pandas as pd

df_temp = pd.read_parquet(
    "/Users/sagar/work/yral/recommendation-engine/data-3-month/transformed/temporal_merged_embeddings/user_merged_embeddings_with_clusters.parquet"
)

# %%
df_temp.iloc[0]
