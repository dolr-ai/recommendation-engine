# %%
import os
import json
import pandas as pd
import asyncio
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
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
VISUALIZATION_DIR = DATA_ROOT / "visualizations" / "merged_embeddings"
TRANSFORMED_DIR = DATA_ROOT / "transformed" / "merged_embeddings"

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
df_alg1_emb.head()

# %%
df_alg2_emb.head()
# %%
df_alg1_emb.shape, df_alg2_emb.shape
# %%
df_req = df_alg1_emb.merge(df_alg2_emb, on="user_id", how="inner")

# %%
df_req["merged_embedding"] = df_req.apply(
    lambda x: np.concatenate([x["embedding"], x["cluster_distributions"]]), axis=1
)

# Normalize the merged embeddings
embeddings_array = np.vstack(df_req["merged_embedding"])
normalized_embeddings = normalize(embeddings_array, norm="l2")
df_req["merged_embedding"] = [emb.tolist() for emb in normalized_embeddings]

# %%
df_req["merged_embedding"].apply(len).value_counts()
# %%

sns.histplot(df_req.iloc[0]["merged_embedding"])
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
    merged_embeddings = np.array(df_with_merged_embeddings["merged_embedding"].tolist())

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


# Function to run the entire analysis pipeline
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

    # Save final results
    df_clustered.to_parquet(
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
