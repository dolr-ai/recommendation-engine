# %%
import os
import json
import pandas as pd
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import pathlib
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import umap
import plotly.express as px
import plotly.graph_objects as go
from kneed import KneeLocator

# Set random seed for reproducibility
np.random.seed(42)

# Load environment variables
load_dotenv(".env")
DATA_ROOT = pathlib.Path(os.getenv("DATA_ROOT"))

# List of experiments to process
experiments = ["exp1", "exp2", "exp3"]


# Function to process a single experiment
def process_experiment(exp):
    print(f"\nProcessing experiment: {exp}")

    # Create output directories for visualizations and transformed data
    VISUALIZATION_DIR = DATA_ROOT / "visualizations" / f"ucf-{exp}"
    TRANSFORMED_DIR = DATA_ROOT / "transformed" / f"ucf-{exp}"

    # Create directories if they don't exist
    VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
    TRANSFORMED_DIR.mkdir(parents=True, exist_ok=True)

    # Load user embedding data
    df_user_avg_emb = pd.read_parquet(
        DATA_ROOT / "emb_analysis" / f"002-user_avg_item_emb-{exp}.parquet"
    )

    # Verify data structure
    print(f"Dataset shape: {df_user_avg_emb.shape}")

    # Convert embeddings to numpy arrays for processing
    embeddings = np.array(df_user_avg_emb["embedding"].tolist())
    print(f"Embedding dimension: {embeddings.shape}")

    # Perform dimensionality reduction with t-SNE
    print("Performing t-SNE dimensionality reduction...")
    tsne = TSNE(
        n_components=2, perplexity=30, n_iter=1000, random_state=42, metric="cosine"
    )
    tsne_results = tsne.fit_transform(embeddings)

    # Save t-SNE results
    df_tsne = pd.DataFrame(tsne_results, columns=["tsne_1", "tsne_2"])
    df_tsne["user_id"] = df_user_avg_emb["user_id"].values
    df_tsne.to_parquet(TRANSFORMED_DIR / "user_tsne_embeddings.parquet")

    # Visualize t-SNE projection
    plt.figure(figsize=(12, 10))
    plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5)
    plt.title(f"t-SNE Projection of User Embeddings - {exp}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.savefig(VISUALIZATION_DIR / "tsne_projection.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Determine optimal number of clusters
    silhouette_scores = []
    inertia_values = []
    k_range = range(2, 15)

    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init="auto", metric="cosine")
        cluster_labels = kmeans.fit_predict(embeddings)

        # Calculate silhouette score
        score = silhouette_score(embeddings, cluster_labels, metric="cosine")
        silhouette_scores.append(score)

        # Calculate inertia
        inertia_values.append(kmeans.inertia_)

        print(f"K={k}, Silhouette Score: {score:.4f}, Inertia: {kmeans.inertia_:.2f}")

    # Plot silhouette scores
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, silhouette_scores, "o-")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title(f"Silhouette Score for Different Cluster Counts - {exp}")
    plt.grid(True)
    plt.savefig(
        VISUALIZATION_DIR / "silhouette_scores.png", dpi=300, bbox_inches="tight"
    )
    plt.close()

    # Plot inertia values
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, inertia_values, "o-")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Inertia (Sum of Squared Distances)")
    plt.title(f"Elbow Method for Optimal k - {exp}")
    plt.grid(True)
    plt.savefig(VISUALIZATION_DIR / "elbow_method.png", dpi=300, bbox_inches="tight")
    plt.close()

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
    except:
        inertia_diffs = np.diff(inertia_values)
        elbow_idx = np.argmax(np.diff(inertia_diffs)) + 1
        optimal_k_elbow = k_range[elbow_idx]
        print(
            f"Optimal number of clusters based on simple elbow heuristic: {optimal_k_elbow}"
        )

    print(
        f"using cluster count max({optimal_k_silhouette}, {optimal_k_elbow}) = {max(optimal_k_silhouette, optimal_k_elbow)}"
    )
    optimal_k = max(optimal_k_silhouette, optimal_k_elbow)

    # Perform KMeans clustering with optimal k
    kmeans = KMeans(
        n_clusters=optimal_k, random_state=42, n_init="auto", metric="cosine"
    )
    cluster_labels = kmeans.fit_predict(embeddings)

    # Add cluster labels
    df_user_avg_emb["cluster"] = cluster_labels
    df_tsne["cluster"] = cluster_labels

    # Save clustered data
    df_user_clusters = df_user_avg_emb[["user_id", "cluster"]]
    df_user_clusters.to_parquet(TRANSFORMED_DIR / "user_clusters.parquet")

    # Visualize clusters with t-SNE
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        tsne_results[:, 0],
        tsne_results[:, 1],
        c=cluster_labels,
        alpha=0.6,
        cmap="viridis",
    )
    plt.colorbar(scatter, label="Cluster")
    plt.title(f"t-SNE Projection with {optimal_k} Clusters - {exp}")
    plt.xlabel("t-SNE Dimension 1")
    plt.ylabel("t-SNE Dimension 2")
    plt.savefig(VISUALIZATION_DIR / "tsne_clusters.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Create interactive plotly visualization
    fig = px.scatter(
        df_tsne,
        x="tsne_1",
        y="tsne_2",
        color="cluster",
        hover_data=["user_id"],
        title=f"t-SNE Projection of User Embeddings ({optimal_k} clusters) - {exp}",
        color_continuous_scale=px.colors.qualitative.G10,
    )
    fig.update_layout(width=1000, height=800, legend_title_text="Cluster")
    fig.write_html(VISUALIZATION_DIR / "interactive_tsne_clusters.html")

    # Analyze cluster characteristics
    cluster_stats = (
        df_user_avg_emb.groupby("cluster")
        .agg(
            user_count=("user_id", "count"),
        )
        .reset_index()
    )

    print("\nCluster Statistics:")
    print(cluster_stats)

    # Visualize cluster sizes
    plt.figure(figsize=(10, 6))
    sns.barplot(x="cluster", y="user_count", data=cluster_stats)
    plt.title(f"User Count per Cluster (k={optimal_k}) - {exp}")
    plt.xlabel("Cluster")
    plt.ylabel("Number of Users")
    plt.savefig(VISUALIZATION_DIR / "cluster_sizes.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Save analysis summary
    summary = {
        "experiment": exp,
        "total_users": int(len(df_user_avg_emb)),
        "embedding_dimension": int(embeddings.shape[1]),
        "optimal_clusters_silhouette": int(optimal_k_silhouette),
        "optimal_clusters_elbow": int(optimal_k_elbow),
        "optimal_clusters_used": int(optimal_k),
        "silhouette_scores": {
            int(k): float(score) for k, score in zip(k_range, silhouette_scores)
        },
        "inertia_values": {
            int(k): float(inertia) for k, inertia in zip(k_range, inertia_values)
        },
        "cluster_sizes": cluster_stats.to_dict(orient="records"),
        "timestamp": datetime.now().isoformat(),
    }

    with open(VISUALIZATION_DIR / "cluster_analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print(
        f"Cluster analysis complete for {exp}. Results saved to {VISUALIZATION_DIR} and {TRANSFORMED_DIR}"
    )
    return summary


# %%
# Process all experiments and collect summaries
all_summaries = []
for exp in experiments:
    summary = process_experiment(exp)
    all_summaries.append(summary)

# Save combined summary for all experiments
combined_summary_path = DATA_ROOT / "visualizations" / "uc-combined_summary.json"
with open(combined_summary_path, "w") as f:
    json.dump(all_summaries, f, indent=2)

print(f"\nAll experiments processed. Combined summary saved to {combined_summary_path}")
# %%
