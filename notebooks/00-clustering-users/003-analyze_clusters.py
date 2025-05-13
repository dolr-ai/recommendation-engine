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

# Create output directories for visualizations and transformed data
VISUALIZATION_DIR = DATA_ROOT / "visualizations" / "user_clusters"
TRANSFORMED_DIR = DATA_ROOT / "transformed" / "user_clusters"

# Create directories if they don't exist
VISUALIZATION_DIR.mkdir(parents=True, exist_ok=True)
TRANSFORMED_DIR.mkdir(parents=True, exist_ok=True)

# %%
# Load user embedding data
df_user_avg_emb = pd.read_parquet(
    DATA_ROOT / "emb_analysis" / "002-user_avg_item_emb.parquet"
)

# %%
# Verify data structure
print(f"Dataset shape: {df_user_avg_emb.shape}")
print(f"Columns: {df_user_avg_emb.columns}")
# The embeddings column contains vectors

# %%
# Convert embeddings to numpy arrays for processing
embeddings = np.array(df_user_avg_emb["embedding"].tolist())
print(f"Embedding dimension: {embeddings.shape}")

# %%
# Normalize embeddings
scaler = StandardScaler()
normalized_embeddings = scaler.fit_transform(embeddings)

# %%
# Perform dimensionality reduction with t-SNE
print("Performing t-SNE dimensionality reduction...")
tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)
tsne_results = tsne.fit_transform(normalized_embeddings)

# Save t-SNE results
df_tsne = pd.DataFrame(tsne_results, columns=["tsne_1", "tsne_2"])
df_tsne["user_id"] = df_user_avg_emb["user_id"].values
df_tsne.to_parquet(TRANSFORMED_DIR / "user_tsne_embeddings.parquet")

# %%
# Visualize t-SNE projection
plt.figure(figsize=(12, 10))
plt.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5)
plt.title("t-SNE Projection of User Embeddings")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.savefig(VISUALIZATION_DIR / "tsne_projection.png", dpi=300, bbox_inches="tight")
plt.close()

# %%
# Determine optimal number of clusters using silhouette score and inertia (sum of squared distances)
silhouette_scores = []
inertia_values = []
k_range = range(2, 15)

for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(normalized_embeddings)

    # Calculate silhouette score
    score = silhouette_score(normalized_embeddings, cluster_labels)
    silhouette_scores.append(score)

    # Calculate inertia (sum of squared distances)
    inertia_values.append(kmeans.inertia_)

    print(f"K={k}, Silhouette Score: {score:.4f}, Inertia: {kmeans.inertia_:.2f}")

# Plot silhouette scores
plt.figure(figsize=(10, 6))
plt.plot(k_range, silhouette_scores, "o-")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Silhouette Score")
plt.title("Silhouette Score for Different Cluster Counts")
plt.grid(True)
plt.savefig(VISUALIZATION_DIR / "silhouette_scores.png", dpi=300, bbox_inches="tight")
plt.close()

# Plot inertia values (Elbow method)
plt.figure(figsize=(10, 6))
plt.plot(k_range, inertia_values, "o-")
plt.xlabel("Number of Clusters (k)")
plt.ylabel("Inertia (Sum of Squared Distances)")
plt.title("Elbow Method for Optimal k")
plt.grid(True)
plt.savefig(VISUALIZATION_DIR / "elbow_method.png", dpi=300, bbox_inches="tight")
plt.close()

# %%
# Identify optimal k from silhouette scores
optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
print(f"Optimal number of clusters based on silhouette score: {optimal_k_silhouette}")

# Try to find the elbow point automatically
try:
    kneedle = KneeLocator(
        list(k_range), inertia_values, curve="convex", direction="decreasing"
    )
    optimal_k_elbow = kneedle.elbow
    print(f"Optimal number of clusters based on elbow method: {optimal_k_elbow}")
except:
    # If KneeLocator fails or is not available, use a simple heuristic
    # Calculate the rate of change in inertia
    inertia_diffs = np.diff(inertia_values)
    # Find where the rate of change starts to slow down significantly
    elbow_idx = np.argmax(np.diff(inertia_diffs)) + 1
    optimal_k_elbow = k_range[elbow_idx]
    print(
        f"Optimal number of clusters based on simple elbow heuristic: {optimal_k_elbow}"
    )

# Choose which method to use for optimal k
# You can comment/uncomment the preferred method
# optimal_k = optimal_k_elbow  # Using elbow method
print(
    f"using cluster count max({optimal_k_silhouette}, 5) = {max(optimal_k_silhouette, 5)}"
)
optimal_k = max(optimal_k_silhouette, 5)  # Using silhouette score

# Perform KMeans clustering with optimal k
kmeans = KMeans(
    n_clusters=optimal_k,
    random_state=42,
    n_init="auto",
)
cluster_labels = kmeans.fit_predict(normalized_embeddings)

# Add cluster labels to the dataframe
df_user_avg_emb["cluster"] = cluster_labels
df_tsne["cluster"] = cluster_labels

# Save clustered data
df_user_clusters = df_user_avg_emb[["user_id", "cluster"]]
df_user_clusters.to_parquet(TRANSFORMED_DIR / "user_clusters.parquet")

# %%
# Visualize clusters with t-SNE
plt.figure(figsize=(12, 10))
scatter = plt.scatter(
    tsne_results[:, 0], tsne_results[:, 1], c=cluster_labels, alpha=0.6, cmap="viridis"
)
plt.colorbar(scatter, label="Cluster")
plt.title(f"t-SNE Projection with {optimal_k} Clusters")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.savefig(VISUALIZATION_DIR / "tsne_clusters.png", dpi=300, bbox_inches="tight")
plt.close()

# %%
# Create an interactive plotly visualization
fig = px.scatter(
    df_tsne,
    x="tsne_1",
    y="tsne_2",
    color="cluster",
    hover_data=["user_id"],
    title=f"t-SNE Projection of User Embeddings ({optimal_k} clusters)",
    color_continuous_scale=px.colors.qualitative.G10,
)
fig.update_layout(width=1000, height=800, legend_title_text="Cluster")
fig.write_html(VISUALIZATION_DIR / "interactive_tsne_clusters.html")

# %%
# Also try UMAP dimensionality reduction for comparison
try:
    print("Performing UMAP dimensionality reduction...")
    umap_reducer = umap.UMAP(
        n_neighbors=15, min_dist=0.1, n_components=2, random_state=42
    )
    umap_results = umap_reducer.fit_transform(normalized_embeddings)

    # Save UMAP results
    df_umap = pd.DataFrame(umap_results, columns=["umap_1", "umap_2"])
    df_umap["user_id"] = df_user_avg_emb["user_id"].values
    df_umap["cluster"] = cluster_labels
    df_umap.to_parquet(TRANSFORMED_DIR / "user_umap_embeddings.parquet")

    # Visualize UMAP projection with clusters
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(
        umap_results[:, 0],
        umap_results[:, 1],
        c=cluster_labels,
        alpha=0.6,
        cmap="viridis",
    )
    plt.colorbar(scatter, label="Cluster")
    plt.title(f"UMAP Projection with {optimal_k} Clusters")
    plt.xlabel("UMAP Dimension 1")
    plt.ylabel("UMAP Dimension 2")
    plt.savefig(VISUALIZATION_DIR / "umap_clusters.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Interactive UMAP visualization
    fig = px.scatter(
        df_umap,
        x="umap_1",
        y="umap_2",
        color="cluster",
        hover_data=["user_id"],
        title=f"UMAP Projection of User Embeddings ({optimal_k} clusters)",
        color_continuous_scale=px.colors.qualitative.G10,
    )
    fig.update_layout(width=1000, height=800, legend_title_text="Cluster")
    fig.write_html(VISUALIZATION_DIR / "interactive_umap_clusters.html")
except:
    print("UMAP visualization skipped - please install umap-learn if needed")

# %%
# Analyze cluster characteristics
cluster_stats = (
    df_user_avg_emb.groupby("cluster")
    .agg(
        user_count=("user_id", "count"),
    )
    .reset_index()
)

print("Cluster Statistics:")
print(cluster_stats)

# Visualize cluster sizes
plt.figure(figsize=(10, 6))
sns.barplot(x="cluster", y="user_count", data=cluster_stats)
plt.title(f"User Count per Cluster (k={optimal_k})")
plt.xlabel("Cluster")
plt.ylabel("Number of Users")
plt.savefig(VISUALIZATION_DIR / "cluster_sizes.png", dpi=300, bbox_inches="tight")
plt.close()

# %%
# Save analysis summary
summary = {
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
    f"Cluster analysis complete. Results saved to {VISUALIZATION_DIR} and {TRANSFORMED_DIR}"
)
# %%
