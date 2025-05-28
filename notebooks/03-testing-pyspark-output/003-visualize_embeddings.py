# %%
import os
import json
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from dotenv import load_dotenv
import pathlib
from tqdm import tqdm
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize, StandardScaler
import plotly.express as px
import plotly.io as pio

# utils
from utils.gcp_utils import GCPUtils

# %%
# setup configs
print(load_dotenv("/Users/sagar/work/yral/recommendation-engine/.env"))

print(os.getenv("DATA_ROOT"))

DATA_ROOT = pathlib.Path(os.getenv("DATA_ROOT"))
GCP_CREDENTIALS_PATH = os.environ.get("GCP_CREDENTIALS_PATH_STAGE")

with open(GCP_CREDENTIALS_PATH, "r") as f:
    _ = json.load(f)
    gcp_credentials_str = json.dumps(_)

gcp = GCPUtils(gcp_credentials=gcp_credentials_str)
del gcp_credentials_str, _

# %%
# Load data from parquet file
df = pd.read_parquet(
    "/Users/sagar/work/yral/recommendation-engine/data-temp/master_dag_output/new_user_emb.parquet"
)

# %%
# Create different embedding types
print("Creating different embedding types...")
df["etype1"] = df["avg_interaction_embedding"]
df["etype2"] = df.apply(
    lambda x: np.concatenate(
        [x["avg_interaction_embedding"], x["cluster_distribution_embedding"]]
    ),
    axis=1,
)
df["etype3"] = df.apply(
    lambda x: np.concatenate(
        [
            x["avg_interaction_embedding"],
            x["cluster_distribution_embedding"],
            x["temporal_embedding"],
        ]
    ),
    axis=1,
)
df["etype4"] = df["user_embedding"]
df["etype5"] = df["cluster_distribution_embedding"]
df["etype6"] = df["temporal_embedding"]

etype_to_type_of_embedding = {
    "etype1": "<Avg Interaction> Embedding",
    "etype2": "<Avg Interaction, Cluster Distribution> Embedding",
    "etype3": "<Avg Interaction, Cluster Distribution, Temporal> Embedding",
    "etype4": "<User> Embedding",
    # exploration purposes
    "etype5": "<Cluster Distribution> Embedding",
    "etype6": "<Temporal> Embedding",
}


# %%
# Function to visualize embeddings with t-SNE and save 3D plots
def visualize_embeddings(
    df,
    embedding_column,
    type_of_embedding="",
    tsne_perplexity=30,
    tsne_n_iter=1000,
    output_dir="/Users/sagar/work/yral/recommendation-engine/data-temp/master_dag_output/visualizations",
):
    print("#" * 100)
    print(f"Visualizing {type_of_embedding}")
    print("#" * 100)

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Extract embeddings as numpy array
    embeddings = np.array(df[embedding_column].tolist())
    print(f"Embedding array shape: {embeddings.shape}")

    # Apply L2 normalization directly
    print("Applying L2 normalization...")
    embeddings = normalize(embeddings, norm="l2")
    print("Embeddings normalized with L2 norm")

    # Use existing cluster IDs
    cluster_labels = df["cluster_id"].values

    # Perform dimensionality reduction with 2D t-SNE
    print("Performing 2D t-SNE dimensionality reduction...")
    tsne = TSNE(
        n_components=2,
        perplexity=tsne_perplexity,
        n_iter=tsne_n_iter,
        metric="cosine",
        early_exaggeration=12,  # Help with initial clustering
    )
    tsne_results = tsne.fit_transform(embeddings)

    # Create dataframe with t-SNE results
    df_tsne = pd.DataFrame(tsne_results, columns=["tsne_1", "tsne_2"])
    df_tsne["user_id"] = df["user_id"].values
    df_tsne["cluster"] = cluster_labels

    # Plot t-SNE projection
    fig, ax = plt.subplots(figsize=(12, 10))
    scatter = ax.scatter(
        tsne_results[:, 0],
        tsne_results[:, 1],
        c=cluster_labels,
        alpha=0.6,
        cmap="viridis",
        s=50,
    )

    # Add cluster number labels at centroids
    centroids = {}
    for i in np.unique(cluster_labels):
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
    ax.set_title(f"t-SNE Projection - {type_of_embedding}")
    plt.show()

    # Create interactive plotly visualization for 2D
    fig_plotly = px.scatter(
        df_tsne,
        x="tsne_1",
        y="tsne_2",
        color="cluster",
        hover_data=["user_id"],
        title=f"t-SNE Projection of {type_of_embedding}",
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

    # Perform 3D t-SNE visualization
    print("Performing 3D t-SNE dimensionality reduction...")
    tsne_3d = TSNE(
        n_components=3,
        perplexity=tsne_perplexity,
        n_iter=tsne_n_iter,
        metric="cosine",
    )
    tsne_results_3d = tsne_3d.fit_transform(embeddings)

    # Create dataframe with 3D t-SNE results
    df_tsne_3d = pd.DataFrame(tsne_results_3d, columns=["tsne_1", "tsne_2", "tsne_3"])
    df_tsne_3d["user_id"] = df["user_id"].values
    df_tsne_3d["cluster"] = cluster_labels

    # Create 3D interactive plotly visualization
    fig_3d = px.scatter_3d(
        df_tsne_3d,
        x="tsne_1",
        y="tsne_2",
        z="tsne_3",
        color="cluster",
        hover_data=["user_id"],
        title=f"3D t-SNE Projection of {type_of_embedding}",
        color_continuous_scale=px.colors.qualitative.G10,
    )

    # Improve 3D plot layout
    fig_3d.update_layout(
        width=1000,
        height=800,
        legend_title_text="Cluster",
        scene=dict(
            xaxis_title="t-SNE 1",
            yaxis_title="t-SNE 2",
            zaxis_title="t-SNE 3",
        ),
    )

    # Make data points smaller for better visibility
    fig_3d.update_traces(
        marker=dict(size=3, opacity=0.7),
        selector=dict(mode="markers"),
    )

    # Add cluster centroids in 3D space
    centroids_3d = {}
    for i in np.unique(cluster_labels):
        mask = cluster_labels == i
        if np.any(mask):
            centroids_3d[i] = (
                np.mean(tsne_results_3d[mask, 0]),
                np.mean(tsne_results_3d[mask, 1]),
                np.mean(tsne_results_3d[mask, 2]),
            )

    # Add centroids as larger markers
    centroid_x = [c[0] for c in centroids_3d.values()]
    centroid_y = [c[1] for c in centroids_3d.values()]
    centroid_z = [c[2] for c in centroids_3d.values()]
    centroid_labels = list(centroids_3d.keys())

    fig_3d.add_trace(
        px.scatter_3d(
            x=centroid_x,
            y=centroid_y,
            z=centroid_z,
            text=centroid_labels,
            color_discrete_sequence=["black"],
        ).data[0]
    )

    # Make centroids larger and more visible
    fig_3d.data[-1].marker.size = 12
    fig_3d.data[-1].marker.symbol = "circle"
    fig_3d.data[-1].marker.opacity = 1.0
    fig_3d.data[-1].marker.line = dict(width=2, color="white")

    fig_3d.show()

    # Save the 3D interactive plot
    output_path = os.path.join(output_dir, f"{embedding_column}_3d_tsne.html")
    pio.write_html(fig_3d, output_path)
    print(f"Saved 3D plot to {output_path}")

    # Analyze cluster characteristics
    cluster_stats = pd.DataFrame(
        {
            "cluster": np.unique(cluster_labels),
            "user_count": [
                np.sum(cluster_labels == i) for i in np.unique(cluster_labels)
            ],
        }
    )

    print("\nCluster Statistics:")
    print(cluster_stats)

    # Visualize cluster sizes
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="cluster", y="user_count", data=cluster_stats, ax=ax)
    ax.set_title(f"User Count per Cluster - {type_of_embedding}")
    plt.show()

    return df_tsne, df_tsne_3d, cluster_stats


# %%
# Visualize all embedding types
visualization_results = {}

for etype in ["etype1", "etype2", "etype3", "etype4", "etype5", "etype6"]:
    print(f"\nProcessing {etype}: {etype_to_type_of_embedding[etype]}")
    df_tsne, df_tsne_3d, cluster_stats = visualize_embeddings(
        df,
        embedding_column=etype,
        type_of_embedding=etype_to_type_of_embedding[etype],
        tsne_perplexity=30,
        tsne_n_iter=1000,
    )

    visualization_results[etype] = {
        "df_tsne": df_tsne,
        "df_tsne_3d": df_tsne_3d,
        "cluster_stats": cluster_stats,
    }

# %%
# Save visualization results
pd.to_pickle(
    visualization_results,
    "/Users/sagar/work/yral/recommendation-engine/data-temp/master_dag_output/embedding_visualization_results.pkl",
)

# %%
print("Visualization complete. 3D interactive plots have been saved.")
