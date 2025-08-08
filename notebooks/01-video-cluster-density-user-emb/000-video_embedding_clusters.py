# %%
import os
import json
import pandas as pd
import asyncio
import numpy as np
from datetime import datetime
from dotenv import load_dotenv
import pathlib
from tqdm import tqdm

# Import the necessary libraries for clustering and visualization
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score
import plotly.express as px
from kneed import KneeLocator

# Set random seed for reproducibility
np.random.seed(42)

# Load environment variables
load_dotenv(".env")
DEFAULT_DATA_ROOT = pathlib.Path(os.getenv("DATA_ROOT"))


def _create_directories(vis_dir, trans_dir):
    """Helper function to create directories."""
    vis_dir.mkdir(parents=True, exist_ok=True)
    trans_dir.mkdir(parents=True, exist_ok=True)


def _save_plot(fig, path, title):
    """Helper function to save matplotlib plots."""
    plt.title(title)
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def _save_json(data, path):
    """Helper function to save data to a JSON file."""
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def cluster_videos(
    video_embeddings_df: pd.DataFrame,
    visualization_dir: pathlib.Path,
    transformed_dir: pathlib.Path,
    tsne_perplexity: int = 30,
    tsne_n_iter: int = 1000,
    min_k_video: int = 2,
    max_k_video: int = 20,
    default_optimal_k_video: int = 5,
):
    """
    Performs clustering on video embeddings.

    Args:
        video_embeddings_df: DataFrame with video_id and embedding.
        visualization_dir: Path to save visualizations.
        transformed_dir: Path to save transformed data.
        tsne_perplexity: Perplexity for t-SNE.
        tsne_n_iter: Number of iterations for t-SNE.
        min_k_video: Minimum number of clusters to test.
        max_k_video: Maximum number of clusters to test.
        default_optimal_k_video: Default optimal k if not found.

    Returns:
        Tuple: (video_cluster_map_df, video_tsne_df, optimal_k_video)
    """
    print("Starting video clustering...")
    _create_directories(visualization_dir, transformed_dir)

    embeddings = np.array(video_embeddings_df["embedding"].tolist())
    print(f"Video embedding dimension: {embeddings.shape}")

    # Normalize embeddings
    scaler = StandardScaler()
    normalized_embeddings = scaler.fit_transform(embeddings)

    # Perform dimensionality reduction with t-SNE
    print("Performing t-SNE on video embeddings...")
    tsne = TSNE(
        n_components=2,
        perplexity=tsne_perplexity,
        n_iter=tsne_n_iter,
        random_state=42,
    )
    tsne_results = tsne.fit_transform(normalized_embeddings)

    df_tsne = pd.DataFrame(tsne_results, columns=["tsne_1", "tsne_2"])
    df_tsne["video_id"] = video_embeddings_df["video_id"].values
    df_tsne.to_parquet(transformed_dir / "video_tsne_embeddings.parquet")

    # Visualize t-SNE projection
    fig, ax = plt.subplots(figsize=(12, 10))
    ax.scatter(tsne_results[:, 0], tsne_results[:, 1], alpha=0.5)
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    _save_plot(
        fig,
        visualization_dir / "video_tsne_projection.png",
        "t-SNE Projection of Video Embeddings",
    )

    # Determine optimal number of clusters
    silhouette_scores = []
    inertia_values = []
    k_range = range(min_k_video, max_k_video)

    print("Determining optimal k for video clusters...")
    for k in tqdm(k_range, desc="Video K-Means"):
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(normalized_embeddings)
        silhouette_scores.append(
            silhouette_score(normalized_embeddings, cluster_labels)
        )
        inertia_values.append(kmeans.inertia_)

    # Plot silhouette scores
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_range, silhouette_scores, "o-")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.grid(True)
    _save_plot(
        fig,
        visualization_dir / "video_silhouette_scores.png",
        "Silhouette Score for Video Clusters",
    )

    # Plot inertia values (Elbow method)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_range, inertia_values, "o-")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia")
    ax.grid(True)
    _save_plot(
        fig,
        visualization_dir / "video_elbow_method.png",
        "Elbow Method for Optimal k (Videos)",
    )

    optimal_k_silhouette = k_range[np.argmax(silhouette_scores)]
    print(f"Optimal video k (Silhouette): {optimal_k_silhouette}")
    try:
        kneedle = KneeLocator(
            list(k_range), inertia_values, curve="convex", direction="decreasing"
        )
        optimal_k_elbow = kneedle.elbow
        print(f"Optimal video k (Elbow): {optimal_k_elbow}")
    except Exception:
        optimal_k_elbow = default_optimal_k_video
        print(f"Elbow method failed, using default k: {optimal_k_elbow}")

    optimal_k_video = max(
        optimal_k_silhouette, optimal_k_elbow, default_optimal_k_video
    )
    print(f"Using optimal_k_video: {optimal_k_video}")

    kmeans = KMeans(n_clusters=optimal_k_video, random_state=42, n_init="auto")
    cluster_labels = kmeans.fit_predict(normalized_embeddings)

    video_embeddings_df["cluster"] = cluster_labels
    df_tsne["cluster"] = cluster_labels

    df_video_clusters = video_embeddings_df[["video_id", "cluster"]]
    df_video_clusters.to_parquet(transformed_dir / "video_clusters.parquet")

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
    ax.set_xlabel("t-SNE Dimension 1")
    ax.set_ylabel("t-SNE Dimension 2")
    _save_plot(
        fig,
        visualization_dir / "video_tsne_clusters.png",
        f"t-SNE Projection with {optimal_k_video} Video Clusters",
    )

    fig_plotly = px.scatter(
        df_tsne,
        x="tsne_1",
        y="tsne_2",
        color="cluster",
        hover_data=["video_id"],
        title=f"Interactive t-SNE of Video Embeddings ({optimal_k_video} clusters)",
        color_continuous_scale=px.colors.qualitative.G10,
    )
    fig_plotly.update_layout(width=1000, height=800, legend_title_text="Cluster")
    fig_plotly.write_html(visualization_dir / "interactive_video_tsne_clusters.html")

    cluster_stats = (
        video_embeddings_df.groupby("cluster")
        .agg(video_count=("video_id", "count"))
        .reset_index()
    )
    print("Video Cluster Statistics:\n", cluster_stats)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x="cluster", y="video_count", data=cluster_stats, ax=ax)
    ax.set_xlabel("Cluster")
    ax.set_ylabel("Number of Videos")
    _save_plot(
        fig,
        visualization_dir / "video_cluster_sizes.png",
        f"Video Count per Cluster (k={optimal_k_video})",
    )

    summary = {
        "total_videos": int(len(video_embeddings_df)),
        "embedding_dimension": int(embeddings.shape[1]),
        "optimal_clusters_silhouette": int(optimal_k_silhouette),
        "optimal_clusters_elbow": int(optimal_k_elbow) if optimal_k_elbow else None,
        "optimal_clusters_used": int(optimal_k_video),
        "silhouette_scores": {
            int(k): float(s) for k, s in zip(k_range, silhouette_scores)
        },
        "inertia_values": {int(k): float(i) for k, i in zip(k_range, inertia_values)},
        "cluster_sizes": cluster_stats.to_dict(orient="records"),
        "timestamp": datetime.now().isoformat(),
    }
    _save_json(summary, visualization_dir / "video_cluster_analysis_summary.json")
    print(
        f"Video cluster analysis complete. Results saved to {visualization_dir} and {transformed_dir}"
    )
    return df_video_clusters, df_tsne, optimal_k_video


def calculate_user_cluster_distributions(
    df_all_data: pd.DataFrame,
    video_cluster_map_df: pd.DataFrame,
    transformed_dir: pathlib.Path,
):
    """
    Calculates user cluster distributions based on video engagement.

    Args:
        df_all_data: DataFrame with user_id, video_id, and engagement metadata.
        video_cluster_map_df: DataFrame mapping video_id to cluster.
        transformed_dir: Path to save transformed data.

    Returns:
        DataFrame with user_id and their video cluster distributions.
    """
    print("Calculating user cluster distributions...")
    video_cluster_label_map = dict(
        zip(video_cluster_map_df["video_id"], video_cluster_map_df["cluster"])
    )
    df_all_data["cluster"] = df_all_data["video_id"].map(video_cluster_label_map)

    # Create engagement metadata, ensuring 'cluster' is serializable if it's a numpy type
    df_all_data["engagement_metadata"] = df_all_data.apply(
        lambda x: {
            "video_id": x["video_id"],
            "last_watched_timestamp": x["last_watched_timestamp"],
            "mean_percentage_watched": x["mean_percentage_watched"],
            "cluster_label": (
                int(x["cluster"]) if pd.notna(x["cluster"]) else None
            ),  # Ensure int
        },
        axis=1,
    )

    df_user_engagement = (
        df_all_data.groupby("user_id")
        .agg(engagement_metadata_list=("engagement_metadata", list))
        .reset_index()
    )
    df_user_engagement.to_parquet(
        transformed_dir / "user_engagement_metadata_with_video_clusters.parquet"
    )

    cluster_min = int(video_cluster_map_df.cluster.min())  # Ensure int
    cluster_max = int(video_cluster_map_df.cluster.max())  # Ensure int

    def get_cluster_distribution(engagement_list):
        cluster_counts = {}
        for engagement in engagement_list:
            cluster = engagement.get("cluster_label")  # Use .get for safety
            if cluster is not None:  # Check if cluster is not None
                cluster_counts[cluster] = cluster_counts.get(cluster, 0) + 1

        total_videos = sum(cluster_counts.values())
        distribution = [
            cluster_counts.get(i, 0) / total_videos if total_videos > 0 else 0
            for i in range(cluster_min, cluster_max + 1)
        ]
        return distribution

    df_user_engagement["cluster_distributions"] = df_user_engagement[
        "engagement_metadata_list"
    ].apply(get_cluster_distribution)
    df_user_engagement.to_parquet(
        transformed_dir
        / "user_engagement_metadata_with_video_clusters_distribution.parquet"
    )
    print("User cluster distributions calculated.")
    return df_user_engagement


def cluster_users(
    df_user_engagement_with_dist: pd.DataFrame,
    visualization_dir: pathlib.Path,
    transformed_dir: pathlib.Path,
    tsne_perplexity: int = 30,
    tsne_n_iter: int = 1000,
    min_k_user: int = 2,
    max_k_user: int = 15,
    default_optimal_k_user: int = 3,
    process_normalized: bool = True,
    process_non_normalized: bool = True,
):
    """
    Performs clustering on user cluster distributions.

    Args:
        df_user_engagement_with_dist: DataFrame with user_id and cluster_distributions.
        visualization_dir: Path to save visualizations.
        transformed_dir: Path to save transformed data.
        tsne_perplexity: Perplexity for t-SNE.
        tsne_n_iter: Number of iterations for t-SNE.
        min_k_user: Minimum number of user clusters.
        max_k_user: Maximum number of user clusters.
        default_optimal_k_user: Default optimal k for users.
        process_normalized: Whether to process normalized distributions.
        process_non_normalized: Whether to process non-normalized distributions.

    Returns:
        DataFrame with user_id and assigned user cluster labels.
    """
    print("Starting user clustering...")
    _create_directories(visualization_dir, transformed_dir)
    user_distributions_raw = np.array(
        df_user_engagement_with_dist["cluster_distributions"].tolist()
    )

    all_user_cluster_summaries = {
        "total_users": int(len(df_user_engagement_with_dist)),
        "distribution_dimension": int(user_distributions_raw.shape[1]),
        "timestamp": datetime.now().isoformat(),
    }

    processing_approaches = []
    if process_non_normalized:
        processing_approaches.append(
            {"name": "non_normalized", "data": user_distributions_raw, "scaler": None}
        )
    if process_normalized:
        scaler_user = StandardScaler()
        normalized_user_distributions = scaler_user.fit_transform(
            user_distributions_raw
        )
        processing_approaches.append(
            {
                "name": "normalized",
                "data": normalized_user_distributions,
                "scaler": scaler_user,
            }
        )

    for approach in processing_approaches:
        name = approach["name"]
        user_distributions = approach["data"]
        print(
            f"Processing {name} user distributions... Shape: {user_distributions.shape}"
        )

        # Perform t-SNE
        tsne_user = TSNE(
            n_components=2,
            perplexity=tsne_perplexity,
            n_iter=tsne_n_iter,
            random_state=42,
        )
        tsne_user_results = tsne_user.fit_transform(user_distributions)

        df_user_tsne = pd.DataFrame(tsne_user_results, columns=["tsne_1", "tsne_2"])
        df_user_tsne["user_id"] = df_user_engagement_with_dist["user_id"].values
        df_user_tsne.to_parquet(transformed_dir / f"user_tsne_{name}.parquet")

        # Determine optimal number of clusters
        silhouette_scores_user = []
        inertia_values_user = []
        k_range_user = range(min_k_user, max_k_user)

        print(f"Determining optimal k for {name} user clusters...")
        for k in tqdm(k_range_user, desc=f"User K-Means ({name})"):
            kmeans_user = KMeans(n_clusters=k, random_state=42, n_init="auto")
            user_cluster_labels_temp = kmeans_user.fit_predict(user_distributions)
            if (
                len(np.unique(user_cluster_labels_temp)) > 1
            ):  # Silhouette score requires at least 2 labels
                score = silhouette_score(user_distributions, user_cluster_labels_temp)
                silhouette_scores_user.append(score)
            else:
                silhouette_scores_user.append(
                    -1
                )  # Assign a low score if only one cluster is formed
            inertia_values_user.append(kmeans_user.inertia_)

        optimal_k_user_silhouette = (
            k_range_user[np.argmax(silhouette_scores_user)]
            if silhouette_scores_user
            else default_optimal_k_user
        )
        print(f"Optimal user k ({name}, Silhouette): {optimal_k_user_silhouette}")
        try:
            kneedle_user = KneeLocator(
                list(k_range_user),
                inertia_values_user,
                curve="convex",
                direction="decreasing",
            )
            optimal_k_user_elbow = kneedle_user.elbow
            print(f"Optimal user k ({name}, Elbow): {optimal_k_user_elbow}")
        except Exception:
            optimal_k_user_elbow = default_optimal_k_user
            print(
                f"Elbow method failed for {name}, using default k: {optimal_k_user_elbow}"
            )

        optimal_k_user = max(
            optimal_k_user_silhouette, optimal_k_user_elbow, default_optimal_k_user
        )
        print(f"Using optimal_k_user for {name}: {optimal_k_user}")

        # Perform KMeans clustering
        kmeans_user = KMeans(n_clusters=optimal_k_user, random_state=42, n_init="auto")
        user_cluster_labels = kmeans_user.fit_predict(user_distributions)

        df_user_engagement_with_dist[f"user_cluster_{name}"] = user_cluster_labels
        df_user_tsne["cluster"] = user_cluster_labels

        # Visualize user clusters with t-SNE
        fig, ax = plt.subplots(figsize=(12, 10))
        scatter = ax.scatter(
            tsne_user_results[:, 0],
            tsne_user_results[:, 1],
            c=user_cluster_labels,
            alpha=0.6,
            cmap="viridis",
        )
        fig.colorbar(scatter, ax=ax, label="User Cluster")
        _save_plot(
            fig,
            visualization_dir / f"tsne_user_clusters_{name}.png",
            f"t-SNE User Distributions - {name} ({optimal_k_user} clusters)",
        )

        fig_plotly_user = px.scatter(
            df_user_tsne,
            x="tsne_1",
            y="tsne_2",
            color="cluster",
            hover_data=["user_id"],
            title=f"Interactive t-SNE User Distributions - {name} ({optimal_k_user} clusters)",
            color_continuous_scale=px.colors.qualitative.G10,
        )
        fig_plotly_user.update_layout(
            width=1000, height=800, legend_title_text="User Cluster"
        )
        fig_plotly_user.write_html(
            visualization_dir / f"interactive_tsne_user_clusters_{name}.html"
        )

        user_cluster_stats = (
            df_user_engagement_with_dist.groupby(f"user_cluster_{name}")
            .agg(user_count=("user_id", "count"))
            .reset_index()
        )
        print(f"User Cluster Statistics ({name}):\n", user_cluster_stats)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(
            x=f"user_cluster_{name}", y="user_count", data=user_cluster_stats, ax=ax
        )
        _save_plot(
            fig,
            visualization_dir / f"user_cluster_sizes_{name}.png",
            f"User Count per Cluster - {name} (k={optimal_k_user})",
        )

        all_user_cluster_summaries[name] = {
            "optimal_clusters_silhouette": int(optimal_k_user_silhouette),
            "optimal_clusters_elbow": (
                int(optimal_k_user_elbow) if optimal_k_user_elbow else None
            ),
            "optimal_clusters_used": int(optimal_k_user),
            "silhouette_scores": {
                int(k): float(s) for k, s in zip(k_range_user, silhouette_scores_user)
            },
            "inertia_values": {
                int(k): float(i) for k, i in zip(k_range_user, inertia_values_user)
            },
            "cluster_sizes": user_cluster_stats.to_dict(orient="records"),
        }

    df_user_engagement_with_dist.to_parquet(
        transformed_dir / "user_clusters_with_distributions.parquet"
    )
    _save_json(
        all_user_cluster_summaries,
        visualization_dir / "user_cluster_analysis_summary.json",
    )
    print(
        f"User cluster analysis complete. Results saved to {visualization_dir} and {transformed_dir}"
    )
    return df_user_engagement_with_dist


def main_video_user_clustering(
    data_root: pathlib.Path = DEFAULT_DATA_ROOT,
    input_parquet_path: str = "emb_analysis/user_item_emb.parquet",
    video_cluster_vis_subdir: str = "video_cluster_based_engagements",
    video_cluster_trans_subdir: str = "video_cluster_based_engagements",
    user_cluster_vis_subdir: str = "user_clusters_by_video_cluster_distribution",
    user_cluster_trans_subdir: str = "user_clusters_by_video_cluster_distribution",
    # Video clustering params
    video_tsne_perplexity: int = 30,
    video_tsne_n_iter: int = 1000,
    video_min_k: int = 2,
    video_max_k: int = 20,
    video_default_optimal_k: int = 8,  # based on original optimal_k_elbow
    # User clustering params
    user_tsne_perplexity: int = 30,
    user_tsne_n_iter: int = 1000,
    user_min_k: int = 2,
    user_max_k: int = 15,
    user_default_optimal_k: int = 3,
    run_user_clustering_normalized: bool = True,
    run_user_clustering_non_normalized: bool = True,
):
    """
    Main orchestrator for video and user clustering pipeline.
    """
    # Define paths
    video_visualization_dir = data_root / "visualizations" / video_cluster_vis_subdir
    video_transformed_dir = data_root / "transformed" / video_cluster_trans_subdir
    user_visualization_dir = data_root / "visualizations" / user_cluster_vis_subdir
    user_transformed_dir = data_root / "transformed" / user_cluster_trans_subdir

    # Load initial data
    print(f"Loading data from {data_root / input_parquet_path}")
    df_input = pd.read_parquet(data_root / input_parquet_path)
    assert (
        df_input["embedding"].isna().sum() == 0
    ), "Embeddings should not contain NaN values"

    # Prepare video embeddings
    video_embedding_map = (
        df_input.groupby("video_id")
        .agg(
            embedding=("embedding", "first"),
        )
        .reset_index()
    )  # df_input already has mean of video embedding, therefore the video_id_nunique >= video_embedding_nunique (> condition for the duplicates)

    # 1. Cluster Videos
    df_video_clusters_map, _, optimal_k_video = cluster_videos(
        video_embeddings_df=video_embedding_map.copy(),  # Pass a copy
        visualization_dir=video_visualization_dir,
        transformed_dir=video_transformed_dir,
        tsne_perplexity=video_tsne_perplexity,
        tsne_n_iter=video_tsne_n_iter,
        min_k_video=video_min_k,
        max_k_video=video_max_k,
        default_optimal_k_video=video_default_optimal_k,
    )

    # Create video_id <-> cluster hash-map
    video_cluster_label_map_dict = dict(
        zip(df_video_clusters_map["video_id"], df_video_clusters_map["cluster"])
    )
    _save_json(
        video_cluster_label_map_dict,
        video_transformed_dir / "video_cluster_label_map.json",
    )
    print(
        f"Video cluster label map saved to {video_transformed_dir / 'video_cluster_label_map.json'}"
    )

    # filtering out videos wtih < 25% watch percentage
    df_input = df_input[df_input["mean_percentage_watched"] >= 0.50]

    # 2. Calculate User Cluster Distributions
    df_user_engagement_dist = calculate_user_cluster_distributions(
        df_all_data=df_input.copy(),  # Pass a copy
        video_cluster_map_df=df_video_clusters_map,
        transformed_dir=video_transformed_dir,  # Save intermediate files here
    )

    # 3. Cluster Users
    df_user_final_clusters = cluster_users(
        df_user_engagement_with_dist=df_user_engagement_dist.copy(),  # Pass a copy
        visualization_dir=user_visualization_dir,
        transformed_dir=user_transformed_dir,  # Main transformed dir for user clusters too
        tsne_perplexity=user_tsne_perplexity,
        tsne_n_iter=user_tsne_n_iter,
        min_k_user=user_min_k,
        max_k_user=user_max_k,
        default_optimal_k_user=user_default_optimal_k,
        process_normalized=run_user_clustering_normalized,
        process_non_normalized=run_user_clustering_non_normalized,
    )

    print("Video and User clustering pipeline complete.")
    print(f"Video artifacts in: {video_visualization_dir} and {video_transformed_dir}")
    print(f"User artifacts in: {user_visualization_dir} and {user_transformed_dir}")

    # Example of how to analyze specific user clusters (from original script)
    if (
        run_user_clustering_normalized
        and "user_cluster_normalized" in df_user_final_clusters.columns
    ):
        # Find a cluster with a decent number of users for example analysis
        # This is just an example, you might pick a specific cluster number
        user_cluster_counts = df_user_final_clusters[
            "user_cluster_normalized"
        ].value_counts()
        if not user_cluster_counts.empty:
            example_cluster_id = user_cluster_counts.index[0]  # Largest cluster
            print(
                f"\nExample Analysis: Top 10 engaged videos for users in normalized cluster {example_cluster_id}"
            )

            # Filter users belonging to the example cluster
            users_in_cluster = df_user_final_clusters[
                df_user_final_clusters["user_cluster_normalized"] == example_cluster_id
            ]

            # Extract their engagement metadata
            engagement_lists = users_in_cluster[
                "engagement_metadata_list"
            ].sum()  # Concatenate all lists

            if engagement_lists:  # Check if there's any engagement data
                engaged_video_ids = [
                    item["video_id"] for item in engagement_lists if "video_id" in item
                ]
                if engaged_video_ids:
                    print(pd.Series(engaged_video_ids).value_counts().head(10))
                else:
                    print("No video IDs found in engagement data for this cluster.")
            else:
                print(
                    f"No engagement metadata found for users in cluster {example_cluster_id}"
                )
        else:
            print("No user clusters found for example analysis.")


# %%
if __name__ == "__main__":
    main_video_user_clustering(
        # You can override default parameters here if needed, e.g.:
        # data_root=pathlib.Path("/custom/path/to/data"),
        # video_max_k=25,
        # user_max_k=25,
    )
