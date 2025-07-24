-- Table 1: User embeddings table
CREATE TABLE
  `hot-or-not-feed-intelligence.yral_ds.recsys_user_embeddings` (
    cluster_id INT64 OPTIONS(
      description = "Cluster identifier for partitioning and performance optimization"
    ),
    user_id STRING NOT NULL OPTIONS(description = "Unique user identifier"),
    user_embedding ARRAY<FLOAT64> OPTIONS(
      description = "Normalized concatenated user embedding vector"
    ),
    avg_interaction_embedding ARRAY<FLOAT64> OPTIONS(
      description = "Average interaction embedding vector"
    ),
    cluster_distribution_embedding ARRAY<FLOAT64> OPTIONS(
      description = "Cluster distribution embedding vector"
    ),
    temporal_embedding ARRAY<FLOAT64> OPTIONS(
      description = "Temporal interaction embedding vector"
    ),
    updated_at TIMESTAMP NOT NULL OPTIONS(
      description = "Timestamp when the embedding was uploaded"
    )
  )
CLUSTER BY
  cluster_id,
  user_id OPTIONS(
    description = "User embeddings table with optimized partitioning and clustering",
    labels = [
      ("environment", "stage"),
      ("data_type", "embeddings")
    ]
  );


-- Table 2: User clusters table with flattened engagement metadata
CREATE TABLE
  `hot-or-not-feed-intelligence.yral_ds.recsys_user_cluster_interaction` (
    cluster_id INT64 OPTIONS(
      description = "Cluster identifier for partitioning and performance optimization"
    ),
    user_id STRING NOT NULL OPTIONS(description = "Unique user identifier"),
    video_id STRING OPTIONS(description = "Video identifier"),
    last_watched_timestamp TIMESTAMP OPTIONS(
      description = "Timestamp when the video was last watched"
    ),
    mean_percentage_watched FLOAT64 OPTIONS(description = "Mean percentage of video watched"),
    liked BOOL OPTIONS(description = "Whether the video was liked"),
    last_liked_timestamp TIMESTAMP OPTIONS(
      description = "Timestamp when the video was last liked"
    ),
    shared BOOL OPTIONS(description = "Whether the video was shared"),
    last_shared_timestamp TIMESTAMP OPTIONS(
      description = "Timestamp when the video was last shared"
    ),
    cluster_label INT64 OPTIONS(description = "Cluster label for the video"),
    updated_at TIMESTAMP NOT NULL OPTIONS(
      description = "Timestamp when the record was updated"
    )
  )
CLUSTER BY
  cluster_id,
  user_id OPTIONS(
    description = "User clusters table with flattened engagement metadata for easier access",
    labels = [("environment", "stage"), ("data_type", "clusters")]
  );
