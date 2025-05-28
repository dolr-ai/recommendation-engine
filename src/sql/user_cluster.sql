CREATE TABLE `hot-or-not-feed-intelligence.yral_ds.test_user_cluster_embeddings` (
  user_id STRING NOT NULL
    OPTIONS(description="Unique user identifier"),
  cluster_id INT64
    OPTIONS(description="Cluster identifier for partitioning and performance optimization"),
  user_embedding ARRAY<FLOAT64>
    OPTIONS(description="Normalized concatenated user embedding vector"),
  avg_interaction_embedding ARRAY<FLOAT64>
    OPTIONS(description="Average interaction embedding vector"),
  cluster_distribution_embedding ARRAY<FLOAT64>
    OPTIONS(description="Cluster distribution embedding vector"),
  temporal_embedding ARRAY<FLOAT64>
    OPTIONS(description="Temporal interaction embedding vector"),
  engagement_metadata_list ARRAY<STRUCT<
    video_id STRING,
    last_watched_timestamp TIMESTAMP,
    mean_percentage_watched FLOAT64,
    liked BOOL,
    last_liked_timestamp TIMESTAMP,
    shared BOOL,
    last_shared_timestamp TIMESTAMP,
    cluster_label INT64
  >>
    OPTIONS(description="List of video engagement metadata for user"),
  updated_at TIMESTAMP NOT NULL
    OPTIONS(description="Timestamp when the embedding was uploaded")
)
CLUSTER BY cluster_id, user_id
OPTIONS(
  description="User embeddings table with optimized partitioning and clustering",
  labels=[("environment", "stage"), ("data_type", "embeddings")]
);

