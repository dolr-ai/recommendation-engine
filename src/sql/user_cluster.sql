CREATE TABLE `hot-or-not-feed-intelligence.yral_ds.test_user_cluster_embeddings` (
  user_id STRING NOT NULL
    OPTIONS(description="Unique user identifier"),
  cluster_id INT64
    OPTIONS(description="Cluster identifier for partitioning and performance optimization"),
  user_embedding ARRAY<FLOAT64>
    OPTIONS(description="Normalized concatenated user embedding vector"),
  updated_at TIMESTAMP NOT NULL
    OPTIONS(description="Timestamp when the embedding was uploaded"),
)
CLUSTER BY cluster_id, user_id
OPTIONS(
  description="User embeddings table with optimized partitioning and clustering",
  labels=[("environment", "stage"), ("data_type", "embeddings")]
);

