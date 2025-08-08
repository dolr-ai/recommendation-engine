-- Create the table structure first
CREATE OR REPLACE TABLE
  `hot-or-not-feed-intelligence.yral_ds.watch_time_quantile_comparison_intermediate` (
    cluster_id INT64,
    bin INT64,
    list_videos_watched ARRAY<STRING>,
    flag_same_cluster BOOL,
    flag_same_bin BOOL,
    shifted_list_videos_watched ARRAY<STRING>,
    flag_compare BOOL,
    videos_to_be_checked_for_tier_progression ARRAY<STRING>,
    num_cx INT64,
    num_cy INT64
  );
