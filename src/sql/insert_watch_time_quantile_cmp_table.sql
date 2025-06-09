-- First truncate the table to remove existing data
TRUNCATE TABLE
  `jay-dhanwant-experiments.stage_test_tables.watch_time_quantile_comparison_intermediate`;


-- Then insert data into the table
INSERT INTO
  `jay-dhanwant-experiments.stage_test_tables.watch_time_quantile_comparison_intermediate` -- VARIABLES
  -- Hardcoded values - equivalent to n_bins=4 in Python code
WITH
  VARS AS (
    SELECT
      4 AS n_bins
  ),
  -- Read data from the clusters table
  clusters AS (
    SELECT
      cluster_id,
      user_id,
      video_id,
      last_watched_timestamp,
      mean_percentage_watched,
      liked,
      last_liked_timestamp,
      shared,
      last_shared_timestamp,
      cluster_label,
      updated_at
    FROM
      `jay-dhanwant-experiments.stage_test_tables.test_user_clusters`
  ),
  -- Get approx seconds watched per user
  clusters_with_time AS (
    SELECT
      *,
      mean_percentage_watched * 60 AS time_watched_seconds_approx
    FROM
      clusters
  ),
  -- Aggregate time watched per user and cluster
  user_cluster_time AS (
    SELECT
      cluster_id,
      user_id,
      SUM(time_watched_seconds_approx) AS total_time_watched_seconds,
      ARRAY_AGG(video_id) AS list_videos_watched
    FROM
      clusters_with_time
    GROUP BY
      cluster_id,
      user_id
  ),
  -- Create a table of all users and their rank percentile within each cluster
  -- This mimics pd.qcut in Python which creates quantiles
  user_quantiles AS (
    SELECT
      cluster_id,
      user_id,
      total_time_watched_seconds,
      list_videos_watched,
      PERCENT_RANK() OVER (
        PARTITION BY
          cluster_id
        ORDER BY
          total_time_watched_seconds
      ) AS percentile_rank,
      NTILE(4) OVER (
        PARTITION BY
          cluster_id
        ORDER BY
          total_time_watched_seconds
      ) - 1 AS bin
    FROM
      user_cluster_time
  ),
  -- Add bin_type column to match pandas implementation
  user_cluster_quantiles AS (
    SELECT
      cluster_id,
      user_id,
      total_time_watched_seconds,
      list_videos_watched,
      percentile_rank AS quantile,
      bin,
      'watch_time' AS bin_type
    FROM
      user_quantiles
  ),
  -- Aggregate by cluster_id and bin - SAME AS PANDAS
  cluser_quantiles_agg_raw AS (
    SELECT
      cluster_id,
      bin,
      ARRAY_CONCAT_AGG(list_videos_watched) AS list_videos_watched_raw
    FROM
      user_cluster_quantiles
    GROUP BY
      cluster_id,
      bin
  ),
  -- Deduplicate videos in list_videos_watched - SAME AS PANDAS list(set(x))
  cluser_quantiles_agg AS (
    SELECT
      cluster_id,
      bin,
      ARRAY (
        SELECT DISTINCT
          video_id
        FROM
          UNNEST (list_videos_watched_raw) AS video_id
      ) AS list_videos_watched
    FROM
      cluser_quantiles_agg_raw
  ),
  -- Add flags exactly as pandas implementation
  -- The key is using shift(1) which translates to LAG in SQL
  cluser_quantiles_with_flags AS (
    SELECT
      cluster_id,
      bin,
      list_videos_watched,
      -- The fillna(False) in pandas becomes COALESCE in SQL
      COALESCE(
        LAG(cluster_id) OVER (
          ORDER BY
            cluster_id,
            bin
        ) = cluster_id,
        FALSE
      ) AS flag_same_cluster,
      COALESCE(
        LAG(bin) OVER (
          ORDER BY
            cluster_id,
            bin
        ) = bin,
        FALSE
      ) AS flag_same_bin,
      -- Get previous row's videos
      IFNULL(
        LAG(list_videos_watched) OVER (
          ORDER BY
            cluster_id,
            bin
        ),
        []
      ) AS shifted_list_videos_watched
    FROM
      cluser_quantiles_agg
  ),
  -- Calculate flag_compare exactly as in the Python function
  cluser_quantiles_with_compare AS (
    SELECT
      *,
      CASE
        WHEN flag_same_cluster = FALSE
        AND flag_same_bin = FALSE THEN FALSE
        WHEN flag_same_cluster = TRUE
        AND flag_same_bin = FALSE THEN TRUE
        ELSE NULL
      END AS flag_compare
    FROM
      cluser_quantiles_with_flags
  ),
  -- Add videos_to_be_checked_for_tier_progression
  -- Exactly matching Python's set difference operation
  final_result AS (
    SELECT
      cluster_id,
      bin,
      list_videos_watched,
      flag_same_cluster,
      flag_same_bin,
      shifted_list_videos_watched,
      flag_compare,
      -- Match Python's list(set(row["shifted_list_videos_watched"]).difference(set(row["list_videos_watched"])))
      CASE
        WHEN flag_compare = TRUE THEN (
          SELECT
            ARRAY_AGG(v)
          FROM
            (
              SELECT
                v
              FROM
                UNNEST (shifted_list_videos_watched) AS v
              WHERE
                v NOT IN (
                  SELECT
                    video_id
                  FROM
                    UNNEST (list_videos_watched) AS video_id
                )
            )
        )
        ELSE []
      END AS videos_to_be_checked_for_tier_progression,
      -- Length calculations
      ARRAY_LENGTH(shifted_list_videos_watched) AS num_cx,
      ARRAY_LENGTH(list_videos_watched) AS num_cy
    FROM
      cluser_quantiles_with_compare
  ) -- Final output in the same order as Python
SELECT
  *
FROM
  final_result
ORDER BY
  cluster_id,
  bin;
