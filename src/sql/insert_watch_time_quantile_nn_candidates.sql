-- First create or replace the table structure
CREATE OR REPLACE TABLE
  `jay-dhanwant-experiments.stage_test_tables.watch_time_quantile_comparison_intermediate` (
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


-- Then insert data into the table
INSERT INTO
  `jay-dhanwant-experiments.stage_test_tables.watch_time_quantile_comparison_intermediate` -- VARIABLES
  -- Hardcoded values - equivalent to n_bins=4 in Python code
WITH
  variables AS (
    SELECT
      4 AS n_bins,
      100 AS min_list_videos_watched,
      100 AS min_shifted_list_videos_watched
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
      ) -1 AS bin
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
  -- First get the shifted data using LAG
  cluser_quantiles_with_lag AS (
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
      -- Get previous row's videos (just the array, no unnesting yet)
      IFNULL(
        LAG(list_videos_watched) OVER (
          ORDER BY
            cluster_id,
            bin
        ),
        []
      ) AS shifted_videos_raw
    FROM
      cluser_quantiles_agg
  ),
  -- Now deduplicate the shifted videos in a separate step
  cluser_quantiles_with_flags AS (
    SELECT
      cluster_id,
      bin,
      list_videos_watched,
      flag_same_cluster,
      flag_same_bin,
      -- Now deduplicate the shifted videos
      ARRAY (
        SELECT DISTINCT
          video_id
        FROM
          UNNEST (shifted_videos_raw) AS video_id
      ) AS shifted_list_videos_watched
    FROM
      cluser_quantiles_with_lag
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
  -- Prepare a CTE to extract videos that are in the shifted list but not in the current list
  videos_not_in_current AS (
    SELECT
      cluster_id,
      bin,
      list_videos_watched,
      flag_same_cluster,
      flag_same_bin,
      shifted_list_videos_watched,
      flag_compare,
      -- For each row, create a filtered array of videos that aren't in the current list
      ARRAY (
        SELECT
          v
        FROM
          UNNEST (shifted_list_videos_watched) AS v
        WHERE
          v NOT IN (
            SELECT
              v2
            FROM
              UNNEST (list_videos_watched) AS v2
          )
      ) AS progression_videos
    FROM
      cluser_quantiles_with_compare
    WHERE
      flag_compare = TRUE
  ),
  -- Add videos_to_be_checked_for_tier_progression
  -- Exactly matching Python's set difference operation
  final_result AS (
    SELECT
      c.cluster_id,
      c.bin,
      c.list_videos_watched,
      c.flag_same_cluster,
      c.flag_same_bin,
      c.shifted_list_videos_watched,
      c.flag_compare,
      -- Use the pre-computed array if this is a row with flag_compare = TRUE
      CASE
        WHEN c.flag_compare = TRUE THEN (
          SELECT
            progression_videos
          FROM
            videos_not_in_current v
          WHERE
            v.cluster_id = c.cluster_id
            AND v.bin = c.bin
        )
        ELSE []
      END AS videos_to_be_checked_for_tier_progression,
      -- Length calculations
      ARRAY_LENGTH(c.shifted_list_videos_watched) AS num_cx,
      ARRAY_LENGTH(c.list_videos_watched) AS num_cy
    FROM
      cluser_quantiles_with_compare c -- Filter here to ensure sufficient videos in both lists
    WHERE
      (c.flag_compare IS NULL)
      OR (
        c.flag_compare = TRUE
        AND ARRAY_LENGTH(c.list_videos_watched) > (
          SELECT
            min_list_videos_watched
          FROM
            variables
        )
        AND ARRAY_LENGTH(c.shifted_list_videos_watched) > (
          SELECT
            min_shifted_list_videos_watched
          FROM
            variables
        )
      )
  ) -- Final output in the same order as Python
SELECT
  *
FROM
  final_result
ORDER BY
  cluster_id,
  bin;
