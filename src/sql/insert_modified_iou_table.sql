-- Modified IoU Score Calculation for Video Recommendation Candidates
-- This SQL replicates the logic from the Python code in 002-iou_algorithm_prep_data.py
-- FOCUS ON CLUSTER 1 ONLY and returns the exact same result as res_dict[1]["candidates"]
-- Delete any existing data for the target cluster (cluster 1)
-- This ensures we don't have duplicate entries when we re-run the calculation
DELETE FROM
  `hot-or-not-feed-intelligence.yral_ds.modified_iou_candidates`
WHERE
  cluster_id = 1;


-- Insert the results from the modified_iou_intermediate_table.sql calculation
INSERT INTO
  `hot-or-not-feed-intelligence.yral_ds.modified_iou_candidates` (
    cluster_id,
    video_id_x,
    user_id_list_min_x,
    user_id_list_success_x,
    d,
    cluster_id_y,
    video_id_y,
    user_id_list_min_y,
    user_id_list_success_y,
    den,
    num,
    iou_modified
  ) -- The query below should match the final SELECT in modified_iou_intermediate_table.sql
WITH
  -- Constants for thresholds
  constants AS (
    SELECT
      0.5 AS watch_percentage_threshold_min,
      -- Minimum watch percentage for basic engagement
      0.75 AS watch_percentage_threshold_success,
      -- Threshold for successful engagement
      2 AS min_req_users_for_video,
      -- Minimum required users per video
      1.0 AS sample_factor -- Sampling factor (1.0 = use all data)
  ),
  -- Filter data for minimum watch threshold - ONLY CLUSTER 1
  min_threshold_data AS (
    SELECT
      cluster_id,
      user_id,
      video_id,
      mean_percentage_watched
    FROM
      `hot-or-not-feed-intelligence.yral_ds.recsys_user_cluster_interaction`
    WHERE
      mean_percentage_watched > 0.5
      AND cluster_id = 1 -- FOCUS ONLY ON CLUSTER 1
  ),
  -- Filter data for success watch threshold - ONLY CLUSTER 1
  success_threshold_data AS (
    SELECT
      cluster_id,
      user_id,
      video_id,
      mean_percentage_watched
    FROM
      `hot-or-not-feed-intelligence.yral_ds.recsys_user_cluster_interaction`
    WHERE
      mean_percentage_watched > 0.75
      AND cluster_id = 1 -- FOCUS ONLY ON CLUSTER 1
  ),
  -- Group by video for minimum threshold
  min_grouped AS (
    SELECT
      cluster_id,
      video_id,
      ARRAY_AGG(user_id) AS user_id_list_min,
      COUNT(DISTINCT user_id) AS num_unique_users
    FROM
      min_threshold_data
    GROUP BY
      cluster_id,
      video_id
    HAVING
      COUNT(DISTINCT user_id) >= 2 -- Use hardcoded value to avoid subquery
  ),
  -- Group by video for success threshold
  success_grouped AS (
    SELECT
      cluster_id,
      video_id,
      ARRAY_AGG(user_id) AS user_id_list_success,
      COUNT(DISTINCT user_id) AS num_unique_users
    FROM
      success_threshold_data
    GROUP BY
      cluster_id,
      video_id
    HAVING
      COUNT(DISTINCT user_id) >= 2 -- Use hardcoded value to avoid subquery
  ),
  -- Create base dataset for pairwise comparison
  base_data AS (
    SELECT
      m.cluster_id,
      m.video_id,
      m.user_id_list_min,
      s.user_id_list_success
    FROM
      min_grouped m
      INNER JOIN success_grouped s ON m.cluster_id = s.cluster_id
      AND m.video_id = s.video_id
  ),
  -- Add a dummy column to replicate the Python df_base["d"] = 1
  base_data_with_d AS (
    SELECT
      cluster_id,
      video_id,
      user_id_list_min,
      user_id_list_success,
      1 AS d -- This replicates df_base["d"] = 1 in Python
    FROM
      base_data
  ),
  -- This replicates the Python df_req = df_base.merge(df_base, on=["d"], suffixes=["_x", "_y"])
  -- Create cartesian product of all videos with all videos (including self-joins)
  all_pairs_raw AS (
    SELECT
      b1.cluster_id AS cluster_id_x,
      b1.video_id AS video_id_x,
      b1.user_id_list_min AS user_id_list_min_x,
      b1.user_id_list_success AS user_id_list_success_x,
      b1.d,
      b2.cluster_id AS cluster_id_y,
      b2.video_id AS video_id_y,
      b2.user_id_list_min AS user_id_list_min_y,
      b2.user_id_list_success AS user_id_list_success_y
    FROM
      base_data_with_d b1
      JOIN base_data_with_d b2 ON b1.d = b2.d
  ),
  -- Create a unique key to deduplicate (replicates df_req["pkey"] = df_req["video_id_x"] + "_" + df_req["video_id_y"])
  all_pairs_with_key AS (
    SELECT
      *,
      CONCAT(video_id_x, "_", video_id_y) AS pkey
    FROM
      all_pairs_raw
  ),
  -- Deduplicate (replicates df_req = df_req.drop_duplicates(subset=["pkey"]))
  all_pairs_deduplicated AS (
    SELECT
      cluster_id_x,
      video_id_x,
      user_id_list_min_x,
      user_id_list_success_x,
      d,
      cluster_id_y,
      video_id_y,
      user_id_list_min_y,
      user_id_list_success_y
    FROM
      (
        SELECT
          *,
          ROW_NUMBER() OVER (
            PARTITION BY
              pkey
            ORDER BY
              video_id_x
          ) AS rn
        FROM
          all_pairs_with_key
      )
    WHERE
      rn = 1
  ),
  -- Filter out same video comparisons (replicates df_req = df_req[df_req["video_id_x"] != df_req["video_id_y"]])
  all_pairs_filtered AS (
    SELECT
      cluster_id_x,
      video_id_x,
      user_id_list_min_x,
      user_id_list_success_x,
      d,
      cluster_id_y,
      video_id_y,
      user_id_list_min_y,
      user_id_list_success_y
    FROM
      all_pairs_deduplicated
    WHERE
      video_id_x != video_id_y
  ),
  -- Calculate denominator and numerator (replicates Python calculations)
  all_pairs_with_calcs AS (
    SELECT
      cluster_id_x,
      video_id_x,
      user_id_list_min_x,
      user_id_list_success_x,
      d,
      cluster_id_y,
      video_id_y,
      user_id_list_min_y,
      user_id_list_success_y,
      -- Calculate denominator (replicates df_req["den"] = ...)
      (
        ARRAY_LENGTH(user_id_list_min_x) + ARRAY_LENGTH(user_id_list_min_y)
      ) AS den,
      -- Calculate numerator (replicates df_req["num"] = ...)
      (
        SELECT
          COUNT(*)
        FROM
          UNNEST (user_id_list_success_x) AS user_x
        WHERE
          user_x IN UNNEST (user_id_list_success_y)
      ) AS num
    FROM
      all_pairs_filtered
  ),
  -- Calculate modified IoU score (replicates df_req["iou_modified"] = ((df_req["num"] / df_req["den"]).round(2)) * 2)
  all_pairs_with_iou AS (
    SELECT
      cluster_id_x,
      video_id_x,
      user_id_list_min_x,
      user_id_list_success_x,
      d,
      cluster_id_y,
      video_id_y,
      user_id_list_min_y,
      user_id_list_success_y,
      den,
      num,
      ROUND((num / CAST(den AS FLOAT64)), 2) * 2 AS iou_modified
    FROM
      all_pairs_with_calcs
    WHERE
      num > 0 -- Only keep pairs with positive IoU (replicates df_req = df_req[df_req["iou_modified"] > 0])
  ),
  -- Calculate the 95th percentile (replicates df_temp["iou_modified"].quantile(0.95))
  percentile_calc AS (
    SELECT
      APPROX_QUANTILES(iou_modified, 100) [OFFSET(95)] AS p95_threshold
    FROM
      all_pairs_with_iou
  ) -- Get final candidates (replicates df_cand = df_temp[df_temp["iou_modified"] > df_temp["iou_modified"].quantile(0.95)])
  -- This is the same as res_dict[1]["candidates"]
SELECT
  cluster_id_x,
  video_id_x,
  user_id_list_min_x,
  user_id_list_success_x,
  d,
  cluster_id_y,
  video_id_y,
  user_id_list_min_y,
  user_id_list_success_y,
  den,
  num,
  iou_modified
FROM
  all_pairs_with_iou a
  CROSS JOIN percentile_calc p
WHERE
  a.iou_modified > p.p95_threshold
ORDER BY
  iou_modified DESC;
