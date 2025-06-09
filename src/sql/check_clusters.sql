-- Check the cluster 0 data in the intermediate table
SELECT
  cluster_id,
  bin,
  flag_same_cluster,
  flag_same_bin,
  flag_compare,
  ARRAY_LENGTH(shifted_list_videos_watched) AS num_shifted_videos,
  ARRAY_LENGTH(list_videos_watched) AS num_videos
FROM
  `jay-dhanwant-experiments.stage_test_tables.user_cluster_watch_time_comparison_intermediate`
WHERE
  cluster_id = 0
ORDER BY
  bin;


-- Check if flag_compare is TRUE for any cluster
SELECT
  cluster_id,
  bin,
  flag_compare,
  ARRAY_LENGTH(shifted_list_videos_watched) AS num_shifted_videos
FROM
  `jay-dhanwant-experiments.stage_test_tables.user_cluster_watch_time_comparison_intermediate`
WHERE
  flag_compare = TRUE
ORDER BY
  cluster_id,
  bin;


-- Check for issue in query_videos CTE - how many videos are being processed per cluster
WITH
  query_videos AS (
    SELECT
      cluster_id,
      bin,
      query_video_id
    FROM
      `jay-dhanwant-experiments.stage_test_tables.user_cluster_watch_time_comparison_intermediate`,
      UNNEST (shifted_list_videos_watched) AS query_video_id
    WHERE
      flag_compare = TRUE
  )
SELECT
  cluster_id,
  bin,
  COUNT(*) AS video_count
FROM
  query_videos
GROUP BY
  cluster_id,
  bin
ORDER BY
  cluster_id,
  bin;


-- Check for missing embeddings for cluster 0 videos (embedding join issues)
WITH
  query_videos AS (
    SELECT
      cluster_id,
      bin,
      query_video_id
    FROM
      `jay-dhanwant-experiments.stage_test_tables.user_cluster_watch_time_comparison_intermediate`,
      UNNEST (shifted_list_videos_watched) AS query_video_id
    WHERE
      flag_compare = TRUE
      AND cluster_id = 0
  ),
  video_embeddings AS (
    SELECT
      `jay-dhanwant-experiments.stage_test_tables.extract_video_id` (vi.uri) AS video_id
    FROM
      `jay-dhanwant-experiments.stage_tables.stage_video_index` vi
    WHERE
      `jay-dhanwant-experiments.stage_test_tables.extract_video_id` (vi.uri) IS NOT NULL
  )
SELECT
  qv.cluster_id,
  qv.bin,
  qv.query_video_id,
  CASE
    WHEN ve.video_id IS NULL THEN 'Missing'
    ELSE 'Found'
  END AS embedding_status
FROM
  query_videos qv
  LEFT JOIN video_embeddings ve ON qv.query_video_id = ve.video_id
ORDER BY
  embedding_status,
  query_video_id;


-- Check final output table for all clusters
SELECT
  cluster_id,
  bin,
  COUNT(*) AS video_count
FROM
  `jay-dhanwant-experiments.stage_test_tables.user_cluster_watch_time_comparison_intermediate_nearest_neighbors`
GROUP BY
  cluster_id,
  bin
ORDER BY
  cluster_id,
  bin;


-- Check similarity distributions for cluster 0's videos with embeddings (before threshold filter)
WITH
  query_videos AS (
    -- Get only cluster 0 videos that have embeddings
    SELECT
      cluster_id,
      bin,
      query_video_id,
      list_videos_watched
    FROM
      `jay-dhanwant-experiments.stage_test_tables.user_cluster_watch_time_comparison_intermediate`,
      UNNEST (shifted_list_videos_watched) AS query_video_id
    WHERE
      flag_compare = TRUE
      AND cluster_id = 0
      AND query_video_id IN (
        SELECT
          `jay-dhanwant-experiments.stage_test_tables.extract_video_id` (vi.uri) AS video_id
        FROM
          `jay-dhanwant-experiments.stage_tables.stage_video_index` vi
        WHERE
          `jay-dhanwant-experiments.stage_test_tables.extract_video_id` (vi.uri) IS NOT NULL
      )
  ),
  embedding_elements AS (
    SELECT
      `jay-dhanwant-experiments.stage_test_tables.extract_video_id` (vi.uri) AS video_id,
      embedding_value,
      pos
    FROM
      `jay-dhanwant-experiments.stage_tables.stage_video_index` vi,
      UNNEST (vi.embedding) AS embedding_value
    WITH
    OFFSET
      pos
    WHERE
      `jay-dhanwant-experiments.stage_test_tables.extract_video_id` (vi.uri) IS NOT NULL
  ),
  video_embeddings AS (
    SELECT
      video_id,
      ARRAY_AGG(
        avg_value
        ORDER BY
          pos
      ) AS avg_embedding
    FROM
      (
        SELECT
          video_id,
          pos,
          AVG(embedding_value) AS avg_value
        FROM
          embedding_elements
        GROUP BY
          video_id,
          pos
      )
    GROUP BY
      video_id
  ),
  query_embeddings AS (
    SELECT
      qv.cluster_id,
      qv.bin,
      qv.query_video_id,
      qv.list_videos_watched,
      ve.avg_embedding AS query_embedding
    FROM
      query_videos qv
      JOIN video_embeddings ve ON qv.query_video_id = ve.video_id
  ),
  search_space_videos AS (
    SELECT
      qe.cluster_id,
      qe.bin,
      qe.query_video_id,
      qe.query_embedding,
      ve.video_id AS candidate_video_id,
      ve.avg_embedding AS candidate_embedding,
      ML.DISTANCE (qe.query_embedding, ve.avg_embedding, 'COSINE') AS distance
    FROM
      query_embeddings qe,
      UNNEST (qe.list_videos_watched) AS watched_video_id
      JOIN video_embeddings ve ON watched_video_id = ve.video_id
    WHERE
      qe.query_video_id != ve.video_id
  )
SELECT
  query_video_id,
  MIN(distance) AS min_distance,
  MAX(distance) AS max_distance,
  AVG(distance) AS avg_distance,
  COUNTIF(distance < 0.336) AS matches_below_threshold,
  COUNT(*) AS total_candidates
FROM
  search_space_videos
GROUP BY
  query_video_id
ORDER BY
  matches_below_threshold DESC;


-- Statistics on missing embeddings across all clusters
WITH
  all_videos AS (
    -- Get all videos from shifted_list_videos_watched that are being processed
    SELECT
      cluster_id,
      bin,
      query_video_id
    FROM
      `jay-dhanwant-experiments.stage_test_tables.user_cluster_watch_time_comparison_intermediate`,
      UNNEST (shifted_list_videos_watched) AS query_video_id
    WHERE
      flag_compare = TRUE
  ),
  video_embedding_status AS (
    SELECT
      av.cluster_id,
      av.bin,
      av.query_video_id,
      CASE
        WHEN ve.video_id IS NULL THEN 'Missing'
        ELSE 'Found'
      END AS embedding_status
    FROM
      all_videos av
      LEFT JOIN (
        SELECT
          `jay-dhanwant-experiments.stage_test_tables.extract_video_id` (vi.uri) AS video_id
        FROM
          `jay-dhanwant-experiments.stage_tables.stage_video_index` vi
        WHERE
          `jay-dhanwant-experiments.stage_test_tables.extract_video_id` (vi.uri) IS NOT NULL
      ) ve ON av.query_video_id = ve.video_id
  )
SELECT
  cluster_id,
  COUNT(DISTINCT query_video_id) AS total_videos,
  COUNTIF(embedding_status = 'Found') AS videos_with_embeddings,
  COUNTIF(embedding_status = 'Missing') AS videos_without_embeddings,
  ROUND(
    COUNTIF(embedding_status = 'Found') / COUNT(DISTINCT query_video_id) * 100,
    2
  ) AS percent_with_embeddings
FROM
  video_embedding_status
GROUP BY
  cluster_id
ORDER BY
  cluster_id;
