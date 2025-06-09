-- Create table for nearest neighbors results
CREATE OR REPLACE TABLE
  `jay-dhanwant-experiments.stage_test_tables.user_cluster_watch_time_comparison_intermediate_nearest_neighbors` (
    cluster_id INT64,
    bin INT64,
    query_video_id STRING,
    comparison_flow STRING,
    nearest_neighbors ARRAY<STRUCT<candidate_video_id STRING, distance FLOAT64>>
  );


-- Insert nearest neighbors results
INSERT INTO
  `jay-dhanwant-experiments.stage_test_tables.user_cluster_watch_time_comparison_intermediate_nearest_neighbors`
WITH
  -- First get the intermediate data to understand the bin relationships
  intermediate_data AS (
    SELECT
      cluster_id,
      bin,
      flag_compare,
      shifted_list_videos_watched,
      list_videos_watched,
      -- The shifted_list_videos_watched comes from the previous bin
      -- The bin where the videos are FROM
      LAG(bin) OVER (
        PARTITION BY
          cluster_id
        ORDER BY
          bin
      ) AS source_bin
    FROM
      `jay-dhanwant-experiments.stage_test_tables.user_cluster_watch_time_comparison_intermediate`
    WHERE
      flag_compare = TRUE -- Only process rows where comparison is flagged
  ),
  query_videos AS (
    -- Flatten the shifted_list_videos_watched to get individual query video_ids
    SELECT
      idata.cluster_id,
      idata.bin AS target_bin,
      -- Current bin is the target (hyperspace)
      idata.source_bin,
      -- Source bin is where the query videos come from
      query_video_id,
      idata.list_videos_watched,
      -- Ensure source_bin is not null, use IFNULL to handle the first bin
      FORMAT(
        '%d->[%d]->[%d]',
        idata.cluster_id,
        IFNULL(idata.source_bin, 0),
        idata.bin
      ) AS comparison_flow
    FROM
      intermediate_data idata,
      UNNEST (idata.shifted_list_videos_watched) AS query_video_id
  ),
  -- First, flatten all embeddings with their video_ids
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
  -- Aggregate embeddings by video_id (average multiple embeddings per video)
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
    -- Get averaged embeddings for query videos
    SELECT
      qv.cluster_id,
      qv.target_bin AS bin,
      -- Use target_bin for bin to maintain compatibility
      qv.query_video_id,
      qv.list_videos_watched,
      qv.comparison_flow,
      -- Make sure to pass this field along
      ve.avg_embedding AS query_embedding
    FROM
      query_videos qv
      JOIN video_embeddings ve ON qv.query_video_id = ve.video_id
  ),
  search_space_videos AS (
    -- Get averaged embeddings for videos in list_videos_watched (search space X)
    SELECT
      qe.cluster_id,
      qe.bin,
      qe.query_video_id,
      qe.comparison_flow,
      -- Make sure to pass this field along
      qe.query_embedding,
      ve.video_id AS candidate_video_id,
      ve.avg_embedding AS candidate_embedding
    FROM
      query_embeddings qe,
      UNNEST (qe.list_videos_watched) AS watched_video_id
      JOIN video_embeddings ve ON watched_video_id = ve.video_id
    WHERE
      qe.query_video_id != ve.video_id -- Exclude self-matches
  ) -- Perform vector similarity search
SELECT
  cluster_id,
  bin,
  query_video_id,
  comparison_flow,
  -- Make sure to include this in the final output
  ARRAY_AGG(
    STRUCT (
      candidate_video_id,
      ML.DISTANCE (query_embedding, candidate_embedding, 'COSINE') AS distance
    )
    ORDER BY
      ML.DISTANCE (query_embedding, candidate_embedding, 'COSINE') ASC
    LIMIT
      10 -- Get top 10 nearest neighbors
  ) AS nearest_neighbors
FROM
  search_space_videos
WHERE
  ML.DISTANCE (query_embedding, candidate_embedding, 'COSINE') < 0.336 -- Only include videos with cosine similarity > 0.664
GROUP BY
  cluster_id,
  bin,
  query_video_id,
  comparison_flow,
  -- Make sure to include this in the GROUP BY
  query_embedding
ORDER BY
  cluster_id,
  bin,
  query_video_id;
