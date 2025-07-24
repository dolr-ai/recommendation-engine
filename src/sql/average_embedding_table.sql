-- # initial creation of the table
CREATE OR REPLACE TABLE
  `jay-dhanwant-experiments.stage_tables.video_embedding_average` (video_id STRING, avg_embedding ARRAY<FLOAT64>);


INSERT INTO
  `jay-dhanwant-experiments.stage_tables.video_embedding_average` (video_id, avg_embedding)
WITH
  flattened_embeddings AS (
    SELECT
      `hot-or-not-feed-intelligence.yral_ds.extract_video_id` (uri) AS video_id,
      pos,
      embedding_value
    FROM
      `jay-dhanwant-experiments.stage_tables.stage_video_index`,
      UNNEST (embedding) AS embedding_value
    WITH
    OFFSET
      pos
    WHERE
      `hot-or-not-feed-intelligence.yral_ds.extract_video_id` (uri) IS NOT NULL
  ),
  averaged_by_position AS (
    SELECT
      video_id,
      pos,
      AVG(embedding_value) AS avg_value
    FROM
      flattened_embeddings
    GROUP BY
      video_id,
      pos
  )
SELECT
  video_id,
  ARRAY_AGG(
    avg_value
    ORDER BY
      pos
  ) AS avg_embedding
FROM
  averaged_by_position
GROUP BY
  video_id;


-- # consecutive updates to the table should use this query
INSERT INTO
  `jay-dhanwant-experiments.stage_tables.video_embedding_average` (video_id, avg_embedding)
WITH
  missing_videos AS (
    SELECT DISTINCT
      `hot-or-not-feed-intelligence.yral_ds.extract_video_id` (uri) AS video_id
    FROM
      `jay-dhanwant-experiments.stage_tables.stage_video_index`
    WHERE
      `hot-or-not-feed-intelligence.yral_ds.extract_video_id` (uri) IS NOT NULL
      AND `hot-or-not-feed-intelligence.yral_ds.extract_video_id` (uri) NOT IN (
        SELECT
          video_id
        FROM
          `jay-dhanwant-experiments.stage_tables.video_embedding_average`
        WHERE
          video_id IS NOT NULL
      )
  ),
  flattened_embeddings AS (
    SELECT
      `hot-or-not-feed-intelligence.yral_ds.extract_video_id` (s.uri) AS video_id,
      pos,
      embedding_value
    FROM
      `jay-dhanwant-experiments.stage_tables.stage_video_index` s
      INNER JOIN missing_videos m ON `hot-or-not-feed-intelligence.yral_ds.extract_video_id` (s.uri) = m.video_id,
      UNNEST (embedding) AS embedding_value
    WITH
    OFFSET
      pos
  ),
  averaged_by_position AS (
    SELECT
      video_id,
      pos,
      AVG(embedding_value) AS avg_value
    FROM
      flattened_embeddings
    GROUP BY
      video_id,
      pos
  )
SELECT
  video_id,
  ARRAY_AGG(
    avg_value
    ORDER BY
      pos
  ) AS avg_embedding
FROM
  averaged_by_position
GROUP BY
  video_id;
