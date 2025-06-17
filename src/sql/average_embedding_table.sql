CREATE OR REPLACE TABLE `jay-dhanwant-experiments.stage_tables.video_embedding_average` (
    video_id STRING,
    avg_embedding ARRAY<FLOAT64>
);

INSERT INTO `jay-dhanwant-experiments.stage_tables.video_embedding_average` (video_id, avg_embedding)
WITH flattened_embeddings AS (
    SELECT
        `jay-dhanwant-experiments.stage_test_tables.extract_video_id`(uri) as video_id,
        pos,
        embedding_value
    FROM `jay-dhanwant-experiments.stage_tables.stage_video_index`,
    UNNEST(embedding) AS embedding_value WITH OFFSET pos
    WHERE `jay-dhanwant-experiments.stage_test_tables.extract_video_id`(uri) IS NOT NULL
),
averaged_by_position AS (
    SELECT
        video_id,
        pos,
        AVG(embedding_value) as avg_value
    FROM flattened_embeddings
    GROUP BY video_id, pos
)
SELECT
    video_id,
    ARRAY_AGG(avg_value ORDER BY pos) as avg_embedding
FROM averaged_by_position
GROUP BY video_id;