CREATE OR REPLACE FUNCTION
  `hot-or-not-feed-intelligence.yral_ds.extract_video_id` (uri STRING) RETURNS STRING LANGUAGE js AS """
  if (!uri) return null;
  const match = uri.match(/([^\\/]+)\\.mp4$/);
  return match ? match[1] : null;
""";


CREATE OR REPLACE FUNCTION
  `hot-or-not-feed-intelligence.yral_ds.video_ids_to_urls` (input_video_ids ARRAY<STRING>) RETURNS ARRAY<STRUCT<video_id STRING, yral_url STRING>> AS (
    ARRAY (
      SELECT
        STRUCT (
          input_video_id AS video_id,
          CONCAT(
            'https://yral.com/hot-or-not/',
            vi.canister_id,
            '/',
            vi.post_id
          ) AS yral_url
        )
      FROM
        UNNEST (input_video_ids) AS input_video_id
        LEFT JOIN `jay-dhanwant-experiments.stage_tables.stage_video_index` vi ON `hot-or-not-feed-intelligence.yral_ds.extract_video_id` (vi.uri) = input_video_id
    )
  );


-- usage of get url from video ids
SELECT DISTINCT
  video_url.video_id,
  video_url.yral_url
FROM
  (
    SELECT
      `hot-or-not-feed-intelligence.yral_ds.video_ids_to_urls` (
        [
          '8ed149ae76de48cdb039bc79c9d352a5',
          'b53a7f55da4148b39738a9eda518ed81',
          '0a94387b126c4dc0bea37d8b4eef791c',
          '762466a701934bc7b5bb241109011f98',
          '3bd4a2d6b5184354bf64519e522a8cae',
          'b72394f8ec9741a18e68615575b835b5',
          'e3f81d8218614a50a63ed0d002992d7d',
          '3dc7a1d933424912ab4a5721ac8f637a',
          '542f4da1d78b41349b2c51ab8701b490',
          '3a77da284ead42ada2327cf71091c6ad'
        ]
      ) AS video_urls
  ),
  UNNEST (video_urls) AS video_url;
