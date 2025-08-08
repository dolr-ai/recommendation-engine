--- to run in prod
WITH
  regional_candidates AS (
    SELECT
      rg.video_id,
      rg.region,
      CAST(rg.within_region_popularity_score AS FLOAT64) AS within_region_popularity_score,
      CASE
        WHEN nsfw.probability >= 0.7 THEN TRUE
        WHEN nsfw.probability < 0.4 THEN FALSE
        ELSE NULL
      END AS is_nsfw,
      nsfw.probability AS probability
    FROM
      `hot-or-not-feed-intelligence.yral_ds.region_grossing_l7d_candidates` rg
      LEFT JOIN `hot-or-not-feed-intelligence.yral_ds.video_nsfw_agg` nsfw ON rg.video_id = nsfw.video_id
    WHERE
      rg.within_region_popularity_score IS NOT NULL
      AND rg.within_region_popularity_score > 0
  )
SELECT
  region,
  COUNTIF(is_nsfw = TRUE) AS nsfw_count,
  COUNTIF(is_nsfw = FALSE) AS clean_count,
  COUNTIF(is_nsfw IS NULL) AS undetermined_count,
  COUNT(*) AS total_count
FROM
  regional_candidates
GROUP BY
  region
ORDER BY
  total_count,
  clean_count,
  nsfw_count DESC --- to run in stage
WITH
  regional_candidates AS (
    SELECT
      rg.video_id,
      rg.region,
      CAST(rg.within_region_popularity_score AS FLOAT64) AS within_region_popularity_score,
      CASE
        WHEN nsfw.probability >= 0.7 THEN TRUE
        WHEN nsfw.probability < 0.4 THEN FALSE
        ELSE NULL
      END AS is_nsfw,
      nsfw.probability AS probability
    FROM
      `hot-or-not-feed-intelligence.yral_ds.region_grossing_l7d_candidates` rg
      LEFT JOIN `hot-or-not-feed-intelligence.yral_ds.video_nsfw_agg` nsfw ON rg.video_id = nsfw.video_id
    WHERE
      rg.within_region_popularity_score IS NOT NULL
      AND rg.within_region_popularity_score > 0
  )
SELECT
  region,
  COUNTIF(is_nsfw = TRUE) AS nsfw_count,
  COUNTIF(is_nsfw = FALSE) AS clean_count,
  COUNTIF(is_nsfw IS NULL) AS undetermined_count,
  COUNT(*) AS total_count
FROM
  regional_candidates
GROUP BY
  region
ORDER BY
  total_count,
  clean_count,
  nsfw_count DESC
