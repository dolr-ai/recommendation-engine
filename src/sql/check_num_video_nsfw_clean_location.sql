--- to run in prod
WITH regional_candidates AS (
  SELECT
    rg.video_id,
    rg.region,
    CAST(rg.within_region_popularity_score AS FLOAT64) as within_region_popularity_score,
    CASE
      WHEN nsfw.probability >= 0.7 THEN true
      WHEN nsfw.probability < 0.4 THEN false
      ELSE NULL
    END as is_nsfw,
    nsfw.probability as probability
  FROM `hot-or-not-feed-intelligence.yral_ds.region_grossing_l7d_candidates` rg
  LEFT JOIN `hot-or-not-feed-intelligence.yral_ds.video_nsfw_agg` nsfw
    ON rg.video_id = nsfw.video_id
  WHERE rg.within_region_popularity_score IS NOT NULL
    AND rg.within_region_popularity_score > 0
)
SELECT
  region,
  COUNTIF(is_nsfw = true) AS nsfw_count,
  COUNTIF(is_nsfw = false) AS clean_count,
  COUNTIF(is_nsfw IS NULL) AS undetermined_count,
  COUNT(*) AS total_count
FROM regional_candidates
GROUP BY region
ORDER BY total_count, clean_count, nsfw_count DESC

--- to run in stage
WITH regional_candidates AS (
  SELECT
    rg.video_id,
    rg.region,
    CAST(rg.within_region_popularity_score AS FLOAT64) as within_region_popularity_score,
    CASE
      WHEN nsfw.probability >= 0.7 THEN true
      WHEN nsfw.probability < 0.4 THEN false
      ELSE NULL
    END as is_nsfw,
    nsfw.probability as probability
  FROM `jay-dhanwant-experiments.stage_tables.stage_region_grossing_l7d_candidates` rg
  LEFT JOIN `jay-dhanwant-experiments.stage_tables.stage_video_nsfw_agg` nsfw
    ON rg.video_id = nsfw.video_id
  WHERE rg.within_region_popularity_score IS NOT NULL
    AND rg.within_region_popularity_score > 0
)
SELECT
  region,
  COUNTIF(is_nsfw = true) AS nsfw_count,
  COUNTIF(is_nsfw = false) AS clean_count,
  COUNTIF(is_nsfw IS NULL) AS undetermined_count,
  COUNT(*) AS total_count
FROM regional_candidates
GROUP BY region
ORDER BY total_count, clean_count, nsfw_count DESC

