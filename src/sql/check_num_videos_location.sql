WITH regional_candidates AS (
    SELECT
    rg.video_id,
    rg.region,
    CAST(rg.within_region_popularity_score AS FLOAT64) as within_region_popularity_score,
    nsfw.probability,
    CASE
        WHEN nsfw.probability >= 0.7 THEN true
        WHEN nsfw.probability < 0.4 THEN false
        ELSE NULL
    END as nsfw_label
    FROM `jay-dhanwant-experiments.stage_tables.stage_region_grossing_l7d_candidates` rg
    LEFT JOIN `jay-dhanwant-experiments.stage_tables.stage_video_nsfw_agg` nsfw
    ON rg.video_id = nsfw.video_id
    WHERE rg.within_region_popularity_score IS NOT NULL
    AND rg.within_region_popularity_score > 0
)
SELECT
    region,
    COUNT(video_id) AS num_items,
    COUNTIF(nsfw_label IS NULL) AS num_null_nsfw_label,
    COUNTIF(nsfw_label IS NOT NULL) AS num_nonnull_nsfw_label,
    COUNTIF(nsfw_label = {self.nsfw_label}) AS num_matching_label,
    ARRAY_AGG(STRUCT(video_id, probability) ORDER BY probability DESC LIMIT 5) AS example_videos
FROM regional_candidates
GROUP BY region
ORDER BY num_items DESC
LIMIT 20