-- derive watch time bin it will be a realtime query
-- we will need to store cluster (for history filtering), quantile data in redis as well
WITH user_watch_time AS (
  SELECT
    cluster_id,
    user_id,
    SUM(mean_percentage_watched) * 60 as total_watch_time
  FROM `jay-dhanwant-experiments.stage_test_tables.test_user_clusters`
  WHERE user_id = '5ewkg-pzewc-dg53q-2mcwx-34cku-mwmf2-dedap-267oj-5cloa-y4ue6-4qe'
  GROUP BY cluster_id, user_id
)
SELECT
  uwt.cluster_id,
  uwt.user_id,
  uwt.total_watch_time,
  cp.percentile_25,
  cp.percentile_50,
  cp.percentile_75,
  cp.percentile_100,
  cp.user_count,
  CASE
    -- todo:
    -- check if these bins fit the logic
    -- will you compare with higher bin or lower bin when you get it?
    -- if you get bin = 3 what will you do? this is the highest bin
    -- will you compare with all the bins out there? or will you skip it?
    -- you will also need to store this metadata in redis for service to use it
    -- have a good structure of keys to identify them plus you can also refresh them easily
    WHEN uwt.total_watch_time <= cp.percentile_25 THEN 0
    WHEN uwt.total_watch_time <= cp.percentile_50 THEN 1
    WHEN uwt.total_watch_time <= cp.percentile_75 THEN 2
    WHEN uwt.total_watch_time <= cp.percentile_100 THEN 3
    ELSE -1
  END as percentile_bin_number
FROM user_watch_time uwt
JOIN `jay-dhanwant-experiments.stage_test_tables.user_watch_time_quantile_bins` cp
  ON uwt.cluster_id = cp.cluster_id


-- get cluster_id of user and corresponding history
SELECT cluster_id, user_id, video_id, last_watched_timestamp, mean_percentage_watched
FROM `jay-dhanwant-experiments.stage_test_tables.test_user_clusters`
WHERE user_id = '5ewkg-pzewc-dg53q-2mcwx-34cku-mwmf2-dedap-267oj-5cloa-y4ue6-4qe'

SELECT *
FROM `jay-dhanwant-experiments.stage_test_tables.watch_time_quantile_candidates`
WHERE cluster_id = 7 -- fetch from above query
AND bin = 3 -- fetch from above query
AND query_video_id IN (
    SELECT video_id
    FROM `jay-dhanwant-experiments.stage_test_tables.test_user_clusters`
    WHERE user_id = '5ewkg-pzewc-dg53q-2mcwx-34cku-mwmf2-dedap-267oj-5cloa-y4ue6-4qe'
)