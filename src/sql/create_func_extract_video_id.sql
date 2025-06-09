CREATE OR REPLACE FUNCTION
  `jay-dhanwant-experiments.stage_test_tables.extract_video_id` (uri STRING) RETURNS STRING LANGUAGE js AS """
  if (!uri) return null;
  const match = uri.match(/([^\\/]+)\\.mp4$/);
  return match ? match[1] : null;
""";
