# Recommendation Service API

A FastAPI service for delivering personalized video recommendations.

## Features

- **Single User Recommendations**: Get personalized recommendations for a single user
- **Batch Processing**: Process multiple users in a single request
- **Automatic Metadata Fetching**: Automatically fetch user metadata (cluster_id, watch_time_quantile_bin_id) when not provided
- **Fallback Candidates**: Provides fallback recommendations when personalized ones are insufficient
- **Configurable Parameters**: Customize recommendation behavior through API parameters

## API Endpoints

### Health Check

```
GET /
```

Returns the status of the service.

### Recommendations for a Single User

```
POST /recommendations
```

Request body:

```json
{
  "user_id": "user123",
  "watch_history": [
    {
      "video_id": "video123",
      "last_watched_timestamp": "2025-05-28 06:33:40.393635+00:00",
      "mean_percentage_watched": "0.64"
    }
  ],
  "top_k": 50,
  "fallback_top_k": 100,
  "threshold": 0.1,
  "enable_deduplication": true,
  "max_workers": 4,
  "max_fallback_candidates": 200,
  "min_similarity_threshold": 0.4,
  "recency_weight": 0.8,
  "watch_percentage_weight": 0.2
}
```

**Note**: The `watch_history` field is optional. If not provided, an empty list will be used.

### Batch Recommendations

```
POST /recommendations/batch
```

Request body:

```json
[
  {
    "user_id": "user123",
    "watch_history": [...],
    "top_k": 50
  },
  {
    "user_id": "user456",
    "top_k": 30
  }
]
```

## Metadata Service

The service includes an internal metadata service that automatically fetches user metadata when needed:

- **Cluster ID**: User's assigned cluster based on their behavior patterns
- **Watch Time Quantile Bin ID**: User's watch time percentile within their cluster (0-3)

The metadata service uses the existing `UserClusterWatchTimeFetcher` and `UserWatchTimeQuantileBinsFetcher` from the candidate cache to retrieve this information from Valkey.

### Configuration

The metadata service can be configured using environment variables:

```bash
# Valkey connection settings (optional, uses defaults if not set)
export VALKEY_HOST="10.128.15.210"
export VALKEY_PORT="6379"

# GCP credentials (required)
export GCP_CREDENTIALS=$(jq -c . '/path/to/credentials.json')
```

## Running the Service

Make sure you have the GCP credentials set:

```bash
export GCP_CREDENTIALS=$(jq -c . '/path/to/credentials.json')
```

Start the service:

```bash
python -m src.service.run
```

By default, the service listens on port 8000. You can change this by setting the `API_PORT` environment variable:

```bash
export API_PORT=8080
python -m src.service.run
```

## Performance Optimization

The service implements several performance optimizations:

- BigQuery-based similarity calculations for efficient vector operations
- Parallel processing of different candidate types
- Video embeddings are cached to avoid repeated vector lookups
- These optimizations significantly improve performance while ensuring fresh recommendations

## Error Handling

The service handles various error scenarios gracefully:

- **Missing User Metadata**: If a user's metadata cannot be fetched, the request will return an error response
- **Invalid User ID**: If the user ID is not found in the system, appropriate error messages are returned
- **Service Unavailable**: If the metadata service or recommendation engine is unavailable, error responses are returned

## todo
1. nsfw filter on/off handle
1. add more candidate types