# Recommendation Service API

A FastAPI service for delivering personalized video recommendations.

## Features

- **Single User Recommendations**: Get personalized recommendations for a single user
- **Batch Processing**: Process multiple users in a single request
- **Fallback Candidates**: Uses optimized caching for fallback candidates with probabilistic refresh
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
  "user_profile": {
    "user_id": "user123",
    "cluster_id": 1,
    "watch_time_quantile_bin_id": 3,
    "watch_history": [
      {
        "video_id": "video123",
        "last_watched_timestamp": "2025-05-28 06:33:40.393635+00:00",
        "mean_percentage_watched": "0.64"
      }
    ]
  },
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

### Batch Recommendations

```
POST /recommendations/batch
```

Request body:

```json
[
  {
    "user_profile": {
      "user_id": "user123",
      "cluster_id": 1,
      "watch_time_quantile_bin_id": 3,
      "watch_history": [...]
    },
    "top_k": 50
  },
  {
    "user_profile": {
      "user_id": "user456",
      "cluster_id": 2,
      "watch_time_quantile_bin_id": 4,
      "watch_history": [...]
    },
    "top_k": 30
  }
]
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

The service implements caching for fallback candidates to avoid repeatedly fetching the same data:

- Fallback candidates are cached by cluster_id
- A probabilistic refresh mechanism ensures that the cache is periodically refreshed
- By default, there's a 10% chance of refreshing the cache on each request
- You can configure this probability using the `FALLBACK_CACHE_REFRESH_PROBABILITY` environment variable:

```bash
# Set to 20% refresh probability
export FALLBACK_CACHE_REFRESH_PROBABILITY=0.2
```

- Video embeddings are also cached to avoid repeated vector lookups
- This significantly improves performance while ensuring fresh recommendations over time

## todo
1. nsfw filter on/off handle
1. add more candidate types