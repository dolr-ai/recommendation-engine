# Recommendation Engine

This module provides a modular recommendation engine for generating personalized video recommendations based on user watch history, clustering, and vector similarity.

## Module Structure

- `__init__.py`: Package initialization
- `config.py`: Configuration management
- `similarity.py`: Vector similarity operations
- `candidates.py`: Candidate fetching and processing
- `reranking.py`: Reranking logic for candidates
- `mixer.py`: Mixer algorithm for blending candidates
- `engine.py`: Main recommendation engine class
- `example.py`: Example script demonstrating usage

## Prerequisites

Before using the recommendation engine, you need to set up the GCP credentials environment variable:

```bash
# Export GCP credentials from a JSON file
export GCP_CREDENTIALS=$(jq -c . '/path/to/credentials.json')
```

## Usage

The recommendation engine can be used as follows:

```python
from recommendation.engine import create_recommendation_engine

# Create recommendation engine
engine = create_recommendation_engine()

# Verify connection to vector service
if not engine.verify_connection():
    print("Failed to connect to vector service")
    exit(1)

# Generate recommendations for a user profile
recommendation_results = engine.generate_recommendations(
    user_profile,
    top_k=50,
    fallback_top_k=100,
    min_similarity_threshold=0.4,
    recency_weight=0.7,
    watch_percentage_weight=0.3,
    max_candidates_per_query=10,
    enable_deduplication=True,
)

# Access recommendations
recommendations = recommendation_results["recommendations"]
fallback_recommendations = recommendation_results["fallback_recommendations"]
```

## User Profile Format

The user profile should be a dictionary with the following structure:

```python
user_profile = {
    "user_id": "123",
    "cluster_id": 5,
    "watch_time_quantile_bin_id": 3,
    "watch_history": [
        {
            "video_id": "video1",
            "last_watched_timestamp": 1625097600,
            "mean_percentage_watched": 0.8
        },
        {
            "video_id": "video2",
            "last_watched_timestamp": 1625011200,
            "mean_percentage_watched": 0.6
        },
        # ...
    ]
}
```

## Recommendation Process

1. **Candidate Fetching**: Fetches candidates from various sources based on user's cluster and watch time quantile bin
2. **Reranking**: Calculates similarity scores for candidates using vector embeddings
3. **Mixing**: Blends candidates from different sources based on weights and scores
4. **Final Recommendations**: Returns top-k recommendations and fallback recommendations

## Performance Optimizations

### Fallback Candidate Caching

The system implements caching for fallback candidates to avoid repeatedly fetching the same data:

- Fallback candidates are cached by cluster_id in the CandidateManager
- Once fetched, the same fallback candidates are reused for all subsequent operations
- Video embeddings are also cached in the SimilarityManager to avoid repeated vector lookups
- This significantly improves performance when processing recommendations for a user

To specify the maximum number of fallback candidates to cache:

```python
recommendations = engine.get_recommendations(
    user_profile,
    max_fallback_candidates=200,  # Will cache up to 200 fallback candidates per cluster/type
    # Other parameters...
)
```

## Configuration

The recommendation engine can be configured with custom candidate types and weights:

```python
candidate_types = {
    1: {"name": "watch_time_quantile", "weight": 1.0},
    2: {"name": "modified_iou", "weight": 0.8},
    3: {"name": "fallback_watch_time_quantile", "weight": 0.6},
    4: {"name": "fallback_modified_iou", "weight": 0.5},
}

engine = create_recommendation_engine(
    candidate_types=candidate_types,
)
```

## Example

See `example.py` for a complete example of using the recommendation engine.