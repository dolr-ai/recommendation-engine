### Purpose

- **What**: Resolve the latest `canister_id` and `post_id` for a piece of content using a Redis mapping, given the previous `canister_id` and `post_id` you already have (typically from `video_index`).
- **Rule**: Build a composite key from the previous values and look it up in Redis. If a mapping exists, use the mapped values (the “current” IDs). If not, keep the previous values.

### Composite key format

- **Key**: `"{old_canister_id}-{old_post_id}"`
- **Example**: `"5jdmx-ciaaa-aaaag-aowxq-cai-506"`
- **Value (Redis Hash)**:
  - `canister_id`: string (the new/current canister ID)
  - `post_id`: string (the new/current post ID; parse to int if your API requires int)

### Relevant environment variables

- These are the variables used by the production service to resolve and connect to Redis for post mapping. Some are operational; the key ones for mapping are noted.

- Redis connection for post mappings (key for this doc):
  - `RECSYS_PROXY_REDIS_HOST` (preferred host used by mapping code)
  - `PROXY_REDIS_HOST` (fallback host if `RECSYS_PROXY_REDIS_HOST` is unset)
  - `RECSYS_PROXY_REDIS_PORT` (port; default 6379 if unset)
  - `RECSYS_SERVICE_REDIS_AUTHKEY` (password/auth key)
  - `RECSYS_SERVICE_REDIS_CLUSTER_ENABLED` (not used by mapping code; mapping uses non-cluster mode)
  - `ENABLE_REDIS_POST_MAPPING` (feature flag used elsewhere; mapping code is effectively always-on if auth is present)

- Other Redis variables you may see (not used by the mapping code path but may exist in your env):
  - `RECSYS_SERVICE_REDIS_HOST` (not used by the mapping code path)
  - `RECSYS_SERVICE_REDIS_PORT` (not used by the mapping code path)
  - `RECSYS_USE_REDIS_PROXY` (informational; not used by the mapping code path)
  - `RECSYS_SERVICE_REDIS_INSTANCE_ID` (informational; not used by the mapping code path)

- GCP/BigQuery (not required to read Redis mapping; provided here for completeness when you also fetch “previous” from `video_index`):
  - `RECSYS_SERVICE_ACCOUNT`
  - `RECSYS_GCP_CREDENTIALS` (JSON string of the service account credential)
  - `RECSYS_PROJECT_ID`
  - `VIDEO_INDEX_TABLE` (optional; defaults to `hot-or-not-feed-intelligence.yral_ds.video_index`)

### Exact lookup logic

- Build composite key using the previous values you already have: `key = f"{old_canister_id}-{old_post_id}"`.
- Perform `HGETALL key` in Redis.
- If the hash exists and contains `canister_id` and `post_id`, use those as the new/current values.
- If the hash does not exist or is empty, keep the previous values (no change).
- Notes:
  - Hash fields are strings. Convert `post_id` to int if your API requires an integer (v1). Keep it as string for v2.
  - Fail fast on missing env vars or connection errors in development.

### Example: redis-cli

```bash
# Using env vars (recommended). Do NOT hardcode secrets in docs or code.
redis-cli -h "$RECSYS_PROXY_REDIS_HOST" -p "$RECSYS_PROXY_REDIS_PORT" -a "$RECSYS_SERVICE_REDIS_AUTHKEY" \
  HGETALL "5jdmx-ciaaa-aaaag-aowxq-cai-506"

# Expected output if mapping exists (fields are strings):
1) "canister_id"
2) "ivkka-7qaaa-aaaas-qbg3q-cai"
3) "post_id"
4) "987654321"
```

### Minimal Python helper (standalone)

```python
import os
from typing import Optional, Dict, Union
import redis

def get_mapped_post(
    old_canister_id: str,
    old_post_id: Union[str, int],
) -> Optional[Dict[str, str]]:
    """
    Fetch new/current canister_id and post_id from Redis, given previous values.

    Algorithm (fail-fast):
      1) Validate inputs are non-empty.
      2) Read connection details from env: RECSYS_PROXY_REDIS_HOST (or PROXY_REDIS_HOST),
         RECSYS_PROXY_REDIS_PORT (default 6379), RECSYS_SERVICE_REDIS_AUTHKEY.
      3) Connect to Redis with decode_responses=True (get strings back).
      4) Compute composite key: "{old_canister_id}-{old_post_id}" (post_id as string).
      5) HGETALL the key; if empty, return None (caller should keep previous values).
      6) If present, return {"canister_id": str, "post_id": str}.

    Args:
      old_canister_id: str; previous canister ID (non-empty)
      old_post_id: str|int; previous post ID

    Returns:
      Dict with schema {"canister_id": str, "post_id": str} if mapping exists; otherwise None.
    """
    if not old_canister_id:
        raise ValueError("old_canister_id must be non-empty")
    if old_post_id is None or old_post_id == "":
        raise ValueError("old_post_id must be provided")

    host = os.environ.get("RECSYS_PROXY_REDIS_HOST") or os.environ.get("PROXY_REDIS_HOST", "localhost")
    port = int(os.environ.get("RECSYS_PROXY_REDIS_PORT", 6379))
    auth = os.environ.get("RECSYS_SERVICE_REDIS_AUTHKEY")
    if not host or not auth:
        raise RuntimeError("Redis mapping env not set: RECSYS_PROXY_REDIS_HOST/PROXY_REDIS_HOST and RECSYS_SERVICE_REDIS_AUTHKEY are required")

    client = redis.Redis(host=host, port=port, password=auth, decode_responses=True, socket_timeout=10)

    key = f"{old_canister_id}-{old_post_id}"
    mapping = client.hgetall(key)
    if not mapping:
        return None  # Keep previous values (no mapping)

    # Only the two fields are relevant for mapping
    canister_id = mapping.get("canister_id")
    post_id = mapping.get("post_id")
    if not canister_id or post_id is None:
        return None

    return {"canister_id": str(canister_id), "post_id": str(post_id)}
```

### End-to-end flow (context)

- Upstream you typically resolve the “previous” pair from BigQuery `video_index` using `video_id`.
- Then apply the Redis mapping in this document to get the “current” pair.
- If Redis has no mapping, you return the “previous” pair.

### Failure modes and handling

- **Missing Redis env**: fail fast in dev; in prod, ensure env is set (see list above).
- **Redis unavailable/timeouts**: treat as “no mapping” and keep previous values; log for observability.
- **Partial/invalid hash fields**: treat as “no mapping”.
- **Type handling**:
  - For v1 APIs: parse `post_id` to int after mapping (raise on failure in dev).
  - For v2 APIs: keep `post_id` as string.

### References in this repository (production path)

- Redis mapping fetch and application are implemented here:
  - `src/recommendation/data/backend.py`:
    - `_get_redis_client(...)`: constructs the Redis client from env.
    - `get_batch_post_mappings_from_redis(...)`: batch HGETALL of composite keys.
    - `get_video_metadata(...)`: merges BigQuery base rows with Redis mappings (overwrites canister/post when mapping exists).

### Quick checklist

- [ ] `RECSYS_PROXY_REDIS_HOST`/`PROXY_REDIS_HOST` set
- [ ] `RECSYS_PROXY_REDIS_PORT` set (or default 6379)
- [ ] `RECSYS_SERVICE_REDIS_AUTHKEY` set
- [ ] `ENABLE_REDIS_POST_MAPPING=true` (recommended for consistency across services)
- [ ] Test with `redis-cli` HGETALL on a known composite key


