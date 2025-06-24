# Docker Setup for Recommendation Service

This document explains how to build and run the recommendation service using Docker with uv for faster package installation.

## Prerequisites

- Docker and Docker Compose installed
- GCP credentials for accessing required GCP services
- BuildKit enabled for Docker (for build caching)

## Dockerfile Options

We provide two Dockerfile options:

1. **Dockerfile** - Basic version that uses uv for faster package installation
2. **Dockerfile.optimized** - Advanced multi-stage build with caching and optimizations

## Building the Docker Image

```bash
# Build the basic image
docker build -t recommendation-service .

# Or build the optimized image with BuildKit enabled
DOCKER_BUILDKIT=1 docker build -f Dockerfile.optimized -t recommendation-service .
```

## Running with Docker

```bash
# Run the container with GCP credentials
docker run -p 8000:8000 -e GCP_CREDENTIALS='$(cat /path/to/credentials.json)' recommendation-service
```

## Running with Docker Compose

1. First, set up your GCP credentials:

```bash
# Option 1: Export credentials as an environment variable
export GCP_CREDENTIALS=$(cat /path/to/credentials.json)

# Option 2: Create a .env file with the credentials
echo "GCP_CREDENTIALS=$(cat /path/to/credentials.json)" > .env
```

2. Run the service using Docker Compose:

```bash
# Use BuildKit for better caching
COMPOSE_DOCKER_CLI_BUILD=1 DOCKER_BUILDKIT=1 docker-compose up --build
```

## Testing the Service

Once the service is running, you can test it with curl:

```bash
# Health check
curl -X GET http://localhost:8000/

# Get recommendations (example)
curl -X POST \
  http://localhost:8000/recommendations \
  -H 'Content-Type: application/json' \
  -d '{
  "user_profile": {
    "user_id": "user123",
    "cluster_id": 1,
    "watch_time_quantile_bin_id": 3,
    "watch_history": [
      {
        "video_id": "video123",
        "last_watched_timestamp": "2025-05-28 06:33:40+00:00",
        "mean_percentage_watched": "0.64"
      }
    ]
  }
}'
```

## Why uv?

We use [uv](https://github.com/astral-sh/uv) instead of pip for several benefits:

1. **Faster installation**: uv can be up to 10-100x faster than pip
2. **Better caching**: uv has improved caching mechanisms
3. **Multi-stage builds**: Better support for Docker multi-stage builds
4. **Reduced image size**: More efficient dependency resolution

## Configuration

The service uses the following environment variables:

- `API_PORT`: Port to run the API server on (default: 8000)
- `GCP_CREDENTIALS`: GCP credentials in JSON format
- `UV_SYSTEM_PYTHON`: Set to 1 to use the system Python environment (already set in Dockerfile)

## Health Check

The Docker Compose configuration includes a health check that periodically checks if the service is responding to requests.