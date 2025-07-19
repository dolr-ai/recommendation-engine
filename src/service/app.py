"""
Optimized FastAPI application for recommendation service.
Performance improvements for high-concurrency scenarios.
"""

import os
import json
import traceback
import asyncio
import time
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError, ResponseValidationError
import uvicorn

from utils.common_utils import get_logger
from service.models import (
    RecommendationRequest,
    RecommendationResponse,
)
from service.recommendation_service import RecommendationService

logger = get_logger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Recommendation Engine API",
    description="API for serving personalized video recommendations",
    version="1.0.0",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global service instance
recommendation_service: Optional[RecommendationService] = None


# Initialize service on startup
@app.on_event("startup")
async def startup_event():
    """Initialize recommendation service on startup."""
    global recommendation_service
    logger.info("Starting up recommendation service")
    try:
        # Initialize the recommendation service singleton
        recommendation_service = RecommendationService.get_instance()
        logger.info("Recommendation service initialized")
    except Exception as e:
        logger.error(f"Failed to initialize recommendation service: {e}")
        raise


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """Handle request validation errors."""
    logger.error(f"Request validation error: {exc}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors(), "body": exc.body},
    )


@app.exception_handler(ResponseValidationError)
async def response_validation_exception_handler(
    request: Request, exc: ResponseValidationError
):
    """Handle response validation errors."""
    logger.error(f"Response validation error: {exc}")
    error_detail = {
        "detail": "Response validation error",
        "errors": exc.errors(),
    }
    return JSONResponse(
        status_code=500,
        content=error_detail,
    )


@app.get("/", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "message": "Recommendation service is running"}


@app.get("/debug", tags=["Debug"])
async def debug_info():
    """Debug endpoint to check service status."""
    global recommendation_service
    import psutil
    import threading
    from utils.gcp_utils import get_bigquery_client_stats

    return {
        "status": "ok",
        "service_initialized": recommendation_service is not None,
        "active_threads": threading.active_count(),
        "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
        "cpu_percent": psutil.Process().cpu_percent(),
        "bigquery_client_stats": get_bigquery_client_stats(),
    }


def process_recommendation_sync(request: RecommendationRequest) -> dict:
    """
    Synchronous recommendation processing to avoid async/await overhead.
    This is the main optimization - your RecommendationService.get_recommendations
    is likely CPU-bound and doesn't benefit from async.
    """
    global recommendation_service

    start_time = time.time()

    try:
        # Convert watch history items to dictionaries
        watch_history = []
        if request.watch_history:
            for item in request.watch_history:
                watch_history.append(
                    {
                        "video_id": item.video_id,
                        "last_watched_timestamp": item.last_watched_timestamp,
                        "mean_percentage_watched": item.mean_percentage_watched,
                    }
                )

        # Create user profile from request
        user_profile = {
            "user_id": request.user_id,
            "cluster_id": None,  # Will be auto-fetched
            "watch_time_quantile_bin_id": None,  # Will be auto-fetched
            "watch_history": watch_history,
        }

        # Get recommendations
        # NOTE: this is for testing purposes only - we should use the async version
        recommendations = recommendation_service.get_recommendations(
            user_profile=user_profile,
            nsfw_label=request.nsfw_label,
            exclude_watched_items=request.exclude_watched_items,
            exclude_reported_items=request.exclude_reported_items,
            exclude_items=request.exclude_items,
            num_results=request.num_results,
        )

        processing_time = (time.time() - start_time) * 1000

        response = {
            "posts": recommendations.get("posts", []),
            "processing_time_ms": processing_time,
            "error": recommendations.get("error"),
            "debug": recommendations.get("debug"),
        }

        logger.info(
            f"Processed recommendation for user {request.user_id} in {processing_time:.2f}ms"
        )
        return response

    except Exception as e:
        processing_time = (time.time() - start_time) * 1000
        logger.error(f"Error processing recommendation request: {e}", exc_info=True)
        return {
            "posts": [],
            "processing_time_ms": processing_time,
            "error": str(e),
            "debug": None,
        }


@app.post(
    "/recommendations",
    tags=["Recommendations"],
    response_model_exclude_none=True,
)
async def get_recommendations(request: RecommendationRequest):
    """
    Get personalized video recommendations for a user.
    Optimized version with better error handling and timing.
    """
    logger.info(f"Received recommendation request for user {request.user_id}")

    if recommendation_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        # Process in thread pool to avoid blocking the event loop
        # This allows FastAPI to handle other requests while this one processes
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, process_recommendation_sync, request
        )

        if response.get("error"):
            raise HTTPException(status_code=500, detail=response["error"])

        return JSONResponse(content=response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing recommendation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


@app.post(
    "/recommendations/batch",
    tags=["Recommendations"],
    response_model_exclude_none=True,
)
async def get_batch_recommendations(
    requests: list[RecommendationRequest], background_tasks: BackgroundTasks
):
    """
    Get recommendations for multiple users in batch.
    Optimized for concurrent processing.
    """
    logger.info(f"Received batch recommendation request for {len(requests)} users")

    if recommendation_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Process requests concurrently using thread pool
    loop = asyncio.get_event_loop()

    # Create tasks for concurrent processing
    tasks = []
    for request in requests:
        task = loop.run_in_executor(None, process_recommendation_sync, request)
        tasks.append(task)

    # Wait for all tasks to complete
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # Handle any exceptions
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.error(f"Batch request {i} failed: {result}")
            processed_results.append(
                {
                    "posts": [],
                    "processing_time_ms": 0,
                    "error": str(result),
                }
            )
        else:
            processed_results.append(result)

    return processed_results


def start():
    """Start the FastAPI application with optimized settings."""
    # CRITICAL: These settings are optimized for your hardware
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "service.app:app",
        # host="localhost",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
        workers=int(os.environ.get("WORKERS", 32)),
        access_log=False,
        # limit_concurrency=200,
        # backlog=500,
        # For multiple workers, use:
        # workers=2,  # 2 workers for 4 vCPUs
        # worker_class="uvicorn.workers.UvicornWorker"
    )


if __name__ == "__main__":
    start()
