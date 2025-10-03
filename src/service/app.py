"""
Optimized FastAPI application for recommendation service.
Performance improvements for high-concurrency scenarios.
"""

import os
import json
import traceback
import asyncio
import time
import datetime
from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError, ResponseValidationError
import uvicorn
import httpx
import sentry_sdk
from sentry_sdk.integrations.starlette import StarletteIntegration
from sentry_sdk.integrations.fastapi import FastApiIntegration

from utils.common_utils import get_logger
from service.models import (
    RecommendationRequest,
    RecommendationResponse,
    create_safe_response,
    validate_and_sanitize_response,
)
from history.get_realtime_history_items import UserRealtimeHistory
from service.recommendation_service import RecommendationService
from service.fallback_recommendation_service import FallbackRecommendationService

# Initialize Sentry for error monitoring, performance tracking, and profiling
sentry_sdk.init(
    integrations=[
        StarletteIntegration(
            transaction_style="endpoint",
            failed_request_status_codes={403, *range(500, 599)},
            http_methods_to_capture=("GET", "POST"),
        ),
        FastApiIntegration(
            transaction_style="endpoint",
            failed_request_status_codes={403, *range(500, 599)},
            http_methods_to_capture=("GET", "POST"),
        ),
    ],
    traces_sample_rate=1.0,  # Capture 100% of transactions for performance monitoring
    enable_tracing=True,
    profiles_sample_rate=0,  # Capture 100% of profiles for performance profiling
)

logger = get_logger(__name__)

# max items to consider while recommending content
MAX_UNIQUE_HISTORY_ITEMS = 100

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
    global user_realtime_history
    global fallback_recommendation_service

    # Initialize services
    logger.info("Starting up recommendation service")
    try:
        # Initialize the recommendation service singleton
        recommendation_service = RecommendationService.get_instance()
        logger.info("Recommendation service initialized")
        user_realtime_history = UserRealtimeHistory()
        logger.info("User realtime history initialized")
        fallback_recommendation_service = FallbackRecommendationService.get_instance()
        logger.info("Fallback recommendation service initialized")
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


async def get_region_from_ip(ip_address: str) -> Optional[str]:
    """
    Get region from IP address using marketing analytics server API.

    Args:
        ip_address: IP address to lookup

    Returns:
        Region string or None if lookup fails
    """
    try:
        auth_token = os.environ.get("MARKETING_ANALYTICS_AUTH_TOKEN")
        async with httpx.AsyncClient() as client:
            response = await client.get(
                f"{os.environ.get('MARKETING_ANALYTICS_SERVER_BASE_URL')}/api/ip/{ip_address}",
                headers={
                    "Content-Type": "application/json",
                    "Authorization": f"Bearer {auth_token}",
                },
                timeout=5.0,  # 5 second timeout
            )

            if response.status_code == 200:
                data = response.json()
                region = data.get("city") or data.get("region") or data.get("country")
                logger.info(f"IP {ip_address} resolved to region: {region}")
                return region
            else:
                logger.warning(
                    f"Failed to get region for IP {ip_address}: {response.status_code}"
                )
                return None

    except Exception as e:
        logger.error(f"Error getting region from IP {ip_address}: {e}")
        return None


@app.get("/", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "message": "Recommendation service is running"}


@app.get("/version", tags=["Health"])
async def version_info():
    """Version endpoint to check deployment info."""
    import subprocess

    # Try to get git commit hash
    git_commit = "unknown"
    try:
        # Find git root by going up from current file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # Go up to project root (src/service -> src -> root)
        git_root = os.path.dirname(os.path.dirname(current_dir))

        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            timeout=2,
            cwd=git_root,
        )
        if result.returncode == 0:
            git_commit = result.stdout.strip()[:8]  # Short hash
    except Exception:
        pass

    # Get deployment timestamp from environment or file modification time
    deploy_time = os.environ.get("DEPLOY_TIMESTAMP", "unknown")

    return {
        "version": "2025-10-03-backend-fixes",
        "git_commit": git_commit,
        "deploy_time": deploy_time,
        "backend_fix_timestamp": "2025-10-03T12:30:00",
        "has_redis_mapping_fixes": True,
        "has_uuid_handling": True,
        "python_version": f"{os.sys.version_info.major}.{os.sys.version_info.minor}.{os.sys.version_info.micro}",
    }


@app.get("/debug", tags=["Debug"])
async def debug_info():
    """Debug endpoint to check service status."""
    global recommendation_service
    import psutil
    import threading
    from utils.gcp_utils import (
        get_bigquery_client_stats,
        ValkeyConnectionManager,
        ValkeyThreadPoolManager,
    )
    from recommendation.data.backend import get_video_metadata_cache_stats

    # Get Valkey connection and thread pool stats
    try:
        valkey_conn_manager = ValkeyConnectionManager()
        valkey_conn_stats = valkey_conn_manager.get_connection_stats()
    except Exception as e:
        valkey_conn_stats = {"error": str(e)}

    try:
        valkey_thread_manager = ValkeyThreadPoolManager()
        valkey_thread_stats = valkey_thread_manager.get_stats()
    except Exception as e:
        valkey_thread_stats = {"error": str(e)}

    # Get video metadata cache stats
    try:
        video_cache_stats = get_video_metadata_cache_stats()
    except Exception as e:
        video_cache_stats = {"error": str(e)}

    return {
        "status": "ok",
        "service_initialized": recommendation_service is not None,
        "active_threads": threading.active_count(),
        "memory_usage_mb": psutil.Process().memory_info().rss / 1024 / 1024,
        "cpu_percent": psutil.Process().cpu_percent(),
        "bigquery_client_stats": get_bigquery_client_stats(),
        "valkey_connection_stats": valkey_conn_stats,
        "valkey_thread_stats": valkey_thread_stats,
        "video_metadata_cache_stats": video_cache_stats,
    }


def process_recommendation_sync(
    request: RecommendationRequest, post_id_as_string: bool = False
) -> dict:
    """
    Synchronous recommendation processing to avoid async/await overhead.
    This is the main optimization - your RecommendationService.get_recommendations
    is likely CPU-bound and doesn't benefit from async.
    """
    global recommendation_service

    start_time = time.time()

    try:
        # Convert watch history items to dictionaries
        epoch_time = int(datetime.datetime.now().timestamp())
        watch_history = (
            user_realtime_history.get_history_items_for_recommendation_service(
                user_id=request.user_id,
                start=0,
                end=epoch_time,
                buffer=10_000,
                max_unique_history_items=MAX_UNIQUE_HISTORY_ITEMS,
                nsfw_label=request.nsfw_label,
            )
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
            region=request.region,
            post_id_as_string=post_id_as_string,
            dev_inject_video_ids=request.dev_inject_video_ids,
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
        return create_safe_response(
            posts=[],
            processing_time_ms=processing_time,
            error=str(e),
            debug=None,
        )


@app.post(
    "/recommendations/cache",
    tags=["CacheRecommendations"],
    response_model_exclude_none=True,
)
async def get_cache_recommendations(
    request: RecommendationRequest, http_request: Request
):
    """
    Get recommendations from cache.
    """
    logger.info(f"Received cache recommendation request for user {request.user_id}")

    # Start profiling for cache recommendations
    sentry_sdk.profiler.start_profiler()

    try:
        # Get X-Forwarded-For header for logging
        x_forwarded_for = http_request.headers.get("X-Forwarded-For")

        # If IP address is not provided, try to get it from X-Forwarded-For header
        ip_address = request.ip_address
        if not ip_address:
            if x_forwarded_for:
                # Take the first IP in case of multiple forwarded IPs
                ip_address = x_forwarded_for.split(",")[0].strip()
                logger.info(
                    f"Using IP from X-Forwarded-For header: {ip_address}, X-Forwarded-For: {x_forwarded_for}"
                )
            else:
                logger.info(
                    "No IP address provided in request and no X-Forwarded-For header found"
                )
        else:
            logger.info(
                f"Using IP from request: {ip_address}, X-Forwarded-For: {x_forwarded_for or 'Not present'}"
            )

        # If IP address is available but region is not, resolve region from IP
        if ip_address and not request.region:
            region = await get_region_from_ip(ip_address)
            if region:
                request.region = region
                logger.info(f"Resolved region '{region}' from IP address: {ip_address}")
            else:
                logger.warning(
                    f"Could not resolve region from IP address: {ip_address}"
                )

        recommendations = fallback_recommendation_service.get_cached_recommendations(
            user_id=request.user_id,
            nsfw_label=request.nsfw_label,
            num_results=request.num_results,
            region=request.region,
            dev_inject_video_ids=request.dev_inject_video_ids,
        )

        # Validate and sanitize response to ensure error is empty string for success
        safe_recommendations = validate_and_sanitize_response(recommendations)
        return JSONResponse(content=safe_recommendations)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting cache recommendations: {e}", exc_info=True)
        # Return safe response instead of raising exception to prevent feed break
        safe_error_response = create_safe_response(error=str(e))
        return JSONResponse(content=safe_error_response, status_code=500)
    finally:
        # Stop profiling for cache recommendations
        sentry_sdk.profiler.stop_profiler()


@app.post(
    "/recommendations",
    tags=["Recommendations"],
    response_model_exclude_none=True,
)
async def get_recommendations(request: RecommendationRequest, http_request: Request):
    """
    Get personalized video recommendations for a user.
    Optimized version with better error handling and timing.

    TODO: REVERT THIS CHANGE IN NEXT FEW ITERATIONS
    TEMPORARY REDIRECT: All /recommendations requests now redirect to /recommendations/cache
    WHAT CHANGED: Added immediate call to cache endpoint as first line, making all below code unreachable
    TO REVERT: Remove the return statement below and let the original logic execute
    """
    # IMMEDIATE REDIRECT: First line calls cache endpoint, bypassing all recommendation model logic
    return await get_cache_recommendations(request, http_request)

    # UNREACHABLE CODE BELOW - Original recommendation logic preserved for easy revert
    logger.info(f"Received recommendation request for user {request.user_id}")

    if recommendation_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Start profiling for main recommendations
    sentry_sdk.profiler.start_profiler()

    try:
        # Get X-Forwarded-For header for logging
        x_forwarded_for = http_request.headers.get("X-Forwarded-For")

        # If IP address is not provided, try to get it from X-Forwarded-For header
        ip_address = request.ip_address
        if not ip_address:
            if x_forwarded_for:
                # Take the first IP in case of multiple forwarded IPs
                ip_address = x_forwarded_for.split(",")[0].strip()
                logger.info(
                    f"Using IP from X-Forwarded-For header: {ip_address}, X-Forwarded-For: {x_forwarded_for}"
                )
            else:
                logger.info(
                    "No IP address provided in request and no X-Forwarded-For header found"
                )
        else:
            logger.info(
                f"Using IP from request: {ip_address}, X-Forwarded-For: {x_forwarded_for or 'Not present'}"
            )

        # If IP address is available but region is not, resolve region from IP
        if ip_address and not request.region:
            region = await get_region_from_ip(ip_address)
            if region:
                request.region = region
                logger.info(f"Resolved region '{region}' from IP address: {ip_address}")
            else:
                logger.warning(
                    f"Could not resolve region from IP address: {ip_address}"
                )

        # Process in thread pool to avoid blocking the event loop
        # This allows FastAPI to handle other requests while this one processes
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, process_recommendation_sync, request
        )

        if response.get("error"):
            # Return safe error response instead of raising exception
            safe_error_response = validate_and_sanitize_response(response)
            return JSONResponse(content=safe_error_response, status_code=500)

        # Validate and sanitize successful response
        safe_response = validate_and_sanitize_response(response)
        return JSONResponse(content=safe_response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing recommendation: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Stop profiling for main recommendations
        sentry_sdk.profiler.stop_profiler()


@app.post(
    "/recommendations/batch",
    tags=["Recommendations"],
    response_model_exclude_none=True,
)
async def get_batch_recommendations(requests: list[RecommendationRequest]):
    """
    Get recommendations for multiple users in batch.
    Optimized for concurrent processing.
    """
    logger.info(f"Received batch recommendation request for {len(requests)} users")

    if recommendation_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Start profiling for batch recommendations
    sentry_sdk.profiler.start_profiler()

    try:
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
                    create_safe_response(
                        posts=[], processing_time_ms=0.0, error=str(result)
                    )
                )
            else:
                # Validate and sanitize successful results too
                safe_result = (
                    validate_and_sanitize_response(result)
                    if isinstance(result, dict)
                    else result
                )
                processed_results.append(safe_result)

        return processed_results
    except Exception as e:
        logger.error(f"Error processing batch recommendations: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Stop profiling for batch recommendations
        sentry_sdk.profiler.stop_profiler()


# V2 API Endpoints (with post_id as string)
@app.post(
    "/v2/recommendations",
    tags=["V2 Recommendations"],
    response_model_exclude_none=True,
)
async def get_recommendations_v2(request: RecommendationRequest, http_request: Request):
    """
    Get personalized video recommendations for a user (V2 API with post_id as string).

    # TODO: Revert once experiment is done
    TEMPORARY REDIRECT: All /v2/recommendations requests now redirect to /v2/recommendations/cache
    WHAT CHANGED: Added immediate call to cache endpoint as first line, making all below code unreachable
    TO REVERT: Remove the return statement below and let the original logic execute
    """
    # IMMEDIATE REDIRECT: First line calls cache endpoint, bypassing all recommendation model logic
    return await get_cache_recommendations_v2(request, http_request)

    # UNREACHABLE CODE BELOW - Original recommendation logic preserved for easy revert
    logger.info(f"Received v2 recommendation request for user {request.user_id}")

    if recommendation_service is None:
        raise HTTPException(status_code=503, detail="Service not initialized")

    # Start profiling for V2 recommendations
    sentry_sdk.profiler.start_profiler()

    try:
        # Get X-Forwarded-For header for logging
        x_forwarded_for = http_request.headers.get("X-Forwarded-For")

        # If IP address is not provided, try to get it from X-Forwarded-For header
        ip_address = request.ip_address
        if not ip_address:
            if x_forwarded_for:
                # Take the first IP in case of multiple forwarded IPs
                ip_address = x_forwarded_for.split(",")[0].strip()
                logger.info(
                    f"Using IP from X-Forwarded-For header: {ip_address}, X-Forwarded-For: {x_forwarded_for}"
                )
            else:
                logger.info(
                    "No IP address provided in request and no X-Forwarded-For header found"
                )
        else:
            logger.info(
                f"Using IP from request: {ip_address}, X-Forwarded-For: {x_forwarded_for or 'Not present'}"
            )

        # If IP address is available but region is not, resolve region from IP
        if ip_address and not request.region:
            region = await get_region_from_ip(ip_address)
            if region:
                request.region = region
                logger.info(f"Resolved region '{region}' from IP address: {ip_address}")
            else:
                logger.warning(
                    f"Could not resolve region from IP address: {ip_address}"
                )

        # Process in thread pool to avoid blocking the event loop
        # This allows FastAPI to handle other requests while this one processes
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None, lambda: process_recommendation_sync(request, post_id_as_string=True)
        )

        if response.get("error"):
            # Return safe error response instead of raising exception
            safe_error_response = validate_and_sanitize_response(response)
            return JSONResponse(content=safe_error_response, status_code=500)

        # Validate and sanitize successful response
        safe_response = validate_and_sanitize_response(response)
        return JSONResponse(content=safe_response)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            f"Unexpected error processing v2 recommendation: {e}", exc_info=True
        )
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Stop profiling for V2 recommendations
        sentry_sdk.profiler.stop_profiler()


@app.post(
    "/v2/recommendations/cache",
    tags=["V2 CacheRecommendations"],
    response_model_exclude_none=True,
)
async def get_cache_recommendations_v2(
    request: RecommendationRequest, http_request: Request
):
    """
    Get recommendations from cache (V2 API with post_id as string).
    """
    logger.info(f"Received v2 cache recommendation request for user {request.user_id}")

    # Start profiling for V2 cache recommendations
    sentry_sdk.profiler.start_profiler()

    try:
        # Get X-Forwarded-For header for logging
        x_forwarded_for = http_request.headers.get("X-Forwarded-For")

        # If IP address is not provided, try to get it from X-Forwarded-For header
        ip_address = request.ip_address
        if not ip_address:
            if x_forwarded_for:
                # Take the first IP in case of multiple forwarded IPs
                ip_address = x_forwarded_for.split(",")[0].strip()
                logger.info(
                    f"Using IP from X-Forwarded-For header: {ip_address}, X-Forwarded-For: {x_forwarded_for}"
                )
            else:
                logger.info(
                    "No IP address provided in request and no X-Forwarded-For header found"
                )
        else:
            logger.info(
                f"Using IP from request: {ip_address}, X-Forwarded-For: {x_forwarded_for or 'Not present'}"
            )

        # If IP address is available but region is not, resolve region from IP
        if ip_address and not request.region:
            region = await get_region_from_ip(ip_address)
            if region:
                request.region = region
                logger.info(f"Resolved region '{region}' from IP address: {ip_address}")
            else:
                logger.warning(
                    f"Could not resolve region from IP address: {ip_address}"
                )

        # Get recommendations with post_id as string for v2 API
        recommendations = fallback_recommendation_service.get_cached_recommendations(
            user_id=request.user_id,
            nsfw_label=request.nsfw_label,
            num_results=request.num_results,
            region=request.region,
            post_id_as_string=True,
            dev_inject_video_ids=request.dev_inject_video_ids,
        )

        # Validate and sanitize response to ensure error is empty string for success
        safe_recommendations = validate_and_sanitize_response(recommendations)
        return JSONResponse(content=safe_recommendations)
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting v2 cache recommendations: {e}", exc_info=True)
        # Return safe response instead of raising exception to prevent feed break
        safe_error_response = create_safe_response(error=str(e))
        return JSONResponse(content=safe_error_response, status_code=500)
    finally:
        # Stop profiling for V2 cache recommendations
        sentry_sdk.profiler.stop_profiler()


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
        # workers=int(os.environ.get("WORKERS", 16)),
        workers=16,
        access_log=False,
        # limit_concurrency=200,
        # backlog=500,
        # For multiple workers, use:
        # workers=2,  # 2 workers for 4 vCPUs
        # worker_class="uvicorn.workers.UvicornWorker"
    )


if __name__ == "__main__":
    start()
