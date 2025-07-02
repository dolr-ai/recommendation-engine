"""
FastAPI application for the recommendation service.

This module provides a FastAPI application for serving recommendations.
"""

import os
import json
import traceback
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


# Initialize service on startup
@app.on_event("startup")
async def startup_event():
    """Initialize recommendation service on startup."""
    logger.info("Starting up recommendation service")
    try:
        # Initialize the recommendation service singleton
        RecommendationService.get_instance()
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
    error_message = str(exc)

    # Extract the raw response to return it directly
    error_detail = {
        "detail": "Response validation error",
        "errors": exc.errors(),
    }

    # Return the raw response to bypass validation
    return JSONResponse(
        status_code=500,
        content=error_detail,
    )


@app.get("/", tags=["Health"])
async def health_check():
    """Health check endpoint."""
    return {"status": "ok", "message": "Recommendation service is running"}


@app.post(
    "/recommendations",
    tags=["Recommendations"],
    response_model_exclude_none=True,
)
async def get_recommendations(request: RecommendationRequest):
    """
    Get personalized video recommendations for a user.
    Metadata (cluster_id and watch_time_quantile_bin_id) will be automatically fetched if not provided.

    Args:
        request: Recommendation request containing user_id and optional parameters including
                exclude_watched_items and exclude_reported_items for real-time filtering

    Returns:
        RecommendationResponse with recommendations and metadata
    """
    logger.info(f"Received recommendation request for user {request.user_id}")

    try:
        # Get recommendation service instance
        service = RecommendationService.get_instance()

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
        recommendations = service.get_recommendations(
            user_profile=user_profile,
            exclude_watched_items=request.exclude_watched_items,
            exclude_reported_items=request.exclude_reported_items,
        )

        # Only include the fields present in the new RecommendationResponse model
        response = {
            "posts": recommendations.get("posts", []),
            "processing_time_ms": recommendations.get("processing_time_ms", 0),
            "error": recommendations.get("error"),
        }
        return JSONResponse(content=response)

    except Exception as e:
        logger.error(f"Error processing recommendation request: {e}", exc_info=True)
        error_detail = {"detail": str(e), "traceback": traceback.format_exc()}
        raise HTTPException(status_code=500, detail=error_detail)


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
    Metadata (cluster_id and watch_time_quantile_bin_id) will be automatically fetched if not provided.

    Args:
        requests: List of recommendation requests with optional exclude_watched_items and exclude_reported_items
        background_tasks: FastAPI background tasks

    Returns:
        List of RecommendationResponse objects
    """
    logger.info(f"Received batch recommendation request for {len(requests)} users")

    results = []
    service = RecommendationService.get_instance()

    for request in requests:
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
            recommendations = service.get_recommendations(
                user_profile=user_profile,
                exclude_watched_items=request.exclude_watched_items,
                exclude_reported_items=request.exclude_reported_items,
            )

            # Only include the fields present in the new RecommendationResponse model
            response = {
                "posts": recommendations.get("posts", []),
                "processing_time_ms": recommendations.get("processing_time_ms", 0),
                "error": recommendations.get("error"),
            }
            results.append(response)

        except Exception as e:
            logger.error(f"Error processing batch request item: {e}", exc_info=True)
            # Return error response for this item
            results.append(
                {
                    "posts": [],
                    "processing_time_ms": 0,
                    "error": str(e),
                }
            )

    return results


def start():
    """Start the FastAPI application."""
    uvicorn.run(
        "service.app:app",
        # host="localhost",
        host="0.0.0.0",
        port=8000,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    start()
