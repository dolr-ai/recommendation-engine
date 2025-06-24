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
from service.models import RecommendationRequest, RecommendationResponse
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

    Args:
        request: Recommendation request containing user profile and parameters

    Returns:
        RecommendationResponse with recommendations and metadata
    """
    logger.info(
        f"Received recommendation request for user {request.user_profile.user_id}"
    )

    try:
        # Get recommendation service instance
        service = RecommendationService.get_instance()

        # Convert user profile to dictionary
        user_profile = request.user_profile.dict()

        # Get recommendations
        recommendations = service.get_recommendations(
            user_profile=user_profile,
            top_k=request.top_k,
            fallback_top_k=request.fallback_top_k,
            threshold=request.threshold,
            enable_deduplication=request.enable_deduplication,
            max_workers=request.max_workers,
            max_fallback_candidates=request.max_fallback_candidates,
            min_similarity_threshold=request.min_similarity_threshold,
            recency_weight=request.recency_weight,
            watch_percentage_weight=request.watch_percentage_weight,
        )

        # Check for error
        if "error" in recommendations:
            raise HTTPException(status_code=500, detail=recommendations["error"])

        # Return response directly to avoid validation issues
        return JSONResponse(content=recommendations)

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

    Args:
        requests: List of recommendation requests
        background_tasks: FastAPI background tasks

    Returns:
        List of RecommendationResponse objects
    """
    logger.info(f"Received batch recommendation request for {len(requests)} users")

    results = []
    service = RecommendationService.get_instance()

    for request in requests:
        try:
            # Convert user profile to dictionary
            user_profile = request.user_profile.dict()

            # Get recommendations
            recommendations = service.get_recommendations(
                user_profile=user_profile,
                top_k=request.top_k,
                fallback_top_k=request.fallback_top_k,
                threshold=request.threshold,
                enable_deduplication=request.enable_deduplication,
                max_workers=request.max_workers,
                max_fallback_candidates=request.max_fallback_candidates,
                min_similarity_threshold=request.min_similarity_threshold,
                recency_weight=request.recency_weight,
                watch_percentage_weight=request.watch_percentage_weight,
            )

            results.append(recommendations)

        except Exception as e:
            logger.error(f"Error processing batch request item: {e}", exc_info=True)
            # Return error response for this item
            results.append(
                {
                    "recommendations": [],
                    "scores": {},
                    "sources": {},
                    "fallback_recommendations": [],
                    "fallback_scores": {},
                    "fallback_sources": {},
                    "processing_time_ms": 0,
                    "error": str(e),
                }
            )

    # Return response directly as JSON
    return JSONResponse(content=results)


def start():
    """Start the API server."""
    # Get port from environment or use default
    port = int(os.environ.get("API_PORT", "8000"))

    # Start server
    uvicorn.run(
        "service.app:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )


if __name__ == "__main__":
    start()
