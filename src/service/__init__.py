"""
Service package for the recommendation engine.

This package contains the service layer components.
"""

from .app import app
from .recommendation_service import RecommendationService
from .metadata_service import MetadataService

__all__ = ["app", "RecommendationService", "MetadataService"]
