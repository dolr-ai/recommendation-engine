"""
Data management components of the recommendation engine.

This submodule contains classes for managing user metadata and backend data transformations.
"""

from recommendation.data.metadata import MetadataManager
from recommendation.data.backend import transform_recommendations_with_metadata

__all__ = ["MetadataManager", "transform_recommendations_with_metadata"]
