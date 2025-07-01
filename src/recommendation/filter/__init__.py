"""
Filtering components of the recommendation engine.

This submodule contains classes for filtering recommendations based on watch history
and deduplication logic.
"""

from recommendation.filter.history import HistoryManager
from recommendation.filter.deduplication import DeduplicationManager

__all__ = ["HistoryManager", "DeduplicationManager"]
