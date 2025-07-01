"""
Filtering components of the recommendation engine.

This submodule contains classes for filtering recommendations based on watch history
and deduplication logic.
"""

from recommendation.filter.history import HistoryManager

__all__ = ["HistoryManager"]
