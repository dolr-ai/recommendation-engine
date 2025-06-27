"""
Candidates module for fetching and processing recommendation candidates.

This module provides functionality for fetching and processing candidates from various sources.
"""

import random
import concurrent.futures
from collections import OrderedDict
from datetime import datetime
from utils.common_utils import get_logger
from candidate_cache.get_candidates import (
    ModifiedIoUCandidateFetcher,
    WatchTimeQuantileCandidateFetcher,
    FallbackCandidateFetcher,
)

logger = get_logger(__name__)


class CandidateService:
    """Service for fetching and processing recommendation candidates."""

    def __init__(self, valkey_config):
        """
        Initialize candidate service.

        Args:
            valkey_config: Valkey configuration dictionary
        """
        self.valkey_config = valkey_config
        self.miou_fetcher = None
        self.wt_fetcher = None
        self.fallback_fetcher = None

        logger.info("CandidateService initialized")

    def _initialize_fetchers(self, candidate_types_dict):
        """
        Initialize candidate fetchers based on candidate types.

        Args:
            candidate_types_dict: Dictionary mapping candidate type numbers to their names and weights
        """
        # Create a reverse mapping from name to type number for lookup
        candidate_type_name_to_num = {
            info["name"]: type_num for type_num, info in candidate_types_dict.items()
        }

        # Check which fetchers we need
        need_miou = "modified_iou" in candidate_type_name_to_num
        need_wt = "watch_time_quantile" in candidate_type_name_to_num
        need_fallback_miou = "fallback_modified_iou" in candidate_type_name_to_num
        need_fallback_wt = "fallback_watch_time_quantile" in candidate_type_name_to_num

        # Only initialize the fetchers we need
        if need_miou and not self.miou_fetcher:
            self.miou_fetcher = ModifiedIoUCandidateFetcher(config=self.valkey_config)

        if need_wt and not self.wt_fetcher:
            self.wt_fetcher = WatchTimeQuantileCandidateFetcher(
                config=self.valkey_config
            )

        if (need_fallback_miou or need_fallback_wt) and not self.fallback_fetcher:
            self.fallback_fetcher = FallbackCandidateFetcher(config=self.valkey_config)

    def _fetch_miou_candidates(self, cluster_id, query_videos):
        """Fetch Modified IoU candidates in parallel."""
        miou_args = [(str(cluster_id), video_id) for video_id in query_videos]
        logger.info(
            f"Fetching Modified IoU candidates for {len(miou_args)} videos\n miou_args: {miou_args}"
        )
        try:
            candidates = self.miou_fetcher.fetch_using_mget(miou_args)
            return candidates
        except Exception as e:
            logger.error(f"Error fetching Modified IoU candidates: {e}", exc_info=True)
            return {}

    def _fetch_wt_candidates(self, cluster_id, bin_id, query_videos):
        """Fetch Watch Time Quantile candidates in parallel."""
        wt_args = [
            (str(cluster_id), str(bin_id), video_id) for video_id in query_videos
        ]
        try:
            candidates = self.wt_fetcher.fetch_using_mget(wt_args)
            return candidates
        except Exception as e:
            logger.error(
                f"Error fetching Watch Time Quantile candidates: {e}", exc_info=True
            )
            return {}

    def _fetch_fallback_miou_candidates(self, cluster_id, max_fallback_candidates):
        """Fetch fallback Modified IoU candidates."""
        cluster_id_str = str(cluster_id)

        try:
            fallback_miou = self.fallback_fetcher.get_fallback_candidates(
                cluster_id_str, "modified_iou"
            )

            # Sample if we have more than max_fallback_candidates
            if len(fallback_miou) > max_fallback_candidates:
                fallback_miou = random.sample(fallback_miou, max_fallback_candidates)

            return fallback_miou
        except Exception as e:
            logger.error(
                f"Error fetching fallback Modified IoU candidates: {e}", exc_info=True
            )
            return []

    def _fetch_fallback_wt_candidates(self, cluster_id, max_fallback_candidates):
        """Fetch fallback Watch Time Quantile candidates."""
        cluster_id_str = str(cluster_id)

        try:
            fallback_wt = self.fallback_fetcher.get_fallback_candidates(
                cluster_id_str, "watch_time_quantile"
            )

            # Sample if we have more than max_fallback_candidates
            if len(fallback_wt) > max_fallback_candidates:
                fallback_wt = random.sample(fallback_wt, max_fallback_candidates)

            return fallback_wt
        except Exception as e:
            logger.error(
                f"Error fetching fallback Watch Time Quantile candidates: {e}",
                exc_info=True,
            )
            return []

    def fetch_candidates(
        self,
        query_videos,
        cluster_id,
        bin_id,
        candidate_types_dict,
        max_fallback_candidates=1000,
        max_workers=4,
    ):
        """
        Fetch candidates for all query videos in parallel.

        Args:
            query_videos: List of query video IDs
            cluster_id: User's cluster ID
            bin_id: User's watch time quantile bin ID
            candidate_types_dict: Dictionary mapping candidate type numbers to their names and weights
            max_fallback_candidates: Maximum number of fallback candidates to sample (if more are available)
            max_workers: Maximum number of worker threads for parallel fetching

        Returns:
            OrderedDict of candidates organized by query video and type
        """
        # Initialize fetchers if needed
        self._initialize_fetchers(candidate_types_dict)

        # Create a reverse mapping from name to type number for lookup
        candidate_type_name_to_num = {
            info["name"]: type_num for type_num, info in candidate_types_dict.items()
        }

        # Check which fetchers we need
        need_miou = "modified_iou" in candidate_type_name_to_num
        need_wt = "watch_time_quantile" in candidate_type_name_to_num
        need_fallback_miou = "fallback_modified_iou" in candidate_type_name_to_num
        need_fallback_wt = "fallback_watch_time_quantile" in candidate_type_name_to_num

        # Prepare tasks for parallel execution
        tasks = {}
        miou_candidates = {}
        wt_candidates = {}
        fallback_candidates = {}

        # Use ThreadPoolExecutor for parallel fetching
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks for parallel execution
            if need_miou:
                tasks["miou"] = executor.submit(
                    self._fetch_miou_candidates, cluster_id, query_videos
                )

            if need_wt:
                tasks["wt"] = executor.submit(
                    self._fetch_wt_candidates, cluster_id, bin_id, query_videos
                )

            if need_fallback_miou:
                tasks["fallback_miou"] = executor.submit(
                    self._fetch_fallback_miou_candidates,
                    cluster_id,
                    max_fallback_candidates,
                )

            if need_fallback_wt:
                tasks["fallback_wt"] = executor.submit(
                    self._fetch_fallback_wt_candidates,
                    cluster_id,
                    max_fallback_candidates,
                )

            # Collect results as they complete
            for name, future in tasks.items():
                try:
                    if name == "miou":
                        miou_candidates = future.result()
                    elif name == "wt":
                        wt_candidates = future.result()
                    elif name == "fallback_miou":
                        fallback_candidates["fallback_modified_iou"] = future.result()
                    elif name == "fallback_wt":
                        fallback_candidates["fallback_watch_time_quantile"] = (
                            future.result()
                        )
                except Exception as e:
                    logger.error(
                        f"Error fetching {name} candidates: {e}", exc_info=True
                    )
                    # Initialize with empty results on error
                    if name == "miou":
                        miou_candidates = {}
                    elif name == "wt":
                        wt_candidates = {}
                    elif name == "fallback_miou":
                        fallback_candidates["fallback_modified_iou"] = []
                    elif name == "fallback_wt":
                        fallback_candidates["fallback_watch_time_quantile"] = []

        # Organize candidates by query video and type in an ordered dictionary
        all_candidates = OrderedDict()

        # Pre-initialize the structure for all query videos to avoid repeated checks
        for video_id in query_videos:
            all_candidates[video_id] = {}

            # Add candidates for each type if available
            if need_miou:
                key = f"{cluster_id}:{video_id}:modified_iou_candidate"
                candidates = miou_candidates.get(key, [])
                all_candidates[video_id]["modified_iou"] = candidates

            if need_wt:
                key = f"{cluster_id}:{bin_id}:{video_id}:watch_time_quantile_bin_candidate"
                candidates = wt_candidates.get(key, [])
                all_candidates[video_id]["watch_time_quantile"] = candidates

        # Add fallback candidates to the first query video only
        if query_videos and fallback_candidates:
            first_video_id = query_videos[0]
            for fallback_type, fallback_list in fallback_candidates.items():
                all_candidates[first_video_id][fallback_type] = fallback_list

        return all_candidates

    def filter_and_sort_watch_history(self, watch_history, threshold):
        """
        Filter watch history by threshold and sort by timestamp.

        Args:
            watch_history: List of watch history items
            threshold: Minimum mean_percentage_watched to consider a video

        Returns:
            Tuple of (query_videos, watch_percentages) where:
            - query_videos: List of video IDs ordered from latest to oldest watched
            - watch_percentages: Dictionary mapping video IDs to their mean_percentage_watched values
        """
        filtered_history = []
        skipped_count = 0
        for item in watch_history:
            try:
                watch_percentage = float(item.get("mean_percentage_watched", 0))
                if watch_percentage >= threshold:
                    filtered_history.append(item)
                else:
                    skipped_count += 1
            except (ValueError, TypeError):
                skipped_count += 1
                continue

        # Sort by last_watched_timestamp (newest first)
        # todo: check if this is being passed by client in the right format
        filtered_history.sort(
            key=lambda x: datetime.fromisoformat(
                x.get("last_watched_timestamp", "1970-01-01T00:00:00+00:00").replace(
                    " ", "T"
                )
            ),
            reverse=True,
        )

        # Extract video IDs in order (from latest to oldest watched)
        query_videos = [item.get("video_id") for item in filtered_history]

        # Create dictionary mapping video IDs to watch percentages
        watch_percentages = {
            item.get("video_id"): float(item.get("mean_percentage_watched", 0))
            for item in filtered_history
        }

        return query_videos, watch_percentages
