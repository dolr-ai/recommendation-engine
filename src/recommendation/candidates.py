"""
Candidates module for fetching and processing recommendation candidates.

This module provides functionality for fetching and processing candidates from various sources.
"""

import random
import concurrent.futures
from collections import OrderedDict
from utils.common_utils import get_logger
from candidate_cache.get_candidates import (
    ModifiedIoUCandidateFetcher,
    WatchTimeQuantileCandidateFetcher,
    FallbackCandidateFetcher,
)

logger = get_logger(__name__)


class CandidateService:
    """Service for fetching and processing recommendation candidates."""

    # Class-level dictionaries to cache fallback candidates by cluster_id and type
    _cached_fallback_miou_candidates = {}
    _cached_fallback_wt_candidates = {}

    # Probability of refreshing the cache (10%)
    _cache_refresh_probability = 0.10

    def __init__(self, valkey_config, cache_refresh_probability=None):
        """
        Initialize candidate service.

        Args:
            valkey_config: Valkey configuration dictionary
            cache_refresh_probability: Probability (0.0-1.0) of refreshing the cache on each request.
                                      If None, uses the class default.
        """
        self.valkey_config = valkey_config
        self.miou_fetcher = None
        self.wt_fetcher = None
        self.fallback_fetcher = None

        # Allow overriding the default cache refresh probability
        if cache_refresh_probability is not None:
            self._cache_refresh_probability = max(
                0.0, min(1.0, cache_refresh_probability)
            )
            logger.info(
                f"Using custom cache refresh probability: {self._cache_refresh_probability}"
            )

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

        logger.info(
            f"Initializing fetchers - MIOU: {need_miou}, WT: {need_wt}, Fallback MIOU: {need_fallback_miou}, Fallback WT: {need_fallback_wt}"
        )

        # Only initialize the fetchers we need
        if need_miou and not self.miou_fetcher:
            self.miou_fetcher = ModifiedIoUCandidateFetcher(config=self.valkey_config)
            logger.info("Initialized Modified IoU candidate fetcher")

        if need_wt and not self.wt_fetcher:
            self.wt_fetcher = WatchTimeQuantileCandidateFetcher(
                config=self.valkey_config
            )
            logger.info("Initialized Watch Time Quantile candidate fetcher")

        if (need_fallback_miou or need_fallback_wt) and not self.fallback_fetcher:
            self.fallback_fetcher = FallbackCandidateFetcher(config=self.valkey_config)
            logger.info("Initialized Fallback candidate fetcher")

    def _fetch_miou_candidates(self, cluster_id, query_videos):
        """Fetch Modified IoU candidates in parallel."""
        logger.info(
            f"Fetching Modified IoU candidates for cluster {cluster_id} with {len(query_videos)} query videos"
        )
        miou_args = [(str(cluster_id), video_id) for video_id in query_videos]
        try:
            candidates = self.miou_fetcher.get_candidates(miou_args)
            logger.info(
                f"Successfully fetched {len(candidates)} Modified IoU candidates"
            )
            return candidates
        except Exception as e:
            logger.error(f"Error fetching Modified IoU candidates: {e}", exc_info=True)
            return {}

    def _fetch_wt_candidates(self, cluster_id, bin_id, query_videos):
        """Fetch Watch Time Quantile candidates in parallel."""
        logger.info(
            f"Fetching Watch Time Quantile candidates for cluster {cluster_id}, bin {bin_id} with {len(query_videos)} query videos"
        )
        wt_args = [
            (str(cluster_id), str(bin_id), video_id) for video_id in query_videos
        ]
        try:
            candidates = self.wt_fetcher.get_candidates(wt_args)
            logger.info(
                f"Successfully fetched {len(candidates)} Watch Time Quantile candidates"
            )
            return candidates
        except Exception as e:
            logger.error(
                f"Error fetching Watch Time Quantile candidates: {e}", exc_info=True
            )
            return {}

    def _fetch_fallback_miou_candidates(self, cluster_id, max_fallback_candidates):
        """Fetch fallback Modified IoU candidates in parallel with caching."""
        cluster_id_str = str(cluster_id)

        # Check if we should refresh the cache
        should_refresh = random.random() < self._cache_refresh_probability

        # Use cache only if it exists and we don't need to refresh
        if (
            cluster_id_str in self._cached_fallback_miou_candidates
            and not should_refresh
        ):
            logger.info(
                f"Using cached fallback Modified IoU candidates for cluster {cluster_id} "
                f"({len(self._cached_fallback_miou_candidates[cluster_id_str])} candidates)"
            )
            return self._cached_fallback_miou_candidates[cluster_id_str]

        # Log appropriate message based on whether we're refreshing or fetching for the first time
        if should_refresh and cluster_id_str in self._cached_fallback_miou_candidates:
            logger.info(
                f"CACHE REFRESH: Bypassing cache for fallback Modified IoU candidates for cluster {cluster_id} (probability={self._cache_refresh_probability})"
            )
        else:
            logger.info(
                f"Fetching fallback Modified IoU candidates for cluster {cluster_id} (not cached)"
            )

        try:
            fallback_miou = self.fallback_fetcher.get_fallback_candidates(
                cluster_id_str, "modified_iou"
            )
            logger.info(
                f"Fetched {len(fallback_miou)} fallback Modified IoU candidates before sampling"
            )

            # Sample if we have more than max_fallback_candidates
            if len(fallback_miou) > max_fallback_candidates:
                fallback_miou = random.sample(fallback_miou, max_fallback_candidates)
                logger.info(
                    f"Sampled down to {len(fallback_miou)} fallback Modified IoU candidates"
                )

            # Cache the candidates for future use
            self._cached_fallback_miou_candidates[cluster_id_str] = fallback_miou
            if (
                should_refresh
                and cluster_id_str in self._cached_fallback_miou_candidates
            ):
                logger.info(
                    f"CACHE REFRESH: Updated cache for fallback Modified IoU candidates for cluster {cluster_id}"
                )
            else:
                logger.info(
                    f"Cached {len(fallback_miou)} fallback Modified IoU candidates for cluster {cluster_id}"
                )

            return fallback_miou
        except Exception as e:
            logger.error(
                f"Error fetching fallback Modified IoU candidates: {e}", exc_info=True
            )
            return []

    def _fetch_fallback_wt_candidates(self, cluster_id, max_fallback_candidates):
        """Fetch fallback Watch Time Quantile candidates in parallel with caching."""
        cluster_id_str = str(cluster_id)

        # Check if we should refresh the cache
        should_refresh = random.random() < self._cache_refresh_probability

        # Use cache only if it exists and we don't need to refresh
        if cluster_id_str in self._cached_fallback_wt_candidates and not should_refresh:
            logger.info(
                f"Using cached fallback Watch Time Quantile candidates for cluster {cluster_id} "
                f"({len(self._cached_fallback_wt_candidates[cluster_id_str])} candidates)"
            )
            return self._cached_fallback_wt_candidates[cluster_id_str]

        # Log appropriate message based on whether we're refreshing or fetching for the first time
        if should_refresh and cluster_id_str in self._cached_fallback_wt_candidates:
            logger.info(
                f"CACHE REFRESH: Bypassing cache for fallback Watch Time Quantile candidates for cluster {cluster_id} (probability={self._cache_refresh_probability})"
            )
        else:
            logger.info(
                f"Fetching fallback Watch Time Quantile candidates for cluster {cluster_id} (not cached)"
            )

        try:
            fallback_wt = self.fallback_fetcher.get_fallback_candidates(
                cluster_id_str, "watch_time_quantile"
            )
            logger.info(
                f"Fetched {len(fallback_wt)} fallback Watch Time Quantile candidates before sampling"
            )

            # Sample if we have more than max_fallback_candidates
            if len(fallback_wt) > max_fallback_candidates:
                fallback_wt = random.sample(fallback_wt, max_fallback_candidates)
                logger.info(
                    f"Sampled down to {len(fallback_wt)} fallback Watch Time Quantile candidates"
                )

            # Cache the candidates for future use
            self._cached_fallback_wt_candidates[cluster_id_str] = fallback_wt
            if should_refresh and cluster_id_str in self._cached_fallback_wt_candidates:
                logger.info(
                    f"CACHE REFRESH: Updated cache for fallback Watch Time Quantile candidates for cluster {cluster_id}"
                )
            else:
                logger.info(
                    f"Cached {len(fallback_wt)} fallback Watch Time Quantile candidates for cluster {cluster_id}"
                )

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
        logger.info(
            f"Starting candidate fetching for {len(query_videos)} query videos with cluster_id={cluster_id}, bin_id={bin_id}"
        )
        logger.info(
            f"Using max_workers={max_workers}, max_fallback_candidates={max_fallback_candidates}"
        )

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
        logger.info(f"Starting parallel fetching with {max_workers} workers")
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit tasks for parallel execution
            if need_miou:
                logger.info("Submitting Modified IoU candidate fetching task")
                tasks["miou"] = executor.submit(
                    self._fetch_miou_candidates, cluster_id, query_videos
                )

            if need_wt:
                logger.info("Submitting Watch Time Quantile candidate fetching task")
                tasks["wt"] = executor.submit(
                    self._fetch_wt_candidates, cluster_id, bin_id, query_videos
                )

            if need_fallback_miou:
                logger.info("Submitting fallback Modified IoU candidate fetching task")
                tasks["fallback_miou"] = executor.submit(
                    self._fetch_fallback_miou_candidates,
                    cluster_id,
                    max_fallback_candidates,
                )

            if need_fallback_wt:
                logger.info(
                    "Submitting fallback Watch Time Quantile candidate fetching task"
                )
                tasks["fallback_wt"] = executor.submit(
                    self._fetch_fallback_wt_candidates,
                    cluster_id,
                    max_fallback_candidates,
                )

            # Collect results as they complete
            logger.info(f"Waiting for {len(tasks)} parallel fetching tasks to complete")
            for name, future in tasks.items():
                try:
                    logger.info(f"Processing results from {name} task")
                    if name == "miou":
                        miou_candidates = future.result()
                        logger.info(
                            f"Received {len(miou_candidates)} Modified IoU candidates"
                        )
                    elif name == "wt":
                        wt_candidates = future.result()
                        logger.info(
                            f"Received {len(wt_candidates)} Watch Time Quantile candidates"
                        )
                    elif name == "fallback_miou":
                        fallback_candidates["fallback_modified_iou"] = future.result()
                        logger.info(
                            f"Received {len(fallback_candidates.get('fallback_modified_iou', []))} fallback Modified IoU candidates"
                        )
                    elif name == "fallback_wt":
                        fallback_candidates["fallback_watch_time_quantile"] = (
                            future.result()
                        )
                        logger.info(
                            f"Received {len(fallback_candidates.get('fallback_watch_time_quantile', []))} fallback Watch Time Quantile candidates"
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

        logger.info(
            f"All parallel fetching tasks completed for {len(query_videos)} query videos"
        )
        if fallback_candidates:
            logger.info(
                f"Fetched fallback candidates: {', '.join([f'{k}: {len(v)}' for k, v in fallback_candidates.items()])}"
            )

        # Organize candidates by query video and type in an ordered dictionary
        logger.info("Organizing candidates by query video and type")
        all_candidates = OrderedDict()

        # Pre-initialize the structure for all query videos to avoid repeated checks
        for video_id in query_videos:
            all_candidates[video_id] = {}

            # Add candidates for each type if available
            if need_miou:
                key = f"{cluster_id}:{video_id}:modified_iou_candidate"
                candidates = miou_candidates.get(key, [])
                all_candidates[video_id]["modified_iou"] = candidates
                logger.debug(
                    f"Added {len(candidates)} Modified IoU candidates for video {video_id}"
                )

            if need_wt:
                key = f"{cluster_id}:{bin_id}:{video_id}:watch_time_quantile_bin_candidate"
                candidates = wt_candidates.get(key, [])
                all_candidates[video_id]["watch_time_quantile"] = candidates
                logger.debug(
                    f"Added {len(candidates)} Watch Time Quantile candidates for video {video_id}"
                )

        # Add fallback candidates to the first query video only
        if query_videos and fallback_candidates:
            first_video_id = query_videos[0]
            logger.info(
                f"Adding fallback candidates to first query video: {first_video_id}"
            )
            for fallback_type, fallback_list in fallback_candidates.items():
                all_candidates[first_video_id][fallback_type] = fallback_list
                logger.debug(
                    f"Added {len(fallback_list)} {fallback_type} candidates to first query video"
                )

        logger.info(
            f"Candidate fetching completed with {len(all_candidates)} query videos processed"
        )
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
        logger.info(
            f"Filtering and sorting {len(watch_history)} watch history items with threshold {threshold}"
        )

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

        logger.info(
            f"Filtered watch history: kept {len(filtered_history)}, skipped {skipped_count} items"
        )

        # Sort by last_watched_timestamp (newest first)
        filtered_history.sort(
            key=lambda x: x.get("last_watched_timestamp", 0), reverse=True
        )
        logger.info("Sorted watch history by timestamp (newest first)")

        # Extract video IDs in order (from latest to oldest watched)
        query_videos = [item.get("video_id") for item in filtered_history]

        # Create dictionary mapping video IDs to watch percentages
        watch_percentages = {
            item.get("video_id"): float(item.get("mean_percentage_watched", 0))
            for item in filtered_history
        }

        logger.info(
            f"Returning {len(query_videos)} query videos after filtering and sorting"
        )
        return query_videos, watch_percentages
