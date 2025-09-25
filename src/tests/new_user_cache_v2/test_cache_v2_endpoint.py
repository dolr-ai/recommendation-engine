#!/usr/bin/env python3
"""
Test script for new user cache v2 endpoint.
Tests /v2/recommendations/cache endpoint with random user ID, excludes previously seen videos,
and logs all requests/responses until an error occurs (max 1000 iterations).

USAGE EXAMPLES:
    # Basic test with localhost
    python test_cache_v2_endpoint.py --base-url http://localhost:8000 --num-results 10

    # Test with production-like settings (30 results, NSFW enabled)
    python test_cache_v2_endpoint.py --base-url http://localhost:8000 --num-results 30 --nsfw --max-iterations 1000

    # Test against production endpoint
    python test_cache_v2_endpoint.py --base-url BASE_URL --num-results 30

EQUIVALENT CURL COMMAND:
    curl --location 'http://localhost:8000/v2/recommendations/cache' \
    --header 'Content-Type: application/json' \
    --data '{
        "user_id": "random-uuid-here",
        "nsfw_label": false,
        "exclude_watched_items": [],
        "exclude_reported_items": [],
        "exclude_items": ["video_id_1", "video_id_2"],
        "num_results": 10,
        "region": null,
        "ip_address": null
    }'

PRODUCTION CURL EXAMPLE:
    curl --location 'BASE_URL/v2/recommendations/cache' \
    --header 'User-Agent: Mozilla/5.0 (X11; Linux x86_64; rv:142.0) Gecko/20100101 Firefox/142.0' \
    --header 'Accept: */*' \
    --header 'Accept-Language: en-US,en;q=0.5' \
    --header 'Accept-Encoding: gzip, deflate, br, zstd' \
    --header 'Referer: https://yral.com/' \
    --header 'content-type: application/json' \
    --header 'Origin: https://yral.com' \
    --header 'Connection: keep-alive' \
    --header 'Sec-Fetch-Dest: empty' \
    --header 'Sec-Fetch-Mode: cors' \
    --header 'Sec-Fetch-Site: cross-site' \
    --header 'Priority: u=4' \
    --header 'Pragma: no-cache' \
    --header 'Cache-Control: no-cache' \
    --header 'TE: trailers' \
    --data '{"user_id":"v6uzq-up7cy-os5rl-oxyp6-vodok-prm66-lbrwu-r6qes-otn5a-kwfeb-hae","exclude_items":[],"nsfw_label":false,"num_results":30,"ip_address":null}'
"""

import requests
import json
import time
import uuid
import os
from datetime import datetime
from typing import List, Dict, Any, Optional
import argparse

# Import the existing ValkeyService and GCPUtils
from utils.valkey_utils import ValkeyService
from utils.gcp_utils import GCPUtils


class CacheV2Tester:
    def __init__(self, base_url: str = "http://localhost:8000", nsfw_enabled: bool = False, num_results: int = 10):
        self.base_url = base_url
        self.user_id = f"temp_test_user-{str(uuid.uuid4())}"  # Random test user ID
        self.nsfw_enabled = nsfw_enabled
        self.num_results = num_results
        self.request_response_log: List[Dict[str, Any]] = []

        # Metrics tracking
        self.total_already_watched_count = 0
        self.total_recommended_count = 0

        # Setup GCP utils and Valkey service (like production does)
        self.gcp_utils = self._setup_gcp_utils()
        self.valkey_service = self._setup_valkey_service()

        # Setup Redis set key for watched videos (mimics production)
        self.watched_set_key = self._get_watched_set_key()

        # Initialize the set with 15-day expiry
        self._init_watched_set()

    def _setup_gcp_utils(self) -> GCPUtils:
        """Setup GCP utils from environment variables (like production does)."""
        gcp_credentials = os.environ.get("RECSYS_GCP_CREDENTIALS")
        if not gcp_credentials:
            print("Warning: No RECSYS_GCP_CREDENTIALS found, using placeholder")
            # For testing purposes, we don't need full GCP credentials
            # Just need the core for ValkeyService initialization
            return None

        return GCPUtils(gcp_credentials=gcp_credentials)

    def _setup_valkey_service(self) -> ValkeyService:
        """Setup Valkey service with production-like configuration."""
        # Use environment variables exactly like the production code does
        config = {
            "host": os.environ.get("RECSYS_PROXY_REDIS_HOST") or os.environ.get("RECSYS_SERVICE_REDIS_HOST", "localhost"),
            "port": int(os.environ.get("RECSYS_PROXY_REDIS_PORT", os.environ.get("RECSYS_SERVICE_REDIS_PORT", 6379))),
            "instance_id": os.environ.get("RECSYS_SERVICE_REDIS_INSTANCE_ID"),
            "authkey": os.environ.get("RECSYS_SERVICE_REDIS_AUTHKEY"),
            "ssl_enabled": False,
            "socket_timeout": 15,
            "socket_connect_timeout": 15,
            "cluster_enabled": os.environ.get("RECSYS_SERVICE_REDIS_CLUSTER_ENABLED", "false").lower() in ("true", "1", "yes"),
        }

        print(f"Connecting to Valkey at {config['host']}:{config['port']} with authkey: {config['authkey'][:8]}..." if config['authkey'] else "no authkey")

        # Create a minimal GCPCore for ValkeyService (required by the constructor but not used when authkey is present)
        minimal_core = type('MinimalCore', (), {
            'credentials': None,
            'project_id': os.environ.get("RECSYS_PROJECT_ID", "test-project")
        })()

        return ValkeyService(core=minimal_core, **config)

    def _get_watched_set_key(self) -> str:
        """Get the Redis set key for watched videos (mimics production format)."""
        suffix = "_set_watch_nsfw_v2" if self.nsfw_enabled else "_set_watch_clean_v2"
        return f"{self.user_id}{suffix}"

    def _init_watched_set(self):
        """Initialize the watched videos set with 15-day expiry."""
        try:
            # Test connection first
            if not self.valkey_service.verify_connection():
                print("Warning: Could not verify Valkey connection")
                return

            # Initialize empty set (will be created when we add first video)
            # Set 15-day expiry (15 * 24 * 60 * 60 = 1,296,000 seconds)
            expiry_seconds = 15 * 24 * 60 * 60

            # Add a dummy member and remove it to create the set with expiry
            self.valkey_service.sadd(self.watched_set_key, "__init__")
            self.valkey_service.srem(self.watched_set_key, "__init__")
            self.valkey_service.expire(self.watched_set_key, expiry_seconds)

            print(f"Initialized watched set: {self.watched_set_key} with 15-day expiry")
        except Exception as e:
            print(f"Warning: Could not initialize Valkey set: {e}")

    def _add_watched_videos(self, video_ids: List[str]):
        """Add video IDs to the watched set in Valkey."""
        if not video_ids:
            return

        try:
            self.valkey_service.sadd(self.watched_set_key, *video_ids)
            # Refresh expiry
            self.valkey_service.expire(self.watched_set_key, 15 * 24 * 60 * 60)
            print(f"Added {len(video_ids)} videos to watched set: {video_ids}")
        except Exception as e:
            print(f"Warning: Could not add videos to Valkey set: {e}")

    def get_watched_count(self) -> int:
        """Get count of watched videos in Valkey set."""
        try:
            return self.valkey_service.scard(self.watched_set_key)
        except Exception as e:
            print(f"Warning: Could not get watched count: {e}")
            return 0

    def _check_already_watched(self, video_ids: List[str]) -> int:
        """Check how many of the provided video IDs are already in the watched set using efficient batch operation."""
        if not video_ids:
            return 0

        try:
            # Use pipeline for batch operation - much faster than individual calls
            pipe = self.valkey_service.pipeline()
            for video_id in video_ids:
                pipe.sismember(self.watched_set_key, video_id)
            results = pipe.execute()

            # Count how many returned True (already watched)
            already_watched = sum(1 for result in results if result)
            return already_watched
        except Exception as e:
            print(f"Warning: Could not check already watched videos: {e}")
            return 0

    def make_request(self, iteration: int) -> Dict[str, Any]:
        """Make a request to the /v2/recommendations/cache endpoint."""
        url = f"{self.base_url}/v2/recommendations/cache"

        payload = {
            "user_id": self.user_id,
            "nsfw_label": self.nsfw_enabled,
            "exclude_watched_items": [],
            "exclude_reported_items": [],
            "exclude_items": [],  # Not using exclude_items anymore - using Valkey set instead
            "num_results": self.num_results,
            "region": None,
            "ip_address": None,
        }

        headers = {"Content-Type": "application/json"}

        request_data = {
            "iteration": iteration,
            "timestamp": datetime.now().isoformat(),
            "user_id": self.user_id,
            "url": url,
            "payload": payload,
            "headers": headers,
        }

        try:
            print(f"\n=== Iteration {iteration} ===")
            print(f"User ID: {self.user_id}")
            print(f"Watched videos count in Valkey: {self.get_watched_count()}")
            print(f"Request payload: {json.dumps(payload, indent=2)}")

            response = requests.post(url, json=payload, headers=headers, timeout=30)

            response_data = {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "content": response.json()
                if response.headers.get("content-type", "").startswith(
                    "application/json"
                )
                else response.text,
                "response_time_ms": response.elapsed.total_seconds() * 1000,
            }

            print(f"Response status: {response.status_code}")
            print(f"Response time: {response_data['response_time_ms']:.2f}ms")
            print(f"Response content: {json.dumps(response_data['content'], indent=2)}")

            log_entry = {
                "request": request_data,
                "response": response_data,
                "error": None,
            }

            # If successful, extract video_ids and track already watched metrics
            already_watched_this_request = 0
            total_videos_this_request = 0

            if response.status_code == 200 and isinstance(
                response_data["content"], dict
            ):
                posts = response_data["content"].get("posts", [])
                new_video_ids = [
                    post.get("video_id") for post in posts if post.get("video_id")
                ]

                total_videos_this_request = len(new_video_ids)
                self.total_recommended_count += total_videos_this_request

                if new_video_ids:
                    # Check how many videos were already watched before adding new ones
                    already_watched_this_request = self._check_already_watched(new_video_ids)
                    self.total_already_watched_count += already_watched_this_request

                    self._add_watched_videos(new_video_ids)
                    print(
                        f"Added {len(new_video_ids)} video_ids to Valkey watched set: {new_video_ids}"
                    )
                    print(
                        f"Already watched in this request: {already_watched_this_request}/{total_videos_this_request}"
                    )
                    print(
                        f"Total already watched: {self.total_already_watched_count}/{self.total_recommended_count}"
                    )
                else:
                    print("No video_ids found in response")

            # Add metrics to log entry
            log_entry["metrics"] = {
                "already_watched_this_request": already_watched_this_request,
                "total_videos_this_request": total_videos_this_request,
                "total_already_watched_count": self.total_already_watched_count,
                "total_recommended_count": self.total_recommended_count,
                "watched_videos_in_redis": self.get_watched_count()
            }

            return log_entry

        except Exception as e:
            error_msg = f"Request failed: {str(e)}"
            print(f"ERROR: {error_msg}")

            log_entry = {"request": request_data, "response": None, "error": error_msg}

            return log_entry

    def run_test(self, max_iterations: int = 1000):
        """Run the test loop until error or max iterations."""
        print(f"Starting cache v2 endpoint test with user_id: {self.user_id}")
        print(f"Base URL: {self.base_url}")
        print(f"NSFW enabled: {self.nsfw_enabled}")
        print(f"Num results per request: {self.num_results}")
        print(f"Valkey watched set key: {self.watched_set_key}")
        print(f"Max iterations: {max_iterations}")

        for iteration in range(1, max_iterations + 1):
            try:
                log_entry = self.make_request(iteration)
                self.request_response_log.append(log_entry)

                # Check if there was an error
                if log_entry["error"]:
                    print(
                        f"\nTest stopped at iteration {iteration} due to error: {log_entry['error']}"
                    )
                    break

                # Check if response indicates an error
                response = log_entry.get("response")
                if response and response.get("status_code") != 200:
                    print(
                        f"\nTest stopped at iteration {iteration} due to HTTP error: {response.get('status_code')}"
                    )
                    break

                # Check if response content has error field
                content = response.get("content") if response else None
                if isinstance(content, dict) and content.get("error"):
                    print(
                        f"\nTest stopped at iteration {iteration} due to API error: {content.get('error')}"
                    )
                    break

                # Small delay between requests
                time.sleep(0.1)

            except KeyboardInterrupt:
                print(f"\nTest interrupted by user at iteration {iteration}")
                break
            except Exception as e:
                error_msg = f"Unexpected error at iteration {iteration}: {str(e)}"
                print(f"\nERROR: {error_msg}")

                log_entry = {
                    "request": {
                        "iteration": iteration,
                        "timestamp": datetime.now().isoformat(),
                    },
                    "response": None,
                    "error": error_msg,
                }
                self.request_response_log.append(log_entry)
                break

        self.save_results()
        self.print_summary()

    def save_results(self):
        """Save all request/response logs to a JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"cache_v2_test_results_{timestamp}.json"

        results = {
            "test_info": {
                "user_id": self.user_id,
                "base_url": self.base_url,
                "nsfw_enabled": self.nsfw_enabled,
                "num_results": self.num_results,
                "valkey_watched_set_key": self.watched_set_key,
                "total_requests": len(self.request_response_log),
                "test_timestamp": datetime.now().isoformat(),
                "final_watched_videos_count": self.get_watched_count(),
                "total_already_watched_count": self.total_already_watched_count,
                "total_recommended_count": self.total_recommended_count,
                "already_watched_percentage": round((self.total_already_watched_count / self.total_recommended_count * 100), 2) if self.total_recommended_count > 0 else 0,
            },
            "logs": self.request_response_log,
        }

        with open(filename, "w") as f:
            json.dump(results, f, indent=2)

        print(f"\nResults saved to: {filename}")

    def print_summary(self):
        """Print test summary."""
        total_requests = len(self.request_response_log)
        successful_requests = sum(
            1
            for log in self.request_response_log
            if log.get("response") and log["response"].get("status_code") == 200
        )
        error_requests = sum(1 for log in self.request_response_log if log.get("error"))

        already_watched_percentage = round((self.total_already_watched_count / self.total_recommended_count * 100), 2) if self.total_recommended_count > 0 else 0

        print("\n=== TEST SUMMARY ===")
        print(f"User ID: {self.user_id}")
        print(f"Base URL: {self.base_url}")
        print(f"NSFW enabled: {self.nsfw_enabled}")
        print(f"Num results per request: {self.num_results}")
        print(f"Total requests: {total_requests}")
        print(f"Successful requests: {successful_requests}")
        print(f"Error requests: {error_requests}")
        print(f"Final watched videos count in Valkey: {self.get_watched_count()}")
        print(f"Total recommended videos: {self.total_recommended_count}")
        print(f"Total already watched videos: {self.total_already_watched_count}")
        print(f"Already watched percentage: {already_watched_percentage}%")
        print(f"Valkey set key: {self.watched_set_key}")

        if self.request_response_log and self.request_response_log[-1].get("error"):
            print(f"Last error: {self.request_response_log[-1]['error']}")


def main():
    parser = argparse.ArgumentParser(description="Test cache v2 endpoint")
    parser.add_argument(
        "--base-url", default="http://localhost:8000", help="Base URL for the API"
    )
    parser.add_argument(
        "--max-iterations", type=int, default=1000, help="Maximum iterations to run"
    )
    parser.add_argument(
        "--nsfw", action="store_true", help="Enable NSFW content (default: False)"
    )
    parser.add_argument(
        "--num-results", type=int, default=10, help="Number of results per request (default: 10)"
    )

    args = parser.parse_args()

    tester = CacheV2Tester(base_url=args.base_url, nsfw_enabled=args.nsfw, num_results=args.num_results)
    tester.run_test(max_iterations=args.max_iterations)


if __name__ == "__main__":
    main()
