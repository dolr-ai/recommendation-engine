#!/usr/bin/env python3
"""
Show detailed content of requests that returned 0 posts.
"""

import json
import os
from datetime import datetime

def show_zero_post_details(filename="cache_v2_test_results_20250917_142207.json"):
    """
    Show detailed content of requests with 0 posts.

    Args:
        filename: JSON file with test results
    """
    # Check if file exists
    if not os.path.exists(filename):
        print(f"Error: File {filename} not found in current directory")
        return

    # Load the JSON data
    with open(filename, 'r') as f:
        data = json.load(f)

    print(f"=== DETAILED ANALYSIS OF ZERO-POST REQUESTS ===")
    print(f"File: {filename}")
    print(f"User ID: {data['test_info']['user_id']}")
    print()

    # Find zero post requests
    zero_post_iterations = [241, 278, 307, 308, 318, 341]

    logs = data['logs']

    for i, log in enumerate(logs):
        iteration = log['request']['iteration']

        if iteration in zero_post_iterations:
            print(f"=" * 80)
            print(f"ITERATION {iteration} - ZERO POSTS REQUEST")
            print(f"=" * 80)

            # Show request details
            request = log['request']
            print(f"Timestamp: {request['timestamp']}")
            print(f"URL: {request['url']}")
            print(f"Payload:")
            print(json.dumps(request['payload'], indent=2))

            # Show response details
            response = log['response']
            print(f"\nResponse Status: {response['status_code']}")
            print(f"Response Time: {response['response_time_ms']:.2f}ms")
            print(f"Response Content:")
            print(json.dumps(response['content'], indent=2))

            # Show error if any
            if log.get('error'):
                print(f"Error: {log['error']}")

            print(f"\n")

if __name__ == "__main__":
    show_zero_post_details()