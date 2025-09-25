#!/usr/bin/env python3
"""
Analyze cache v2 test results to find requests that returned 0 posts.
"""

import pandas as pd
import json
import os
from datetime import datetime

def analyze_zero_posts(filename="cache_v2_test_results_20250917_142207.json"):
    """
    Analyze test results to find requests with 0 posts.

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

    print(f"=== Analysis of {filename} ===")
    print(f"Test info: User ID {data['test_info']['user_id']}")
    print(f"Total requests: {data['test_info']['total_requests']}")
    print(f"Base URL: {data['test_info']['base_url']}")
    print(f"NSFW enabled: {data['test_info']['nsfw_enabled']}")
    print(f"Num results requested: {data['test_info']['num_results']}")
    print()

    # Extract logs into DataFrame
    logs = data['logs']

    # Process each log entry
    zero_post_requests = []
    successful_requests = []
    error_requests = []

    for log in logs:
        iteration = log['request']['iteration']

        # Check if there's an error
        if log.get('error'):
            error_requests.append({
                'iteration': iteration,
                'error': log['error']
            })
            continue

        # Check response
        response = log.get('response')
        if not response:
            continue

        status_code = response.get('status_code')
        content = response.get('content')

        if status_code == 200 and isinstance(content, dict):
            posts = content.get('posts', [])
            num_posts = len(posts)

            request_info = {
                'iteration': iteration,
                'status_code': status_code,
                'num_posts': num_posts,
                'response_time_ms': response.get('response_time_ms', 0)
            }

            if num_posts == 0:
                zero_post_requests.append(request_info)
            else:
                successful_requests.append(request_info)

    # Convert to DataFrames
    if zero_post_requests:
        zero_posts_df = pd.DataFrame(zero_post_requests)
    else:
        zero_posts_df = pd.DataFrame()

    if successful_requests:
        successful_df = pd.DataFrame(successful_requests)
    else:
        successful_df = pd.DataFrame()

    # Print results
    print(f"=== SUMMARY ===")
    print(f"Total requests processed: {len(logs)}")
    print(f"Successful requests (200 OK): {len(successful_requests) + len(zero_post_requests)}")
    print(f"Error requests: {len(error_requests)}")
    print(f"Requests with 0 posts: {len(zero_post_requests)}")
    print(f"Requests with >0 posts: {len(successful_requests)}")
    print()

    # Show zero post requests
    if not zero_posts_df.empty:
        print(f"=== REQUESTS WITH 0 POSTS ({len(zero_post_requests)} total) ===")
        print(zero_posts_df.to_string(index=False))
        print()

        # Save zero post requests to CSV
        csv_filename = f"zero_posts_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        zero_posts_df.to_csv(csv_filename, index=False)
        print(f"Zero post requests saved to: {csv_filename}")
    else:
        print("=== NO REQUESTS WITH 0 POSTS FOUND ===")

    # Show error requests
    if error_requests:
        print(f"\n=== ERROR REQUESTS ({len(error_requests)} total) ===")
        error_df = pd.DataFrame(error_requests)
        print(error_df.to_string(index=False))

    # Show statistics for successful requests
    if not successful_df.empty:
        print(f"\n=== STATISTICS FOR SUCCESSFUL REQUESTS ===")
        print(f"Average posts per request: {successful_df['num_posts'].mean():.2f}")
        print(f"Min posts: {successful_df['num_posts'].min()}")
        print(f"Max posts: {successful_df['num_posts'].max()}")
        print(f"Average response time: {successful_df['response_time_ms'].mean():.2f}ms")

        # Show distribution of post counts
        print(f"\n=== POST COUNT DISTRIBUTION ===")
        post_distribution = successful_df['num_posts'].value_counts().sort_index()
        print(post_distribution)

    return zero_posts_df

if __name__ == "__main__":
    # You can specify a different filename here if needed
    zero_posts_df = analyze_zero_posts()