# %%
import pandas as pd
import requests
import json
import time
import concurrent.futures
import statistics
from tqdm import tqdm


def make_simple_request(request_params, api_url, timeout=30):
    """Simple request with basic timing"""
    headers = {"accept": "application/json", "Content-Type": "application/json"}

    start_time = time.time()
    try:
        response = requests.post(
            api_url, headers=headers, json=request_params, timeout=timeout
        )
        end_time = time.time()

        latency = end_time - start_time
        success = response.status_code == 200

        # Extract key timings from response
        debug_info = {}
        if success:
            try:
                resp_data = response.json()
                debug = resp_data.get("debug", {})
                # Collect all timing keys present in the debug info
                debug_info = {
                    k: debug.get(k, 0)
                    for k in [
                        "backend_time",
                        "bq_similarity_time",
                        "rerank_time",
                        "candidate_fetching_time",
                        "mixer_time",
                        "filter_time",
                        "reported_time",
                        "total_time",
                    ]
                    if k in debug
                }
            except Exception:
                pass

        return {
            "success": success,
            "latency": latency,
            "status_code": response.status_code,
            "debug": debug_info,
        }
    except Exception as e:
        end_time = time.time()
        return {
            "success": False,
            "latency": end_time - start_time,
            "error": str(e)[:50],
            "status_code": 0,
            "debug": {},
        }


def prepare_request(row):
    """Extract request from dataframe row"""
    if isinstance(row["request_params"], str):
        params = json.loads(row["request_params"])
    else:
        params = row["request_params"]

    if "num_results" not in params:
        params["num_results"] = 2

    return params


def run_concurrent_test(df, api_url, concurrent_requests, timeout=30):
    """Run simple concurrent test"""
    print(f"\n🧪 Testing {concurrent_requests} concurrent requests...")

    # Get random requests
    sample_requests = []
    for _ in range(concurrent_requests):
        row = df.sample(1).iloc[0]
        sample_requests.append(prepare_request(row))

    # Run all requests concurrently
    start_time = time.time()
    with concurrent.futures.ThreadPoolExecutor(
        max_workers=concurrent_requests
    ) as executor:
        futures = [
            executor.submit(make_simple_request, req, api_url, timeout)
            for req in sample_requests
        ]

        results = []
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures)
        ):
            results.append(future.result())

    end_time = time.time()
    test_duration = end_time - start_time

    # Analyze results
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    if not successful:
        print("❌ ALL REQUESTS FAILED!")
        return None

    # Calculate detailed percentiles
    latencies = [r["latency"] for r in successful]
    success_rate = len(successful) / len(results) * 100
    throughput = len(successful) / test_duration

    def calc_percentiles(data):
        if not data:
            return {}
        sorted_data = sorted(data)
        n = len(sorted_data)
        return {
            "min": min(data),
            "p50": sorted_data[int(n * 0.50)],
            "p90": sorted_data[int(n * 0.90)],
            "p95": sorted_data[int(n * 0.95)],
            "p99": sorted_data[int(n * 0.99)] if n >= 100 else sorted_data[-1],
            "max": max(data),
            "mean": statistics.mean(data),
        }

    latency_stats = calc_percentiles(latencies)

    # Component analysis with percentiles
    # Collect all possible timing keys for reporting
    all_timing_keys = [
        "backend_time",
        "bq_similarity_time",
        "rerank_time",
        "candidate_fetching_time",
        "mixer_time",
        "filter_time",
        "reported_time",
        "total_time",
    ]
    components = {k: [] for k in all_timing_keys}
    for result in successful:
        for comp in all_timing_keys:
            time_val = result["debug"].get(comp, 0)
            if time_val > 0:
                components[comp].append(time_val)
    # Remove empty lists
    components = {k: v for k, v in components.items() if v}

    component_stats = {}
    for comp, times in components.items():
        component_stats[comp] = calc_percentiles(times)

    # Print results
    print(f"✅ Success: {len(successful)}/{len(results)} ({success_rate:.1f}%)")
    print(f"🚀 Throughput: {throughput:.1f} requests/second")
    print(f"⏱️  Latency percentiles:")
    print(f"   Min: {latency_stats['min']:.3f}s")
    print(f"   P50: {latency_stats['p50']:.3f}s")
    print(f"   P90: {latency_stats['p90']:.3f}s")
    print(f"   P95: {latency_stats['p95']:.3f}s")
    print(f"   P99: {latency_stats['p99']:.3f}s")
    print(f"   Max: {latency_stats['max']:.3f}s")
    print(f"   Avg: {latency_stats['mean']:.3f}s")

    if component_stats:
        print(f"🔍 Component breakdown (with percentiles):")
        for comp in all_timing_keys:
            if comp in component_stats:
                stats = component_stats[comp]
                print(f"   {comp}:")
                print(
                    f"     P50: {stats['p50']:.3f}s, P90: {stats['p90']:.3f}s, P95: {stats['p95']:.3f}s, Avg: {stats['mean']:.3f}s"
                )

    return {
        "concurrent_requests": concurrent_requests,
        "success_rate": success_rate,
        "latency_stats": latency_stats,
        "throughput": throughput,
        "component_stats": component_stats,
    }


# =============================================================================
# SIMPLE BENCHMARK
# =============================================================================

if __name__ == "__main__":
    # API_URL = "https://recommendation-service-749244211103.us-central1.run.app/recommendations"
    API_URL = "http://localhost:8000/recommendations"

    # Load test data
    df = pd.read_json(
        "/root/recommendation-engine/data-root/df_user_profiles.json", lines=True
    )
    print(f"Loaded {len(df)} user profiles")

    test_levels = [1, 20, 50, 100, 500, 1000, 2000]
    results = []

    for level in test_levels:
        result = run_concurrent_test(df, API_URL, level)
        if result:
            results.append(result)

            # Stop if success rate drops below 90%
            if result["success_rate"] < 90:
                print(
                    f"⚠️  Success rate dropped to {result['success_rate']:.1f}% - stopping"
                )
                break
        else:
            print(f"❌ Test failed at {level} concurrent requests")
            break

        time.sleep(2)  # Brief pause between tests

    # Summary
    if results:
        print(f"\n{'='*80}")
        print("📊 BENCHMARK SUMMARY - LATENCY PERCENTILES")
        print(f"{'='*80}")
        print(
            f"{'Load':<6} {'Success%':<9} {'P50':<8} {'P90':<8} {'P95':<8} {'P99':<8} {'Avg':<8} {'Throughput'}"
        )
        print("-" * 80)

        for r in results:
            stats = r["latency_stats"]
            print(
                f"{r['concurrent_requests']:<6} "
                f"{r['success_rate']:<8.1f}% "
                f"{stats['p50']:<7.3f}s "
                f"{stats['p90']:<7.3f}s "
                f"{stats['p95']:<7.3f}s "
                f"{stats['p99']:<7.3f}s "
                f"{stats['mean']:<7.3f}s "
                f"{r['throughput']:<9.1f}/s"
            )

        # Find best performance
        best = max(
            results, key=lambda x: x["success_rate"] if x["success_rate"] >= 95 else 0
        )

        print(f"\n💡 RECOMMENDATIONS:")
        print(f"✅ Max stable load: {best['concurrent_requests']} concurrent requests")
        print(f"   - {best['success_rate']:.1f}% success rate")
        print(
            f"   - P50: {best['latency_stats']['p50']:.3f}s, P95: {best['latency_stats']['p95']:.3f}s"
        )
        print(f"   - {best['throughput']:.1f} requests/second")

        # Component insights
        if best.get("component_stats"):
            # Only consider components that actually exist in the best result
            slowest_comp = max(
                best["component_stats"].items(), key=lambda x: x[1]["mean"]
            )
            print(
                f"   - Slowest component: {slowest_comp[0]} (P95: {slowest_comp[1]['p95']:.3f}s)"
            )
    else:
        print("❌ No successful tests completed!")

# %%
