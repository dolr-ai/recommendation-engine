#!/usr/bin/env python3
"""
Analysis script for cache v2 endpoint test results.

This script analyzes JSON dumps from test_cache_v2_endpoint.py to derive meaningful
product metrics around recommendation system performance, content discovery, and user experience.

USAGE:
    # Analyze a single result file
    python analyze_cache_test_results.py --file cache_v2_test_results_20241219_143022.json

    # Analyze all JSON files in current directory
    python analyze_cache_test_results.py --directory .

    # Generate detailed report with plots
    python analyze_cache_test_results.py --file results.json --detailed --plots

METRICS PROVIDED:
    - Per-request already watched probability
    - Content discovery rate and freshness
    - Recommendation system effectiveness
    - User experience quality indicators
    - Performance and latency analysis
"""

import json
import os
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
import statistics
from dataclasses import dataclass


@dataclass
class RequestMetrics:
    """Metrics for a single request."""
    iteration: int
    already_watched_count: int
    total_videos: int
    already_watched_rate: float
    response_time_ms: float
    success: bool
    error: Optional[str] = None


@dataclass
class AnalysisResults:
    """Complete analysis results for a test run."""
    # Basic test info
    test_info: Dict[str, Any]

    # Request-level metrics
    request_metrics: List[RequestMetrics]

    # Aggregate metrics
    total_requests: int
    successful_requests: int
    failed_requests: int

    # Content discovery metrics
    avg_already_watched_rate: float
    content_discovery_rate: float  # Percentage of truly new content
    recommendation_effectiveness: float  # How well the system avoids repeats

    # Performance metrics
    avg_response_time_ms: float
    p95_response_time_ms: float
    p99_response_time_ms: float

    # User experience metrics
    freshness_score: float  # How much new content per request
    diversity_trend: List[float]  # Already watched rate over time
    recommendation_saturation_point: Optional[int]  # When repeats start dominating

    # Per-request already watched rate statistics
    already_watched_rates: List[float]  # All per-request rates
    median_already_watched_rate: float
    p25_already_watched_rate: float
    p75_already_watched_rate: float
    p90_already_watched_rate: float
    p95_already_watched_rate: float
    p99_already_watched_rate: float
    std_already_watched_rate: float

    # Per-request video count statistics
    video_counts: List[int]  # Total videos returned per request
    avg_videos_per_request: float
    median_videos_per_request: float
    min_videos_per_request: int
    max_videos_per_request: int
    std_videos_per_request: float

    # Content availability metrics
    expected_videos_per_request: int  # What was requested
    requests_with_fewer_videos: int  # Requests that returned fewer than expected
    content_shortage_rate: float  # Percentage of requests with content shortage

    # User consumption metrics
    total_videos_watched: int  # Total videos consumed by user during test
    final_watched_videos_in_redis: int  # Final count in Redis set
    avg_videos_watched_per_request: float  # Average consumption rate
    max_videos_watched_in_session: int  # Peak videos watched in single request
    watch_velocity_trend: List[int]  # Cumulative videos watched over time


class CacheTestAnalyzer:
    """Analyzer for cache v2 endpoint test results."""

    def __init__(self, json_file_path: str):
        self.json_file_path = json_file_path
        self.data = self._load_json_data()
        self.results = self._analyze()

    def _load_json_data(self) -> Dict[str, Any]:
        """Load and validate JSON test results."""
        try:
            with open(self.json_file_path, 'r') as f:
                data = json.load(f)

            required_fields = ['test_info', 'logs']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing required field: {field}")

            print(f"✅ Loaded test data from {self.json_file_path}")
            print(f"   Test timestamp: {data['test_info'].get('test_timestamp', 'unknown')}")
            print(f"   Total requests: {len(data.get('logs', []))}")

            return data

        except Exception as e:
            print(f"❌ Failed to load JSON file: {e}")
            raise

    def _analyze(self) -> AnalysisResults:
        """Perform comprehensive analysis of the test results."""
        test_info = self.data['test_info']
        logs = self.data['logs']

        # Extract request-level metrics
        request_metrics = []
        response_times = []

        for log_entry in logs:
            if not log_entry.get('request') or not log_entry.get('metrics'):
                continue

            request = log_entry['request']
            metrics = log_entry['metrics']
            response = log_entry.get('response', {})

            iteration = request.get('iteration', 0)
            already_watched = metrics.get('already_watched_this_request', 0)
            total_videos = metrics.get('total_videos_this_request', 0)
            response_time = response.get('response_time_ms', 0)
            success = response.get('status_code') == 200
            error = log_entry.get('error')

            already_watched_rate = (already_watched / total_videos * 100) if total_videos > 0 else 0

            request_metric = RequestMetrics(
                iteration=iteration,
                already_watched_count=already_watched,
                total_videos=total_videos,
                already_watched_rate=already_watched_rate,
                response_time_ms=response_time,
                success=success,
                error=error
            )

            request_metrics.append(request_metric)
            if response_time > 0:
                response_times.append(response_time)

        # Calculate aggregate metrics
        successful_requests = sum(1 for rm in request_metrics if rm.success)
        failed_requests = len(request_metrics) - successful_requests

        # Content discovery metrics
        successful_metrics = [rm for rm in request_metrics if rm.success]
        already_watched_rates = [rm.already_watched_rate for rm in successful_metrics]
        video_counts = [rm.total_videos for rm in successful_metrics]

        avg_already_watched_rate = statistics.mean(already_watched_rates) if already_watched_rates else 0
        content_discovery_rate = 100 - avg_already_watched_rate
        recommendation_effectiveness = max(0, 100 - avg_already_watched_rate)

        # Video count statistics
        expected_videos = test_info.get('num_results', 30)  # Default to 30 if not specified
        avg_videos = statistics.mean(video_counts) if video_counts else 0
        median_videos = statistics.median(video_counts) if video_counts else 0
        min_videos = min(video_counts) if video_counts else 0
        max_videos = max(video_counts) if video_counts else 0
        std_videos = statistics.stdev(video_counts) if len(video_counts) > 1 else 0

        # Content shortage analysis
        requests_with_fewer = sum(1 for count in video_counts if count < expected_videos)
        content_shortage_rate = (requests_with_fewer / len(video_counts) * 100) if video_counts else 0

        # User consumption metrics
        total_videos_watched = sum(rm.total_videos for rm in successful_metrics)
        final_watched_count = test_info.get('final_watched_videos_count', 0)
        avg_watched_per_request = total_videos_watched / len(successful_metrics) if successful_metrics else 0
        max_watched_in_session = max((rm.total_videos for rm in successful_metrics), default=0)

        # Calculate cumulative watch velocity trend
        cumulative_watched = 0
        watch_velocity_trend = []
        for rm in successful_metrics:
            cumulative_watched += rm.total_videos
            watch_velocity_trend.append(cumulative_watched)

        # Calculate quantiles and distribution statistics for already watched rates
        if already_watched_rates:
            median_rate = statistics.median(already_watched_rates)
            std_rate = statistics.stdev(already_watched_rates) if len(already_watched_rates) > 1 else 0

            # Calculate quantiles
            if len(already_watched_rates) >= 4:
                quantiles = statistics.quantiles(already_watched_rates, n=100)
                p25_rate = quantiles[24]  # 25th percentile (index 24 for 0-99)
                p75_rate = quantiles[74]  # 75th percentile
                p90_rate = quantiles[89]  # 90th percentile
                p95_rate = quantiles[94]  # 95th percentile
                p99_rate = quantiles[98]  # 99th percentile
            else:
                # Not enough data for quantiles
                sorted_rates = sorted(already_watched_rates)
                p25_rate = sorted_rates[0] if sorted_rates else 0
                p75_rate = sorted_rates[-1] if sorted_rates else 0
                p90_rate = sorted_rates[-1] if sorted_rates else 0
                p95_rate = sorted_rates[-1] if sorted_rates else 0
                p99_rate = sorted_rates[-1] if sorted_rates else 0
        else:
            median_rate = std_rate = p25_rate = p75_rate = p90_rate = p95_rate = p99_rate = 0

        # Performance metrics
        avg_response_time = statistics.mean(response_times) if response_times else 0
        p95_response_time = statistics.quantiles(response_times, n=20)[18] if len(response_times) >= 20 else (max(response_times) if response_times else 0)
        p99_response_time = statistics.quantiles(response_times, n=100)[98] if len(response_times) >= 100 else (max(response_times) if response_times else 0)

        # User experience metrics
        freshness_score = self._calculate_freshness_score(successful_metrics)
        diversity_trend = self._calculate_diversity_trend(successful_metrics)
        saturation_point = self._find_saturation_point(successful_metrics)

        return AnalysisResults(
            test_info=test_info,
            request_metrics=request_metrics,
            total_requests=len(request_metrics),
            successful_requests=successful_requests,
            failed_requests=failed_requests,
            avg_already_watched_rate=avg_already_watched_rate,
            content_discovery_rate=content_discovery_rate,
            recommendation_effectiveness=recommendation_effectiveness,
            avg_response_time_ms=avg_response_time,
            p95_response_time_ms=p95_response_time,
            p99_response_time_ms=p99_response_time,
            freshness_score=freshness_score,
            diversity_trend=diversity_trend,
            recommendation_saturation_point=saturation_point,
            already_watched_rates=already_watched_rates,
            median_already_watched_rate=median_rate,
            p25_already_watched_rate=p25_rate,
            p75_already_watched_rate=p75_rate,
            p90_already_watched_rate=p90_rate,
            p95_already_watched_rate=p95_rate,
            p99_already_watched_rate=p99_rate,
            std_already_watched_rate=std_rate,
            video_counts=video_counts,
            avg_videos_per_request=avg_videos,
            median_videos_per_request=median_videos,
            min_videos_per_request=min_videos,
            max_videos_per_request=max_videos,
            std_videos_per_request=std_videos,
            expected_videos_per_request=expected_videos,
            requests_with_fewer_videos=requests_with_fewer,
            content_shortage_rate=content_shortage_rate,
            total_videos_watched=total_videos_watched,
            final_watched_videos_in_redis=final_watched_count,
            avg_videos_watched_per_request=avg_watched_per_request,
            max_videos_watched_in_session=max_watched_in_session,
            watch_velocity_trend=watch_velocity_trend
        )

    def _calculate_freshness_score(self, metrics: List[RequestMetrics]) -> float:
        """Calculate content freshness score (0-100, higher = more fresh content)."""
        if not metrics:
            return 0

        total_new_content = sum(rm.total_videos - rm.already_watched_count for rm in metrics)
        total_content = sum(rm.total_videos for rm in metrics)

        return (total_new_content / total_content * 100) if total_content > 0 else 0

    def _calculate_diversity_trend(self, metrics: List[RequestMetrics]) -> List[float]:
        """Calculate rolling average of already watched rate to show trend over time."""
        if not metrics:
            return []

        window_size = min(10, len(metrics))  # 10-request rolling window
        trend = []

        for i in range(len(metrics)):
            start_idx = max(0, i - window_size + 1)
            window_metrics = metrics[start_idx:i+1]
            avg_rate = statistics.mean([rm.already_watched_rate for rm in window_metrics])
            trend.append(round(avg_rate, 2))

        return trend

    def _find_saturation_point(self, metrics: List[RequestMetrics]) -> Optional[int]:
        """Find the request number where already watched rate consistently exceeds 50%."""
        if len(metrics) < 10:
            return None

        window_size = 5
        saturation_threshold = 50.0

        for i in range(window_size, len(metrics)):
            window_metrics = metrics[i-window_size:i]
            avg_rate = statistics.mean([rm.already_watched_rate for rm in window_metrics])

            if avg_rate >= saturation_threshold:
                return metrics[i].iteration

        return None

    def print_summary_report(self):
        """Print a comprehensive summary report."""
        r = self.results

        print("\n" + "="*80)
        print("🎯 RECOMMENDATION SYSTEM ANALYSIS REPORT")
        print("="*80)

        # Test Overview
        print(f"\n📊 TEST OVERVIEW")
        print(f"   User ID: {r.test_info.get('user_id', 'unknown')}")
        print(f"   Content Type: {'NSFW' if r.test_info.get('nsfw_enabled') else 'Clean'}")
        print(f"   Results per request: {r.test_info.get('num_results', 'unknown')}")
        print(f"   Test duration: {r.total_requests} requests")
        print(f"   Success rate: {r.successful_requests}/{r.total_requests} ({r.successful_requests/r.total_requests*100:.1f}%)")

        # Content Discovery Metrics
        print(f"\n🔍 CONTENT DISCOVERY METRICS")
        print(f"   Average already watched rate: {r.avg_already_watched_rate:.2f}%")
        print(f"   Median already watched rate: {r.median_already_watched_rate:.2f}%")
        print(f"   Standard deviation: {r.std_already_watched_rate:.2f}%")
        print(f"   Content discovery rate: {r.content_discovery_rate:.2f}%")
        print(f"   Recommendation effectiveness: {r.recommendation_effectiveness:.2f}%")
        print(f"   Content freshness score: {r.freshness_score:.2f}%")

        # Per-Request Already Watched Rate Distribution
        print(f"\n📊 PER-REQUEST ALREADY WATCHED RATE DISTRIBUTION")
        print(f"   P25 (25th percentile): {r.p25_already_watched_rate:.2f}%")
        print(f"   P50 (median): {r.median_already_watched_rate:.2f}%")
        print(f"   P75 (75th percentile): {r.p75_already_watched_rate:.2f}%")
        print(f"   P90 (90th percentile): {r.p90_already_watched_rate:.2f}%")
        print(f"   P95 (95th percentile): {r.p95_already_watched_rate:.2f}%")
        print(f"   P99 (99th percentile): {r.p99_already_watched_rate:.2f}%")

        # Video Count Analysis
        print(f"\n📹 VIDEO COUNT ANALYSIS")
        print(f"   Expected videos per request: {r.expected_videos_per_request}")
        print(f"   Average videos returned: {r.avg_videos_per_request:.1f}")
        print(f"   Median videos returned: {r.median_videos_per_request:.0f}")
        print(f"   Min/Max videos returned: {r.min_videos_per_request}/{r.max_videos_per_request}")
        print(f"   Video count standard deviation: {r.std_videos_per_request:.2f}")
        print(f"   Requests with fewer than expected: {r.requests_with_fewer_videos}/{r.total_requests}")
        print(f"   Content shortage rate: {r.content_shortage_rate:.2f}%")

        if r.content_shortage_rate > 10:
            print(f"   ⚠️  High content shortage rate - system running out of content")
        elif r.content_shortage_rate > 0:
            print(f"   ✅ Low content shortage - system mostly providing expected content count")
        else:
            print(f"   ✅ No content shortage - system consistently providing expected content count")

        # User Consumption Analysis
        print(f"\n👤 USER CONSUMPTION ANALYSIS")
        print(f"   Total videos watched during test: {r.total_videos_watched:,}")
        print(f"   Final videos in Redis watched set: {r.final_watched_videos_in_redis:,}")
        print(f"   Average videos watched per request: {r.avg_videos_watched_per_request:.1f}")
        print(f"   Max videos watched in single request: {r.max_videos_watched_in_session}")

        if len(r.watch_velocity_trend) > 10:
            early_velocity = r.watch_velocity_trend[9] / 10  # First 10 requests average
            total_requests = len(r.watch_velocity_trend)
            late_velocity = (r.watch_velocity_trend[-1] - r.watch_velocity_trend[-11]) / 10 if total_requests > 10 else 0
            print(f"   Early session velocity: {early_velocity:.1f} videos/request")
            print(f"   Late session velocity: {late_velocity:.1f} videos/request")

            velocity_change = late_velocity - early_velocity
            if velocity_change > 5:
                print(f"   📈 Increasing consumption velocity (+{velocity_change:.1f} videos/request)")
            elif velocity_change < -5:
                print(f"   📉 Decreasing consumption velocity ({velocity_change:.1f} videos/request)")
            else:
                print(f"   ➡️  Stable consumption velocity ({velocity_change:+.1f} videos/request)")

        if r.recommendation_saturation_point:
            print(f"   ⚠️  Recommendation saturation at request: {r.recommendation_saturation_point}")
        else:
            print(f"   ✅ No recommendation saturation detected")

        # Performance Metrics
        print(f"\n⚡ PERFORMANCE METRICS")
        print(f"   Average response time: {r.avg_response_time_ms:.2f}ms")
        print(f"   P95 response time: {r.p95_response_time_ms:.2f}ms")
        print(f"   P99 response time: {r.p99_response_time_ms:.2f}ms")

        # User Experience Assessment
        print(f"\n👤 USER EXPERIENCE ASSESSMENT")
        if r.avg_already_watched_rate < 20:
            ux_rating = "🟢 EXCELLENT - High content discovery"
        elif r.avg_already_watched_rate < 40:
            ux_rating = "🟡 GOOD - Moderate content discovery"
        elif r.avg_already_watched_rate < 60:
            ux_rating = "🟠 FAIR - Limited content discovery"
        else:
            ux_rating = "🔴 POOR - High content repetition"

        print(f"   Overall rating: {ux_rating}")

        # Product Insights
        print(f"\n💡 PRODUCT INSIGHTS")

        if r.content_discovery_rate > 80:
            print("   ✅ Strong content discovery - users likely to stay engaged")
        elif r.content_discovery_rate > 60:
            print("   ⚠️  Moderate content discovery - room for improvement")
        else:
            print("   🚨 Low content discovery - risk of user churn")

        if r.recommendation_saturation_point and r.recommendation_saturation_point < 20:
            print("   🚨 Early saturation - recommendation pool may be too small")

        if r.avg_response_time_ms > 1000:
            print("   ⚠️  High response times - may impact user experience")

        # Trending Analysis
        if len(r.diversity_trend) > 20:
            early_trend = statistics.mean(r.diversity_trend[:10])
            late_trend = statistics.mean(r.diversity_trend[-10:])
            trend_change = late_trend - early_trend

            print(f"\n📈 TRENDING ANALYSIS")
            print(f"   Early requests already watched rate: {early_trend:.2f}%")
            print(f"   Recent requests already watched rate: {late_trend:.2f}%")
            print(f"   Trend: {'+' if trend_change > 0 else ''}{trend_change:.2f}% change")

            if trend_change > 10:
                print("   📈 Increasing repetition over time - content pool exhaustion")
            elif trend_change < -10:
                print("   📉 Improving diversity over time - good recommendation learning")
            else:
                print("   ➡️  Stable recommendation patterns")

    def print_detailed_report(self):
        """Print detailed per-request analysis."""
        r = self.results

        print(f"\n📋 DETAILED REQUEST ANALYSIS")
        print(f"{'Req':<4} {'Videos':<7} {'Watched':<8} {'Rate':<6} {'Time':<8} {'Status':<8}")
        print("-" * 50)

        for rm in r.request_metrics[:20]:  # Show first 20 requests
            status = "✅ OK" if rm.success else "❌ FAIL"
            print(f"{rm.iteration:<4} {rm.total_videos:<7} {rm.already_watched_count:<8} {rm.already_watched_rate:>5.1f}% {rm.response_time_ms:>7.1f}ms {status:<8}")

        if len(r.request_metrics) > 20:
            print(f"... ({len(r.request_metrics) - 20} more requests)")

    def save_analysis_report(self, output_file: Optional[str] = None):
        """Save analysis results to JSON file."""
        if not output_file:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            base_name = os.path.splitext(os.path.basename(self.json_file_path))[0]
            output_file = f"{base_name}_analysis_{timestamp}.json"

        # Prepare data for JSON serialization
        analysis_data = {
            "source_file": self.json_file_path,
            "analysis_timestamp": datetime.now().isoformat(),
            "test_info": self.results.test_info,
            "summary_metrics": {
                "total_requests": self.results.total_requests,
                "successful_requests": self.results.successful_requests,
                "failed_requests": self.results.failed_requests,
                "avg_already_watched_rate": self.results.avg_already_watched_rate,
                "median_already_watched_rate": self.results.median_already_watched_rate,
                "std_already_watched_rate": self.results.std_already_watched_rate,
                "content_discovery_rate": self.results.content_discovery_rate,
                "recommendation_effectiveness": self.results.recommendation_effectiveness,
                "avg_response_time_ms": self.results.avg_response_time_ms,
                "p95_response_time_ms": self.results.p95_response_time_ms,
                "p99_response_time_ms": self.results.p99_response_time_ms,
                "freshness_score": self.results.freshness_score,
                "recommendation_saturation_point": self.results.recommendation_saturation_point,
                "already_watched_rate_quantiles": {
                    "p25": self.results.p25_already_watched_rate,
                    "p50": self.results.median_already_watched_rate,
                    "p75": self.results.p75_already_watched_rate,
                    "p90": self.results.p90_already_watched_rate,
                    "p95": self.results.p95_already_watched_rate,
                    "p99": self.results.p99_already_watched_rate
                }
            },
            "diversity_trend": self.results.diversity_trend,
            "request_details": [
                {
                    "iteration": rm.iteration,
                    "already_watched_count": rm.already_watched_count,
                    "total_videos": rm.total_videos,
                    "already_watched_rate": rm.already_watched_rate,
                    "response_time_ms": rm.response_time_ms,
                    "success": rm.success,
                    "error": rm.error
                }
                for rm in self.results.request_metrics
            ]
        }

        with open(output_file, 'w') as f:
            json.dump(analysis_data, f, indent=2)

        print(f"\n💾 Analysis saved to: {output_file}")


def analyze_multiple_files(directory: str, pattern: str = "cache_v2_test_results_*.json"):
    """Analyze multiple test result files in a directory."""
    import glob

    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        print(f"❌ No files found matching pattern: {pattern}")
        return

    print(f"📁 Found {len(files)} test result files")

    all_results = []
    for file_path in sorted(files):
        try:
            print(f"\n🔍 Analyzing: {os.path.basename(file_path)}")
            analyzer = CacheTestAnalyzer(file_path)
            analyzer.print_summary_report()
            all_results.append((file_path, analyzer.results))
        except Exception as e:
            print(f"❌ Failed to analyze {file_path}: {e}")

    if len(all_results) > 1:
        print_comparative_analysis(all_results)


def print_comparative_analysis(results: List[Tuple[str, AnalysisResults]]):
    """Print comparative analysis across multiple test runs."""
    print("\n" + "="*80)
    print("🔄 COMPARATIVE ANALYSIS")
    print("="*80)

    print(f"{'File':<30} {'Discovery%':<12} {'Effectiveness%':<15} {'AvgTime(ms)':<12} {'Requests':<10}")
    print("-" * 80)

    for file_path, result in results:
        filename = os.path.basename(file_path)[:28]
        print(f"{filename:<30} {result.content_discovery_rate:>10.1f}% {result.recommendation_effectiveness:>13.1f}% {result.avg_response_time_ms:>10.1f}ms {result.total_requests:>8}")

    # Calculate overall statistics
    all_discovery_rates = [r.content_discovery_rate for _, r in results]
    all_response_times = [r.avg_response_time_ms for _, r in results]

    print("\n📊 OVERALL STATISTICS")
    print(f"   Average content discovery rate: {statistics.mean(all_discovery_rates):.2f}%")
    print(f"   Best content discovery rate: {max(all_discovery_rates):.2f}%")
    print(f"   Worst content discovery rate: {min(all_discovery_rates):.2f}%")
    print(f"   Average response time: {statistics.mean(all_response_times):.2f}ms")


def main():
    parser = argparse.ArgumentParser(description="Analyze cache v2 endpoint test results")
    parser.add_argument("--file", help="JSON file to analyze")
    parser.add_argument("--directory", help="Directory containing multiple JSON files to analyze")
    parser.add_argument("--detailed", action="store_true", help="Show detailed per-request analysis")
    parser.add_argument("--save", help="Save analysis results to specified JSON file")

    args = parser.parse_args()

    if args.directory:
        analyze_multiple_files(args.directory)
    elif args.file:
        try:
            analyzer = CacheTestAnalyzer(args.file)
            analyzer.print_summary_report()

            if args.detailed:
                analyzer.print_detailed_report()

            if args.save:
                analyzer.save_analysis_report(args.save)
            else:
                analyzer.save_analysis_report()

        except Exception as e:
            print(f"❌ Analysis failed: {e}")
            return 1
    else:
        print("❌ Please specify either --file or --directory")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())