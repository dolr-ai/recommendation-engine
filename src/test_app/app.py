"""
Streamlit App for Recommendation Engine Testing (Scroller Style, Cloud Run API)

This app provides a user interface to test the recommendation engine
by uploading user profiles and viewing recommendations in a scroller (Instagram/TikTok) style.
It uses a remote API endpoint for recommendations.
"""

import streamlit as st
import json
import pandas as pd
import sys
import os
import requests

from utils.common_utils import get_logger
from video_utils import (
    transform_video_id_to_url,
    format_video_display_name,
    validate_video_id,
    transform_multiple_video_ids_to_urls,
    video_transformer,
)

st.set_page_config(
    page_title="Recommendation Engine",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

logger = get_logger(__name__)

RECOMMENDATION_API_URL = os.getenv("RECOMMENDATION_API_URL")

st.markdown(
    """
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .video-metadata {
        font-size: 0.95rem;
        color: #555;
        margin-bottom: 0.5rem;
    }
    .video-card {
        background-color: #fff;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        padding: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.07);
    }
    .video-title {
        font-weight: bold;
        font-size: 1.1rem;
        margin-bottom: 0.3rem;
    }
    .video-scroll {
        max-height: 80vh;
        overflow-y: auto;
        padding-right: 10px;
    }
    .video-player {
        width: 100%;
        max-width: 350px;
        border-radius: 0.5rem;
        margin-bottom: 0.5rem;
    }
    .video-url-link {
        font-size: 0.9rem;
        color: #1f77b4;
        word-break: break-all;
    }
    .watched-label {
        color: #388e3c;
        font-weight: bold;
        font-size: 1rem;
    }
    .recommended-label {
        color: #d84315;
        font-weight: bold;
        font-size: 1rem;
    }
    .nsfw-badge {
        background-color: #ff6b6b;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        font-weight: bold;
    }
    .clean-badge {
        background-color: #51cf66;
        color: white;
        padding: 0.3rem 0.6rem;
        border-radius: 0.3rem;
        font-size: 0.8rem;
        font-weight: bold;
    }
</style>
""",
    unsafe_allow_html=True,
)


def load_user_profiles(uploaded_file):
    """Load user profiles from JSONL file (lines=True format)."""
    try:
        content = uploaded_file.read().decode("utf-8")
        profiles = []
        lines = content.strip().split("\n")

        for line_num, line in enumerate(lines, 1):
            if line.strip():  # Skip empty lines
                try:
                    data = json.loads(line.strip())
                    if isinstance(data, dict):
                        # Check if this is a request_params format
                        if "request_params" in data:
                            request_params = data["request_params"]
                            if isinstance(request_params, str):
                                request_params = json.loads(request_params)

                            user_profile = {
                                "user_id": request_params.get("user_id"),
                                "watch_history": request_params.get(
                                    "watch_history", []
                                ),
                                "nsfw_label": request_params.get("nsfw_label", False),
                                "request_params": request_params,
                            }
                            profiles.append(user_profile)
                        else:
                            # Regular user profile format
                            profiles.append(data)
                    else:
                        st.warning(f"Invalid JSON object at line {line_num}")
                except json.JSONDecodeError as e:
                    st.warning(f"Invalid JSON at line {line_num}: {e}")
                    continue

        return profiles
    except Exception as e:
        st.error(f"Error loading file: {e}")
        return []


def display_cache_stats():
    cache_stats = video_transformer.get_cache_stats()
    st.sidebar.markdown("### Cache Statistics")
    st.sidebar.metric("Cached URLs", cache_stats["cache_size"])
    if cache_stats["cache_size"] > 0:
        with st.sidebar.expander("Cached Video IDs"):
            for video_id in cache_stats["cached_video_ids"][:10]:
                st.write(f"• {video_id[:12]}...")
            if len(cache_stats["cached_video_ids"]) > 10:
                st.write(f"... and {len(cache_stats['cached_video_ids']) - 10} more")
    if st.sidebar.button("Clear Cache"):
        video_transformer.clear_cache()
        st.sidebar.success("Cache cleared!")


def render_video_player(url, height=320):
    st.components.v1.html(
        f'<iframe src="{url}" allowfullscreen controls style="width:100%;max-width:350px;height:{height}px;border-radius:8px;border:none;"></iframe>',
        height=height,
    )


def render_video_card(video_id, url, metadata: dict, label: str = None):
    st.markdown(f'<div class="video-card">', unsafe_allow_html=True)
    if label:
        st.markdown(
            f'<span class="{label}">{label.replace("-label", "").capitalize()}</span>',
            unsafe_allow_html=True,
        )
    st.markdown(f'<div class="video-title">{video_id}</div>', unsafe_allow_html=True)
    render_video_player(url)
    st.markdown(f'<div class="video-url-link">{url}</div>', unsafe_allow_html=True)
    # Metadata
    meta_lines = []
    for k, v in metadata.items():
        meta_lines.append(f"<b>{k.replace('_',' ').capitalize()}</b>: {v}")
    st.markdown(
        '<div class="video-metadata">' + "<br/>".join(meta_lines) + "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)


def fetch_recommendations_from_api(user_profile, request_params):
    """Fetch recommendations using the request_params from the uploaded data."""
    payload = {
        "user_id": user_profile["user_id"],
        "watch_history": [
            {
                "video_id": v.get("video_id"),
                "last_watched_timestamp": v.get("last_watched_timestamp"),
                "mean_percentage_watched": v.get("mean_percentage_watched"),
            }
            for v in user_profile.get("watch_history", [])
        ],
        "nsfw_label": request_params.get("nsfw_label"),
    }
    try:
        resp = requests.post(RECOMMENDATION_API_URL, json=payload, timeout=180)
        if resp.status_code == 200:
            return resp.json()
        else:
            st.error(f"API Error: {resp.status_code} {resp.text}")
            return None
    except Exception as e:
        st.error(f"API Request failed: {e}")
        return None


def main():
    st.markdown(
        '<h1 class="main-header">Recommendation Engine</h1>',
        unsafe_allow_html=True,
    )

    # Sidebar: Upload, user select
    st.sidebar.markdown("## Upload Data")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a JSON or CSV file with user profiles",
        type=["json", "csv"],
        help="Upload a JSON file containing user profile(s) or CSV file with request_params column.",
    )
    user_profiles = []
    user_options = []
    selected_user_idx = 0
    if uploaded_file is not None:
        user_profiles = load_user_profiles(uploaded_file)
        if user_profiles:
            user_options = [
                f"{i+1}: {u.get('user_id','N/A')[:16]}..."
                for i, u in enumerate(user_profiles)
            ]
            selected_user_idx = st.sidebar.selectbox(
                "Select User",
                range(len(user_profiles)),
                format_func=lambda i: user_options[i],
            )

    display_cache_stats()

    # Main layout: 3 columns (Watched | Spacer | Recommended)
    if user_profiles:
        profile = user_profiles[selected_user_idx]
        user_profile = {
            "user_id": profile.get("user_id"),
            "watch_history": profile.get("watch_history", []),
        }

        # Get request_params from the profile
        request_params = profile.get("request_params", {})
        nsfw_label = profile.get("nsfw_label", False)

        # Display request type badge
        request_type = "NSFW Request" if nsfw_label else "Clean Request"
        badge_class = "nsfw-badge" if nsfw_label else "clean-badge"
        st.markdown(
            f'<div style="text-align: center; margin-bottom: 1rem;">'
            f'<span class="{badge_class}">{request_type}</span>'
            f"</div>",
            unsafe_allow_html=True,
        )

        # Display request parameters info
        with st.expander("Request Parameters"):
            st.json(request_params)

        # Get recommendations from API
        recommendations = fetch_recommendations_from_api(user_profile, request_params)
        if not recommendations:
            st.error("No recommendations returned from API.")
            return

        # Watched video IDs
        watched_history = profile.get("watch_history", [])
        watched_video_ids = [v.get("video_id") for v in watched_history]
        watched_url_map = transform_multiple_video_ids_to_urls(watched_video_ids)

        # Recommended video IDs
        rec_posts = recommendations.get("posts", [])  # Get the posts array
        rec_video_ids = [
            post.get("video_id") for post in rec_posts if post.get("video_id")
        ]  # Extract video_ids
        rec_url_map = transform_multiple_video_ids_to_urls(rec_video_ids)
        rec_scores = recommendations.get("scores", {})
        rec_sources = recommendations.get("sources", {})

        # Layout
        col1, col_spacer, col2 = st.columns([1.2, 0.1, 1.2])
        with col1:
            st.markdown(
                '<div class="section-header">Watched Videos</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="video-scroll">', unsafe_allow_html=True)
            for i, v in enumerate(watched_history):
                vid = v.get("video_id")
                url = watched_url_map.get(vid, transform_video_id_to_url(vid))
                meta = {
                    "Watch %": f"{float(v.get('mean_percentage_watched', 0)) * 100:.1f}%",
                    "Last Watched": (
                        v.get("last_watched_timestamp", "N/A")[:19]
                        if v.get("last_watched_timestamp")
                        else "N/A"
                    ),
                }
                render_video_card(vid, url, meta, label="watched-label")
            st.markdown("</div>", unsafe_allow_html=True)
        with col2:
            st.markdown(
                '<div class="section-header">Recommended Videos</div>',
                unsafe_allow_html=True,
            )
            st.markdown('<div class="video-scroll">', unsafe_allow_html=True)
            for i, post in enumerate(rec_posts):
                vid = post.get("video_id")
                if not vid:
                    continue

                # Construct URL directly from canister_id and post_id
                canister_id = post.get("canister_id")
                post_id = post.get("post_id")
                if canister_id and post_id is not None:
                    url = f"https://yral.com/hot-or-not/{canister_id}/{post_id}"
                else:
                    # Fallback to video_id transformation if canister_id or post_id is missing
                    url = rec_url_map.get(vid, transform_video_id_to_url(vid))

                meta = {
                    "Score": f"{rec_scores.get(vid, 0):.4f}",
                    "NSFW Probability": f"{post.get('nsfw_probability', 0):.2f}",
                    "Post ID": str(post.get("post_id", "N/A")),
                    "Publisher": post.get("publisher_user_id", "N/A")[:16] + "...",
                }
                # Add source info if available
                source_info = rec_sources.get(vid, {})
                if source_info:
                    if isinstance(source_info, list):
                        meta["Source"] = ", ".join(
                            [
                                s.get("candidate_type", "Unknown")
                                for s in source_info[:2]
                            ]
                        )
                    else:
                        meta["Source"] = source_info.get("candidate_type", "Unknown")
                render_video_card(vid, url, meta, label="recommended-label")
            st.markdown("</div>", unsafe_allow_html=True)

    # Instructions
    with st.expander("How to Use"):
        st.markdown(
            """
        How to Use This App
        1. Upload Data: Upload a JSON file containing user profile(s) or CSV file with request_params column
        2. Select User: Use the sidebar to select a user
        3. View Request Type: The app will show if this is an NSFW or Clean request
        4. View Request Parameters: Expand the "Request Parameters" section to see the full request
        5. View Watched and Recommended Videos: Scroll through the center and right columns
        6. All video links are rendered as playable videos (iframe)

        CSV Format: The CSV file should have a 'request_params' column containing JSON data with the request parameters.
        """
        )


if __name__ == "__main__":
    main()
