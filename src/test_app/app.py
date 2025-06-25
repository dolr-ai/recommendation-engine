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
</style>
""",
    unsafe_allow_html=True,
)


def load_user_profiles(uploaded_file):
    try:
        content = uploaded_file.read()
        data = json.loads(content.decode("utf-8"))
        if isinstance(data, dict):
            return [data]
        elif isinstance(data, list):
            return data
        else:
            st.error("Invalid JSON format. Expected object or array.")
            return []
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


def fetch_recommendations_from_api(user_profile, params):
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
        **params,
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

    # Sidebar: Upload, user select, parameters
    st.sidebar.markdown("## Upload User Profiles")
    uploaded_file = st.sidebar.file_uploader(
        "Choose a JSON file with user profiles",
        type=["json"],
        help="Upload a JSON file containing user profile(s). Can be a single profile object or an array of profiles.",
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

    st.sidebar.markdown("## Parameters")
    top_k = st.sidebar.slider("Top K", min_value=5, max_value=100, value=25, step=5)
    fallback_top_k = st.sidebar.slider(
        "Fallback Top K", min_value=10, max_value=200, value=100, step=10
    )
    threshold = st.sidebar.slider(
        "Threshold", min_value=0.0, max_value=1.0, value=0.1, step=0.05
    )
    min_similarity = st.sidebar.slider(
        "Min Similarity", min_value=0.0, max_value=1.0, value=0.4, step=0.1
    )
    with st.sidebar.expander("Advanced Parameters"):
        enable_deduplication = st.checkbox("Enable Deduplication", value=True)
        max_workers = st.slider("Max Workers", min_value=1, max_value=8, value=4)
        max_fallback_candidates = st.slider(
            "Max Fallback Candidates", min_value=50, max_value=500, value=200, step=50
        )
        recency_weight = st.slider(
            "Recency Weight", min_value=0.0, max_value=1.0, value=0.8, step=0.1
        )
        watch_percentage_weight = st.slider(
            "Watch % Weight", min_value=0.0, max_value=1.0, value=0.2, step=0.1
        )
    display_cache_stats()

    # Main layout: 3 columns (Watched | Spacer | Recommended)
    if user_profiles:
        profile = user_profiles[selected_user_idx]
        user_profile = {
            "user_id": profile.get("user_id"),
            "watch_history": profile.get("watch_history", []),
        }
        params = {
            "top_k": top_k,
            "fallback_top_k": fallback_top_k,
            "threshold": threshold,
            "enable_deduplication": enable_deduplication,
            "max_workers": max_workers,
            "max_fallback_candidates": max_fallback_candidates,
            "min_similarity_threshold": min_similarity,
            "recency_weight": recency_weight,
            "watch_percentage_weight": watch_percentage_weight,
        }
        # Get recommendations from API
        recommendations = fetch_recommendations_from_api(user_profile, params)
        if not recommendations:
            st.error("No recommendations returned from API.")
            return
        # Watched video IDs
        watched_history = profile.get("watch_history", [])
        watched_video_ids = [v.get("video_id") for v in watched_history]
        watched_url_map = transform_multiple_video_ids_to_urls(watched_video_ids)
        # Recommended video IDs
        rec_video_ids = recommendations.get("recommendations", [])
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
            for i, vid in enumerate(rec_video_ids):
                url = rec_url_map.get(vid, transform_video_id_to_url(vid))
                meta = {
                    "Score": f"{rec_scores.get(vid, 0):.4f}",
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
        1. Upload User Profile: Upload a JSON file containing user profile(s)
        2. Select User: Use the sidebar to select a user
        3. Configure Parameters: Adjust parameters in the sidebar
        4. View Watched and Recommended Videos: Scroll through the center and right columns
        5. All video links are rendered as playable videos (iframe)
        """
        )


if __name__ == "__main__":
    main()
