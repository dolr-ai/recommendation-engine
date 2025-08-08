import os
import sys
from huggingface_hub import login


def setup_huggingface_auth():
    """Set up authentication with Hugging Face if token is available"""
    hf_token = os.environ.get("HF_TOKEN")

    if hf_token:
        print(
            f"HF_TOKEN is set. Value: {hf_token[:4]}{'*' * (len(hf_token) - 8)}{hf_token[-4:]}"
        )
        try:
            login(token=hf_token)
            print("Successfully logged in to Hugging Face Hub")
        except Exception as e:
            print(f"Error logging in to Hugging Face Hub: {e}", file=sys.stderr)
    else:
        print(
            "HF_TOKEN environment variable not set. Skipping Hugging Face authentication."
        )
