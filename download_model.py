#!/usr/bin/env python3
"""
Download Mistral-7B-Instruct-v0.3 model from Hugging Face.
This will download the model to the default Hugging Face cache directory.
"""

from huggingface_hub import snapshot_download
import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
LOCAL_DIR = "./models/Mistral-7B-Instruct-v0.3"


def download_model():
    print(f"Downloading {MODEL_NAME}...")
    print("This will download ~14GB of model files.")
    print(f"Files will be saved to: {LOCAL_DIR}")
    print()

    # Get HF token from environment
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN not found in .env file")
        print("You may need to authenticate if the model requires it")
        print()

    try:
        # Create local directory if it doesn't exist
        os.makedirs(LOCAL_DIR, exist_ok=True)

        # Download the model
        path = snapshot_download(
            repo_id=MODEL_NAME,
            local_dir=LOCAL_DIR,
            local_dir_use_symlinks=False,
            token=hf_token,
        )

        print()
        print(f"✓ Model downloaded successfully to: {path}")
        print()
        print("You can now run the vLLM server with:")
        print("  python server.py")

    except Exception as e:
        print(f"Error downloading model: {e}")
        print()
        print("Make sure you have enough disk space (~14GB) and internet connection.")
        return False

    return True


if __name__ == "__main__":
    download_model()
