#!/usr/bin/env python3
"""
Download models from Hugging Face.
Downloads both Mistral-7B-Instruct-v0.3 and Gemma 3 4B-IT.
"""

from huggingface_hub import snapshot_download
import os
from dotenv import load_dotenv
from models_config import MODELS

# Load environment variables from .env file
load_dotenv()


def download_model(model_config: dict) -> bool:
    """Download a single model from Hugging Face."""
    model_name = model_config["name"]
    hf_repo = model_config["hf_repo"]
    local_dir = model_config["local_path"]
    size_gb = model_config["size_gb"]

    print(f"Downloading {model_name}...")
    print(f"Repository: {hf_repo}")
    print(f"This will download ~{size_gb}GB of model files.")
    print(f"Files will be saved to: {local_dir}")
    print()

    # Check if model already exists
    if os.path.exists(local_dir) and os.listdir(local_dir):
        print(f"Model already exists at {local_dir}")
        print("Skipping download.")
        print()
        return True

    # Get HF token from environment
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        print("Warning: HF_TOKEN not found in .env file")
        print("You may need to authenticate if the model requires it")
        print()

    try:
        # Create local directory if it doesn't exist
        os.makedirs(local_dir, exist_ok=True)

        # Download the model
        path = snapshot_download(
            repo_id=hf_repo,
            local_dir=local_dir,
            local_dir_use_symlinks=False,
            token=hf_token,
        )

        print()
        print(f"✓ Model downloaded successfully to: {path}")
        print()
        return True

    except Exception as e:
        print(f"Error downloading model: {e}")
        print()
        print("Make sure you have:")
        print(f"- Enough disk space (~{size_gb}GB)")
        print("- Internet connection")
        print("- Accepted the model license on Hugging Face (if required)")
        if "gemma" in hf_repo.lower():
            print(f"- For Gemma models, visit: https://huggingface.co/{hf_repo}")
            print("  and accept the license terms")
        print()
        return False


def download_all_models():
    """Download all configured models."""
    print("=" * 70)
    print("vLLM Model Downloader")
    print("=" * 70)
    print()
    print(f"Will download {len(MODELS)} models:")
    for key, config in MODELS.items():
        print(f"  - {config['name']} (~{config['size_gb']}GB)")
    print()
    print(f"Total download size: ~{sum(m['size_gb'] for m in MODELS.values())}GB")
    print()

    success_count = 0
    for key, config in MODELS.items():
        if download_model(config):
            success_count += 1
        print("-" * 70)
        print()

    print("=" * 70)
    print(f"Download complete: {success_count}/{len(MODELS)} models ready")
    print("=" * 70)
    print()

    if success_count == len(MODELS):
        print("All models downloaded successfully!")
        print()
        print("You can now run the vLLM server with:")
        print("  uv run python server.py --model-name mistral")
        print("  uv run python server.py --model-name gemma3")
    else:
        print(f"Warning: {len(MODELS) - success_count} model(s) failed to download")
        print("Please check the errors above and try again.")


if __name__ == "__main__":
    download_all_models()
