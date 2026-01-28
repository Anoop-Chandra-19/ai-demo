#!/usr/bin/env python3
"""
vLLM server for Mistral-7B-Instruct-v0.3
Runs an OpenAI-compatible API server on localhost:8000
"""

import subprocess
import sys
import os

MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MODEL_PATH = "./models/Mistral-7B-Instruct-v0.3"
HOST = "0.0.0.0"
PORT = 8000


def start_server():
    # Check if model exists locally
    if os.path.exists(MODEL_PATH):
        print(f"Using local model from: {MODEL_PATH}")
        model_arg = MODEL_PATH
    else:
        print(
            f"Local model not found. vLLM will download from Hugging Face: {MODEL_NAME}"
        )
        print(
            "Consider running 'python download_model.py' first to download the model."
        )
        model_arg = MODEL_NAME

    print()
    print("Starting vLLM server...")
    print(f"Model: {model_arg}")
    print(f"API endpoint: http://localhost:{PORT}")
    print(f"OpenAI-compatible endpoint: http://localhost:{PORT}/v1")
    print()
    print("Press Ctrl+C to stop the server")
    print()

    # vLLM command with optimizations for RTX 4090
    # Using --enforce-eager for better compatibility
    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_arg,
        "--served-model-name",
        MODEL_NAME,  # Serve with HuggingFace model name
        "--host",
        HOST,
        "--port",
        str(PORT),
        "--max-model-len",
        "4096",  # Context length
        "--gpu-memory-utilization",
        "0.8",  # Use 80% of available GPU memory
        "--enforce-eager",  # Disable CUDA graphs for compatibility
        "--dtype",
        "auto",  # Automatic precision selection
    ]

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nServer stopped.")
        sys.exit(0)


if __name__ == "__main__":
    start_server()
