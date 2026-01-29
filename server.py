#!/usr/bin/env python3
"""
vLLM server launcher with multi-model support.
Runs an OpenAI-compatible API server on localhost:8000.
"""

import subprocess
import sys
import os
import argparse
from models_config import MODELS, DEFAULT_MODEL, get_model_config

HOST = "0.0.0.0"
PORT = 8000


def start_server(model_key: str, port: int) -> None:
    """Start vLLM server with the specified model."""
    # Get model configuration
    try:
        config = get_model_config(model_key)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)

    model_name = config["name"]
    hf_repo = config["hf_repo"]
    local_path = config["local_path"]
    context_length = config["context_length"]
    gpu_memory = config["gpu_memory"]
    dtype = config["dtype"]

    # Check if model exists locally
    if os.path.exists(local_path):
        print(f"Using local model from: {local_path}")
        model_arg = local_path
    else:
        print(f"Local model not found. vLLM will download from Hugging Face: {hf_repo}")
        print(
            "Consider running 'python download_model.py' first to download the model."
        )
        model_arg = hf_repo

    print()
    print("Starting vLLM server...")
    print(f"Model: {model_name}")
    print(f"Context Length: {context_length} tokens")
    print(f"GPU Memory Utilization: {gpu_memory * 100}%")
    print(f"Data Type: {dtype}")
    print(f"API endpoint: http://localhost:{port}")
    print(f"OpenAI-compatible endpoint: http://localhost:{port}/v1")
    print()
    print("Press Ctrl+C to stop the server")
    print()

    # vLLM command with model-specific optimizations
    cmd = [
        "python",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        model_arg,
        "--served-model-name",
        hf_repo,  # Serve with HuggingFace model name
        "--host",
        HOST,
        "--port",
        str(port),
        "--max-model-len",
        str(context_length),
        "--gpu-memory-utilization",
        str(gpu_memory),
        "--enforce-eager",  # Disable CUDA graphs for compatibility
        "--dtype",
        dtype,
    ]

    try:
        subprocess.run(cmd)
    except KeyboardInterrupt:
        print("\nServer stopped.")
        sys.exit(0)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="vLLM server launcher with multi-model support"
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL,
        choices=list(MODELS.keys()),
        help=f"Model to load (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=PORT,
        help=f"Port to run server on (default: {PORT})",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available models and exit",
    )

    args = parser.parse_args()

    # Handle list models
    if args.list_models:
        print("Available models:")
        for key, config in MODELS.items():
            default_marker = " (default)" if key == DEFAULT_MODEL else ""
            print(f"  {key}: {config['name']}{default_marker}")
            print(f"    - Parameters: {config['parameters']}")
            print(f"    - Context: {config['context_length']} tokens")
            print(f"    - Size: ~{config['size_gb']}GB")
        return

    # Update PORT if specified
    # Start server
    start_server(args.model_name, args.port)


if __name__ == "__main__":
    main()
