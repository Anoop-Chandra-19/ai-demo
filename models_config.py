#!/usr/bin/env python3
"""
Model configuration for vLLM chat application.
Defines available models and their parameters.
"""

MODELS = {
    "mistral": {
        "name": "Mistral-7B-Instruct-v0.3",
        "hf_repo": "mistralai/Mistral-7B-Instruct-v0.3",
        "local_path": "./models/Mistral-7B-Instruct-v0.3",
        "context_length": 4096,
        "gpu_memory": 0.8,
        "size_gb": 14,
        "parameters": "7B",
        "dtype": "auto",
    },
    "gemma3": {
        "name": "Gemma 3 4B-IT",
        "hf_repo": "google/gemma-3-4b-it",
        "local_path": "./models/gemma-3-4b-it",
        "context_length": 8192,
        "gpu_memory": 0.7,
        "size_gb": 8,
        "parameters": "4B",
        "dtype": "bfloat16",
    },
}

DEFAULT_MODEL = "mistral"


def get_model_config(model_key: str) -> dict:
    """Get configuration for a specific model."""
    if model_key not in MODELS:
        available = ", ".join(MODELS.keys())
        raise ValueError(f"Unknown model '{model_key}'. Available models: {available}")
    return MODELS[model_key]


def list_models() -> list[str]:
    """List all available model keys."""
    return list(MODELS.keys())
