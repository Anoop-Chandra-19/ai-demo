#!/usr/bin/env python3
"""
CLI chat client for vLLM server
Connects to localhost:8000 and provides interactive chat with streaming responses.
"""

import argparse
from openai import OpenAI

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"


def chat(port: int, model_name: str | None) -> None:
    """Interactive chat loop with streaming responses and history."""

    # Configure OpenAI client to point to vLLM server
    client = OpenAI(
        base_url=f"http://localhost:{port}/v1",
        api_key="dummy-key",  # vLLM doesn't require auth by default
    )

    if model_name is None:
        try:
            models = client.models.list()
            if models.data:
                model_name = models.data[0].id
            else:
                model_name = DEFAULT_MODEL
        except Exception:
            model_name = DEFAULT_MODEL

    # Chat history (list of message dicts with 'role' and 'content')
    messages = []

    print("=" * 70)
    print(f"vLLM Chat Demo - {model_name}")
    print("=" * 70)
    print("Type your message and press Enter. Type 'quit' or 'exit' to end.")
    print()

    while True:
        # Get user input
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nGoodbye!")
            break

        # Check for exit commands
        if user_input.lower() in ["quit", "exit", "q"]:
            print("Goodbye!")
            break

        # Skip empty input
        if not user_input:
            continue

        # Add user message to history
        messages.append({"role": "user", "content": user_input})

        # Get streaming response from vLLM
        print("Assistant: ", end="", flush=True)

        try:
            stream = client.chat.completions.create(
                model=model_name,
                messages=messages,
                stream=True,
                max_tokens=512,
                temperature=0.7,
            )

            # Collect full response while streaming
            full_response = ""
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    print(content, end="", flush=True)
                    full_response += content

            print()  # Newline after response
            print()  # Extra spacing

            # Add assistant response to history
            messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            print(f"\n\nError: {e}")
            print("\nMake sure the vLLM server is running: python server.py")
            print()
            # Remove the user message since we didn't get a response
            messages.pop()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="vLLM CLI chat client")
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port of the vLLM server (default: 8000)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model to use (default: auto-detect from server)",
    )
    args = parser.parse_args()

    chat(args.port, args.model)
