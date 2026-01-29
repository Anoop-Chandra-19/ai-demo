#!/usr/bin/env python3
"""
CLI chat client for vLLM server
Connects to localhost:8000 and provides interactive chat with streaming responses.
"""

import argparse
import sys
from openai import OpenAI

DEFAULT_MODEL = "mistralai/Mistral-7B-Instruct-v0.3"

USER_COLOR = "\033[32m"
ASSISTANT_COLOR = "\033[36m"
ERROR_COLOR = "\033[31m"
BOLD = "\033[1m"
RESET_BOLD = "\033[22m"
RESET = "\033[0m"


def format_prefix(label: str, color: str, use_color: bool) -> str:
    if not use_color:
        return label
    return f"{BOLD}{color}{label}{RESET_BOLD}"


def colorize(text: str, color: str, use_color: bool) -> str:
    if not use_color:
        return text
    return f"{color}{text}{RESET}"


def init_tokenizer(model_name: str):
    try:
        from transformers import AutoTokenizer
    except Exception:
        return None, "Transformers is not installed"
    try:
        return AutoTokenizer.from_pretrained(model_name, use_fast=True), None
    except Exception as exc:
        return None, str(exc)


def update_status_line(text: str, enabled: bool) -> None:
    if not enabled:
        return
    sys.stdout.write("\0337")
    sys.stdout.write("\033[1E")
    sys.stdout.write("\r\033[K" + text)
    sys.stdout.write("\0338")
    sys.stdout.flush()


def finish_status_line(enabled: bool) -> None:
    if not enabled:
        print()
        print()
        return
    sys.stdout.write("\033[1E")
    sys.stdout.write("\r")
    sys.stdout.flush()
    print()
    print()


def chat(
    port: int,
    model_name: str | None,
    use_color: bool,
    enable_token_count: bool,
) -> None:
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

    tokenizer = None
    token_count_enabled = enable_token_count and sys.stdout.isatty()
    if token_count_enabled:
        tokenizer, tokenizer_error = init_tokenizer(model_name)
        if tokenizer is None:
            token_count_enabled = False
            print(
                f"Token counting disabled: {tokenizer_error}."
                " Install transformers to enable it."
            )

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
            prompt = format_prefix("You: ", USER_COLOR, use_color)
            user_input = input(prompt).strip()
            if use_color:
                print(RESET, end="", flush=True)
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
        assistant_prefix = format_prefix("Assistant: ", ASSISTANT_COLOR, use_color)
        print(assistant_prefix, end="", flush=True)

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
            token_count = 0
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    content = chunk.choices[0].delta.content
                    if use_color:
                        print(f"{ASSISTANT_COLOR}{content}{RESET}", end="", flush=True)
                    else:
                        print(content, end="", flush=True)
                    full_response += content
                    if token_count_enabled and tokenizer is not None:
                        token_count = len(tokenizer.encode(full_response))
                        update_status_line(f"Tokens: {token_count}", True)

            finish_status_line(token_count_enabled)

            # Add assistant response to history
            messages.append({"role": "assistant", "content": full_response})

        except Exception as e:
            error_text = colorize(f"\n\nError: {e}", ERROR_COLOR, use_color)
            print(error_text)
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
    parser.add_argument(
        "--no-color",
        action="store_true",
        help="Disable ANSI colors in output",
    )
    parser.add_argument(
        "--token-count",
        action="store_true",
        help="Enable live token counting in output",
    )
    args = parser.parse_args()

    use_color = sys.stdout.isatty() and not args.no_color
    enable_token_count = args.token_count
    chat(args.port, args.model, use_color, enable_token_count)
