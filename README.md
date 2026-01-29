# vLLM Demo with Mistral-7B-Instruct-v0.3

A simple demo for running Mistral-7B-Instruct-v0.3 locally using vLLM with an interactive CLI chat interface.

## Features

- 🚀 Fast inference with vLLM optimization
- 💬 Streaming chat responses
- 📝 Conversation history maintained across turns
- 🎯 OpenAI-compatible API server
- 🖥️ Simple terminal-based UI

## Requirements

- Python 3.11+
- NVIDIA GPU with CUDA support (tested on RTX 4090 with 24GB VRAM)
- ~14GB disk space for model files
- ~8-10GB GPU VRAM for inference

## Installation

1. Install dependencies:

```bash
pip install -e .
```

2. Download the Mistral-7B-Instruct-v0.3 model:

```bash
python download_model.py
```

This will download ~14GB of model files to `./models/Mistral-7B-Instruct-v0.3/`

## Usage

### Step 1: Start the vLLM Server

In one terminal, start the vLLM server:

```bash
python server.py
```

The server will start on `http://localhost:8000` and expose an OpenAI-compatible API.

Wait for the model to load (you'll see "Application startup complete" message).

### Step 2: Start the Chat Client

In another terminal, start the interactive chat client:

```bash
python client.py
# or explicitly set the model:
python client.py --model google/gemma-3-4b-it
# or connect to a custom port:
python client.py --port 8080
```

Now you can chat with the model! Type your messages and press Enter. The model will respond with streaming output.

Type `quit`, `exit`, or `q` to end the conversation.

## Example

```
======================================================================
vLLM Chat Demo - Mistral-7B-Instruct-v0.3
======================================================================
Type your message and press Enter. Type 'quit' or 'exit' to end.

You: What is the capital of France?
Assistant: The capital of France is Paris. It is the largest city in France and serves as the country's political, economic, and cultural center.

You: Tell me a fun fact about it
Assistant: Here's a fun fact about Paris: The Eiffel Tower was originally intended to be a temporary structure! It was built for the 1889 World's Fair and was supposed to be dismantled after 20 years. However, it proved too useful as a radio transmission tower and has become one of the most iconic landmarks in the world.

You: quit
Goodbye!
```

## Project Structure

```
ai-demo/
├── download_model.py   # Script to download the model from Hugging Face
├── server.py          # vLLM server launcher
├── client.py          # Interactive CLI chat client
├── models/            # Downloaded model files (gitignored)
├── pyproject.toml     # Python dependencies
└── README.md          # This file
```

## Configuration

You can modify these parameters in the respective files:

**server.py:**
- `--gpu-memory-utilization`: GPU memory usage (default: 0.9 = 90%)
- `--max-model-len`: Maximum context length (default: 4096 tokens)
- `--port`: Server port (default: 8000)

**client.py:**
- `max_tokens`: Maximum tokens per response (default: 512)
- `temperature`: Sampling temperature (default: 0.7)

## Troubleshooting

**Server fails to start:**
- Make sure you have CUDA installed and GPU is available
- Check GPU memory usage with `nvidia-smi`
- Try reducing `--gpu-memory-utilization` in server.py

**Client connection error:**
- Make sure the server is running first
- Check that the server shows "Application startup complete"
- Verify the server is on http://localhost:8000

**Model download fails:**
- Check internet connection and disk space (~14GB needed)
- You may need to accept the model license on Hugging Face first

## License

This demo project is provided as-is. The Mistral-7B-Instruct-v0.3 model is subject to its own license terms on Hugging Face.
