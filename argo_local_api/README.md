# Argo Local API

A Flask server that provides an OpenAI-compatible API interface for the internal ANL Argo API.

## Features

- OpenAI-compatible `/v1/chat/completions` endpoint
- Translates between OpenAI API format and internal Argo API format
- Supports multi-turn conversations with context preservation
- Extensive test suite for all functionality

## Files

- `server_chat.py` - The main server implementation with OpenAI-compatible interface
- `server.py` - Earlier version of server implementation
- `test_server_chat.py` - Comprehensive test suite with mocked responses

## Usage

### Setup

1. Install required dependencies:
```bash
pip install flask pytest requests
```

2. Run the server:
```bash
python server_chat.py
```

The server will start on port 5001 by default.

### Example Request

```bash
curl http://localhost:5001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gpto1preview",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Hello, who are you?"}
    ],
    "temperature": 0.7
  }'
```

### Running Tests

```bash
python -m pytest test_server_chat.py -v
```

## Notes

This service acts as an adapter between the OpenAI Chat API format and the internal Argo API, allowing tools built for OpenAI's API to work with ANL's internal LLM services. 