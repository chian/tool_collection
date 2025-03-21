from flask import Flask, request, jsonify
import requests
import json
import time
import uuid
import os

app = Flask(__name__)

# The internal endpoint you currently have
INTERNAL_API_URL = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/"

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """
    Emulates the OpenAI Chat endpoint: POST /v1/chat/completions
    
    Expected JSON body (example):
    {
        "model": "gpto1preview",
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."}
        ],
        "temperature": 0.7,
        "top_p": 1.0,
        "n": 1,
        "stream": false,
        "stop": ["\\n"],
        "max_tokens": 2056,
        "presence_penalty": 0.0,
        "frequency_penalty": 0.0,
        "logit_bias": {},
        "user": "some_user_id"
    }
    """

    data = request.get_json()

    # Extract relevant fields with fallbacks
    model = data.get("model", "gpto1preview")
    messages = data.get("messages", [])
    temperature = data.get("temperature", 0.0)
    top_p = data.get("top_p", 1.0)
    n = data.get("n", 1)
    stream = data.get("stream", False)
    stop = data.get("stop", [])
    max_tokens = data.get("max_tokens", 2056)

    # Optional fields often used in Chat completions, but may not be supported by your backend:
    presence_penalty = data.get("presence_penalty", 0.0)
    frequency_penalty = data.get("frequency_penalty", 0.0)
    logit_bias = data.get("logit_bias", {})
    user_id = data.get("user", os.getenv("USER"))  # Use OS username

    # Separate system vs. user vs. assistant messages:
    system_prompt = ""
    user_prompts = []
    assistant_contexts = []
    
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            system_prompt = content
        elif role == "user":
            user_prompts.append(content)
        elif role == "assistant":
            # Optionally add assistant role messages to context if your backend supports it.
            assistant_contexts.append(content)
        # You can ignore other roles or handle them as needed.

    # Build payload for your internal API.
    # (Adjust the fields to match what your internal API expects.)
    # For example, let's assume your internal endpoint only uses:
    #   user, model, system, prompt, stop, temperature, top_p, max_tokens
    # If your internal endpoint can handle or needs assistant messages,
    # you might include them as part of 'prompt' or a separate field.

    payload = {
        "user": user_id,               # from "user" in OpenAI request or fallback
        "model": model,
        "system": system_prompt,
        "prompt": user_prompts,
        "stop": stop,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
        # presence_penalty, frequency_penalty, etc. can be added if your internal API supports them.
    }

    headers = {
        "Content-Type": "application/json"
    }

    # Forward the request to the internal endpoint
    try:
        response = requests.post(INTERNAL_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        response_json = response.json()
    except requests.exceptions.RequestException as e:
        return jsonify({"error": str(e)}), 500
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON response from internal API"}), 500

    # Extract the AI's response from your backend
    # (Adjust key name if your backend returns something different)
    ai_response = response_json.get('response')
    if not ai_response:
        return jsonify({"error": "No response from the AI"}), 500

    # Construct an OpenAI-compliant response
    # Typically includes an ID, timestamp, model, and an array of choices.
    # You may also include usage tokens if your internal API provides them.

    openai_response = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [],
        # 'usage' can be computed or provided by your internal API if available:
        "usage": {
            "prompt_tokens": 0,        # replace with actual usage if known
            "completion_tokens": 0,    # replace with actual usage if known
            "total_tokens": 0          # replace with actual usage if known
        }
    }

    # Since the user can request n completions, handle n responses.
    # If your internal API can only handle 1 response at a time, you might
    # replicate the same answer n times or change your backend to support n.
    for i in range(n):
        choice = {
            "index": i,
            "message": {
                "role": "assistant",
                "content": ai_response  # or a loop if your backend returns multiple completions
            },
            "finish_reason": "stop"
        }
        openai_response["choices"].append(choice)

    # If streaming is requested, you'd typically return a chunked response.
    # For simplicity, we return a single JSON if stream=False.
    # In a real scenario, you would implement SSE or chunked responses.

    return jsonify(openai_response), 200

if __name__ == '__main__':
    # In production, use a production server (e.g. gunicorn).
    app.run(host='0.0.0.0', port=5001, debug=True)
