from flask import Flask, request, jsonify
import requests
import json
import time
import uuid

app = Flask(__name__)

# The internal endpoint you currently have
INTERNAL_API_URL = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/"

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    # Expecting a JSON body similar to OpenAI format:
    # {
    #   "model": "gpto1preview",
    #   "messages": [
    #       {"role": "system", "content": "..."},
    #       {"role": "user", "content": "..."}
    #   ],
    #   "temperature": 0.0,
    #   "top_p": 1.0,
    #   "max_tokens": 2056,
    #   "stop": [...]
    # }

    data = request.get_json()

    # Extract relevant fields with fallbacks
    model = data.get("model", "gpto1preview")
    messages = data.get("messages", [])
    temperature = data.get("temperature", 0.0)
    top_p = data.get("top_p", 1.0)
    max_tokens = data.get("max_tokens", 2056)
    stop = data.get("stop", [])

    # Convert messages into a format suitable for your internal API
    # The internal code snippet uses 'system' as a separate field,
    # and 'prompt' as a list (presumably just user content).
    # We will extract the system message if present and the user content.
    
    system_prompt = ""
    user_prompts = []

    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            system_prompt = content
        elif role == "user":
            # Append user message(s) to prompt list
            user_prompts.append(content)
        elif role == "assistant":
            # If there's assistant role messages (e.g. from a conversation),
            # you may want to include them as context. Here we just ignore or append them as needed.
            # Since the original code sets "prompt" to user messages only, we typically wouldn't
            # send assistant messages to the "prompt" field. 
            pass

    # Build the payload for the internal API
    payload = {
        "user": "cels", 
        "model": model,
        "system": system_prompt,
        "prompt": user_prompts,
        "stop": stop,
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens
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

    # Extract the AI's response
    ai_response = response_json.get('response')
    if not ai_response:
        return jsonify({"error": "No response from the AI"}), 500

    # Construct an OpenAI-compliant response
    # Example response format:
    # {
    #   "id": "chatcmpl-12345",
    #   "object": "chat.completion",
    #   "created": 1234567890,
    #   "model": "gpto1preview",
    #   "choices": [
    #       {
    #         "index": 0,
    #         "message": {
    #           "role": "assistant",
    #           "content": ai_response
    #         },
    #         "finish_reason": "stop"
    #       }
    #   ]
    # }

    openai_response = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": ai_response
                },
                "finish_reason": "stop"
            }
        ]
    }

    return jsonify(openai_response), 200

if __name__ == '__main__':
    # Run the server on port 5000 or change as needed.
    # In production, you'd likely run behind a WSGI server like gunicorn.
    app.run(host='0.0.0.0', port=5001, debug=True)
