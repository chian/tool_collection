from flask import Flask, request, jsonify
import requests
import json
import time
import uuid
import os
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import HumanMessage, AIMessage, SystemMessage

app = Flask(__name__)

# The internal endpoint you currently have
INTERNAL_API_URL = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/"

# Store conversations in memory
# In production, you might want to use Redis or another persistent store
conversation_memories = {}

def get_memory_for_session(session_id):
    """Get or create a window-based conversation memory for a session"""
    if session_id not in conversation_memories:
        # Create a memory system that keeps the last k conversation turns
        conversation_memories[session_id] = ConversationBufferWindowMemory(
            k=10,  # Store last 10 interactions (can be adjusted)
            return_messages=True,
            memory_key="chat_history"
        )
    return conversation_memories[session_id]

@app.route('/v1/chat/completions', methods=['POST'])
def chat_completions():
    """
    Emulates the OpenAI Chat endpoint: POST /v1/chat/completions
    
    Expected JSON body (example):
    {
        "model": "gpto1preview",
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."},
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

    # Use the user_id as the session identifier for conversation tracking
    session_id = user_id
    
    # Get memory for this session
    memory = get_memory_for_session(session_id)
    
    # Process incoming messages
    system_prompt = ""
    latest_user_message = None
    
    # If client is sending full conversation history
    if len(messages) > 1:
        # Clear existing memory and load from messages
        memory.clear()
        
        # Process each message in the conversation history
        for msg in messages:
            role = msg.get("role")
            content = msg.get("content", "")
            
            if role == "system":
                system_prompt = content
            elif role == "user":
                # For saving to memory
                if latest_user_message is None or msg == messages[-1]:
                    latest_user_message = content
                
                # Add to memory - recreates conversation
                memory.save_context({"input": content}, {"output": ""})
            elif role == "assistant":
                # Update the previous empty output with the actual assistant response
                memory.chat_memory.messages[-1] = AIMessage(content=content)
    
    # If client is only sending a new message (stateful mode)
    elif len(messages) == 1 and messages[0].get("role") == "user":
        latest_user_message = messages[0].get("content", "")
        # System prompt might have been set earlier
        system_msgs = [m for m in memory.chat_memory.messages if isinstance(m, SystemMessage)]
        if system_msgs:
            system_prompt = system_msgs[0].content
    
    # Extract the chat history
    chat_history = memory.load_memory_variables({})
    
    # Get all accumulated user messages for context
    user_prompts = []
    assistant_contexts = []
    
    # Process memory to get user and assistant messages
    for msg in memory.chat_memory.messages:
        if isinstance(msg, HumanMessage):
            user_prompts.append(msg.content)
        elif isinstance(msg, AIMessage):
            assistant_contexts.append(msg.content)
    
    # If we have a latest user message and it's not already in the prompts, add it
    if latest_user_message and (not user_prompts or user_prompts[-1] != latest_user_message):
        user_prompts.append(latest_user_message)
    
    # Format the conversation for the Argo API
    if assistant_contexts:
        # Create a formatted conversation history alternating between user and assistant
        formatted_prompts = []
        # We need to combine user and assistant messages in order
        # Ensure we don't go beyond the available messages
        last_user_idx = min(len(user_prompts), len(assistant_contexts) + 1)
        
        # Format in pairs: user message followed by assistant response
        for i in range(len(assistant_contexts)):
            if i < len(user_prompts):
                formatted_prompts.append(f"USER: {user_prompts[i]}")
                formatted_prompts.append(f"ASSISTANT: {assistant_contexts[i]}")
        
        # Add the latest user message if it exists (the one waiting for a response)
        if last_user_idx < len(user_prompts):
            formatted_prompts.append(f"USER: {user_prompts[last_user_idx]}")
        
        # Create a single prompt string
        combined_prompt = "\n\n".join(formatted_prompts)
        
        # Ensure the prompt ends with "ASSISTANT: " to prompt the AI to respond as the assistant
        if not combined_prompt.endswith("ASSISTANT: "):
            combined_prompt += "\n\nASSISTANT: "
            
        # Replace the user_prompts with our formatted conversation
        user_prompts = [combined_prompt]
    elif latest_user_message:
        # If no history but we have a message, use it directly
        user_prompts = [latest_user_message]
    
    # Build payload for the internal API
    payload = {
        "user": user_id,
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

    print(f"Sending request to Argo API: {json.dumps(payload, indent=2)}")

    # Forward the request to the internal endpoint
    try:
        response = requests.post(INTERNAL_API_URL, headers=headers, data=json.dumps(payload))
        response.raise_for_status()
        response_json = response.json()
        print(f"Received response from Argo API: {json.dumps(response_json, indent=2)}")
    except requests.exceptions.RequestException as e:
        print(f"Error from Argo API: {str(e)}")
        return jsonify({"error": str(e)}), 500
    except json.JSONDecodeError:
        print("Invalid JSON response from Argo API")
        return jsonify({"error": "Invalid JSON response from internal API"}), 500

    # Extract the AI's response from your backend
    ai_response = response_json.get('response')
    if not ai_response:
        print("No response field in Argo API response")
        return jsonify({"error": "No response from the AI"}), 500

    # Save the latest context - both the user input and the AI response
    if latest_user_message:
        memory.save_context({"input": latest_user_message}, {"output": ai_response})

    # Construct an OpenAI-compliant response
    openai_response = {
        "id": f"chatcmpl-{uuid.uuid4()}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [],
        "usage": {
            "prompt_tokens": response_json.get("usage", {}).get("prompt_tokens", 0),
            "completion_tokens": response_json.get("usage", {}).get("completion_tokens", 0),
            "total_tokens": response_json.get("usage", {}).get("total_tokens", 0)
        }
    }

    # Handle multiple completions if requested
    for i in range(n):
        choice = {
            "index": i,
            "message": {
                "role": "assistant",
                "content": ai_response
            },
            "finish_reason": "stop"
        }
        openai_response["choices"].append(choice)

    print(f"Returning OpenAI-formatted response: {json.dumps(openai_response, indent=2)}")
    return jsonify(openai_response), 200

# Memory cleanup route to manually clear conversations
@app.route('/v1/chat/memory/clear', methods=['POST'])
def clear_memory():
    """Clear conversation memory for a user"""
    data = request.get_json()
    user_id = data.get("user", os.getenv("USER"))
    
    if user_id in conversation_memories:
        # Delete the memory entirely rather than just clearing it
        del conversation_memories[user_id]
        return jsonify({"status": "success", "message": f"Memory cleared for user {user_id}"}), 200
    else:
        return jsonify({"status": "not_found", "message": f"No memory found for user {user_id}"}), 404

# Memory management route to list active conversations
@app.route('/v1/chat/memory/list', methods=['GET'])
def list_memories():
    """List all active conversation sessions"""
    return jsonify({
        "status": "success", 
        "active_sessions": list(conversation_memories.keys()),
        "count": len(conversation_memories)
    }), 200

# Memory inspection route for debugging
@app.route('/v1/chat/memory/debug', methods=['GET'])
def debug_memory():
    """Debug endpoint to inspect memory contents"""
    user_id = request.args.get("user", os.getenv("USER"))
    
    if user_id in conversation_memories:
        memory = conversation_memories[user_id]
        # Get message contents in a readable format
        messages = []
        for msg in memory.chat_memory.messages:
            if isinstance(msg, HumanMessage):
                messages.append({"role": "user", "content": msg.content})
            elif isinstance(msg, AIMessage):
                messages.append({"role": "assistant", "content": msg.content})
            elif isinstance(msg, SystemMessage):
                messages.append({"role": "system", "content": msg.content})
        
        return jsonify({
            "status": "success",
            "user": user_id,
            "messages": messages,
            "window_size": memory.k,
            "message_count": len(memory.chat_memory.messages)
        }), 200
    else:
        return jsonify({
            "status": "not_found",
            "message": f"No memory found for user {user_id}"
        }), 404

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001, debug=True)
