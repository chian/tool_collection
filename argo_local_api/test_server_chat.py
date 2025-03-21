import json
import os
import sys
import time
import requests
import subprocess
import signal
import atexit

# Define the server URL - this will be the local Flask server
SERVER_URL = "http://localhost:5001"

# Class that mimics the OpenAI client for chat completions
class ArgoOpenAIClient:
    """A client that mimics OpenAI's client interface for chat completions."""
    
    def __init__(self, base_url=None):
        """Initialize the client with a base URL."""
        self.base_url = base_url or SERVER_URL
    
    def chat_completion_create(self, model, messages, temperature=0, max_tokens=2056, n=1, user=None):
        """
        Create a chat completion using the OpenAI-compatible endpoint.
        
        Args:
            model: The model to use
            messages: A list of message dictionaries with 'role' and 'content' keys
            temperature: The sampling temperature
            max_tokens: The maximum number of tokens to generate
            n: The number of completions to generate
            user: A user identifier
            
        Returns:
            A dictionary containing the response data
        """
        url = f"{self.base_url}/v1/chat/completions"
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "n": n
        }
        
        if user:
            payload["user"] = user
        
        headers = {
            "Content-Type": "application/json"
        }
        
        print(f"Sending request to {url} with payload: {json.dumps(payload, indent=2)}")
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Request failed with status code: {response.status_code} {response.text}")
    
    def clear_memory(self, user=None):
        """
        Clear the conversation memory for a user.
        
        Args:
            user: The user identifier whose memory to clear
            
        Returns:
            The response data
        """
        url = f"{self.base_url}/v1/chat/memory/clear"
        
        payload = {}
        if user:
            payload["user"] = user
        
        headers = {
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Request failed with status code: {response.status_code} {response.text}")
    
    def list_memories(self):
        """
        List all active conversation sessions.
        
        Returns:
            The response data with active sessions
        """
        url = f"{self.base_url}/v1/chat/memory/list"
        
        response = requests.get(url)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Request failed with status code: {response.status_code} {response.text}")

# Server process management
server_process = None

def start_server():
    """Start the Flask server in a subprocess."""
    global server_process
    print("Starting server_chat.py Flask server...")
    server_process = subprocess.Popen(
        ["python", "argo_local_api/server_chat.py"], 
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Give the server time to start up
    time.sleep(3)
    print("Server started.")

def stop_server():
    """Stop the Flask server subprocess."""
    global server_process
    if server_process:
        print("Stopping server...")
        server_process.send_signal(signal.SIGTERM)
        server_process.wait()
        print("Server stopped.")

# Register the cleanup function
atexit.register(stop_server)

def test_chat_completions_basic():
    """Test the basic functionality of the chat completions endpoint with real API calls."""
    print("\nRunning test: Basic chat completions")
    
    client = ArgoOpenAIClient()
    
    # Use the client to make a request
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello, who are you?"}
    ]
    
    response = client.chat_completion_create(
        model="gpto1preview",
        messages=messages,
        temperature=0.5,
        max_tokens=1000
    )
    
    # Validate the response
    assert "id" in response, "Response missing 'id' field"
    assert response["object"] == "chat.completion", f"Expected object 'chat.completion', got {response.get('object')}"
    assert "created" in response, "Response missing 'created' field"
    assert response["model"] == "gpto1preview", f"Expected model 'gpto1preview', got {response.get('model')}"
    assert len(response["choices"]) == 1, f"Expected 1 choice, got {len(response.get('choices', []))}"
    assert response["choices"][0]["message"]["role"] == "assistant", f"Expected role 'assistant', got {response['choices'][0]['message'].get('role')}"
    assert isinstance(response["choices"][0]["message"]["content"], str), "Response content is not a string"
    assert len(response["choices"][0]["message"]["content"]) > 0, "Response content is empty"
    
    print("✓ Test passed: Basic chat completions")
    return True

def test_server_memory():
    """Test the server's memory capabilities with stateful conversations."""
    print("\nRunning test: Server-side memory")
    
    client = ArgoOpenAIClient()
    
    # Create a unique session identifier
    session_id = f"memory_test_{time.time()}"
    
    # First, clear any existing memory for this session
    try:
        client.clear_memory(user=session_id)
        print(f"Cleared memory for session {session_id}")
    except Exception as e:
        # It's okay if there's no memory to clear yet
        print(f"Note: {str(e)}")
    
    # Step 1: Send initial message
    print("\n----- INITIAL MESSAGE -----")
    initial_message = "My name is Taylor and I work as a software engineer."
    
    response1 = client.chat_completion_create(
        model="gpto1preview",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": initial_message}
        ],
        user=session_id
    )
    
    print(f"USER: {initial_message}")
    first_response = response1["choices"][0]["message"]["content"]
    print(f"ASSISTANT: {first_response}")
    
    # Step 2: Send a follow-up message WITHOUT including conversation history
    # The server should remember the context
    print("\n----- FOLLOW-UP (SERVER MEMORY) -----")
    follow_up = "What is my profession?"
    
    response2 = client.chat_completion_create(
        model="gpto1preview",
        messages=[
            {"role": "user", "content": follow_up}
        ],
        user=session_id
    )
    
    print(f"USER: {follow_up}")
    second_response = response2["choices"][0]["message"]["content"]
    print(f"ASSISTANT: {second_response}")
    
    # Verify the response contains context from the first message
    assert "software engineer" in second_response.lower() or "engineer" in second_response.lower(), "Server memory failed to maintain context"
    
    # Step 3: Check active sessions and verify our session exists
    sessions = client.list_memories()
    assert "active_sessions" in sessions, "Missing active_sessions in response"
    assert session_id in sessions["active_sessions"], f"Session {session_id} not found in active sessions"
    print(f"Session found in active sessions: {session_id}")
    
    # Step 4: Clear the memory
    clear_result = client.clear_memory(user=session_id)
    assert clear_result["status"] == "success", "Failed to clear memory"
    print(f"Successfully cleared memory for session {session_id}")
    
    # Step 5: Verify session was removed
    sessions_after = client.list_memories()
    assert session_id not in sessions_after["active_sessions"], f"Session {session_id} still exists after clearing"
    print("Session successfully removed from active sessions")
    
    print("✓ Test passed: Server-side memory")
    return True

def run_all_tests():
    """Run all tests and report results."""
    tests = [
        test_chat_completions_basic,
        test_server_memory  # New test for server memory capabilities
    ]
    
    total_tests = len(tests)
    passed_tests = 0
    failed_tests = []
    
    print(f"\n=== Running {total_tests} tests ===\n")
    
    for test_func in tests:
        try:
            if test_func():
                passed_tests += 1
        except AssertionError as e:
            failed_tests.append((test_func.__name__, str(e)))
            print(f"✗ Test failed: {test_func.__name__} - {e}")
        except Exception as e:
            failed_tests.append((test_func.__name__, f"Unexpected error: {str(e)}"))
            print(f"✗ Test failed: {test_func.__name__} - Unexpected error: {e}")
    
    # Print test summary
    print(f"\n=== Test Summary ===")
    print(f"Total tests: {total_tests}")
    print(f"Passed: {passed_tests}")
    print(f"Failed: {len(failed_tests)}")
    
    if failed_tests:
        print("\nFailed tests:")
        for name, error in failed_tests:
            print(f"- {name}: {error}")
    
    return passed_tests == total_tests

if __name__ == "__main__":
    # Start the server process
    start_server()
    
    try:
        # Run the tests
        success = run_all_tests()
    finally:
        # Ensure the server is stopped
        stop_server()
    
    sys.exit(0 if success else 1) 