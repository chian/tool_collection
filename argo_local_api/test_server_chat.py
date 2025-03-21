import pytest
import json
import uuid
import time
import os
from unittest.mock import patch, MagicMock
from flask import Flask
import requests

# Import the app from the server_chat module
from server_chat import app

@pytest.fixture
def client():
    """Create a test client for the Flask app."""
    app.config['TESTING'] = True
    with app.test_client() as client:
        yield client

def test_chat_completions_basic(client):
    """Test the basic functionality of the chat completions endpoint."""
    # Mock the uuid.uuid4 to return a fixed value for testing
    with patch('uuid.uuid4', return_value="test-uuid-123"):
        # Mock the time.time to return a fixed timestamp
        with patch('time.time', return_value=1234567890):
            # Mock the requests.post to avoid actual API calls
            with patch('requests.post') as mock_post:
                # Configure the mock response
                mock_response = MagicMock()
                mock_response.json.return_value = {"response": "This is a test response from the AI."}
                mock_post.return_value = mock_response

                # Define the test request data (OpenAI format)
                request_data = {
                    "model": "gpto1preview",
                    "messages": [
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": "Hello, who are you?"}
                    ],
                    "temperature": 0.5,
                    "max_tokens": 1000
                }

                # Make the request to the endpoint
                response = client.post(
                    '/v1/chat/completions',
                    data=json.dumps(request_data),
                    content_type='application/json'
                )

                # Assert the response status code is 200 (OK)
                assert response.status_code == 200

                # Parse the response data
                response_data = json.loads(response.data)

                # Assert the response structure matches the expected OpenAI format
                assert response_data["id"] == "chatcmpl-test-uuid-123"
                assert response_data["object"] == "chat.completion"
                assert response_data["created"] == 1234567890
                assert response_data["model"] == "gpto1preview"
                assert len(response_data["choices"]) == 1
                assert response_data["choices"][0]["message"]["role"] == "assistant"
                assert response_data["choices"][0]["message"]["content"] == "This is a test response from the AI."

                # Verify the mock was called with the correct arguments
                mock_post.assert_called_once()
                # Extract the kwargs used in the call
                call_kwargs = mock_post.call_args.kwargs
                # Validate the payload conversion from OpenAI format to internal format
                payload = json.loads(call_kwargs['data'])
                assert payload["user"] == os.getenv("USER") 
                assert payload["model"] == "gpto1preview"
                assert payload["system"] == "You are a helpful assistant."
                assert payload["prompt"] == ["Hello, who are you?"]
                assert payload["temperature"] == 0.5
                assert payload["max_tokens"] == 1000

def test_chat_completions_error_handling(client):
    """Test error handling in the chat completions endpoint."""
    # Test case for request exceptions
    with patch('requests.post') as mock_post:
        # Configure the mock to raise a requests.RequestException
        mock_post.side_effect = requests.exceptions.RequestException("Test exception")

        # Make the request to the endpoint
        response = client.post(
            '/v1/chat/completions',
            data=json.dumps({"messages": [{"role": "user", "content": "Hello"}]}),
            content_type='application/json'
        )

        # Assert the response status code is 500 (Internal Server Error)
        assert response.status_code == 500
        # Assert the error message is in the response
        assert b"Test exception" in response.data

def test_chat_completions_n_parameter(client):
    """Test the 'n' parameter for multiple completions."""
    with patch('requests.post') as mock_post:
        # Configure the mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Test response"}
        mock_post.return_value = mock_response

        # Define the test request data with n=3
        request_data = {
            "model": "gpto1preview",
            "messages": [{"role": "user", "content": "Generate three variations"}],
            "n": 3
        }

        # Make the request to the endpoint
        response = client.post(
            '/v1/chat/completions',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        # Assert the response status code is 200 (OK)
        assert response.status_code == 200

        # Parse the response data
        response_data = json.loads(response.data)

        # Assert there are 3 choices in the response
        assert len(response_data["choices"]) == 3
        # Verify each choice has the correct structure and content
        for i in range(3):
            assert response_data["choices"][i]["index"] == i
            assert response_data["choices"][i]["message"]["role"] == "assistant"
            assert response_data["choices"][i]["message"]["content"] == "Test response"

def test_chat_completions_parameter_defaults(client):
    """Test that default parameters are set correctly."""
    with patch('requests.post') as mock_post:
        # Configure the mock response
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "Test response"}
        mock_post.return_value = mock_response

        # Define the minimal test request data
        request_data = {
            "messages": [{"role": "user", "content": "Hello"}]
        }

        # Make the request to the endpoint
        response = client.post(
            '/v1/chat/completions',
            data=json.dumps(request_data),
            content_type='application/json'
        )

        # Parse the response data
        response_data = json.loads(response.data)

        # Assert the response contains default model
        assert response_data["model"] == "gpto1preview"

        # Verify the mock was called with the correct default arguments
        mock_post.assert_called_once()
        call_kwargs = mock_post.call_args.kwargs
        payload = json.loads(call_kwargs['data'])
        assert payload["temperature"] == 0.0
        assert payload["top_p"] == 1.0
        assert payload["max_tokens"] == 2056

def test_chat_completions_missing_ai_response(client):
    """Test handling of missing AI response."""
    with patch('requests.post') as mock_post:
        # Configure the mock response without the 'response' field
        mock_response = MagicMock()
        mock_response.json.return_value = {"something_else": "Not what we need"}
        mock_post.return_value = mock_response

        # Make the request to the endpoint
        response = client.post(
            '/v1/chat/completions',
            data=json.dumps({"messages": [{"role": "user", "content": "Hello"}]}),
            content_type='application/json'
        )

        # Assert the response status code is 500 (Internal Server Error)
        assert response.status_code == 500
        # Assert the error message is in the response
        assert b"No response from the AI" in response.data

def test_chat_completions_multi_turn_conversation(client):
    """Test multi-turn conversation with context preservation."""
    print("\n====== TESTING MULTI-TURN CONVERSATION ======\n")
    
    with patch('requests.post') as mock_post:
        # First turn: User introduces themselves
        mock_response_1 = MagicMock()
        mock_response_1.json.return_value = {"response": "Hello Alex! It's nice to meet a data scientist."}
        mock_post.return_value = mock_response_1
        
        # Initial message from user
        initial_request = {
            "model": "gpto1preview",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "My name is Alex and I'm a data scientist."}
            ],
            "user": os.getenv("USER")  # Use the current user's username from environment
        }
        
        print("\n----- TURN 1 -----")
        print("USER: My name is Alex and I'm a data scientist.")
        
        initial_response = client.post(
            '/v1/chat/completions',
            data=json.dumps(initial_request),
            content_type='application/json'
        )
        
        assert initial_response.status_code == 200
        initial_response_data = json.loads(initial_response.data)
        initial_ai_message = initial_response_data["choices"][0]["message"]["content"]
        
        print(f"ASSISTANT: {initial_ai_message}")
        
        # Show what was sent to the internal API
        call_kwargs = mock_post.call_args.kwargs
        payload = json.loads(call_kwargs['data'])
        print("\nInternal API payload:")
        print(f"- System: {payload['system']}")
        print(f"- User prompts: {payload['prompt']}")
        
        assert "Alex" in initial_ai_message
        assert "data scientist" in initial_ai_message
        
        # Reset mock to prepare for second call
        mock_post.reset_mock()
        
        # Second turn: User asks a question that relies on previous context
        mock_response_2 = MagicMock()
        mock_response_2.json.return_value = {"response": "You're a data scientist named Alex."}
        mock_post.return_value = mock_response_2
        
        # Follow-up message that includes previous conversation
        follow_up_request = {
            "model": "gpto1preview",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "My name is Alex and I'm a data scientist."},
                {"role": "assistant", "content": initial_ai_message},
                {"role": "user", "content": "What's my profession?"}
            ]
        }
        
        print("\n----- TURN 2 -----")
        print("USER: What's my profession?")
        
        follow_up_response = client.post(
            '/v1/chat/completions',
            data=json.dumps(follow_up_request),
            content_type='application/json'
        )
        
        assert follow_up_response.status_code == 200
        follow_up_response_data = json.loads(follow_up_response.data)
        follow_up_ai_message = follow_up_response_data["choices"][0]["message"]["content"]
        
        print(f"ASSISTANT: {follow_up_ai_message}")
        
        # Verify context preservation in second prompt sent to internal API
        call_kwargs = mock_post.call_args.kwargs
        payload = json.loads(call_kwargs['data'])
        
        print("\nInternal API payload:")
        print(f"- System: {payload['system']}")
        print(f"- User prompts: {payload['prompt']}")
        
        # Verify that multiple user messages from the conversation history were sent
        assert len(payload["prompt"]) == 2
        assert payload["prompt"][0] == "My name is Alex and I'm a data scientist."
        assert payload["prompt"][1] == "What's my profession?"
        
        # Verify AI response contains context from previous messages
        assert "data scientist" in follow_up_ai_message
        
        # Third turn: Further build on the conversation
        mock_post.reset_mock()
        mock_response_3 = MagicMock()
        mock_response_3.json.return_value = {"response": "I understand you're a data scientist named Alex working on a climate change ML project."}
        mock_post.return_value = mock_response_3
        
        # Add a third message to the conversation
        third_request = {
            "model": "gpto1preview",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "My name is Alex and I'm a data scientist."},
                {"role": "assistant", "content": initial_ai_message},
                {"role": "user", "content": "What's my profession?"},
                {"role": "assistant", "content": follow_up_ai_message},
                {"role": "user", "content": "I'm working on a machine learning project about climate change."}
            ]
        }
        
        print("\n----- TURN 3 -----")
        print("USER: I'm working on a machine learning project about climate change.")
        
        third_response = client.post(
            '/v1/chat/completions',
            data=json.dumps(third_request),
            content_type='application/json'
        )
        
        assert third_response.status_code == 200
        third_response_data = json.loads(third_response.data)
        third_ai_message = third_response_data["choices"][0]["message"]["content"]
        
        print(f"ASSISTANT: {third_ai_message}")
        
        # Verify the prompt sent to internal API for the third message
        call_kwargs = mock_post.call_args.kwargs
        payload = json.loads(call_kwargs['data'])
        
        print("\nInternal API payload:")
        print(f"- System: {payload['system']}")
        print(f"- User prompts: {payload['prompt']}")
        
        # Verify all user messages were included
        assert len(payload["prompt"]) == 3
        assert "climate change" in payload["prompt"][2]
        
        # Fourth turn: Test if the model remembers multiple turns
        mock_post.reset_mock()
        mock_response_4 = MagicMock()
        mock_response_4.json.return_value = {"response": "You are Alex, a data scientist working on a machine learning project related to climate change."}
        mock_post.return_value = mock_response_4
        
        # Final question that tests memory across all conversation turns
        fourth_request = {
            "model": "gpto1preview",
            "messages": [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "My name is Alex and I'm a data scientist."},
                {"role": "assistant", "content": initial_ai_message},
                {"role": "user", "content": "What's my profession?"},
                {"role": "assistant", "content": follow_up_ai_message},
                {"role": "user", "content": "I'm working on a machine learning project about climate change."},
                {"role": "assistant", "content": third_ai_message},
                {"role": "user", "content": "Based on our conversation, who am I and what am I working on?"}
            ]
        }
        
        print("\n----- TURN 4 -----")
        print("USER: Based on our conversation, who am I and what am I working on?")
        
        fourth_response = client.post(
            '/v1/chat/completions',
            data=json.dumps(fourth_request),
            content_type='application/json'
        )
        
        assert fourth_response.status_code == 200
        fourth_response_data = json.loads(fourth_response.data)
        fourth_ai_message = fourth_response_data["choices"][0]["message"]["content"]
        
        print(f"ASSISTANT: {fourth_ai_message}")
        
        # Verify the prompt sent to internal API for the fourth message
        call_kwargs = mock_post.call_args.kwargs
        payload = json.loads(call_kwargs['data'])
        
        print("\nInternal API payload:")
        print(f"- System: {payload['system']}")
        print(f"- User prompts: {payload['prompt']}")
        
        # Output complete conversation history
        print("\n====== FULL CONVERSATION HISTORY ======")
        print("1. USER: My name is Alex and I'm a data scientist.")
        print(f"2. ASSISTANT: {initial_ai_message}")
        print("3. USER: What's my profession?")
        print(f"4. ASSISTANT: {follow_up_ai_message}")
        print("5. USER: I'm working on a machine learning project about climate change.")
        print(f"6. ASSISTANT: {third_ai_message}")
        print("7. USER: Based on our conversation, who am I and what am I working on?")
        print(f"8. ASSISTANT: {fourth_ai_message}")
        print("\n=======================================")
        
        # Verify all user messages were included
        assert len(payload["prompt"]) == 4
        assert "Based on our conversation" in payload["prompt"][3]
        
        # Verify AI response shows memory of the entire conversation context
        assert "Alex" in fourth_ai_message
        assert "data scientist" in fourth_ai_message
        assert "climate change" in fourth_ai_message

if __name__ == "__main__":
    pytest.main() 