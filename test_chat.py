# test_argo_chat.py
from argo.ArgoLLM import ArgoChatInterface, ModelType

def test_chat_history():
    # Initialize the chat interface
    chat = ArgoChatInterface(model_type=ModelType.GPT4)
    
    # First message to establish context
    print("USER: My name is Alex and I'm a data scientist.")
    response1 = chat.send_message("My name is Alex and I'm a data scientist.")
    print(f"ASSISTANT: {response1}\n")
    
    # Second message that relies on previous context
    print("USER: What's my profession?")
    response2 = chat.send_message("What's my profession?")
    print(f"ASSISTANT: {response2}\n")
    
    # Third message to further build on context
    print("USER: I'm working on a machine learning project about climate change.")
    response3 = chat.send_message("I'm working on a machine learning project about climate change.")
    print(f"ASSISTANT: {response3}\n")
    
    # Fourth message that tests if the model remembers multiple turns of conversation
    print("USER: Based on our conversation, who am I and what am I working on?")
    response4 = chat.send_message("Based on our conversation, who am I and what am I working on?")
    print(f"ASSISTANT: {response4}\n")
    
    # Print out the conversation history for verification
    print("CONVERSATION HISTORY:")
    for i, message in enumerate(chat.get_conversation_history()):
        print(f"{i+1}. {message['role']}: {message['content']}")
    
    # Test clearing history
    chat.clear_conversation_history()
    print("\nAfter clearing history, length:", len(chat.get_conversation_history()))
    
    # Test that new conversation doesn't have previous context
    print("\nAfter clearing history:")
    print("USER: What was I working on again?")
    response5 = chat.send_message("What was I working on again?")
    print(f"ASSISTANT: {response5}")

if __name__ == "__main__":
    test_chat_history()