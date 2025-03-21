# test_argo_chat.py
from argo.ArgoLLM import ArgoChatInterface, ModelType

def test_chat_history():
    # Initialize the chat interface
    chat = ArgoChatInterface(model_type=ModelType.O1_MINI)
    
    # First message to establish context
    print("USER: What do you know about the trpB?")
    response1 = chat.send_message("What do you know about the trpB?.")
    print(f"ASSISTANT: {response1}\n")
    
    # Second message that relies on previous context
    print("USER: Based on the information you provided, how can we design better synthetic mutants of this proteins?")
    response2 = chat.send_message("Based on the information you provided, how can we design better synthetic mutants of this protein?")
    print(f"ASSISTANT: {response2}\n")
    
    
    # Print out the conversation history for verification
    print("CONVERSATION HISTORY:")
    for i, message in enumerate(chat.get_conversation_history()):
        print(f"{i+1}. {message['role']}: {message['content']}")
    
    # Test clearing history
    chat.clear_conversation_history()
    print("\nAfter clearing history, length:", len(chat.get_conversation_history()))
    
    # Test that new conversation doesn't have previous context
    print("\nAfter clearing history:")
    print("USER: Which protein did I ask you about??")
    response5 = chat.send_message("Which protein did I ask you about?")
    print(f"ASSISTANT: {response5}")
    
    

if __name__ == "__main__":
    test_chat_history()
