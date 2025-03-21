from typing import Any, List, Mapping, Optional, Dict
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langchain_core.language_models.llms import LLM
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    ChatMessage,
    HumanMessage,
    SystemMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
import requests
import json
import os
from pydantic import Field

from enum import Enum

class ModelType(Enum):
    GPT35 = 'gpt35'
    GPT4 = 'gpt4'
    O3_MINI = 'gpto3mini'
    O1_MINI = 'gpto1mini'
    
class ArgoLLM(LLM):

    model_type: ModelType = ModelType.GPT35
    url: str = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/"
    temperature: Optional[float] = 0.8
    system: Optional[str]
    top_p: Optional[float]= 0.7
    user: str = os.getenv("USER")
    
    @property
    def _llm_type(self) -> str:
        return "ArgoLLM"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:

        headers = {
            "Content-Type": "application/json"
        }
        params = {
            **self._get_model_default_parameters,
            **kwargs,
            "prompt": [prompt],
            "stop": []
        }

        params_json = json.dumps(params);
        print(params_json)
        response = requests.post(self.url, headers=headers, data=params_json)

        if response.status_code == 200:
            parsed = json.loads(response.text)
            return parsed['response']
        else:
            raise Exception(f"Request failed with status code: {response.status_code} {response.text}")

    @property
    def _get_model_default_parameters(self):
        return {
            "user": self.user,
            "model": self.model,
            "system": "" if self.system is None else self.system,
            "temperature": self.temperature,
            "top_p":  self.top_p
        }

    @property
    def model(self):
        return self.model_type.value
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}

    @property
    def _generations(self):
        return

class ArgoChatInterface:
    """
    A class for handling chat conversations with the Argo LLM service.
    This maintains conversation history and provides context for each new message.
    """
    
    def __init__(self, 
                model_type: ModelType = ModelType.GPT35, 
                url: str = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/",
                temperature: float = 0.8,
                system: Optional[str] = None,
                top_p: float = 0.7,
                user: str = os.getenv("USER")):
        self.model_type = model_type
        self.url = url
        self.temperature = temperature
        self.system = system
        self.top_p = top_p
        self.user = user
        self.conversation_history = []
        
    def add_user_message(self, message: str) -> None:
        """
        Add a user message to the conversation history.
        
        Args:
            message: The user's message
        """
        self.conversation_history.append({"role": "user", "content": message})
        
    def add_assistant_message(self, message: str) -> None:
        """
        Add an assistant message to the conversation history.
        
        Args:
            message: The assistant's message
        """
        self.conversation_history.append({"role": "assistant", "content": message})
    
    def format_conversation_for_prompt(self) -> str:
        """
        Format the conversation history into a prompt string.
        
        Returns:
            A formatted conversation history string
        """
        formatted_prompt = ""
        for message in self.conversation_history:
            role = message["role"]
            content = message["content"]
            formatted_prompt += f"{role}: {content}\n\n"
        return formatted_prompt
    
    def send_message(self, message: str) -> str:
        """
        Send a new message and get a response, maintaining conversation history.
        
        Args:
            message: The new message to send
            
        Returns:
            The model's response
        """
        # Add the user message to history
        self.add_user_message(message)
        
        # Format the full conversation history for the prompt
        conversation_prompt = self.format_conversation_for_prompt()
        
        # Add a prompt for the assistant to continue
        prompt = f"{conversation_prompt}assistant: "
        
        # Send the request to the API
        headers = {
            "Content-Type": "application/json"
        }
        
        params = {
            "user": self.user,
            "model": self.model_type.value,
            "system": "" if self.system is None else self.system,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "prompt": [prompt],
            "stop": []
        }
        
        params_json = json.dumps(params)
        response = requests.post(self.url, headers=headers, data=params_json)
        
        if response.status_code == 200:
            parsed = json.loads(response.text)
            response_text = parsed['response']
            
            # Add the assistant's response to the history
            self.add_assistant_message(response_text)
            
            return response_text
        else:
            raise Exception(f"Request failed with status code: {response.status_code} {response.text}")
    
    def get_conversation_history(self) -> List[Dict[str, str]]:
        """
        Get the current conversation history.
        
        Returns:
            The conversation history as a list of message dictionaries
        """
        return self.conversation_history
    
    def clear_conversation_history(self) -> None:
        """
        Clear the conversation history.
        """
        self.conversation_history = []

class ArgoChatModel(BaseChatModel):
    """
    A chat model implementation for the Argo LLM service that uses LangChain's chat model interface.
    This maintains conversation history and provides context for each new message.
    """
    
    model_type: ModelType = ModelType.GPT35
    url: str = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/chat/"
    temperature: Optional[float] = 0.8
    system: Optional[str] = None
    top_p: Optional[float] = 0.7
    user: str = os.getenv("USER")
    
    @property
    def _llm_type(self) -> str:
        return "ArgoChatModel"
    
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[Any] = None,
        **kwargs: Any,
    ) -> ChatResult:
        """
        Generate a chat response from the given messages.
        
        Args:
            messages: A list of BaseMessage objects representing the chat history
            stop: Optional list of stop sequences
            run_manager: Optional callback manager
            **kwargs: Additional keyword arguments
            
        Returns:
            A ChatResult containing the response message
        """
        # Format the messages into a prompt
        prompt = self._format_messages_to_prompt(messages)
        
        # Send the request to the API
        headers = {
            "Content-Type": "application/json"
        }
        
        params = {
            **self._get_model_default_parameters,
            **kwargs,
            "prompt": [prompt],
            "stop": stop or []
        }
        
        params_json = json.dumps(params)
        response = requests.post(self.url, headers=headers, data=params_json)
        
        if response.status_code == 200:
            parsed = json.loads(response.text)
            response_text = parsed['response']
            
            message = AIMessage(content=response_text)
            generation = ChatGeneration(message=message)
            
            return ChatResult(generations=[generation])
        else:
            raise Exception(f"Request failed with status code: {response.status_code} {response.text}")
    
    def _format_messages_to_prompt(self, messages: List[BaseMessage]) -> str:
        """
        Format a list of messages into a single prompt string.
        
        Args:
            messages: A list of BaseMessage objects representing the chat history
            
        Returns:
            A formatted prompt string
        """
        prompt = ""
        for message in messages:
            if isinstance(message, SystemMessage):
                prompt += f"system: {message.content}\n\n"
            elif isinstance(message, HumanMessage):
                prompt += f"user: {message.content}\n\n"
            elif isinstance(message, AIMessage):
                prompt += f"assistant: {message.content}\n\n"
            elif isinstance(message, ChatMessage):
                prompt += f"{message.role}: {message.content}\n\n"
        
        # Add the prompt for the assistant to respond
        prompt += "assistant: "
        
        return prompt
    
    @property
    def _get_model_default_parameters(self):
        return {
            "user": self.user,
            "model": self.model_type.value,
            "system": "" if self.system is None else self.system,
            "temperature": self.temperature,
            "top_p": self.top_p
        }
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "model_type": self.model_type.value,
            "temperature": self.temperature,
            "top_p": self.top_p,
        }

class ArgoEmbeddingWrapper():
    url: str = "https://apps-dev.inside.anl.gov/argoapi/api/v1/resource/embed/"
    user: str = os.getenv("USER")

    @property
    def _llm_type(self) -> str:
        return "ArgoLLM"

    def _call(
        self, 
        prompts: List[str], 
        run_manager: Optional[CallbackManagerForLLMRun] = None, 
        **kwargs: Any
    ) -> str:
        headers = { "Content-Type": "application/json" }
        params = { 
            "user": self.user, 
            "prompt": prompts
        }
        params_json = json.dumps(params)
        response = requests.post(self.url, headers=headers, data=params_json)
        if response.status_code == 200:
            parsed = json.loads(response.text)
            return parsed
        else:
            raise Exception(f"Request failed with status code: {response.status_code} {response.text}")

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {}

    @property
    def _generations(self):
        return

    def embed_documents(self, texts):
        return self.invoke(texts)

    def embed_query(self, query):
        return self.invoke(query)[0]
