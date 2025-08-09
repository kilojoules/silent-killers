# src/silent_killers/llm_api.py
import os
import abc
from openai import OpenAI
from anthropic import Anthropic
import google.generativeai as genai

class LLMProvider(abc.ABC):
    """Abstract base class for all LLM API providers."""
    @abc.abstractmethod
    def get_completion(self, prompt: str, temperature: float = 0.7, max_tokens: int = 4096) -> str:
        """Takes a prompt and returns the model's text response."""
        pass

class OpenAIProvider(LLMProvider):
    def __init__(self, model_name: str):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name

    def get_completion(self, prompt: str, temperature: float = 0.7, max_tokens: int = 4096) -> str: # Add max_tokens
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens, # Pass it to the API call
        )
        return response.choices[0].message.content or ""

class AnthropicProvider(LLMProvider):
    def __init__(self, model_name: str):
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model_name = model_name

    def get_completion(self, prompt: str, temperature: float = 0.7, max_tokens: int = 4096) -> str: # Add max_tokens
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens, # Pass it to the API call
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.content[0].text or ""

class GoogleProvider(LLMProvider):
    def __init__(self, model_name: str):
        # Configure the API key when the provider is created
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model_name = model_name
        self.client = genai.GenerativeModel(self.model_name)

    def get_completion(self, prompt: str, temperature: float = 0.7, max_tokens: int = 4096) -> str:
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens 
        )
        response = self.client.generate_content(
            prompt,
            generation_config=generation_config
        )
        return response.text
