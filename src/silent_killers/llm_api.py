# src/silent_killers/llm_api.py
import os
import abc

class LLMProvider(abc.ABC):
    """Abstract base class for all LLM API providers."""
    @abc.abstractmethod
    def get_completion(self, prompt: str, temperature: float = 0.7, max_tokens: int = 4096) -> str:
        """Takes a prompt and returns the model's text response."""
        pass

class OpenAIProvider(LLMProvider):
    """Provider for OpenAI models (GPT-4o, etc.).

    For reasoning models like o3-mini that reject the temperature param,
    set ``supports_temperature=False`` when constructing.
    """
    def __init__(self, model_name: str, *, supports_temperature: bool = True):
        from openai import OpenAI
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model_name = model_name
        self.supports_temperature = supports_temperature

    def get_completion(self, prompt: str, temperature: float = 0.7, max_tokens: int = 4096) -> str:
        kwargs = dict(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
        )
        if self.supports_temperature:
            kwargs["temperature"] = temperature
            kwargs["max_tokens"] = max_tokens
        else:
            # Reasoning models (o3-mini, etc.) use max_completion_tokens
            kwargs["max_completion_tokens"] = max_tokens
        response = self.client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

class AnthropicProvider(LLMProvider):
    def __init__(self, model_name: str):
        from anthropic import Anthropic
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        self.model_name = model_name

    def get_completion(self, prompt: str, temperature: float = 0.7, max_tokens: int = 4096) -> str:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
        )
        return response.content[0].text or ""

class GoogleProvider(LLMProvider):
    def __init__(self, model_name: str):
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model_name = model_name
        self.client = genai.GenerativeModel(self.model_name)
        self._genai = genai

    def get_completion(self, prompt: str, temperature: float = 0.7, max_tokens: int = 4096) -> str:
        generation_config = self._genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens
        )
        response = self.client.generate_content(
            prompt,
            generation_config=generation_config
        )
        return response.text

class DeepSeekProvider(LLMProvider):
    """Provider for DeepSeek models via their OpenAI-compatible API."""
    def __init__(self, model_name: str):
        from openai import OpenAI
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        self.model_name = model_name

    def get_completion(self, prompt: str, temperature: float = 0.7, max_tokens: int = 4096) -> str:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""
