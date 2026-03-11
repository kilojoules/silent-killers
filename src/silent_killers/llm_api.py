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


class HuggingFaceProvider(LLMProvider):
    """Provider for local HuggingFace models (Llama, etc.).

    Loads a model once and reuses it across calls.
    Supports 4-bit quantization to fit larger models on smaller GPUs.
    """

    _instances: dict[str, "HuggingFaceProvider"] = {}

    def __init__(self, model_name: str, *, load_in_4bit: bool = False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

        # Reuse already-loaded model if same name requested
        if model_name in HuggingFaceProvider._instances:
            other = HuggingFaceProvider._instances[model_name]
            self.model = other.model
            self.tokenizer = other.tokenizer
            self.model_name = model_name
            self._device = other._device
            return

        self.model_name = model_name
        token = os.getenv("HF_TOKEN", "")

        if load_in_4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
            )
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=bnb_config,
                device_map="auto",
                token=token,
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map="auto",
                token=token,
            )

        self.model.eval()
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self._device = self.model.device
        HuggingFaceProvider._instances[model_name] = self

    def get_completion(self, prompt: str, temperature: float = 0.7, max_tokens: int = 4096) -> str:
        import torch

        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True,
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                top_p=0.9 if temperature > 0 else 1.0,
                pad_token_id=self.tokenizer.pad_token_id,
            )

        new_tokens = outputs[0, inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
