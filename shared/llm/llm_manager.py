"""LLMManager using direct clients for Azure OpenAI, Anthropic, Grok, and Gemini.

Supports:
- Azure OpenAI (GPT, DeepSeek, Llama, Phi)
- Azure Anthropic (Claude)
- Grok (xAI)
- Google Gemini
"""

import os
import asyncio
import base64
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml
from dotenv import load_dotenv

# Client Libraries
from openai import AsyncAzureOpenAI, AsyncOpenAI
from anthropic import AsyncAnthropic
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# Use relative imports within shared module
from ..core.logger import setup_logger
from .cost_logger import CostLogger
from ..utils.preprocessor import TextPreprocessor

logger = setup_logger("llm_manager")

load_dotenv()


def _load_model_deployments() -> Dict[str, str]:
    """Load model deployment mappings from YAML config file."""
    config_path = Path(__file__).parent / "config" / "model_deployments.yaml"
    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        deployments = {}
        for category, models in config.items():
            if isinstance(models, dict):
                deployments.update(models)
        logger.info(f"Loaded {len(deployments)} model deployments from config")
        return deployments
    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}, using empty mappings")
        return {}
    except Exception as e:
        logger.error(f"Error loading model deployments: {e}")
        return {}


MODEL_DEPLOYMENTS = _load_model_deployments()


class LLMManager:
    """Manager for multiple LLM providers using direct SDK clients."""

    def __init__(self, cost_logger: Optional[CostLogger] = None):
        self.cost_logger = cost_logger or CostLogger()
        self.preprocessor = TextPreprocessor()

        # Initialize clients lazily or on init? 
        # Better to initialize on demand or verify env vars here.
        # We will initialize configured clients as needed to allow for dynamic env var changes if necessary,
        # but for performance, we could cache them.
        self._clients = {}

    def _get_deployment_name(self, model: str) -> str:
        return MODEL_DEPLOYMENTS.get(model, model)

    def _get_azure_openai_client(self) -> AsyncAzureOpenAI:
        if "azure_openai" not in self._clients:
            endpoint = os.getenv("AZURE_PROJECT_ENDPOINT")
            api_key = os.getenv("AZURE_API_KEY")
            if not endpoint or not api_key:
                raise ValueError("AZURE_PROJECT_ENDPOINT and AZURE_API_KEY must be set for Azure OpenAI.")
            self._clients["azure_openai"] = AsyncAzureOpenAI(
                azure_endpoint=endpoint,
                api_key=api_key,
                api_version="2024-12-01-preview"
            )
        return self._clients["azure_openai"]
    
    def _get_azure_phi_client(self) -> AsyncOpenAI:
        # Phi on Azure sometimes uses a different endpoint/format (OpenAI compatible)
        if "azure_phi" not in self._clients:
            endpoint = os.getenv("AZURE_PROJECT_EXTRA_ENDPOINT")
            api_key = os.getenv("AZURE_API_KEY")
            if not endpoint or not api_key:
                raise ValueError("AZURE_PROJECT_EXTRA_ENDPOINT and AZURE_API_KEY must be set for Phi models.")
            # Usually strict OpenAI client pointing to the endpoint
            self._clients["azure_phi"] = AsyncOpenAI(
                base_url=endpoint,
                api_key=api_key
            )
        return self._clients["azure_phi"]

    def _get_anthropic_client(self) -> AsyncAnthropic:
        if "anthropic" not in self._clients:
            endpoint = os.getenv("AZURE_PROJECT_ANTHROPIC_ENDPOINT")
            api_key = os.getenv("AZURE_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
            if not endpoint or not api_key:
                raise ValueError("AZURE_PROJECT_ANTHROPIC_ENDPOINT and API Key must be set for Claude.")
            
            # Azure Anthropic setup usually requires base_url and potentially custom headers
            # Assuming standard Anthropic client component compatible with the provided endpoint
            self._clients["anthropic"] = AsyncAnthropic(
                base_url=endpoint,
                api_key=api_key
            )
        return self._clients["anthropic"]

    def _get_grok_client(self) -> AsyncOpenAI:
        if "grok" not in self._clients:
            api_key = os.getenv("XAI_API_KEY")
            if not api_key:
                raise ValueError("XAI_API_KEY must be set for Grok.")
            self._clients["grok"] = AsyncOpenAI(
                api_key=api_key,
                base_url="https://api.x.ai/v1"
            )
        return self._clients["grok"]

    def _get_ollama_client(self) -> AsyncOpenAI:
        if "ollama" not in self._clients:
            # If backend runs in Docker and Ollama runs on your host, use: host.docker.internal:11434
            host = os.getenv("OLLAMA_HOST", "localhost:11434")
            if not host.startswith(("http://", "https://")):
                host = f"http://{host}"
            
            if not host.endswith("/v1"):
                host = f"{host}/v1"
                
            self._clients["ollama"] = AsyncOpenAI(
                base_url=host,
                api_key="ollama"  # Ollama doesn't require a real API key, but client needs one
            )
        return self._clients["ollama"]

    def _configure_gemini(self):
        if "gemini_configured" not in self._clients:
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("GEMINI_API_KEY must be set for Gemini.")
            genai.configure(api_key=api_key)
            self._clients["gemini_configured"] = True

    async def invoke_model(
        self, 
        model_alias: str, 
        messages: List[Dict[str, Any]], 
        temperature: Optional[float] = None, 
        metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Invoke LLM with specific provider logic."""
        model_lower = model_alias.lower()
        deployment_name = self._get_deployment_name(model_alias)
        
        try:
            # 1. Ollama (Local via OpenAI compatible API)
            if "ollama" in model_lower:
                return await self._invoke_ollama(deployment_name, messages, temperature, metadata, model_alias)

            # 2. Grok (xAI)
            elif "grok" in model_lower:
                return await self._invoke_grok(deployment_name, messages, temperature, metadata, model_alias)
            
            # 2. Gemini
            elif "gemini" in model_lower:
                return await self._invoke_gemini(deployment_name, messages, temperature, metadata, model_alias)
            
            # 3. Claude (Azure Anthropic)
            elif "claude" in model_lower:
                return await self._invoke_anthropic(deployment_name, messages, temperature, metadata, model_alias)
            
            # 4. Phi (Azure Extra Endpoint)
            elif "phi" in model_lower:
                return await self._invoke_phi(deployment_name, messages, temperature, metadata, model_alias)

            # 5. Default: Azure OpenAI (GPT, DeepSeek, Llama)
            else:
                return await self._invoke_azure_openai(deployment_name, messages, temperature, metadata, model_alias)

        except Exception as e:
            logger.error(f"Error invoking model {model_alias}: {e}")
            raise

    async def _invoke_azure_openai(self, deployment_name: str, messages: List[Dict], temperature: Optional[float], metadata: Optional[Dict], original_model: str):
        client = self._get_azure_openai_client()
        
        # Azure OpenAI Reasoning models (o1, o3, etc) often require temperature=1
        # GPT-5.1 in this context seems to be one of them based on user logs
        # We will enforce temperature=1 for known reasoning/preview models or if user hasn't specified one (default 0.7 -> 1 for these)
        
        # List of models that likely require temperature=1 (Reasoning models)
        reasoning_models = ["gpt-5.1", "o1", "o3", "deepseek-r1"] 
        
        # Check if original model or deployment name indicates a reasoning model
        is_reasoning = any(x in original_model.lower() for x in reasoning_models) or \
                       any(x in deployment_name.lower() for x in reasoning_models)

        if is_reasoning:
            temp = 1.0
        else:
            temp = temperature if temperature is not None else 0.7
        
        response = await client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            temperature=temp
        )
        
        # Handle reasoning models (like DeepSeek-R1) that may have different response structures
        message = response.choices[0].message
        content = message.content
        
        # If content is None, try to get text from other fields (for reasoning models)
        if content is None:
            # Some reasoning models may have content in different attributes
            if hasattr(message, 'text'):
                content = message.text
            elif hasattr(message, 'reasoning'):
                content = message.reasoning
            else:
                # Fallback: convert to string if it's not None but unexpected type
                content = str(content) if content is not None else ""
        
        # Ensure content is a string
        if content is None:
            content = ""
        elif not isinstance(content, str):
            content = str(content)
        
        usage = response.usage
        
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        
        self.cost_logger.log_request(original_model, input_tokens, output_tokens, metadata)
        
        return {
            "response_text": content,
            "tokens": {"input": input_tokens, "output": output_tokens, "total": input_tokens + output_tokens},
            "token_source": "api_response"
        }

        return {
            "response_text": content,
            "tokens": {"input": input_tokens, "output": output_tokens, "total": input_tokens + output_tokens},
            "token_source": "api_response"
        }

    async def _invoke_ollama(self, deployment_name: str, messages: List[Dict], temperature: Optional[float], metadata: Optional[Dict], original_model: str):
        client = self._get_ollama_client()
        temp = temperature if temperature is not None else 0.7
        
        # deployment_name should be the actual model name for Ollama (e.g., 'llama3.1:latest')
        response = await client.chat.completions.create(
            model=deployment_name,
            messages=messages,
            temperature=temp
        )
        
        content = response.choices[0].message.content
        usage = response.usage
        
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0
        
        self.cost_logger.log_request(original_model, input_tokens, output_tokens, metadata)
        
        return {
            "response_text": content,
            "tokens": {"input": input_tokens, "output": output_tokens, "total": input_tokens + output_tokens},
            "token_source": "api_response"
        }

    async def _invoke_grok(self, deployment_name: str, messages: List[Dict], temperature: Optional[float], metadata: Optional[Dict], original_model: str):
        client = self._get_grok_client()
        temp = temperature if temperature is not None else 0.7
        
        # Determine actual model name for Grok API
        # config mapping: grok-4 -> grok-4
        model_id = deployment_name
        if deployment_name.startswith("xai/"): # Old litellm artifact check
            model_id = deployment_name[4:]
            
        response = await client.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temp
        )
        
        content = response.choices[0].message.content
        usage = response.usage
        
        input_tokens = usage.prompt_tokens if usage else 0
        output_tokens = usage.completion_tokens if usage else 0

        self.cost_logger.log_request(original_model, input_tokens, output_tokens, metadata)

        return {
            "response_text": content,
            "tokens": {"input": input_tokens, "output": output_tokens, "total": input_tokens + output_tokens},
            "token_source": "api_response"
        }

    async def _invoke_anthropic(self, deployment_name: str, messages: List[Dict], temperature: Optional[float], metadata: Optional[Dict], original_model: str):
        client = self._get_anthropic_client()
        temp = temperature if temperature is not None else 0.7
        
        # Extract system message
        system_prompt = ""
        filtered_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system_prompt += msg["content"] + "\n"
            else:
                filtered_messages.append(msg)
        
        # Remove anthropic/ prefix if present from config mapping
        model_id = deployment_name
        if model_id.startswith("anthropic/"):
            model_id = model_id[10:]
            
        kwargs = {
            "model": model_id,
            "messages": filtered_messages,
            "max_tokens": 4096, # Required parameter for Anthropic
            "temperature": temp
        }
        if system_prompt:
            kwargs["system"] = system_prompt.strip()

        response = await client.messages.create(**kwargs)
        
        content = response.content[0].text
        usage = response.usage
        
        input_tokens = usage.input_tokens
        output_tokens = usage.output_tokens
        
        self.cost_logger.log_request(original_model, input_tokens, output_tokens, metadata)
        
        return {
            "response_text": content,
            "tokens": {"input": input_tokens, "output": output_tokens, "total": input_tokens + output_tokens},
            "token_source": "api_response"
        }

    async def _invoke_gemini(self, deployment_name: str, messages: List[Dict], temperature: Optional[float], metadata: Optional[Dict], original_model: str):
        self._configure_gemini()
        temp = temperature if temperature is not None else 0.7

        # Convert messages to Gemini format
        # [{'role': 'user', 'parts': ['text']}]
        gemini_history = []
        system_instruction = None
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            # Handle list of content (multimodal) or string
            parts = []
            if isinstance(content, list):
                for part in content:
                    if part.get("type") == "text":
                        parts.append(part["text"])
                    elif part.get("type") == "image_url":
                        # Simplification: In a real app we'd fetch the image data. 
                        # Assuming base64 data URI might be passed or URL.
                        # For now, just logging warning if not text.
                        # Ideally, we should handle this.
                         parts.append("[Image Content Not Fully Supported in Migration]")
            else:
                parts.append(str(content))
                
            if role == "system":
                # Accumulate system instruction
                if system_instruction is None:
                    system_instruction = parts[0]
                else:
                    system_instruction += "\n" + parts[0]
            elif role == "user":
                gemini_history.append({"role": "user", "parts": parts})
            elif role == "assistant":
                gemini_history.append({"role": "model", "parts": parts})

        # Remove gemini/ prefix
        model_id = deployment_name
        if model_id.startswith("gemini/"):
            model_id = model_id[7:]

        model = genai.GenerativeModel(
            model_name=model_id,
            system_instruction=system_instruction
        )
        
        generation_config = genai.types.GenerationConfig(
            temperature=temp
        )

        # Gemini chat session
        chat = model.start_chat(history=gemini_history[:-1] if gemini_history else [])
        
        last_msg = gemini_history[-1] if gemini_history else {"role": "user", "parts": [""]}
        
        response = await chat.send_message_async(
            last_msg["parts"],
            generation_config=generation_config
        )
        
        content = response.text
        usage = response.usage_metadata
        
        input_tokens = usage.prompt_token_count
        output_tokens = usage.candidates_token_count
        
        self.cost_logger.log_request(original_model, input_tokens, output_tokens, metadata)

        return {
            "response_text": content,
            "tokens": {"input": input_tokens, "output": output_tokens, "total": input_tokens + output_tokens},
            "token_source": "api_response"
        }

    def get_session_summary(self) -> Dict:
        """Get current session summary."""
        stats = self.cost_logger.get_session_stats()
        self.cost_logger.print_session_summary()
        return stats

    def save_session_log(self, filename: Optional[str] = None):
        """Save session log to file."""
        return self.cost_logger.save_session(filename)

