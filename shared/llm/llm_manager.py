"""LLMManager using LiteLLM via LangChain for unified LLM provider support.

Supports Azure OpenAI (GPT, DeepSeek, Llama, Phi), Azure Anthropic (Claude),
Google Gemini, and Grok (xAI) through a single interface.
"""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import litellm
import yaml
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_litellm import ChatLiteLLM

# Use relative imports within shared module
from ..core.logger import setup_logger
from .cost_logger import CostLogger
from ..utils.preprocessor import TextPreprocessor

litellm.suppress_debug_info = True


logger = setup_logger("llm_manager")

load_dotenv()

LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY")
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST")


def _load_model_deployments() -> Dict[str, str]:
    """Load model deployment mappings from YAML config file.

    Flattens nested structure into single-level dict.

    Returns:
        Dict mapping model names to deployment/API names
    """
    config_path = Path(__file__).parent / "config" / "model_deployments.yaml"

    try:
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Flatten nested structure
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
    """Manager for multiple LLM providers using LiteLLM with cost tracking."""

    def __init__(self, cost_logger: Optional[CostLogger] = None):
        """Initialize LLM Manager.

        Args:
            cost_logger: Optional CostLogger instance for tracking costs
        """
        self.cost_logger = cost_logger or CostLogger()
        self.preprocessor = TextPreprocessor()
        self.llm_cache = {}

        if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
            try:
                if not LANGFUSE_HOST:
                    os.environ["LANGFUSE_HOST"] = "http://localhost:3000"

                if "langfuse" not in litellm.success_callback:
                    litellm.success_callback.append("langfuse")
                if "langfuse" not in litellm.failure_callback:
                    litellm.failure_callback.append("langfuse")

                logger.info(f"✅ Langfuse monitoring enabled (Host: {os.getenv('LANGFUSE_HOST', 'http://localhost:3000')})")
                logger.info(f"   Callbacks registered: {litellm.success_callback}")
            except Exception as e:
                logger.error(f"❌ Failed to initialize Langfuse: {e}")
        else:
            logger.debug("⚠️ Langfuse credentials not found. Monitoring disabled.")

    def _get_deployment_name(self, model: str) -> str:
        """Get deployment/API model name for a given model alias."""
        return MODEL_DEPLOYMENTS.get(model, model)

    def _resolve_model_config(self, model_alias: str, deployment_name: str) -> Dict[str, Any]:
        """Resolve provider-specific configuration for LiteLLM.

        Args:
            model_alias: Original model name (e.g., 'gpt-5-mini', 'claude-sonnet')
            deployment_name: Resolved deployment name from config

        Returns:
            Dict containing 'model' (for LiteLLM) and provider kwargs (api_base, api_key, etc.)
        """
        model_lower = model_alias.lower()
        config = {}

        # Azure OpenAI (GPT, DeepSeek, Llama)
        # Note: Phi is also Azure OpenAI compatible but on a different endpoint in this setup
        if "gpt" in model_lower or "deepseek" in model_lower or "llama" in model_lower:
            endpoint = os.getenv("AZURE_PROJECT_ENDPOINT")
            api_key = os.getenv("AZURE_API_KEY")
            if not endpoint or not api_key:
                raise ValueError("AZURE_PROJECT_ENDPOINT and AZURE_API_KEY must be set.")

            config.update(
                {
                    "model": f"azure/{deployment_name}",
                    "api_base": endpoint,
                    "api_key": api_key,
                    "api_version": "2024-12-01-preview",
                }
            )

        elif "phi" in model_lower:
            endpoint = os.getenv("AZURE_PROJECT_EXTRA_ENDPOINT")
            api_key = os.getenv("AZURE_API_KEY")
            if not endpoint or not api_key:
                raise ValueError("AZURE_PROJECT_EXTRA_ENDPOINT and AZURE_API_KEY must be set.")

            config.update({"model": f"openai/{deployment_name}", "api_base": endpoint, "api_key": api_key})

        elif "claude" in model_lower:
            endpoint = os.getenv("AZURE_PROJECT_ANTHROPIC_ENDPOINT")
            api_key = os.getenv("AZURE_API_KEY")
            if not endpoint or not api_key:
                raise ValueError("AZURE_PROJECT_ANTHROPIC_ENDPOINT and AZURE_API_KEY must be set.")

            # If using Azure Anthropic/Foundry, it typically emulates Anthropic API or OpenAI.
            # Assuming standard Anthropic API exposed on Azure here (based on previous ChatAnthropic usage)
            # LiteLLM needs 'anthropic/' prefix usually, but with custom API base.
            config.update({"model": f"anthropic/{deployment_name}", "api_base": endpoint, "api_key": api_key})

        elif "grok" in model_lower:
            api_key = os.getenv("GROK_API_KEY")
            if not api_key:
                raise ValueError("GROK_API_KEY must be set.")

            # Grok is OpenAI compatible
            config.update({"model": f"openai/{deployment_name}", "api_base": "https://api.x.ai/v1", "api_key": api_key})

        elif "gemini" in model_lower:
            api_key = os.getenv("GEMINI_AI_API_KEY")
            if api_key:
                config.update({"model": f"gemini/{deployment_name}", "api_key": api_key})
            else:
                project = os.getenv("VERTEX_PROJECT_ID")
                location = os.getenv("GOOGLE_CLOUD_LOCATION", "us-central1")
                if not project:
                    raise ValueError("VERTEX_PROJECT_ID or GEMINI_AI_API_KEY must be set.")

                config.update({"model": f"vertex_ai/{deployment_name}", "vertex_project": project, "vertex_location": location})

        else:
            endpoint = os.getenv("AZURE_PROJECT_ENDPOINT")
            api_key = os.getenv("AZURE_API_KEY")
            config.update(
                {
                    "model": f"azure/{deployment_name}",
                    "api_base": endpoint,
                    "api_key": api_key,
                    "api_version": "2024-12-01-preview",
                }
            )

        return config

    def get_llm(self, model: str, temperature: Optional[float] = None, **kwargs) -> ChatLiteLLM:
        """Get ChatLiteLLM instance for the specified model.

        Args:
            model: Model name/alias (e.g., "gpt-5-mini", "claude-sonnet")
            temperature: Sampling temperature
            **kwargs: Additional arguments for ChatLiteLLM
        """
        deployment_name = self._get_deployment_name(model)

        # Determine strict temperature caching strategy
        # Some models don't support temperature or treat None differently
        temp_str = str(temperature) if temperature is not None else "default"
        cache_key = f"litellm_{deployment_name}_{temp_str}"

        if cache_key in self.llm_cache:
            return self.llm_cache[cache_key]

        config = self._resolve_model_config(model, deployment_name)

        llm_kwargs = {
            "model": config.pop("model"),
            **config,  # Inject api_base, api_key, etc.
            **kwargs,
        }

        if temperature is not None:
            llm_kwargs["temperature"] = temperature

        # Langfuse metadata delivery (Information included)
        llm_kwargs["model_kwargs"] = {
            "metadata": {
                "model_alias": model,
                "deployment": deployment_name,
                "generation_name": model,
                "trace_name": model,
            }
        }

        logger.info(f"Creating LiteLLM instance: {model} -> {llm_kwargs['model']}")

        llm = ChatLiteLLM(**llm_kwargs)
        self.llm_cache[cache_key] = llm
        return llm

    def _extract_token_usage(self, response: Any, messages: Optional[List] = None) -> Dict[str, int]:
        """Extract token usage from LLM response.

        LiteLLM standardizes this in response.response_metadata['token_usage'].
        """
        input_tokens = 0
        output_tokens = 0

        # Check LiteLLM/LangChain standard metadata
        if hasattr(response, "response_metadata"):
            usage = response.response_metadata.get("token_usage", {})
            if usage:
                input_tokens = usage.get("prompt_tokens", 0)
                output_tokens = usage.get("completion_tokens", 0)

        if input_tokens == 0 and output_tokens == 0:
            if hasattr(response, "usage_metadata") and response.usage_metadata:
                meta = response.usage_metadata
                if isinstance(meta, dict):
                    input_tokens = meta.get("input_tokens", 0)
                    output_tokens = meta.get("output_tokens", 0)

        # Fallback estimation if still 0 (and checking if it wasn't just a 0-token response logic error)
        # However, purely 0 usually implies missing data for most models.
        if input_tokens == 0 and output_tokens == 0:
            # Basic estimation logic from before
            prompt_text = ""
            if messages:
                prompt_text = " ".join([m.content if hasattr(m, "content") else str(m) for m in messages])

            response_text = response.content if hasattr(response, "content") else str(response)

            input_tokens = int(len(prompt_text.split()) * 1.33) if prompt_text else 0
            output_tokens = int(len(response_text.split()) * 1.33) if response_text else 0

        return {"input": input_tokens, "output": output_tokens, "total": input_tokens + output_tokens}

    async def invoke_with_tracking(
        self, llm: ChatLiteLLM, messages: Union[List, str], model_name: str, metadata: Optional[Dict] = None
    ) -> Dict[str, Any]:
        """Async invoke LLM with cost tracking using LiteLLM + LangChain.

        Note: LiteLLM automatically sends traces to Langfuse with name 'litellm-acompletion'.
        The actual model name is available in the metadata.
        """
        if isinstance(messages, str):
            messages = [HumanMessage(content=messages)]
        elif isinstance(messages, list) and all(isinstance(m, str) for m in messages):
            messages = [HumanMessage(content=m) for m in messages]

        try:
            logger.info(f"Invoking LLM: {model_name}")

            current_model_kwargs = llm.model_kwargs.copy() if llm.model_kwargs else {}
            current_metadata = current_model_kwargs.get("metadata", {}).copy()

            # Merge metadata (User ID, Session ID, Trace ID injection)
            if metadata:
                # Langfuse v2 field mapping
                if "trace_id" in metadata:
                    current_metadata["trace_id"] = metadata["trace_id"]
                if "user_id" in metadata and "trace_user_id" not in metadata:
                    current_metadata["trace_user_id"] = metadata["user_id"]
                if "session_id" in metadata:
                    current_metadata["session_id"] = metadata["session_id"]
                if "tags" in metadata:
                    current_metadata["tags"] = metadata["tags"]

            current_model_kwargs["metadata"] = current_metadata

            # Use model_copy instead of deprecated copy() method
            llm_for_request = llm.model_copy(update={"model_kwargs": current_model_kwargs})

            response = await llm_for_request.ainvoke(messages)

            response_text = response.content if hasattr(response, "content") else str(response)

            if not response_text or not str(response_text).strip():
                logger.error("Empty response from LLM!")
                raise ValueError("LLM returned empty response")

            # Extract usage
            token_usage = self._extract_token_usage(response, messages)

            cost_info = self.cost_logger.log_request(
                model=model_name,
                input_tokens=token_usage["input"],
                output_tokens=token_usage["output"],
                metadata=metadata,
            )

            token_source = "api_response" if hasattr(response, "response_metadata") and response.response_metadata.get("token_usage") else "estimated"

            return {
                "response": response,
                "response_text": response_text,
                "cost_info": cost_info,
                "tokens": token_usage,
                "token_source": token_source,
            }

        except Exception as e:
            logger.error(f"Error invoking LLM {model_name}: {e}")
            raise

    def get_session_summary(self) -> Dict:
        """Get current session summary."""
        stats = self.cost_logger.get_session_stats()
        self.cost_logger.print_session_summary()
        return stats

    def save_session_log(self, filename: Optional[str] = None):
        """Save session log to file."""
        return self.cost_logger.save_session(filename)
