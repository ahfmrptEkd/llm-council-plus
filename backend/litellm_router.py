"""LiteLLM router for multi-provider LLM support via shared/llm module.

Supports Azure OpenAI, Azure Anthropic, Google Gemini, and Grok (xAI)
through a unified interface compatible with openrouter.py.
"""

import logging
import sys
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional, Union, AsyncGenerator

# Add project root to path for shared module imports
_project_root = Path(__file__).parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

# Import shared module as a package
from shared.llm.llm_manager import LLMManager
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

from .config import DEFAULT_TIMEOUT

logger = logging.getLogger(__name__)

# Retry configuration for rate-limited requests (429 errors)
MAX_RETRIES = 2
INITIAL_BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 30.0

# Global LLMManager instance (singleton pattern)
_llm_manager: Optional[LLMManager] = None


def _get_llm_manager() -> LLMManager:
    """Get or create LLMManager instance."""
    global _llm_manager
    if _llm_manager is None:
        _llm_manager = LLMManager()
    return _llm_manager


def build_message_content(
    text: str,
    images: Optional[List[Dict[str, str]]] = None
) -> Union[str, List[Dict[str, Any]]]:
    """
    Build message content for multimodal LLM API.

    For text-only messages, returns a string.
    For multimodal messages (with images), returns an array of content parts.

    Args:
        text: The text content of the message
        images: Optional list of image dicts with 'content' (base64 data URI) and 'filename'

    Returns:
        Either a string (text only) or a list of content parts (multimodal)
    """
    if not images:
        return text

    # Build multimodal content array (OpenAI-compatible format)
    content = [
        {"type": "text", "text": text}
    ]

    for image in images:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": image["content"]  # base64 data URI
            }
        })

    return content


def _convert_messages_to_langchain(messages: List[Dict[str, Any]]) -> List:
    """Convert API message format to LangChain message format.

    Args:
        messages: List of dicts with 'role' and 'content' keys

    Returns:
        List of LangChain message objects
    """
    langchain_messages = []
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")

        if role == "system":
            langchain_messages.append(SystemMessage(content=content))
        elif role == "assistant":
            langchain_messages.append(AIMessage(content=content))
        else:  # user or default
            # Handle multimodal content (list of content parts)
            if isinstance(content, list):
                # LangChain HumanMessage supports list of content parts
                # Format: [{"type": "text", "text": "..."}, {"type": "image_url", "image_url": {"url": "..."}}]
                langchain_messages.append(HumanMessage(content=content))
            else:
                # Simple text content
                langchain_messages.append(HumanMessage(content=content))

    return langchain_messages


async def query_model(
    model: str,
    messages: List[Dict[str, Any]],
    timeout: float = None,
    stage: str = None,
    retry_on_rate_limit: bool = True
) -> Optional[Dict[str, Any]]:
    """
    Query a single model via LiteLLM with retry on rate limits.

    Args:
        model: Model alias (e.g., "gpt-5-mini", "claude-sonnet-4.5", "gemini-2.5-pro")
        messages: List of message dicts with 'role' and 'content'.
                  Content can be a string (text only) or an array of content parts
                  for multimodal messages (see build_message_content).
        timeout: Request timeout in seconds (defaults to DEFAULT_TIMEOUT from config)
        stage: Optional stage identifier for debugging (e.g., "STAGE1", "STAGE2", "STAGE3")
        retry_on_rate_limit: If True, retry on rate limit errors with exponential backoff

    Returns:
        Response dict with 'content' and optional 'reasoning_details', or error dict if failed
    """
    if timeout is None:
        timeout = DEFAULT_TIMEOUT

    if stage:
        logger.debug("[%s] Querying model: %s", stage, model)

    manager = _get_llm_manager()

    # Convert messages to LangChain format
    try:
        langchain_messages = _convert_messages_to_langchain(messages)
    except Exception as e:
        logger.error("Error converting messages: %s", e)
        return {
            'error': True,
            'error_type': 'invalid_input',
            'error_message': f'Invalid message format: {str(e)}'
        }

    # Get LLM instance
    try:
        llm = manager.get_llm(model)
    except ValueError as e:
        logger.error("Configuration error for model %s: %s", model, e)
        return {
            'error': True,
            'error_type': 'config',
            'error_message': str(e)
        }
    except Exception as e:
        logger.error("Error getting LLM instance for model %s: %s", model, e)
        return {
            'error': True,
            'error_type': 'unknown',
            'error_message': f'Failed to initialize model: {str(e)}'
        }

    # Retry loop for rate limits
    retries = 0
    backoff = INITIAL_BACKOFF_SECONDS

    while True:
        try:
            # Invoke with tracking
            result = await manager.invoke_with_tracking(
                llm=llm,
                messages=langchain_messages,
                model_name=model,
                metadata={"stage": stage} if stage else None
            )

            response_text = result.get("response_text", "")
            if not response_text or not str(response_text).strip():
                logger.error("Empty response from LLM!")
                return {
                    'error': True,
                    'error_type': 'empty_response',
                    'error_message': 'LLM returned empty response'
                }

            return {
                'content': response_text,
                'reasoning_details': None  # LiteLLM doesn't provide reasoning details
            }

        except Exception as e:
            error_msg = str(e)
            error_type = 'unknown'

            # Check for rate limit errors
            if 'rate limit' in error_msg.lower() or '429' in error_msg:
                error_type = 'rate_limit'
                if retry_on_rate_limit and retries < MAX_RETRIES:
                    retries += 1
                    logger.warning("[%s] Rate limit for model %s. Retry %d/%d in %.1fs...",
                                 stage or "API", model, retries, MAX_RETRIES, backoff)
                    await asyncio.sleep(backoff)
                    backoff = min(backoff * 2, MAX_BACKOFF_SECONDS)
                    continue

            # Check for timeout errors
            if 'timeout' in error_msg.lower() or 'timed out' in error_msg.lower():
                error_type = 'timeout'
                logger.error("Timeout error querying model %s: %s", model, error_msg)
                return {
                    'error': True,
                    'error_type': error_type,
                    'error_message': f'Request timed out: {error_msg}'
                }

            # Check for authentication errors
            if 'api key' in error_msg.lower() or 'unauthorized' in error_msg.lower() or '401' in error_msg:
                error_type = 'auth'
                logger.error("Authentication error for model %s: %s", model, error_msg)
                return {
                    'error': True,
                    'error_type': error_type,
                    'error_message': 'Invalid API key or authentication failed'
                }

            # Check for model not found errors
            if 'not found' in error_msg.lower() or '404' in error_msg:
                error_type = 'not_found'
                logger.error("Model not found: %s", model)
                return {
                    'error': True,
                    'error_type': error_type,
                    'error_message': f'Model not available: {model}'
                }

            # Generic error
            logger.error("Error querying model %s: %s: %s", model, type(e).__name__, error_msg)
            return {
                'error': True,
                'error_type': error_type,
                'error_message': error_msg
            }


async def query_models_parallel(
    models: List[str],
    messages: List[Dict[str, Any]],
    stage: str = None
) -> Dict[str, Optional[Dict[str, Any]]]:
    """
    Query multiple models in parallel.

    Args:
        models: List of model aliases (e.g., ["gpt-5-mini", "claude-sonnet-4.5"])
        messages: List of message dicts to send to each model
        stage: Optional stage identifier for debugging (e.g., "STAGE1", "STAGE2")

    Returns:
        Dict mapping model identifier to response dict (or error dict if failed)
    """
    if stage:
        logger.debug("[%s] Querying %d models in parallel...", stage, len(models))

    # Create tasks for all models
    tasks = [query_model(model, messages, stage=stage) for model in models]

    # Wait for all to complete
    responses = await asyncio.gather(*tasks)

    # Map models to their responses
    return {model: response for model, response in zip(models, responses)}


async def query_models_streaming(
    models: List[str],
    messages: List[Dict[str, Any]]
) -> AsyncGenerator[tuple[str, Optional[Dict[str, Any]]], None]:
    """
    Query multiple models in parallel and yield results as they complete.

    Args:
        models: List of model aliases
        messages: List of message dicts to send to each model

    Yields:
        Tuple of (model, response) as each model completes
    """
    import time

    start_time = time.time()
    logger.debug("[PARALLEL] Starting %d model queries at t=0.0s", len(models))

    # Create named tasks so we can identify which model completed
    async def query_with_name(model: str):
        req_start = time.time() - start_time
        logger.debug("[PARALLEL] Starting request to %s at t=%.2fs", model, req_start)
        response = await query_model(model, messages)
        req_end = time.time() - start_time
        logger.debug("[PARALLEL] Got response from %s at t=%.2fs", model, req_end)
        return (model, response)

    # Create ALL tasks at once - they start executing immediately in parallel
    tasks = [asyncio.create_task(query_with_name(model)) for model in models]
    logger.debug("[PARALLEL] All %d tasks created and running in parallel", len(tasks))

    # Yield results as they complete (first finished = first yielded)
    for coro in asyncio.as_completed(tasks):
        model, response = await coro
        yield_time = time.time() - start_time
        logger.debug("[PARALLEL] Yielding %s at t=%.2fs", model, yield_time)
        yield (model, response)

