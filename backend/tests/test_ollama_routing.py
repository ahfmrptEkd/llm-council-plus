import os
import sys
import asyncio
import logging
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path

# Add project root to path
_project_root = Path(__file__).parent.parent.parent
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

from shared.llm.llm_manager import LLMManager

async def test_ollama_routing():
    print("Testing Ollama routing...")
    
    # Mock environment variable
    with patch.dict(os.environ, {"OLLAMA_HOST": "localhost:11434"}):
        manager = LLMManager()
        
        # Mock the AsyncOpenAI client
        mock_client = AsyncMock()
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Hello from Ollama"))]
        mock_response.usage = MagicMock(prompt_tokens=10, completion_tokens=20)
        mock_client.chat.completions.create.return_value = mock_response
        
        # Patch _get_ollama_client to return our mock
        with patch.object(manager, '_get_ollama_client', return_value=mock_client):
            result = await manager.invoke_model("ollama/llama3.1:latest", [{"role": "user", "content": "hi"}])
            
            print(f"Result: {result['response_text']}")
            assert result['response_text'] == "Hello from Ollama"
            
            # Check if client was called with correct parameters
            mock_client.chat.completions.create.assert_called_once()
            args, kwargs = mock_client.chat.completions.create.call_args
            assert kwargs['model'] == "llama3.1:latest"
            print("Ollama routing test passed!")

async def test_ollama_client_init():
    print("Testing Ollama client initialization...")
    
    with patch.dict(os.environ, {"OLLAMA_HOST": "some-remote-host:11434"}):
        manager = LLMManager()
        with patch('shared.llm.llm_manager.AsyncOpenAI') as MockAsyncOpenAI:
            client = manager._get_ollama_client()
            MockAsyncOpenAI.assert_called_once()
            args, kwargs = MockAsyncOpenAI.call_args
            assert kwargs['base_url'] == "http://some-remote-host:11434/v1"
            assert kwargs['api_key'] == "ollama"
            print("Ollama client initialization test passed!")

if __name__ == "__main__":
    asyncio.run(test_ollama_routing())
    asyncio.run(test_ollama_client_init())
