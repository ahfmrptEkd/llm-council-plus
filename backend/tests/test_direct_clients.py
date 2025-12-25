import unittest
from unittest.mock import MagicMock, patch, AsyncMock
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from shared.llm.llm_manager import LLMManager

class TestDirectClients(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self):
        # Set dummy env vars to pass validation
        os.environ["AZURE_PROJECT_ENDPOINT"] = "https://example.openai.azure.com"
        os.environ["AZURE_PROJECT_EXTRA_ENDPOINT"] = "https://example.phi.azure.com"
        os.environ["AZURE_PROJECT_ANTHROPIC_ENDPOINT"] = "https://example.anthropic.azure.com"
        os.environ["AZURE_API_KEY"] = "dummy-key"
        os.environ["XAI_API_KEY"] = "dummy-key"
        os.environ["GEMINI_API_KEY"] = "dummy-key"
        os.environ["ANTHROPIC_API_KEY"] = "dummy-key"

        self.manager = LLMManager()

    @patch("shared.llm.llm_manager.AsyncAzureOpenAI")
    async def test_azure_openai(self, mock_client_cls):
        # Setup mock
        mock_client = AsyncMock()
        mock_client_cls.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Azure Response"))]
        mock_response.usage.prompt_tokens = 10
        mock_response.usage.completion_tokens = 20
        
        mock_client.chat.completions.create.return_value = mock_response

        # Invoke
        result = await self.manager.invoke_model(
            "gpt-5", 
            [{"role": "user", "content": "Hello"}],
            metadata={}
        )

        # Verify
        self.assertEqual(result["response_text"], "Azure Response")
        self.assertEqual(result["tokens"]["input"], 10)
        self.assertEqual(result["token_source"], "api_response")
        mock_client.chat.completions.create.assert_called_once()

    @patch("shared.llm.llm_manager.AsyncOpenAI")
    async def test_grok(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client_cls.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.choices = [MagicMock(message=MagicMock(content="Grok Response"))]
        mock_response.usage.prompt_tokens = 5
        mock_response.usage.completion_tokens = 5
        
        mock_client.chat.completions.create.return_value = mock_response

        result = await self.manager.invoke_model(
            "grok-4", 
            [{"role": "user", "content": "Hello"}],
            metadata={}
        )

        self.assertEqual(result["response_text"], "Grok Response")
        # Check if base_url was set correctly for Grok
        args, kwargs = mock_client_cls.call_args
        self.assertEqual(kwargs.get("base_url"), "https://api.x.ai/v1")

    @patch("shared.llm.llm_manager.AsyncAnthropic")
    async def test_anthropic(self, mock_client_cls):
        mock_client = AsyncMock()
        mock_client_cls.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.content = [MagicMock(text="Claude Response")]
        mock_response.usage.input_tokens = 15
        mock_response.usage.output_tokens = 25
        
        mock_client.messages.create.return_value = mock_response

        result = await self.manager.invoke_model(
            "claude-sonnet-4.5", 
            [{"role": "user", "content": "Hello"}],
            metadata={}
        )

        self.assertEqual(result["response_text"], "Claude Response")
        mock_client.messages.create.assert_called_once()

    @patch("shared.llm.llm_manager.genai")
    async def test_gemini(self, mock_genai):
        mock_model = MagicMock()
        mock_genai.GenerativeModel.return_value = mock_model
        
        mock_chat = MagicMock()
        mock_model.start_chat.return_value = mock_chat
        
        mock_response = MagicMock()
        mock_response.text = "Gemini Response"
        mock_response.usage_metadata.prompt_token_count = 30
        mock_response.usage_metadata.candidates_token_count = 40
        
        mock_chat.send_message_async = AsyncMock(return_value=mock_response)

        result = await self.manager.invoke_model(
            "gemini-2.5-pro", 
            [{"role": "user", "content": "Hello"}],
            metadata={}
        )

        self.assertEqual(result["response_text"], "Gemini Response")
        mock_genai.configure.assert_called_once()
        mock_chat.send_message_async.assert_called_once()

if __name__ == "__main__":
    unittest.main()
