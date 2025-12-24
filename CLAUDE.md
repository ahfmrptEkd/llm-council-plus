# CLAUDE.md - Technical Notes for LLM Council Plus

This file contains technical details, architectural decisions, and important implementation notes for future development sessions.

## Project Overview

LLM Council Plus is a 3-stage deliberation system where multiple LLMs collaboratively answer user questions. The key innovation is anonymized peer review in Stage 2, preventing models from playing favorites.

**Application Name:** LLM Council Plus (formerly "Hea LLM")

## Architecture

### Deployment

The application runs entirely in Docker containers:
- **Frontend:** Nginx serving React app on port 80 (HTTP by default)
- **Backend:** FastAPI Python app on port 8001

```bash
# Start the application
APP_VERSION="1.2.12" docker compose up -d

# Access at http://localhost
```

### LLM Router Architecture

The system supports **three router types** via `ROUTER_TYPE` environment variable:

1. **OpenRouter** (default): Cloud-based access to 200+ models
2. **LiteLLM**: Multi-provider unified interface (Azure, GCP, xAI, Ollama)
3. **Ollama** (deprecated): Use `ROUTER_TYPE=litellm` with `USE_OLLAMA_MODELS=true` instead

**LiteLLM Benefits:**
- Enterprise multi-cloud support (Azure OpenAI, Azure Anthropic, Google Gemini, xAI Grok)
- Local model support via Ollama integration
- Unified cost tracking via Langfuse monitoring
- Model alias system for deployment abstraction
- Rate limit handling with exponential backoff

### Backend Structure (`backend/`)

**`config.py`**
- Contains `COUNCIL_MODELS` (list of model identifiers, format varies by router type)
- Contains `CHAIRMAN_MODEL` (model that synthesizes final answer)
- `ROUTER_TYPE` selector: "openrouter", "litellm" (supports Ollama via `USE_OLLAMA_MODELS=true`)
- `reload_config()` function for hot reload without container restart
- Default models per router:
  - OpenRouter: `openai/gpt-5.1`, `google/gemini-3-pro-preview`, `anthropic/claude-sonnet-4.5`, `x-ai/grok-4`
  - LiteLLM (cloud): `gpt-5.1`, `gemini-2.5-pro`, `claude-sonnet-4.5`, `grok-4`
  - LiteLLM (Ollama): `ollama/deepseek-r1:latest`, `ollama/llama3.1:latest`, `ollama/qwen3:latest`, `ollama/gemma3:latest`

**`auth.py`**
- JWT-based authentication system
- `reload_auth()` function for hot reload of auth configuration
- Users configured via `AUTH_USERS_JSON` environment variable
- 60-day token expiry with auto-logout

**`openrouter.py`**
- OpenRouter-specific implementation (used when `ROUTER_TYPE=openrouter`)
- `query_model()`: Single async model query
- `query_models_parallel()`: Parallel queries using `asyncio.gather()`
- `query_models_streaming()`: SSE streaming for real-time responses
- Uses dynamic config access (`config.OPENROUTER_API_KEY`) for hot reload support
- Graceful degradation: returns None on failure, continues with successful responses

**`litellm_router.py`** (New)
- LiteLLM-specific implementation (used when `ROUTER_TYPE=litellm`)
- Compatible interface with `openrouter.py` for seamless switching
- `query_model()`: Single async query with retry logic and rate limit handling
- `query_models_parallel()`: Parallel queries across multiple providers
- `query_models_streaming()`: SSE streaming with provider-agnostic interface
- `build_message_content()`: Multimodal support (text + images)
- Provider-specific configuration via `_resolve_model_config()`:
  - Azure OpenAI: GPT, DeepSeek, Llama (via `AZURE_PROJECT_ENDPOINT`)
  - Azure Anthropic: Claude models (via `AZURE_PROJECT_ANTHROPIC_ENDPOINT`)
  - Azure Extra: Phi models (via `AZURE_PROJECT_EXTRA_ENDPOINT`)
  - Google Gemini: Via Google AI Studio API or Vertex AI
  - xAI Grok: Direct API integration
  - Ollama: Local models via LiteLLM adapter
- Error handling: Categorizes errors (rate_limit, timeout, auth, not_found)
- Retry strategy: Max 2 retries with exponential backoff (2s → 30s cap)

**`council.py`** - The Core Logic
- Router-agnostic implementation (works with OpenRouter, LiteLLM, Ollama)
- Dynamically imports correct router module based on `ROUTER_TYPE`
- `stage1_collect_responses()`: Parallel queries to all council models
- `stage2_collect_rankings()`:
  - Anonymizes responses as "Response A, B, C, etc."
  - Creates `label_to_model` mapping for de-anonymization
  - Prompts models to evaluate and rank (with strict format requirements)
  - Returns tuple: (rankings_list, label_to_model_dict)
  - Each ranking includes both raw text and `parsed_ranking` list
- `stage3_synthesize_final()`: Chairman synthesizes from all responses + rankings
- `parse_ranking_from_text()`: Extracts "FINAL RANKING:" section
- `calculate_aggregate_rankings()`: Computes average rank position

**`storage.py`**
- JSON-based conversation storage in `data/conversations/`
- Each conversation: `{id, created_at, messages[], username}`
- Assistant messages contain: `{role, stage1, stage2, stage3}`
- Thread-safe with file locking

**`main.py`**
- FastAPI app with CORS enabled
- SSE streaming endpoints for real-time responses
- Setup wizard endpoint (`POST /api/setup/config`)
- Hot reload triggers after setup config save

### Shared Module Structure (`shared/llm/`)

The shared LLM module provides unified multi-provider support for LiteLLM router:

**`llm_manager.py`** - Core LLM Manager
- `LLMManager` class: Singleton pattern for managing LLM instances
- `_load_model_deployments()`: Loads model aliases from YAML config
- `get_llm()`: Factory method returning `ChatLiteLLM` instances with provider-specific config
- `_resolve_model_config()`: Provider detection and configuration resolution
- `invoke_with_tracking()`: Async invocation with cost tracking and Langfuse integration
- `_extract_token_usage()`: Token usage extraction from LLM responses
- Langfuse integration: Auto-configured if `LANGFUSE_PUBLIC_KEY` and `LANGFUSE_SECRET_KEY` set
- LLM instance caching for performance

**`cost_logger.py`**
- Tracks token usage and costs per model
- Uses pricing from `config/model_pricing.yaml`
- Provides session summaries and detailed cost breakdowns

**`config/model_deployments.yaml`**
- Maps user-friendly model aliases to provider-specific deployment names
- Example: `gpt-5-mini` → `gpt-5-mini-prod` (Azure deployment name)
- Categories: GPT, Claude, DeepSeek, Llama, Phi, Gemini, Grok, Ollama, Embeddings
- Supports 20+ model aliases across 5+ providers

**`config/model_pricing.yaml`**
- Defines input/output token costs per 1K tokens (USD)
- Used by `CostLogger` for accurate cost tracking
- Ollama models: Free (0.0000 cost)
- Cloud models: Provider-specific pricing

### Frontend Structure (`frontend/src/`)

**`App.jsx`**
- Main orchestration: manages conversations list and current conversation
- Shows SetupWizard if not configured
- Shows LoginScreen if auth enabled and not authenticated
- Handles message sending with SSE streaming

**`components/SetupWizard.jsx`**
- First-time configuration UI
- Configures: LLM provider (OpenRouter/Ollama), API keys, Tavily, Authentication
- Saves to `.env` file and triggers hot reload

**`components/LoginScreen.jsx`**
- JWT-based authentication
- User selection or manual username input
- SVG logo (council nodes design)

**`components/Sidebar.jsx`**
- Conversation list with user filter
- 3-dot menu for edit/delete
- Inline title editing
- SVG logo with "LLM Council Plus" text

**`components/ChatInterface.jsx`**
- Multiline textarea (3 rows, resizable)
- Enter to send, Shift+Enter for new line
- File upload support (PDF, TXT, MD, images)
- Web search toggle (requires Tavily)
- Google Drive upload button
- Real-time streaming display

**`components/Stage1.jsx`, `Stage2.jsx`, `Stage3.jsx`**
- Tab views for each stage of council deliberation
- Real-time updates during streaming
- ReactMarkdown rendering

**Styling**
- Dark theme (CSS variables in `index.css`)
- Primary color: #4a90e2 (blue)
- SVG logos for sidebar and login

## Key Features

### Hot Reload
Configuration changes via Setup Wizard are applied without container restart:
- `reload_config()` in config.py reloads all environment variables
- `reload_auth()` in auth.py reloads JWT and user settings
- Dynamic config access in openrouter.py (`config.OPENROUTER_API_KEY`)

### Authentication (Optional)
- JWT tokens with 60-day expiry
- Users defined via `AUTH_USERS_JSON` env var
- Enable/disable via `AUTH_ENABLED` flag
- Auto-logout on token expiry

### File Upload
- Supports: PDF, TXT, MD, JPG, PNG, GIF, WebP
- Size limits: 10MB files, 5MB images
- Content extracted and sent to council

### Web Search (Tavily)
- Optional Tavily API integration
- Toggle per-message in chat interface
- Search results fed to Stage 1 queries

### Google Drive Integration
- Upload conversation exports to Drive
- Requires `credentials/google-credentials.json`
- Configure `GOOGLE_DRIVE_FOLDER_ID`

### Export
- Export to Markdown format
- Download or upload to Google Drive

## Key Design Decisions

### Stage 2 Prompt Format
The Stage 2 prompt is very specific to ensure parseable output:
```
1. Evaluate each response individually first
2. Provide "FINAL RANKING:" header
3. Numbered list format: "1. Response C", "2. Response A", etc.
4. No additional text after ranking section
```

### De-anonymization Strategy
- Models receive: "Response A", "Response B", etc.
- Backend creates mapping: `{"Response A": "openai/gpt-5.1", ...}`
- Frontend displays model names in **bold** for readability
- This prevents bias while maintaining transparency

### SSE Streaming
- Real-time response streaming via Server-Sent Events
- Chunk buffering handles split JSON messages
- Graceful fallback on connection errors

## Port Configuration
- **Frontend:** 80 (HTTP, redirects to HTTPS), 443 (HTTPS)
- **Backend:** 8001

## Docker Volumes
```yaml
volumes:
  - ./data/conversations:/app/data/conversations  # Conversation storage
  - ./credentials:/app/credentials:ro              # Google Drive creds
  - ./.env:/app/.env                               # Hot reload support
```

## Environment Variables

### Router Configuration

**Router Type Selection:**
- `ROUTER_TYPE` - Router type: "openrouter" (default), "litellm"
  - For local models: Use `ROUTER_TYPE=litellm` with `USE_OLLAMA_MODELS=true`

**OpenRouter (when `ROUTER_TYPE=openrouter`):**
- `OPENROUTER_API_KEY` - **Required** for OpenRouter
- `OPENROUTER_API_URL` - API endpoint (default: `https://openrouter.ai/api/v1/chat/completions`)

**LiteLLM Multi-Provider (when `ROUTER_TYPE=litellm`):**

*Azure Configuration:*
- `AZURE_PROJECT_ENDPOINT` - Azure OpenAI endpoint (for GPT, DeepSeek, Llama)
- `AZURE_PROJECT_ANTHROPIC_ENDPOINT` - Azure Anthropic endpoint (for Claude)
- `AZURE_PROJECT_EXTRA_ENDPOINT` - Azure extra endpoint (for Phi models)
- `AZURE_API_KEY` - Shared API key for all Azure services

*Google Gemini (choose one):*
- `GEMINI_AI_API_KEY` - Google AI Studio direct API key
- OR use Vertex AI:
  - `VERTEX_PROJECT_ID` - GCP project ID
  - `GOOGLE_CLOUD_LOCATION` - GCP region (default: `us-central1`)

*xAI Grok:*
- `GROK_API_KEY` - Grok API key

*Ollama (local models):*
- `USE_OLLAMA_MODELS` - Set to "true" to use Ollama models with LiteLLM
- `OLLAMA_HOST` - Ollama server address (default: `localhost:11434`)
  - For Docker: Use `host.docker.internal:11434` if Ollama runs on host

### Model Configuration

- `COUNCIL_MODELS` - Comma-separated model list (format depends on router type)
  - OpenRouter: `openai/gpt-5.1,anthropic/claude-sonnet-4.5`
  - LiteLLM cloud: `gpt-5-mini,gemini-2.5-pro,claude-sonnet-4.5`
  - LiteLLM Ollama: `ollama/deepseek-r1:latest,ollama/llama3.1:latest`
- `CHAIRMAN_MODEL` - Model for final synthesis
- `MAX_COUNCIL_MODELS` - Maximum council models allowed (default: 5)

### Monitoring & Observability

- `LANGFUSE_PUBLIC_KEY` - Langfuse public key for LLM monitoring
- `LANGFUSE_SECRET_KEY` - Langfuse secret key
- `LANGFUSE_HOST` - Langfuse host URL (default: `http://localhost:3000`)

### Authentication (Optional)

- `AUTH_ENABLED` - "true" to enable authentication (default: false)
- `JWT_SECRET` - Secret for JWT tokens (required when auth enabled)
- `AUTH_USERS_JSON` - JSON object of username:password pairs

### Features

- `TAVILY_API_KEY` - For web search feature (Tavily)
- `EXA_API_KEY` - For web search feature (Exa, alternative to Tavily)
- `ENABLE_TAVILY` - Enable Tavily integration (default: false)
- `ENABLE_EXA` - Enable Exa integration (default: false)
- `ENABLE_MEMORY` - Enable memory system (default: true)
- `ENABLE_OPENAI_EMBEDDINGS` - Use OpenAI embeddings instead of local (default: false)
- `OPENAI_API_KEY` - OpenAI API key (for embeddings if enabled)
- `ENABLE_LANGGRAPH` - Enable LangGraph workflows (default: false)

### Storage

- `DATABASE_TYPE` - Storage backend: "json" (default), "postgresql", "mysql"
- `POSTGRESQL_URL` - PostgreSQL connection URL
- `MYSQL_URL` - MySQL connection URL
- `DATA_DIR` - Data directory for JSON storage (default: `data/conversations`)

### Integration

- `GOOGLE_DRIVE_FOLDER_ID` - Google Drive folder ID for exports
- `GOOGLE_SERVICE_ACCOUNT_FILE` - Path to service account JSON

### Performance

- `DEFAULT_TIMEOUT` - Default API timeout in seconds (default: 120.0)
- `TITLE_GENERATION_TIMEOUT` - Title generation timeout (default: 180.0)

## Common Gotchas

1. **Hot Reload**: After using Setup Wizard, config is reloaded automatically
2. **CORS Issues**: Frontend must match allowed origins in CORS middleware
3. **Docker Restart**: Use `docker compose up -d`, not `docker restart`
4. **HTTPS**: If you enable HTTPS in nginx, ensure certs are mounted correctly
5. **LiteLLM Model Names**: Use aliases from `model_deployments.yaml`, not raw provider names
   - Correct: `gpt-5-mini`, `claude-sonnet-4.5`, `gemini-2.5-pro`
   - Incorrect: `gpt-5-mini-prod`, `claude-sonnet-4-5-prod` (these are deployment names)
6. **Ollama Integration**: When using Ollama with LiteLLM:
   - Set `ROUTER_TYPE=litellm` (not "ollama")
   - Set `USE_OLLAMA_MODELS=true`
   - Model names must include prefix: `ollama/deepseek-r1:latest`
7. **Langfuse Monitoring**: Auto-enabled if both keys are set, no manual configuration needed
8. **Azure Endpoints**: Multiple Azure endpoints supported for different model families
   - Main: GPT, DeepSeek, Llama
   - Anthropic: Claude models
   - Extra: Phi models

## Data Flow Summary

```
User Query
    ↓
Stage 1: Parallel queries → [individual responses] (SSE streaming)
    ↓
Stage 2: Anonymize → Parallel ranking queries → [evaluations + parsed rankings]
    ↓
Aggregate Rankings Calculation → [sorted by avg position]
    ↓
Stage 3: Chairman synthesis with full context
    ↓
Return: {stage1, stage2, stage3, metadata}
    ↓
Frontend: Display with tabs + validation UI
```

The entire flow is async/parallel where possible to minimize latency.
