---
title: LLM Council Plus
emoji: 🏛️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: 5.9.1
app_file: app.py
pinned: false
---

# LLM Council Plus

![llmcouncil](header.jpg)

> **Inspired by [Andrej Karpathy's LLM Council](https://github.com/karpathy/llm-council)** - see his [original tweet](https://x.com/karpathy/status/1992381094667411768) about the concept.

The idea of this repo is that instead of asking a question to your favorite LLM provider (e.g. OpenAI GPT 5.1, Google Gemini 3.0 Pro, Anthropic Claude Sonnet 4.5, xAI Grok 4, etc.), you can group them into your "LLM Council Plus". This is a containerized web app with a Setup Wizard that guides you through configuration. It uses OpenRouter to send your query to multiple LLMs, asks them to review and rank each other's work, and finally a Chairman LLM produces the final response.

In a bit more detail, here is what happens when you submit a query:

1. **Stage 1: First opinions**. The user query is given to all LLMs individually, and the responses are collected. The individual responses are shown in a "tab view", so that the user can inspect them all one by one.
2. **Stage 2: Review**. Each individual LLM is given the responses of the other LLMs. Under the hood, the LLM identities are anonymized so that the LLM can't play favorites when judging their outputs. The LLM is asked to rank them in accuracy and insight.
3. **Stage 3: Final response**. The designated Chairman of the LLM Council takes all of the model's responses and compiles them into a single final answer that is presented to the user.

## What's Different from the Original

This fork extends [Andrej Karpathy's llm-council](https://github.com/karpathy/llm-council) with production-ready features:

| Feature                 | Original                | LLM Council Plus                               |
| ----------------------- | ----------------------- | ---------------------------------------------- |
| **Deployment**          | Manual Python/npm setup | Docker Compose (one command)                   |
| **Configuration**       | Edit config.py manually | Visual Setup Wizard                            |
| **Models**              | 4 hardcoded models      | Full OpenRouter catalog (100+ models)          |
| **Direct Connection**   | ❌                      | ✅ Azure, Google, xAI support                  |
| **Local Models**        | ❌                      | ✅ Ollama support                              |
| **Authentication**      | ❌                      | ✅ JWT multi-user system                       |
| **Storage**             | JSON files only         | JSON, PostgreSQL, MySQL                        |
| **Token Optimization**  | ❌                      | ✅ TOON format (20-60% savings)                |
| **Web Search**          | ❌                      | ✅ Tavily + Exa AI integration                 |
| **File Attachments**    | ❌                      | ✅ PDF, TXT, MD, images                        |
| **Tools**               | ❌                      | ✅ Calculator, Wikipedia, ArXiv, Yahoo Finance |
| **Real-time Streaming** | Basic                   | SSE with state persistence                     |
| **Error Handling**      | Silent failures         | Visual error indicators per model              |
| **Hot Reload**          | ❌                      | ✅ Config changes without restart              |

## Quick Start

**Easiest way — use the Setup Wizard:**

```bash
cp .env.example .env
docker compose up --build
# Open http://localhost
# The Setup Wizard will guide you through configuration
```

The Setup Wizard lets you:

- Choose LLM provider (OpenRouter or Direct/Ollama)
- Enter API keys
- Optionally enable authentication
- Optionally configure web search (Tavily or Exa)

**Alternative: Gradio & MCP Mode (Standalone)**

```bash
uv run python app.py
# Open http://localhost:7860
# This mode includes an MCP server for integration with other AI tools
```

**Alternative: Manual configuration**

```bash
cp .env.example .env
# Edit .env (see examples below)
docker compose up --build
# Open http://localhost
```

> **Note:** Authentication is disabled by default for easy setup. For production deployment with user authentication, see [SECURITY.md](SECURITY.md).

## Vibe Code Alert

This project was 99% vibe coded as a fun Saturday hack because I wanted to explore and evaluate a number of LLMs side by side in the process of [reading books together with LLMs](https://x.com/karpathy/status/1990577951671509438). It's nice and useful to see multiple responses side by side, and also the cross-opinions of all LLMs on each other's outputs. I'm not going to support it in any way, it's provided here as is for other people's inspiration and I don't intend to improve it. Code is ephemeral now and libraries are over, ask your LLM to change it in whatever way you like.

## Detailed Setup

### 1. Install Dependencies

The project uses [uv](https://docs.astral.sh/uv/) for project management.

**Backend:**

```bash
uv sync
```

**Frontend:**

```bash
cd frontend
npm install
cd ..
```

### 2. Configure API Key

Create a `.env` file in the project root:

```bash
OPENROUTER_API_KEY=sk-or-v1-...
```

Get your API key at [openrouter.ai](https://openrouter.ai/). Make sure to purchase the credits you need, or sign up for automatic top up.

Alternatively, you can use **Direct Connection** (Azure, Google, etc.) or Ollama for local models by setting `ROUTER_TYPE=direct` in your `.env` file.

Models are selected via the Setup Wizard or dynamically in the UI. Browse all available models at [openrouter.ai/models](https://openrouter.ai/models).

## Running the Application

### Docker (Recommended)

```bash
# Start services
docker compose up --build

# Access the application at http://localhost
```

Backend API is available at http://localhost:8001

### Development Mode (without Docker)

**Option A: Split Mode (FastAPI + React)**

```bash
# Backend:
uv run python -m backend.main

# Frontend:
cd frontend && npm run dev
```

**Option B: Unified Mode (Gradio + MCP)**

```bash
uv run python app.py
```

_Access at http://localhost:7860._

## Common Configuration

### OpenRouter (cloud)

```bash
ROUTER_TYPE=openrouter
OPENROUTER_API_KEY=sk-or-v1-...
```

### Ollama (local)

```bash
ROUTER_TYPE=direct
# Or legacy: ROUTER_TYPE=litellm
```

If you're running the backend in Docker and Ollama is running on your host machine, set:

```bash
OLLAMA_HOST=host.docker.internal:11434
```

## Web Search (Optional)

The UI can run web search and inject results into Stage 1 as context:

- **Heuristic tool usage**: search tools run only when the prompt has explicit search intent.
- **Web Search toggle**: forces a web search; the Chairman optimizes the query and runs Tavily/Exa.

Enable one of:

```bash
ENABLE_TAVILY=true
TAVILY_API_KEY=tvly-...
```

or:

```bash
ENABLE_EXA=true
EXA_API_KEY=...
```

## Storage

Default storage is local JSON files under `data/conversations/` (mounted into the backend container).

Database storage is supported via `DATABASE_TYPE=postgresql` or `DATABASE_TYPE=mysql` (see `backend/storage.py`).

## Tech Stack

- **Backend:** FastAPI (Python 3.10+), async httpx, OpenRouter API or Ollama
- **Frontend:** React + Vite, react-markdown for rendering
- **Storage:** JSON files in `data/conversations/` (optional DB storage)
- **Package Management:** uv for Python, npm for JavaScript

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for development setup and guidelines.

## Security

See [SECURITY.md](SECURITY.md) for security considerations and reporting vulnerabilities.

## License

MIT License - see [LICENSE](LICENSE) for details.
