<div align="center">

#  Multi-Agent Planner

[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/downloads/)
[![Groq](https://img.shields.io/badge/Groq-Llama%203.3%2070B-orange.svg)](https://console.groq.com/)
[![Flask](https://img.shields.io/badge/Flask-Web%20UI-green.svg)](http://localhost:8080)
[![MIT License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**AI-powered multi-agent system that turns natural language into working Python code**

[Quick Start](#-quick-start) • [Features](#-features) • [Web UI](#-web-interface) • [Architecture](#-architecture) • [API](#-api-usage)

</div>

---

##  Overview

Multi-Agent Planner is an intelligent code generation system that orchestrates **8 specialized AI agents** to transform your ideas into production-ready Python code. Simply describe what you want to build, and the system will:

1. **Plan** the project structure
2. **Design** the architecture
3. **Generate** the code
4. **Test** for correctness
5. **Review** and improve
6. **Document** everything

>  **Free to use** - Powered by Groq's free Llama 3.3 70B API with automatic fallback

---

##  Features

| Feature | Description |
|---------|-------------|
|  **Free by Default** | Uses Groq's free Llama 3.3 70B API with auto-fallback to backup models |
|  **DAG Pipeline (opt-in)** | Replace the linear pipeline with a dependency graph that develops independent modules in parallel and supports mid-run replanning (`USE_DAG_PIPELINE=1`) |
|  **Human-in-the-Loop** | Optional approve / edit / regenerate gates after the Planner and Architect stages (`CHECKPOINT_MODE=cli`) |
|  **Live Streaming UI** | Server-Sent Events stream pipeline progress to the browser — every stage transition, task pass/fail, and cost update appears in a live timeline |
|  **Project + User Memory** | Cross-run learning: Architect consults remembered user preferences (libs, style, blacklists); ProjectMemory records decisions per prompt for future runs (`MEMORY_BACKEND=chroma` for embeddings, JSON default) |
|  **Web Research (opt-in)** | ResearcherAgent runs web searches between Planner and Architect to surface current docs and best practices (`TAVILY_API_KEY` or `BRAVE_API_KEY`; no-op without a key) |
|  **Cloud Sandbox (opt-in)** | Real isolation via E2B — only mode that can validate code doing real I/O (`E2B_API_KEY` + `pip install e2b-code-interpreter`) |
|  **Best-of-N Generation** | `BEST_OF_N=3` generates N candidates per task in parallel, scores with the critic, keeps best — trades cost for quality |
|  **Architecture-First** | Architect agent creates high-level design before any code is written |
|  **Smart Retry** | Failed tasks retry up to 3x with critic feedback for self-healing |
|  **Shared Context** | Agents share knowledge of defined classes/functions via AST analysis |
|  **Sandboxed Execution** | Three tiers: `restricted` (real isolation), `docker` (strongest), `crash_isolated` (subprocess fallback for GUI/input) |
|  **Multi-Provider** | Switch between Groq, Gemini, Ollama, OpenAI, or OpenRouter |
|  **Multi-File Output** | Generate organized project structures (models.py, services.py, main.py) |
|  **Auto-Generated Tests** | TestGenerator creates comprehensive pytest test suites |
|  **Verified Tests** | Generated tests are executed against the generated code; pass/fail recorded in the session log |
|  **Auto Documentation** | Documenter agent creates README and adds docstrings |
|  **Parallel Execution** | Tests and documentation generated concurrently |
|  **Cost & Token Tracking** | Per-agent and per-model token counts and USD cost reported every run; optional budget caps |
|  **Web Interface** | Flask-based UI with rate limiting and OpenAPI/Swagger docs |
|  **Docker Ready** | Dockerfile + docker-compose for containerized deployment |
|  **CI/CD Pipeline** | GitHub Actions for automated testing and linting |
|  **Bounded Cache** | Agent response cache uses LRU + TTL (no more unbounded memory files) |

---

##  Agent Pipeline

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              USER PROMPT                                     │
│            "Create a todo list manager with priorities"                      │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│   PLANNER          Breaks prompt into 2-4 logical modules                  │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│   ARCHITECT        Creates high-level design (classes, interfaces)        │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│   DEVELOPER        Writes Python code for each module                      │
│                      ↺ Retries with CRITIC feedback (up to 3x)              │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│   QA AGENT         Validates execution in sandbox                          │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│   INTEGRATOR       Merges all code into cohesive program                   │
│                      Creates multi-file structure with import validation     │
└─────────────────────────────────────────────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    │         PARALLEL              │
                    ▼                               ▼
┌───────────────────────────────┐   ┌───────────────────────────────┐
│   TEST GENERATOR            │   │   DOCUMENTER                 │
│  Creates pytest test suite    │   │  Creates README + docstrings  │
└───────────────────────────────┘   └───────────────────────────────┘
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                            OUTPUT                                          │
│   output/final_program.py  |  output/test_program.py  |  output/README.md   │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

##  Quick Start

### 1. Clone and Setup

```bash
git clone https://github.com/josepeon/multi_agent_planner.git
cd multi_agent_planner

# Using conda (recommended)
conda env create -f environment.yml
conda activate multi_agent_planner
pip install -r requirements.txt
```

### 2. Configure LLM Provider

```bash
cp .env.example .env
```

**Groq (FREE - Recommended):**
```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your-key-here
```

Get your free API key at [console.groq.com](https://console.groq.com/)

### 3. Run

**CLI Mode:**
```bash
python main.py
```

**Web UI:**
```bash
python web/app.py
# Open http://localhost:8080
```

**Docker:**
```bash
docker compose up
# Open http://localhost:8080
```

---

##  Testing

```bash
# Run all tests
python -m pytest tests/ -v

# Run with coverage
python -m pytest tests/ --cov=agents --cov=core

# 215 tests covering agents, core modules, sandbox, integration, test execution, cache, cost tracking, and DAG executor
```

---

##  Web Interface

<div align="center">

| Input | Output |
|-------|--------|
| Simple form for project descriptions | Tabbed view: Code, Tests, README |
| Example prompts for quick testing | Download project as ZIP |

</div>

```bash
python web/app.py
# Open http://localhost:8080
```

### API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI |
| `/api/generate` | POST | Start code generation |
| `/api/status/{job_id}` | GET | Check job status |
| `/api/download/{job_id}` | GET | Download project ZIP |
| `/api/recent` | GET | List recent jobs |
| `/api/health` | GET | Health check |
| `/swagger` | GET | OpenAPI/Swagger documentation |

### Rate Limiting
- 10 requests per minute per IP address
- Configurable via `RATE_LIMIT_MAX` and `RATE_LIMIT_WINDOW` env vars

---

##  Project Structure

```
multi_agent_planner/
├── agents/                    # AI Agents
│   ├── planner.py             # Task decomposition
│   ├── architect.py           # High-level design
│   ├── developer.py           # Code generation
│   ├── qa.py                  # Code validation
│   ├── critic.py              # Code review & feedback
│   ├── integrator.py          # Code merging + import validation
│   ├── test_generator.py      # Pytest generation
│   ├── documenter.py          # README generation
│   └── base_agent.py          # Abstract base class
├── core/                      # Core System
│   ├── orchestrator.py        # Pipeline with parallel execution
│   ├── shared_context.py      # AST-based shared memory
│   ├── llm_provider.py        # Multi-provider LLM (auto-fallback)
│   ├── sandbox.py             # Sandboxed execution
│   ├── retry.py               # Exponential backoff
│   ├── memory.py              # Persistent JSON cache
│   └── logger.py              # Structured logging (color + JSON)
├── web/                       # Web Interface
│   ├── app.py                 # Flask server with rate limiting
│   ├── openapi.yml            # OpenAPI 3.0 specification
│   └── templates/index.html   # Web UI
├── tests/                     # Test Suite (215 tests)
│   ├── test_agents.py         # Agent unit tests
│   └── test_core.py           # Core module tests
├── .github/workflows/         # CI/CD
│   └── ci.yml                 # GitHub Actions pipeline
├── output/                    # Generated Output
│   ├── final_program.py       # Main code
│   ├── test_program.py        # Pytest tests
│   ├── README.md              # Documentation
│   └── project/               # Multi-file output
├── Dockerfile                 # Docker image
├── docker-compose.yml         # Docker Compose config
├── main.py                    # CLI entry point
├── requirements.txt
└── environment.yml
```

---

##  Architecture

### Agent Roles

| Agent | Responsibility |
|-------|----------------|
| **PlannerAgent** | Breaks prompts into 2-4 logical modules |
| **ArchitectAgent** | Creates high-level design with classes and interfaces |
| **DeveloperAgent** | Writes Python code with sandboxed execution |
| **QAAgent** | Validates code execution and correctness |
| **CriticAgent** | Reviews failed code, provides actionable feedback |
| **IntegratorAgent** | LLM-powered code merging with AST import validation |
| **TestGeneratorAgent** | Generates comprehensive pytest test suites |
| **DocumenterAgent** | Creates README.md and adds docstrings |

### LLM Providers

| Provider | Cost | Models | Auto-Fallback |
|----------|------|--------|---------------|
| **Groq** |  FREE | llama-3.3-70b-versatile | → llama-3.1-8b-instant → gemma2-9b-it |
| **Gemini** |  FREE tier | gemini-2.0-flash | - |
| **Ollama** |  FREE (local) | llama3.2, codellama | - |
| **OpenAI** |  Paid | gpt-4o, gpt-4 | - |
| **OpenRouter** |  Pay-per-use | All models | - |

### Security

| Sandbox Method | What it actually provides | Requirements |
|---|---|---|
| `restricted` (default) | **Real security isolation.** AST allowlist blocks network, file I/O, subprocess, `eval`/`exec`, dangerous dunders. Safe for untrusted input. | None |
| `docker` | **Strongest isolation.** Container with no network, read-only FS, memory + pid limits. | Docker on host |
| `crash_isolated` (alias: `subprocess`) | **Crash isolation only — NOT a security boundary.** A subprocess can read/write the host filesystem, open sockets, and exec arbitrary binaries. Used as a fallback when restricted-mode code requires `input()` / GUI / `eval`. Hosted deployments serving untrusted input should disable it (see below). | None |
| `cloud` | **Hosted ephemeral container via E2B.** Real isolation, real network, real filesystem — the only mode that can validate code that does I/O (scrapers, API clients, DB code). | `E2B_API_KEY` + `pip install e2b-code-interpreter` |

**Disabling the crash-isolated fallback in production**

Set `MAP_FORBID_CRASH_ISOLATED=1` in your environment. With that flag set,
code that would otherwise fall back to subprocess execution is reported as
unverifiable instead of executed. Recommended for any web-facing deployment
where the prompt source is not fully trusted.

**Blocked in restricted mode:** `os.system`, `subprocess`, `eval`, `exec`,
`__import__`, file I/O, network access, dangerous dunders.

---

##  API Usage

### LLM Client

```python
from core.llm_provider import get_llm_client

client = get_llm_client()  # Uses default from .env
response = client.chat("Write a hello world")

# Specify provider
client = get_llm_client(provider="groq", model="llama-3.3-70b-versatile")
```

### Sandboxed Execution

```python
from core.sandbox import execute_code_safely

result = execute_code_safely('print("Hello")')
# {"success": True, "output": "Hello"}

result = execute_code_safely('import os; os.system("rm -rf /")')
# {"success": False, "error": "Security violation: 'os.system' is not allowed"}
```

### Full Pipeline

```python
from core.orchestrator import Orchestrator

orchestrator = Orchestrator()
result = orchestrator.run("Create a calculator with basic operations")
# Generates code, tests, and documentation
```

---

##  Output Options

### Single File (default)
```
output/
├── final_program.py    # All code
├── test_program.py     # Pytest tests
└── README.md           # Documentation
```

### Multi-File Project
Enable in `core/orchestrator.py`:
```python
MULTI_FILE_OUTPUT = True
```
```
output/project/
├── models.py           # Data models and enums
├── services.py         # Business logic
├── main.py             # Entry point
├── test_program.py     # Tests
└── README.md           # Documentation
```

---

##  Tech Stack

| Component | Technology |
|-----------|------------|
| **Language** | Python 3.11 |
| **LLM** | Groq (Llama 3.3 70B), Gemini, Ollama, OpenAI |
| **Web Framework** | Flask with rate limiting |
| **API Docs** | OpenAPI 3.0 / Swagger UI |
| **Code Analysis** | AST (Abstract Syntax Tree) |
| **Testing** | pytest (215 tests) |
| **CI/CD** | GitHub Actions |
| **Containerization** | Docker + Docker Compose |
| **Concurrency** | ThreadPoolExecutor |
| **Environment** | Conda + pip |

---

## [Docker] Docker Deployment

### Quick Start with Docker Compose

```bash
# Start the web service
docker compose up

# Or run in CLI mode
docker compose run --rm cli
```

### Build and Run Manually

```bash
# Build the image
docker build -t multi-agent-planner .

# Run with your API key
docker run -p 8080:8080 -e GROQ_API_KEY=your-key multi-agent-planner
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GROQ_API_KEY` | - | Your Groq API key (required) |
| `LLM_PROVIDER` | `groq` | LLM provider to use |
| `PORT` | `8080` | Web server port |
| `LOG_LEVEL` | `INFO` | Logging level |
| `RATE_LIMIT_MAX` | `10` | Max requests per window |
| `RATE_LIMIT_WINDOW` | `60` | Rate limit window (seconds) |

---

## [Docker] Requirements

- Python 3.11+
- Groq API key (free) or other LLM provider
- Docker (optional, for maximum sandbox security)

---

##  License

MIT License - feel free to use this project for any purpose.

---

##  Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

<div align="center">

[ Back to top](#-multi-agent-planner)

</div>