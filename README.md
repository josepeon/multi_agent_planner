# Multi-Agent Planner

**LLM-powered system that solves coding tasks through agent collaboration.**

This project implements a multi-agent architecture using LLM-based agents, where each role has a specific function. A **Planner Agent** breaks down tasks, an **Architect Agent** creates high-level designs, a **Developer Agent** writes code, a **QA Agent** validates execution, a **Critic Agent** suggests improvements, and an **Integrator Agent** merges code into a cohesive program.

## ✨ Key Features

- 🆓 **Free by Default** - Uses Groq's free Llama 3.3 70B API
- 🏗️ **Architecture-First** - Architect agent creates design before coding
- 🔄 **Smart Retry** - Failed tasks retry up to 3x with critic feedback
- 🧠 **Shared Context** - Agents share knowledge of defined classes/functions
- 🔒 **Sandboxed Execution** - Safe code execution with multiple isolation methods
- 🔌 **Multi-Provider** - Switch between Groq, Gemini, Ollama, OpenAI, OpenRouter
- 💾 **Persistent Memory** - Caches results to avoid redundant API calls
- 🧩 **Modular Design** - Easy to extend with new agents

---

## 🚀 Supported LLM Providers

| Provider | Cost | Models | Setup |
|----------|------|--------|-------|
| **Groq** | ✅ FREE | Llama 3.3 70B, Mixtral | [Get API Key](https://console.groq.com/) |
| **Google Gemini** | ✅ FREE tier | Gemini 2.0 Flash, 1.5 Pro | [Get API Key](https://aistudio.google.com/apikey) |
| **Ollama** | ✅ FREE (local) | Llama 3.2, CodeLlama, Mistral | [Install Ollama](https://ollama.ai/) |
| **OpenAI** | 💰 Paid | GPT-4o, GPT-4 | [Get API Key](https://platform.openai.com/api-keys) |
| **OpenRouter** | 💰 Pay-per-use | All models | [Get API Key](https://openrouter.ai/) |

---

## 🔒 Security Features

### Sandboxed Code Execution

Generated code runs in isolated environments to prevent malicious operations:

| Method | Security | Requirements | Use Case |
|--------|----------|--------------|----------|
| `restricted` | ⭐⭐⭐ | None | Default, blocks dangerous operations |
| `docker` | ⭐⭐⭐⭐⭐ | Docker installed | Production, full isolation |
| `subprocess` | ⭐⭐ | None | Basic timeout protection |

**Blocked operations:** `os.system`, `subprocess`, `eval`, `exec`, `__import__`, file I/O, network access

---

## Example Workflow

**Example user prompt:**  
*"Create a command-line tool that parses a CSV file and returns JSON-formatted summary statistics."*

System output:
1. **Planner** → Breaks task into 2-4 logical modules
2. **Architect** → Creates high-level design (classes, interfaces, dependencies)
3. **Developer** → Writes Python code for each module
4. **QA Agent** → Validates execution in sandbox
5. **Critic** → Reviews failed code, provides feedback
6. **Developer** → Retries with critic feedback (up to 3 attempts)
7. **Integrator** → Intelligently merges all code into final program

---

## 📁 Project Structure

```
multi_agent_planner/
├── agents/
│   ├── planner.py       # Task decomposition (2-4 logical modules)
│   ├── architect.py     # High-level design
│   ├── developer.py     # Code generation + sandboxed execution
│   ├── qa.py            # Code validation
│   ├── critic.py        # Code review & feedback
│   ├── integrator.py    # Intelligent code merging
│   ├── test_generator.py # Pytest test generation (NEW)
│   ├── documenter.py    # README & docstring generation (NEW)
│   └── base_agent.py    # Abstract base class
├── core/
│   ├── orchestrator.py  # Pipeline with parallel execution
│   ├── shared_context.py # Shared memory with code snippets
│   ├── llm_provider.py  # Multi-provider LLM abstraction
│   ├── sandbox.py       # Sandboxed code execution
│   ├── retry.py         # Exponential backoff retry logic
│   ├── memory.py        # Persistent JSON memory
│   ├── task_schema.py   # Task dataclass
│   └── assembler.py     # Legacy code assembly (deprecated)
├── web/                 # Web Interface (NEW)
│   ├── app.py           # Flask web server
│   └── templates/
│       └── index.html   # Web UI
├── output/              # Generated outputs
│   ├── final_program.py # Single-file output
│   ├── test_program.py  # Generated tests
│   ├── README.md        # Generated documentation
│   └── project/         # Multi-file output (if enabled)
├── memory/              # Agent memory caches
├── tests/               # Unit tests
├── main.py              # CLI entry point
├── requirements.txt
├── environment.yml      # Conda environment
├── .env.example         # Environment template
├── IMPROVEMENTS.md      # Progress tracker
└── README.md
```

## 🤖 Implemented Agents

| Agent | Role |
|-------|------|
| `PlannerAgent` | Breaks user prompts into 2-4 logical modules (not micro-tasks) |
| `ArchitectAgent` | Creates high-level design with classes, interfaces, dependencies |
| `DeveloperAgent` | Writes Python code with sandboxed execution |
| `QAAgent` | Verifies code execution and correctness |
| `CriticAgent` | Reviews failed code and provides actionable feedback |
| `IntegratorAgent` | LLM-powered intelligent code merging with AST fallback |
| `TestGeneratorAgent` | Generates comprehensive pytest test suites |
| `DocumenterAgent` | Creates README.md and adds docstrings |

---

## 🌐 Web Interface

The system includes a web-based UI for easier interaction:

```bash
# Install Flask if not already installed
pip install flask

# Run the web server
cd multi_agent_planner
python web/app.py
```

Then open http://localhost:5000 in your browser.

**Features:**
- Simple input form for project descriptions
- Real-time generation status
- Tabbed output view (Code, Tests, README)
- Download project as ZIP
- Example prompts for quick testing

---

## 📦 Output Options

### Single File (default)
```python
# output/final_program.py - All code in one file
```

### Multi-File Project
Enable multi-file output in `core/orchestrator.py`:
```python
MULTI_FILE_OUTPUT = True  # Creates output/project/ with models.py, services.py, main.py
```

---

## 🚀 Quick Start

### 1. Clone and setup environment

```bash
git clone https://github.com/josepeon/multi_agent_planner.git
cd multi_agent_planner

# Option A: Using conda (recommended)
conda env create -f environment.yml
conda activate multi_agent_planner
pip install -r requirements.txt

# Option B: Using pip
pip install -r requirements.txt
```

### 2. Configure your LLM provider

Copy `.env.example` to `.env` and add your API key:

```bash
cp .env.example .env
```

**Option A: Use Groq (FREE & Fast - Recommended)**
```env
LLM_PROVIDER=groq
GROQ_API_KEY=gsk_your-key-here
```

**Option B: Use Google Gemini (FREE tier)**
```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your-key-here
```

**Option C: Use Ollama (FREE, runs locally)**
```bash
# First install Ollama and pull a model
ollama pull llama3.2
```
```env
LLM_PROVIDER=ollama
```

**Option D: Use OpenAI (Paid)**
```env
LLM_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
```

### 3. Run the planner
```bash
python main.py
```

### 4. Follow the interactive prompt to generate your desired program!

---

## 📖 API Usage

### LLM Provider

```python
from core.llm_provider import get_llm_client

# Use default provider from .env (Groq)
client = get_llm_client()
response = client.chat("Write a hello world in Python")

# Specify provider and model
client = get_llm_client(provider="groq", model="llama-3.3-70b-versatile")
response = client.chat(
    "Write a function to parse CSV",
    system_message="You are a senior Python developer"
)
```

### Sandboxed Execution

```python
from core.sandbox import execute_code_safely

# Execute code safely
result = execute_code_safely('print("Hello World")')
print(result["output"])  # "Hello World"
print(result["success"]) # True

# Dangerous code is blocked
result = execute_code_safely('import os; os.system("rm -rf /")')
print(result["success"]) # False
print(result["error"])   # "Security violation: 'os.system' is not allowed"

# Use Docker for maximum isolation
result = execute_code_safely(code, method="docker")
```

### Retry Logic

```python
from core.retry import retry_with_backoff, retry_llm_call

# Decorator pattern
@retry_with_backoff(max_retries=3, base_delay=1.0)
def call_api():
    return requests.get("https://api.example.com")

# Direct usage for LLM calls
result = retry_llm_call(lambda: client.chat("Hello"), max_retries=3)
```

---

## 🧪 Testing

```bash
# Run a simple test
python -c "from core.llm_provider import get_llm_client; print(get_llm_client().chat('Say hello'))"

# Test sandbox
python -c "from core.sandbox import execute_code_safely; print(execute_code_safely('print(1+1)'))"
```

---

## 🗺️ Roadmap

- [ ] Web UI (Streamlit/Gradio)
- [ ] Async pipeline for parallel task execution
- [ ] RAG integration for documentation context
- [ ] Multi-file project generation
- [ ] Git integration for auto-commits
- [ ] Streaming output support

---

## 📄 License

MIT License - feel free to use this project for any purpose.

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.