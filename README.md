# Multi-Agent Planner

**LLM-powered system that solves coding tasks through agent collaboration.**

This project implements a multi-agent architecture using LLM-based agents, where each role has a specific function. A **Planner Agent** breaks down tasks, a **Developer Agent** writes code, a **QA Agent** validates execution, and a **Critic Agent** suggests improvements.

## ✨ Key Features

- 🆓 **Free by Default** - Uses Groq's free Llama 3.3 70B API
- 🔒 **Sandboxed Execution** - Safe code execution with multiple isolation methods
- 🔄 **Auto-Retry** - Exponential backoff for API resilience
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
1. **Planner** → Breaks task into atomic subtasks
2. **Developer** → Writes Python code for each subtask (sandboxed execution)
3. **QA Agent** → Validates execution and correctness
4. **Critic** → Reviews failed code, suggests fixes
5. **Developer** → Revises and retries failed tasks (with exponential backoff)
6. **Assembler** → Generates clean, deduplicated final program

---

## 📁 Project Structure

```
multi_agent_planner/
├── agents/
│   ├── planner.py       # Task decomposition
│   ├── developer.py     # Code generation + sandboxed execution
│   ├── qa.py            # Code validation
│   ├── critic.py        # Code review
│   └── base_agent.py    # Abstract base class
├── core/
│   ├── orchestrator.py  # Pipeline coordinator
│   ├── llm_provider.py  # Multi-provider LLM abstraction
│   ├── sandbox.py       # Sandboxed code execution
│   ├── retry.py         # Exponential backoff retry logic
│   ├── memory.py        # Persistent JSON memory
│   ├── task_schema.py   # Task dataclass
│   └── assembler.py     # Code assembly
├── output/              # Generated outputs
├── memory/              # Agent memory caches
├── logs/                # Session logs
├── main.py              # Entry point
├── requirements.txt
├── environment.yml      # Conda environment
├── .env.example         # Environment template
└── README.md
```

## 🤖 Implemented Agents

| Agent | Role |
|-------|------|
| `PlannerAgent` | Breaks user prompts into atomic, executable subtasks |
| `DeveloperAgent` | Writes Python code with sandboxed execution |
| `QAAgent` | Verifies code execution and correctness |
| `CriticAgent` | Reviews failed code and suggests improvements |
| `Assembler` | Deduplicates and combines code into final output |

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