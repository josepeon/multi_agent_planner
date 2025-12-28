# Multi-Agent Planner - Architecture Improvements

## Progress Tracker

| # | Improvement | Priority | Status | Notes |
|---|-------------|----------|--------|-------|
| 1 | Parallel task execution | Medium | ✅ Done | Test + Documentation run in parallel using ThreadPoolExecutor |
| 2 | File-based output | Medium | ✅ Done | Multi-file project structure (models.py, services.py, main.py) |
| 3 | Unit test generation | Medium | ✅ Done | agents/test_generator.py, generates pytest tests |
| 4 | Documentation agent | Medium | ✅ Done | agents/documenter.py, generates README.md |
| 5 | Better shared context | Medium | ✅ Done | Now includes actual code snippets |
| 6 | Web UI / API interface | Large | ✅ Done | Flask web app at web/app.py |

---

## Completed (Previous Round)

- ✅ Expand sandbox allowed imports
- ✅ Add retry loop with critic feedback
- ✅ Add shared context memory
- ✅ Smarter planner (fewer tasks)
- ✅ Add Architect agent
- ✅ Add Integrator agent
- ✅ Fix QA to trust sandbox results

---

## Problem Details

### 1. Parallel Task Execution
**Current**: Tasks run sequentially, one after another.

**Fix**: Run independent tasks in parallel using asyncio or ThreadPoolExecutor.

**Benefits**: Faster execution, better API utilization.

**Files**: `core/orchestrator.py`

---

### 2. File-Based Output
**Current**: All code goes into a single `final_program.py` file.

**Fix**: Generate multi-file projects with proper structure:
- `models.py` - data classes
- `services.py` - business logic
- `main.py` - entry point

**Files**: `agents/integrator.py`, `core/orchestrator.py`

---

### 3. Unit Test Generation
**Current**: No automated tests for generated code.

**Fix**: Add a TestGenerator agent that creates pytest tests for the generated code.

**Files**: `agents/test_generator.py` (NEW)

---

### 4. Documentation Agent
**Current**: Code has minimal documentation.

**Fix**: Add a Documentation agent that:
- Adds comprehensive docstrings
- Generates a README.md for the project
- Adds type hints where missing

**Files**: `agents/documenter.py` (NEW)

---

### 5. Better Shared Context
**Current**: SharedContext only stores class/function names.

**Fix**: Store actual code snippets so later tasks can reference exact implementations.

**Files**: `core/shared_context.py`, `core/orchestrator.py`

---

### 6. Web UI / API Interface
**Current**: CLI-only interface.

**Fix**: Add a simple Flask/FastAPI web interface:
- Input: project description
- Output: generated code with syntax highlighting
- Download as ZIP

**Files**: `web/app.py` (NEW), `web/templates/` (NEW)

---

## Implementation Log

### Started: [timestamp]

(Updates will be added here as each improvement is completed)
