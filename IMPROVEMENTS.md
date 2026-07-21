# Multi-Agent Planner - Architecture Improvements

## Current Level: Production-Grade (9/10)

All Round 3 production hardening improvements are complete!

---

## Round 3 - Production Hardening [DONE]

| # | Improvement | Priority | Effort | Status |
|---|-------------|----------|--------|--------|
| 1 | Unit tests for the system |  High | 2-3 hrs | [DONE] Done |
| 2 | CI/CD pipeline (GitHub Actions) |  High | 1 hr | [DONE] Done |
| 3 | Structured logging |  Medium | 1 hr | [DONE] Done |
| 4 | Type hints throughout |  Medium | 2 hrs | [DONE] Done |
| 5 | Dockerize the system |  Medium | 1 hr | [DONE] Done |
| 6 | API rate limiting |  Low | 30 min | [DONE] Done |
| 7 | OpenAPI/Swagger docs |  Low | 1 hr | [DONE] Done |

---

## Round 2 - Feature Complete [DONE]

| # | Improvement | Priority | Status | Notes |
|---|-------------|----------|--------|-------|
| 1 | Parallel task execution | Medium | [DONE] Done | Test + Documentation run in parallel using ThreadPoolExecutor |
| 2 | File-based output | Medium | [DONE] Done | Multi-file project structure (models.py, services.py, main.py) |
| 3 | Unit test generation | Medium | [DONE] Done | agents/test_generator.py, generates pytest tests |
| 4 | Documentation agent | Medium | [DONE] Done | agents/documenter.py, generates README.md |
| 5 | Better shared context | Medium | [DONE] Done | Now includes actual code snippets |
| 6 | Web UI / API interface | Large | [DONE] Done | Flask web app at web/app.py |

**Bonus**: Auto-fallback for Groq rate limits, AST-based import validation for multi-file output.

---

## Round 1 - Core Architecture [DONE]

- [DONE] Expand sandbox allowed imports
- [DONE] Add retry loop with critic feedback
- [DONE] Add shared context memory
- [DONE] Smarter planner (fewer tasks)
- [DONE] Add Architect agent
- [DONE] Add Integrator agent
- [DONE] Fix QA to trust sandbox results

---

## Round 3 Details

### 1. Unit Tests for the System
**Current**: No tests for agents, orchestrator, or core modules.

**Fix**: Add pytest tests covering:
- Each agent (mock LLM responses)
- Orchestrator pipeline
- Sandbox execution
- LLM provider fallback logic

**Files**: `tests/test_agents.py`, `tests/test_orchestrator.py`, `tests/test_sandbox.py`, `tests/test_llm_provider.py`

**Goal**: 70%+ code coverage

---

### 2. CI/CD Pipeline
**Current**: No automated testing or deployment.

**Fix**: Add GitHub Actions workflow:
- Run pytest on push/PR
- Run linting (ruff or flake8)
- Check type hints (mypy)

**Files**: `.github/workflows/ci.yml`

---

### 3. Structured Logging
**Current**: Uses `print()` statements throughout.

**Fix**: Replace with Python `logging` module:
- Different log levels (DEBUG, INFO, WARNING, ERROR)
- Configurable log output (file, console)
- Structured log format with timestamps

**Files**: `core/logger.py` (NEW), update all agents and core modules

---

### 4. Type Hints Throughout
**Current**: Partial type hints, inconsistent usage.

**Fix**: Add comprehensive type hints:
- All function parameters and returns
- Class attributes
- Use `typing` module for complex types

**Files**: All `.py` files in `agents/` and `core/`

**Validation**: Run `mypy` to verify

---

### 5. Dockerize the System
**Current**: No containerization for the system itself.

**Fix**: Add Docker support:
- `Dockerfile` for the application
- `docker-compose.yml` for easy startup
- Include all dependencies

**Files**: `Dockerfile`, `docker-compose.yml`

---

### 6. API Rate Limiting
**Current**: Web API has no rate limiting.

**Fix**: Add rate limiting to prevent abuse:
- Limit requests per IP
- Use Flask-Limiter or custom middleware

**Files**: `web/app.py`

---

### 7. OpenAPI/Swagger Documentation
**Current**: No API documentation.

**Fix**: Add OpenAPI spec:
- Document all endpoints
- Add request/response schemas
- Serve Swagger UI

**Files**: `web/app.py`, `web/openapi.yml`

---

## Implementation Log

### Round 3 Started: 2025-12-28

(Updates will be added here as each improvement is completed)

### Round 4 — Hardening & correctness audit (2026-07-20)

Full-codebase audit (4 parallel reviewers over orchestrators, provider
layer, agents, and web/sandbox/memory) followed by targeted fixes. Tests
315 → 349; ruff lint + format now clean and blocking in CI. Version 2.0.0.

**Security**
- Web app enables the sandbox gate (`MAP_FORBID_CRASH_ISOLATED=1`) by
  default — previously LLM-generated code containing eval/tkinter fell
  through to an unsandboxed subprocess on any network request.
- Loopback bind by default; FLASK_DEBUG refused off-box; rate limiter no
  longer trusts spoofable X-Forwarded-For; repo ingestion blocks the git
  `ext::` RCE transport, option injection, and symlink file leaks.

**Correctness**
- BoundedCache (shared across worker threads) is now lock-guarded with
  atomic saves — concurrent runs could corrupt the cache file.
- EventBus: late SSE subscribers to finished jobs no longer hang forever;
  crashed pipelines can't strand subscribers (@ends_bus); history bounded.
- Critic was reviewing the developer's result-dict repr instead of code;
  SharedContext advertised methods as top-level functions; Architect JSON
  with trailing prose collapsed the design; Integrator's builtins check
  never matched; test runner now aliases `import X` too.
- DAG orchestrator parity: researcher stage wired in, architecture in
  session_log, unified passed/failed vocabulary, replans re-validated.
- Provider layer: temperature=0 respected, Groq rate-limit cooldown
  (was benched-forever), groq default everywhere, Gemini/Ollama calls now
  cost-tracked and interaction-logged, import-safe logger.

**Infra**
- `python -m evals.run` exists (full + --offline modes) — the corpus
  previously referenced a runner that didn't exist.
- Packaging: wheel now ships main.py (console script crashed before);
  MIT LICENSE file added; Docker healthcheck fixed (curl isn't in the
  image); requirements.txt gained the missing `requests`.
