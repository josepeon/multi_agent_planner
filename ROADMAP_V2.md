# Multi-Agent Planner — Roadmap v2.0

> Forward-looking improvement plan. `IMPROVEMENTS.md` covers what's already been built (Rounds 1–3, current state = Production-Grade 9/10). This document covers what's next.

## Context

The v1.0 product is feature-complete as a polished, single-shot Python code generator:
8-agent pipeline (Planner → Architect → Developer×N → QA → Critic → Integrator → TestGenerator + Documenter), multi-provider LLM with auto-fallback, 3 sandbox strategies, retry-with-critic loop, parallel test/doc generation, Flask UI, Docker, GitHub Actions CI, OpenAPI/Swagger, 60 tests.

The next wave of improvements is **conceptual, not infrastructural** — the scaffolding is solid; what's missing is depth in how the agents reason, coordinate, learn, and produce shippable output.

---

## Bugs to fix first

These should land before or alongside Tier 1 work — they undermine current correctness claims.

### B1. Generated tests are never executed
`TestGeneratorAgent` writes `output/test_program.py` and `DocumenterAgent` writes a README claiming the project is tested — but the orchestrator never runs the tests against `output/final_program.py`. So "tests pass" is unverified.

**Fix:** After `test_generator.generate_tests(...)`, add a pytest execution step in `core/orchestrator.py`. Capture pass/fail counts, include in the session log and the final README. If tests fail, optionally feed the failures back to the Developer or Integrator for one repair pass.

**Effort:** Small (½ day).

### B2. The "subprocess" sandbox isn't a sandbox
`core/sandbox.py` exposes three modes: `restricted` (RestrictedPython, real isolation), `docker` (real isolation), and `subprocess` (just `python -c …` in a child process — *zero* security isolation, only crash isolation). README labels it "Low" security; in practice it's none. On a hosted web deployment this is a foot-gun.

**Fix:** Either remove `subprocess` mode entirely or rename to `crash-isolated` and document clearly that it's unsafe for untrusted input. Update web UI to forbid `subprocess` mode when the request originates from a non-localhost IP.

**Effort:** Small (½ day).

### B3. Agent-response cache is unbounded
`memory/developer_memory.json` and `memory/critic_memory.json` are unbounded JSON dicts keyed by full prompt strings. They will grow forever; no TTL, no eviction.

**Fix:** LRU + size cap (e.g. 1000 entries) or SQLite with TTL. The cache itself is fine and useful — it just needs bounds.

**Effort:** Small (½ day).

---

## Tier 1 — Transformational

These three together constitute the "v2.0" release. Highest leverage on output quality.

### T1.1. Replace the linear pipeline with a DAG

**Problem:** Planner produces 2–4 modules, but `core/orchestrator.run_pipeline` develops them serially in a `for task in tasks:` loop. Independent modules wait for each other. There's no way to express conditional agents (e.g. "if this project needs a DB, add a `DBAgent` step") or to re-plan when QA finds the architecture is wrong.

**Plan:**
- Define a `PipelineGraph` abstraction: nodes are agent invocations, edges are dependencies, each node declares its inputs (other node outputs) and outputs.
- Planner emits a graph (not a flat task list). Independent dev tasks become parallel branches.
- Add conditional nodes (Planner or Architect can decide to inject specialist agents: `DBAgent`, `AuthAgent`, `APIClientAgent`).
- Add a "re-plan" edge from QA back to Planner/Architect when a failure looks architectural rather than implementation-level (heuristic: 3 dev retries all failed → re-plan).
- Reuse the existing `ThreadPoolExecutor` pattern; just generalize it from the hardcoded test+docs parallelism to any graph layer.

**Library options:** custom (small DAG ~150 LOC) or LangGraph. Recommend **custom first** — keeps dependency footprint low and matches existing codebase style.

**Effort:** Medium-Large (3–5 days).
**Affected files:** `core/orchestrator.py` (rewrite), `agents/planner.py` (output schema change), `core/task_schema.py` (extend), new `core/pipeline_graph.py`.

### T1.2. Human-in-the-loop checkpoints

**Problem:** Prompt → code with no intermediate review. For non-trivial projects the architecture stage is where the wrong decision usually happens (wrong storage choice, wrong framework, wrong decomposition). Once code is generated, fixing those decisions costs many more tokens than approving them up front.

**Plan:**
- Add two optional gates: **after Planner** (review the module breakdown) and **after Architect** (review classes/interfaces). Default: gated in web UI, ungated in CLI.
- Each gate returns one of: `approve`, `edit` (user provides corrected text, becomes the new artifact), `regenerate` (re-run that agent with extra hints).
- Web UI: turn the existing single-page Flask form into a stepwise flow with a "Continue / Edit / Regenerate" panel between stages.
- Add `--auto` flag and `?auto=1` query param to skip all gates (current behavior).

**Effort:** Medium (2–3 days).
**Affected files:** `web/app.py`, `web/templates/index.html`, `core/orchestrator.py` (gate hooks), new `web/templates/review.html`.

### T1.3. Project memory that actually learns

**Problem:** `core/memory.py` is named "Memory" but it's a response-deduplication cache, not learning. The system has no notion of "this user prefers FastAPI over Flask," "last time we built a CLI tool we used `click`," or "this project type usually needs X."

**Plan:**
- Split memory into three layers, modeled on the conventional cognitive architecture:
  - **Response cache** (current behavior, renamed for clarity)
  - **Project memory** (per-project: chosen libs, architectural decisions, files written)
  - **User memory** (across projects: preferred stack, common patterns, libraries blacklisted)
- Memory is consulted by the **Architect** (to bias toward known-good choices) and the **Planner** (to recognize project archetypes).
- Surface memory in the UI as "remembered preferences," editable by the user.
- **Cross-repo opportunity:** the `self-improving-agent` repo has the exact infrastructure (ChromaDB + sentence-transformers, RAG over a knowledge base, lifecycle management) that this needs. Don't rebuild it — import or vendor the relevant modules from `self-improving-agent`.

**Effort:** Medium (3–4 days), or Small (1–2 days) if reusing `self-improving-agent` modules.
**Affected files:** `core/memory.py` (split into 3 modules), `agents/architect.py`, `agents/planner.py`, `web/app.py`, new `core/project_memory.py`, `core/user_memory.py`.

---

## Tier 2 — High-impact additions

Each is independently valuable; can be sequenced after T1 or in parallel.

### T2.1. Researcher agent with web + docs retrieval

**Problem:** All 8 agents generate from training prior. APIs are hallucinated, library versions drift, current best practices are missed.

**Plan:**
- New `ResearcherAgent` that runs between Planner and Architect (and optionally before Developer for unfamiliar libraries).
- Uses a web search API (Tavily, Exa, or Brave Search) + a curated RAG store of common library docs (FastAPI, Flask, SQLAlchemy, Pydantic, requests, pandas, pytest).
- Output: a short "research brief" injected into Architect's and Developer's context.

**Effort:** Medium (3 days).
**Affected files:** new `agents/researcher.py`, new `core/web_search.py`, new `core/docs_rag.py`, `core/orchestrator.py`.

### T2.2. Real sandbox via E2B / Modal / Daytona

**Problem:** `RestrictedPython` blocks subprocess, network, and most file I/O — so QA can't validate any project that does real I/O (scrapers, API clients, DB code, anything fun). This is a *severe* limitation on what the system can credibly generate.

**Plan:**
- Add a fourth sandbox mode: `cloud` (E2B is the closest fit — purpose-built for AI-generated code execution).
- The cloud sandbox gets real network access and ephemeral filesystem; cleanup on completion.
- Default `cloud` mode when network or subprocess is detected in generated code.

**Effort:** Medium (2–3 days).
**Affected files:** `core/sandbox.py`, `.env.example`, `requirements.txt`.

### T2.3. Streaming output in the web UI

**Problem:** Current UX is "submit, wait 30–90 seconds, see result." Boring and unconvincing.

**Plan:**
- Server-Sent Events (or WebSocket) endpoint that streams: stage transitions, agent thinking tokens, task pass/fail events, file writes.
- UI shows a live timeline ("Planner: defining tasks…", "Developer attempt 2/3…", "QA: passed").
- Token streaming on the final integrated code as it's written.

**Effort:** Medium (2 days).
**Affected files:** `web/app.py`, `web/templates/index.html`, `core/orchestrator.py` (event emitter hooks), new `core/events.py`.

### T2.4. Deployer agent

**Problem:** The system generates a project then stops. There's a 9th agent missing: **deploy it somewhere live.**

**Plan:**
- New `DeployerAgent` that picks a target based on project type:
  - CLI tool → publish to PyPI (or skip)
  - Flask/FastAPI service → Railway or Modal
  - Streamlit app → Streamlit Cloud
  - Static site → Vercel
- Generates the appropriate config (`Procfile`, `vercel.json`, `modal_app.py`, etc.) and runs the deploy.
- Returns a live URL.

**Effort:** Medium-Large (3–5 days, mostly per-target integration work).
**Affected files:** new `agents/deployer.py`, new `core/deploy_targets/{railway,modal,vercel,streamlit,pypi}.py`, `core/orchestrator.py`.

### T2.5. Best-of-N generation with critic scoring

**Problem:** One shot per task, retry only on hard failure. The critic only sees failed code. Many "passed" outputs are mediocre.

**Plan:**
- Per task, generate N candidates (configurable, default N=3) in parallel.
- Critic scores each (0–10 on correctness, clarity, idiomatic-ness).
- Highest-scoring candidate wins; others go to memory for negative signal.
- Trades cost (Nx tokens) for quality. Should be a per-request flag in the UI.

**Effort:** Small-Medium (1–2 days).
**Affected files:** `core/orchestrator.develop_with_retry`, `agents/critic.py` (add `score` method), `web/app.py` (expose flag).

### T2.6. "Modify existing codebase" mode

**Problem:** Always generates from scratch. Can't say "add OAuth to this repo" or "refactor this code." That's most real-world use.

**Plan:**
- Accept a Git URL or local path as input alongside the prompt.
- New ingestion step: clone, parse with AST (extend `core/shared_context.py`), build a project map (files, classes, functions, dependencies).
- Architect's job changes: "design a change" instead of "design from scratch."
- Developer modifies existing files rather than creating new ones. Integrator produces a diff/patch instead of a new project.
- Output is a PR-ready diff.

**Effort:** Large (5–7 days). This is essentially a second product mode.
**Affected files:** new `core/repo_ingestion.py`, new `core/diff_generator.py`, major changes to `agents/architect.py`, `agents/developer.py`, `agents/integrator.py`.

### T2.7. Cost & token tracking

**Problem:** No visibility into per-run cost. Bad UX for anyone on paid LLM tiers.

**Plan:**
- Track tokens (input + output) per agent invocation. Most provider SDKs return this in the response.
- Multiply by per-provider rate cards. Aggregate per run, per session, per user.
- Surface in UI footer: "Cost: $0.04 (12,400 input + 3,200 output tokens)."
- Add `--budget` flag and `?budget=` param: hard-stop if exceeded.

**Effort:** Small (1 day).
**Affected files:** `core/llm_provider.py` (return usage), new `core/cost_tracker.py`, `web/app.py`, `core/orchestrator.py`.

---

## Tier 3 — Polish & infrastructure

### T3.1. Model routing per agent role

Different agents need different model capabilities. Planner needs reasoning (Llama 70B is fine, GPT-4 / Claude Opus better). Documenter is fine on a cheap small model. Single-knob router:

```yaml
# config/model_routing.yml
planner: groq/llama-3.3-70b-versatile
architect: groq/llama-3.3-70b-versatile
developer: groq/llama-3.3-70b-versatile
critic: groq/llama-3.1-8b-instant
test_generator: groq/llama-3.3-70b-versatile
documenter: groq/llama-3.1-8b-instant
```

**Effort:** Small (1 day).

### T3.2. Eval harness for the pipeline itself

Right now there's no way to know if a change to a prompt or to the pipeline structure improved things. Need:
- A corpus of 20–30 representative prompts ("build a CLI todo app", "build a Flask API with auth", etc.).
- Golden outputs (acceptable solution structures, not exact code).
- Automated scoring: tests pass, code runs, structure matches, idiomatic-ness (judged by GPT-4 / Claude).
- A `make eval` target that runs the full suite and produces a regression report.

The `self-improving-agent` repo has this *exact* pattern (golden-test YAML rubrics, 92% coverage across 291 tests). **Reuse it.**

**Effort:** Medium (2–3 days), or Small (1 day) if reusing `self-improving-agent`'s harness.

### T3.3. Multi-language / multi-stack output

Hardcoded to Python. Most interesting modern projects are full-stack (TypeScript frontend + Python or Node backend). Even just adding "TypeScript CLI" or "Next.js scaffold" as output targets would 10× the demo surface.

**Plan:** Generalize Developer and Integrator over a `Language` enum. Add per-language prompt variants and per-language sandbox runners.

**Effort:** Large (5–7 days per added language/stack — pick the first one carefully).

### T3.4. Agent fine-tuning via self-improving-agent

Currently every agent is just prompt-engineered. The big unlock: each agent role is a fine-tuned MLX model trained on its own role-specific transcripts.

`self-improving-agent` is already built for this exact pattern (MLX + LoRA + Groq-as-judge self-play). Cross-repo opportunity:
- Capture all agent invocations to a structured log.
- Periodically run `self-improving-agent`'s pipeline on each agent role's logs.
- Promote fine-tuned models to the `model_router` (T3.1) when they beat the base model on the eval harness (T3.2).

This is the portfolio narrative: **"my agents train each other."** No one else has it.

**Effort:** Large (1–2 weeks), but mostly infrastructure-glue work — the heavy lifting is already in `self-improving-agent`.

---

## Recommended phasing

### Sprint 1 (~1 week): "Trust the output"
- B1 (run the tests)
- B2 (sandbox honesty)
- B3 (bounded cache)
- T2.7 (cost tracking)

After this, every claim the README makes is true.

### Sprint 2 (~2 weeks): "v2.0 — the DAG release"
- T1.1 (DAG pipeline)
- T1.2 (human-in-the-loop checkpoints)
- T2.3 (streaming UI)

After this, the product *feels* different — parallelism is visible, the user is in control, the pipeline streams.

### Sprint 3 (~2 weeks): "Smarter agents"
- T1.3 (project + user memory, reusing `self-improving-agent`)
- T2.1 (Researcher agent + web search)
- T2.2 (real sandbox via E2B)
- T2.5 (best-of-N)

After this, generation quality is meaningfully higher and the system can build projects that do real I/O.

### Sprint 4 (~2 weeks): "End-to-end"
- T2.4 (Deployer agent — at least Railway and Streamlit Cloud)
- T2.6 (modify-existing-codebase mode)

After this, the system ships, not just generates.

### Sprint 5 (~2 weeks): "Self-improvement loop"
- T3.1 (model routing)
- T3.2 (eval harness, reusing `self-improving-agent`)
- T3.4 (fine-tune each agent role)

After this, the system improves itself between releases.

---

## Total scope

- **Bugs:** 1.5 days
- **Tier 1:** ~10 days
- **Tier 2:** ~15–20 days
- **Tier 3:** ~15+ days

≈ **6–10 weeks** of focused work to land everything in this document.

## Cross-repo dependencies

Three improvements *should* reuse code from `self-improving-agent` rather than rebuild:
- T1.3 (project/user memory → ChromaDB + sentence-transformers stack)
- T3.2 (eval harness → golden-test YAML rubrics + 92%-coverage test infra)
- T3.4 (agent fine-tuning → MLX + LoRA + Groq-as-judge pipeline)

Treat `self-improving-agent` as a library this project depends on, not a separate experiment.
